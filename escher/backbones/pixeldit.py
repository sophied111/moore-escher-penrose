"""escher.backbones.pixeldit -- PixelDiT adapter implementing :class:`Backbone`.

A pixel-space text-to-image DiT (NVlabs/PixelDiT). Unlike FLUX there is **no
VAE**: the model's working tensor *is* the RGB image in ``[-1, 1]``, so
``to_pixels``/``from_pixels`` are the identity and ``State.latents_or_pixels``
holds a ``(1, 3, H, W)`` pixel tensor throughout.

The ordinary diffusion is a plain fp32 flow-matching Euler step driven by a
``FlowMatchEulerDiscreteScheduler(shift=4.0)`` sigma grid -- exactly the v2-engine
PixelDiT reference (``pixeldit_pipe.py``), NOT PixelDiT's native DPM-Solver. The
velocity is a two-branch classifier-free call on the model's
``forward_with_dpmsolver`` (``v = v_u + g*(v_c - v_u)`` in fp32, no CFG at
``sigma == 1``), the clean prediction is ``x0 = x - sigma*v``, and re-noising is
the flow interpolation ``(1 - sigma_next)*x0 + sigma_next*noise`` in pixel space
-- the same contract as the FLUX backbone. Latents stay fp32 end to end; the
model input is cast to the model's weight dtype (bf16) only for the forward pass.

Import safety: this module imports only ``sys``/``torch`` at load time. The
PixelDiT submodule, ``huggingface_hub``, ``pyrallis`` and every model weight are
imported/loaded lazily inside ``__init__``. Importing this module therefore
never requires a GPU, the ``[pixeldit]`` dependency stack, or downloaded
weights -- so ``escher`` still imports cleanly in a FLUX-only environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .base import Backbone, Cond, State

# Submodule layout: <repo>/third_party/PixelDiT/t2i holds the ``diffusion``
# package (``from diffusion import DPMS`` etc.). Resolved relative to this file
# so it works regardless of the caller's working directory.
_T2I_DIR = Path(__file__).resolve().parents[2] / "third_party" / "PixelDiT" / "t2i"

# The stage-3 (multiscale ~1024px) pixel-diffusion config shipped in the submodule.
_DEFAULT_CONFIG = _T2I_DIR / "configs" / "PixelDiT_1024px_pixel_diffusion_stage3.yaml"

# Weights filename published in the HF repo; resolved via hf_hub_download.
_WEIGHTS_FILENAME = "pixeldit_t2i_v1.pth"


def _ensure_t2i_on_path() -> None:
    """Put the submodule's ``t2i`` dir on ``sys.path`` (idempotent, lazy)."""
    p = str(_T2I_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


class PixelDiTBackbone(Backbone):
    """PixelDiT backbone: pixel-space flow matching driven by the flow DPM-Solver.

    Working space is the RGB image itself -- a ``(1, 3, H, W)`` tensor in
    ``[-1, 1]`` -- so ``to_pixels``/``from_pixels`` are no-ops and the sampler's
    pixel-space conformal warps apply directly to ``State.latents_or_pixels``.
    """

    def __init__(
        self,
        model_id: str = "nvidia/PixelDiT-1300M-1024px",
        config: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: str = "cuda",
        num_inference_steps: int = 50,
    ):
        # --- lazy heavy imports (keep the module import GPU-/dep-free) --------
        _ensure_t2i_on_path()
        from huggingface_hub import hf_hub_download
        import pyrallis

        from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
        from diffusion.model.utils import get_weight_dtype
        from diffusion.utils.config import PixDiTConfig, model_init_config

        torch.set_grad_enabled(False)

        config_path = str(config) if config is not None else str(_DEFAULT_CONFIG)
        # ``args=[]`` -> ignore process argv; load purely from the YAML file.
        cfg = pyrallis.parse(config_class=PixDiTConfig, config_path=config_path, args=[])

        self.config = cfg
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_type = "classifier-free"
        self.interval_guidance = [0, 1]

        # Weight dtype: honour an explicit override, else the config's precision.
        self.weight_dtype = dtype if dtype is not None else get_weight_dtype(cfg.model.mixed_precision)
        self.flow_shift = float(cfg.scheduler.flow_shift)

        self.image_size = int(cfg.model.image_size)
        self.max_sequence_length = int(cfg.text_encoder.model_max_length)

        # --- text encoder + tokenizer ---------------------------------------
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            name=cfg.text_encoder.text_encoder_name, device=device
        )

        # Null (unconditional) embedding, computed once (prompt-independent).
        null_token = self.tokenizer(
            "",
            max_length=self.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        self.null_caption_embs = self.text_encoder(
            null_token.input_ids, null_token.attention_mask
        )[0]

        # --- diffusion model -------------------------------------------------
        model_kwargs = model_init_config(cfg, latent_size=self.image_size)
        model = build_model(
            cfg.model.model,
            use_fp32_attention=cfg.model.get("fp32_attention", False),
            **model_kwargs,
        ).to(device)

        weights_path = hf_hub_download(model_id, _WEIGHTS_FILENAME)
        state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
        if isinstance(state_dict, dict) and "state_dict" not in state_dict:
            # ``.bin``-style flat checkpoint: wrap to match the expected layout.
            state_dict = {"state_dict": state_dict}
        # ``pos_embed`` is resolution-dependent and rebuilt by the model.
        if "pos_embed" in state_dict["state_dict"]:
            del state_dict["state_dict"]["pos_embed"]
        model.load_state_dict(state_dict["state_dict"], strict=False)
        model.eval().to(self.weight_dtype)
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model

    # ------------------------------------------------------------------ VAE
    @torch.inference_mode()
    def decode_state(self, state: State) -> Tensor:
        """Return the settled working tensor as pixels (identity; no extra step)."""
        return self.to_pixels(state.latents_or_pixels)

    @torch.inference_mode()
    def to_pixels(self, latents: Tensor) -> Tensor:
        """Identity: the working tensor already *is* the pixel image ([-1, 1])."""
        return latents

    @torch.inference_mode()

    def from_pixels(self, pixels: Tensor) -> Tensor:
        """Identity: no VAE in a pixel-space model."""
        return pixels

    # -------------------------------------------------------------- prompting
    @torch.inference_mode()
    def encode_prompt(self, prompt: str, cfg_scale: float) -> Cond:
        """Encode ``prompt`` into conditioning + null embeddings for CFG.

        Mirrors the reference inference path: optional CHI prompt-prefixing, the
        gemma decoder embeddings sliced by ``select_index``, and the matching
        attention mask. The uncond branch reuses the cached null embedding.
        """
        cfg = self.config
        device = self.device
        model_max_length = self.max_sequence_length

        if not cfg.text_encoder.chi_prompt:
            max_length_all = model_max_length
            prompts_all = [prompt]
        else:
            chi_prompt = "\n".join(cfg.text_encoder.chi_prompt)
            prompts_all = [chi_prompt + prompt]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            # magic number 2: [bos], [_] (matches upstream inference).
            max_length_all = num_chi_prompt_tokens + model_max_length - 2

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        select_index = [0] + list(range(-model_max_length + 1, 0))
        caption_embs = self.text_encoder(
            caption_token.input_ids, caption_token.attention_mask
        )[0][:, None][:, :, select_index]
        emb_masks = caption_token.attention_mask[:, select_index]
        null_y = self.null_caption_embs.repeat(len(prompts_all), 1, 1)[:, None]

        return Cond(
            embeds=caption_embs,
            cfg_scale=cfg_scale,
            extra={"null_y": null_y, "mask": emb_masks},
        )

    # ---------------------------------------------------------------- schedule
    def init_state(self, size: int, seed: int) -> State:
        """Initial fp32 pixel noise + the ``FlowMatchEuler(shift)`` sigma grid (square ``size``).

        Mirrors the reference ``pixeldit_pipe`` exactly:
        ``FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
        shift=flow_shift).set_timesteps(num_steps)`` gives the decreasing sigma
        grid (``sigmas[0] ~= 1`` ... ``sigmas[-1] = 0``, length ``num_steps + 1``)
        and ``timesteps = sigma * 1000`` (length ``num_steps``). The initial noise
        is drawn in **float32** (not the model's bf16 weight dtype) and the working
        tensor stays fp32 end to end -- the model input is down-cast to bf16 only
        inside the forward pass (see ``_velocity``).
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        gen = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(
            (1, 3, size, size),
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=self.flow_shift
        )
        scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        sigmas = scheduler.sigmas.to(self.device)
        timesteps = scheduler.timesteps.to(self.device)

        return State(
            latents_or_pixels=x,
            sigmas=sigmas,
            timesteps=timesteps,
            gen=gen,
            height=size,
            width=size,
        )

    # ------------------------------------------------------------------ velocity
    def _velocity(self, x: Tensor, timestep: Tensor, cond: Cond) -> Tensor:
        """Two-branch classifier-free velocity, ported verbatim from
        ``pixeldit_pipe.velocity``: cast the input to the model's weight dtype for
        the forward, run one batched ``[uncond, cond]`` call on
        ``forward_with_dpmsolver`` and combine ``v = v_u + g*(v_c - v_u)`` in
        float32. ``timestep`` is the model time ``sigma * 1000``; when ``g == 1`` or
        ``sigma`` is outside ``(0, 1)`` (i.e. the first step at ``sigma == 1``) it
        runs the conditional branch only (matches the DPMS ``interval_guidance``
        gate). ``mask`` is the conditional mask, passed for both branches."""
        h, w = int(x.shape[-2]), int(x.shape[-1])
        info = {
            "img_hw": torch.tensor([[h, w]], device=x.device).float(),
            "aspect_ratio": torch.tensor([[w / h]], device=x.device).float(),
        }
        t = timestep.reshape(1).to(x.device).float()
        xm = x.to(self.weight_dtype)
        g = cond.cfg_scale
        fwd = self.model.forward_with_dpmsolver
        mask = cond.extra["mask"]
        if g == 1.0 or not (0.0 < float(t) / 1000.0 < 1.0):
            out = fwd(xm, t, cond.embeds, data_info=info, mask=mask)
            return (out[0] if isinstance(out, (tuple, list)) else out).float()
        out = fwd(
            torch.cat([xm, xm]),
            t.expand(2),
            torch.cat([cond.extra["null_y"], cond.embeds]),
            data_info=info,
            mask=mask,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        v_u, v_c = out.float().chunk(2)
        return v_u + g * (v_c - v_u)

    # ------------------------------------------------------------- sampling ops
    @torch.inference_mode()
    def predict_x0(self, state: State, cond: Cond, step_i: int) -> Tensor:
        """Clean pixel prediction ``x0 = x - sigma * v`` at ``sigmas[step_i]``
        (identity VAE, so ``x0`` is already ``(1, 3, H, W)`` pixels in ``[-1, 1]``),
        with the two-branch CFG velocity. Reference ``sampling_core`` contract."""
        x = state.latents_or_pixels
        sigma = state.sigmas[step_i]
        v = self._velocity(x, state.timesteps[step_i], cond)
        return x - sigma * v

    @torch.inference_mode()

    def renoise(self, state: State, x0_pixels: Tensor, step_i: int) -> State:
        """Flow re-noise to ``sigmas[step_i + 1]`` in pixel space:
        ``(1 - sigma_next) * x0 + sigma_next * noise`` with fresh noise from
        ``gen`` -- the FLUX backbone's convention, VAE-free."""
        sigma_next = state.sigmas[step_i + 1]   # fp32 tensor (match reference, not float())
        noise = torch.randn(
            x0_pixels.shape,
            generator=state.gen,
            device=x0_pixels.device,
            dtype=x0_pixels.dtype,
        )
        state.latents_or_pixels = (1.0 - sigma_next) * x0_pixels + sigma_next * noise
        return state

    @torch.inference_mode()
    def renoise_latent(self, state: State, step_i: int) -> State:
        """Re-noise the current working tensor in place to ``sigmas[step_i]``
        (pixel space is the working space here): ``(1 - sigma) * z + sigma * noise``.
        Used by post-loop time travel; no warp/round-trip, one draw from ``gen``."""
        z = state.latents_or_pixels
        sigma = state.sigmas[step_i]   # fp32 tensor (match reference, not float())
        noise = torch.randn(z.shape, generator=state.gen, device=z.device, dtype=z.dtype)
        state.latents_or_pixels = (1.0 - sigma) * z + sigma * noise
        return state

    @torch.inference_mode()

    def denoise_block(
        self, state: State, cond: Cond, from_i: int, to_i: int, fp32: bool = True
    ) -> State:
        """Advance ordinary diffusion over ``[from_i, to_i)`` with plain fp32
        flow-matching Euler steps: ``x <- x + (sigma_next - sigma) * v`` per step,
        the velocity from the two-branch CFG call. This reproduces the reference's
        ``pipe.scheduler.step`` (FlowMatchEuler, fp32) -- latents are fp32 and the
        velocity is fp32, so the step is fully fp32. The ``fp32`` flag is accepted
        for protocol parity but ignored: PixelDiT is fp32 end to end (including the
        time-travel tail, unlike FLUX whose tail is plain fp16)."""
        x = state.latents_or_pixels
        for i in range(from_i, to_i):
            sigma = state.sigmas[i]
            sigma_next = state.sigmas[i + 1]
            v = self._velocity(x, state.timesteps[i], cond)
            x = x + (sigma_next - sigma) * v
        state.latents_or_pixels = x
        return state
