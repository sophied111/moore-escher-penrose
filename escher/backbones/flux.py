"""escher.backbones.flux -- FLUX.1-dev adapter implementing :class:`Backbone`.

Ports the FLUX-only branches of the reference pipeline helpers into the
sampler-facing protocol: prompt encoding with guidance distillation, a
mu-shifted flow-matching schedule, latent init, a single packed transformer
call, the ``x0 = latents - sigma * v`` clean prediction, VAE decode/encode, the
``(1 - sigma_next) * z0 + sigma_next * noise`` re-noise, and a FlowMatch Euler
denoise block.

Import safety: this module imports only ``torch`` at load time. ``FluxPipeline``
(and therefore any model weights / CUDA context) is imported lazily inside
``__init__`` -- importing this module never requires a GPU or a model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Backbone, Cond, State


class FluxBackbone(Backbone):
    """FLUX.1-dev backbone: latent-space flow matching with guidance distillation.

    Working space is the unpacked ``(1, 16, H/vae_sf, W/vae_sf)`` VAE latent;
    packing to the transformer's token layout happens inside each transformer
    call, so ``State.latents_or_pixels`` is always a plain 4-D spatial tensor.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_inference_steps: int = 28,
    ):
        # Lazy heavy import: keeps module import GPU-/model-free.
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.transformer.eval()
        self._freeze(pipe)
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps

    @classmethod
    def from_pipe(cls, pipe, num_inference_steps: int = 28) -> "FluxBackbone":
        """Wrap an already-loaded ``FluxPipeline`` (frozen + eval'd) as a backbone."""
        self = cls.__new__(cls)
        pipe.transformer.eval()
        cls._freeze(pipe)
        self.pipe = pipe
        self.device = next(pipe.transformer.parameters()).device
        self.dtype = pipe.transformer.dtype
        self.num_inference_steps = num_inference_steps
        return self

    @staticmethod
    def _freeze(pipe) -> None:
        for param in pipe.transformer.parameters():
            param.requires_grad_(False)
        for param in pipe.vae.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ VAE
    @torch.inference_mode()
    def decode_state(self, state: State) -> Tensor:
        """Decode the settled final latent directly (VAE decode, no extra step)."""
        return self.to_pixels(state.latents_or_pixels)

    @torch.inference_mode()
    def to_pixels(self, latents: Tensor) -> Tensor:
        """Decode latents to a ``(1, 3, H, W)`` pixel image in ``[-1, 1]``."""
        sf = self.pipe.vae.config.scaling_factor
        sh = getattr(self.pipe.vae.config, "shift_factor", 0.0)
        return self.pipe.vae.decode(latents / sf + sh, return_dict=False)[0]

    @torch.inference_mode()

    def from_pixels(self, pixels: Tensor) -> Tensor:
        """Encode a ``[-1, 1]`` pixel image to normalised latents (posterior mode)."""
        sf = self.pipe.vae.config.scaling_factor
        sh = getattr(self.pipe.vae.config, "shift_factor", 0.0)
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        z = self.pipe.vae.encode(pixels.to(vae_dtype)).latent_dist.mode()
        return (z - sh) * sf

    # -------------------------------------------------------------- prompting
    @torch.inference_mode()
    def encode_prompt(self, prompt: str, cfg_scale: float) -> Cond:
        """Encode ``prompt`` for FLUX (positive-only; guidance via distillation)."""
        embeds, pooled, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
        )
        guidance = None
        if getattr(self.pipe.transformer.config, "guidance_embeds", False):
            guidance = torch.tensor(
                [cfg_scale], device=embeds.device, dtype=embeds.dtype
            )
        return Cond(
            embeds=embeds,
            pooled=pooled,
            text_ids=text_ids,
            guidance=guidance,
            cfg_scale=cfg_scale,
        )

    # ------------------------------------------------------------- transformer
    def _call_transformer(
        self, latents: Tensor, timestep: Tensor, cond: Cond, height: int, width: int
    ) -> Tensor:
        """One transformer forward: pack -> call -> unpack; velocity in unpacked
        ``(1, 16, lh, lw)`` space (FLUX guidance-distillation call, no CFG concat)."""
        pipe = self.pipe
        lh = height // pipe.vae_scale_factor
        lw = width // pipe.vae_scale_factor
        packed = pipe._pack_latents(latents, 1, latents.shape[1], lh, lw)
        t_in = (timestep / 1000).expand(packed.shape[0]).to(packed.dtype)
        img_ids = pipe._prepare_latent_image_ids(
            1, lh // 2, lw // 2, latents.device, packed.dtype
        )
        kwargs = dict(
            hidden_states=packed,
            timestep=t_in,
            encoder_hidden_states=cond.embeds,
            pooled_projections=cond.pooled,
            txt_ids=cond.text_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        if cond.guidance is not None:
            kwargs["guidance"] = cond.guidance.expand(packed.shape[0])
        v_packed = pipe.transformer(**kwargs)[0]
        return pipe._unpack_latents(v_packed, height, width, pipe.vae_scale_factor)

    # ---------------------------------------------------------------- schedule
    def _set_timesteps(self, height: int, width: int) -> None:
        """Set the FLUX flow-matching schedule with resolution-dependent mu-shift."""
        pipe = self.pipe
        lh = height // pipe.vae_scale_factor
        lw = width // pipe.vae_scale_factor
        image_seq_len = (lh // 2) * (lw // 2)
        s = pipe.scheduler.config
        # Linear shift in log-space -- matches diffusers' calculate_shift().
        m = (s.max_shift - s.base_shift) / (
            math.log(s.max_image_seq_len) - math.log(s.base_image_seq_len)
        )
        mu = m * math.log(image_seq_len) + (
            s.base_shift - m * math.log(s.base_image_seq_len)
        )
        pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device, mu=mu)

    def init_state(self, size: int, seed: int) -> State:
        """Initial latent noise + mu-shifted schedule for a square ``size`` run."""
        pipe = self.pipe
        gen = torch.Generator(device=self.device).manual_seed(seed)
        ls = size // pipe.vae_scale_factor
        latents = torch.randn(
            (1, 16, ls, ls),
            generator=gen,
            device=self.device,
            dtype=pipe.transformer.dtype,
        )
        self._set_timesteps(size, size)
        sigmas = pipe.scheduler.sigmas.to(self.device)
        timesteps = pipe.scheduler.timesteps.to(self.device)
        return State(
            latents_or_pixels=latents,
            sigmas=sigmas,
            timesteps=timesteps,
            gen=gen,
            height=size,
            width=size,
        )

    # ------------------------------------------------------------- sampling ops
    @torch.inference_mode()
    def predict_x0(self, state: State, cond: Cond, step_i: int) -> Tensor:
        """Clean prediction ``x0 = latents - sigma * v``, decoded to pixels."""
        latents = state.latents_or_pixels
        sigma = state.sigmas[step_i]
        timestep = state.timesteps[step_i]
        v = self._call_transformer(latents, timestep, cond, state.height, state.width)
        x0_latents = latents - sigma * v
        return self.to_pixels(x0_latents)

    @torch.inference_mode()

    def renoise(self, state: State, x0_pixels: Tensor, step_i: int) -> State:
        """Encode ``x0_pixels`` and re-noise to ``sigma_next``:
        ``(1 - sigma_next) * z0 + sigma_next * noise`` (fresh noise from ``gen``)."""
        z0 = self.from_pixels(x0_pixels)
        sigma_next = state.sigmas[step_i + 1]
        noise = torch.randn(
            z0.shape, generator=state.gen, device=z0.device, dtype=z0.dtype
        )
        state.latents_or_pixels = (1 - sigma_next) * z0 + sigma_next * noise
        return state

    @torch.inference_mode()
    def renoise_latent(self, state: State, step_i: int) -> State:
        """Re-noise the current latent in place to ``sigmas[step_i]`` with no VAE
        round trip: ``(1 - sigma) * z + sigma * noise`` (reference time-travel,
        ``sampling_core`` ``(1 - sigma_tt) * latents + sigma_tt * randn_like``)."""
        z = state.latents_or_pixels
        sigma = state.sigmas[step_i]
        noise = torch.randn(z.shape, generator=state.gen, device=z.device, dtype=z.dtype)
        state.latents_or_pixels = (1 - sigma) * z + sigma * noise
        return state

    @torch.inference_mode()

    def denoise_block(
        self, state: State, cond: Cond, from_i: int, to_i: int, fp32: bool = True
    ) -> State:
        """FlowMatch Euler over ``[from_i, to_i)``: ``x <- x + (sigma_next - sigma) * v``.

        ``fp32=True`` (ordinary steps) reproduces ``diffusers``
        ``FlowMatchEulerDiscreteScheduler.step`` bit-for-bit: upcast the sample to
        float32, add ``dt * v`` (``dt = sigma_next - sigma``), then cast the result
        back to ``v``'s dtype -- i.e. the latent is re-quantized to fp16 every step
        (scheduling_flow_match_euler_discrete.py:486/506/513/519). ``fp32=False``
        is the plain fp16 Euler the reference uses in the time-travel tail
        (``sampling_core`` re-denoise), which does not go through the scheduler."""
        latents = state.latents_or_pixels
        for i in range(from_i, to_i):
            sigma = state.sigmas[i]
            sigma_next = state.sigmas[i + 1]
            timestep = state.timesteps[i]
            v = self._call_transformer(
                latents, timestep, cond, state.height, state.width
            )
            if fp32:
                latents = (latents.to(torch.float32) + (sigma_next - sigma) * v).to(v.dtype)
            else:
                latents = latents + (sigma_next - sigma) * v
        state.latents_or_pixels = latents
        return state
