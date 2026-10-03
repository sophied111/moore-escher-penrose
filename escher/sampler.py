"""escher.sampler -- the braided T-cycle sampler (the integration crux).

This is where every earlier phase is assembled into the paper's recipe: an
ordinary diffusion trajectory is *braided* with pixel-space conformal warps so
the frozen backbone renders a recursive (Print-Gallery / Droste) image. The
loop collapses the reference ``run_denoising`` denoise loop and the
``TCycleModifier.modify_x0`` step-modifier into one direct function that speaks
only three vocabularies:

* the :class:`~escher.backbones.base.Backbone` protocol (Phase 2) -- the model
  is driven exclusively through ``encode_prompt`` / ``init_state`` /
  ``predict_x0`` / ``renoise`` / ``denoise_block``; the sampler never imports
  diffusers, a scheduler or a VAE;
* :mod:`escher.transforms` (Phase 1) -- ``get_family(name)`` returns a
  ``TransformFamily`` instance (``.T`` / ``.Tp``) for the operator window, and
  the warm-up uses ``ConformalFamily.T`` with ``periods=0`` (untwisted Droste;
  ``WARMUP_CONF_ZOOM`` lives in ``escher/transforms/base.py``); and
* :mod:`escher.schedule` (Task 3.1) -- ``build_op_steps`` / ``build_warmup_steps``
  decide *which* step indices get a warp.

The three phases of a run, walked in one pass over ``0..num_steps-1``:

1. **Warm-up** (steps resolved from ``warmup_sigma`` up to the first op step):
   predict the clean image, apply an *untwisted* conformal Droste zoom
   (``periods=0``, ``zoom=WARMUP_CONF_ZOOM``, no super-resolution) and re-noise.
   Always conformal ``periods=0`` regardless of ``family`` -- matching the
   reference ``_droste_steps`` always-conformal rule.
2. **Operator window** (the ``build_op_steps`` schedule): predict, optionally
   super-resolve (lazy, Phase 4), apply the family's ``T`` or ``T†`` per the
   schedule label, re-noise.
3. **Ordinary diffusion** (every other step): a plain ``denoise_block``.

Then the paper's post-loop **time travel** (optional): re-noise the finished
estimate back to ``tt_sigma`` and re-denoise the schedule tail ``tt_n`` times
with *no* operators -- pure refinement that sharpens the soft pixel-roundtrip
output without changing global structure (ported from ``sampling_core`` 228-256).

Fresh noise is drawn *only* inside ``backbone.renoise`` (paper section "Updating
the noisy state"); ordinary steps never re-noise.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from escher.monitor import Monitor
from escher.schedule import build_op_steps, build_warmup_steps, sigma_to_step
from escher.transforms import get_family, WARMUP_CONF_ZOOM


def _sigmas_of(state: Any) -> Any:
    """Read ``sigmas`` from a backbone state (dataclass attribute or dict key).

    ``State`` (Phase 2) exposes ``sigmas`` as an attribute; a lightweight
    dict-based fake backbone exposes it as a key. The sampler treats the state
    as otherwise opaque, so this is the only field it needs to read directly.
    """
    if isinstance(state, dict):
        return state["sigmas"]
    return state.sigmas


def _to_pil(x0_pixels: torch.Tensor) -> Image.Image:
    """Convert a ``(1, 3, H, W)`` pixel tensor in ``[-1, 1]`` to a PIL image.

    Byte-for-byte matches ``diffusers`` ``VaeImageProcessor.postprocess`` +
    ``numpy_to_pil`` (the reference's final image path), so a byte-identical
    latent yields a byte-identical PNG. That means, in order: denormalize
    ``x / 2 + 0.5`` in the tensor's *native* dtype (the VAE decode dtype, e.g.
    fp16 -- NOT up-cast to float32 first, and ``x/2 + 0.5`` not ``(x+1)/2``),
    clamp to ``[0, 1]``, move to a float32 NumPy array, then ``(a*255).round()``
    with NumPy's round-half-to-even and cast to uint8. Doing the denormalize in
    float32 or rounding in torch shifts ~4% of pixels by 1 LSB.
    """
    x = x0_pixels.detach()
    x = (x / 2 + 0.5).clamp(0, 1)                      # denormalize, native dtype
    arr = x.cpu().permute(0, 2, 3, 1).float().numpy()  # (1, H, W, 3) float32
    arr = (arr * 255).round().astype("uint8")[0]       # (H, W, 3) uint8
    return Image.fromarray(arr)


@torch.inference_mode()


def sample(
    backbone: Any,
    prompt: str,
    *,
    family: str = "conformal",
    inset_scale: float = 1 / 4,
    periods: int = 1,
    sigma_hi: float = 0.87,
    sigma_lo: float = 0.5,
    n_ops: int = 3,
    op_gap: int = 9,
    warmup_sigma: Optional[float] = 0.95,
    focus: Tuple[float, float] = (0.5, 0.5),
    upres: str = "off",
    zoom: float = 0.5,
    supersample: int = 2,
    time_travel: Optional[Tuple[int, float]] = None,
    seed: int = 0,
    size: int = 1024,
    cfg_scale: float = 3.5,
    family_params: Optional[dict] = None,
    monitor_every: int = 0,
    monitor_path: Optional[str] = None,
) -> Image.Image:
    """Run the braided warm-up -> T/T† window -> time-travel sampling loop.

    Args:
        backbone: any object satisfying the :class:`Backbone` protocol.
        prompt: text prompt (encoded once via ``backbone.encode_prompt``).
        family: transform family name for the operator window (``get_family``);
            the warm-up is always conformal ``periods=0`` regardless of this.
        inset_scale: paper inset ratio ``1/lambda`` passed to every warp.
        periods: twist periods ``p`` for the operator-window warps.
        sigma_hi, sigma_lo: sigma-unit bounds of the operator window.
        n_ops, op_gap: operator-schedule shape (see ``build_op_steps``).
        warmup_sigma: sigma at which the untwisted-Droste warm-up begins, or
            ``None`` to disable the warm-up phase entirely.
        focus: normalized fixed-point pan shared by every warp.
        upres: super-resolution mode for the op window (``"off"`` disables the
            lazy Phase-4 SR path entirely).
        zoom: radial zoom for the operator-window warps.
        supersample: antialiasing supersample factor for every warp.
        time_travel: ``None`` or ``(tt_n, tt_sigma)`` -- re-noise the finished
            estimate to ``tt_sigma`` and re-denoise the tail ``tt_n`` times.
        seed: RNG seed forwarded to ``backbone.init_state``.
        size: output pixel side length (images are square, ``size`` x ``size``).
        cfg_scale: guidance scale forwarded to ``backbone.encode_prompt``.
        family_params: extra keyword arguments forwarded to the family's
            ``T``/``T†`` (e.g. poles' ``centers``, mobius'/square's ``k``/``l``).
        monitor_every: ``0`` (default) disables the diagnostic monitor. ``N > 0``
            captures the decoded x0 at every warm-up step, before AND after
            every operator warp, every ``N``-th ordinary denoise step, the
            pre/post time-travel frames and the final image, and writes them as
            one labelled montage PNG. Purely observational: no RNG draws, no
            change to the sampling math or the returned image.
        monitor_path: montage output path (default ``escher_monitor.png``; the
            CLI derives ``<output_stem>_monitor.png``). Ignored when disabled.

    Returns:
        The decoded image as a :class:`PIL.Image.Image` of size ``(size, size)``.
    """
    cond = backbone.encode_prompt(prompt, cfg_scale)
    state = backbone.init_state(size, seed)
    sigmas = _sigmas_of(state)
    num_steps = len(sigmas) - 1

    # --- schedule: which steps warp, and how -------------------------------
    op_steps = build_op_steps(
        sigmas=sigmas,
        sigma_hi=sigma_hi,
        sigma_lo=sigma_lo,
        n_ops=n_ops,
        op_gap=op_gap,
    )
    first_op = min(op_steps)
    # warmup_sigma=None disables the warm-up phase entirely (e.g. the rimrings
    # single-T recipe runs no untwisted-Droste warm-up).
    warmup = set() if warmup_sigma is None else set(build_warmup_steps(
        sigmas=sigmas,
        warmup_sigma=warmup_sigma,
        first_op_step=first_op,
    ))

    op_fam = get_family(family)
    warmup_fam = get_family("conformal")
    fam_kw = family_params or {}

    mon = Monitor(monitor_path or "escher_monitor.png", monitor_every) if monitor_every > 0 else None

    def _sig(k: int) -> float:
        return float(sigmas[k])

    # --- the braided loop --------------------------------------------------
    for i in range(num_steps):
        if i in warmup:
            # Untwisted Droste warm-up: always conformal periods=0, no SR.
            x0 = backbone.predict_x0(state, cond, i)
            # Reference parity (step_modifiers.modify_x0): decode then
            # ``out = x0_pixels.float()`` -- every pixel-space warp runs in
            # float32 regardless of the VAE's fp16/bf16 decode dtype, so the
            # bilinear resample is bit-for-bit the reference's. ``renoise`` casts
            # back to the latent dtype on re-encode.
            x0 = x0.float()
            if mon is not None:
                mon.capture(x0, i, _sig(i), "warmup T0")
            x0 = warmup_fam.T(
                x0,
                inset_scale=inset_scale,
                periods=0,
                out=size,
                zoom=WARMUP_CONF_ZOOM,
                supersample=supersample,
                focus=focus,
            )
            # Reference parity (step_modifiers.modify_x0): the warped pixels are
            # clamped to [-1, 1] *before* the re-encode -- ``out.clamp(-1, 1)`` in
            # float32, then cast to the VAE dtype. The warp overshoots [-1, 1]
            # (border pad + bilinear), so without this clamp the re-encoded z0
            # diverges on the overshoot pixels and the whole trajectory drifts.
            x0 = x0.clamp(-1.0, 1.0)
            state = backbone.renoise(state, x0, i)

        elif i in op_steps:
            x0 = backbone.predict_x0(state, cond, i)
            # Reference parity: pixel-space warps run in float32 (see warm-up).
            x0 = x0.float()
            fn = op_fam.T if op_steps[i] == "T" else op_fam.Tp
            if mon is not None:
                mon.capture(x0, i, _sig(i), f"pre {op_steps[i]}")

            # Only the conformal family's T/T† accept ``periods`` (the twist
            # exponent p). The other families (mobius/poles/square/rimrings)
            # take their twist from ``family_params`` (k/l/centers/A/R0, all
            # defaulted) and raise ``TypeError`` on an unexpected ``periods``
            # kwarg -- so route ``periods`` to conformal only.
            twist_kw = {"periods": periods} if op_fam.name == "conformal" else {}

            if upres != "off":
                # Ruling P1: lazy, function-local import -- escher.sr is an
                # optional Phase-4 module; the sampler must import with it
                # absent. SR-upscale -> warp at the upscaled resolution ->
                # area-down back to the target size.
                from escher.sr import upscale4

                x0 = upscale4(x0, mode=upres)
                out_up = x0.shape[-1]
                x0 = fn(
                    x0,
                    inset_scale=inset_scale,
                    out=out_up,
                    supersample=supersample,
                    zoom=zoom,
                    focus=focus,
                    **twist_kw,
                    **fam_kw,
                )
                x0 = F.interpolate(x0, size=(size, size), mode="area")
            else:
                x0 = fn(
                    x0,
                    inset_scale=inset_scale,
                    out=size,
                    supersample=supersample,
                    zoom=zoom,
                    focus=focus,
                    **twist_kw,
                    **fam_kw,
                )

            if mon is not None:
                mon.capture(x0, i, _sig(i), f"post {op_steps[i]}")
            # Reference parity: clamp warped pixels to [-1, 1] before re-encode.
            x0 = x0.clamp(-1.0, 1.0)
            state = backbone.renoise(state, x0, i)

        else:
            if mon is not None and mon.due(i):
                mon.capture(backbone.predict_x0(state, cond, i), i, _sig(i), "denoise")
            state = backbone.denoise_block(state, cond, i, i + 1)

    # --- post-loop time travel (paper's refinement pass) -------------------
    # Reference (sampling_core time travel): re-noise the finished LATENT directly
    # to sigma[i_tt] -- ``(1 - sigma) * z + sigma * noise`` -- then re-denoise the
    # schedule tail [i_tt, num_steps) with NO operators. It does NOT decode to
    # pixels and re-encode, and does NOT run an extra transformer/predict_x0 pass;
    # a pixel round trip here (lossy VAE decode+encode) flattens contrast and
    # washes out the final image. Drive it entirely in latent space via
    # ``renoise_latent``. The single RNG draw matches ``renoise``'s.
    if time_travel is not None:
        tt_n, tt_sigma = time_travel
        i_tt = sigma_to_step(sigmas, tt_sigma)
        for k in range(tt_n):
            if mon is not None and k == 0:
                mon.capture(backbone.decode_state(state),
                            num_steps - 1, _sig(num_steps - 1), "pre TT")
            state = backbone.renoise_latent(state, i_tt)
            # Reference time-travel tail uses plain fp16 Euler (not the scheduler),
            # so fp32=False here (ordinary steps above keep the scheduler's fp32).
            state = backbone.denoise_block(state, cond, i_tt, num_steps, fp32=False)

    # --- decode final clean estimate ---------------------------------------
    # Decode the settled final latent directly (reference:
    # ``decode_latents(pipe, latents)`` on the loop's final ``latents``). The
    # last denoise step already advanced to sigma=0, so the final latent *is*
    # the clean x0; re-running ``predict_x0`` here would apply a spurious extra
    # transformer correction and diverge from the reference image.
    x0 = backbone.decode_state(state)
    if mon is not None:
        tt_ran = time_travel is not None and time_travel[0] > 0
        mon.capture(x0, num_steps - 1, _sig(num_steps - 1),
                    "post TT / final" if tt_ran else "final")
        mon.save()
    return _to_pil(x0)
