"""escher.backbones.base -- the ``Backbone`` protocol and its data containers.

A *backbone* is a diffusion model wrapped in a small, sampler-facing interface.
The braided sampler (Task 3.2) never touches a pipeline, a scheduler, or a VAE
directly: it drives the model **only** through the methods declared here. This
keeps the sampler model-agnostic, so the very same loop runs on:

* :class:`escher.backbones.flux.FluxBackbone` -- a latent-space flow-matching
  model (FLUX.1-dev), where ``State.latents_or_pixels`` holds packed-VAE
  latents and ``to_pixels``/``from_pixels`` are the real VAE decode/encode; and
* a future pixel-space DiT backbone, where the "latent" *is* the image, the VAE
  is the identity, and ``to_pixels``/``from_pixels`` are no-ops.

Because both satisfy the same protocol, the operator-braiding sampler applies
its conformal warps in **pixel space** (the space every backbone agrees on) via
``predict_x0`` -> warp -> ``renoise``, and advances the ordinary diffusion with
``denoise_block``.

Design constraints
------------------
* Importing this module must NOT require a GPU or a loaded model. It only needs
  ``torch`` for type aliases; no CUDA context is created and no weights load.
* ``State`` and ``Cond`` are plain dataclasses -- transparent, backbone-owned
  bags of tensors. The sampler treats ``Cond`` as opaque and reads only the
  documented ``State`` fields (``sigmas``/``timesteps`` for scheduling,
  ``gen`` for reproducible re-noising, ``height``/``width`` for geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass
class State:
    """Mutable per-generation state threaded through a backbone's methods.

    ``latents_or_pixels`` is the working tensor the backbone advances: packed or
    unpacked VAE latents for a latent model, or the pixel image itself for a
    pixel-space model. Everything else is fixed at ``init_state`` time.

    Attributes:
        latents_or_pixels: the current working tensor (shape is backbone-defined;
            for FLUX an unpacked ``(1, 16, H/vae_sf, W/vae_sf)`` latent).
        sigmas: flow-matching noise levels, shape ``(num_steps + 1,)``; the
            sampler indexes ``sigmas[i]`` / ``sigmas[i + 1]`` per step.
        timesteps: scheduler timesteps aligned with ``sigmas``, shape
            ``(num_steps,)``.
        gen: the ``torch.Generator`` that seeds every fresh-noise draw, so a run
            is reproducible from ``init_state``'s ``seed``.
        height: output image height in pixels.
        width: output image width in pixels.
    """

    latents_or_pixels: Tensor
    sigmas: Tensor
    timesteps: Tensor
    gen: torch.Generator
    height: int
    width: int


@dataclass
class Cond:
    """Encoded conditioning produced by ``encode_prompt``, opaque to the sampler.

    The fields cover FLUX's needs (prompt embeddings, pooled projections, RoPE
    text ids, and the guidance-distillation scalar); other backbones populate
    whatever subset they use and leave the rest ``None``. ``extra`` is a free
    slot for backbone-specific conditioning without changing this contract.

    Attributes:
        embeds: sequence prompt embeddings.
        pooled: pooled prompt projection.
        text_ids: RoPE positional ids for the text stream (FLUX); ``None`` when
            the model has no such ids.
        guidance: the FLUX.1-dev guidance-embedding tensor (``tensor([scale])``)
            when ``transformer.config.guidance_embeds`` is set; else ``None``.
        cfg_scale: the requested guidance scale, retained for reference.
        extra: optional backbone-specific extras.
    """

    embeds: Optional[Tensor] = None
    pooled: Optional[Tensor] = None
    text_ids: Optional[Tensor] = None
    guidance: Optional[Tensor] = None
    cfg_scale: float = 1.0
    extra: Optional[dict[str, Any]] = None


@runtime_checkable
class Backbone(Protocol):
    """Sampler-facing interface every diffusion backbone implements.

    The braided sampler composes exactly these calls: ``encode_prompt`` once,
    ``init_state`` once, then per step either ``denoise_block`` (ordinary
    diffusion) or the operator braid ``predict_x0`` -> (pixel-space warp) ->
    ``renoise``. ``to_pixels``/``from_pixels`` expose the model's image<->working
    space so callers can move between them; for a pixel-space backbone they are
    the identity.
    """

    def encode_prompt(self, prompt: str, cfg_scale: float) -> Cond:
        """Encode ``prompt`` (with guidance ``cfg_scale``) into a :class:`Cond`."""
        ...

    def init_state(self, size: int, seed: int) -> State:
        """Create initial noise + schedule for a square ``size`` x ``size`` run.

        (``State`` keeps ``height``/``width`` internally, both set to ``size``.)"""
        ...

    def predict_x0(self, state: State, cond: Cond, step_i: int) -> Tensor:
        """Clean-image prediction at ``step_i``, decoded to pixels.

        Returns a ``(1, 3, H, W)`` tensor in ``[-1, 1]`` (image space), so
        sampler-level operators act in the space every backbone shares.
        """
        ...

    def renoise(self, state: State, x0_pixels: Tensor, step_i: int) -> State:
        """Re-noise a (possibly warped) clean pixel image back to the noise
        level of the step after ``step_i`` and store it in ``state``."""
        ...

    def renoise_latent(self, state: State, step_i: int) -> State:
        """Re-noise the *current working tensor in place* to the noise level of
        ``step_i`` -- ``(1 - sigmas[step_i]) * z + sigmas[step_i] * noise`` -- with
        no decode/warp/encode round trip. Used by the post-loop time-travel pass,
        which re-noises the finished latent directly (reference
        ``sampling_core`` time travel), unlike :meth:`renoise` which re-encodes a
        warped pixel image. Draws exactly one fresh-noise tensor from ``gen``."""
        ...

    def denoise_block(
        self, state: State, cond: Cond, from_i: int, to_i: int, fp32: bool = True
    ) -> State:
        """Advance ordinary diffusion over the half-open step range
        ``[from_i, to_i)`` and store the result in ``state``. ``fp32`` selects the
        step precision: ``True`` (ordinary steps) matches the reference scheduler's
        float32 accumulate; ``False`` is the plain fp16 Euler used in the
        time-travel tail. Backbones that fix their own precision may ignore it."""
        ...

    def decode_state(self, state: State) -> Tensor:
        """Decode the current working tensor in ``state`` to a ``(1, 3, H, W)``
        pixel image in ``[-1, 1]``, without any extra model step. Used for the
        sampler's final image (the settled final latent *is* the clean x0, so this
        is a plain decode, not another ``predict_x0``). Keeps ``state`` opaque to
        the sampler (equivalent to ``to_pixels(state.latents_or_pixels)`` for the
        real backbones)."""
        ...

    def to_pixels(self, latents: Tensor) -> Tensor:
        """Map a working-space tensor to a ``(1, 3, H, W)`` pixel image in
        ``[-1, 1]`` (VAE decode for a latent model; identity for pixel space)."""
        ...

    def from_pixels(self, pixels: Tensor) -> Tensor:
        """Inverse of :meth:`to_pixels` (VAE encode for a latent model;
        identity for pixel space)."""
        ...
