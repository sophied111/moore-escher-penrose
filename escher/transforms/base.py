"""escher.transforms.base -- shared render core: mesh build, band fold, and
bilinear resample used by every transform family.

A "family" (conformal twist, poles, mobius, square, rimrings -- later tasks)
supplies only ``coord_fn``: a pure function mapping an output-mesh point
(as two real tensors, the real/imaginary parts of a complex number in
``[-1, 1]^2``) to a *complex source log-position* ``Wc = u + i*theta``,
where ``exp(u)`` is a source-frame radius and ``theta`` the corresponding
direction, both measured from the source image's own centre.

``render`` performs everything family-agnostic: building the (optionally
supersampled) output mesh, offsetting it for ``focus``, converting ``Wc``
into a source-pixel radius/direction, folding that radius into the visible
band of ``src`` (picking the representative, in-canvas copy of the position
via a floor-based band index -- the same trick every "Escher zoom" renderer
uses, ported from ``escher_zoo.escher``'s shared fold+sample tail), and
sampling with bilinear ``grid_sample`` + ``avg_pool2d`` supersampling.

``beta_p``/``L_of`` are the paper's twist-exponent building blocks: plain,
torch-free helpers a family's ``coord_fn`` closes over (e.g.
``beta_p(periods, L) * torch.log(z)``). ``render`` itself does not need a
family's periodicity ``L`` -- its own fold uses a single, family-agnostic
band width tied to the source image's own resolution
(``log(min(H, W))``), independent of whatever twist a given family bakes
into ``coord_fn``.

Fixed convention (do not vary per call): floor fold, bilinear
``grid_sample``, border padding, ``align_corners=True``.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def geom_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype the (dtype-agnostic) transform geometry runs in.

    ``torch.complex`` accepts only Half/Float/Double and ``grid_sample`` /
    complex ``log`` are not reliably available for half precision, so bfloat16
    (PixelDiT) and float16 (FLUX) inputs are handled in float32. float64 stays
    float64; float32 is unchanged. Callers restore the input dtype on output.
    """
    return torch.float64 if dtype == torch.float64 else torch.float32


def L_of(inset_scale: float) -> float:
    """L = log(lambda), where lambda = 1/inset_scale is the paper's inset ratio."""
    return math.log(1.0 / inset_scale)


def beta_p(p: float, L: float) -> complex:
    """Twist exponent beta_p = 1 - i*p*L/(2*pi) (paper convention)."""
    return 1 - 1j * p * L / (2 * math.pi)


def render(
    src: torch.Tensor,
    coord_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    out: int,
    supersample: int = 2,
    zoom: float,
    focus: Tuple[float, float] = (0.5, 0.5),
    inset_scale: float | None = None,
) -> torch.Tensor:
    """Shared fold + resample core for every transform family.

    ``coord_fn(zx, zy) -> Wc`` maps the focus-shifted output mesh (given as
    the real/imaginary parts of a complex number) to a complex source
    log-position ``Wc``. ``render`` owns everything else: the supersampled
    mesh, the focus offset (``_oz = 2*focus - 1``), the zoom radial scale
    (``rho_pre = r*zoom*min(H, W)``), the floor-band fold that keeps every
    sample inside ``src``'s bounds, and the bilinear resample.

    Args:
        src: source image, shape (1, C, H, W).
        coord_fn: output-mesh -> complex source log-position.
        out: output side length (output is always square, out x out).
        supersample: supersampling factor for antialiasing (default 2).
        zoom: radial zoom applied before folding into the source bounds.
        focus: normalized (x, y) in [0, 1] output-frame pan of the fixed point.
        inset_scale: when given, the floor-band fold uses the paper's true
            scale-period ``L = L_of(inset_scale) = log(1/inset_scale)`` instead
            of the resolution-tied default. Conformal families (which bake a
            genuine scale-periodicity into ``coord_fn``) must pass this so the
            fold picks the correct scale-equivalent copy; families without a
            defined period leave it ``None`` and keep the legacy default.

    Returns:
        Tensor of shape (1, C, out, out), in ``src``'s dtype (the geometry and
        resample run in float32 internally; see ``geom_dtype``).
    """
    _, _, H, W = src.shape
    device, in_dtype = src.device, src.dtype
    dtype = geom_dtype(in_dtype)  # geometry in float32 (float64 kept), see geom_dtype
    n = out * supersample

    lin = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    z = torch.complex(xx, yy)

    # focus offset: pan the fixed point in the output frame, before the
    # family's own coordinate map ever sees the mesh.
    oz = complex(2.0 * focus[0] - 1.0, 2.0 * focus[1] - 1.0)
    z = z - oz
    z = torch.where(z == 0, torch.full_like(z, 1e-12), z)

    # ``coord_fn`` returns the *complex exponent* ``Wc = beta_p * log(z)`` (the
    # conformal family builds it as ``alpha * log(complex(zx, zy))``). Recover
    # the source radius/direction the reference's exact way -- form the complex
    # ``w = exp(Wc)`` and read ``r = |w|`` / ``ang = arg(w)`` -- rather than
    # taking ``r = exp(Re Wc)`` / ``theta = Im Wc`` directly. The two are
    # mathematically equal (``cos``/``sin`` are 2*pi-periodic, so the ``angle``
    # wrap is irrelevant), but only the ``exp -> abs/angle`` route reproduces
    # the reference ``droste_conformal.T`` bit-for-bit; the direct route drifts
    # by ~5e-7, which the fp16 VAE re-encode amplifies to a divergent trajectory.
    Wc = coord_fn(z.real, z.imag)
    w = torch.exp(Wc)
    r = w.abs().clamp_min(1e-12)
    ang = w.angle()
    cos, sin = torch.cos(ang), torch.sin(ang)

    fx, fy = (W - 1) / 2.0, (H - 1) / 2.0
    rx = torch.where(
        cos >= 0,
        (W - 1 - fx) / cos.clamp_min(1e-9),
        -fx / cos.clamp_max(-1e-9),
    )
    ry = torch.where(
        sin >= 0,
        (H - 1 - fy) / sin.clamp_min(1e-9),
        -fy / sin.clamp_max(-1e-9),
    )
    r_max = torch.minimum(rx, ry) * 0.995
    rho = r * zoom * min(W, H)

    # floor-band fold: pick the largest in-canvas scale-equivalent copy of
    # ``rho``. Fold at the true scale-period ``s = 1/inset_scale`` when a family
    # supplies it (conformal), else a resolution-tied fallback. The op sequence
    # (``floor(log(r_max/rho)/log(s))`` then ``rho * s**k``) mirrors the
    # reference ``T`` exactly -- ``log(a/b)`` not ``log a - log b``, and ``s**k``
    # not ``exp(k*log s)`` -- to stay bit-identical.
    s_fold = (1.0 / inset_scale) if inset_scale is not None else max(min(H, W), 2)
    k = torch.floor(torch.log(r_max / rho) / math.log(s_fold))
    rho = rho * s_fold ** k

    grid = torch.stack(
        [
            (fx + rho * cos) / (W - 1) * 2 - 1,
            (fy + rho * sin) / (H - 1) * 2 - 1,
        ],
        dim=-1,
    )[None]

    o = F.grid_sample(
        src.to(dtype), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return F.avg_pool2d(o, supersample).to(in_dtype)


from abc import ABC, abstractmethod


# Warm-up radial zoom, applied to the untwisted-Droste warm-up for EVERY window
# family (the sampler's warm-up is always conformal p=0, but this value is a
# sampler-level knob, not a property of the conformal family). It is the
# reference's effective_zoom clamp of the requested 1.3 at the 1024^2 warm-up
# raster, centred focus; baked in rather than ported (see the design spec).
WARMUP_CONF_ZOOM = 0.49701416015625


class TransformFamily(ABC):
    """A paper transform family: a matched forward ``T`` and its (exact or, for
    rimrings, pseudo) generalized inverse ``Tp``, both acting on a ``(1, C, H, W)``
    image. Families are stateless; family-specific parameters are passed per call
    as keyword arguments (conformal ``periods``; mobius/square ``k``/``l``; poles
    ``centers``; rimrings ``A``/``R0``)."""

    name: str

    @abstractmethod
    def T(self, src: torch.Tensor, *, inset_scale: float, out: int,
          supersample: int = 2, zoom: float = 0.5,
          focus: Tuple[float, float] = (0.5, 0.5), **params) -> torch.Tensor: ...

    @abstractmethod
    def Tp(self, src: torch.Tensor, *, inset_scale: float, out: int,
           supersample: int = 2, zoom: float = 0.5,
           focus: Tuple[float, float] = (0.5, 0.5), **params) -> torch.Tensor: ...
