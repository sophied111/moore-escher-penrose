"""escher.transforms.conformal -- the conformal twist family and its matched
generalized inverse: the carrier of the paper's Penrose identity ``T T† T = T``.

This module supplies the *matched pair* ``(T, T†)`` (here ``ConformalFamily.T`` and
``ConformalFamily.Tp``). They are **not** free coordinate maps that happen to look like
inverses: ``T = E₂B₁`` and ``T† = E₁B₂`` share deck-transformation closure
constants so that ``T†`` is a genuine generalized inverse of ``T`` and the
composite projector ``P = T∘T†`` is idempotent (``P² = P`` to the bilinear
resample floor). Those closure constants -- the complex deck ``gamma``, the
integer-``s``-lattice scale closure ``(2·zoom_T)(2·zoom_T†) = sʲ``, and the
phase-closure read-rotation -- are ported faithfully from the reference
implementation and must not be simplified away (dropping them, e.g. folding
``T†`` radially instead of along the complex deck, drives the idempotence
residual from ~0.03 up past 0.09 -- broken closure).

Convention (shared with ``escher.transforms.base.render``): the family bakes its
periodicity into ``coord_fn``; ``render`` owns the supersampled mesh, focus
offset, radial zoom, floor-band fold and bilinear resample. ``ConformalFamily.T`` runs
entirely through ``render`` (radial deck, band ``L = log(1/inset_scale)`` passed
via ``inset_scale``). ``ConformalFamily.Tp`` needs a *complex* deck ``gamma`` -- a
spiral fold that rotates as well as scales -- which ``render``'s radial-only fold
cannot represent; it therefore performs its own fold+resample tail, but folds at
the same true scale-period (via ``s = 1/inset_scale`` in ``gamma``/``z2``) rather
than any resolution-tied default. See the task report for the measured residuals.

Only the Cartesian ``z**alpha`` conformal path is implemented here. The
log-strip backend, band backend, super-resolution and the other transform
families are later tasks and are deliberately absent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import beta_p, geom_dtype, L_of, render
from .base import TransformFamily


class ConformalFamily(TransformFamily):
    """Conformal twist family: matched ``T = E2 B1`` / ``Tp = E1 B2`` and the
    idempotent projector ``P = T o Tp`` (Penrose identity ``T Tp T = T``)."""

    name = "conformal"

    def T(self, src, *, inset_scale, periods=1, out, supersample=2,
          zoom=0.5, focus=(0.5, 0.5)):
        """Matched conformal forward T."""
        L = L_of(inset_scale)
        alpha = torch.tensor(beta_p(periods, L), dtype=torch.cfloat, device=src.device)

        def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
            # Parity-critical: keep ``log(z)`` on the left of ``* alpha`` (1-ulp kernel difference).
            return torch.log(torch.complex(zx, zy)) * alpha

        return render(
            src,
            coord_fn,
            out=out,
            supersample=supersample,
            zoom=zoom,
            focus=focus,
            inset_scale=inset_scale,
        )

    def Tp(self, src, *, inset_scale, periods=1, out, supersample=2,
           zoom=0.5, focus=(0.5, 0.5)):
        """Matched conformal generalized inverse T-dagger."""
        _, _, H, W = src.shape
        device, in_dtype = src.device, src.dtype
        dtype = geom_dtype(in_dtype)  # geometry in float32; input dtype restored below

        s = 1.0 / inset_scale
        alpha = beta_p(periods, L_of(inset_scale))          # python complex
        t_zoom = zoom                                       # = zoom_T

        # complex deck and its log-modulus (real fold band along the spiral)
        gamma = torch.tensor(s, dtype=torch.cfloat) ** (1 / alpha)   # 1/alpha: python complex, matches reference
        log_abs_gamma = math.log(abs(gamma))

        # scale closure on the integer-s lattice: 2*zoom_Tp such that
        # (2*zoom_T)*(2*zoom_Tp) = s**j  (exact power of s)
        j = round(math.log(2.0 * t_zoom * 0.8) / math.log(s))
        z2 = (s ** j) / (2.0 * t_zoom)                      # = 2*zoom_Tp

        # supersampled output mesh, focus-panned like render
        n = out * supersample
        lin = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        z = torch.complex(xx, yy)
        oz = complex(2.0 * focus[0] - 1.0, 2.0 * focus[1] - 1.0)
        z = z - oz
        z = torch.where(z == 0, torch.full_like(z, 1e-12), z)

        # reciprocal-exponent map + phase-closure read-rotation
        inv_alpha = torch.tensor(1.0 / alpha, dtype=torch.cfloat, device=device)
        w = torch.exp(torch.log(z) * inv_alpha)
        # Parity-critical: this constant is built on the CPU (device-less tensor) on purpose.
        p0 = w * z2 * torch.exp(torch.tensor(-1j * alpha.imag * math.log(z2)))

        fx, fy = (W - 1) / 2.0, (H - 1) / 2.0
        half = min(W, H) / 2.0

        # spiral deck fold: largest in-canvas copy along gamma**k (per-pixel Rmax; parity-critical).
        _ap = p0 / p0.abs().clamp_min(1e-12)
        _cx, _sx = _ap.real, _ap.imag
        _rx = torch.where(_cx >= 0, (W - 1 - fx) / _cx.clamp_min(1e-9), -fx / _cx.clamp_max(-1e-9))
        _ry = torch.where(_sx >= 0, (H - 1 - fy) / _sx.clamp_min(1e-9), -fy / _sx.clamp_max(-1e-9))
        Rmax = torch.minimum(_rx, _ry) * 0.995 / half
        k = torch.floor(torch.log(Rmax / p0.abs().clamp_min(1e-12)) / log_abs_gamma)
        p = p0 * gamma ** k

        grid = torch.stack(
            [
                (fx + p.real * half) / (W - 1) * 2 - 1,
                (fy + p.imag * half) / (H - 1) * 2 - 1,
            ],
            dim=-1,
        )[None]

        o = F.grid_sample(
            src.to(dtype), grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return F.avg_pool2d(o, supersample).to(in_dtype)

    def P(self, src, **kw):
        """Idempotent projector P = T o Tp."""
        return self.T(self.Tp(src, **kw), **kw)
