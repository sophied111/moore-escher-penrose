"""escher.transforms.rimrings -- twist family with a rim-condensing term:
periods "condense" (bunch up) against a circle of radius ``R0``.

Forward ``Wc = beta_p*log(z) + A*rc/(R0-rc)``, ``rc = r.clamp(max=R0-1e-3)``,
``r = |z|``, ``beta_p = beta_p(1, ls)`` (paper's twist exponent, ``k=1, l=1``
default), ``ls = log(1/inset_scale)``. Ported verbatim from
``escher_zoo.wfield``'s ``'rimrings'`` branch (``escher_zoo.py`` @54-56;
defaults ``A=1.2, R0=0.97``).

Pseudo-inverse (radial term frozen at the twist seed); approximate, and NOT
used by default in the paper's Rimrings recipe. Kept per project decision.
The radial correction term ``A*rc/(R0-rc)`` has no elementary inverse once
composed with the twist log-map, so ``RimringsFamily.Tp`` freezes it at the
leading-order (pure-twist) seed ``exp(T/beta_p)`` and inverts the remainder
exactly, one-shot -- no iteration. Ported from
``escher_zoo_inverse.wfield_inv_point``'s ``'rimrings'`` branch (@81-85).
"""

from __future__ import annotations

import math

import torch

from .base import beta_p, L_of, render
from .base import TransformFamily


def _rimrings_coord_fn(alpha: torch.Tensor, A: float, R0: float):
    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        r = z.abs()
        rc = r.clamp(max=R0 - 1e-3)
        return alpha * torch.log(z) + A * rc / (R0 - rc)

    return coord_fn


def _rimrings_inv_point(T: torch.Tensor, alpha: complex, A: float, R0: float) -> torch.Tensor:
    seed = torch.exp(T / alpha)  # PSEUDO: freeze the radial term at the twist seed
    rc = seed.abs().clamp(max=R0 - 1e-3)
    P = A * rc / (R0 - rc)  # real, added to Re(W)
    return torch.exp((T - P) / alpha)


def _rimrings_inv_coord_fn(alpha: complex, A: float, R0: float, H: int, W: int, zoom: float):
    lam = zoom * min(H, W)
    logK = math.log(2.0 * lam / (W - 1))

    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        T = torch.log(z) - logK
        w = _rimrings_inv_point(T, alpha, A, R0)
        w = torch.where(w == 0, torch.full_like(w, 1e-12), w)
        return torch.log(w) - logK

    return coord_fn


class RimringsFamily(TransformFamily):
    name = "rimrings"

    def T(self, src, *, inset_scale, out, supersample=2, zoom=0.5, focus=(0.5, 0.5), A=1.2, R0=0.97):
        """Forward rimrings map: twist plus a rim-condensing term."""
        L = L_of(inset_scale)
        alpha = torch.tensor(beta_p(1, L), dtype=torch.cfloat, device=src.device)
        coord_fn = _rimrings_coord_fn(alpha, A, R0)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )

    def Tp(self, src, *, inset_scale, out, supersample=2, zoom=0.5, focus=(0.5, 0.5), A=1.2, R0=0.97):
        """Pseudo-inverse rimrings map (radial term frozen at the twist seed)."""
        L = L_of(inset_scale)
        alpha = beta_p(1, L)  # python complex
        _, _, H, W = src.shape
        coord_fn = _rimrings_inv_coord_fn(alpha, A, R0, H, W, zoom)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )
