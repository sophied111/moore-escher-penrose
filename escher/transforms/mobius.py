"""escher.transforms.mobius -- Mobius/cross-ratio family with two fixed points.

Forward ``Q(z) = alpha * (log(z - a) - log(z - b))``, ``alpha = l -
i*k*ls/(2*pi)``, ``ls = log(1/inset_scale)``. Ported verbatim from
``escher_zoo.wfield``'s ``'mobius'`` branch (``escher_zoo.py`` @77-82; there the
fixed points are written ``-c``/``+c`` with default ``c = 0.5``, i.e. ``a = -c``,
``b = c`` here).

Inverse: closed form via the cross-ratio ``(w-a)/(w-b) = exp(T/alpha) = E``,
solved linearly for ``w = (a - b*E)/(1 - E)`` (ported from
``escher_zoo_inverse.wfield_inv_point``'s ``'mobius'`` branch, @54-60).
"""

from __future__ import annotations

import math

import torch

from .base import L_of, render
from .base import TransformFamily


def _alpha(k: float, l: float, ls: float) -> complex:
    return complex(l, -k * ls / (2 * math.pi))


def _mobius_coord_fn(a: complex, b: complex, alpha: complex):
    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        z1 = z - a
        z1 = torch.where(z1 == 0, torch.full_like(z1, 1e-9), z1)
        z2 = z - b
        z2 = torch.where(z2 == 0, torch.full_like(z2, 1e-9), z2)
        return alpha * (torch.log(z1) - torch.log(z2))

    return coord_fn


def _mobius_inv_point(T: torch.Tensor, a: complex, b: complex, alpha: complex) -> torch.Tensor:
    E = torch.exp(T / alpha)
    den = 1.0 - E
    den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
    return (a - b * E) / den


def _mobius_inv_coord_fn(a: complex, b: complex, alpha: complex, H: int, W: int, zoom: float):
    lam = zoom * min(H, W)
    logK = math.log(2.0 * lam / (W - 1))

    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        T = torch.log(z) - logK
        w = _mobius_inv_point(T, a, b, alpha)
        w = torch.where(w == 0, torch.full_like(w, 1e-12), w)
        return torch.log(w) - logK

    return coord_fn


class MobiusFamily(TransformFamily):
    name = "mobius"

    def T(self, src, *, inset_scale, a=-0.5, b=0.5, k=1.0, l=1.0, out, supersample=2, zoom=0.5, focus=(0.5, 0.5)):
        """Forward Mobius map Q(z) = alpha*(log(z-a) - log(z-b))."""
        ls = L_of(inset_scale)
        alpha = _alpha(k, l, ls)
        coord_fn = _mobius_coord_fn(complex(a), complex(b), alpha)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )

    def Tp(self, src, *, inset_scale, a=-0.5, b=0.5, k=1.0, l=1.0, out, supersample=2, zoom=0.5, focus=(0.5, 0.5)):
        """Inverse Mobius map (exact cross-ratio closed form)."""
        ls = L_of(inset_scale)
        alpha = _alpha(k, l, ls)
        _, _, H, W = src.shape
        coord_fn = _mobius_inv_coord_fn(complex(a), complex(b), alpha, H, W, zoom)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )
