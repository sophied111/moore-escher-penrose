"""escher.transforms.poles -- multi-center divisor family ("dipole", "tripole", ...).

Forward ``Q(z) = sum_j (l_j - i*k_j*ls/(2*pi)) * log(z - p_j)``, ``ls =
log(1/inset_scale)``. Ported verbatim from ``escher_zoo.wfield``'s ``'poles'``
branch (``escher_zoo.py`` @49-53).

Inverse: when exactly two centers share the same coefficient
``a = l_j - i*k_j*ls/(2*pi)`` the sum of two logs collapses to a quadratic in
``w`` (product form ``(w-p)(w-q) = exp(T/a)``), giving an exact closed-form root
pick (ported from ``escher_zoo_inverse.wfield_inv_point``'s ``'poles'`` branch,
@94-105). For any other configuration (non-uniform coefficients, or a center
count other than two) there is no elementary inverse; a complex-Newton
iteration (ported ``_cnewton``, @34-45) refines the leading-order seed
``exp(T / sum(coeffs))``.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch

from .base import L_of, render
from .base import TransformFamily

Center = Tuple[complex, float, float]  # (position p_j, k_j, l_j)

_DEFAULT_CENTERS: Tuple[Center, ...] = ((-0.5 + 0j, 1.0, 1.0), (0.5 + 0j, 1.0, 1.0))


def _coeffs(centers: Sequence[Center], ls: float):
    return [(complex(p), complex(l, -k * ls / (2 * math.pi))) for p, k, l in centers]


def _poles_coord_fn(centers: Sequence[Center], ls: float):
    cs = _coeffs(centers, ls)

    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        Wc = torch.zeros_like(z)
        for p, a in cs:
            zz = z - p
            zz = torch.where(zz == 0, torch.full_like(zz, 1e-9), zz)
            Wc = Wc + a * torch.log(zz)
        return Wc

    return coord_fn


def _cnewton(T, Wf, Wpf, seed, iters: int = 12):
    """Complex Newton for ``W(w) = T`` with analytic derivative ``Wpf``.

    Ported verbatim from ``escher_zoo_inverse._cnewton``.
    """
    w = seed.clone()
    for _ in range(iters):
        num = Wf(w) - T
        den = Wpf(w)
        den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        step = num / den
        step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
        w = w - step
        w = torch.where(torch.isfinite(w) & (w.abs() > 1e-9), w, seed)
    return w


def _poles_inv_point(T: torch.Tensor, centers: Sequence[Center], ls: float) -> torch.Tensor:
    cs = _coeffs(centers, ls)
    coeffs = [a for _, a in cs]
    uniform = all(abs(a - coeffs[0]) < 1e-12 for a in coeffs)
    if uniform and len(cs) == 2:
        a = coeffs[0]
        p, q = cs[0][0], cs[1][0]
        C = torch.exp(T / a)
        disc = torch.sqrt((p - q) ** 2 + 4.0 * C)
        r1 = ((p + q) + disc) / 2
        r2 = ((p + q) - disc) / 2
        seed = torch.exp(T / (2 * a))
        return torch.where((r1 - seed).abs() <= (r2 - seed).abs(), r1, r2)

    # fallback: complex Newton for non-uniform coefficients or center-count != 2
    a_tot = sum(a for _, a in cs)

    def _safe(zz):
        return torch.where(zz.abs() < 1e-9, torch.full_like(zz, 1e-9), zz)

    def Wf(w):
        out = torch.zeros_like(w)
        for p, a in cs:
            out = out + a * torch.log(_safe(w - p))
        return out

    def Wpf(w):
        out = torch.zeros_like(w)
        for p, a in cs:
            out = out + a / _safe(w - p)
        return out

    return _cnewton(T, Wf, Wpf, torch.exp(T / a_tot))


def _poles_inv_coord_fn(centers: Sequence[Center], ls: float, H: int, W: int, zoom: float):
    lam = zoom * min(H, W)
    logK = math.log(2.0 * lam / (W - 1))

    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        T = torch.log(z) - logK
        w = _poles_inv_point(T, centers, ls)
        w = torch.where(w == 0, torch.full_like(w, 1e-12), w)
        return torch.log(w) - logK

    return coord_fn


class PolesFamily(TransformFamily):
    name = "poles"

    def T(self, src, *, inset_scale, centers=None, out, supersample=2, zoom=0.5, focus=(0.5, 0.5)):
        """Forward poles map: multi-center divisor Q(z) = sum_j a_j*log(z-p_j)."""
        centers = tuple(centers) if centers is not None else _DEFAULT_CENTERS
        ls = L_of(inset_scale)
        coord_fn = _poles_coord_fn(centers, ls)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )

    def Tp(self, src, *, inset_scale, centers=None, out, supersample=2, zoom=0.5, focus=(0.5, 0.5)):
        """Inverse poles map (closed-form for a uniform 2-pole pair, else Newton)."""
        centers = tuple(centers) if centers is not None else _DEFAULT_CENTERS
        ls = L_of(inset_scale)
        _, _, H, W = src.shape
        coord_fn = _poles_inv_coord_fn(centers, ls, H, W, zoom)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )
