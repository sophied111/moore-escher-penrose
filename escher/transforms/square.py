"""escher.transforms.square -- square-spiral chart family: nested squares, one
period per lap.

Forward ``Q(z)``: ``rho = max(|x|, |y|)`` (Chebyshev radius), ``psi`` in
``[0, 8)`` an arclength parameter around the unit-square boundary through the
point's octant, ``tau = (psi mod 8)/8`` in ``[0, 1)`` the fractional lap.
``Wc = l*log(rho) + k*ls*tau + 2*pi*i*l*tau``, ``ls = log(1/inset_scale)``.
Ported verbatim from ``escher_zoo.wfield``'s ``'square'`` branch
(``escher_zoo.py`` @89-97).

Inverse: closed form. ``tau = Im(T)/(2*pi*l)`` recovers the lap fraction,
``rho = exp((Re(T) - k*ls*tau)/l)`` the Chebyshev radius, and ``_sq_psi_inv``
(ported from ``escher_zoo_inverse.py`` @21-31) maps ``psi = 8*tau`` back onto
the unit-square boundary direction ``(uf, vf)``; ``w = rho*(uf + i*vf)``
(ported from ``escher_zoo_inverse.wfield_inv_point``'s ``'square'`` branch,
@75-80).
"""

from __future__ import annotations

import math

import torch

from .base import L_of, render
from .base import TransformFamily


def _square_coord_fn(k: float, l: float, ls: float):
    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        xx, yy = zx, zy
        rho = torch.maximum(xx.abs(), yy.abs()).clamp_min(1e-9)
        uf, vf = xx / rho, yy / rho
        psi = torch.where(
            uf >= vf.abs(),
            torch.where(vf >= 0, vf, 8 + vf),
            torch.where(
                vf >= uf.abs(),
                2 - uf,
                torch.where(-uf >= vf.abs(), 4 - vf, 6 + uf),
            ),
        )
        tau = (psi % 8) / 8
        return torch.complex(l * torch.log(rho) + k * ls * tau, 2 * math.pi * l * tau)

    return coord_fn


def _sq_psi_inv(psi: torch.Tensor):
    """Inverse of the forward ``psi``: ``psi`` in ``[0, 8)`` -> ``(uf, vf)`` on
    the unit-square boundary. Ported verbatim from
    ``escher_zoo_inverse._sq_psi_inv``.
    """
    uf = torch.where(
        psi < 1,
        torch.ones_like(psi),
        torch.where(
            psi < 3,
            2 - psi,
            torch.where(psi < 5, -torch.ones_like(psi), torch.where(psi < 7, psi - 6, torch.ones_like(psi))),
        ),
    )
    vf = torch.where(
        psi < 1,
        psi,
        torch.where(
            psi < 3,
            torch.ones_like(psi),
            torch.where(psi < 5, 4 - psi, torch.where(psi < 7, -torch.ones_like(psi), psi - 8)),
        ),
    )
    return uf, vf


def _square_inv_point(T: torch.Tensor, k: float, l: float, ls: float) -> torch.Tensor:
    tau = T.imag / (2 * math.pi * l)
    rho = torch.exp((T.real - k * ls * tau) / l)
    psi = (tau * 8.0) % 8.0
    uf, vf = _sq_psi_inv(psi)
    return torch.complex(rho * uf, rho * vf)


def _square_inv_coord_fn(k: float, l: float, ls: float, H: int, W: int, zoom: float):
    lam = zoom * min(H, W)
    logK = math.log(2.0 * lam / (W - 1))

    def coord_fn(zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        z = torch.complex(zx, zy)
        T = torch.log(z) - logK
        w = _square_inv_point(T, k, l, ls)
        w = torch.where(w == 0, torch.full_like(w, 1e-12), w)
        return torch.log(w) - logK

    return coord_fn


class SquareFamily(TransformFamily):
    name = "square"

    def T(self, src, *, inset_scale, k=1.0, l=1.0, out, supersample=2, zoom=0.5, focus=(0.5, 0.5)):
        """Forward square-spiral map."""
        ls = L_of(inset_scale)
        coord_fn = _square_coord_fn(k, l, ls)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )

    def Tp(self, src, *, inset_scale, k=1.0, l=1.0, out, supersample=2, zoom=0.5, focus=(0.5, 0.5)):
        """Inverse square-spiral map (exact closed form)."""
        ls = L_of(inset_scale)
        _, _, H, W = src.shape
        coord_fn = _square_inv_coord_fn(k, l, ls, H, W, zoom)
        return render(
            src, coord_fn, out=out, supersample=supersample, zoom=zoom, focus=focus, inset_scale=inset_scale
        )
