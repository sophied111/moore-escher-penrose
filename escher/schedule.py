"""escher.schedule -- pure schedule builders for the T-cycle sampling loop.

Decides WHICH denoising step indices get warm-up (forward-T zoom, p=0) or an
operator application (T / T-dagger, the paper's forward/inverse conformal
warp) -- no model, no torch autograd, CPU-only. Ported from the legacy
``step_modifiers._sigma_to_step`` / ``_nops_op_steps_gap`` and the warm-up
range construction inside ``run_sampling``, stripped of the CLI ``args``
object and turned into pure functions of ``sigmas``.

Conventions carried over from the legacy script:
  * ``sigmas`` descends from ~1.0 (pure noise) to 0.0 (clean), length
    steps + 1 (the scheduler's own convention -- diffusers, k-diffusion).
  * a "step index" i is resolved by :func:`sigma_to_step` as the FIRST
    index where ``sigmas[i] <= target + eps`` -- i.e. mapping a sigma-unit
    window bound to the step index where the schedule first reaches or
    passes it. This is stable across step counts, unlike a step fraction.
  * op labels alternate ``"T"`` (forward warp) / ``"Tp"`` (T-dagger, the
    flagged pseudo-inverse) starting at ``"T"``; the LAST op in any window
    is always forced back to ``"T"`` (the paper's warped end).
"""

from __future__ import annotations

from typing import Sequence


def sigma_to_step(sigmas: Sequence[float], sigma: float) -> int:
    """First step index i where ``sigmas[i] <= sigma`` (+ eps).

    ``sigmas`` descends from ~1.0 to 0.0 with length steps + 1. Maps a
    sigma-unit window bound (schedule time) to a step index -- reproducible
    across step counts, unlike a step fraction. Falls back to the last step
    index if ``sigma`` is below every entry.
    """
    n = len(sigmas)
    for i in range(n - 1):
        if float(sigmas[i]) <= sigma + 1e-9:
            return i
    return n - 1


def build_op_steps(
    *,
    sigmas: Sequence[float],
    sigma_hi: float,
    sigma_lo: float,
    n_ops: int = 3,
    op_gap: int = 9,
    mode: str = "alt",
) -> dict[int, str]:
    """Build the operator-step schedule across ``[sigma_hi, sigma_lo]``.

    Resolves the sigma window to step indices ``i_hi = sigma_to_step(sigma_hi)``
    and ``i_lo = sigma_to_step(sigma_lo)`` (``i_hi`` must be strictly less
    than ``i_lo`` -- an empty or inverted window is a caller error), then
    places ``n_ops`` operator applications: the first op ("T") lands at
    ``i_hi``, the second op ("Tp") lands ``op_gap`` steps later (clamped to
    ``i_lo``, a tight T->Tp pair), and any remaining ops (``n_ops > 2``) are
    evenly spaced from there to ``i_lo``. Labels alternate T/Tp starting at
    T; the LAST op is always forced back to "T" (the paper's warped end).
    Positions are de-duplicated (rounding collisions), so the returned dict
    can have fewer than ``n_ops`` entries when the window is narrower than
    the requested spacing.

    ``n_ops=1`` is a special case: a single forward ``T`` at ``i_hi`` with no
    ``Tp`` (the rimrings single-T recipe).

    Only ``mode="alt"`` (T/Tp alternation) is implemented -- the only mode
    the paper's conformal schedule uses.
    """
    if mode != "alt":
        raise NotImplementedError(f"build_op_steps: mode={mode!r} not implemented (only 'alt')")
    if n_ops < 1:
        raise ValueError(f"n_ops must be >= 1 (got {n_ops})")
    if op_gap < 1:
        raise ValueError(f"op_gap must be >= 1 (got {op_gap})")

    i_hi = sigma_to_step(sigmas, sigma_hi)
    i_lo = sigma_to_step(sigmas, sigma_lo)
    if not i_hi < i_lo:
        raise ValueError(f"op window empty: hi_step={i_hi} >= lo_step={i_lo}")

    if n_ops == 1:
        return {i_hi: "T"}          # single forward T at the window's high edge

    second = min(i_hi + op_gap, i_lo)
    pts = {i_hi, second}
    if n_ops > 2:
        for k in range(1, n_ops - 1):
            pts.add(round(second + k * (i_lo - second) / (n_ops - 2)))
    positions = sorted(pts)

    ops = {st: ("T" if j % 2 == 0 else "Tp") for j, st in enumerate(positions)}
    ops[positions[-1]] = "T"
    return ops


def build_warmup_steps(
    *,
    sigmas: Sequence[float],
    warmup_sigma: float,
    first_op_step: int,
) -> list[int]:
    """Every step index from ``warmup_sigma`` up to (not including) ``first_op_step``.

    Each of these steps gets a forward-T application at periods=0 (a pure
    zoom-Droste, no twist) before the T-cycle window opens -- ported from
    the ``run_sampling`` warm-up range construction. Raises if the resolved
    warm-up start does not land strictly before ``first_op_step``.
    """
    i_w = sigma_to_step(sigmas, warmup_sigma)
    if not i_w < first_op_step:
        raise ValueError(
            f"warmup_sigma {warmup_sigma} (step {i_w}) must land before "
            f"the first op step ({first_op_step})"
        )
    return list(range(i_w, first_op_step))
