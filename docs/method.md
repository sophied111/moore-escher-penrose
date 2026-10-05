# Method: conformal T-cycle braided sampling

This document is the technical companion to the top-level `README.md`. It
describes the braided sampling schedule, the matched forward/inverse
operator pair and the Penrose identity that makes it work, and the five
transform families' coordinate maps. It accompanies the paper "Moore,
Escher, Penrose: A Conformal Golden Braid" and its appendix; code pointers
below are into this repository's `escher/` package.

## 1. The braided schedule

`escher.sampler.sample` (`escher/sampler.py`) walks the backbone's ordinary
denoising trajectory step by step and "braids" pixel-space conformal warps
into a handful of those steps, chosen by `escher.schedule`
(`escher/schedule.py`) from sigma-unit bounds rather than step fractions (so
the same schedule reproduces across step counts). One run looks like:

```
free-denoise ... warm-up (p=0 Droste) ... T -> T† -> T window ... free-denoise ... [time travel]
```

1. **Free denoising.** Every step outside the two windows below is an
   ordinary `backbone.denoise_block` call -- no warp, no re-noising outside
   the model's own trajectory.

2. **Warm-up (`warmup_sigma` down to the first operator step).** At each of
   these steps the sampler predicts the clean image (`predict_x0`), applies
   an *untwisted* conformal Droste zoom (`ConformalFamily.T` in `escher/transforms/conformal.py`
   with `periods=0`, i.e. no twist -- a pure recursive zoom) at a fixed
   `WARMUP_CONF_ZOOM` (defined in `escher/transforms/base.py`), and re-noises. This is always the conformal family
   regardless of `--family`, planting a Droste copy before the chosen
   family's twist ever appears.

3. **Operator window (`build_op_steps`, bounded by `sigma_hi`/`sigma_lo`).**
   `n_ops` operator applications are placed across the window: the first
   lands at the `sigma_hi` step, the second `op_gap` steps later, and any
   remaining ones are evenly spaced out to the `sigma_lo` step. Labels
   alternate `T`, `T†`, `T`, `T†`, ... starting from `T`, but **the last
   operator in the window is always forced back to `T`** -- the paper's
   "warped end" convention, so the window always closes on the forward
   twist rather than its inverse. The paper's own conformal schedule uses
   `sigma_hi=0.87`, `op_gap=9`, `sigma_lo=0.5` (see
   `escher/configs/flux_conformal.yaml`). At each of these steps the sampler
   predicts `x0`, optionally super-resolves it (`--upres`), applies the
   scheduled `T` or `T†` from the selected family, and re-noises.

4. **Optional time travel (`time_travel = (tt_n, tt_sigma)`).** After the
   main loop, the finished estimate is re-noised back to `tt_sigma` and the
   schedule's tail is re-denoised `tt_n` times with *no* operators applied
   -- a pure refinement pass that lets the backbone smooth over the pixel
   warps' seams without disturbing the global recursive structure it just
   built. `escher/configs/pixeldit_conformal.yaml` uses one such pass
   (`time_travel: [1, 0.82]`); the FLUX preset does not use time travel.

Fresh noise is only ever injected inside `backbone.renoise`; ordinary steps
never re-noise.

## 2. The matched operator pair and the Penrose identity

The forward warp `T` and the "inverse" warp `T†` are **not** two
independently chosen coordinate maps that happen to approximately undo each
other -- they are a *matched pair*, in the paper's notation `T = E₂B₁` and
`T† = E₁B₂` (an "encode"/"blend" factorization shared across families:
`B` folds a source position into the visible fundamental-domain band, `E`
expands it back out to the output frame). Sharing the deck-transformation
closure constants between the two directions is what makes `T†` a genuine
generalized inverse of `T`, in the precise sense of the **Penrose identity**

```
T T† T = T
```

which in turn makes the composite `P = T ∘ T†` an **idempotent projector**
(`P² = P`, up to the bilinear-resample floor) onto the manifold of
self-consistent, admissible recursive images: applying `P` twice does no
more than applying it once. `ConformalFamily.P` (`escher/transforms/conformal.py`) is
exactly this composite for the conformal family.

Concretely, for the conformal family (`escher/transforms/conformal.py`):

* `ConformalFamily.T` maps the output mesh through `w = z**β_p` (i.e.
  `coord_fn(z) = β_p · log z`) and folds radially at the true scale period
  `L = log(1/inset_scale)`.
* `ConformalFamily.Tp` (`T†`) uses the reciprocal exponent `1/β_p` plus a matched
  **complex** deck `γ = s**(1/β_p)` (`s = 1/inset_scale`): a spiral fold
  that scales *and* rotates at once, which a radial-only fold cannot
  represent. It also carries an integer-lattice **scale closure**
  (`(2·zoom_T)(2·zoom_T†) = s^j` for an integer `j`) and a **phase-closure**
  read-rotation that pre-aligns the phase the forward twist will impose.
  Dropping any of these three closure constants breaks the identity above
  and drives the idempotence residual from ~0.01-0.03 up past 0.09.

Every family below defines its own twist exponent `β_p` from the same
convention:

```
β_p = 1 - i·p·L/(2π),   L = log(1/inset_scale)
```

(`escher.transforms.base.beta_p` / `L_of`), where `p` is `--periods` and
`inset_scale` is `--inset-scale` (the paper's inset ratio `1/λ`).

## 3. The five transform families

Each family is a `TransformFamily` subclass (`escher.transforms.base`) exposing
a matched `T` / `Tp` method pair (`ConformalFamily` also a `P = T∘T†`
projector); `get_family(name)` returns the family instance. A family's `T`/`Tp`
builds a pure coordinate map `coord_fn` — an output-mesh point `z` (in the
family's own working frame) to a complex source log-position `Q(z) = u + iθ` —
and hands it to the shared `escher.transforms.base.render` core, which owns the
supersampled mesh, focus offset, radial zoom and the floor-band fold + bilinear
resample. `render` is family-agnostic; only `Q(z)` (and, for the spiral-deck
families, the inverse's own fold) differs. The warm-up zoom constant
`WARMUP_CONF_ZOOM` is a global in `base.py`.

| Family | Forward `Q(z)` | Inverse |
| --- | --- | --- |
| **conformal** (`conformal.py`) | `Q(z) = β_p · log z` | exact: reciprocal exponent `1/β_p` + matched complex-deck fold (`T†`, see §2) |
| **poles** (`poles.py`) | `Q(z) = Σⱼ aⱼ · log(z − pⱼ)`, `aⱼ = lⱼ − i·kⱼ·L/(2π)` (sum over `j` pole centres) | exact closed form for a uniform two-pole pair (quadratic in `w`, `(w−p)(w−q) = exp(T/a)`); complex-Newton refinement of the leading-order seed `exp(T/Σaⱼ)` otherwise |
| **mobius** (`mobius.py`) | `Q(z) = α·(log(z − a) − log(z − b))`, `α = l − i·k·L/(2π)` | exact: cross-ratio closed form, `w = (a − b·E)/(1 − E)`, `E = exp(T/α)` |
| **square** (`square.py`) | `Q(z) = l·log ρ + k·L·τ + 2πi·l·τ`, Chebyshev radius `ρ = max(|x|,|y|)` and unit-square arclength fraction `τ ∈ [0,1)` | exact closed form: `τ` from `Im(T)`, `ρ` from `Re(T)`, then the square-boundary direction inverted lap-wise |
| **rimrings** (`rimrings.py`) | `Q(z) = β_p·log z + A·r_c/(R₀ − r_c)`, twist plus a term that condenses periods against a rim of radius `R₀` (`r_c = min(\|z\|, R₀−ε)`) | pseudo-inverse only: the radial rim term is frozen at the leading-order twist seed `exp(T/β_p)` and the remainder inverted exactly, one-shot; **not used by default** in the paper's Rimrings recipe |

`conformal`, `poles`, `mobius` and `square` all have exact (or Newton-refined
exact) inverses and are used with the full `T→T†→T` operator window.
`rimrings` ships an inverse for completeness but the paper's own Rimrings
schedule does not exercise it by default (see the module docstring in
`escher/transforms/rimrings.py`).
