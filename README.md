# moore-escher-penrose

Code for the paper **"Moore, Escher, Penrose: A Conformal Golden Braid"**.
<img src="assets/assorted-results.png" alt="assorted results" width="800">

## Method

This repository turns a frozen, off-the-shelf diffusion backbone into a generator of
recursive Print-Gallery/Droste images, with no fine-tuning. Between denoising
steps it warps the model's clean-image prediction (`x0`) in pixel space: a short
untwisted-Droste warm-up (pure zoom), then a window alternating a conformal warp
`T` and its matched inverse `T†` (`T→T†→T`), optionally followed by a
"time-travel" refinement pass. `T` and `T†` share deck-transformation closure
constants, so the Penrose identity `T T† T = T` holds and `P = T∘T†` is an
idempotent projector onto self-consistent recursive images. Full schedule,
operator construction, and the five families: [`docs/method.md`](docs/method.md).

## Install

Clone, then install one backbone from its pinned lock file. FLUX and PixelDiT
share the same core `torch` / `diffusers` / `transformers` stack, so one venv can
serve both, but a separate venv each is recommended (isolation, smaller
installs). Needs a CUDA GPU and Python >= 3.11.

```bash
git clone https://github.com/sophied111/moore-escher-penrose.git moore-escher-penrose
cd moore-escher-penrose
```

**uv**:

```bash
uv venv .venv-flux
source .venv-flux/bin/activate
# PixelDiT only: git submodule update --init third_party/PixelDiT
uv pip install -r requirements-flux.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
# PixelDiT only: use requirements-pixeldit.lock.txt above
uv pip install -e . --no-deps
```


Notes:
- **Weights are not shipped.** FLUX: `black-forest-labs/FLUX.1-dev`. PixelDiT:
  `pixeldit_t2i_v1.pth` from `nvidia/PixelDiT-1300M-1024px`.
- **Super-resolution** (`--upres sr`, the FLUX default) needs no extra install:
  the RRDBNet is vendored, and `RealESRGAN_x4plus.pth` auto-downloads to
  `~/.cache/escher/` (offline: set `ESCHER_SR_WEIGHTS`).
- **VRAM:** ~45 GB for the default FLUX preset; `--upres bicubic`/`off` for less.
- PixelDiT is a pinned git submodule (`third_party/PixelDiT`, upstream
  NVlabs/PixelDiT); no upstream code is copied in.

## Run

```bash
python -m cli --preset flux_conformal \
  --prompt "a photorealistic gallery whose far wall is a photo of the same gallery, no text" \
  --seed 100 --output-dir outputs --output smoke_flux.png
```

PixelDiT is the same command with `--preset pixeldit_conformal`. The PNG lands at
`<output-dir>/<output>`; `python -m cli` and the installed `escher` command are
equivalent. [`scripts/smoke.sh`](scripts/smoke.sh) runs both backbones;
[`scripts/README.md`](scripts/README.md) covers cluster runs and the visual
acceptance gate.

### Example: the antique-map figure

The `flux_conformal` preset carries the full recipe (clamped zoom, `periods=2`,
`upres=sr`, time travel), so the paper's antique-map figure is just the prompt
and the seed:

```bash
python -m cli --preset flux_conformal \
  --prompt "A photorealistic old map spread on a wooden table; an explorer drawn in ink walks off the edge of the map onto the real table, and the map depicts this same table, map, and walking explorer; candle light, parchment texture, ink turning into cloth." \
  --seed 7 --output-dir outputs --output antique_map.png
```

Installed from the lock, this reproduces the paper's reference for this seed on
compatible hardware; a different library stack gives a different but equivalent
recursion.

## Interactive REPL

Iterate on prompts and knobs without reloading the backbone between runs.

```bash
python -i interactive.py
```

`go()` takes any `SampleConfig` field as a keyword override, runs one sample
against the already-loaded backbone, and saves to
`outputs/interactive/<tag>/output.png`:

```python
>>> go(
...     "A photorealistic artist's desk with an open comic page; the inked "
...     "character steps out of its panel into full photographic reality on the "
...     "desk, the page still showing this same desk and the same stepping "
...     "character; drafting lamp, halftone dots resolving into real fabric.",
...     seed=1234,
... )
```

Example output:

<img src="output.png" alt="comic_stepout" width="300">

## Knobs

`--preset` names a YAML under [`escher/configs/`](escher/configs/) (default `flux_conformal`);
any flag you pass overrides it. Recursion knobs:

| Flag | Meaning |
| --- | --- |
| `--family {conformal,poles,mobius,square,rimrings}` | transform family for the window `T`/`T†` (the warm-up is always an untwisted conformal zoom) |
| `--inset-scale` | inset ratio `1/λ` (e.g. `0.25` -> `λ=4`); sets the twist exponent and fold scale period |
| `--periods` | twist periods `p` per turn, operator window |
| `--focus FX FY` | normalized `(x, y)` fixed point the recursion pans around |
| `--op-gap` | step spacing between operator applications |
| `--sigma-hi` / `--sigma-lo` | sigma bounds of the operator window |
| `--warmup-sigma` | sigma where the untwisted warm-up begins |
| `--upres {off,bicubic,sr}` | super-resolution before warping (`sr` = Real-ESRGAN x4) |
| `--time-travel-n` / `--time-travel-sigma` | post-loop refinement: re-noise to the sigma, re-denoise the tail N times |

Also `--model`, `--backbone`, `--steps`, `--cfg-scale`, `--seed`, `--size`,
`--n-ops`, `--zoom`, `--supersample`, `--output`, `--output-dir`, plus
`--monitor-every N` / `--monitor-path PATH` (dump an `x̂0` montage for debugging).
`python -m cli --help` is authoritative.

Two presets ship, both the paper's own defaults:
`escher/configs/flux_conformal.yaml` (FLUX.1-dev, Tables 1-4) and
`escher/configs/pixeldit_conformal.yaml` (PixelDiT-1300M, re-matched schedule). Copy one
and point `--preset` at it for your own.

## Extending: transform families

Families live under `escher/transforms/`: a `TransformFamily` ABC (`base.py`)
with a matched `T` / `Tp` pair and one class each (`ConformalFamily`,
`PolesFamily`, `MobiusFamily`, `SquareFamily`, `RimringsFamily`;
`ConformalFamily` adds the `P = T∘T†` projector). To add one: subclass
`TransformFamily`, implement `T`/`Tp` (build a `coord_fn`, call `base.render`),
set `name`, and register the instance in `transforms/__init__.py`'s `FAMILIES`.
The sampler touches families only via `get_family` + `T`/`Tp`, so no sampler
change is needed. The warm-up is always `ConformalFamily.T` with `periods=0`;
`WARMUP_CONF_ZOOM` is a global in `base.py`.

## Further reading

[`docs/method.md`](docs/method.md) — braided schedule, matched-operator
construction, the Penrose identity, and the five families' coordinate maps.
[`scripts/README.md`](scripts/README.md) — the visual acceptance gate.
