# scripts/

## `smoke.sh` -- headline smoke test

Runs the paper's headline conformal Print-Gallery repro on both shipped
backbone presets (`flux_conformal`, `pixeldit_conformal`) and writes the
resulting PNGs to `outputs/`. It is a *visual* acceptance gate, not a unit
test -- see "Acceptance gate" below for what to look for.

A **separate environment per backbone is recommended** (isolation, smaller
installs; the `[flux]` and `[pixeldit]` extras share one core stack, so a
single env can also hold both), so run one section at a time:

```bash
bash scripts/smoke.sh flux       # in the [flux] env
bash scripts/smoke.sh pixeldit   # in the separate [pixeldit] env
bash scripts/smoke.sh            # both, from a single env that holds both stacks
```

Prerequisites:

* **FLUX section:** `pip install -e .[flux]` (add `.[sr]` for the FLUX
  preset's default Real-ESRGAN upres path -- see the main `README.md`).
* **PixelDiT section:** `pip install -e .[pixeldit]` in a *separate*
  environment, plus the vendored submodule and weights:
  ```bash
  git submodule update --init third_party/PixelDiT
  # weights: HF repo nvidia/PixelDiT-1300M-1024px, file pixeldit_t2i_v1.pth
  ```
  `accelerate` may turn out to be a needed transitive dependency at GPU run
  time. It is deliberately *not* pinned in `[pixeldit]` yet; if the smoke run
  surfaces an `ImportError` for it, add it to the `[pixeldit]` extra in
  `pyproject.toml`.
* Model weights cached locally (the script sets `HF_HUB_OFFLINE=1` so it
  fails fast instead of hanging on a network fetch; download the weights
  ahead of time with `huggingface-cli download` if they are not already
  cached on the host you run this on).
* A CUDA GPU with roughly **45GB of VRAM** for the FLUX.1-dev preset with
  its default `upres: sr` (Real-ESRGAN x4) path. Drop to `--upres bicubic`
  or `--upres off` if you are VRAM-constrained; note that changes what the
  acceptance gate is judging (no learned SR pass).

### Running on a GPU cluster

The script itself has no cluster-specific assumptions (no SLURM directives,
no hostnames) -- submit it the way you'd submit any other job on your own
scheduler. For example, with Slurm:

```bash
# interactive, for a quick check:
srun --partition=<partition> --gres=gpu:1 --mem=64G --time=00:30:00 \
  bash scripts/smoke.sh

# or as a batch job:
sbatch --partition=<partition> --gres=gpu:1 --mem=64G --time=00:30:00 \
  --job-name=escher-smoke --output=smoke.%j.log \
  --wrap="bash scripts/smoke.sh"

# to target a specific node:
srun --partition=<partition> --nodelist=<gpu-node> --gres=gpu:1 --mem=64G \
  bash scripts/smoke.sh
```

Replace `<partition>` and `<gpu-node>` with whatever your own cluster calls
them; nothing in this repo hardcodes an institution's queue names or node
lists.

### Acceptance gate (visual, Q6a)

This script's job is to catch a broken *recipe* (wrong backbone wiring,
broken schedule, mis-registered transform) that only surfaces in a real
generated image end to end, not in the transform math on its own.

Open `outputs/smoke_flux.png` and `outputs/smoke_pixeldit.png` (whichever
you ran) and check, by eye. FLUX is judged against the paper's
general-results figure; PixelDiT against the paper's PixelDiT gallery figure:

* The image reads as a single coherent scene matching the prompt (a
  photorealistic gallery), not a broken/degenerate collage.
* Somewhere inside that scene there is a visibly smaller, rotated copy of
  the same scene nested inward -- the Print-Gallery/Droste recursion the
  conformal twist is supposed to produce (the "spiralling into itself"
  look of the paper's own headline examples).
* The recursion should read as *smooth* (no hard seam ring, no obviously
  duplicated/pasted edge) -- a symptom of a correctly closed matched
  operator pair (`T`/`T†`), not a cosmetic nicety.

There is no automated pixel-diff for this: it is a human correctness check
that the faithfully-ported recipe still produces the paper's qualitative
result on a real backbone.

### PixelDiT-specific notes

**Visual gate.** `outputs/smoke_pixeldit.png` should show a coherent gallery
scene containing a smaller, rotated nested copy of itself (the Print-Gallery
recursion), judged against the paper's PixelDiT gallery figure. The same
smoothness criteria as above apply (no hard seam ring).

**Re-matched schedule.** The `pixeldit_conformal` preset uses the paper's
re-matched sigmas for this smaller backbone, not FLUX's: warm-up sigma
**0.96**, `sigma_hi` **0.92**, `sigma_lo` **0.69** (128 steps, `cfg_scale`
3.5, post-loop time travel `[1, 0.82]`). If the result looks wrong, first
confirm these values were not overridden on the command line.

**Solver.** `denoise_block` runs a `FlowMatchEulerDiscreteScheduler`
(`shift=4`), stepped between operator switches. If the PixelDiT outputs
diverge from the paper (e.g. a weaker or less consistent recursion than the
paper's figure), first confirm the preset's schedule and `cfg_scale` were not
overridden on the command line.
