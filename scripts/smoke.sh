#!/usr/bin/env bash
# scripts/smoke.sh -- headline smoke test for the escher conformal
# T-cycle sampler: runs the paper's headline conformal Print-Gallery repro
# on the shipped backbones and writes the images to outputs/.
#
# This is the VISUAL acceptance gate (see scripts/README.md): only a human
# eyeballing the PNGs against the paper's general-results figure confirms the
# full recipe (backbone + braided sampler + schedule) reproduces the
# recursive Print-Gallery look end to end.
#
# Usage:
#   bash scripts/smoke.sh            # both sections (one env can hold both stacks -- see below)
#   bash scripts/smoke.sh flux       # FLUX section only
#   bash scripts/smoke.sh pixeldit   # PixelDiT section only
#
# A separate environment per backbone is recommended (isolation, smaller
# installs; the [flux] and [pixeldit] extras share one core stack, so a single
# env can also hold both). In practice you run each section from its own env:
#   * FLUX section:     `pip install -e .[flux]` (+ `.[sr]` for the Real-ESRGAN
#                        upres path) in the FLUX env.
#   * PixelDiT section: `pip install -e .[pixeldit]` in a SEPARATE env, with the
#                        submodule initialized
#                          git submodule update --init third_party/PixelDiT
#                        and the weights available (HF repo
#                        nvidia/PixelDiT-1300M-1024px, file pixeldit_t2i_v1.pth).
# Model weights must already be cached locally (see HF_HUB_OFFLINE below) or
# reachable over the network.
set -euo pipefail

TARGET="${1:-all}"

# Do not attempt to hit the Hugging Face Hub for a fresh download on a
# (typically offline) compute node -- fail fast if the weights are not
# already cached locally instead of hanging on a network call.
export HF_HUB_OFFLINE=1

PROMPT="a photorealistic gallery whose far wall is a photo of the same gallery, no text"

if [[ "${TARGET}" == "all" || "${TARGET}" == "flux" ]]; then
  # --- FLUX section (run in the [flux] environment) -----------------------
  # Headline FLUX.1-dev conformal Print-Gallery repro (paper Tables 1-4
  # defaults, via escher/configs/flux_conformal.yaml). ~45GB VRAM with the SR upres
  # path this preset uses by default -- see scripts/README.md.
  python -m cli --preset flux_conformal     --prompt "${PROMPT}"     --seed 100 --output-dir outputs --output smoke_flux.png
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "pixeldit" ]]; then
  # --- PixelDiT section (run in a SEPARATE [pixeldit] environment) --------
  # PixelDiT-1300M conformal repro (escher/configs/pixeldit_conformal.yaml), the
  # paper's re-matched schedule for the smaller backbone (warm-up 0.96,
  # sigma_hi 0.92, sigma_lo 0.69). Requires the [pixeldit] extra, the
  # initialized third_party/PixelDiT submodule and the HF weights
  # (nvidia/PixelDiT-1300M-1024px, pixeldit_t2i_v1.pth) -- see the header.
  python -m cli --preset pixeldit_conformal     --prompt "${PROMPT}"     --seed 100 --output-dir outputs --output smoke_pixeldit.png
fi

echo "Smoke run (${TARGET}) finished; images are in outputs/ (smoke_flux.png / smoke_pixeldit.png)."
echo "Now eyeball them against the paper's general-results figure -- see scripts/README.md."
