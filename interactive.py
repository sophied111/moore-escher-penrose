"""interactive -- load-once Escher REPL: build the backbone ONCE, render on demand.

Usage:
    export ESCHER_PRESET=flux_conformal    # optional; default is flux_conformal
    python -i interactive.py               # loads the backbone once, drops to >>>

Then call go() as many times as you like -- the backbone stays loaded (no
reload between calls):

    go("a photorealistic snake coming out of a framed picture, no text")
    go("...", seed=7, periods=2)                 # any SampleConfig field
    go("...", upres="sr", zoom=0.4, focus=(0.5, 0.5))

Each call deep-copies the preset's base ``SampleConfig``, applies the given
overrides (validated against the dataclass's own fields -- an unknown name
raises), runs ``escher.sampler.sample`` against the already-loaded backbone,
saves to ``outputs/interactive/<tag>/output.png``, prints the path, and
returns the rendered ``PIL.Image.Image``.

Only ``escher.config`` (pure dataclass + YAML) is touched at import time
before the backbone build; the heavy backbone construction below is the one
GPU/model-loading step this module performs, and it happens exactly once, at
import. (Do not import this module from a test -- it always loads a model.
The parsing-only pieces this REPL shares in spirit with the batch runner live
in ``batch_run.py``, which stays importable without a model.)
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import fields
from typing import Any, Optional

from PIL import Image

from escher.config import SampleConfig, load_preset

_PRESET = os.environ.get("ESCHER_PRESET", "flux_conformal")
_FIELD_NAMES = {f.name for f in fields(SampleConfig)}


def _build_backbone(cfg: SampleConfig) -> Any:
    if cfg.backbone == "flux":
        from escher.backbones.flux import FluxBackbone

        return FluxBackbone(model_id=cfg.model_id, num_inference_steps=cfg.num_steps)
    elif cfg.backbone == "pixeldit":
        from escher.backbones.pixeldit import PixelDiTBackbone

        return PixelDiTBackbone(model_id=cfg.model_id, num_inference_steps=cfg.num_steps)
    raise ValueError(f"unknown backbone: {cfg.backbone!r}")


print(f"[interactive] loading preset {_PRESET!r} backbone (once)...", flush=True)
BASE: SampleConfig = load_preset(_PRESET)
BACKBONE: Any = _build_backbone(BASE)
print("[interactive] backbone loaded. call go('your prompt', **overrides)", flush=True)


def go(prompt: Optional[str] = None, out: Optional[str] = None, **overrides: Any) -> Image.Image:
    """Render one config on the already-loaded backbone. Returns the PIL image.

    ``prompt`` is a shortcut for the ``prompt`` override; any other
    ``SampleConfig`` field name works as a kwarg (``seed=``, ``periods=``,
    ``upres=``, ``family_params=``, ...). Unknown names raise ``AttributeError``
    rather than being silently ignored.
    """
    from escher.sampler import sample

    if prompt is not None:
        overrides = {**overrides, "prompt": prompt}
    for name in overrides:
        if name not in _FIELD_NAMES:
            raise AttributeError(f"unknown SampleConfig field: {name!r}")

    cfg = copy.deepcopy(BASE)
    if overrides:
        cfg = cfg.overrides(**overrides)

    image = sample(
        BACKBONE,
        cfg.prompt,
        family=cfg.family,
        inset_scale=cfg.inset_scale,
        periods=cfg.periods,
        sigma_hi=cfg.sigma_hi,
        sigma_lo=cfg.sigma_lo,
        n_ops=cfg.n_ops,
        op_gap=cfg.op_gap,
        warmup_sigma=cfg.warmup_sigma,
        focus=cfg.focus,
        upres=cfg.upres,
        zoom=cfg.zoom,
        supersample=cfg.supersample,
        time_travel=cfg.time_travel,
        seed=cfg.seed,
        size=cfg.size,
        cfg_scale=cfg.cfg_scale,
        family_params=cfg.family_params,
    )

    tag = out or f"i_{time.strftime('%H%M%S')}"
    out_dir = os.path.join("outputs", "interactive", tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.png")
    image.save(out_path)
    print(
        f"[interactive] saved {out_path} "
        f"(seed={cfg.seed} family={cfg.family} periods={cfg.periods} upres={cfg.upres})",
        flush=True,
    )
    return image


r: Optional[Image.Image] = None  # last-result convenience holder for the REPL
