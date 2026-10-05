"""escher.config -- SampleConfig dataclass + YAML preset loader.

A ``SampleConfig`` is the serializable, overridable bag of knobs for one
``escher.sampler.sample()`` run: every keyword argument ``sample()`` accepts,
plus the three fields needed to pick and drive a backbone (``model_id``,
``backbone``, ``num_steps``) that live outside ``sample()``'s own signature
(``sample()`` derives its step count from the backbone's own noise schedule;
``num_steps`` here is the config-level knob a CLI/backbone constructor reads
to build that schedule in the first place -- see Phase 6).

Two presets ship with the package and are the paper's own defaults, not
placeholders:

* ``escher/configs/flux_conformal.yaml`` -- FLUX.1-dev, Tables 1-4 of the paper.
* ``escher/configs/pixeldit_conformal.yaml`` -- PixelDiT-1300M, the paper's
  re-matched schedule for that backbone (different step count and sigma
  schedule; see the file's comments for the source of each value).

``load_preset`` resolves a bare name (no path separator, no ``.yaml``/``.yml``
suffix) against the shipped ``escher/configs/`` package data via
``importlib.resources`` (so it works from a wheel, not just a source checkout);
anything else (an absolute path, a relative path, or an explicit
``.yaml``/``.yml`` filename) is treated as a literal filesystem path, so a
user's own preset file works the same way as a shipped one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any, Optional, Tuple

import yaml


@dataclass
class SampleConfig:
    """Every knob for one ``escher.sampler.sample()`` run, plus backbone selection.

    Fields with no default (``model_id``, ``backbone``, ``num_steps``) select
    and size the backbone; everything else mirrors a ``sample()`` keyword
    argument 1:1, with the same default as ``sample()`` itself.
    """

    # --- backbone selection (outside sample()'s own signature) -------------
    model_id: str
    backbone: str
    num_steps: int

    # --- sample() kwargs (defaults mirror escher.sampler.sample) -----------
    prompt: str = ""
    family: str = "conformal"
    inset_scale: float = 1 / 4
    periods: int = 1
    sigma_hi: float = 0.87
    sigma_lo: float = 0.5
    n_ops: int = 3
    op_gap: int = 9
    warmup_sigma: Optional[float] = 0.95
    focus: Tuple[float, float] = (0.5, 0.5)
    upres: str = "off"
    zoom: float = 0.5
    supersample: int = 2
    time_travel: Optional[Tuple[int, float]] = None
    seed: int = 0
    size: int = 1024   # output images are square (size x size)
    cfg_scale: float = 3.5
    family_params: Optional[dict] = None

    def overrides(self, **kw: Any) -> "SampleConfig":
        """Return a new ``SampleConfig`` with ``kw`` applied. Does not mutate ``self``."""
        return replace(self, **kw)


def load_preset(name_or_path: str) -> SampleConfig:
    """Load a ``SampleConfig`` from a shipped preset name or a YAML file path.

    A bare name (e.g. ``"flux_conformal"``, no path separator or ``.yaml``/
    ``.yml`` suffix) resolves to the shipped ``escher/configs/<name>.yaml``
    package data via ``importlib.resources`` (wheel-safe). Anything else -- an
    absolute path, a relative path, or an explicit ``.yaml``/``.yml`` filename
    -- is opened as-is.
    """
    is_bare_name = (
        os.sep not in name_or_path
        and (os.altsep is None or os.altsep not in name_or_path)
        and not name_or_path.lower().endswith((".yaml", ".yml"))
    )
    if is_bare_name:
        text = files("escher").joinpath("configs", f"{name_or_path}.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    else:
        with open(name_or_path, "r") as f:
            data = yaml.safe_load(f) or {}

    # YAML has no tuple type; sample()'s (x, y)-shaped kwargs come back as
    # lists and are normalized here so equality/usage matches sample()'s own
    # Tuple[...] annotations.
    if data.get("focus") is not None:
        data["focus"] = tuple(data["focus"])
    if data.get("time_travel") is not None:
        data["time_travel"] = tuple(data["time_travel"])

    return SampleConfig(**data)
