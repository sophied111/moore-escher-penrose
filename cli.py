"""cli -- minimal command-line front end over ``escher.config`` presets.

``escher <flags>`` (see ``pyproject.toml``'s ``[project.scripts]``) starts from
a named preset (``--preset``, default ``flux_conformal``) and applies only the
flags the user actually typed as overrides on top of it, then runs one T-cycle
sample and saves the image. No W&B, no ``ESCHER_ENGINE``, no legacy flags --
just the clean flag surface over :class:`escher.config.SampleConfig`.

``build_config`` is deliberately free of any model/GPU import so it stays
CPU/unit-testable: it only touches ``escher.config`` (pure dataclass + YAML).
Backbone construction (``escher.backbones``) and the sampling call
(``escher.sampler.sample``) happen lazily, inside :func:`main`, which is the
only function that ever needs a GPU or loaded weights.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional, Tuple

from escher.config import SampleConfig, load_preset

_FAMILIES = ["conformal", "poles", "mobius", "square", "rimrings"]
_BACKBONES = ["flux", "pixeldit"]
_UPRES_MODES = ["off", "bicubic", "sr"]

# Fallback (tt_n, tt_sigma) used when only one of --time-travel-n /
# --time-travel-sigma is given and the starting preset has no time_travel of
# its own to fill in the other half from (mirrors the legacy CLI's own
# tt_n=0 / tt_sigma=0.3 defaults).
_TIME_TRAVEL_DEFAULT: Tuple[int, float] = (0, 0.3)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="escher",
        description="Conformal T-cycle braided sampling over a SampleConfig preset.",
    )
    # --preset always has a real default: it picks which SampleConfig we
    # start from, so it is not itself an "override".
    p.add_argument("--preset", default="flux_conformal",
                    help="preset name (configs/<name>.yaml) or a path to a YAML file")

    # Every flag below defaults to argparse.SUPPRESS so an unset flag leaves
    # no attribute on the parsed namespace at all -- that absence is exactly
    # how build_config() tells "the user explicitly passed this" apart from
    # "leave the preset's own value alone".
    p.add_argument("--model", dest="model_id", metavar="MODEL", default=argparse.SUPPRESS,
                    help="backbone model id/checkpoint")
    p.add_argument("--backbone", choices=_BACKBONES, default=argparse.SUPPRESS)
    p.add_argument("--prompt", default=argparse.SUPPRESS)
    p.add_argument("--family", choices=_FAMILIES, default=argparse.SUPPRESS)
    p.add_argument("--inset-scale", type=float, default=argparse.SUPPRESS)
    p.add_argument("--periods", type=int, default=argparse.SUPPRESS)
    p.add_argument("--sigma-hi", type=float, default=argparse.SUPPRESS)
    p.add_argument("--sigma-lo", type=float, default=argparse.SUPPRESS)
    p.add_argument("--n-ops", type=int, default=argparse.SUPPRESS)
    p.add_argument("--op-gap", type=int, default=argparse.SUPPRESS)
    p.add_argument("--warmup-sigma", type=float, default=argparse.SUPPRESS)
    p.add_argument("--focus", nargs=2, type=float, metavar=("FX", "FY"),
                    default=argparse.SUPPRESS)
    p.add_argument("--zoom", type=float, default=argparse.SUPPRESS)
    p.add_argument("--supersample", type=int, default=argparse.SUPPRESS)
    p.add_argument("--upres", choices=_UPRES_MODES, default=argparse.SUPPRESS)
    p.add_argument("--time-travel-n", dest="time_travel_n", type=int, metavar="N",
                    default=argparse.SUPPRESS)
    p.add_argument("--time-travel-sigma", dest="time_travel_sigma", type=float,
                    metavar="SIGMA", default=argparse.SUPPRESS)
    p.add_argument("--steps", dest="num_steps", type=int, metavar="STEPS",
                    default=argparse.SUPPRESS)
    p.add_argument("--cfg-scale", type=float, default=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--size", type=int, default=argparse.SUPPRESS,
                    help="output image side length (images are square)")
    p.add_argument("--output", default=argparse.SUPPRESS,
                    help="output filename (default: output.png)")
    p.add_argument("--output-dir", dest="output_dir", metavar="DIR",
                    default=argparse.SUPPRESS,
                    help="directory to write --output into (default: none, "
                         "i.e. --output is used as-is)")
    p.add_argument("--monitor-every", dest="monitor_every", type=int, metavar="N",
                    default=argparse.SUPPRESS,
                    help="diagnostic: save a labelled montage of intermediate x0 "
                         "frames (warm-up, pre/post every warp, every Nth denoise "
                         "step, time-travel, final). 0 = off (default)")
    p.add_argument("--monitor-path", dest="monitor_path", metavar="PATH",
                    default=argparse.SUPPRESS,
                    help="montage PNG path (default: <output_stem>_monitor.png)")
    return p


def build_config(argv: Optional[List[str]] = None) -> Tuple[SampleConfig, str]:
    """Parse ``argv``, apply explicitly-passed flags onto ``--preset``.

    Starts from ``load_preset(args.preset)`` and applies ``.overrides(...)``
    with ONLY the flags the user actually typed -- every other flag defaults
    to ``argparse.SUPPRESS``, so it is simply absent from the parsed
    namespace rather than present with some sentinel value. This function
    touches only ``escher.config`` (pure dataclass + YAML): no backbone, no
    model, no GPU, so it is safe to call from CPU-only unit tests.

    Returns the resolved ``SampleConfig`` and the output path to save to
    (``--output-dir``/``--output`` joined, or ``--output`` alone, or the
    ``output.png`` default).
    """
    args = _build_parser().parse_args(argv)
    kw: dict[str, Any] = vars(args).copy()
    kw.pop("preset", None)
    # Monitor flags are run-time diagnostics, not SampleConfig fields; main()
    # reads them straight from the parsed args.
    kw.pop("monitor_every", None)
    kw.pop("monitor_path", None)

    output = kw.pop("output", None)
    output_dir = kw.pop("output_dir", None)

    cfg = load_preset(args.preset)

    # --time-travel-n / --time-travel-sigma independently fill in the two
    # halves of SampleConfig.time_travel (tt_n, tt_sigma); whichever half is
    # not given falls back to the preset's own time_travel, or the legacy
    # (0, 0.3) default if the preset has none.
    tt_n = kw.pop("time_travel_n", None)
    tt_sigma = kw.pop("time_travel_sigma", None)
    if tt_n is not None or tt_sigma is not None:
        base_n, base_sigma = cfg.time_travel if cfg.time_travel is not None else _TIME_TRAVEL_DEFAULT
        kw["time_travel"] = (
            tt_n if tt_n is not None else base_n,
            tt_sigma if tt_sigma is not None else base_sigma,
        )

    if "focus" in kw:
        kw["focus"] = tuple(kw["focus"])

    cfg = cfg.overrides(**kw) if kw else cfg

    out_path = os.path.join(output_dir, output) if output_dir else (output or "output.png")
    return cfg, out_path


def main(argv: Optional[List[str]] = None) -> str:
    """Build the config, run one T-cycle sample, and save it.

    Backbone construction and the ``escher.sampler``/model imports are
    deliberately lazy (only reached here, never from ``build_config``) so
    that unit-testing argument parsing never needs a GPU or loaded weights.
    """
    cfg, out_path = build_config(argv)
    margs = _build_parser().parse_args(argv)
    monitor_every = getattr(margs, "monitor_every", 0)
    monitor_path = getattr(margs, "monitor_path", None)
    if monitor_every > 0 and monitor_path is None:
        monitor_path = os.path.splitext(out_path)[0] + "_monitor.png"

    print(
        f"[escher] {cfg.backbone} | family={cfg.family} seed={cfg.seed} "
        f"steps={cfg.num_steps} size={cfg.size} -> {out_path}",
        flush=True,
    )

    print(f"[escher] loading {cfg.backbone} backbone weights...", flush=True)

    if cfg.backbone == "flux":
        from escher.backbones.flux import FluxBackbone

        backbone = FluxBackbone(model_id=cfg.model_id, num_inference_steps=cfg.num_steps)
    elif cfg.backbone == "pixeldit":
        from escher.backbones.pixeldit import PixelDiTBackbone

        backbone = PixelDiTBackbone(model_id=cfg.model_id, num_inference_steps=cfg.num_steps)
    else:
        raise ValueError(f"unknown backbone: {cfg.backbone!r}")

    from escher.sampler import sample

    print("[escher] backbone ready; sampling...", flush=True)

    image = sample(
        backbone,
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
        monitor_every=monitor_every,
        monitor_path=monitor_path,
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(out_path)
    print(f"[escher] saved {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    main()
