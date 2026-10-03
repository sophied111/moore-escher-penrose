"""batch_run -- slim load-once batch runner over ``escher`` CLI config lines.

Each non-comment/non-blank line in a config file is exactly the flag surface
of one ``escher <flags>`` invocation (parsed with ``cli.build_config``, so a
sweep file is literally one CLI call per line -- no separate flag grammar to
maintain). Configs that share a load key -- ``(model_id, backbone)`` -- are
grouped so the expensive backbone build happens once per group and is reused
across every config in it; only the RNG state is reset between configs
sharing a backbone (``reset_for_config``), mirroring the reference
load-once "fast mode" batch runner this was ported from.

Stripped relative to that reference implementation: SLURM array-job env
sharding (``--shard`` here is only ever an explicit ``i/NW`` string -- nothing
reads a ``SLURM_ARRAY_TASK_*`` env var), Weights & Biases, the
``main_escher`` subprocess entry point, and the strict
(one-subprocess-per-config) reproduction mode -- this runner only ever runs
in a single process, load-once ("fast" mode).

``read_config_lines`` / ``parse_line`` / ``group_by_load_key`` touch only
``cli.build_config`` and ``escher.config`` (pure dataclass + YAML, no model,
no GPU) so they stay CPU/unit-testable; the backbone build and
``escher.sampler.sample`` call are reached only from inside ``run``'s group
loop.
"""

from __future__ import annotations

import gc
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cli import build_config
from escher.config import SampleConfig

ConfigItem = Tuple[SampleConfig, str]


def read_config_lines(path: str) -> List[str]:
    """Return non-blank, non-comment lines from ``path``, stripped.

    A line whose first non-whitespace character is ``#`` is a comment; blank
    (whitespace-only) lines are skipped too.
    """
    lines: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def parse_line(line: str) -> ConfigItem:
    """Parse one config-file line into a ``(SampleConfig, out_path)`` pair.

    Shlex-splits the line into argv tokens and hands them to
    ``cli.build_config`` -- the same preset-plus-overrides resolution the
    ``escher`` CLI itself uses, so a sweep-file line and a CLI invocation
    never drift apart. No model/GPU import happens here.
    """
    return build_config(shlex.split(line))


def load_key(cfg: SampleConfig) -> Tuple[str, str]:
    """The fields that define the expensive load. Configs sharing this key
    reuse one backbone instead of paying for a rebuild."""
    return (cfg.model_id, cfg.backbone)


def group_by_load_key(configs: List[ConfigItem]) -> List[Tuple[Tuple[str, str], List[ConfigItem]]]:
    """Group ``(config, out_path)`` items by :func:`load_key`.

    Preserves first-seen key order and stable within-group order, so a sweep
    file mixing backbones still runs deterministically -- one backbone build
    per distinct ``(model_id, backbone)``, each followed by every config that
    wants it. Returns ``[(key, [(cfg, out_path), ...]), ...]``.
    """
    order: List[Tuple[str, str]] = []
    buckets: Dict[Tuple[str, str], List[ConfigItem]] = {}
    for cfg, out_path in configs:
        k = load_key(cfg)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append((cfg, out_path))
    return [(k, buckets[k]) for k in order]


def _select_shard(lines: List[str], shard: Optional[str]) -> List[str]:
    """Return this worker's lines for an explicit ``"i/NW"`` --shard spec.

    ``shard is None`` means "run every line" -- there is no SLURM env
    fallback here (single-machine / manual use only).
    """
    if shard is None:
        return lines
    slot_s, _, nw_s = shard.partition("/")
    slot, nw = int(slot_s), int(nw_s)
    if nw < 1:
        raise ValueError(f"num_workers must be >= 1, got {nw}")
    if not (0 <= slot < nw):
        raise ValueError(f"shard slot {slot} out of range for {nw} workers")
    return [line for idx, line in enumerate(lines) if idx % nw == slot]


def _build_backbone(cfg: SampleConfig) -> Any:
    """Construct the backbone ``cfg`` selects. Heavy/model-loading; only
    ever called from inside :func:`run`'s group loop, never at import."""
    if cfg.backbone == "flux":
        from escher.backbones.flux import FluxBackbone

        return FluxBackbone(model_id=cfg.model_id, num_inference_steps=cfg.num_steps)
    elif cfg.backbone == "pixeldit":
        from escher.backbones.pixeldit import PixelDiTBackbone

        return PixelDiTBackbone(model_id=cfg.model_id, num_inference_steps=cfg.num_steps)
    raise ValueError(f"unknown backbone: {cfg.backbone!r}")


def reset_for_config(cfg: SampleConfig) -> None:
    """Re-seed global RNG state before a config that shares a backbone with
    the previous one in its group.

    Belt-and-suspenders, not the real isolation: ``escher.sampler.sample``
    draws its own per-call generator from ``cfg.seed`` via
    ``backbone.init_state``, so each call already re-derives its full RNG
    stream from the config's own seed and cannot leak state from a prior
    config in the same group. This only guards any stray global-RNG use.
    """
    import torch

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class ConfigResult:
    out_path: str
    status: str  # "ok" | "failed"
    error: Optional[str]


def run(config_file: str, shard: Optional[str] = None) -> None:
    """Run every config line in ``config_file`` (or this worker's ``shard`` of it).

    Groups configs by load key so each distinct backbone is built exactly
    once; every other config sharing that group only pays for a seed reset
    before sampling. A single config's failure is caught, logged, and never
    aborts the rest of the sweep -- ``gc.collect()`` (+ CUDA cache clear)
    runs after every config, and again after a group's backbone is freed
    before the next group loads, so a sweep mixing backbones never stacks
    two models in memory.
    """
    from escher.sampler import sample

    lines = _select_shard(read_config_lines(config_file), shard)
    configs = [parse_line(line) for line in lines]

    results: List[ConfigResult] = []
    for _key, group in group_by_load_key(configs):
        backbone = None
        for cfg, out_path in group:
            try:
                if backbone is None:
                    backbone = _build_backbone(cfg)
                else:
                    reset_for_config(cfg)

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
                )

                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                image.save(out_path)

                results.append(ConfigResult(out_path, "ok", None))
                print(f"[batch_run] ok: {out_path}", flush=True)
            except BaseException as exc:  # noqa: BLE001 - one config must never abort the sweep
                kind = f"{type(exc).__name__}: {exc}"
                print(f"[batch_run] FAILED {out_path} ({kind}); continuing.", flush=True)
                results.append(ConfigResult(out_path, "failed", kind))
            finally:
                gc.collect()
                _empty_cuda_cache()

        if backbone is not None:
            del backbone
            _empty_cuda_cache()

    ok = sum(r.status == "ok" for r in results)
    failed = sum(r.status == "failed" for r in results)
    print(f"[batch_run] done: ok={ok} failed={failed}", flush=True)
    for r in results:
        if r.status == "failed":
            print(f"  FAILED {r.out_path}: {r.error}", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Load-once batch runner over escher config lines")
    p.add_argument("config_file")
    p.add_argument("--shard", default=None,
                    help="explicit i/NW shard spec (default: run every line)")
    args = p.parse_args()
    run(args.config_file, shard=args.shard)


if __name__ == "__main__":
    main()
