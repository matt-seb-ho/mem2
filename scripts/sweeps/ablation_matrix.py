"""Ablation-matrix sweep driver for Phase-1 axis validation.

Per the live plan (``../../raw_ideas/context/surveys/03_mem2/06_ablation_plan.md``):
Phase-1 validates each axis 1-6 individually by holding all others at a
sensible default and varying the axis of interest. (Axes were renumbered
from letters A-F on 2026-04-26 to reflect execution priority — see
`configs/axes/_index.yaml`. Old letter labels are retired; port IDs like
"A.2 reorg_dreamcoder" remain stable historical identifiers.)

Usage::

    python scripts/sweeps/ablation_matrix.py --axis 2 --seeds 3 --limit 20
    python scripts/sweeps/ablation_matrix.py --axis 1 --seeds 3 --benchmark arc_agi
    python scripts/sweeps/ablation_matrix.py --axis 4 --seeds 3 --variants all

ARC-3 note
----------
ARC-3 SDK is not integrated at the time of writing (2026-04-19). Until the
SDK lands, this driver defaults to benchmark=``arc_agi`` (ARC-1/2 data),
which provides the same relative-gain signal Phase-1 needs. When the SDK is
wired, pass ``--benchmark arc3_sdk`` to switch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- #
#                       Catalog-driven conditions                      #
# ------------------------------------------------------------------- #
#
# As of Phase 0 refactor (2026-04-22), condition catalogs live in
# `configs/axes/<axis>.yaml`. This file used to hardcode six
# AXIS_*_CONDITIONS lists and a `build_conditions_for_axis` dispatch
# with per-axis special cases — now a thin loader + wrapper.
#
# To add an axis 7 (next-available numeric, 2026-04-26 rename):
#   1. Drop `configs/axes/7.yaml` (see catalog schema in src/mem2/sweeps/catalog.py)
#   2. Append `- "7"` to `configs/axes/_index.yaml`'s `order` list
#   3. Run this driver with `--axis 7`. No Python edits required.

from mem2.sweeps.catalog import (
    AxisCatalog,
    conditions_from_catalog,
    load_axis_catalog,
    load_axis_index,
)


@dataclass(slots=True)
class Condition:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)


def _deep_set(cfg: dict, dotted: str, value: Any) -> None:
    """Set a dotted-path key.

    Semantics:
      - If the leaf value is a dict AND the existing key is also a dict → merge.
      - If the leaf value is None → replace with None (strip the inherited key
        when the Python-level wiring validator's None-stripping kicks in).
        Critical for axis 6's `empty_start` (init ablation; was axis E pre-rename)
        which explicitly nulls inherited `seed_memory_file` / `seed_annotations_file`.
      - Otherwise → replace.

    The None-as-replace rule means YAML `null` leaves propagate as explicit
    None values into the merged config, where `_build_component` in
    orchestrator/wiring.py strips them before registry dispatch.
    """
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    leaf = parts[-1]
    if isinstance(value, dict) and isinstance(cur.get(leaf), dict):
        cur[leaf].update(value)
    else:
        cur[leaf] = value


def build_conditions_for_axis(
    axis: str,
    *,
    variants: list[str] | None,
    axes_dir: Path | str = "configs/axes",
    include_spec_only: bool = False,
) -> list[Condition]:
    """Load the axis catalog YAML and return a list of Conditions.

    `variants` acts as a label-filter over the catalog. Backward compat for
    axis 4 (format; was axis D pre-rename): short variant names like
    "minimal" get translated to full labels `variant_minimal`, and baseline
    short names `arcmemo_oe`/`arcmemo_ps` pass through unchanged.
    """
    catalog: AxisCatalog = load_axis_catalog(axis, axes_dir)
    label_filter: list[str] | None = None
    if variants:
        label_filter = []
        for v in variants:
            if axis == "D" and v not in {"arcmemo_oe", "arcmemo_ps"}:
                label_filter.append(f"variant_{v}")
            else:
                label_filter.append(v)
    entries = conditions_from_catalog(
        catalog, variants=label_filter, include_spec_only=include_spec_only,
    )
    return [Condition(label=lbl, overrides=dict(ov)) for lbl, ov in entries]


def load_base_config(path: Path) -> dict:
    from mem2.cli.run import deep_merge, load_yaml
    cfg = load_yaml(path)
    if "_base_" in cfg:
        base = load_yaml((path.parent / cfg["_base_"]).resolve())
        cfg = deep_merge(base, {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


def load_arc_eval_100_ids() -> list[str]:
    """Load the 100-problem ArcMemo-paper eval split."""
    splits_path = Path("data/arc_agi/splits.json")
    if not splits_path.exists():
        return []
    splits = json.loads(splits_path.read_text())
    return list(splits.get("eval_100", {}).get("ids", []))


def apply_condition(
    base_cfg: dict,
    cond: Condition,
    *,
    seed: int,
    benchmark: str,
    limit: int,
    problem_split: str = "eval_100",
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # Never sweep with strict_arcmemo_compat on — the guard rejects new builders.
    cfg.setdefault("run", {})["strict_arcmemo_compat"] = False
    cfg["run"]["seed"] = int(seed)
    cfg["run"]["run_type"] = f"phase1_sweep_{cond.label}_s{seed}"
    cfg.setdefault("components", {}).setdefault("benchmark", {})["limit"] = int(limit)
    cfg.setdefault("pipeline", {})["benchmark"] = benchmark

    # ARC-AGI-1 100-set: load the eval_100 problem IDs and point benchmark at
    # the evaluation dir. Only touch arc_agi; other benchmarks keep their own defaults.
    if benchmark == "arc_agi" and problem_split == "eval_100":
        eval_ids = load_arc_eval_100_ids()
        if eval_ids:
            bm = cfg["components"]["benchmark"]
            bm.setdefault("data_root", "data/arc_agi/evaluation")
            bm["include_ids"] = eval_ids
            if limit and limit > 0:
                bm["include_ids"] = eval_ids[:limit]
            bm["limit"] = 0  # 0 = no extra cap beyond include_ids

    for dotted, value in cond.overrides.items():
        _deep_set(cfg, dotted, value)
    # Inference engine seed — most mem2 runs tie this to run.seed.
    ie = cfg.setdefault("components", {}).setdefault("inference_engine", {})
    ie.setdefault("gen_cfg", {})["seed"] = int(seed)
    return cfg


# ------------------------------------------------------------------- #
#                       Sweep runner                                  #
# ------------------------------------------------------------------- #

def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_sweep(
    axis: str,
    *,
    seeds: list[int],
    limit: int,
    benchmark: str,
    base_config_path: Path,
    output_dir: Path,
    variants: list[str] | None = None,
    dry_run: bool = False,
    axes_dir: Path | str = "configs/axes",
    include_spec_only: bool = False,
) -> list[dict]:
    from mem2.orchestrator.runner import run_sync
    from mem2.orchestrator.wiring import resolve_components

    conditions = build_conditions_for_axis(
        axis, variants=variants, axes_dir=axes_dir, include_spec_only=include_spec_only,
    )
    base = load_base_config(base_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for cond in conditions:
        for seed in seeds:
            cfg = apply_condition(base, cond, seed=seed, benchmark=benchmark, limit=limit)
            chash = config_hash(cfg)
            entry = {
                "axis": axis,
                "condition": cond.label,
                "seed": seed,
                "limit": limit,
                "benchmark": benchmark,
                "config_hash": chash,
            }
            run_id = f"phase1_{axis}_{cond.label}_s{seed}_{chash}"
            run_dir = output_dir / run_id
            if dry_run:
                entry["run_dir"] = str(run_dir)
                entry["status"] = "dry_run"
                results.append(entry)
                logger.info("[dry-run] axis=%s cond=%s seed=%s", axis, cond.label, seed)
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

            t0 = time.time()
            try:
                components = resolve_components(cfg)
                bundle = run_sync(cfg, components)
                entry.update({
                    "status": "ok",
                    "summary": bundle.summary,
                    "duration_s": round(time.time() - t0, 1),
                })
            except Exception as exc:
                entry.update({
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_s": round(time.time() - t0, 1),
                })
                logger.exception("sweep run failed: %s", cond.label)
            results.append(entry)
            (run_dir / "result.json").write_text(json.dumps(entry, indent=2, default=str))

    (output_dir / f"sweep_axis_{axis}.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    return results


STEP_DEFAULTS = {
    # Stage semantics encoded from docs/phase1_grid_search_plan / go-make-the-plan
    "4a": {"seeds": [42],               "limit": 10,  "purpose": "cheap screen, 1 seed × 10 problems"},
    "4b": {"seeds": [42, 43, 44],       "limit": 25,  "purpose": "small signal, 3 seeds × 25 problems"},
    "4c": {"seeds": [42, 43, 44],       "limit": 100, "purpose": "confirmatory, 3 seeds × full eval_100"},
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    # --axis accepts any axis name declared in the axes-dir _index.yaml.
    # Names are numeric strings ("1".."6", execution-priority order, 2026-04-26
    # rename); historically were letters ("A".."F"). Drop a 7.yaml to add axis 7.
    p.add_argument("--axis", required=True,
                   help="Axis name (numeric string, e.g. '1'..'6'); must match a YAML file in --axes-dir")
    p.add_argument("--step", choices=list(STEP_DEFAULTS.keys()),
                   help="Stage preset. Overrides --seeds + --limit with the plan's stage defaults.")
    p.add_argument("--seeds", default="42,43,44", help="comma-separated seeds (ignored if --step set)")
    p.add_argument("--limit", type=int, default=20,
                   help="problems per run (ignored if --step set; takes deterministic prefix of eval_100 for arc_agi)")
    p.add_argument("--problem-split-head", type=int, default=None,
                   help="alias for --limit, clearer name: take first N ids of eval_100")
    p.add_argument("--benchmark", default="arc_agi",
                   help="benchmark adapter name (arc_agi until ARC-3 SDK is wired)")
    p.add_argument("--base-config", default="configs/experiments/phase1_arc_base.yaml")
    p.add_argument("--output-dir", default="outputs/phase1_sweeps")
    p.add_argument("--axes-dir", default="configs/axes",
                   help="Directory with per-axis YAML catalogs + _index.yaml")
    p.add_argument("--variants", default="all",
                   help="Axis-D convenience: 'all' or csv of variant short names "
                        "(auto-prefixed to variant_<name>); for other axes, csv "
                        "of exact condition labels to filter to.")
    p.add_argument("--include-spec-only", action="store_true",
                   help="Run conditions flagged spec_only=true in the catalog "
                        "(default skips them since no local implementation).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Validate axis against _index.yaml when possible; tolerate index absence
    # for non-standard setups (unit tests, one-off axis files).
    try:
        idx = load_axis_index(Path(args.axes_dir))
        if idx.order and args.axis not in idx.order:
            logger.warning(
                "axis %r not in %s/_index.yaml order %s — proceeding anyway",
                args.axis, args.axes_dir, idx.order,
            )
    except FileNotFoundError:
        pass

    if args.step:
        preset = STEP_DEFAULTS[args.step]
        seeds = list(preset["seeds"])
        limit = int(preset["limit"])
        logger.info(f"--step {args.step}: {preset['purpose']} (seeds={seeds}, limit={limit})")
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        limit = int(args.problem_split_head) if args.problem_split_head is not None else int(args.limit)
    variants = None
    if args.variants and args.variants != "all":
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Organize outputs by step when a step preset is active
    axis_out = Path(args.output_dir) / (f"step_{args.step}" if args.step else "") / f"axis_{args.axis}"

    results = run_sweep(
        args.axis,
        seeds=seeds,
        limit=limit,
        benchmark=args.benchmark,
        base_config_path=Path(args.base_config),
        output_dir=axis_out,
        variants=variants,
        dry_run=args.dry_run,
        axes_dir=args.axes_dir,
        include_spec_only=args.include_spec_only,
    )

    # Console summary
    print(f"\n=== Axis {args.axis} sweep summary ({len(results)} runs) ===")
    for r in results:
        head = f"{r['condition']:<40}  s{r['seed']}  {r['status']}"
        if r.get("summary"):
            head += f"  score={r['summary'].get('official_score', '?')}"
        if r.get("error"):
            head += f"  ERR={r['error'][:120]}"
        print(head)


if __name__ == "__main__":
    main()
