"""Ablation-matrix sweep driver for Phase-1 axis validation.

Per the live plan (``../../raw_ideas/context/surveys/03_mem2/06_ablation_plan.md``):
Phase-1 validates each axis A-F individually by holding all others at a
sensible default and varying the axis of interest.

Usage::

    python scripts/sweeps/ablation_matrix.py --axis A --seeds 3 --limit 20
    python scripts/sweeps/ablation_matrix.py --axis B --seeds 3 --benchmark arc_agi
    python scripts/sweeps/ablation_matrix.py --axis D --seeds 3 --variants all

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
#                       Axis condition catalogue                      #
# ------------------------------------------------------------------- #

AXIS_A_CONDITIONS = [
    # off: use plain PS builder; on: reorg with various sub-axis settings.
    {"label": "reorg_off", "builder": "arcmemo_ps", "builder_cfg": {}},
    {
        "label": "reorg_on_graph_mdl_global_plateau",
        "builder": "arcmemo_reorg",
        "builder_cfg": {
            "input_basis": "graph_intrinsic",
            "objective": "mdl",
            "scope": "global_rebuild",
            "trigger": "plateau",
        },
    },
    {
        "label": "reorg_on_trace_mdl_accretive_everyk",
        "builder": "arcmemo_reorg",
        "builder_cfg": {
            "input_basis": "trace_based",
            "objective": "mdl",
            "scope": "accretive",
            "trigger": "every_k",
            "every_k": 20,
        },
    },
]

AXIS_B_CONDITIONS = [
    {"label": "flat_topk", "retriever": "ps_selector", "retriever_cfg": {"top_k": 10}},
    {
        "label": "graph_traversal",
        "retriever": "graph_traversal",
        "retriever_cfg": {"top_k": 10, "bfs_depth": 3, "prefer_aggregates": True},
    },
]

AXIS_C_CONDITIONS = [
    {"label": "one_shot", "retriever": "ps_selector", "retriever_cfg": {"top_k": 10}},
    {
        "label": "rrmc_multi_round",
        "retriever": "rrmc_interactive",
        "retriever_cfg": {
            "top_k": 10,
            "per_round_k": 3,
            "max_rounds": 5,
            "convergence_patience": 2,
        },
    },
]

AXIS_D_VARIANTS = [
    "arcmemo_oe",  # baseline 1
    "arcmemo_ps",  # baseline 2
    "minimal",
    "typed_only",
    "cue_heavy",
    "free_text",
    "structured_routine",
]

AXIS_E_CONDITIONS = [
    {"label": "empty_start", "builder": "arcmemo_ps", "builder_cfg": {}},
    {
        "label": "barc_seeded",
        "builder": "barc_ingest",
        "builder_cfg": {"barc_dir": "../arc_memo/data/dataset/src/BARC/seeds"},
    },
]

AXIS_F_CONDITIONS = [
    {
        "label": "hand_coded_reorg",
        "builder": "arcmemo_reorg",
        "builder_cfg": {
            "input_basis": "graph_intrinsic",
            "objective": "mdl",
            "scope": "global_rebuild",
            "trigger": "every_k",
            "every_k": 20,
        },
    },
    {
        "label": "alma_style_metaedit",
        "builder": "alma_style_metaedit",
        "builder_cfg": {
            "input_basis": "graph_intrinsic",
            "objective": "mdl",
            "scope": "global_rebuild",
            "trigger": "every_k",
            "every_k": 20,
        },
    },
]


# ------------------------------------------------------------------- #
#                       Config construction                           #
# ------------------------------------------------------------------- #

@dataclass(slots=True)
class Condition:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)


def _deep_set(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def build_conditions_for_axis(axis: str, *, variants: list[str] | None) -> list[Condition]:
    if axis == "A":
        return [
            Condition(c["label"], {
                "pipeline.memory_builder": c["builder"],
                "components.memory_builder": c["builder_cfg"],
            })
            for c in AXIS_A_CONDITIONS
        ]
    if axis == "B":
        return [
            Condition(c["label"], {
                "pipeline.memory_builder": "arcmemo_ps",
                "components.memory_builder": {},
                "pipeline.memory_retriever": c["retriever"],
                "components.memory_retriever": c["retriever_cfg"],
            })
            for c in AXIS_B_CONDITIONS
        ]
    if axis == "C":
        return [
            Condition(c["label"], {
                "pipeline.memory_builder": "arcmemo_ps",
                "components.memory_builder": {},
                "pipeline.memory_retriever": c["retriever"],
                "components.memory_retriever": c["retriever_cfg"],
            })
            for c in AXIS_C_CONDITIONS
        ]
    if axis == "D":
        chosen = variants or AXIS_D_VARIANTS
        out: list[Condition] = []
        for v in chosen:
            if v in {"arcmemo_oe", "arcmemo_ps"}:
                out.append(Condition(v, {
                    "pipeline.memory_builder": v,
                    "components.memory_builder": {},
                }))
            else:
                out.append(Condition(f"variant_{v}", {
                    "pipeline.memory_builder": "variant_format",
                    "components.memory_builder": {"variant": v},
                }))
        return out
    if axis == "E":
        return [
            Condition(c["label"], {
                "pipeline.memory_builder": c["builder"],
                "components.memory_builder": c["builder_cfg"],
            })
            for c in AXIS_E_CONDITIONS
        ]
    if axis == "F":
        return [
            Condition(c["label"], {
                "pipeline.memory_builder": c["builder"],
                "components.memory_builder": c["builder_cfg"],
            })
            for c in AXIS_F_CONDITIONS
        ]
    raise ValueError(f"unknown axis '{axis}' — use A/B/C/D/E/F")


def load_base_config(path: Path) -> dict:
    from mem2.cli.run import deep_merge, load_yaml
    cfg = load_yaml(path)
    if "_base_" in cfg:
        base = load_yaml((path.parent / cfg["_base_"]).resolve())
        cfg = deep_merge(base, {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


def apply_condition(
    base_cfg: dict,
    cond: Condition,
    *,
    seed: int,
    benchmark: str,
    limit: int,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # Never sweep with strict_arcmemo_compat on — the guard rejects new builders.
    cfg.setdefault("run", {})["strict_arcmemo_compat"] = False
    cfg["run"]["seed"] = int(seed)
    cfg["run"]["run_type"] = f"phase1_sweep_{cond.label}_s{seed}"
    cfg.setdefault("components", {}).setdefault("benchmark", {})["limit"] = int(limit)
    cfg.setdefault("pipeline", {})["benchmark"] = benchmark
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
) -> list[dict]:
    from mem2.orchestrator.runner import run_sync
    from mem2.orchestrator.wiring import resolve_components

    conditions = build_conditions_for_axis(axis, variants=variants)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=list("ABCDEF"))
    p.add_argument("--seeds", default="42,43,44", help="comma-separated seeds")
    p.add_argument("--limit", type=int, default=20, help="problems per run")
    p.add_argument("--benchmark", default="arc_agi",
                   help="benchmark adapter name (arc_agi until ARC-3 SDK is wired)")
    p.add_argument("--base-config", default="configs/experiments/arcmemo_arc_strict.yaml")
    p.add_argument("--output-dir", default="outputs/phase1_sweeps")
    p.add_argument("--variants", default="all", help="axis D only: 'all' or csv")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = None
    if args.axis == "D" and args.variants and args.variants != "all":
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    results = run_sweep(
        args.axis,
        seeds=seeds,
        limit=args.limit,
        benchmark=args.benchmark,
        base_config_path=Path(args.base_config),
        output_dir=Path(args.output_dir) / f"axis_{args.axis}",
        variants=variants,
        dry_run=args.dry_run,
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
