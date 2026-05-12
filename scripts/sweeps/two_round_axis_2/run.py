"""Two-round axis-2 (memory reorganization) sweep runner.

Why this exists
---------------
All axis-2 reorg builders run their actual mechanism in `consolidate()`,
which fires ONCE in `runner.py:714` AFTER all 3 scoring passes. Within a
single run, reorg cannot affect scores. To measure reorg effect, we need
two rounds:

  Round 1 (warmup): condition's builder + baseline retriever, 3 iters
                    → solve, accumulate outcomes, consolidate fires
                    → save reorganized memory (memory/final.json)

  Round 2 (eval):   passive arcmemo_ps builder + chosen retriever, 1 iter
                    → load round-1 memory as seed, solve fresh
                    → score = official accuracy of round 2

Per condition, we run BOTH:
  (a) Round 2 with baseline retriever (ps_topk) — lower-bound: "does this
      reorg help any retriever?"
  (b) Round 2 with the condition's matching retriever — upper-bound:
      "does this reorg work with its paper-paired retrieval?"

For 8 of 12 axis-2 conditions, the matching retriever IS ps_topk, so (a)
and (b) collapse to one run. Only 4 conditions need both.

Usage
-----
  cd mem2 && source .env
  .venv/bin/python scripts/sweeps/two_round_axis_2/run.py \
      [--seeds 42] [--limit 50] [--variants reorg_lrll,reorg_memp]

Outputs
-------
- outputs/two_round_axis_2/<seed>/<condition>__<round2_retriever>/
  - round1/  (full mem2 run dir with memory/final.json)
  - round2/  (full mem2 run dir with result.json)
  - summary.json  (round1 score, round2 score, delta vs baseline)
- outputs/two_round_axis_2/<seed>/_aggregate.json (cross-condition summary)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]  # mem2/

# Add mem2/src to path for imports when called as a script
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mem2.cli.run import _load_config_recursive
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components

BASE_CFG = ROOT / "configs" / "experiments" / "phase1_arc_base.yaml"

# Per-condition matching retriever (paper-paired). 8 of 12 default to
# baseline (ps_topk); 4 specify alternates that better match the paper's
# intent.
# Per-condition matching retriever (paper-paired). Covers axes 2 (memory
# reorganization) and 5 (LLM-driven metaedit) — both have the same
# consolidate-post-iter timing issue solved by the two-round design.
CONDITIONS: list[tuple[str, str, str]] = [
    # (axis, condition label, matching retriever name)
    # --- Axis 2: memory reorganization ---
    ("2", "accretive_prune", "ps_topk"),
    ("2", "reorg_dreamcoder", "ps_topk"),
    ("2", "reorg_stitch", "ps_topk"),
    ("2", "reorg_lilo", "ps_topk"),
    ("2", "reorg_lrll", "ps_topk"),
    ("2", "reorg_memp", "ps_topk"),
    ("2", "reorg_sleepgate", "ps_topk"),
    ("2", "reorg_amem", "colbert_rerank"),
    ("2", "reorg_evolver", "colbert_rerank"),
    ("2", "reorg_memtree", "hmem_hierarchical"),
    ("2", "reorg_on_graph_mdl_global_plateau", "graph_traversal"),
    ("2", "reorg_on_trace_mdl_accretive_everyk", "graph_traversal"),
    # --- Axis 5: LLM-driven metaedit ---
    ("5", "alma_style_metaedit", "colbert_rerank"),
    ("5", "adas_style_search", "colbert_rerank"),
]

REFERENCE_CONDITION = ("2", "arcmemo_ps_no_reorg", "ps_topk")  # passive control


def load_axis_yaml(axis: str) -> dict:
    path = ROOT / "configs" / "axes" / f"{axis}.yaml"
    return yaml.safe_load(path.read_text())


def find_condition_overrides(axis: str, label: str) -> dict:
    """Look up the dotted-override map for a condition in the axis YAML."""
    cat = load_axis_yaml(axis)
    for cond in cat.get("conditions", []):
        if cond.get("label") == label:
            return _flatten_condition(cond)
    raise ValueError(f"Condition {label!r} not found in axis {axis}.yaml")


def _flatten_condition(cond: dict) -> dict:
    """Translate axis-yaml condition spec into dotted-override dict."""
    overrides: dict[str, object] = {}
    group = cond.get("override_group", "builder")
    if group == "builder":
        if "builder" in cond:
            overrides["pipeline.memory_builder"] = cond["builder"]
        if "builder_cfg" in cond:
            overrides["components.memory_builder"] = cond["builder_cfg"]
    elif group == "retriever":
        if "retriever" in cond:
            overrides["pipeline.memory_retriever"] = cond["retriever"]
        if "retriever_cfg" in cond:
            overrides["components.memory_retriever"] = cond["retriever_cfg"]
    elif group == "combo":
        if "builder" in cond:
            overrides["pipeline.memory_builder"] = cond["builder"]
        if "builder_cfg" in cond:
            overrides["components.memory_builder"] = cond["builder_cfg"]
        if "retriever" in cond:
            overrides["pipeline.memory_retriever"] = cond["retriever"]
        if "retriever_cfg" in cond:
            overrides["components.memory_retriever"] = cond["retriever_cfg"]
    return overrides


def deep_set(d: dict, dotted: str, val: object) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    leaf = parts[-1]
    if isinstance(val, dict) and isinstance(cur.get(leaf), dict):
        cur[leaf].update(val)
    else:
        cur[leaf] = val


def load_arc_eval_100_ids() -> list[str]:
    splits_path = ROOT / "data" / "arc_agi" / "splits.json"
    if not splits_path.exists():
        return []
    splits = json.loads(splits_path.read_text())
    return list(splits.get("eval_100", {}).get("ids", []))


def build_round1_config(axis: str, label: str, seed: int, limit: int, run_root: Path, cache_dir: Path | None = None) -> dict:
    """Round 1: condition's builder + ps_topk retriever + 3 iters."""
    cfg = _load_config_recursive(BASE_CFG)
    cfg = copy.deepcopy(cfg)

    # Identify condition overrides from the condition's axis YAML
    overrides = find_condition_overrides(axis, label)
    for k, v in overrides.items():
        deep_set(cfg, k, v)

    # RN-005 finding 3: axis-2/5 reorg conditions default to every_k=20, which
    # doesn't fire at typical n=50 step counts (e.g. step=95 % 20 != 0). Force
    # every_k=1 so the trigger fires at the final consolidate. Only applies if
    # the builder takes an `every_k` parameter (axis-2 reorgs + axis-5 metaedit).
    # `accretive_prune` is a passive capacity-pressure variant that doesn't
    # accept `every_k` — skip the override for it.
    BUILDERS_WITHOUT_EVERY_K = {"accretive_prune"}
    if axis in ("2", "5"):
        builder_name = cfg.get("pipeline", {}).get("memory_builder", "")
        if builder_name not in BUILDERS_WITHOUT_EVERY_K:
            bcfg = cfg.get("components", {}).get("memory_builder", {}) or {}
            if isinstance(bcfg, dict):
                bcfg["every_k"] = 1
                deep_set(cfg, "components.memory_builder", bcfg)

    # RN-005 follow-up: per-cell cache dir to eliminate cross-cell contamination.
    if cache_dir is not None:
        deep_set(cfg, "components.provider.cache_dir", str(cache_dir.resolve()))

    # Force baseline retriever in round 1 for consistent outcome accumulation
    cfg.setdefault("pipeline", {})["memory_retriever"] = "ps_topk"
    deep_set(cfg, "components.memory_retriever", {"top_k": 10})

    # Run config
    cfg.setdefault("run", {})["seed"] = seed
    cfg["run"]["run_type"] = f"two_round_r1_{label}_s{seed}"
    cfg["run"]["strict_arcmemo_compat"] = False
    cfg["run"]["retry_policy"] = {
        "max_passes": 3,
        "criterion": "train",
        "error_feedback": "all",
        "num_feedback_passes": 1,
        "include_past_outcomes": True,
    }
    cfg["run"]["max_passes"] = 3
    deep_set(cfg, "components.inference_engine.gen_cfg.seed", seed)

    # Benchmark: eval_100 first N
    eval_ids = load_arc_eval_100_ids()
    if eval_ids and limit > 0:
        bm = cfg["components"]["benchmark"]
        bm["data_root"] = "data/arc_agi/evaluation"
        bm["include_ids"] = eval_ids[:limit]
        bm["limit"] = 0

    # Pipe runner's output root into our nested folder
    cfg["run"]["output_root"] = str(run_root.resolve())
    return cfg


def build_round2_config(
    label: str,
    round2_retriever: str,
    seed: int,
    limit: int,
    seed_memory_file: Path,
    run_root: Path,
    cache_dir: Path | None = None,
) -> dict:
    """Round 2: passive arcmemo_ps + chosen retriever + 1 iter."""
    cfg = _load_config_recursive(BASE_CFG)
    cfg = copy.deepcopy(cfg)

    # Passive base builder (no consolidate effects)
    cfg.setdefault("pipeline", {})["memory_builder"] = "arcmemo_ps"
    deep_set(cfg, "components.memory_builder", {
        "seed_memory_file": str(seed_memory_file.resolve()),
        "domain": "arc",
    })

    # Round-2 retriever choice
    cfg["pipeline"]["memory_retriever"] = round2_retriever
    if round2_retriever == "ps_topk":
        deep_set(cfg, "components.memory_retriever", {"top_k": 10})
    else:
        # Apply axis-1 condition cfg if defined
        try:
            ax1 = load_axis_yaml("1")
            for c in ax1.get("conditions", []):
                if c.get("label") == round2_retriever:
                    rcfg = c.get("retriever_cfg") or {}
                    deep_set(cfg, "components.memory_retriever", rcfg)
                    break
        except FileNotFoundError:
            pass

    cfg.setdefault("run", {})["seed"] = seed
    cfg["run"]["run_type"] = f"two_round_r2_{label}_{round2_retriever}_s{seed}"
    cfg["run"]["strict_arcmemo_compat"] = False
    # 1-iter round 2 — clean signal, no retry-policy noise
    cfg["run"]["retry_policy"] = {
        "max_passes": 1,
        "criterion": "train",
        "error_feedback": "all",
        "num_feedback_passes": 1,
        "include_past_outcomes": False,
    }
    cfg["run"]["max_passes"] = 1
    deep_set(cfg, "components.inference_engine.gen_cfg.seed", seed)

    eval_ids = load_arc_eval_100_ids()
    if eval_ids and limit > 0:
        bm = cfg["components"]["benchmark"]
        bm["data_root"] = "data/arc_agi/evaluation"
        bm["include_ids"] = eval_ids[:limit]
        bm["limit"] = 0

    cfg["run"]["output_root"] = str(run_root.resolve())
    if cache_dir is not None:
        deep_set(cfg, "components.provider.cache_dir", str(cache_dir.resolve()))
    return cfg


def run_one_round(cfg: dict, label: str) -> dict:
    """Run a single mem2 pipeline round. Returns the bundle.summary dict."""
    components = resolve_components(cfg)
    bundle = run_sync(cfg, components)
    return bundle.summary


def memory_final_to_seed_file(memory_final_path: Path, dst_path: Path) -> None:
    """Convert a memory/final.json artifact into a seed_memory_file format.

    `memory/final.json` schema: {schema_name, schema_version, payload, metadata}
    `seed_memory_file` schema (extended): {concepts, solutions, custom_types,
                                            categories?, reorg?}

    `ConceptMemory.load_from_file` only reads concepts/solutions/custom_types
    at load time, but round-2 RETRIEVERS may inspect `memory.payload["reorg"]`
    for lineage edges (graph_traversal, others). Pass the full payload through
    so round-2 retrievers see reorg state. Extra keys are silently ignored
    when not needed. (RN-005 finding 4.)
    """
    obj = json.loads(memory_final_path.read_text())
    payload = obj.get("payload", {})
    # Pass the full payload through (concepts/solutions/custom_types required;
    # categories/reorg/anything-else preserved for downstream consumers).
    seed_doc = dict(payload)
    seed_doc.setdefault("concepts", {})
    seed_doc.setdefault("solutions", {})
    seed_doc.setdefault("custom_types", {})
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(seed_doc, indent=2))


def run_one_condition(
    axis: str,
    label: str,
    round2_retriever: str,
    seed: int,
    limit: int,
    out_root: Path,
) -> dict:
    cell_dir = out_root / f"ax{axis}_{label}__{round2_retriever}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cell_dir / "summary.json"
    if summary_path.exists():
        print(f"  [skip] ax{axis}/{label} × {round2_retriever}: summary.json exists")
        return json.loads(summary_path.read_text())

    print(f"  [start] ax{axis}/{label} × {round2_retriever}")
    t0 = time.monotonic()

    # Cache strategy (RN-005 follow-up):
    # - Round 1 SHARES a cache across all cells in the sweep, because round-1
    #   prompts are identical across conditions by design (same retriever
    #   = ps_topk, same starting memory). Cache replay here is correct.
    # - Round 2 uses a PER-CELL cache, because post-consolidate memory differs
    #   per condition. Isolation is critical to avoid cross-cell contamination.
    shared_round1_cache = out_root / ".shared_round1_cache"
    shared_round1_cache.mkdir(parents=True, exist_ok=True)
    per_cell_round2_cache = cell_dir / ".llm_cache_round2"
    per_cell_round2_cache.mkdir(parents=True, exist_ok=True)

    # Round 1
    r1_root = cell_dir / "round1"
    r1_root.mkdir(parents=True, exist_ok=True)
    cfg1 = build_round1_config(axis, label, seed, limit, r1_root, cache_dir=shared_round1_cache)
    cfg1_path = r1_root / "config.yaml"
    cfg1_path.write_text(yaml.safe_dump(cfg1, sort_keys=False))
    r1_summary = run_one_round(cfg1, f"{label}/r1")
    r1_score = r1_summary.get("official_score", 0.0)
    print(f"    [r1 done] official={r1_score} ({time.monotonic()-t0:.1f}s)")

    # Find round-1 memory/final.json (path = <r1_root>/<run_type>/<run_id>/memory/final.json)
    r1_runs = list(r1_root.glob("*/*/memory/final.json"))
    if not r1_runs:
        raise RuntimeError(f"No memory/final.json under {r1_root}")
    r1_mem_final = r1_runs[0]

    # Convert to seed_memory_file format for round 2
    seed_file = cell_dir / "round1_memory_seed.json"
    memory_final_to_seed_file(r1_mem_final, seed_file)

    # Round 2
    r2_root = cell_dir / "round2"
    r2_root.mkdir(parents=True, exist_ok=True)
    cfg2 = build_round2_config(label, round2_retriever, seed, limit, seed_file, r2_root, cache_dir=per_cell_round2_cache)
    cfg2_path = r2_root / "config.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2, sort_keys=False))
    r2_summary = run_one_round(cfg2, f"{label}/r2")
    r2_score = r2_summary.get("official_score", 0.0)
    elapsed = time.monotonic() - t0
    print(f"    [r2 done] official={r2_score} (total {elapsed:.1f}s)")

    out = {
        "axis": axis,
        "label": label,
        "round2_retriever": round2_retriever,
        "seed": seed,
        "limit": limit,
        "round1_score": r1_score,
        "round2_score": r2_score,
        "round1_strict": r1_summary.get("strict_score"),
        "round2_strict": r2_summary.get("strict_score"),
        "wall_time_s": elapsed,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42", help="Comma list of seeds")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--variants", default=None,
                    help="Comma list of axis-2 labels to run (default: all 12)")
    ap.add_argument("--also-baseline", action="store_true",
                    help="For matching!=baseline conds, also run with ps_topk (lower bound)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",")}
        cells = [(a, l, r) for (a, l, r) in CONDITIONS if l in wanted]
    else:
        cells = list(CONDITIONS)

    # Optionally add baseline-retriever cell for non-ps_topk conditions
    if args.also_baseline:
        extra = []
        for axis, label, retriever in cells:
            if retriever != "ps_topk":
                extra.append((axis, label, "ps_topk"))
        cells = cells + extra

    out_base = ROOT / "outputs" / "two_round_axis_2"
    summary_rows: list[dict] = []

    for seed in seeds:
        seed_root = out_base / f"s{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== seed {seed} ({len(cells)} cells) ===")
        for (axis, label, retriever) in cells:
            try:
                row = run_one_condition(axis, label, retriever, seed, args.limit, seed_root)
                summary_rows.append(row)
            except Exception as e:
                print(f"  [FAIL] ax{axis}/{label} × {retriever}: {e}")
                summary_rows.append({
                    "axis": axis, "label": label, "round2_retriever": retriever,
                    "seed": seed, "error": str(e)[:200],
                })

        agg_path = seed_root / "_aggregate.json"
        agg_path.write_text(json.dumps({
            "seed": seed,
            "limit": args.limit,
            "results": summary_rows,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2))
        print(f"\n[aggregate] wrote {agg_path}")

    # Final scoreboard
    print("\n=== SCOREBOARD ===")
    print(f"{'ax':>2s} {'condition':36s} {'r2_retriever':22s} {'r1':>5s} {'r2':>5s}")
    for row in summary_rows:
        if "error" in row:
            print(f"{row.get('axis','?'):>2s} {row['label']:36s} {row['round2_retriever']:22s} ERR: {row['error'][:40]}")
        else:
            print(f"{row['axis']:>2s} {row['label']:36s} {row['round2_retriever']:22s} "
                  f"{row['round1_score']:5.1f} {row['round2_score']:5.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
