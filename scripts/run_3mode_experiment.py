#!/usr/bin/env python3
"""
3-Mode Experiment Runner: Baseline vs Full Concept vs Hybrid
4 independent runs per mode, each with max_passes=2, n=1.
Benchmarks: Math (200 problems) and LCB (100 problems).

Usage:
    python scripts/run_3mode_experiment.py --smoke       # smoketest (5 problems, 1 run)
    python scripts/run_3mode_experiment.py --full        # full experiment (all problems, 4 runs)
    python scripts/run_3mode_experiment.py --full --benchmark math  # math only
    python scripts/run_3mode_experiment.py --full --benchmark lcb   # lcb only
    python scripts/run_3mode_experiment.py --full --mode baseline   # baseline only
"""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ── Base config templates ────────────────────────────────────────────────

MATH_BASELINE = {
    "_base_": "../../base.yaml",
    "pipeline": {
        "task_adapter": "math_ps",
        "benchmark": "competition_math_ps",
        "memory_builder": "none",
        "memory_retriever": "none",
        "trajectory_policy": "single_path",
        "provider": "llmplus_openai",
        "inference_engine": "math_reason",
        "feedback_engine": "math_reason_gt",
        "evaluator": "math_reason_eval",
        "artifact_sink": "json_local",
    },
    "run": {
        "run_type": "m3_baseline_math_nano",
        "seed": 42,
        "max_passes": 2,
        "execution_mode": "arc_batch",
        "retry_criterion": "test",
        "retry_policy": {
            "max_passes": 2,
            "criterion": "test",
            "error_feedback": "all",
            "num_feedback_passes": 1,
            "include_past_outcomes": True,
        },
    },
    "components": {
        "task_adapter": {"task_name": "math_ps"},
        "benchmark": {
            "data_root": "data/competition_math_all_l5",
            "types": [
                "Algebra",
                "Counting & Probability",
                "Geometry",
                "Intermediate Algebra",
                "Number Theory",
                "Prealgebra",
                "Precalculus",
            ],
            "limit": 0,
        },
        "memory_builder": {},
        "memory_retriever": {},
        "trajectory_policy": {"retry_paths": 1},
        "provider": {
            "profile_name": "llmplus_openai",
            "dotenv_path": ".env.example",
            "default_max_concurrency": 16,
        },
        "inference_engine": {
            "model": "gpt-5-nano",
            "gen_cfg": {
                "n": 1,
                "max_tokens": 16384,
                "batch_size": 16,
                "seed": 42,
                "ignore_cache": True,
            },
            "error_feedback": "all",
            "num_feedback_passes": 1,
            "include_past_outcomes": True,
            "include_reselected_lessons": False,
            "prompt_options": None,
        },
        "feedback_engine": {},
        "evaluator": {"require_all_tests": None},
        "artifact_sink": {},
    },
}

LCB_BASELINE = {
    "_base_": "../../base.yaml",
    "pipeline": {
        "task_adapter": "livecodebench",
        "benchmark": "livecodebench",
        "memory_builder": "none",
        "memory_retriever": "none",
        "trajectory_policy": "single_path",
        "provider": "llmplus_openai",
        "inference_engine": "lcb_solve",
        "feedback_engine": "lcb_gt",
        "evaluator": "lcb_exec",
        "artifact_sink": "json_local",
    },
    "run": {
        "run_type": "m3_baseline_lcb_nano",
        "seed": 42,
        "max_passes": 2,
        "execution_mode": "arc_batch",
        "retry_criterion": "test",
        "retry_policy": {
            "max_passes": 2,
            "criterion": "test",
            "error_feedback": "all",
            "num_feedback_passes": 1,
            "include_past_outcomes": True,
        },
    },
    "components": {
        "task_adapter": {"task_name": "livecodebench"},
        "benchmark": {
            "data_root": "data/livecodebench_all",
            "limit": 0,
        },
        "memory_builder": {},
        "memory_retriever": {},
        "trajectory_policy": {"retry_paths": 1},
        "provider": {
            "profile_name": "llmplus_openai",
            "dotenv_path": ".env.example",
            "default_max_concurrency": 16,
        },
        "inference_engine": {
            "model": "gpt-5-nano",
            "gen_cfg": {
                "n": 1,
                "max_tokens": 16384,
                "batch_size": 16,
                "seed": 42,
                "ignore_cache": True,
            },
            "error_feedback": "all",
            "num_feedback_passes": 1,
            "include_past_outcomes": True,
            "include_reselected_lessons": False,
            "prompt_options": None,
        },
        "feedback_engine": {},
        "evaluator": {
            "timeout_s": 30.0,
            "require_all_tests": None,
        },
        "artifact_sink": {},
    },
}

# Problem IDs from existing configs
MATH_INCLUDE_IDS = None  # None = use all from data_root (limit: 0 loads eval set)
LCB_INCLUDE_IDS = None   # Will be loaded from existing config

SMOKE_MATH_IDS = ["cmath_10098", "cmath_10143", "cmath_10170", "cmath_2097", "cmath_5298"]
SMOKE_LCB_IDS = ["2757", "2921", "3025", "abc307_e", "abc366_e"]


def load_include_ids(config_path: str) -> list[str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["components"]["benchmark"].get("include_ids", [])


def make_concept_config(base: dict, benchmark: str) -> dict:
    cfg = copy.deepcopy(base)
    prefix = "m3_concept"
    cfg["pipeline"]["memory_builder"] = "arcmemo_ps"
    cfg["pipeline"]["memory_retriever"] = "ps_selector"
    if benchmark == "math":
        cfg["run"]["run_type"] = f"{prefix}_math_nano"
        cfg["components"]["memory_builder"] = {
            "seed_memory_file": "data/competition_math_all_l5/concept_memory_gpt5nano/extracted_v1.json",
            "domain": "math",
            "max_concepts": 500,
        }
        cfg["components"]["memory_retriever"] = {
            "domain": "math",
            "prompt_info_file": "data/competition_math_all_l5/concept_memory_gpt5nano/selection_v1/prompt_info.json",
        }
    else:
        cfg["run"]["run_type"] = f"{prefix}_lcb_nano"
        cfg["components"]["memory_builder"] = {
            "seed_memory_file": "data/livecodebench_all/concept_memory_gpt5nano/extracted_v1.json",
            "domain": "code",
            "max_concepts": 500,
        }
        cfg["components"]["memory_retriever"] = {
            "domain": "code",
            "prompt_info_file": "data/livecodebench_all/concept_memory_gpt5nano/selection_v2/prompt_info.json",
        }
    return cfg


def make_hybrid_config(base: dict, benchmark: str) -> dict:
    cfg = make_concept_config(base, benchmark)
    prefix = "m3_hybrid"
    if benchmark == "math":
        cfg["run"]["run_type"] = f"{prefix}_math_nano"
    else:
        cfg["run"]["run_type"] = f"{prefix}_lcb_nano"
    cfg["run"]["hybrid_concept_mode"] = True
    # Enable reselected lessons so retry prompts include concept hints
    cfg["components"]["inference_engine"]["include_reselected_lessons"] = True
    return cfg


def apply_smoke(cfg: dict, benchmark: str) -> dict:
    cfg = copy.deepcopy(cfg)
    ids = SMOKE_MATH_IDS if benchmark == "math" else SMOKE_LCB_IDS
    cfg["components"]["benchmark"]["include_ids"] = ids
    cfg["run"]["run_type"] = "smoke_" + cfg["run"]["run_type"]
    return cfg


def apply_include_ids(cfg: dict, benchmark: str) -> dict:
    """Add include_ids from existing configs to ensure same problem sets."""
    cfg = copy.deepcopy(cfg)
    if benchmark == "math":
        ids = load_include_ids("configs/experiments/eval_math_reason_gpt5nano_s42.yaml")
    else:
        ids = load_include_ids("configs/experiments/baseline_lcb_all_gpt5nano_s42.yaml")
    if ids:
        cfg["components"]["benchmark"]["include_ids"] = ids
    return cfg


def write_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_experiment(config_path: str, run_label: str) -> dict | None:
    """Run a single experiment and return the summary."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {run_label}")
    print(f"Config:  {config_path}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "mem2.cli.run",
        "--config", config_path,
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: {run_label} failed with return code {result.returncode}")
        return None

    # Find the output directory from the run
    # Parse from the last line of output that contains "Output directory:"
    print(f"\nCOMPLETED: {run_label}")
    return {"label": run_label, "returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="3-Mode Experiment Runner")
    parser.add_argument("--smoke", action="store_true", help="Smoketest (5 problems, 1 run)")
    parser.add_argument("--full", action="store_true", help="Full experiment (all problems, 4 runs)")
    parser.add_argument("--benchmark", choices=["math", "lcb"], help="Run only one benchmark")
    parser.add_argument("--mode", choices=["baseline", "concept", "hybrid"], help="Run only one mode")
    parser.add_argument("--num-runs", type=int, default=4, help="Number of runs per mode (default: 4)")
    parser.add_argument("--start-run", type=int, default=1, help="Start from run N (default: 1)")
    args = parser.parse_args()

    if not args.smoke and not args.full:
        parser.error("Specify --smoke or --full")

    config_dir = Path("configs/experiments/m3")
    benchmarks = [args.benchmark] if args.benchmark else ["math", "lcb"]
    modes = [args.mode] if args.mode else ["baseline", "concept", "hybrid"]
    num_runs = 1 if args.smoke else args.num_runs

    # Generate configs
    all_configs = []
    for bm in benchmarks:
        base = copy.deepcopy(MATH_BASELINE if bm == "math" else LCB_BASELINE)

        for mode in modes:
            if mode == "baseline":
                cfg = copy.deepcopy(base)
            elif mode == "concept":
                cfg = make_concept_config(base, bm)
            else:
                cfg = make_hybrid_config(base, bm)

            if args.smoke:
                cfg = apply_smoke(cfg, bm)
            else:
                cfg = apply_include_ids(cfg, bm)

            for run_idx in range(args.start_run, args.start_run + num_runs):
                run_cfg = copy.deepcopy(cfg)
                # Add run_index to config so each run gets a unique run_id hash
                run_cfg["run"]["run_index"] = run_idx
                label = f"{mode}_{bm}_run{run_idx}"
                config_path = config_dir / f"{label}.yaml"
                write_config(run_cfg, config_path)
                all_configs.append((str(config_path), label))

    print(f"Generated {len(all_configs)} configs in {config_dir}/")
    for path, label in all_configs:
        print(f"  {label}: {path}")

    # Run experiments
    results = []
    for i, (config_path, label) in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] Starting {label}...")
        result = run_experiment(config_path, label)
        results.append(result)
        if result is None:
            print(f"\nFATAL: {label} failed. Stopping.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ALL {len(results)} RUNS COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
