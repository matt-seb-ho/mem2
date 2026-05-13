from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import (
    REPO_ROOT,
    RUNS_ROOT,
    load_yaml,
    slugify,
    utc_run_stamp,
)
from mem2.cli.run import _load_config_recursive
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components


def _axis_paths() -> list[Path]:
    return sorted((REPO_ROOT / "configs" / "axes").glob("*.yaml"))


def find_condition(port: str) -> tuple[Path, dict[str, Any]]:
    for axis_path in _axis_paths():
        data = load_yaml(axis_path)
        for condition in data.get("conditions", []) or []:
            if condition.get("label") == port:
                return axis_path, condition
    known = []
    for axis_path in _axis_paths():
        data = load_yaml(axis_path)
        known.extend(
            str(condition.get("label"))
            for condition in data.get("conditions", []) or []
            if condition.get("label")
        )
    raise ValueError(f"Unknown port '{port}'. Known ports: {', '.join(sorted(known))}")


def _set_component(cfg: dict[str, Any], *, pipeline_key: str, component_key: str, value: str, settings: dict[str, Any]) -> None:
    cfg.setdefault("pipeline", {})[pipeline_key] = value
    cfg.setdefault("components", {})[component_key] = copy.deepcopy(settings or {})


def apply_condition(cfg: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    group = str(condition.get("override_group", ""))
    if group in {"builder", "combo"}:
        _set_component(
            cfg,
            pipeline_key="memory_builder",
            component_key="memory_builder",
            value=str(condition["builder"]),
            settings=condition.get("builder_cfg") or {},
        )
    if group in {"retriever", "combo"}:
        _set_component(
            cfg,
            pipeline_key="memory_retriever",
            component_key="memory_retriever",
            value=str(condition["retriever"]),
            settings=condition.get("retriever_cfg") or {},
        )
    return cfg


def build_case_study_config(
    *,
    port: str,
    n_problems: int,
    seed: int,
    iters: int,
    base_config: Path,
    label: str,
    now: datetime | None = None,
) -> tuple[dict[str, Any], Path]:
    base_config = base_config.expanduser()
    if not base_config.is_absolute():
        base_config = REPO_ROOT / base_config
    base_cfg = _load_config_recursive(base_config)
    axis_path, condition = find_condition(port)
    cfg = apply_condition(base_cfg, condition)

    label_slug = slugify(label)
    run_id = f"{utc_run_stamp(now)}_{port}_n{n_problems}_seed{seed}_{label_slug}"
    trace_dir = RUNS_ROOT / run_id

    run_cfg = cfg.setdefault("run", {})
    run_cfg["run_type"] = "case_study"
    run_cfg["run_id"] = run_id
    run_cfg["seed"] = seed
    run_cfg["max_passes"] = iters
    run_cfg.setdefault("retry_policy", {})["max_passes"] = iters
    cfg.setdefault("components", {}).setdefault("benchmark", {})["limit"] = n_problems
    cfg.setdefault("components", {}).setdefault("provider", {})["trace_dir"] = str(trace_dir)
    cfg["case_studies"] = {
        "trace_dir": str(trace_dir),
        "run_id": run_id,
        "port": port,
        "label": label,
        "axis_config": str(axis_path.relative_to(REPO_ROOT)),
        "condition": copy.deepcopy(condition),
    }
    return cfg, trace_dir


def dry_run_summary(cfg: dict[str, Any], trace_dir: Path) -> dict[str, Any]:
    return {
        "run_id": cfg.get("run", {}).get("run_id"),
        "trace_dir": str(trace_dir),
        "pipeline": cfg.get("pipeline", {}),
        "benchmark_limit": cfg.get("components", {}).get("benchmark", {}).get("limit"),
        "seed": cfg.get("run", {}).get("seed"),
        "iters": cfg.get("run", {}).get("max_passes"),
        "case_studies": cfg.get("case_studies", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trace-persisted mem2 case study")
    parser.add_argument("--port", required=True, help="Axis condition label to run")
    parser.add_argument("--n-problems", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/experiments/phase1_arc_base.yaml"),
    )
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, trace_dir = build_case_study_config(
        port=args.port,
        n_problems=args.n_problems,
        seed=args.seed,
        iters=args.iters,
        base_config=args.base_config,
        label=args.label,
    )
    if args.dry_run:
        print(json.dumps(dry_run_summary(cfg, trace_dir), indent=2, sort_keys=False))
        return

    trace_dir.mkdir(parents=True, exist_ok=True)
    components = resolve_components(cfg)
    bundle = run_sync(cfg, components)
    print(json.dumps(bundle.summary, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
