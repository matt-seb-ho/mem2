from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .entities import RunContext


@dataclass(slots=True)
class RunnerSettings:
    seed: int
    max_passes: int
    output_root: str



def build_run_id(config: dict[str, Any]) -> str:
    cfg_repr = repr(config).encode("utf-8")
    return hashlib.sha1(cfg_repr).hexdigest()[:12]



def build_run_context(config: dict[str, Any]) -> RunContext:
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    run_id = run_cfg.get("run_id") or build_run_id(config)
    output_root = run_cfg.get("output_root", "outputs/_runs")
    run_type = run_cfg.get("run_type", "adhoc")
    out_dir = Path(output_root) / run_type / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = {
        "run_type": str(run_type),
        "benchmark": str(config.get("pipeline", {}).get("benchmark", "unknown")),
    }
    return RunContext(run_id=run_id, seed=seed, config=config, output_dir=str(out_dir), tags=tags)
