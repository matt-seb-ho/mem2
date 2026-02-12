from pathlib import Path

from mem2.cli.run import deep_merge, load_yaml
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components


def test_smoke_arc_runs(tmp_path):
    cfg = load_yaml(Path("configs/experiments/smoke_arc.yaml"))
    if "_base_" in cfg:
        base = load_yaml(Path("configs/base.yaml"))
        cfg = deep_merge(base, {k: v for k, v in cfg.items() if k != "_base_"})

    cfg.setdefault("run", {})["output_root"] = str(tmp_path / "runs")
    components = resolve_components(cfg)
    bundle = run_sync(cfg, components)
    assert bundle.summary["total_attempts"] > 0
