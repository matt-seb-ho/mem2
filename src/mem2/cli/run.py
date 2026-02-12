from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_config_recursive(cfg_path: Path, stack: set[Path] | None = None) -> dict[str, Any]:
    stack = stack or set()
    cfg_path = cfg_path.resolve()
    if cfg_path in stack:
        cycle = " -> ".join(str(p) for p in [*stack, cfg_path])
        raise ValueError(f"Cycle detected in _base_ config chain: {cycle}")
    stack.add(cfg_path)

    cfg = load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping at {cfg_path}")

    base_ref = cfg.get("_base_")
    if not base_ref:
        stack.remove(cfg_path)
        return cfg

    base_path = Path(str(base_ref)).expanduser()
    if not base_path.is_absolute():
        base_path = (cfg_path.parent / base_path).resolve()
    base_cfg = _load_config_recursive(base_path, stack=stack)
    merged = deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "_base_"})
    stack.remove(cfg_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mem2 pipeline")
    parser.add_argument("--config", required=True, help="Path to top-level YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config_recursive(cfg_path)

    components = resolve_components(cfg)
    bundle = run_sync(cfg, components)
    print(f"Run complete: {bundle.summary}")


if __name__ == "__main__":
    main()
