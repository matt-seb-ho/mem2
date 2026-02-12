from __future__ import annotations

import argparse
from pathlib import Path

from mem2.cli.run import deep_merge, load_yaml
from mem2.io.hashing import stable_hash
from mem2.io.json_io import write_json
from mem2.prompting import render as prompt_render


def _load_cfg(path: Path) -> dict:
    cfg = load_yaml(path)
    if "_base_" in cfg:
        base_cfg = load_yaml((path.parent / cfg["_base_"]).resolve())
        cfg = deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


def _prompt_template_hashes() -> dict[str, str]:
    values = {
        "ARC_INTRO": prompt_render.ARC_INTRO,
        "EXAMPLE_GRIDS_INTRO": prompt_render.EXAMPLE_GRIDS_INTRO,
        "CODE_INSTR_DEFAULT": prompt_render.CODE_INSTR_DEFAULT,
        "RETRY_INSTRUCTION": prompt_render.RETRY_INSTRUCTION,
    }
    return {k: stable_hash(v) for k, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze ARC default parity baseline")
    parser.add_argument(
        "--strict-config",
        default="configs/experiments/arcmemo_arc_strict.yaml",
        help="Strict parity config",
    )
    parser.add_argument(
        "--parity-spec",
        default="configs/parity/arcmemo_default_spec.yaml",
        help="Machine-readable parity spec",
    )
    args = parser.parse_args()

    strict_cfg = _load_cfg(Path(args.strict_config))
    spec = load_yaml(Path(args.parity_spec))
    baseline = {
        "strict_config_path": args.strict_config,
        "strict_config_hash": stable_hash(strict_cfg),
        "parity_spec_path": args.parity_spec,
        "parity_spec_hash": stable_hash(spec),
        "prompt_template_hashes": _prompt_template_hashes(),
    }

    write_json("outputs/parity/parity_baseline_frozen.json", baseline)
    write_json("configs/parity/parity_baseline_frozen.json", baseline)
    print("baseline frozen at outputs/parity/parity_baseline_frozen.json")


if __name__ == "__main__":
    main()

