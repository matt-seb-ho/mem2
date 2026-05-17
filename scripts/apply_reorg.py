#!/usr/bin/env python3
"""Standalone Stage 2 memory-bank reorganization runner."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from mem2.core.entities import MemoryState, RunContext
from mem2.providers.llmplus_client import LLMPlusProviderClient
from mem2.providers.meta_edit_adapter import SyncMetaEditProviderAdapter
from mem2.registry.memory_builder import MEMORY_BUILDERS


_LLM_AWARE_BUILDERS = {
    "reorg_lilo",
    "reorg_amem",
    "reorg_memp",
    "alma_style_metaedit",
    "adas_style_search",
}
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    return _SLUG_RE.sub("_", value.strip()).strip("_") or "variant"


def _parse_json_arg(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("@"):
        text = Path(text[1:]).expanduser().read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON config override must be an object")
    return parsed


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_axis_condition(label: str) -> tuple[str, dict[str, Any]] | None:
    axis_root = _REPO_ROOT / "configs" / "axes"
    for axis_path in sorted(axis_root.glob("*.yaml")):
        data = yaml.safe_load(axis_path.read_text(encoding="utf-8")) or {}
        for condition in data.get("conditions", []) or []:
            if condition.get("label") == label:
                return str(condition["builder"]), dict(condition.get("builder_cfg") or {})
    return None


def _resolve_builder(label: str, override_cfg: dict[str, Any]) -> tuple[str, type, dict[str, Any]]:
    if label in MEMORY_BUILDERS:
        builder_key = label
        base_cfg: dict[str, Any] = {}
    else:
        condition = _load_axis_condition(label)
        if condition is None:
            known = ", ".join(sorted(MEMORY_BUILDERS))
            raise ValueError(f"Unknown memory-builder variant '{label}'. Known builders: {known}")
        builder_key, base_cfg = condition
    if builder_key not in MEMORY_BUILDERS:
        known = ", ".join(sorted(MEMORY_BUILDERS))
        raise ValueError(f"Axis condition '{label}' references unknown builder '{builder_key}'. Known: {known}")
    return builder_key, MEMORY_BUILDERS[builder_key], _deep_update(base_cfg, override_cfg)


def _load_memory_state(path: Path) -> MemoryState:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Input bank must be a JSON object: {path}")
    if "schema_name" in raw and "payload" in raw:
        return MemoryState.from_dict(raw)
    return MemoryState(
        schema_name="arcmemo_ps",
        schema_version="v1",
        payload=raw,
        metadata={"source_file": str(path)},
    )


def _maybe_wire_provider(ctx: RunContext, builder_key: str, args: argparse.Namespace) -> None:
    if args.disable_provider or builder_key not in _LLM_AWARE_BUILDERS:
        return
    provider = LLMPlusProviderClient(
        profile_cfg={
            "profile_name": args.provider,
            "default_max_concurrency": args.concurrency,
            "dotenv_path": str(args.dotenv_path) if args.dotenv_path else None,
        }
    )
    ctx.config["_meta_edit_provider"] = SyncMetaEditProviderAdapter(
        provider,
        model=args.model,
        gen_cfg={
            "n": 1,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "ignore_cache": args.ignore_cache,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply one mem2 memory-builder reorganization to a bank"
    )
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--builder-cfg", default=None, help="JSON object or @path JSON object")
    parser.add_argument("--provider", default="llmplus_openrouter")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--dotenv-path", type=Path, default=Path(".env"))
    parser.add_argument("--disable-provider", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder_key, builder_cls, builder_cfg = _resolve_builder(
        args.variant,
        _parse_json_arg(args.builder_cfg),
    )
    memory = _load_memory_state(args.input_bank)
    ctx = RunContext(
        run_id=f"stage2_{_slug(args.variant)}_seed{args.seed}",
        seed=args.seed,
        config={"stage": "2", "variant": args.variant, "builder": builder_key},
        output_dir=str(args.output_dir),
        tags={"stage": "2"},
    )
    _maybe_wire_provider(ctx, builder_key, args)

    builder = builder_cls(**builder_cfg)
    output = builder.consolidate(ctx, memory)
    output.metadata.setdefault("stage2", {})
    output.metadata["stage2"].update(
        {"variant": args.variant, "builder": builder_key, "seed": args.seed}
    )

    out_path = args.output_dir / f"bank_reorg_{_slug(args.variant)}_seed{args.seed}.json"
    output.to_file(out_path)
    result = {
        "success": True,
        "variant": args.variant,
        "builder": builder_key,
        "seed": args.seed,
        "output": str(out_path),
        "concept_count": len((output.payload.get("concepts") or {})),
    }
    print("APPLY_REORG_RESULT_JSON=" + json.dumps(result, sort_keys=False))


if __name__ == "__main__":
    main()
