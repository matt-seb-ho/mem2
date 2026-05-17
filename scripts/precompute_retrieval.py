#!/usr/bin/env python3
"""Standalone Stage 2.5 retrieval-bundle pre-renderer."""
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

from mem2.core.entities import MemoryState, ProblemSpec, RunContext
from mem2.registry.memory_retriever import MEMORY_RETRIEVERS


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def _load_axis_retriever(label: str) -> tuple[str, dict[str, Any]] | None:
    axis_root = _REPO_ROOT / "configs" / "axes"
    for axis_path in sorted(axis_root.glob("*.yaml")):
        data = yaml.safe_load(axis_path.read_text(encoding="utf-8")) or {}
        for condition in data.get("conditions", []) or []:
            if condition.get("label") == label and condition.get("retriever"):
                return str(condition["retriever"]), dict(condition.get("retriever_cfg") or {})
    return None


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_retriever(label: str, override_cfg: dict[str, Any]) -> tuple[str, type, dict[str, Any]]:
    if label in MEMORY_RETRIEVERS:
        retriever_key = label
        base_cfg: dict[str, Any] = {}
    else:
        condition = _load_axis_retriever(label)
        if condition is None:
            known = ", ".join(sorted(MEMORY_RETRIEVERS))
            raise ValueError(f"Unknown retriever '{label}'. Known retrievers: {known}")
        retriever_key, base_cfg = condition
    if retriever_key not in MEMORY_RETRIEVERS:
        known = ", ".join(sorted(MEMORY_RETRIEVERS))
        raise ValueError(f"Axis condition '{label}' references unknown retriever '{retriever_key}'. Known: {known}")
    return retriever_key, MEMORY_RETRIEVERS[retriever_key], _deep_update(base_cfg, override_cfg)


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


def _problem_from_raw(uid: str, raw: Any) -> ProblemSpec:
    if isinstance(raw, str):
        return ProblemSpec(uid=uid, train_pairs=[], test_pairs=[], metadata={"problem_text": raw})
    if not isinstance(raw, dict):
        return ProblemSpec(uid=uid, train_pairs=[], test_pairs=[], metadata={"raw": raw})
    metadata = dict(raw.get("metadata") or {})
    for key in ["puzzle_text", "problem_text", "text"]:
        if key in raw and key not in metadata:
            metadata["problem_text"] = raw[key]
            break
    return ProblemSpec(
        uid=str(raw.get("uid") or uid),
        train_pairs=list(raw.get("train_pairs") or raw.get("train") or []),
        test_pairs=list(raw.get("test_pairs") or raw.get("test") or []),
        metadata=metadata,
    )


def _load_problems(path: Path) -> dict[str, ProblemSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    problems: dict[str, ProblemSpec] = {}
    if isinstance(raw, dict):
        for uid, item in raw.items():
            problem = _problem_from_raw(str(uid), item)
            problems[problem.uid] = problem
        return problems
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            uid = str(item.get("uid") if isinstance(item, dict) else idx)
            problem = _problem_from_raw(uid, item)
            problems[problem.uid] = problem
        return problems
    raise ValueError(f"Problems JSON must contain an object or list: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute mem2 retrieval bundles for a problem list"
    )
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-cfg", default=None, help="JSON object or @path JSON object")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever_cfg = _parse_json_arg(args.retriever_cfg)
    retriever_cfg.setdefault("top_k", args.top_k)
    retriever_key, retriever_cls, retriever_cfg = _resolve_retriever(args.retriever, retriever_cfg)
    memory = _load_memory_state(args.input_bank)
    problems = _load_problems(args.problems)
    ctx = RunContext(
        run_id=f"stage25_{_SLUG_RE.sub('_', args.retriever)}_seed{args.seed}",
        seed=args.seed,
        config={"stage": "2.5", "retriever": retriever_key},
        output_dir=str(args.output.parent),
        tags={"stage": "2.5"},
    )
    retriever = retriever_cls(**retriever_cfg)

    bundles: dict[str, dict[str, Any]] = {}
    for uid, problem in problems.items():
        bundle = retriever.retrieve(ctx, memory, problem, previous_attempts=[])
        bundles[uid] = bundle.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundles, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    non_null_hints = sum(1 for bundle in bundles.values() if bundle.get("hint_text"))
    result = {
        "success": True,
        "retriever": args.retriever,
        "resolved_retriever": retriever_key,
        "seed": args.seed,
        "output": str(args.output),
        "problem_count": len(bundles),
        "non_null_hints": non_null_hints,
    }
    print("PRECOMPUTE_RETRIEVAL_RESULT_JSON=" + json.dumps(result, sort_keys=False))


if __name__ == "__main__":
    main()
