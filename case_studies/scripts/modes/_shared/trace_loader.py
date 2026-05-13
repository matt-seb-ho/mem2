from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import RUNS_ROOT, iter_trace_dirs, load_json, relative_link


@dataclass(frozen=True)
class TraceRecord:
    run_dir: Path
    trace_dir: Path
    task_id: str
    iter_name: str
    prompt: str
    response: str
    retrieval_bundle: dict[str, Any]
    parsed: dict[str, Any]
    eval_data: dict[str, Any]
    call_meta: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.task_id}/{self.iter_name}"

    @property
    def correct(self) -> bool:
        return bool(self.eval_data.get("correct", False))

    @property
    def trace_link(self) -> str:
        return relative_link(self.trace_dir, self.run_dir)


@dataclass(frozen=True)
class CaseRun:
    run_dir: Path
    meta: dict[str, Any]
    traces: list[TraceRecord]

    @property
    def run_id(self) -> str:
        return str(self.meta.get("run_id") or self.run_dir.name)

    @property
    def port(self) -> str:
        return str(self.meta.get("port") or "unknown")

    @property
    def label(self) -> str:
        return str(self.meta.get("label") or "unlabeled")

    def trace_map(self) -> dict[str, TraceRecord]:
        return {trace.key: trace for trace in self.traces}


def resolve_run_dir(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    candidate = RUNS_ROOT / str(value)
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Run not found: {value}")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_case_run(value: str | Path) -> CaseRun:
    run_dir = resolve_run_dir(value)
    traces: list[TraceRecord] = []
    for trace_dir in iter_trace_dirs(run_dir):
        traces.append(
            TraceRecord(
                run_dir=run_dir,
                trace_dir=trace_dir,
                task_id=trace_dir.parent.name,
                iter_name=trace_dir.name,
                prompt=_read_text(trace_dir / "prompt.txt"),
                response=_read_text(trace_dir / "response.txt"),
                retrieval_bundle=load_json(trace_dir / "retrieval_bundle.json", default={}) or {},
                parsed=load_json(trace_dir / "parsed.json", default={}) or {},
                eval_data=load_json(trace_dir / "eval.json", default={}) or {},
                call_meta=load_json(trace_dir / "call_meta.json", default={}) or {},
            )
        )
    return CaseRun(run_dir=run_dir, meta=load_json(run_dir / "meta.json", default={}) or {}, traces=traces)


def metadata_summary(bundle: dict[str, Any], *, limit: int = 6) -> str:
    metadata = bundle.get("metadata") or {}
    parts: list[str] = []
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                parts.append(f"{key}={value}")
            if len(parts) >= limit:
                break
    return ", ".join(parts) if parts else "none"


def response_preview(text: str, *, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def extraction_preview(data: Any, *, limit: int = 300) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=True)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def retrieval_items(bundle: dict[str, Any], *, limit: int = 6) -> list[str]:
    for key in ("items", "retrieved", "concepts", "entries", "documents", "matches"):
        value = bundle.get(key)
        if isinstance(value, list):
            return [_item_preview(item) for item in value[:limit]]
    return []


def _item_preview(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("id", "concept_id", "title", "name", "label"):
            if item.get(key):
                return f"{key}={item[key]}"
        for key in ("text", "content", "summary"):
            if item.get(key):
                return response_preview(str(item[key]), limit=120)
        return extraction_preview(item, limit=120)
    return response_preview(str(item), limit=120)


def write_analysis_report(run_dir: Path, mode_name: str, markdown: str) -> Path:
    out_path = run_dir / "analyses" / f"{mode_name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return out_path


def append_summary_link(run_dir: Path, mode_name: str, report_path: Path, title: str) -> None:
    summary_path = run_dir / "summary.md"
    entry = f"- [{title}]({relative_link(report_path, run_dir)})"
    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8")
    else:
        text = "# Case Study Summary\n"

    if entry in text:
        return

    section = "## Mode analyses"
    if section not in text:
        text = text.rstrip() + f"\n\n{section}\n"
    text = text.rstrip() + f"\n{entry}\n"
    summary_path.write_text(text, encoding="utf-8")
