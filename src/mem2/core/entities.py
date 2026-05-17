from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunContext:
    run_id: str
    seed: int
    config: dict[str, Any]
    output_dir: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProblemSpec:
    uid: str
    train_pairs: list[dict[str, Any]]
    test_pairs: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSpec:
    task_name: str
    task_description: str
    sample_format: dict[str, Any]
    feedback_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryState:
    schema_name: str
    schema_version: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MemoryState":
        return cls(
            schema_name=str(raw.get("schema_name", "")),
            schema_version=str(raw.get("schema_version", "")),
            payload=dict(raw.get("payload") or {}),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_file(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return out

    @classmethod
    def from_file(cls, path: str | Path) -> "MemoryState":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"MemoryState file must contain a JSON object: {path}")
        return cls.from_dict(raw)


@dataclass(slots=True)
class RetrievalBundle:
    problem_uid: str
    hint_text: str | None
    retrieved_items: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RetrievalBundle":
        return cls(
            problem_uid=str(raw.get("problem_uid", "")),
            hint_text=raw.get("hint_text"),
            retrieved_items=list(raw.get("retrieved_items") or []),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_file(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return out

    @classmethod
    def from_file(cls, path: str | Path) -> "RetrievalBundle":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"RetrievalBundle file must contain a JSON object: {path}")
        return cls.from_dict(raw)


@dataclass(slots=True)
class TrajectoryPlan:
    num_paths: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptRecord:
    """Single model attempt persisted to attempts.jsonl.

    retrieval_metadata stores one snapshot per retrieval call that contributed
    to this attempt. Each entry starts from RetrievalBundle.metadata and adds
    retrieved_items plus hint_present for audit attribution.
    """

    problem_uid: str
    pass_idx: int
    branch_id: str
    completion: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackRecord:
    problem_uid: str
    attempt_idx: int
    feedback_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRecord:
    problem_uid: str
    attempt_idx: int
    is_correct: bool
    train_details: list[dict[str, Any]]
    test_details: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventRecord:
    ts_utc: str
    stage: str
    component: str
    level: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunBundle:
    task_spec: TaskSpec
    problems: dict[str, ProblemSpec]
    attempts: list[AttemptRecord]
    eval_records: list[EvalRecord]
    feedback_records: list[FeedbackRecord]
    memory_state: MemoryState
    summary: dict[str, Any]
    events: list[EventRecord]


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_primitive(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_primitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_primitive(v) for v in value]
    if isinstance(value, tuple):
        return [to_primitive(v) for v in value]
    # Handle non-native numeric types (e.g. sympy.Integer, numpy int64)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if hasattr(value, "__int__"):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return value
