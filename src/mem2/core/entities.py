from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
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


@dataclass(slots=True)
class RetrievalBundle:
    problem_uid: str
    hint_text: str | None
    retrieved_items: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryPlan:
    num_paths: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptRecord:
    problem_uid: str
    pass_idx: int
    branch_id: str
    completion: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
