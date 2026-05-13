"""MemTree adapted-memory retriever hook."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mem2.concepts.artifacts import CONCEPT_MEMORY_DIR
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import AttemptRecord, MemoryState, ProblemSpec, RetrievalBundle, RunContext


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "memtree_memory_v1.json"


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _problem_text(problem: ProblemSpec) -> str:
    parts = [str(getattr(problem, "uid", ""))]
    for value in (getattr(problem, "metadata", {}) or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


class MemTreeAdaptedRetriever:
    """Collapsed-tree retrieval over MemTree adapted concept nodes."""

    name = "memtree"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 3,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = True,
        usage_threshold: int = 0,
        adapted_memory_path: str | Path | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)
        self.adapted_memory_path = self._resolve_path(adapted_memory_path, _DEFAULT_ADAPTED_MEMORY_PATH)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        mem = ConceptMemory.from_payload(memory.payload)
        if not mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"retriever": self.name, "reason": "empty_memory"},
            )
        records, source = self._load_adapted_records(mem)
        q = _tokens(_problem_text(problem))
        scored = []
        for idx, name in enumerate(mem.concepts):
            text = self._record_text(records[name]) if name in records else mem.concepts[name].to_string()
            score = len(q & _tokens(text)) if q else len(mem.concepts[name].used_in or [])
            depth_bonus = self._depth_bonus(records.get(name))
            scored.append((score + depth_bonus, -idx, name))
        scored.sort(reverse=True)
        selected = [name for _, _, name in scored[: max(self.top_k, 0)]]
        hint = self._render_adapted_hint(selected, records)
        if not hint:
            hint = mem.to_string(
                concept_names=selected,
                include_description=self.include_description,
                skip_cues=self.skip_cues,
                skip_implementation=self.skip_implementation,
                usage_threshold=self.usage_threshold,
            )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": name} for name in selected],
            metadata={
                "retriever": self.name,
                "adapted_memory_source": source,
                "adapted_records_loaded": len(records),
                "adapted_nodes_rendered": sum(1 for name in selected if name in records),
                "tree_paths_rendered": sum(
                    len(records.get(name, {}).get("path_to_root") or [])
                    for name in selected
                ),
                "retrieval_mode": "collapsed_tree",
                "num_selected": len(selected),
            },
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        return self.retrieve(ctx, memory, problem, previous_attempts)

    @staticmethod
    def _resolve_path(path: str | Path | None, default: Path) -> Path:
        if path is None:
            return default
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _load_adapted_records(self, mem: ConceptMemory) -> tuple[dict[str, dict[str, Any]], str]:
        path = self.adapted_memory_path
        if not path.exists():
            return {}, "flat"
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"invalid MemTree adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != "memtree":
            raise RuntimeError(f"invalid MemTree adapted memory schema: {path}")
        records = {
            raw["concept_id"]: raw
            for raw in data.get("adapted_concepts") or []
            if isinstance(raw, dict) and isinstance(raw.get("concept_id"), str) and raw["concept_id"] in mem.concepts
        }
        for name, record in records.items():
            position = record.get("tree_position")
            content = record.get("node_content")
            if not isinstance(position, dict) or not str(position.get("parent_node_id") or "").strip():
                raise RuntimeError(f"adapted memory missing tree_position for {name}")
            if not isinstance(content, dict) or not str(content.get("embedding_text") or "").strip():
                raise RuntimeError(f"adapted memory missing node_content.embedding_text for {name}")
        return (records, "memtree_memory_v1") if records else ({}, "flat")

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        position = record.get("tree_position") or {}
        content = record.get("node_content") or {}
        sibling = record.get("sibling_group") or {}
        parts = [
            str(position.get("leaf_node_id") or ""),
            str(position.get("parent_node_id") or ""),
            str(position.get("insertion_decision") or ""),
            str(position.get("depth_threshold_rationale") or ""),
            str(content.get("leaf_content") or ""),
            str(content.get("embedding_text") or ""),
            str(content.get("aggregate_contribution") or ""),
            str(record.get("collapsed_retrieval_card") or ""),
            " ".join(str(t) for t in record.get("retrieval_keywords") or []),
            str(sibling.get("sibling_role") or ""),
            " ".join(str(t) for t in sibling.get("near_sibling_concepts") or []),
        ]
        for path_item in record.get("path_to_root") or []:
            if isinstance(path_item, dict):
                parts.append(" ".join(str(path_item.get(k) or "") for k in ("node_id", "content_summary", "update_role")))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _depth_bonus(record: dict[str, Any] | None) -> float:
        if not record:
            return 0.0
        position = record.get("tree_position") or {}
        try:
            depth = int(position.get("depth", 0) or 0)
        except (TypeError, ValueError):
            depth = 0
        return min(max(depth, 0), 5) * 0.05

    @staticmethod
    def _render_adapted_hint(selected: list[str], records: dict[str, dict[str, Any]]) -> str:
        blocks: list[str] = []
        for name in selected:
            record = records.get(name)
            if not record:
                continue
            position = record.get("tree_position") or {}
            content = record.get("node_content") or {}
            lines = [
                f"- concept: {name}",
                f"  memtree_leaf: {position.get('leaf_node_id', '')}",
                f"  parent_node: {position.get('parent_node_id', '')}",
                f"  insertion_decision: {position.get('insertion_decision', '')}",
                f"  collapsed_retrieval_card: {record.get('collapsed_retrieval_card', '')}",
                f"  aggregate_contribution: {content.get('aggregate_contribution', '')}",
            ]
            path = []
            for path_item in record.get("path_to_root") or []:
                if isinstance(path_item, dict):
                    node_id = str(path_item.get("node_id") or "").strip()
                    summary = str(path_item.get("content_summary") or "").strip()
                    if node_id:
                        path.append(f"{node_id}: {summary}")
            if path:
                lines.append("  path_to_root: " + " | ".join(path[:5]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
