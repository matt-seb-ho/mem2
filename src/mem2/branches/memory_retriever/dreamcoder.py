"""DreamCoder adapted-memory retriever hook.

This module is intentionally a best-effort bridge for the adapter artifact.
The score-bearing DreamCoder port is a memory builder (`reorg_dreamcoder`);
DreamCoder itself does not expose a native ARC retriever. This hook loads
`ports/dreamcoder_memory_v1.json` and renders fragment-compression cards so
the adapted substrate can be inspected and used as a retrieval surface without
claiming executable DreamCoder frontiers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mem2.concepts.artifacts import CONCEPT_MEMORY_DIR
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "dreamcoder_memory_v1.json"


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _problem_text(problem: ProblemSpec) -> str:
    parts: list[str] = [str(getattr(problem, "uid", ""))]
    for value in (getattr(problem, "metadata", {}) or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


class DreamCoderAdaptedRetriever:
    """Rank DreamCoder fragment cards by query overlap."""

    name = "dreamcoder"
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
        self.adapted_memory_path = self._resolve_path(
            adapted_memory_path,
            _DEFAULT_ADAPTED_MEMORY_PATH,
        )

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
        scored: list[tuple[int, int, str]] = []
        for idx, name in enumerate(mem.concepts):
            if name in records:
                text = self._record_text(records[name])
            else:
                text = mem.concepts[name].to_string(
                    include_description=True,
                    skip_cues=False,
                    skip_implementation=False,
                )
            scored.append((len(q & _tokens(text)) if q else len(mem.concepts[name].used_in or []), -idx, name))
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
                "adapted_cards_rendered": sum(1 for name in selected if name in records),
                "substrate_gap": "best_effort_non_executable_frontier_cards",
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
            raise RuntimeError(f"invalid DreamCoder adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != "dreamcoder":
            raise RuntimeError(f"invalid DreamCoder adapted memory schema: {path}")
        records: dict[str, dict[str, Any]] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if isinstance(concept_id, str) and concept_id in mem.concepts:
                records[concept_id] = raw
        return (records, "dreamcoder_memory_v1") if records else ({}, "flat")

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        parts = [
            str(record.get("frontier_signature") or ""),
            str(record.get("mdl_notes") or ""),
            " ".join(str(t) for t in record.get("fragment_terms") or []),
        ]
        primitive = record.get("invented_primitive_candidate") or {}
        if isinstance(primitive, dict):
            parts.extend(str(primitive.get(k) or "") for k in (
                "name_hint", "typed_output", "reusable_behavior",
            ))
            parts.extend(str(t) for t in primitive.get("typed_inputs") or [])
        for role in record.get("compression_roles") or []:
            if isinstance(role, dict):
                parts.append(f"{role.get('role', '')} {role.get('text', '')}")
        return "\n".join(part for part in parts if part.strip())

    @classmethod
    def _render_adapted_hint(
        cls,
        selected: list[str],
        records: dict[str, dict[str, Any]],
    ) -> str:
        blocks: list[str] = []
        for name in selected:
            record = records.get(name)
            if not record:
                continue
            primitive = record.get("invented_primitive_candidate") or {}
            lines = [f"- concept: {name}"]
            lines.append(f"  dreamcoder_frontier_signature: {record.get('frontier_signature', '')}")
            if isinstance(primitive, dict):
                lines.append(
                    "  invented_primitive: "
                    f"{primitive.get('name_hint', '')} -> {primitive.get('reusable_behavior', '')}"
                )
            roles = [
                f"{role.get('role', '')}: {role.get('text', '')}"
                for role in record.get("compression_roles") or []
                if isinstance(role, dict)
            ]
            if roles:
                lines.append("  compression_roles: " + "; ".join(roles[:5]))
            terms = [str(t).strip() for t in record.get("fragment_terms") or [] if str(t).strip()]
            if terms:
                lines.append("  fragment_terms: " + ", ".join(terms[:8]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
