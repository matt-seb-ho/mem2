"""A-Mem adapted-memory retriever hook."""
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
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "amem_memory_v1.json"


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _problem_text(problem: ProblemSpec) -> str:
    parts = [str(getattr(problem, "uid", ""))]
    for value in (getattr(problem, "metadata", {}) or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


class AMEMAdaptedRetriever:
    """Rank A-Mem Zettelkasten notes by query overlap and linked-note text."""

    name = "amem"
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
            link_bonus = self._link_bonus(records.get(name), q)
            score = (len(q & _tokens(text)) if q else len(mem.concepts[name].used_in or [])) + link_bonus
            scored.append((score, -idx, name))
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
                "adapted_notes_rendered": sum(1 for name in selected if name in records),
                "zettel_links_rendered": sum(
                    len(records.get(name, {}).get("zettel_links") or [])
                    for name in selected
                ),
                "substrate": "zettelkasten_note_v1",
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
            raise RuntimeError(f"invalid A-Mem adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != "amem":
            raise RuntimeError(f"invalid A-Mem adapted memory schema: {path}")
        records = {
            raw["concept_id"]: raw
            for raw in data.get("adapted_concepts") or []
            if isinstance(raw, dict) and isinstance(raw.get("concept_id"), str) and raw["concept_id"] in mem.concepts
        }
        for name, record in records.items():
            note = record.get("note")
            if not isinstance(note, dict) or not str(note.get("contextual_description") or "").strip():
                raise RuntimeError(f"adapted memory missing note.contextual_description for {name}")
            if not record.get("zettel_links"):
                raise RuntimeError(f"adapted memory missing zettel_links for {name}")
        return (records, "amem_memory_v1") if records else ({}, "flat")

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        note = record.get("note") or {}
        evolution = record.get("memory_evolution") or {}
        parts = [
            str(note.get("content") or ""),
            str(note.get("contextual_description") or ""),
            " ".join(str(t) for t in note.get("keywords") or []),
            " ".join(str(t) for t in note.get("tags") or []),
            " ".join(
                " ".join(str(link.get(k) or "") for k in ("target_concept", "link_type", "rationale"))
                for link in record.get("zettel_links") or []
                if isinstance(link, dict)
            ),
            str(evolution.get("context_update") or ""),
            " ".join(str(t) for t in evolution.get("tag_updates") or []),
            str(record.get("retrieval_text") or ""),
        ]
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _link_bonus(record: dict[str, Any] | None, q_tokens: set[str]) -> float:
        if not record or not q_tokens:
            return 0.0
        score = 0.0
        for link in record.get("zettel_links") or []:
            if not isinstance(link, dict):
                continue
            text = " ".join(str(link.get(k) or "") for k in ("target_concept", "link_type", "rationale"))
            overlap = len(q_tokens & _tokens(text))
            score += overlap * float(link.get("confidence") or 0.0)
        return score

    @staticmethod
    def _render_adapted_hint(selected: list[str], records: dict[str, dict[str, Any]]) -> str:
        blocks: list[str] = []
        for name in selected:
            record = records.get(name)
            if not record:
                continue
            note = record.get("note") or {}
            lines = [
                f"- concept: {name}",
                f"  amem_note: {note.get('content', '')}",
                f"  contextual_description: {note.get('contextual_description', '')}",
            ]
            keywords = [str(t).strip() for t in note.get("keywords") or [] if str(t).strip()]
            if keywords:
                lines.append("  keywords: " + ", ".join(keywords[:8]))
            tags = [str(t).strip() for t in note.get("tags") or [] if str(t).strip()]
            if tags:
                lines.append("  tags: " + ", ".join(tags[:8]))
            links = []
            for link in record.get("zettel_links") or []:
                if not isinstance(link, dict):
                    continue
                target = str(link.get("target_concept") or "").strip()
                link_type = str(link.get("link_type") or "").strip()
                rationale = str(link.get("rationale") or "").strip()
                if target and link_type:
                    links.append(f"{link_type} {target}: {rationale}")
            if links:
                lines.append("  zettel_links: " + " | ".join(links[:5]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
