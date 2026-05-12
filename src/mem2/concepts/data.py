"""Concept and ParameterSpec dataclasses.

Ported from arc_memo/concept_mem/memory/v4/concept.py with minimal changes:
- Removed arc_memo-specific imports
- kind is plain str (not enum), validated by builders
"""
from __future__ import annotations

import itertools
import logging
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# --------------------------- Utilities --------------------------------- #
_TYPE_DEF_RE = re.compile(r"^\s*([^:=\s]+)\s*:=\s*(.+)$")


def _coerce_line(x) -> str:
    """Collapse a cue/impl list entry into a string. Mirrors the `_merge_lines`
    convention: {k: v} → "k: v", str → str (stripped), anything else → str(x).
    """
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict) and len(x) == 1:
        k, v = next(iter(x.items()))
        return f"{k}: {v}".strip() if isinstance(v, str) else f"{k}: {v}"
    return str(x)


def maybe_parse_typedef(s: str | None) -> tuple[str, str] | None:
    """If `s` matches 'Name := annotation', return (Name, annotation). Else None."""
    if not s:
        return None
    m = _TYPE_DEF_RE.match(s)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


# ------------------------ Data structures ------------------------------ #
@dataclass
class ParameterSpec:
    name: str
    typing: str | None = None
    description: str | None = None


@dataclass
class Concept:
    """A typed concept with rich annotations.

    Fields
    ------
    - name: str
    - kind: str  (e.g. 'structure', 'routine', 'theorem', 'technique', 'algorithm', 'pattern')
    - routine_subtype: str | None
    - output_typing: str | None
    - parameters: list[ParameterSpec]
    - description: str | None
    - cues: list[str]
    - implementation: list[str]
    - used_in: list[str]
    """

    name: str
    kind: str
    routine_subtype: str | None = None
    output_typing: str | None = None
    parameters: list[ParameterSpec] = field(default_factory=list)
    description: str | None = None
    cues: list[str] = field(default_factory=list)
    implementation: list[str] = field(default_factory=list)
    used_in: list[str] = field(default_factory=list)

    # ----------------------- Init / validation ------------------------- #
    def __post_init__(self):
        # Payloads serialized from older runs may store optional list fields
        # as null; coerce to [] so downstream `.extend()` / iteration is safe.
        if self.parameters is None:
            self.parameters = []
        if self.cues is None:
            self.cues = []
        if self.implementation is None:
            self.implementation = []
        if self.used_in is None:
            self.used_in = []
        # cues/implementation may contain dict entries from old serializations
        # (before `_merge_lines` canonicalization). Normalize to strings so
        # downstream dedup / aggregation is hashable-safe.
        self.cues = [_coerce_line(x) for x in self.cues]
        self.implementation = [_coerce_line(x) for x in self.implementation]
        assert isinstance(self.parameters, list)
        fixed_params: list[ParameterSpec] = []
        for p in self.parameters:
            if isinstance(p, dict):
                p = ParameterSpec(**p)
            elif not isinstance(p, ParameterSpec):
                raise TypeError(f"Expected ParameterSpec or dict, got {type(p)}: {p}")
            fixed_params.append(p)
        self.parameters = fixed_params

    # ------------------------ Merge logic ------------------------------ #
    def update(self, problem_id: str, annotation: dict) -> None:
        if problem_id not in self.used_in:
            self.used_in.append(problem_id)

        self.description = self.description or annotation.get("description")
        self.output_typing = self.output_typing or annotation.get("output_typing")

        if self.kind == "routine":
            self.routine_subtype = self.routine_subtype or annotation.get(
                "routine_subtype"
            )

        # Merge parameters
        if "parameters" in annotation and annotation["parameters"]:
            merged = {p.name: p for p in self.parameters}
            for raw in annotation["parameters"]:
                if not isinstance(raw, dict):
                    raw = {"name": str(raw)}
                merged[raw["name"]] = ParameterSpec(
                    name=raw["name"],
                    typing=raw.get("typing"),
                    description=raw.get("description"),
                )
            self.parameters = list(merged.values())

        if "cues" in annotation and annotation["cues"]:
            self.cues = self._merge_lines(self.cues, annotation["cues"])
        if "implementation" in annotation and annotation["implementation"]:
            self.implementation = self._merge_lines(
                self.implementation, annotation["implementation"]
            )

    @staticmethod
    def _merge_lines(curr: list[str], new_lines: list[str]) -> list[str]:
        if isinstance(new_lines, str):
            new_lines = [new_lines]
        cleaned_new_lines = []
        for line in new_lines:
            if isinstance(line, dict):
                if len(line) == 1:
                    k, v = next(iter(line.items()))
                    line = f"{k}: {v}"
                else:
                    logger.info(
                        f"merge list[str] expects a string but received a dict with multiple keys: {line}"
                    )
                    line = str(line)
            if isinstance(line, str):
                cleaned_new_lines.append(line.strip())
            else:
                logger.info(f"merge list[str] expects a string but received: {line}")

        return list(dict.fromkeys(itertools.chain(curr, cleaned_new_lines)))

    # --------------------- Rendering helpers --------------------------- #
    def to_string(
        self,
        *,
        include_description: bool = True,
        indentation: int = 0,
        skip_kind: bool = False,
        skip_routine_subtype: bool = False,
        skip_parameters: bool = False,
        skip_parameter_description: bool = False,
        skip_cues: bool = False,
        skip_implementation: bool = False,
    ) -> str:
        ind = " " * indentation
        lines: list[str] = [f"{ind}- concept: {self.name}"]

        if not skip_kind:
            lines.append(f"{ind}  kind: {self.kind}")

        if self.kind == "routine" and not skip_routine_subtype and self.routine_subtype:
            lines.append(f"{ind}  routine_subtype: {self.routine_subtype}")

        if include_description and self.description:
            lines.append(f"{ind}  description: {self.description}")

        if self.output_typing:
            lines.append(f"{ind}  output_typing: {self.output_typing}")

        if self.parameters and not skip_parameters:
            lines.append(f"{ind}  parameters:")
            for p in self.parameters:
                line = f"{ind}    - {p.name}"
                if p.typing:
                    line += f" (type: {p.typing})"
                if p.description and not skip_parameter_description:
                    line += f": {p.description}"
                lines.append(line)

        if self.cues and not skip_cues:
            lines.append(f"{ind}  cues:")
            for c in self.cues:
                lines.append(f"{ind}    - {c}")

        if self.implementation and not skip_implementation:
            lines.append(f"{ind}  implementation:")
            for note in self.implementation:
                lines.append(f"{ind}    - {note}")

        return "\n".join(lines)

    # -------------------------- Misc ----------------------------------- #
    def asdict(self) -> dict:
        return asdict(self)
