"""DomainProfile: domain-agnostic rendering configuration for ConceptMemory."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainProfile:
    """Controls how ConceptMemory.to_string() renders concepts per domain.

    Attributes
    ----------
    valid_kinds : set of kind strings accepted by builders for this domain
    section_order : ordered list of kind keys to render
    section_headers : mapping from kind key to markdown header text
    """

    valid_kinds: set[str] = field(default_factory=set)
    section_order: list[str] = field(default_factory=list)
    section_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def arc_profile(cls) -> "DomainProfile":
        return cls(
            valid_kinds={"structure", "routine"},
            section_order=["structure", "types", "routine"],
            section_headers={
                "structure": "## structure concepts",
                "routine": "## routines",
                "types": "## types",
            },
        )

    @classmethod
    def math_profile(cls) -> "DomainProfile":
        return cls(
            valid_kinds={"theorem", "technique", "definition"},
            section_order=["theorem", "technique", "definition"],
            section_headers={
                "theorem": "## theorems",
                "technique": "## techniques",
                "definition": "## definitions",
                "types": "## types",
            },
        )

    @classmethod
    def code_profile(cls) -> "DomainProfile":
        return cls(
            valid_kinds={"algorithm", "pattern", "data_structure"},
            section_order=["algorithm", "pattern", "data_structure"],
            section_headers={
                "algorithm": "## algorithms",
                "pattern": "## patterns",
                "data_structure": "## data structures",
                "types": "## types",
            },
        )
