"""Tests for the concepts core: Concept, ParameterSpec, ConceptMemory, DomainProfile."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mem2.concepts.data import Concept, ParameterSpec, maybe_parse_typedef
from mem2.concepts.domain import DomainProfile
from mem2.concepts.memory import ConceptMemory, ProblemSolution


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------
class TestParameterSpec:
    def test_basic(self):
        p = ParameterSpec(name="color", typing="Color", description="fill color")
        assert p.name == "color"
        assert p.typing == "Color"
        assert p.description == "fill color"

    def test_defaults(self):
        p = ParameterSpec(name="x")
        assert p.typing is None
        assert p.description is None


# ---------------------------------------------------------------------------
# maybe_parse_typedef
# ---------------------------------------------------------------------------
class TestMaybeParseTypedef:
    def test_valid(self):
        assert maybe_parse_typedef("Foo := list[int]") == ("Foo", "list[int]")

    def test_extra_spaces(self):
        result = maybe_parse_typedef("  Bar  :=  dict[str, int]  ")
        assert result is not None
        assert result[0] == "Bar"
        assert result[1] == "dict[str, int]"

    def test_none(self):
        assert maybe_parse_typedef(None) is None

    def test_no_match(self):
        assert maybe_parse_typedef("just a string") is None


# ---------------------------------------------------------------------------
# Concept
# ---------------------------------------------------------------------------
class TestConcept:
    def test_creation(self):
        c = Concept(
            name="tiling",
            kind="routine",
            routine_subtype="grid manipulation",
            output_typing="Grid",
            parameters=[ParameterSpec(name="pattern")],
            description="Tile a pattern across the grid",
            cues=["repeating pattern"],
            implementation=["np.tile(pattern, (n, m))"],
            used_in=["puzzle_001"],
        )
        assert c.name == "tiling"
        assert c.kind == "routine"
        assert len(c.parameters) == 1
        assert c.parameters[0].name == "pattern"

    def test_post_init_converts_dicts(self):
        c = Concept(
            name="test",
            kind="structure",
            parameters=[{"name": "x", "typing": "int"}],
        )
        assert isinstance(c.parameters[0], ParameterSpec)
        assert c.parameters[0].typing == "int"

    def test_update_merges(self):
        c = Concept(name="fill", kind="routine", used_in=["p1"])
        c.update("p2", {
            "description": "Fill a region",
            "output_typing": "Grid",
            "routine_subtype": "grid manipulation",
            "cues": ["solid region"],
            "implementation": ["np.fill"],
            "parameters": [{"name": "color", "typing": "Color"}],
        })
        assert "p2" in c.used_in
        assert c.description == "Fill a region"
        assert c.output_typing == "Grid"
        assert c.routine_subtype == "grid manipulation"
        assert len(c.cues) == 1
        assert len(c.parameters) == 1

    def test_update_deduplicates(self):
        c = Concept(name="fill", kind="routine", cues=["solid region"])
        c.update("p1", {"cues": ["solid region", "new cue"]})
        assert c.cues == ["solid region", "new cue"]

    def test_update_keeps_first_nonnull(self):
        c = Concept(name="fill", kind="routine", description="orig")
        c.update("p1", {"description": "new"})
        assert c.description == "orig"  # keeps first

    def test_merge_lines_handles_dicts(self):
        result = Concept._merge_lines(["a"], [{"key": "val"}])
        assert "key: val" in result

    def test_to_string(self):
        c = Concept(
            name="tiling",
            kind="routine",
            routine_subtype="grid manipulation",
            output_typing="Grid",
            parameters=[ParameterSpec(name="pattern", typing="Grid", description="the tile")],
            cues=["repeated pattern"],
            implementation=["np.tile(...)"],
        )
        s = c.to_string()
        assert "tiling" in s
        assert "routine" in s
        assert "grid manipulation" in s
        assert "pattern" in s
        assert "repeated pattern" in s
        assert "np.tile" in s

    def test_to_string_skip_flags(self):
        c = Concept(
            name="test",
            kind="structure",
            cues=["cue1"],
            implementation=["impl1"],
            parameters=[ParameterSpec(name="x")],
        )
        s = c.to_string(skip_kind=True, skip_cues=True, skip_implementation=True, skip_parameters=True)
        assert "kind" not in s
        assert "cue1" not in s
        assert "impl1" not in s

    def test_asdict(self):
        c = Concept(name="test", kind="structure")
        d = c.asdict()
        assert d["name"] == "test"
        assert d["kind"] == "structure"


# ---------------------------------------------------------------------------
# ConceptMemory
# ---------------------------------------------------------------------------
class TestConceptMemory:
    def _sample_annotations(self):
        return {
            "puzzle_001": {
                "summary": "Transform by tiling",
                "pseudocode": "tile the pattern",
                "concepts": [
                    {
                        "concept": "tiling",
                        "kind": "routine",
                        "routine_subtype": "grid manipulation",
                        "output_typing": "Grid",
                        "parameters": [{"name": "pattern", "typing": "Grid"}],
                        "cues": ["repeating pattern"],
                        "implementation": ["np.tile(...)"],
                    },
                    {
                        "concept": "color_region",
                        "kind": "structure",
                        "description": "A connected region of same color",
                        "cues": ["uniform color block"],
                        "implementation": ["flood fill"],
                        "parameters": [],
                    },
                ],
            },
            "puzzle_002": {
                "summary": "Fill regions",
                "concepts": [
                    {
                        "concept": "tiling",
                        "kind": "routine",
                        "routine_subtype": "grid manipulation",
                        "cues": ["symmetry"],
                        "implementation": ["np.tile(...)"],
                        "parameters": [{"name": "count", "typing": "int"}],
                    },
                ],
            },
        }

    def test_initialize_from_annotations(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(self._sample_annotations())
        assert "tiling" in mem.concepts
        assert "color_region" in mem.concepts
        assert len(mem.concepts["tiling"].used_in) == 2
        assert "puzzle_001" in mem.solutions
        assert "puzzle_002" in mem.solutions

    def test_write_concept_and_update(self):
        mem = ConceptMemory()
        mem.write_concept("p1", {
            "concept": "test",
            "kind": "structure",
            "cues": ["cue1"],
        })
        assert "test" in mem.concepts
        mem.write_concept("p2", {
            "concept": "test",
            "kind": "structure",
            "cues": ["cue2"],
        })
        assert len(mem.concepts["test"].used_in) == 2
        assert "cue1" in mem.concepts["test"].cues
        assert "cue2" in mem.concepts["test"].cues

    def test_write_concept_invalid_kind(self):
        mem = ConceptMemory()
        mem.write_concept("p1", {"concept": "bad", "kind": "invalid"})
        assert "bad" not in mem.concepts

    def test_write_solution(self):
        mem = ConceptMemory()
        mem.write_solution("p1", "def solve(): pass", {"summary": "solves it"})
        assert "p1" in mem.solutions
        assert mem.solutions["p1"].summary == "solves it"

    def test_to_string_basic(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(self._sample_annotations())
        s = mem.to_string()
        assert "tiling" in s
        assert "color_region" in s

    def test_to_string_with_concept_names(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(self._sample_annotations())
        s = mem.to_string(concept_names=["tiling"], show_other_concepts=True)
        assert "tiling" in s
        assert "other concepts" in s

    def test_to_string_usage_threshold(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(self._sample_annotations())
        # color_region used only in 1 puzzle, threshold=2 means it goes to low usage
        s = mem.to_string(usage_threshold=2)
        assert "lower usage concepts" in s

    def test_roundtrip_file(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(self._sample_annotations())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mem.save_to_file(path)

            mem2 = ConceptMemory()
            mem2.load_from_file(path)

            assert set(mem.concepts.keys()) == set(mem2.concepts.keys())
            assert set(mem.solutions.keys()) == set(mem2.solutions.keys())
            assert mem.custom_types == mem2.custom_types
            for name in mem.concepts:
                assert mem.concepts[name].used_in == mem2.concepts[name].used_in

    def test_roundtrip_payload(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(self._sample_annotations())
        payload = mem.to_payload()
        mem2 = ConceptMemory.from_payload(payload)
        assert set(mem.concepts.keys()) == set(mem2.concepts.keys())
        assert set(mem.solutions.keys()) == set(mem2.solutions.keys())

    def test_harvest_typedefs(self):
        mem = ConceptMemory()
        mem.write_concept("p1", {
            "concept": "typed_thing",
            "kind": "routine",
            "output_typing": "MyType := list[int]",
            "parameters": [{"name": "x", "typing": "Foo := dict[str, int]"}],
        })
        assert "MyType" in mem.custom_types
        assert "Foo" in mem.custom_types

    def test_to_string_types_section(self):
        mem = ConceptMemory()
        mem.custom_types["MyType"] = "list[int]"
        s = mem.to_string()
        assert "MyType := list[int]" in s

    def test_other_routines_grouped(self):
        mem = ConceptMemory()
        mem.write_concept("p1", {
            "concept": "helper",
            "kind": "routine",
            "routine_subtype": "logic",
            "cues": ["c1"],
            "implementation": ["i1"],
        })
        mem.write_concept("p2", {
            "concept": "helper",
            "kind": "routine",
            "cues": ["c2"],
            "implementation": ["i2"],
        })
        s = mem.to_string(usage_threshold=0)
        assert "other routines" in s
        assert "logic" in s


# ---------------------------------------------------------------------------
# DomainProfile
# ---------------------------------------------------------------------------
class TestDomainProfile:
    def test_arc_profile(self):
        p = DomainProfile.arc_profile()
        assert "structure" in p.valid_kinds
        assert "routine" in p.valid_kinds
        assert len(p.section_order) == 3

    def test_math_profile(self):
        p = DomainProfile.math_profile()
        assert "theorem" in p.valid_kinds
        assert "technique" in p.valid_kinds
        assert "definition" in p.valid_kinds

    def test_code_profile(self):
        p = DomainProfile.code_profile()
        assert "algorithm" in p.valid_kinds
        assert "pattern" in p.valid_kinds
        assert "data_structure" in p.valid_kinds

    def test_profile_rendering(self):
        mem = ConceptMemory()
        # Add concepts with math kinds
        mem.concepts["pythagorean"] = Concept(
            name="pythagorean",
            kind="theorem",
            cues=["right triangle"],
            used_in=["p1", "p2"],
        )
        mem.categories["theorem"].append("pythagorean")

        profile = DomainProfile.math_profile()
        s = mem.to_string(profile=profile, usage_threshold=0)
        assert "pythagorean" in s
        assert "theorem" in s.lower()
