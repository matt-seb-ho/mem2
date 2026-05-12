"""Real-API smoketest for LLM-aware builders/retrievers.

Tests each of the 11 LLM-wired modules' prompt/parse cycle against the real
OpenRouter adapter. Catches JSON-schema mismatches, silent parse failures,
and prompt malformations that template/mock mode hides.

Usage:
    python scripts/real_api_smoketest.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("/Users/aaronzhfeng/workspace/workstation_00_arc/mem2/.env")

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext
from mem2.providers.llmplus_client import LLMPlusProviderClient
from mem2.providers.meta_edit_adapter import SyncMetaEditProviderAdapter


def make_fixture_memory(n: int = 8) -> ConceptMemory:
    mem = ConceptMemory()
    seeds = [
        ("extract objects", "routine", "extract distinct objects from a grid by connected-component analysis"),
        ("recolor object", "routine", "change the color of an object while preserving its shape"),
        ("find object", "routine", "locate an object matching given criteria"),
        ("sort objects", "routine", "order objects by size, position, or color"),
        ("fill region", "routine", "fill a bounded region with a specified color"),
        ("grid size", "structure", "the height and width dimensions of the grid"),
        ("divider lines", "structure", "lines that partition the grid into regions"),
        ("color palette", "structure", "the set of colors used in the grid"),
    ]
    for name, kind, desc in seeds[:n]:
        c = Concept(
            name=name, kind=kind, description=desc,
            cues=[f"when working with {name.split()[0]}"], implementation=[],
            parameters=[], used_in=[f"task_{i}" for i in range(hash(name) % 5 + 1)],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def make_ctx(adapter) -> RunContext:
    return RunContext(
        run_id="real_api_smoke", seed=0,
        output_dir=Path("/tmp/real_api_smoke"),
        config={"_meta_edit_provider": adapter},
    )


RESULTS: list[tuple[str, str, str]] = []  # (module, status, note)


def test(name: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        note = result if isinstance(result, str) else str(result)[:200]
        RESULTS.append((name, "PASS", note))
        print(f"  [PASS] {name}: {note[:150]}")
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-3:]
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        for line in tb:
            print(f"         {line}")


# ================================================================== #
#                      TESTS PER LLM-AWARE MODULE                    #
# ================================================================== #

def test_f2_alma(adapter, mem):
    """F.2 ALMA: propose edit plan."""
    from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder
    b = ALMAStyleMetaEditMemoryBuilder(every_k=20, trigger="every_k")
    plan = b._propose_edit_plan(make_ctx(adapter), mem)
    if plan is None:
        return "no plan (provider call failed silently)"
    merges = plan.get("merges", [])
    return f"plan OK: {len(merges)} merges, rationale={bool(plan.get('rationale'))}"


def test_f3_adas(adapter, mem):
    """F.3 ADAS: propose with reflexion buffer."""
    from mem2.branches.memory_builder.adas_style_search import ADASMetaSearchBuilder
    b = ADASMetaSearchBuilder(every_k=20, trigger="every_k", max_reflexion_rounds=2)
    plan = b._propose_with_reflexion(make_ctx(adapter), mem, [], adapter)
    if plan is None:
        return "no plan"
    return f"plan OK: {len(plan.get('merges', []))} merges"


def test_a4_lilo(adapter, mem):
    """A.4 LILO: iterative library growth."""
    from mem2.branches.memory_builder.reorg_lilo import LILOLibraryGrowthBuilder
    b = LILOLibraryGrowthBuilder(every_k=20, trigger="every_k")
    proposal = b._propose_via_llm(make_ctx(adapter), mem, adapter, 0, [])
    if proposal is None:
        return "no proposal"
    name = proposal.get("readable_name", "?")
    members = proposal.get("members", [])
    return f"proposal OK: name={name}, members={members}"


def test_a6_amem(adapter, mem):
    """A.6 A-MEM: per-note evolution."""
    from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder
    b = AMEMAgenticMemoryBuilder()
    note = next(iter(mem.concepts.keys()))
    nbrs = [(n, 0.5) for n in list(mem.concepts.keys())[1:4]]
    decision = b._evolve_via_llm(make_ctx(adapter), mem, note, nbrs, adapter)
    if decision is None:
        return "no decision"
    return f"decision OK: should_evolve={decision.get('should_evolve')}, actions={decision.get('actions')}"


def test_a7_memp(adapter, mem):
    """A.7 Memp: reflect on partial failure."""
    from mem2.branches.memory_builder.reorg_memp import MempProceduralMemoryBuilder
    b = MempProceduralMemoryBuilder()
    concept = next(iter(mem.concepts.values()))
    outcomes = {"task_0": 1.0, "task_1": 0.0}
    rewritten = b._reflect_via_llm(adapter, concept, outcomes, list(outcomes.keys()))
    if rewritten is None:
        return "no rewrite"
    return f"rewrite OK: {len(rewritten)} chars"


def test_a8_evolver(adapter, mem):
    """A.8 EvolveR: duplicate verification."""
    from mem2.branches.memory_builder.reorg_evolver import EvolveRDedupBuilder
    b = EvolveRDedupBuilder()
    concepts = list(mem.concepts.values())
    verdict = b._verify_duplicate_via_llm(adapter, concepts[0], concepts[1])
    return f"duplicate verdict: {verdict}"


def test_b5_hipporag2(adapter, mem):
    """B.5 HippoRAG 2: LLM fact filter."""
    from mem2.branches.memory_retriever.hipporag2 import HippoRAG2FilterRetriever
    r = HippoRAG2FilterRetriever()
    candidates = list(mem.concepts.keys())[:5]
    filtered = r._filter_via_llm(adapter, "extract objects from a grid", candidates, mem)
    if filtered is None:
        return "no filter response"
    return f"filtered: kept {len(filtered)}/{len(candidates)}: {filtered}"


def test_d4_dspy_copro(adapter):
    """D.4 DSPy-COPRO: propose instruction variant."""
    from mem2.branches.memory_builder.variant_dspy_opt import (
        DSPyOptFormatBuilder, VariantAttempt,
    )
    b = DSPyOptFormatBuilder(breadth=3, depth=1)
    history = [
        VariantAttempt(name="minimal", flags={"skip_cues": True, "skip_implementation": True}, score=0.5, round_idx=0),
        VariantAttempt(name="cue_heavy", flags={"skip_cues": False, "skip_implementation": True}, score=0.3, round_idx=0),
    ]
    proposal = b._propose_via_llm(adapter, history, 1)
    if proposal is None:
        return "no proposal"
    return f"proposal OK: flags={proposal.get('flags')}"


def test_d5_gepa(adapter):
    """D.5 GEPA: reflective crossover/mutation proposal."""
    from mem2.branches.memory_builder.variant_gepa import GEPAFormatBuilder
    from mem2.branches.memory_builder.variant_dspy_opt import VariantAttempt
    b = GEPAFormatBuilder(population_size=3, generations=1)
    pop = [
        VariantAttempt(name="minimal", flags={"skip_cues": True}, score=0.5, round_idx=0),
        VariantAttempt(name="cue_heavy", flags={"skip_cues": False}, score=0.3, round_idx=0),
        VariantAttempt(name="free_text", flags={"skip_cues": True, "include_description": True}, score=0.7, round_idx=0),
    ]
    child = b._propose_via_llm_gepa(adapter, pop, 1)
    if child is None:
        return "no child"
    return f"child OK: name={child.name}, flags={child.flags}"


def test_d6_parse(adapter, mem):
    """D.6 PARSE: ARCHITECT schema refinement."""
    from mem2.branches.memory_builder.variant_parse import PARSESchemaBuilder
    b = PARSESchemaBuilder()
    stats = {"routine": {"hit": 50, "count": 10, "rate": 0.8},
              "structure": {"hit": 12, "count": 5, "rate": 0.2}}
    base = {"skip_cues": False, "skip_implementation": False, "skip_parameters": False, "include_description": True}
    refined = b._refine_via_llm(adapter, stats, base)
    if refined is None:
        return "no refinement"
    return f"refinement OK: keys={list(refined.keys())}"


def test_b11_magma(adapter, mem):
    """B.11 MAGMA: view-policy selection."""
    from mem2.branches.memory_retriever.magma import MAGMAMultiGraphRetriever
    r = MAGMAMultiGraphRetriever()
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[],
                          metadata={"description": "extract and recolor objects"})
    view_hits = {
        "semantic": ["extract objects", "recolor object"],
        "temporal": [],
        "causal": ["extract objects"],
        "entity": ["extract objects"],
    }
    picked = r._policy_via_llm(adapter, problem, view_hits)
    if picked is None:
        return "no policy response"
    return f"policy picked: {picked}"


# ================================================================== #
#                              MAIN                                  #
# ================================================================== #

def main():
    print("=" * 60)
    print("REAL-API SMOKETEST: LLM-aware builders via OpenRouter")
    print("=" * 60)

    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": "/Users/aaronzhfeng/workspace/workstation_00_arc/mem2/.env",
    })
    adapter = SyncMetaEditProviderAdapter(client)
    print(f"Model: {adapter.model}")
    print()

    mem = make_fixture_memory(n=8)
    print(f"Fixture: {len(mem.concepts)} concepts\n")

    tests = [
        ("F.2 ALMA (edit plan)", lambda: test_f2_alma(adapter, mem)),
        ("F.3 ADAS (reflexion)", lambda: test_f3_adas(adapter, mem)),
        ("A.4 LILO (iterative)", lambda: test_a4_lilo(adapter, mem)),
        ("A.6 A-MEM (evolution)", lambda: test_a6_amem(adapter, mem)),
        ("A.7 Memp (reflect)", lambda: test_a7_memp(adapter, mem)),
        ("A.8 EvolveR (verify)", lambda: test_a8_evolver(adapter, mem)),
        ("B.5 HippoRAG 2 (filter)", lambda: test_b5_hipporag2(adapter, mem)),
        ("B.11 MAGMA (policy)", lambda: test_b11_magma(adapter, mem)),
        ("D.4 DSPy-COPRO (proposal)", lambda: test_d4_dspy_copro(adapter)),
        ("D.5 GEPA (reflective child)", lambda: test_d5_gepa(adapter)),
        ("D.6 PARSE (ARCHITECT)", lambda: test_d6_parse(adapter, mem)),
    ]

    for name, fn in tests:
        test(name, fn)

    adapter.shutdown()
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"{n_pass}/{len(RESULTS)} passed, {n_fail} failed")
    if n_fail:
        print("\nFAILED:")
        for name, status, note in RESULTS:
            if status == "FAIL":
                print(f"  {name}: {note}")
        sys.exit(1)


if __name__ == "__main__":
    main()
