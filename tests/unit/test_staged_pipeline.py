from __future__ import annotations

import copy

from mem2.branches.memory_builder.accretive_prune import AccretivePruneMemoryBuilder
from mem2.branches.memory_builder.adas_style_search import ADASMetaSearchBuilder
from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder
from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder
from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder
from mem2.branches.memory_builder.reorg_evolver import EvolveRDedupBuilder
from mem2.branches.memory_builder.reorg_lilo import LILOLibraryGrowthBuilder
from mem2.branches.memory_builder.reorg_lrll import LRLLWakeSleepBuilder
from mem2.branches.memory_builder.reorg_memp import MempProceduralMemoryBuilder
from mem2.branches.memory_builder.reorg_memtree import MemTreeHierarchicalBuilder
from mem2.branches.memory_builder.reorg_sleepgate import SleepGateForgettingBuilder
from mem2.branches.memory_builder.reorg_stitch import StitchReorgBuilder
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    MemoryState,
    RetrievalBundle,
    RunContext,
)


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=42, config={}, output_dir="/tmp/mem2-unit")


def _memory() -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps",
        schema_version="v1",
        payload={
            "concepts": {
                "concept_a": {
                    "name": "concept_a",
                    "kind": "routine",
                    "description": "A seeded routine.",
                    "cues": ["cue a"],
                    "implementation": ["impl a"],
                    "used_in": ["p1"],
                }
            },
            "solutions": {},
            "custom_types": {},
            "categories": {"routine": ["concept_a"]},
            "reorg": {"history": [], "step": 20, "outcomes": [{"problem_id": "p1", "score": 0.0}]},
            "dreamcoder_reorg": {"history": [], "step": 20, "outcomes": [0.0]},
            "stitch_reorg": {"history": [], "step": 20, "outcomes": [0.0]},
        },
        metadata={"source": "unit"},
    )


def _attempts() -> tuple[list[AttemptRecord], list[EvalRecord]]:
    attempts = [
        AttemptRecord(
            problem_uid="p2",
            pass_idx=0,
            branch_id="unit",
            completion="def solve(grid): return grid",
            prompt="prompt",
        )
    ]
    evals = [
        EvalRecord(
            problem_uid="p2",
            attempt_idx=0,
            is_correct=True,
            train_details=[{"correct": True}],
            test_details=[{"correct": True}],
        )
    ]
    return attempts, evals


def test_memory_state_and_retrieval_bundle_dict_round_trip(tmp_path):
    memory = _memory()
    memory_path = tmp_path / "memory.json"

    memory.to_file(memory_path)

    assert MemoryState.from_file(memory_path) == memory

    bundle = RetrievalBundle(
        problem_uid="p1",
        hint_text="hint",
        retrieved_items=[{"name": "concept_a"}],
        metadata={"selector_mode": "unit"},
    )
    bundle_path = tmp_path / "bundle.json"

    bundle.to_file(bundle_path)

    assert RetrievalBundle.from_file(bundle_path) == bundle


def test_freeze_memory_keeps_update_and_consolidate_noop_for_staged_builders():
    builder_classes = [
        ArcMemoPsMemoryBuilder,
        ArcMemoReorgMemoryBuilder,
        AccretivePruneMemoryBuilder,
        DreamCoderReorgBuilder,
        StitchReorgBuilder,
        LRLLWakeSleepBuilder,
        EvolveRDedupBuilder,
        SleepGateForgettingBuilder,
        MemTreeHierarchicalBuilder,
        MempProceduralMemoryBuilder,
        AMEMAgenticMemoryBuilder,
        LILOLibraryGrowthBuilder,
        ALMAStyleMetaEditMemoryBuilder,
        ADASMetaSearchBuilder,
    ]
    attempts, evals = _attempts()

    for builder_cls in builder_classes:
        builder = builder_cls(freeze_memory=True)
        memory = _memory()
        expected = copy.deepcopy(memory)

        updated = builder.update(
            _ctx(),
            memory,
            attempts=attempts,
            eval_records=evals,
            feedback_records=[],
        )
        consolidated = builder.consolidate(_ctx(), updated)

        assert updated == expected, builder_cls.__name__
        assert consolidated == expected, builder_cls.__name__
