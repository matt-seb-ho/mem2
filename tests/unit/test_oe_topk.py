from __future__ import annotations

from mem2.branches.memory_retriever.oe_topk import OeTopKRetriever
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="test", config={}, seed=0, output_dir="/tmp")


def _problem() -> ProblemSpec:
    return ProblemSpec(uid="p1", train_pairs=[], test_pairs=[])


def test_oe_topk_first_call_is_empty_without_seeded_history():
    retriever = OeTopKRetriever(top_k=3)
    memory = MemoryState(
        schema_name="arcmemo_oe",
        schema_version="v1",
        payload={"entries": []},
    )

    bundle = retriever.retrieve(_ctx(), memory, _problem(), previous_attempts=[])

    assert bundle.hint_text is None
    assert bundle.retrieved_items == []
    assert bundle.metadata["scoring_mode"] == "open_ended_topk"
    assert bundle.metadata["history_attempts"] == 0
    assert bundle.metadata["scoped_to_problem"] is False
