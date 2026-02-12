from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from mem2.core.entities import AttemptRecord, FeedbackRecord
from mem2.io.json_io import write_json
from mem2.prompting.render import make_retry_prompt as make_mem2_retry_prompt


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_meta_item(item: Any) -> tuple[str | None, str | None, str | None, int | None]:
    if isinstance(item, dict):
        return (
            item.get("puzzle_id"),
            item.get("branch_id"),
            item.get("thread_id"),
            item.get("step_idx"),
        )
    if isinstance(item, (list, tuple)):
        if len(item) >= 2:
            puzzle_id = item[0]
            branch_id = item[1]
            thread_id = item[2] if len(item) > 2 else None
            step_idx = item[3] if len(item) > 3 else None
            return puzzle_id, branch_id, thread_id, step_idx
    return None, None, None, None


def _load_prompt_by_metadata(
    prompts_path: Path,
    metadata_path: Path,
    puzzle_id: str,
    branch_id: str,
    thread_id: str | None = None,
    step_idx: int | None = None,
) -> str:
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for i, md in enumerate(metadata):
        pz, br, th, st = _parse_meta_item(md)
        if pz != puzzle_id or br != branch_id:
            continue
        if thread_id is not None and th is not None and str(th) != str(thread_id):
            continue
        if step_idx is not None and st is not None and int(st) != int(step_idx):
            continue
        return str(prompts[i])
    raise ValueError(
        f"Could not locate prompt for metadata puzzle={puzzle_id}, branch={branch_id}, "
        f"thread={thread_id}, step_idx={step_idx} in {metadata_path}"
    )


def _arc_step_to_feedback_metadata(step: Any) -> dict[str, Any]:
    errors: list[str] = []
    mismatches: list[dict[str, Any]] = []

    parsing_error = getattr(step, "parsing_error", None)
    if parsing_error:
        errors.append(str(parsing_error))
        return {"errors": errors, "mismatches": mismatches}

    for res in list(getattr(step, "train_results", [])):
        if bool(getattr(res, "correct", False)):
            continue
        err = getattr(res, "error", None)
        if err:
            errors.append(str(err))
            continue
        mismatches.append(
            {
                "example_idx": int(getattr(res, "pair_idx", -1)) + 1,
                "output": getattr(res, "output", None),
            }
        )
    return {"errors": errors, "mismatches": mismatches}


def _build_mem2_history_from_arc_thread(
    puzzle_id: str,
    branch_id: str,
    thread: Any,
) -> tuple[list[AttemptRecord], list[FeedbackRecord]]:
    attempts: list[AttemptRecord] = []
    feedback: list[FeedbackRecord] = []
    for idx, step in enumerate(list(getattr(thread, "steps", []))):
        attempts.append(
            AttemptRecord(
                problem_uid=puzzle_id,
                pass_idx=int(getattr(step, "step_idx", idx)),
                branch_id=branch_id,
                completion=str(getattr(step, "completion", "") or ""),
                prompt="",
                metadata={},
            )
        )
        feedback.append(
            FeedbackRecord(
                problem_uid=puzzle_id,
                attempt_idx=idx,
                feedback_type="gt",
                content="",
                metadata=_arc_step_to_feedback_metadata(step),
            )
        )
    return attempts, feedback


def _import_arcmemo_helpers(arc_repo: Path):
    if str(arc_repo) not in sys.path:
        sys.path.insert(0, str(arc_repo))
    from concept_mem.evaluation.prompts import make_retry_prompt as make_arc_retry_prompt
    from concept_mem.evaluation.solution_tree import create_solution_tree_from_serialized_dict

    return make_arc_retry_prompt, create_solution_tree_from_serialized_dict


def compare_retry_replay(
    arc_repo: Path,
    arc_run_dir: Path,
    puzzle_id: str | None,
    branch_id: str | None,
    thread_id: str | None,
    num_feedback_passes: int,
    error_feedback: str,
    include_past_outcomes: bool,
    new_concepts: str | None,
) -> dict[str, Any]:
    make_arc_retry_prompt, create_solution_tree_from_serialized_dict = _import_arcmemo_helpers(
        arc_repo
    )

    iter1 = arc_run_dir / "iteration_1"
    iter2 = arc_run_dir / "iteration_2"
    trees_path = iter1 / "solution_trees.json"
    trees_data = json.loads(trees_path.read_text(encoding="utf-8"))
    selected_puzzle_id = puzzle_id or next(iter(trees_data.keys()))
    tree = create_solution_tree_from_serialized_dict(trees_data[selected_puzzle_id])

    selected_branch_id = branch_id or next(iter(tree.prompt_branches.keys()))
    branch = tree.prompt_branches[selected_branch_id]
    selected_thread_id = thread_id or next(iter(branch.threads.keys()))
    thread = branch.threads[selected_thread_id]

    initial_prompt = _load_prompt_by_metadata(
        prompts_path=iter1 / "prompts.json",
        metadata_path=iter1 / "metadata.json",
        puzzle_id=selected_puzzle_id,
        branch_id=selected_branch_id,
    )
    expected_retry_prompt = _load_prompt_by_metadata(
        prompts_path=iter2 / "prompts.json",
        metadata_path=iter2 / "metadata.json",
        puzzle_id=selected_puzzle_id,
        branch_id=selected_branch_id,
        thread_id=selected_thread_id,
        step_idx=len(thread.steps),
    )

    arc_replayed_retry_prompt = make_arc_retry_prompt(
        initial_prompt=initial_prompt,
        solution_thread=thread,
        num_feedback_passes=num_feedback_passes,
        error_feedback=error_feedback,  # type: ignore[arg-type]
        include_past_outcomes=include_past_outcomes,
        new_concepts=new_concepts,
    )

    mem_attempts, mem_feedback = _build_mem2_history_from_arc_thread(
        puzzle_id=selected_puzzle_id,
        branch_id=selected_branch_id,
        thread=thread,
    )
    mem2_replayed_retry_prompt = make_mem2_retry_prompt(
        initial_prompt=initial_prompt,
        attempts=mem_attempts,
        feedback=mem_feedback,
        num_feedback_passes=num_feedback_passes,
        error_feedback=error_feedback,
        include_past_outcomes=include_past_outcomes,
        new_concepts=new_concepts,
    )

    return {
        "arc_repo": str(arc_repo),
        "arc_run_dir": str(arc_run_dir),
        "selector": {
            "puzzle_id": selected_puzzle_id,
            "branch_id": selected_branch_id,
            "thread_id": str(selected_thread_id),
        },
        "policy": {
            "num_feedback_passes": num_feedback_passes,
            "error_feedback": error_feedback,
            "include_past_outcomes": include_past_outcomes,
            "new_concepts": new_concepts,
        },
        "hashes": {
            "initial_prompt": _sha256(initial_prompt),
            "expected_retry_prompt": _sha256(expected_retry_prompt),
            "arc_replayed_retry_prompt": _sha256(arc_replayed_retry_prompt),
            "mem2_replayed_retry_prompt": _sha256(mem2_replayed_retry_prompt),
        },
        "equalities": {
            "arc_replay_matches_expected_retry_prompt": (
                arc_replayed_retry_prompt == expected_retry_prompt
            ),
            "mem2_replay_matches_arc_replay": (
                mem2_replayed_retry_prompt == arc_replayed_retry_prompt
            ),
            "mem2_replay_matches_expected_retry_prompt": (
                mem2_replayed_retry_prompt == expected_retry_prompt
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically replay ArcMemo retry prompt generation and compare to mem2."
    )
    parser.add_argument("--arc-run-dir", required=True, help="Path to arc_memo output run dir")
    parser.add_argument(
        "--arc-repo",
        default="/root/workspace/arc_memo",
        help="Path to arc_memo repository root",
    )
    parser.add_argument("--puzzle-id", default=None)
    parser.add_argument("--branch-id", default=None)
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--num-feedback-passes", type=int, default=1)
    parser.add_argument("--error-feedback", choices=["first", "all"], default="all")
    parser.add_argument(
        "--include-past-outcomes",
        dest="include_past_outcomes",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--exclude-past-outcomes",
        dest="include_past_outcomes",
        action="store_false",
    )
    parser.add_argument("--new-concepts", default=None)
    parser.add_argument(
        "--out",
        default="outputs/parity/replay_retry_prompt_parity_report.json",
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    report = compare_retry_replay(
        arc_repo=Path(args.arc_repo).expanduser().resolve(),
        arc_run_dir=Path(args.arc_run_dir).expanduser().resolve(),
        puzzle_id=args.puzzle_id,
        branch_id=args.branch_id,
        thread_id=args.thread_id,
        num_feedback_passes=int(args.num_feedback_passes),
        error_feedback=str(args.error_feedback),
        include_past_outcomes=bool(args.include_past_outcomes),
        new_concepts=args.new_concepts,
    )
    write_json(args.out, report)
    print(f"report -> {args.out}")
    print("arc_replay_matches_expected:", report["equalities"]["arc_replay_matches_expected_retry_prompt"])
    print("mem2_replay_matches_arc:", report["equalities"]["mem2_replay_matches_arc_replay"])


if __name__ == "__main__":
    main()

