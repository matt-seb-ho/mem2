"""Lockstep replay infrastructure for deterministic prompt-parity auditing.

Loads artifacts from a previous arc_memo (or mem2) run and replays completions,
reselected lessons, and feedback metadata so that downstream prompts are built
from identical inputs, enabling byte-level parity comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mem2.core.entities import FeedbackRecord, RetrievalBundle


# ---------------------------------------------------------------------------
# Helpers for loading arc_memo iteration artifacts
# ---------------------------------------------------------------------------

def _load_json_or_default(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _uid_from_metadata_item(item: object) -> str | None:
    if isinstance(item, dict):
        uid = item.get("puzzle_id") or item.get("problem_uid")
        if uid is None:
            return None
        return str(uid)
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, str):
        return item
    return None


def _map_values_by_uid(metadata: list[object], values: list[object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for md, value in zip(metadata, values):
        uid = _uid_from_metadata_item(md)
        if uid:
            mapped[uid] = value
    return mapped


def _first_step_completion_from_tree(tree_payload: object) -> str | None:
    if not isinstance(tree_payload, dict):
        return None
    prompt_branches = tree_payload.get("prompt_branches")
    if not isinstance(prompt_branches, dict) or not prompt_branches:
        return None
    branch = prompt_branches.get("0")
    if not isinstance(branch, dict):
        branch = next((v for v in prompt_branches.values() if isinstance(v, dict)), None)
    if not isinstance(branch, dict):
        return None
    threads = branch.get("threads")
    if not isinstance(threads, dict) or not threads:
        return None
    thread = threads.get("0")
    if not isinstance(thread, dict):
        thread = next((v for v in threads.values() if isinstance(v, dict)), None)
    if not isinstance(thread, dict):
        return None
    steps = thread.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    first_step = steps[0]
    if not isinstance(first_step, dict):
        return None
    completion = first_step.get("completion")
    if completion is None:
        return None
    return str(completion)


def _extract_feedback_metadata_from_arc_step(step_payload: object) -> dict[str, object]:
    errors: list[str] = []
    mismatches: list[dict[str, object]] = []
    if not isinstance(step_payload, dict):
        return {"errors": errors, "mismatches": mismatches}

    parsing_error = step_payload.get("parsing_error")
    if parsing_error:
        errors.append(str(parsing_error))
        return {"errors": errors, "mismatches": mismatches}

    train_results = step_payload.get("train_results")
    if not isinstance(train_results, list):
        return {"errors": errors, "mismatches": mismatches}

    for result in train_results:
        if not isinstance(result, dict):
            continue
        if bool(result.get("correct", False)):
            continue
        err = result.get("error")
        if err:
            errors.append(str(err))
            continue
        pair_idx = result.get("pair_idx")
        try:
            ex_idx = int(pair_idx) + 1
        except Exception:
            ex_idx = 1
        mismatches.append(
            {
                "example_idx": ex_idx,
                "output": result.get("output"),
            }
        )
    return {"errors": errors, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# LockstepReplayArtifacts
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LockstepReplayArtifacts:
    enabled: bool = False
    source_run_dir: str | None = None
    replay_initial_completions: bool = True
    replay_reselected_lessons: bool = True
    initial_completions_by_uid: dict[str, list[str]] = field(default_factory=dict)
    reselected_lessons_by_uid: dict[str, str] = field(default_factory=dict)
    reselect_prompt_by_uid: dict[str, str] = field(default_factory=dict)
    reselect_completion_by_uid: dict[str, str] = field(default_factory=dict)
    reselect_concept_uids_by_uid: dict[str, list[object]] = field(default_factory=dict)
    reselect_parsing_error_by_uid: dict[str, str] = field(default_factory=dict)
    feedback_metadata_by_uid: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "LockstepReplayArtifacts":
        run_cfg = config.get("run", {})
        replay_cfg = run_cfg.get("lockstep_replay", {}) or {}
        enabled = bool(replay_cfg.get("enabled", False))
        if not enabled:
            return cls(enabled=False)
        source_run_dir = replay_cfg.get("source_run_dir")
        if not source_run_dir:
            raise ValueError(
                "run.lockstep_replay.enabled=true requires run.lockstep_replay.source_run_dir"
            )
        run_dir = Path(str(source_run_dir)).expanduser()
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
        if not run_dir.exists():
            raise ValueError(f"Lockstep replay source run dir does not exist: {run_dir}")

        replay = cls(
            enabled=True,
            source_run_dir=str(run_dir),
            replay_initial_completions=bool(replay_cfg.get("replay_initial_completions", True)),
            replay_reselected_lessons=bool(replay_cfg.get("replay_reselected_lessons", True)),
        )
        replay._load_from_run_dir(run_dir)
        return replay

    def _load_from_run_dir(self, run_dir: Path) -> None:
        iter1 = run_dir / "iteration_1"
        md1 = _load_json_or_default(iter1 / "metadata.json", [])
        out1 = _load_json_or_default(iter1 / "model_outputs.json", [])
        if isinstance(md1, list) and isinstance(out1, list):
            mapped = _map_values_by_uid(md1, out1)
            for uid, value in mapped.items():
                if isinstance(value, list):
                    completions = [str(v) for v in value]
                elif value is None:
                    completions = []
                else:
                    completions = [str(value)]
                self.initial_completions_by_uid[uid] = completions
        # Arc retry prompts are built from solution tree step completions.
        # Prefer that source for lockstep replay to avoid subtle completion-text drift.
        trees = _load_json_or_default(iter1 / "solution_trees.json", {})
        if isinstance(trees, dict):
            for uid, tree_payload in trees.items():
                completion = _first_step_completion_from_tree(tree_payload)
                if completion is not None:
                    self.initial_completions_by_uid[str(uid)] = [completion]
                if not isinstance(tree_payload, dict):
                    continue
                prompt_branches = tree_payload.get("prompt_branches")
                if not isinstance(prompt_branches, dict) or not prompt_branches:
                    continue
                branch = prompt_branches.get("0")
                if not isinstance(branch, dict):
                    branch = next((v for v in prompt_branches.values() if isinstance(v, dict)), None)
                if not isinstance(branch, dict):
                    continue
                threads = branch.get("threads")
                if not isinstance(threads, dict) or not threads:
                    continue
                thread = threads.get("0")
                if not isinstance(thread, dict):
                    thread = next((v for v in threads.values() if isinstance(v, dict)), None)
                if not isinstance(thread, dict):
                    continue
                steps = thread.get("steps")
                if not isinstance(steps, list):
                    continue
                md_list: list[dict[str, object]] = []
                for step in steps:
                    md_list.append(_extract_feedback_metadata_from_arc_step(step))
                self.feedback_metadata_by_uid[str(uid)] = md_list

        reselect_dir = run_dir / "iteration_2" / "reselect_concepts"
        md2 = _load_json_or_default(reselect_dir / "metadata.json", [])
        prompts = _load_json_or_default(reselect_dir / "prompts.json", [])
        outputs = _load_json_or_default(reselect_dir / "model_outputs.json", [])
        if isinstance(md2, list) and isinstance(prompts, list):
            for uid, value in _map_values_by_uid(md2, prompts).items():
                self.reselect_prompt_by_uid[uid] = str(value)
        if isinstance(md2, list) and isinstance(outputs, list):
            for uid, value in _map_values_by_uid(md2, outputs).items():
                if isinstance(value, list):
                    completion = str(value[0]) if value else ""
                elif value is None:
                    completion = ""
                else:
                    completion = str(value)
                self.reselect_completion_by_uid[uid] = completion

        lessons = _load_json_or_default(reselect_dir / "reselected_lessons.json", {})
        if isinstance(lessons, dict):
            self.reselected_lessons_by_uid = {str(k): str(v) for k, v in lessons.items()}

        concept_uids = _load_json_or_default(reselect_dir / "concept_uids.json", {})
        if isinstance(concept_uids, dict):
            cleaned: dict[str, list[object]] = {}
            for uid, value in concept_uids.items():
                if isinstance(value, list):
                    cleaned[str(uid)] = list(value)
                else:
                    cleaned[str(uid)] = []
            self.reselect_concept_uids_by_uid = cleaned

        parsing_errors = _load_json_or_default(reselect_dir / "parsing_errors.json", [])
        if isinstance(parsing_errors, list):
            for item in parsing_errors:
                if not isinstance(item, (list, tuple)) or not item:
                    continue
                uid = str(item[0])
                msg = str(item[-1]) if len(item) > 1 else "unknown"
                self.reselect_parsing_error_by_uid[uid] = msg

    def initial_completions(self, problem_uid: str) -> list[str] | None:
        if not (self.enabled and self.replay_initial_completions):
            return None
        return self.initial_completions_by_uid.get(problem_uid)

    def retry_retrieval_bundle(self, problem_uid: str, history_len: int) -> RetrievalBundle | None:
        if not (self.enabled and self.replay_reselected_lessons):
            return None
        if problem_uid not in self.reselected_lessons_by_uid:
            return None
        metadata = {
            "selector_mode": "lockstep_replay",
            "query_source": "lockstep_replay",
            "candidate_source": "lockstep_replay",
            "candidate_count": 0,
            "history_attempts": history_len,
            "selector_prompt": self.reselect_prompt_by_uid.get(problem_uid, ""),
            "selector_completion": self.reselect_completion_by_uid.get(problem_uid, ""),
            "selector_parsing_error": self.reselect_parsing_error_by_uid.get(problem_uid),
            "selector_selected_uids": self.reselect_concept_uids_by_uid.get(problem_uid, []),
            "lockstep_source_run_dir": self.source_run_dir,
        }
        return RetrievalBundle(
            problem_uid=problem_uid,
            hint_text=self.reselected_lessons_by_uid.get(problem_uid),
            retrieved_items=[],
            metadata=metadata,
        )

    def feedback_history(self, problem_uid: str, history_len: int) -> list[dict[str, object]] | None:
        if not self.enabled:
            return None
        items = self.feedback_metadata_by_uid.get(problem_uid)
        if items is None:
            return None
        if history_len <= 0:
            return []
        return [dict(x) for x in items[:history_len]]
