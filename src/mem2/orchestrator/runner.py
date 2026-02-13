from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from mem2.core.context import build_run_context
from mem2.core.entities import FeedbackRecord, RetrievalBundle, RunBundle
from mem2.core.lifecycle import LifecycleLogger
from mem2.core.retry_policy import ArcMemoRetryPolicy
from mem2.io.hashing import stable_hash
from mem2.io.json_io import write_json
from mem2.orchestrator.lockstep_replay import LockstepReplayArtifacts
from mem2.orchestrator.wiring import PipelineComponents
from mem2.prompting.render import prompt_fingerprint


@dataclass(slots=True)
class PassArtifacts:
    prompts: list[str] = field(default_factory=list)
    metadata: list[object] = field(default_factory=list)
    model_outputs: list[list[str]] = field(default_factory=list)
    reselect_prompts: list[str] = field(default_factory=list)
    reselect_metadata: list[str] = field(default_factory=list)
    reselect_model_outputs: list[list[str]] = field(default_factory=list)
    reselect_concept_uids: dict[str, list[object]] = field(default_factory=dict)
    reselected_lessons: dict[str, str] = field(default_factory=dict)
    reselect_parsing_errors: list[object] = field(default_factory=list)


class PipelineRunner:
    def __init__(self, config: dict, components: PipelineComponents):
        self.config = config
        self.components = components
        self.lifecycle = LifecycleLogger()
        self.retry_policy = ArcMemoRetryPolicy.from_config(config)
        self.execution_mode = str(config.get("run", {}).get("execution_mode", "sequential"))
        self.driver_log_lines: list[str] = []
        self.lockstep_replay = LockstepReplayArtifacts.from_config(config)
        self._reselect_hint_enabled = bool(
            getattr(self.components.inference_engine, "include_reselected_lessons", False)
        )
        ok, err = self.retry_policy.is_valid()
        if not ok:
            raise ValueError(f"Invalid retry policy config: {err}")
        if self.execution_mode not in {"sequential", "arc_batch"}:
            raise ValueError(
                f"Invalid execution mode: {self.execution_mode}. "
                "Supported modes: sequential, arc_batch"
            )

    @staticmethod
    def _log_ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    def _driver_log(self, component: str, message: str, level: str = "INFO") -> None:
        line = f"[{self._log_ts()}][{component}][{level}] - {message}"
        self.driver_log_lines.append(line)
        print(line)

    def _driver_log_raw(self, line: str) -> None:
        self.driver_log_lines.append(line)
        print(line)

    def _write_driver_log_file(self, run_root: Path) -> None:
        log_path = run_root / "driver.log"
        content = ""
        if self.driver_log_lines:
            content = "\n".join(self.driver_log_lines) + "\n"
        log_path.write_text(content, encoding="utf-8")

    def _provider_usage_snapshot(self) -> dict:
        try:
            return copy.deepcopy(self.components.provider.get_usage_snapshot())
        except Exception:
            return {}

    @staticmethod
    def _iteration_metadata_entry(pass_idx: int, problem_uid: str, step_idx: int) -> list[str] | dict:
        if pass_idx == 0:
            return [problem_uid, "0"]
        return {
            "puzzle_id": problem_uid,
            "branch_id": "0",
            "thread_id": "0",
            "step_idx": step_idx,
        }

    def _record_pass_prompt_artifacts(
        self,
        *,
        pass_idx: int,
        job: dict,
        attempts: list,
        pass_artifacts: PassArtifacts,
    ) -> None:
        step_idx = len(job.get("history", []))
        problem_uid = job["problem"].uid
        pass_artifacts.metadata.append(
            self._iteration_metadata_entry(pass_idx=pass_idx, problem_uid=problem_uid, step_idx=step_idx)
        )
        if attempts:
            pass_artifacts.prompts.append(attempts[0].prompt)
            pass_artifacts.model_outputs.append([a.completion for a in attempts])
        else:
            pass_artifacts.prompts.append("")
            pass_artifacts.model_outputs.append([])

    def _record_reselect_artifacts(self, *, pass_idx: int, job: dict, pass_artifacts: PassArtifacts) -> None:
        if pass_idx == 0 or not self._reselect_hint_enabled:
            return
        retrieval = job.get("retrieval")
        if retrieval is None:
            return
        md = retrieval.metadata or {}
        selector_prompt = str(md.get("selector_prompt", "")).strip()
        selector_completion = str(md.get("selector_completion", "")).strip()
        parse_error = md.get("selector_parsing_error")
        selected_uids = md.get("selector_selected_uids")
        if (
            not retrieval.hint_text
            and not selector_prompt
            and not selector_completion
            and not selected_uids
        ):
            return
        uid = job["problem"].uid
        if uid in pass_artifacts.reselected_lessons:
            return
        pass_artifacts.reselected_lessons[uid] = retrieval.hint_text or ""
        pass_artifacts.reselect_metadata.append(uid)
        pass_artifacts.reselect_prompts.append(
            selector_prompt
            or f"[mem2_retriever] puzzle_id={uid} history_len={len(job.get('history', []))}"
        )
        pass_artifacts.reselect_model_outputs.append([selector_completion or (retrieval.hint_text or "")])
        if isinstance(selected_uids, list):
            pass_artifacts.reselect_concept_uids[uid] = selected_uids
        else:
            pass_artifacts.reselect_concept_uids[uid] = []
        if parse_error:
            pass_artifacts.reselect_parsing_errors.append(
                {"puzzle_id": uid, "error": str(parse_error)}
            )

    @staticmethod
    def _strict_display_value(value: object) -> object:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _build_solution_trees_snapshot(
        self,
        *,
        per_problem_attempts: dict[str, list],
        per_problem_evals: dict[str, list],
    ) -> dict:
        trees: dict[str, dict] = {}
        for problem_uid, attempts in per_problem_attempts.items():
            evals = per_problem_evals.get(problem_uid, [])
            threads: dict[str, dict] = {}
            for idx, attempt in enumerate(attempts):
                thread_id = str(attempt.metadata.get("path_idx", 0))
                if thread_id not in threads:
                    threads[thread_id] = {"thread_id": thread_id, "steps": []}

                eval_record = evals[idx] if idx < len(evals) else None
                if eval_record is None:
                    train_results = []
                    test_results = []
                    parsing_error = None
                else:
                    train_results = list(eval_record.train_details)
                    test_results = list(eval_record.test_details)
                    parsing_error = eval_record.metadata.get("parsing_error")

                step_idx = int(attempt.metadata.get("global_step_idx", idx))
                threads[thread_id]["steps"].append(
                    {
                        "step_idx": step_idx,
                        "thread_id": thread_id,
                        "branch_id": "0",
                        "puzzle_id": problem_uid,
                        "completion": attempt.completion,
                        "parsing_error": parsing_error,
                        "train_results": train_results,
                        "test_results": test_results,
                    }
                )

            trees[problem_uid] = {
                "puzzle_id": problem_uid,
                "prompt_branches": {
                    "0": {
                        "branch_id": "0",
                        "threads": threads,
                    }
                },
            }
        return trees

    def _write_iteration_artifacts(
        self,
        *,
        run_root: Path,
        pass_idx: int,
        pass_artifacts: PassArtifacts,
        usage_before: dict,
        usage_after: dict,
        per_problem_attempts: dict[str, list],
        per_problem_evals: dict[str, list],
    ) -> None:
        iteration_idx = pass_idx + 1
        iter_dir = run_root / f"iteration_{iteration_idx}"
        write_json(iter_dir / "prompts.json", pass_artifacts.prompts)
        write_json(iter_dir / "metadata.json", pass_artifacts.metadata)
        write_json(iter_dir / "model_outputs.json", pass_artifacts.model_outputs)
        write_json(
            iter_dir / "gen_progress.json",
            {"total": len(pass_artifacts.prompts), "completed": len(pass_artifacts.prompts)},
        )
        write_json(iter_dir / "token_usage.json", {"before": usage_before, "after": usage_after})
        write_json(
            iter_dir / "solution_trees.json",
            self._build_solution_trees_snapshot(
                per_problem_attempts=per_problem_attempts,
                per_problem_evals=per_problem_evals,
            ),
        )

        if pass_idx == 0 or not self._reselect_hint_enabled:
            return

        reselect_dir = iter_dir / "reselect_concepts"
        write_json(reselect_dir / "metadata.json", pass_artifacts.reselect_metadata)
        write_json(reselect_dir / "prompts.json", pass_artifacts.reselect_prompts)
        write_json(reselect_dir / "model_outputs.json", pass_artifacts.reselect_model_outputs)
        concept_uids = dict(pass_artifacts.reselect_concept_uids)
        for uid in pass_artifacts.reselect_metadata:
            concept_uids.setdefault(uid, [])
        write_json(reselect_dir / "concept_uids.json", concept_uids)
        write_json(reselect_dir / "reselected_lessons.json", pass_artifacts.reselected_lessons)
        write_json(reselect_dir / "parsing_errors.json", pass_artifacts.reselect_parsing_errors)
        write_json(
            reselect_dir / "gen_progress.json",
            {
                "total": len(pass_artifacts.reselect_metadata),
                "completed": len(pass_artifacts.reselect_metadata),
            },
        )
        write_json(reselect_dir / "token_usage.json", {"before": usage_before, "after": usage_before})
        self._driver_log(
            "concept_mem.selection.description.select",
            f"Parsing error count: {len(pass_artifacts.reselect_parsing_errors)}",
        )
        self._driver_log(
            "concept_mem.selection.description.select",
            f"Wrote to {reselect_dir}",
        )

    async def _run_jobs_with_progress(self, jobs: list[dict], ctx) -> list[object]:
        if not jobs:
            return []

        async def _indexed_job(index: int, job: dict) -> tuple[int, object]:
            try:
                out = await self._run_inference_job(ctx, job)
                return index, out
            except Exception as exc:  # keep parity with current error-handling behavior
                return index, exc

        indexed_tasks = [
            asyncio.create_task(_indexed_job(i, job), name=f"mem2-job-{i}")
            for i, job in enumerate(jobs)
        ]
        results: list[object] = [None] * len(indexed_tasks)

        with tqdm(total=len(indexed_tasks)) as pbar:
            for fut in asyncio.as_completed(indexed_tasks):
                idx, value = await fut
                results[idx] = value
                pbar.update(1)
        return results

    @staticmethod
    def _attempt_passes_criterion(eval_record, criterion: str) -> bool:
        criterion = criterion.lower().strip()
        if criterion == "train":
            details = eval_record.train_details
        elif criterion == "test":
            details = eval_record.test_details
        else:
            details = list(eval_record.train_details) + list(eval_record.test_details)
        if not details:
            return False
        return all(bool(d.get("correct", False)) for d in details)

    def _problem_solved_in_batch(self, eval_records, criterion: str) -> bool:
        return any(self._attempt_passes_criterion(rec, criterion=criterion) for rec in eval_records)

    def _build_problem_job(self, ctx, memory, problem, history, feedback_hist):
        if history:
            plan = self.components.trajectory_policy.plan_retry(ctx, problem, history, feedback_hist)
        else:
            plan = self.components.trajectory_policy.plan_initial(ctx, problem)
        if history and self.lockstep_replay.enabled:
            replay_feedback = self.lockstep_replay.feedback_history(problem.uid, len(history))
            if replay_feedback is None:
                raise RuntimeError(
                    "Lockstep replay missing feedback metadata for puzzle "
                    f"{problem.uid} in {self.lockstep_replay.source_run_dir}"
                )
            feedback_hist = [
                FeedbackRecord(
                    problem_uid=problem.uid,
                    attempt_idx=i,
                    feedback_type="gt",
                    content="",
                    metadata=md,
                )
                for i, md in enumerate(replay_feedback)
            ]
        prompt_options = getattr(self.components.inference_engine, "prompt_options", None)
        problem_data_enabled = bool(getattr(prompt_options, "problem_data", None))
        retriever = self.components.memory_retriever
        if not history and problem_data_enabled:
            retrieval = RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "disabled_problem_data"},
            )
        elif hasattr(retriever, "async_retrieve"):
            retrieval = None
        else:
            retrieval = retriever.retrieve(ctx, memory, problem, history)
        return {
            "problem": problem,
            "history": history,
            "feedback_hist": feedback_hist,
            "plan": plan,
            "memory_snapshot": memory,
            "retrieval": retrieval,
            "is_retry": bool(history),
        }

    async def _run_inference_job(self, ctx, job):
        problem_uid = job["problem"].uid
        retrieval = job.get("retrieval")
        if (
            retrieval is None
            and job["is_retry"]
            and self._reselect_hint_enabled
            and self.lockstep_replay.enabled
            and self.lockstep_replay.replay_reselected_lessons
        ):
            retrieval = self.lockstep_replay.retry_retrieval_bundle(
                problem_uid=problem_uid,
                history_len=len(job["history"]),
            )
            if retrieval is None:
                raise RuntimeError(
                    "Lockstep replay missing retry reselected lesson for puzzle "
                    f"{problem_uid} in {self.lockstep_replay.source_run_dir}"
                )
            job["retrieval"] = retrieval

        if retrieval is None:
            retriever = self.components.memory_retriever
            if hasattr(retriever, "async_retrieve"):
                retrieval = await retriever.async_retrieve(
                    ctx=ctx,
                    provider=self.components.provider,
                    memory=job["memory_snapshot"],
                    problem=job["problem"],
                    previous_attempts=job["history"],
                    selector_model=str(getattr(self.components.inference_engine, "model", "") or ""),
                )
            else:
                retrieval = retriever.retrieve(
                    ctx,
                    job["memory_snapshot"],
                    job["problem"],
                    job["history"],
                )
            job["retrieval"] = retrieval

        if job["is_retry"]:
            return await self.components.inference_engine.retry_attempt(
                ctx=ctx,
                provider=self.components.provider,
                problem=job["problem"],
                retrieval=retrieval,
                attempt_history=job["history"],
                feedback_history=job["feedback_hist"],
                trajectory_plan=job["plan"],
            )
        preset_completions = self.lockstep_replay.initial_completions(problem_uid)
        if self.lockstep_replay.enabled and self.lockstep_replay.replay_initial_completions:
            if preset_completions is None:
                raise RuntimeError(
                    "Lockstep replay missing initial completion for puzzle "
                    f"{problem_uid} in {self.lockstep_replay.source_run_dir}"
                )
            try:
                return await self.components.inference_engine.initial_attempt(
                    ctx=ctx,
                    provider=self.components.provider,
                    problem=job["problem"],
                    retrieval=retrieval,
                    trajectory_plan=job["plan"],
                    preset_completions=preset_completions,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Inference engine does not support lockstep preset completions "
                    "(missing `preset_completions` parameter)."
                ) from exc
        return await self.components.inference_engine.initial_attempt(
            ctx=ctx,
            provider=self.components.provider,
            problem=job["problem"],
            retrieval=retrieval,
            trajectory_plan=job["plan"],
        )

    async def _finalize_problem_result(
        self,
        *,
        ctx,
        memory,
        pass_idx: int,
        retry_criterion: str,
        job: dict,
        attempts,
        per_problem_attempts,
        per_problem_evals,
        per_problem_feedback,
        solved_problems: set[str],
        prompt_fingerprints: list[dict],
        all_attempts: list,
        all_evals: list,
        all_feedback: list,
    ):
        problem = job["problem"]
        history = job["history"]

        global_step_base = len(history)
        for att in attempts:
            att.pass_idx = pass_idx
        for i, att in enumerate(attempts):
            global_step_idx = global_step_base + i
            att.metadata["global_step_idx"] = global_step_idx
            att.metadata["prompt_fingerprint"] = prompt_fingerprint(att.prompt)
            prompt_fingerprints.append(
                {
                    "problem_uid": problem.uid,
                    "pass_idx": pass_idx,
                    "global_step_idx": global_step_idx,
                    "prompt_fingerprint": att.metadata["prompt_fingerprint"],
                }
            )

        eval_records = self.components.evaluator.evaluate(ctx, problem, attempts)
        for i, rec in enumerate(eval_records):
            rec.metadata["global_step_idx"] = global_step_base + i
            rec.metadata["pass_idx"] = pass_idx

        feedback_records = await self.components.feedback_engine.generate(
            ctx=ctx,
            provider=self.components.provider,
            problem=problem,
            attempts=attempts,
            eval_records=eval_records,
        )

        memory = self.components.memory_builder.update(
            ctx,
            memory,
            attempts=attempts,
            eval_records=eval_records,
            feedback_records=feedback_records,
        )

        per_problem_attempts[problem.uid].extend(attempts)
        per_problem_evals[problem.uid].extend(eval_records)
        per_problem_feedback[problem.uid].extend(feedback_records)
        all_attempts.extend(attempts)
        all_evals.extend(eval_records)
        all_feedback.extend(feedback_records)

        if self._problem_solved_in_batch(eval_records, criterion=retry_criterion):
            solved_problems.add(problem.uid)
        return memory

    async def _run_pass(
        self,
        *,
        ctx,
        memory,
        problems,
        pass_idx: int,
        retry_criterion: str,
        per_problem_attempts,
        per_problem_evals,
        per_problem_feedback,
        solved_problems: set[str],
        prompt_fingerprints: list[dict],
        all_attempts: list,
        all_evals: list,
        all_feedback: list,
        pass_artifacts: PassArtifacts,
    ):
        jobs = []
        for problem in problems.values():
            if problem.uid in solved_problems:
                continue
            history = per_problem_attempts[problem.uid]
            feedback_hist = per_problem_feedback[problem.uid]
            job = self._build_problem_job(ctx, memory, problem, history, feedback_hist)
            jobs.append(job)

        if not jobs:
            return memory

        results = await self._run_jobs_with_progress(jobs, ctx)
        for job, result in zip(jobs, results):
            if isinstance(result, Exception):
                self.lifecycle.emit(
                    "inference",
                    "runner",
                    "error",
                    f"inference job failed in {self.execution_mode} mode",
                    {
                        "problem_uid": job["problem"].uid,
                        "error_type": type(result).__name__,
                        "error": str(result),
                    },
                )
                attempts = []
            else:
                attempts = result
            self._record_reselect_artifacts(pass_idx=pass_idx, job=job, pass_artifacts=pass_artifacts)
            self._record_pass_prompt_artifacts(
                pass_idx=pass_idx,
                job=job,
                attempts=attempts,
                pass_artifacts=pass_artifacts,
            )
            memory = await self._finalize_problem_result(
                ctx=ctx,
                memory=memory,
                pass_idx=pass_idx,
                retry_criterion=retry_criterion,
                job=job,
                attempts=attempts,
                per_problem_attempts=per_problem_attempts,
                per_problem_evals=per_problem_evals,
                per_problem_feedback=per_problem_feedback,
                solved_problems=solved_problems,
                prompt_fingerprints=prompt_fingerprints,
                all_attempts=all_attempts,
                all_evals=all_evals,
                all_feedback=all_feedback,
            )
        return memory

    async def run(self) -> RunBundle:
        ctx = build_run_context(self.config)
        run_root = Path(ctx.output_dir)

        self._driver_log("__main__", f"Output directory: {run_root}")
        if self.lockstep_replay.enabled:
            self._driver_log(
                "__main__",
                f"Lockstep replay enabled from {self.lockstep_replay.source_run_dir} "
                f"(initial_completions={self.lockstep_replay.replay_initial_completions}, "
                f"reselected_lessons={self.lockstep_replay.replay_reselected_lessons})",
            )
        prompt_opts = (
            self.config.get("components", {})
            .get("inference_engine", {})
            .get("prompt_options") or {}
        )
        sys_prompt = prompt_opts.get("system_prompt_key", "default")
        self._driver_log("__main__", f"Using system prompt: {sys_prompt}")

        self.lifecycle.emit("run", "runner", "info", "run started", {"run_id": ctx.run_id})

        self.components.artifact_sink.write_stage_artifact(ctx, "frozen_config", self.config)
        write_json(run_root / "run_context.json", ctx)

        task_spec = self.components.task_adapter.get_task_spec(ctx)
        self.components.artifact_sink.write_stage_artifact(ctx, "task_spec", task_spec)

        problems = self.components.benchmark.load(ctx)
        self.components.benchmark.validate(problems)
        self.components.artifact_sink.write_stage_artifact(ctx, "problems", problems)

        memory = self.components.memory_builder.initialize(ctx, problems)
        self.components.artifact_sink.write_stage_artifact(ctx, "memory/initial", memory)

        all_attempts = []
        all_evals = []
        all_feedback = []

        per_problem_attempts = defaultdict(list)
        per_problem_evals = defaultdict(list)
        per_problem_feedback = defaultdict(list)
        solved_problems: set[str] = set()
        prompt_fingerprints: list[dict] = []

        if hasattr(self.components.inference_engine, "set_retry_policy"):
            self.components.inference_engine.set_retry_policy(self.retry_policy)

        max_passes = self.retry_policy.max_passes
        retry_criterion = self.retry_policy.criterion

        for pass_idx in range(max_passes):
            self._driver_log("__main__", f"running iteration {pass_idx + 1} of {max_passes}")
            if pass_idx > 0 and self._reselect_hint_enabled:
                self._driver_log(
                    "concept_mem.selection.description.select",
                    "Reselecting concepts based on previous completion...",
                )

            usage_before_pass = self._provider_usage_snapshot()
            pass_artifacts = PassArtifacts()

            self.lifecycle.emit(
                "pass",
                "runner",
                "info",
                "starting pass",
                {"pass_idx": pass_idx, "execution_mode": self.execution_mode},
            )
            memory = await self._run_pass(
                ctx=ctx,
                memory=memory,
                problems=problems,
                pass_idx=pass_idx,
                retry_criterion=retry_criterion,
                per_problem_attempts=per_problem_attempts,
                per_problem_evals=per_problem_evals,
                per_problem_feedback=per_problem_feedback,
                solved_problems=solved_problems,
                prompt_fingerprints=prompt_fingerprints,
                all_attempts=all_attempts,
                all_evals=all_evals,
                all_feedback=all_feedback,
                pass_artifacts=pass_artifacts,
            )

            usage_after_pass = self._provider_usage_snapshot()
            self._write_iteration_artifacts(
                run_root=run_root,
                pass_idx=pass_idx,
                pass_artifacts=pass_artifacts,
                usage_before=usage_before_pass,
                usage_after=usage_after_pass,
                per_problem_attempts=per_problem_attempts,
                per_problem_evals=per_problem_evals,
            )

            pass_summary = self.components.evaluator.aggregate(ctx, all_evals)
            self._driver_log("__main__", f"Score Report (iter {pass_idx + 1}):")
            self._driver_log_raw(
                f"  official: {pass_summary.get('official_score', 0.0)}"
                f"  strict: {self._strict_display_value(pass_summary.get('strict_score', 0.0))}"
                f"  problems: {len(problems)}"
            )

            self.components.artifact_sink.write_stage_artifact(ctx, f"memory/pass_{pass_idx + 1}", memory)
            if len(solved_problems) == len(problems):
                self.lifecycle.emit(
                    "pass",
                    "runner",
                    "info",
                    "early stop: all problems solved for retry criterion",
                    {"pass_idx": pass_idx, "criterion": retry_criterion},
                )
                break

        memory = self.components.memory_builder.consolidate(ctx, memory)
        self.components.artifact_sink.write_stage_artifact(ctx, "memory/final", memory)

        usage = self.components.provider.get_usage_snapshot()
        write_json(run_root / "provider_usage.json", usage)

        summary = self.components.evaluator.aggregate(ctx, all_evals)
        summary["attempt_count"] = len(all_attempts)
        summary["feedback_count"] = len(all_feedback)
        summary["problem_count"] = len(problems)
        summary["prompt_fingerprint_count"] = len(prompt_fingerprints)
        summary["config_hash"] = stable_hash(self.config)

        self.components.artifact_sink.write_stage_artifact(ctx, "attempts", all_attempts)
        self.components.artifact_sink.write_stage_artifact(ctx, "eval_records", all_evals)
        self.components.artifact_sink.write_stage_artifact(ctx, "feedback_records", all_feedback)
        self.components.artifact_sink.write_stage_artifact(
            ctx, "prompt_fingerprints", prompt_fingerprints
        )

        self.lifecycle.emit("run", "runner", "info", "run finished", summary)
        self.components.artifact_sink.write_stage_artifact(ctx, "events", self.lifecycle.events)
        self.components.artifact_sink.write_run_summary(ctx, summary)
        self._driver_log("__main__", f"Output directory: {run_root}")
        self._write_driver_log_file(run_root)

        return RunBundle(
            task_spec=task_spec,
            problems=problems,
            attempts=all_attempts,
            eval_records=all_evals,
            feedback_records=all_feedback,
            memory_state=memory,
            summary=summary,
            events=self.lifecycle.events,
        )



def run_sync(config: dict, components: PipelineComponents) -> RunBundle:
    return asyncio.run(PipelineRunner(config=config, components=components).run())
