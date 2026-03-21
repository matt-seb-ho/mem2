from __future__ import annotations

import concurrent.futures
import asyncio
import logging
import re
from typing import Any

from mem2.core.entities import AttemptRecord, EvalRecord, ProblemSpec, RunContext

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

JUDGE_PROMPT = """\
You are a math answer equivalence checker. Given a reference answer and a student answer, determine if they are mathematically equivalent.

### Reference Answer
{expected}

### Student Answer
{student}

### Instructions
- Consider mathematical equivalence, not just string equality.
- Equivalent forms: 1/2 = 0.5, (2,4,6) = (2, 4, 6), x^2+1 = 1+x^2, etc.
- If the student answer is a simplified or expanded form of the reference, it is correct.
- If the student answer is numerically equal to the reference, it is correct.
- Respond with ONLY "TRUE" or "FALSE" on a single line. No explanation.
"""


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _run_async(coro):
    """Run an async coroutine from sync code, even if an event loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class OlympiadEvaluator:
    """Evaluate olympiad math attempts using LLM-based answer equivalence checking.

    Extracts the last \\boxed{} from the model output and uses an LLM judge
    to determine if it is mathematically equivalent to the ground truth.
    Falls back to exact string/integer comparison first.
    """

    name = "olympiad_eval"
    DOMAIN_NAME = "math"

    def __init__(
        self,
        judge_model: str = "qwen/qwen3.5-flash-02-23",
        judge_provider_profile: str = "llmplus_openrouter",
        timeout_s: float = 30.0,
        require_all_tests: Any = None,
    ):
        self.judge_model = judge_model
        self.judge_provider_profile = judge_provider_profile
        self.timeout_s = float(timeout_s)
        self._judge_client = None

    def _get_judge(self):
        if self._judge_client is None:
            from mem2.providers.llmplus_client import LLMPlusProviderClient
            self._judge_client = LLMPlusProviderClient(
                profile_cfg={
                    "profile_name": self.judge_provider_profile,
                    "dotenv_path": ".env",
                    "default_max_concurrency": 16,
                }
            )
        return self._judge_client

    async def _judge_equivalence(self, expected: str, student: str) -> bool:
        # Fast path: exact string match
        if student.strip() == expected.strip():
            return True

        # Fast path: both parse as equal integers
        try:
            s = int(student.replace(",", "").replace(" ", "").strip())
            e = int(expected.replace(",", "").replace(" ", "").strip())
            if s == e:
                return True
        except (ValueError, TypeError):
            pass

        # LLM judge
        judge = self._get_judge()
        prompt = JUDGE_PROMPT.format(expected=expected, student=student)
        try:
            results = await judge.async_generate(
                prompt=prompt,
                model=self.judge_model,
                gen_cfg={"max_tokens": 32, "temperature": 0.0, "n": 1},
            )
            if results and results[0]:
                text = results[0].strip().upper()
                # Check last non-empty line for verdict
                for line in reversed(text.split("\n")):
                    line = line.strip()
                    if not line:
                        continue
                    if "TRUE" in line:
                        return True
                    if "FALSE" in line:
                        return False
                    break
            logger.warning("Judge ambiguous for '%s' vs '%s': %s", expected, student, results)
            return False
        except Exception as e:
            logger.warning("Judge call failed: %s", e)
            return False

    def evaluate(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
    ) -> list[EvalRecord]:
        expected = problem.metadata.get("answer_str") or problem.metadata.get("answer", "")
        records = []

        for idx, attempt in enumerate(attempts):
            completion = attempt.completion or ""
            boxed = extract_boxed_answer(completion)

            if boxed is None:
                records.append(EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=False,
                    train_details=[],
                    test_details=[{
                        "is_train": False, "pair_idx": 0, "correct": False,
                        "error": "no \\boxed{} answer found",
                        "output": None, "expected": expected,
                    }],
                    metadata={"evaluator": self.name, "parsing_error": "no boxed answer"},
                ))
                continue

            is_correct = _run_async(self._judge_equivalence(str(expected), boxed))

            records.append(EvalRecord(
                problem_uid=problem.uid,
                attempt_idx=idx,
                is_correct=is_correct,
                train_details=[],
                test_details=[{
                    "is_train": False, "pair_idx": 0,
                    "correct": is_correct, "error": None,
                    "output": boxed, "expected": expected,
                }],
                metadata={
                    "evaluator": self.name,
                    "parsing_error": None,
                    "raw_boxed": boxed,
                    "judge_model": self.judge_model,
                },
            ))

        return records

    def aggregate(self, ctx: RunContext, records: list[EvalRecord]) -> dict[str, Any]:
        if not records:
            return {
                "accuracy_per_attempt": 0.0, "official_score": 0.0,
                "strict_score": 0.0, "total_attempts": 0, "correct_attempts": 0,
            }
        total = len(records)
        correct = sum(1 for r in records if r.is_correct)
        solved_by_puzzle: dict[str, bool] = {}
        for rec in records:
            solved_by_puzzle.setdefault(rec.problem_uid, False)
            solved_by_puzzle[rec.problem_uid] = solved_by_puzzle[rec.problem_uid] or rec.is_correct
        n_solved = sum(1 for ok in solved_by_puzzle.values() if ok)
        n_puzzles = len(solved_by_puzzle)
        return {
            "accuracy_per_attempt": correct / total,
            "official_score": float(n_solved),
            "strict_score": float(n_solved),
            "strict_solved_puzzles": n_solved,
            "official_score_sum": float(n_solved),
            "total_attempts": total,
            "correct_attempts": correct,
            "total_puzzles": n_puzzles,
            "solve_rate": n_solved / n_puzzles if n_puzzles > 0 else 0.0,
        }
