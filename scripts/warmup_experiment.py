"""
Context Warm-up Experiment (I06)

Validates whether injecting a random worked math solution improves
math performance on Omni-MATH, and whether the effect is math-specific.

Conditions:
  A: baseline     — no injected context
  B: math-warmup  — random Omni-MATH problem + solution
  C: problem-only — random Omni-MATH problem (no solution)
                    Tests if reasoning chain matters or just seeing a problem

Design:
  - Full d1-d9 stratified sampling (matches devlog 34)
  - N problems per difficulty level (default 12 = 108 total)
  - Multiple seeds
  - Omni-MATH self as bank (leave-one-out for math-warmup)

Usage:
    source .env
    python scripts/warmup_experiment.py --condition baseline --seed 42
    python scripts/warmup_experiment.py --condition math-warmup --seed 42
    python scripts/warmup_experiment.py --condition problem-only --seed 42
    python scripts/warmup_experiment.py --compare
"""
import asyncio
import json
import os
import re
import argparse
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

BASE = Path(__file__).resolve().parent.parent
OMNI_PATH = BASE / "data" / "omni_math" / "problems.jsonl"
OUTPUT_DIR = BASE / "outputs" / "_runs" / "warmup_experiment"

MODEL = "qwen/qwen3.5-flash-02-23"
JUDGE_MODEL = "qwen/qwen3.5-flash-02-23"

SYSTEM_PROMPT = (
    "You are an expert competition math solver. "
    "You solve problems by writing clear mathematical reasoning."
)

BASELINE_TEMPLATE = """\
### Problem
{problem_text}

### Instructions
Solve the problem above using clear mathematical reasoning.
- Show your work step by step, naming any theorems or techniques you use.
- Present your final answer inside \\boxed{{}}, e.g. \\boxed{{42}}.
"""

WARMUP_TEMPLATE = """\
### Worked Example
Here is a solved math problem. Study the reasoning — it may help warm up your mathematical thinking.

**Problem:** {warmup_problem}

**Solution:** {warmup_solution}

---

### Target Problem
{problem_text}

### Instructions
Solve the target problem above using clear mathematical reasoning.
- Show your work step by step, naming any theorems or techniques you use.
- Present your final answer inside \\boxed{{}}, e.g. \\boxed{{42}}.
"""

PROBLEM_ONLY_TEMPLATE = """\
### Reference Problem
Here is a math problem for context. Consider what techniques might be relevant.

**Problem:** {warmup_problem}

---

### Target Problem
{problem_text}

### Instructions
Solve the target problem above using clear mathematical reasoning.
- Show your work step by step, naming any theorems or techniques you use.
- Present your final answer inside \\boxed{{}}, e.g. \\boxed{{42}}.
"""

JUDGE_PROMPT = """\
You are a math answer equivalence checker. Determine if the student's answer is mathematically equivalent to the expected answer.

Expected answer: {expected}
Student's answer: {student}

Reply with exactly one word: "equivalent" or "different".
"""

_BOXED_RE = re.compile(r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def load_problems(path: Path) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def extract_difficulty(prob: dict) -> float | None:
    level = prob.get("level", "")
    if "Difficulty" in level:
        try:
            return float(level.split("Difficulty")[1].strip())
        except (ValueError, IndexError):
            return None
    return None


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def strip_boxed(text: str) -> str:
    """Remove \\boxed{} from text to avoid parser pollution."""
    return re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"[\1]", text)


def select_stratified(
    problems: list[dict],
    per_level: int = 12,
    seed: int = 42,
    levels: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> list[dict]:
    """Select per_level problems per integer difficulty level, stratified."""
    import random
    rng = random.Random(seed)

    by_level = {}
    for p in problems:
        d = extract_difficulty(p)
        if d is not None:
            bucket = int(d)
            if bucket in levels:
                by_level.setdefault(bucket, []).append(p)

    selected = []
    for level in sorted(levels):
        pool = by_level.get(level, [])
        n = min(per_level, len(pool))
        sample = rng.sample(pool, n)
        selected.extend(sample)
        print(f"  d{level}: {n} problems (pool: {len(pool)})")

    return selected


def pick_random_warmup(
    target: dict,
    all_problems: list[dict],
    rng,
) -> dict:
    """Pick a random problem that isn't the target."""
    candidates = [p for p in all_problems if p["uid"] != target["uid"]]
    return rng.choice(candidates)


async def generate(client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=16384,
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"


async def judge_equivalence(
    client: AsyncOpenAI, expected: str, student: str, sem: asyncio.Semaphore
) -> bool:
    if expected.strip().lower() == student.strip().lower():
        return True
    try:
        e_int = int(expected.replace(",", "").strip())
        s_int = int(student.replace(",", "").strip())
        if e_int == s_int:
            return True
    except (ValueError, TypeError):
        pass

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    expected=expected, student=student
                )}],
                max_tokens=10,
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip().lower()
            return "equivalent" in answer
        except Exception:
            return False


async def run_condition(
    condition: str,
    eval_problems: list[dict],
    all_problems: list[dict],
    seed: int,
    output_path: Path,
):
    import random as _random
    rng = _random.Random(seed)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(16)

    tasks = []
    for prob in eval_problems:
        if condition == "math-warmup":
            warmup = pick_random_warmup(prob, all_problems, rng)
            sol = strip_boxed(warmup.get("solution", "(no solution)"))
            prompt = WARMUP_TEMPLATE.format(
                warmup_problem=warmup["problem"],
                warmup_solution=sol,
                problem_text=prob["problem"],
            )
        elif condition == "problem-only":
            warmup = pick_random_warmup(prob, all_problems, rng)
            prompt = PROBLEM_ONLY_TEMPLATE.format(
                warmup_problem=warmup["problem"],
                problem_text=prob["problem"],
            )
        else:  # baseline
            warmup = None
            prompt = BASELINE_TEMPLATE.format(problem_text=prob["problem"])

        tasks.append({
            "prob": prob,
            "prompt": prompt,
            "warmup_uid": warmup["uid"] if warmup else None,
            "warmup_answer": warmup.get("answer", "") if warmup else None,
        })

    print(f"Running {condition} (seed={seed}) on {len(tasks)} problems...")
    completions = await asyncio.gather(
        *[generate(client, t["prompt"], sem) for t in tasks]
    )

    print("Evaluating...")
    results = []
    for task, completion in zip(tasks, completions):
        prob = task["prob"]
        student_answer = extract_boxed(completion)
        expected = prob.get("answer", "")

        is_correct = False
        if student_answer:
            is_correct = await judge_equivalence(client, expected, student_answer, sem)

        d = extract_difficulty(prob)
        result = {
            "uid": prob["uid"],
            "difficulty": d,
            "difficulty_bucket": int(d) if d else None,
            "condition": condition,
            "seed": seed,
            "expected": expected,
            "student_answer": student_answer,
            "is_correct": is_correct,
            "warmup_uid": task["warmup_uid"],
            "warmup_answer": task["warmup_answer"],
            "answer_leaked": (
                task["warmup_answer"] is not None
                and str(task["warmup_answer"]).strip() == str(expected).strip()
            ),
            "prompt_len": len(task["prompt"]),
        }
        results.append(result)

        status = "OK" if is_correct else "WRONG"
        print(f"  {prob['uid']} d={d}: {status} (exp={expected}, got={student_answer})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    print(f"\n{condition.upper()} (seed={seed}): {correct}/{total} ({100*correct/total:.1f}%)")

    # Per-difficulty breakdown
    by_d = {}
    for r in results:
        b = r["difficulty_bucket"]
        by_d.setdefault(b, []).append(r)
    print("\nPer-difficulty:")
    for d in sorted(by_d):
        dr = by_d[d]
        dc = sum(1 for r in dr if r["is_correct"])
        print(f"  d{d}: {dc}/{len(dr)} ({100*dc/len(dr):.0f}%)")

    leaked = sum(1 for r in results if r.get("answer_leaked"))
    print(f"\nAnswer leakage: {leaked}/{total}")

    return results


def compare_all():
    """Compare all conditions across seeds."""
    result_files = sorted(OUTPUT_DIR.glob("*_s*_results.json"))
    if not result_files:
        print("No result files found in", OUTPUT_DIR)
        return

    all_results = {}
    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
        if data:
            key = (data[0]["condition"], data[0]["seed"])
            all_results[key] = data

    conditions = sorted(set(k[0] for k in all_results))
    seeds = sorted(set(k[1] for k in all_results))

    print("=" * 80)
    print("WARM-UP EXPERIMENT RESULTS")
    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    # Overall scores
    print("\n## Overall Accuracy")
    for cond in conditions:
        scores = []
        for seed in seeds:
            data = all_results.get((cond, seed))
            if data:
                correct = sum(1 for r in data if r["is_correct"])
                total = len(data)
                scores.append(correct / total)
                print(f"  {cond} s{seed}: {correct}/{total} ({100*correct/total:.1f}%)")
        if scores:
            print(f"  {cond} MEAN: {100*np.mean(scores):.1f}% (±{100*np.std(scores):.1f})")

    # Per-difficulty comparison (averaged across seeds)
    print("\n## Per-Difficulty (averaged across seeds)")
    all_diffs = set()
    for data in all_results.values():
        for r in data:
            all_diffs.add(r["difficulty_bucket"])

    header = "| Diff |"
    for cond in conditions:
        header += f" {cond} |"
    print(header)
    print("|" + "---|" * (len(conditions) + 1))

    for d in sorted(all_diffs):
        row = f"| d{d} |"
        for cond in conditions:
            rates = []
            for seed in seeds:
                data = all_results.get((cond, seed))
                if data:
                    dr = [r for r in data if r["difficulty_bucket"] == d]
                    if dr:
                        rates.append(sum(1 for r in dr if r["is_correct"]) / len(dr))
            if rates:
                row += f" {100*np.mean(rates):.0f}% |"
            else:
                row += " — |"
        print(row)

    # Per-problem overlap (first seed only)
    if len(conditions) >= 2 and seeds:
        s = seeds[0]
        base_data = all_results.get(("baseline", s))
        for cond in conditions:
            if cond == "baseline":
                continue
            cond_data = all_results.get((cond, s))
            if base_data and cond_data:
                b_by_uid = {r["uid"]: r["is_correct"] for r in base_data}
                c_by_uid = {r["uid"]: r["is_correct"] for r in cond_data}
                both = sum(1 for uid in b_by_uid if b_by_uid.get(uid) and c_by_uid.get(uid))
                only_b = sum(1 for uid in b_by_uid if b_by_uid.get(uid) and not c_by_uid.get(uid))
                only_c = sum(1 for uid in b_by_uid if not b_by_uid.get(uid) and c_by_uid.get(uid))
                neither = sum(1 for uid in b_by_uid if not b_by_uid.get(uid) and not c_by_uid.get(uid))
                print(f"\n## Overlap: baseline vs {cond} (seed={s})")
                print(f"  Both correct: {both}")
                print(f"  Only baseline: {only_b}")
                print(f"  Only {cond}: {only_c}")
                print(f"  Neither: {neither}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "math-warmup", "problem-only"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-level", type=int, default=12,
                       help="Problems per difficulty level (default 12 = 108 total)")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_all()
        return

    if not args.condition:
        parser.error("--condition is required unless --compare is used")

    print("Loading Omni-MATH...")
    all_problems = load_problems(OMNI_PATH)
    print(f"  {len(all_problems)} total problems")

    print(f"\nSelecting stratified eval set (per_level={args.per_level}):")
    eval_problems = select_stratified(all_problems, per_level=args.per_level, seed=42)
    print(f"  Total: {len(eval_problems)} eval problems")

    output_path = OUTPUT_DIR / f"{args.condition}_s{args.seed}_results.json"

    await run_condition(
        condition=args.condition,
        eval_problems=eval_problems,
        all_problems=all_problems,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
