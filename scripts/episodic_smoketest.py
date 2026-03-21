"""
Episodic Memory Smoketest (I05)

Tests whether injecting a worked solution to a similar problem
improves math performance on Omni-MATH.

Design:
- 20 Omni-MATH problems (d3-d5 from existing stratified set)
- TF-IDF retrieval: top-1 from rest of corpus (leave-one-out)
- Inject worked example as prompt preamble
- Compare to baseline (no worked example)
- Track per-problem similarity for segmented analysis

Usage:
    source .env
    python scripts/episodic_smoketest.py --mode baseline
    python scripts/episodic_smoketest.py --mode episodic
    python scripts/episodic_smoketest.py --mode compare
"""
import asyncio
import json
import os
import re
import time
import argparse
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
OMNI_PATH = BASE / "data" / "omni_math" / "problems.jsonl"
OUTPUT_DIR = BASE / "outputs" / "_runs" / "episodic_smoketest"

MODEL = "qwen/qwen3.5-flash-02-23"
JUDGE_MODEL = "qwen/qwen3.5-flash-02-23"

SYSTEM_PROMPT = (
    "You are an expert competition math solver. "
    "You solve problems by writing clear mathematical reasoning."
)

INITIAL_TEMPLATE = """\
### Problem
{problem_text}

### Instructions
Solve the problem above using clear mathematical reasoning.
- Show your work step by step, naming any theorems or techniques you use.
- You may use intermediate calculations, but focus on mathematical reasoning, not code.
- Present your final answer inside \\boxed{{}}, e.g. \\boxed{{42}}.
"""

EPISODIC_TEMPLATE = """\
### Similar Solved Problem
Here is a similar problem that has been solved. Study the approach — it may help you solve the target problem below.

**Problem:** {similar_problem}

**Solution:** {similar_solution}

---

### Target Problem
{problem_text}

### Instructions
Solve the target problem above using clear mathematical reasoning.
- The similar problem above may suggest useful techniques or approaches.
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


def select_eval_problems(problems: list[dict], n: int = 20, seed: int = 42) -> list[dict]:
    """Select n problems from d3-d5 range, stratified."""
    import random
    rng = random.Random(seed)
    by_diff = {}
    for p in problems:
        d = extract_difficulty(p)
        if d is not None and 3.0 <= d <= 5.0:
            bucket = int(d)
            by_diff.setdefault(bucket, []).append(p)

    selected = []
    per_bucket = n // len(by_diff) + 1
    for d in sorted(by_diff.keys()):
        sample = rng.sample(by_diff[d], min(per_bucket, len(by_diff[d])))
        selected.extend(sample)

    rng.shuffle(selected)
    return selected[:n]


def build_retrieval_index(problems: list[dict]):
    """Build TF-IDF index over all problems."""
    texts = [p["problem"] for p in problems]
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_similar(
    query_prob: dict,
    all_problems: list[dict],
    vectorizer,
    matrix,
    uid_to_idx: dict[str, int],
) -> tuple[dict, float]:
    """Find most similar problem (leave-one-out)."""
    query_vec = vectorizer.transform([query_prob["problem"]])
    sims = cosine_similarity(query_vec, matrix)[0]
    # Zero out self
    self_idx = uid_to_idx.get(query_prob["uid"])
    if self_idx is not None:
        sims[self_idx] = -1
    best_idx = int(np.argmax(sims))
    return all_problems[best_idx], float(sims[best_idx])


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
    """Use LLM judge to check answer equivalence."""
    # Fast path: exact match
    if expected.strip().lower() == student.strip().lower():
        return True
    # Try integer comparison
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
                messages=[
                    {"role": "user", "content": JUDGE_PROMPT.format(
                        expected=expected, student=student
                    )},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip().lower()
            return "equivalent" in answer
        except Exception:
            return False


async def run_condition(
    mode: str,
    eval_problems: list[dict],
    all_problems: list[dict],
    vectorizer,
    matrix,
    uid_to_idx: dict[str, int],
    output_path: Path,
):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Run: source .env")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(16)

    results = []
    tasks = []

    import random as _random
    rng = _random.Random(42)

    for prob in eval_problems:
        if mode == "episodic":
            similar, sim_score = retrieve_similar(
                prob, all_problems, vectorizer, matrix, uid_to_idx
            )
            # Strip \boxed{} from retrieved solution to avoid parser pollution
            sol = similar.get("solution", "(no solution available)")
            sol = re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"[\1]", sol)
            prompt = EPISODIC_TEMPLATE.format(
                similar_problem=similar["problem"],
                similar_solution=sol,
                problem_text=prob["problem"],
            )
        elif mode == "random":
            # Random control: inject a random solved problem (not most similar)
            candidates = [p for p in all_problems if p["uid"] != prob["uid"]]
            similar = rng.choice(candidates)
            sim_score = -1.0  # sentinel for random
            sol = similar.get("solution", "(no solution available)")
            sol = re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"[\1]", sol)
            prompt = EPISODIC_TEMPLATE.format(
                similar_problem=similar["problem"],
                similar_solution=sol,
                problem_text=prob["problem"],
            )
        else:
            similar, sim_score = None, 0.0
            prompt = INITIAL_TEMPLATE.format(problem_text=prob["problem"])

        tasks.append({
            "prob": prob,
            "prompt": prompt,
            "similar_uid": similar["uid"] if similar else None,
            "sim_score": sim_score,
        })

    print(f"Running {mode} condition on {len(tasks)} problems...")
    completions = await asyncio.gather(
        *[generate(client, t["prompt"], sem) for t in tasks]
    )

    print("Evaluating answers...")
    for task, completion in zip(tasks, completions):
        prob = task["prob"]
        student_answer = extract_boxed(completion)
        expected = prob.get("answer", "")

        if student_answer:
            is_correct = await judge_equivalence(client, expected, student_answer, sem)
        else:
            is_correct = False

        result = {
            "uid": prob["uid"],
            "difficulty": extract_difficulty(prob),
            "mode": mode,
            "expected": expected,
            "student_answer": student_answer,
            "is_correct": is_correct,
            "similar_uid": task["similar_uid"],
            "sim_score": task["sim_score"],
            "prompt_len": len(task["prompt"]),
            "completion_len": len(completion),
        }
        results.append(result)
        status = "OK" if is_correct else "WRONG"
        sim_info = f" sim={task['sim_score']:.3f}" if mode == "episodic" else ""
        print(f"  {prob['uid']} d={result['difficulty']}: {status}"
              f" (expected={expected}, got={student_answer}){sim_info}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    print(f"\n{mode.upper()} RESULT: {correct}/{total} ({100*correct/total:.1f}%)")
    return results


def compare_results(baseline_path: Path, episodic_path: Path):
    """Compare baseline, random, and episodic results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(episodic_path) as f:
        episodic = json.load(f)

    random_path = baseline_path.parent / "random_results.json"
    random_results = None
    if random_path.exists():
        with open(random_path) as f:
            random_results = json.load(f)

    b_by_uid = {r["uid"]: r for r in baseline}
    e_by_uid = {r["uid"]: r for r in episodic}
    r_by_uid = {r["uid"]: r for r in random_results} if random_results else {}

    print("=" * 80)
    print("EPISODIC MEMORY SMOKETEST COMPARISON")
    print("=" * 80)

    b_correct = sum(1 for r in baseline if r["is_correct"])
    e_correct = sum(1 for r in episodic if r["is_correct"])
    n = len(baseline)
    print(f"\nBaseline:  {b_correct}/{n} ({100*b_correct/n:.1f}%)")
    if random_results:
        rand_correct = sum(1 for r in random_results if r["is_correct"])
        print(f"Random:    {rand_correct}/{n} ({100*rand_correct/n:.1f}%)  [context control]")
        print(f"Episodic:  {e_correct}/{n} ({100*e_correct/n:.1f}%)")
        print(f"\nDelta (episodic - baseline): {e_correct - b_correct:+d} ({100*(e_correct-b_correct)/n:+.1f}pp)")
        print(f"Delta (episodic - random):   {e_correct - rand_correct:+d} ({100*(e_correct-rand_correct)/n:+.1f}pp)")
        print(f"Delta (random - baseline):   {rand_correct - b_correct:+d} ({100*(rand_correct-b_correct)/n:+.1f}pp)")
        print(f"\nInterpretation:")
        if e_correct > rand_correct > b_correct:
            print(f"  RELEVANCE MATTERS: episodic > random > baseline")
        elif e_correct > b_correct and e_correct == rand_correct:
            print(f"  CONTEXT EFFECT ONLY: episodic = random > baseline (extra context helps, relevance doesn't)")
        elif e_correct == rand_correct == b_correct:
            print(f"  NO EFFECT: all conditions equal")
        elif e_correct > rand_correct and rand_correct == b_correct:
            print(f"  PURE RELEVANCE: episodic > random = baseline (only relevant examples help)")
        else:
            print(f"  MIXED: check per-problem details")
    else:
        print(f"Episodic:  {e_correct}/{n} ({100*e_correct/n:.1f}%)")
        print(f"Delta:     {e_correct - b_correct:+d} ({100*(e_correct-b_correct)/n:+.1f}pp)")
        print(f"\n  (Run --mode random to add the context control condition)")

    # Per-problem comparison
    both = 0
    only_base = 0
    only_epis = 0
    neither = 0
    for uid in b_by_uid:
        b = b_by_uid[uid]["is_correct"]
        e = e_by_uid.get(uid, {}).get("is_correct", False)
        if b and e:
            both += 1
        elif b and not e:
            only_base += 1
        elif not b and e:
            only_epis += 1
        else:
            neither += 1

    print(f"\nPer-problem overlap:")
    print(f"  Both correct:    {both}")
    print(f"  Only baseline:   {only_base}")
    print(f"  Only episodic:   {only_epis}")
    print(f"  Neither:         {neither}")

    # Segment by similarity
    print(f"\nSegmented by retrieval similarity:")
    for lo, hi, label in [(0.5, 1.0, "high (>0.5)"), (0.3, 0.5, "mid (0.3-0.5)"), (0.0, 0.3, "low (<0.3)")]:
        seg = [r for r in episodic if lo <= r.get("sim_score", 0) < hi]
        if seg:
            seg_correct = sum(1 for r in seg if r["is_correct"])
            seg_base_correct = sum(
                1 for r in seg if b_by_uid.get(r["uid"], {}).get("is_correct", False)
            )
            print(f"  {label}: n={len(seg)}, baseline={seg_base_correct}/{len(seg)}, "
                  f"episodic={seg_correct}/{len(seg)}, delta={seg_correct-seg_base_correct:+d}")

    # Show flipped problems
    print(f"\nFlipped problems:")
    for uid in sorted(b_by_uid):
        b = b_by_uid[uid]["is_correct"]
        e = e_by_uid.get(uid, {}).get("is_correct", False)
        if b != e:
            er = e_by_uid.get(uid, {})
            sim = er.get("sim_score", 0)
            diff = er.get("difficulty", "?")
            direction = "baseline→episodic" if e else "episodic→baseline"
            print(f"  {uid} d={diff} sim={sim:.3f}: GAINED by {direction}")


async def main():
    parser = argparse.ArgumentParser(description="Episodic memory smoketest")
    parser.add_argument("--mode", choices=["baseline", "episodic", "random", "compare"], required=True)
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "compare":
        compare_results(
            OUTPUT_DIR / "baseline_results.json",
            OUTPUT_DIR / "episodic_results.json",
        )
        return

    print("Loading Omni-MATH...")
    all_problems = load_problems(OMNI_PATH)
    print(f"  {len(all_problems)} total problems")

    eval_problems = select_eval_problems(all_problems, n=args.n_problems, seed=args.seed)
    print(f"  Selected {len(eval_problems)} eval problems (d3-d5)")
    for p in eval_problems:
        print(f"    {p['uid']} d={extract_difficulty(p)}")

    print("\nBuilding TF-IDF index...")
    vectorizer, matrix = build_retrieval_index(all_problems)
    uid_to_idx = {p["uid"]: i for i, p in enumerate(all_problems)}

    output_path = OUTPUT_DIR / f"{args.mode}_results.json"

    await run_condition(
        mode=args.mode,
        eval_problems=eval_problems,
        all_problems=all_problems,
        vectorizer=vectorizer,
        matrix=matrix,
        uid_to_idx=uid_to_idx,
        output_path=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
