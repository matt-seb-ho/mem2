"""
GPQA Diamond Concept Memory Pilot (I07)

Tests whether concept memory improves graduate-level science QA.

Conditions:
  baseline   — standard MCQ prompt
  concept    — inject science concepts before MCQ (future)
  explanation — inject explanations from similar solved problems (future)

Usage:
    source .env
    python scripts/gpqa_pilot.py --condition baseline
    python scripts/gpqa_pilot.py --compare
"""
import asyncio
import csv
import json
import os
import random
import re
import argparse
from pathlib import Path

from openai import AsyncOpenAI

BASE = Path(__file__).resolve().parent.parent
GPQA_PATH = BASE / "data" / "gpqa_diamond" / "gpqa_diamond.csv"
OUTPUT_DIR = BASE / "outputs" / "_runs" / "gpqa_pilot"

MODEL = "qwen/qwen3.5-flash-02-23"

SYSTEM_PROMPT = (
    "You are an expert scientist. Answer multiple-choice questions by selecting "
    "the correct option. Think step by step, then give your final answer."
)

BASELINE_TEMPLATE = """\
### Question
{question}

### Options
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

### Instructions
Think through this step by step. After your reasoning, state your final answer as:
ANSWER: X
where X is A, B, C, or D.
"""

CONCEPT_TEMPLATE = """\
### Relevant Science Concepts
{concepts}

---

### Question
{question}

### Options
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

### Instructions
The concepts above may help with this question. Think through this step by step.
After your reasoning, state your final answer as:
ANSWER: X
where X is A, B, C, or D.
"""


def load_gpqa() -> list[dict]:
    """Load GPQA Diamond questions."""
    questions = []
    with open(GPQA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            q = {
                "uid": f"gpqa_{i:03d}",
                "question": row["Question"],
                "correct": row["Correct Answer"],
                "incorrect_1": row["Incorrect Answer 1"],
                "incorrect_2": row["Incorrect Answer 2"],
                "incorrect_3": row["Incorrect Answer 3"],
                "explanation": row.get("Explanation", ""),
                "domain": row.get("High-level domain", ""),
                "subdomain": row.get("Subdomain", ""),
                "difficulty": row.get("Writer's Difficulty Estimate", ""),
            }
            questions.append(q)
    return questions


def shuffle_options(q: dict, seed: int) -> tuple[dict, str]:
    """Shuffle options and return (option_map, correct_letter)."""
    rng = random.Random(seed)
    options = [
        ("correct", q["correct"]),
        ("inc1", q["incorrect_1"]),
        ("inc2", q["incorrect_2"]),
        ("inc3", q["incorrect_3"]),
    ]
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    option_map = {}
    correct_letter = ""
    for letter, (label, text) in zip(letters, options):
        option_map[letter] = text
        if label == "correct":
            correct_letter = letter
    return option_map, correct_letter


def extract_answer(text: str) -> str | None:
    """Extract the model's answer letter from its response."""
    # Look for "ANSWER: X" pattern
    match = re.search(r'ANSWER:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: last standalone letter A-D
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()
    return None


async def generate(client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"


async def run_condition(
    condition: str,
    questions: list[dict],
    seed: int,
    output_path: Path,
):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(16)

    tasks = []
    for q in questions:
        option_map, correct_letter = shuffle_options(q, seed=hash(q["uid"]) + seed)

        if condition == "baseline":
            prompt = BASELINE_TEMPLATE.format(
                question=q["question"],
                option_a=option_map["A"],
                option_b=option_map["B"],
                option_c=option_map["C"],
                option_d=option_map["D"],
            )
        else:
            # concept/explanation conditions — to be implemented
            prompt = BASELINE_TEMPLATE.format(
                question=q["question"],
                option_a=option_map["A"],
                option_b=option_map["B"],
                option_c=option_map["C"],
                option_d=option_map["D"],
            )

        tasks.append({
            "q": q,
            "prompt": prompt,
            "correct_letter": correct_letter,
            "option_map": option_map,
        })

    print(f"Running {condition} (seed={seed}) on {len(tasks)} questions...")
    completions = await asyncio.gather(
        *[generate(client, t["prompt"], sem) for t in tasks]
    )

    print("Evaluating...")
    results = []
    for task, completion in zip(tasks, completions):
        q = task["q"]
        model_answer = extract_answer(completion)
        correct_letter = task["correct_letter"]
        is_correct = model_answer == correct_letter

        result = {
            "uid": q["uid"],
            "domain": q["domain"],
            "subdomain": q["subdomain"],
            "difficulty": q["difficulty"],
            "condition": condition,
            "seed": seed,
            "correct_letter": correct_letter,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "prompt_len": len(task["prompt"]),
        }
        results.append(result)

        status = "OK" if is_correct else "WRONG"
        ans_str = f"model={model_answer}" if model_answer else "model=NONE"
        print(f"  {q['uid']} [{q['domain'][:4]}]: {status} (correct={correct_letter}, {ans_str})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    print(f"\n{condition.upper()} (seed={seed}): {correct}/{total} ({100*correct/total:.1f}%)")

    # Per-domain breakdown
    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)
    print("\nPer-domain:")
    for domain in sorted(by_domain):
        dr = by_domain[domain]
        dc = sum(1 for r in dr if r["is_correct"])
        print(f"  {domain}: {dc}/{len(dr)} ({100*dc/len(dr):.0f}%)")

    # Parse failure rate
    no_answer = sum(1 for r in results if r["model_answer"] is None)
    print(f"\nParse failures: {no_answer}/{total}")

    return results


def compare_all():
    """Compare all conditions."""
    result_files = sorted(OUTPUT_DIR.glob("*_results.json"))
    if not result_files:
        print("No result files found")
        return

    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
        if data:
            correct = sum(1 for r in data if r["is_correct"])
            total = len(data)
            cond = data[0]["condition"]
            seed = data[0]["seed"]
            print(f"{f.name}: {cond} s{seed} = {correct}/{total} ({100*correct/total:.1f}%)")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "concept", "explanation"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_all()
        return

    if not args.condition:
        parser.error("--condition required")

    print("Loading GPQA Diamond...")
    questions = load_gpqa()
    print(f"  {len(questions)} questions")

    # Domain distribution
    domains = {}
    for q in questions:
        domains[q["domain"]] = domains.get(q["domain"], 0) + 1
    for d, n in sorted(domains.items()):
        print(f"  {d}: {n}")

    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"  Limited to {len(questions)} questions")

    output_path = OUTPUT_DIR / f"{args.condition}_s{args.seed}_results.json"

    await run_condition(
        condition=args.condition,
        questions=questions,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
