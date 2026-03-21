"""
GPQA Diamond Concept Experiment (I07)

Tests whether injecting science domain knowledge (from expert explanations
of similar solved problems) improves GPQA performance.

Conditions:
  baseline       — standard MCQ prompt
  relevant-hint  — inject explanation from most similar build-set question
  random-hint    — inject random explanation (controls for relevance)

Usage:
    source .env
    python scripts/gpqa_concept_experiment.py --condition baseline
    python scripts/gpqa_concept_experiment.py --condition relevant-hint
    python scripts/gpqa_concept_experiment.py --condition random-hint
    python scripts/gpqa_concept_experiment.py --compare
"""
import asyncio
import csv
import json
import os
import random
import re
import argparse
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
GPQA_PATH = BASE / "data" / "gpqa_diamond" / "gpqa_diamond.csv"
OUTPUT_DIR = BASE / "outputs" / "_runs" / "gpqa_concept"

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

HINT_TEMPLATE = """\
### Relevant Domain Knowledge
Here is an explanation from a related science problem that may contain useful concepts:

{explanation}

---

### Question
{question}

### Options
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

### Instructions
The domain knowledge above may help. Think through this step by step.
After your reasoning, state your final answer as:
ANSWER: X
where X is A, B, C, or D.
"""


def load_gpqa() -> list[dict]:
    questions = []
    with open(GPQA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            questions.append({
                "uid": f"gpqa_{i:03d}",
                "idx": i,
                "question": row["Question"],
                "correct": row["Correct Answer"],
                "incorrect_1": row["Incorrect Answer 1"],
                "incorrect_2": row["Incorrect Answer 2"],
                "incorrect_3": row["Incorrect Answer 3"],
                "explanation": row.get("Explanation", ""),
                "domain": row.get("High-level domain", ""),
                "subdomain": row.get("Subdomain", ""),
            })
    return questions


def split_build_eval(questions: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_domain = {}
    for q in questions:
        by_domain.setdefault(q["domain"], []).append(q)

    build, eval_set = [], []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        rng.shuffle(pool)
        mid = len(pool) // 2
        build.extend(pool[:mid])
        eval_set.extend(pool[mid:])

    return build, eval_set


def shuffle_options(q: dict, seed: int) -> tuple[dict, str]:
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
    match = re.search(r'ANSWER:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text)
    return matches[-1].upper() if matches else None


def build_retrieval_index(build_set: list[dict]):
    texts = [q["question"] for q in build_set]
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_similar(query_q: dict, build_set: list[dict], vectorizer, matrix) -> tuple[dict, float]:
    query_vec = vectorizer.transform([query_q["question"]])
    sims = cosine_similarity(query_vec, matrix)[0]
    # Prefer same domain
    for i, bq in enumerate(build_set):
        if bq["domain"] != query_q["domain"]:
            sims[i] *= 0.5  # penalize cross-domain
    best_idx = int(np.argmax(sims))
    return build_set[best_idx], float(sims[best_idx])


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
    eval_set: list[dict],
    build_set: list[dict],
    vectorizer, matrix,
    seed: int,
    output_path: Path,
):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(16)
    rng = random.Random(seed)

    tasks = []
    for q in eval_set:
        option_map, correct_letter = shuffle_options(q, seed=hash(q["uid"]) + seed)

        if condition == "relevant-hint":
            similar, sim_score = retrieve_similar(q, build_set, vectorizer, matrix)
            prompt = HINT_TEMPLATE.format(
                explanation=similar["explanation"],
                question=q["question"],
                option_a=option_map["A"],
                option_b=option_map["B"],
                option_c=option_map["C"],
                option_d=option_map["D"],
            )
            hint_uid = similar["uid"]
        elif condition == "random-hint":
            # Random hint from same domain when possible
            same_domain = [bq for bq in build_set if bq["domain"] == q["domain"]]
            pool = same_domain if same_domain else build_set
            similar = rng.choice(pool)
            sim_score = -1.0
            prompt = HINT_TEMPLATE.format(
                explanation=similar["explanation"],
                question=q["question"],
                option_a=option_map["A"],
                option_b=option_map["B"],
                option_c=option_map["C"],
                option_d=option_map["D"],
            )
            hint_uid = similar["uid"]
        else:
            sim_score = 0.0
            hint_uid = None
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
            "hint_uid": hint_uid,
            "sim_score": sim_score,
        })

    print(f"Running {condition} (seed={seed}) on {len(tasks)} questions...")
    completions = await asyncio.gather(
        *[generate(client, t["prompt"], sem) for t in tasks]
    )

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
            "condition": condition,
            "seed": seed,
            "correct_letter": correct_letter,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "hint_uid": task["hint_uid"],
            "sim_score": task["sim_score"],
        }
        results.append(result)

        status = "OK" if is_correct else "WRONG"
        print(f"  {q['uid']} [{q['domain'][:4]}]: {status} (correct={correct_letter}, model={model_answer})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    print(f"\n{condition.upper()} (seed={seed}): {correct}/{total} ({100*correct/total:.1f}%)")

    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)
    for domain in sorted(by_domain):
        dr = by_domain[domain]
        dc = sum(1 for r in dr if r["is_correct"])
        print(f"  {domain}: {dc}/{len(dr)} ({100*dc/len(dr):.0f}%)")

    return results


def compare_all():
    result_files = sorted(OUTPUT_DIR.glob("*_results.json"))
    all_data = {}
    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
        if data:
            cond = data[0]["condition"]
            all_data[cond] = data

    print("=" * 70)
    print("GPQA CONCEPT EXPERIMENT RESULTS")
    print("=" * 70)

    for cond in ["baseline", "random-hint", "relevant-hint"]:
        if cond in all_data:
            data = all_data[cond]
            correct = sum(1 for r in data if r["is_correct"])
            total = len(data)
            print(f"\n{cond}: {correct}/{total} ({100*correct/total:.1f}%)")
            by_domain = {}
            for r in data:
                by_domain.setdefault(r["domain"], []).append(r)
            for domain in sorted(by_domain):
                dr = by_domain[domain]
                dc = sum(1 for r in dr if r["is_correct"])
                print(f"  {domain}: {dc}/{len(dr)} ({100*dc/len(dr):.0f}%)")

    # Overlap analysis
    if "baseline" in all_data and "relevant-hint" in all_data:
        b = {r["uid"]: r["is_correct"] for r in all_data["baseline"]}
        h = {r["uid"]: r["is_correct"] for r in all_data["relevant-hint"]}
        both = sum(1 for uid in b if b.get(uid) and h.get(uid))
        only_b = sum(1 for uid in b if b.get(uid) and not h.get(uid))
        only_h = sum(1 for uid in b if not b.get(uid) and h.get(uid))
        neither = sum(1 for uid in b if not b.get(uid) and not h.get(uid))
        print(f"\nOverlap (baseline vs relevant-hint):")
        print(f"  Both correct: {both}")
        print(f"  Only baseline: {only_b}")
        print(f"  Only relevant-hint: {only_h}")
        print(f"  Neither: {neither}")

    if "baseline" in all_data and "random-hint" in all_data:
        b = {r["uid"]: r["is_correct"] for r in all_data["baseline"]}
        r_data = {r["uid"]: r["is_correct"] for r in all_data["random-hint"]}
        only_b = sum(1 for uid in b if b.get(uid) and not r_data.get(uid))
        only_r = sum(1 for uid in b if not b.get(uid) and r_data.get(uid))
        print(f"\nOverlap (baseline vs random-hint):")
        print(f"  Only baseline: {only_b}")
        print(f"  Only random-hint: {only_r}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "relevant-hint", "random-hint"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_all()
        return

    if not args.condition:
        parser.error("--condition required")

    print("Loading GPQA Diamond...")
    questions = load_gpqa()
    print(f"  {len(questions)} total questions")

    print("Splitting build/eval...")
    build_set, eval_set = split_build_eval(questions)
    print(f"  Build: {len(build_set)}, Eval: {len(eval_set)}")

    print("Building retrieval index...")
    vectorizer, matrix = build_retrieval_index(build_set)

    output_path = OUTPUT_DIR / f"{args.condition}_s{args.seed}_results.json"

    await run_condition(
        condition=args.condition,
        eval_set=eval_set,
        build_set=build_set,
        vectorizer=vectorizer,
        matrix=matrix,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
