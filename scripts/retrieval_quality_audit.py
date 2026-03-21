"""
Retrieval Quality Audit for I05 (Episodic Memory)

Tests whether embedding-based retrieval can find structurally similar
math problems. Uses TF-IDF as a cheap baseline — if surface similarity
can't find useful matches, neural embeddings are unlikely to either.

Usage:
    python scripts/retrieval_quality_audit.py [--n-queries 30] [--top-k 3]
"""
import json
import random
import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    omni_path = base / "data" / "omni_math" / "problems.jsonl"
    math_l5_path = base / "data" / "competition_math_all_l5" / "problems.jsonl"

    print("Loading Omni-MATH...")
    omni = load_problems(omni_path)
    print(f"  {len(omni)} problems")

    print("Loading Math L5 (retrieval bank)...")
    math_l5 = load_problems(math_l5_path)
    print(f"  {len(math_l5)} problems")

    # Sample query problems from Omni-MATH, stratified by difficulty
    rng = random.Random(args.seed)
    # Get problems at difficulty levels 3-7 (the interesting range)
    by_diff = {}
    for p in omni:
        d = extract_difficulty(p)
        if d is not None and 3 <= d <= 7:
            by_diff.setdefault(int(d), []).append(p)

    queries = []
    for d in sorted(by_diff.keys()):
        sample = rng.sample(by_diff[d], min(6, len(by_diff[d])))
        queries.extend(sample)
    queries = queries[:args.n_queries]
    print(f"\nSelected {len(queries)} query problems (d3-d7)")

    # Build TF-IDF over the bank (Math L5) + queries
    bank_texts = [p["problem"] for p in math_l5]
    query_texts = [p["problem"] for p in queries]

    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    all_texts = bank_texts + query_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    bank_matrix = tfidf_matrix[:len(bank_texts)]
    query_matrix = tfidf_matrix[len(bank_texts):]

    print("Computing similarities...")
    sims = cosine_similarity(query_matrix, bank_matrix)

    # Also compute Omni-MATH self-retrieval (leave-one-out)
    print("\nAlso computing Omni-MATH self-retrieval...")
    omni_texts = [p["problem"] for p in omni]
    omni_matrix = vectorizer.transform(omni_texts)
    omni_sims = cosine_similarity(
        vectorizer.transform(query_texts), omni_matrix
    )

    # Report
    print("\n" + "=" * 80)
    print("RETRIEVAL QUALITY AUDIT")
    print("=" * 80)

    results = []
    for i, q in enumerate(queries):
        q_diff = extract_difficulty(q)
        q_type = q.get("source", q.get("type", "unknown"))

        # Math L5 retrieval
        top_indices = np.argsort(sims[i])[::-1][:args.top_k]
        top_scores = sims[i][top_indices]

        # Omni-MATH self-retrieval (skip self)
        omni_scores_i = omni_sims[i].copy()
        # Find self and zero it out
        for j, op in enumerate(omni):
            if op["uid"] == q["uid"]:
                omni_scores_i[j] = -1
        omni_top_indices = np.argsort(omni_scores_i)[::-1][:args.top_k]
        omni_top_scores = omni_scores_i[omni_top_indices]

        print(f"\n{'─' * 80}")
        print(f"QUERY [{q['uid']}] (difficulty {q_diff}, {q_type})")
        print(f"  {q['problem'][:200]}...")
        print(f"\n  Top-{args.top_k} from Math L5 bank:")
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
            mp = math_l5[idx]
            print(f"    [{rank+1}] sim={score:.3f} type={mp.get('type','?')} | {mp['problem'][:150]}...")

        print(f"\n  Top-{args.top_k} from Omni-MATH (self-retrieval, leave-one-out):")
        for rank, (idx, score) in enumerate(zip(omni_top_indices, omni_top_scores)):
            op = omni[idx]
            op_diff = extract_difficulty(op)
            print(f"    [{rank+1}] sim={score:.3f} d={op_diff} | {op['problem'][:150]}...")

        results.append({
            "query_uid": q["uid"],
            "query_diff": q_diff,
            "math_l5_top1_sim": float(top_scores[0]),
            "omni_self_top1_sim": float(omni_top_scores[0]),
        })

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    l5_sims = [r["math_l5_top1_sim"] for r in results]
    omni_sims_list = [r["omni_self_top1_sim"] for r in results]
    print(f"Math L5 retrieval — top-1 similarity: mean={np.mean(l5_sims):.3f}, "
          f"median={np.median(l5_sims):.3f}, min={np.min(l5_sims):.3f}, max={np.max(l5_sims):.3f}")
    print(f"Omni self-retrieval — top-1 similarity: mean={np.mean(omni_sims_list):.3f}, "
          f"median={np.median(omni_sims_list):.3f}, min={np.min(omni_sims_list):.3f}, max={np.max(omni_sims_list):.3f}")
    print(f"\nInterpretation guide:")
    print(f"  sim > 0.3: likely topically related (same math domain)")
    print(f"  sim > 0.5: strong surface similarity (may share problem structure)")
    print(f"  sim < 0.1: essentially unrelated")
    print(f"\nKey question: Do high-similarity matches share solution APPROACH,")
    print(f"not just topic? Inspect the pairs above to determine this.")

    # Save for later analysis
    out_path = base / "data" / "omni_math" / "retrieval_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
