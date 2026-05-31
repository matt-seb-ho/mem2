"""Oracle@2 (pairwise-ensemble ceiling) over the 5 independent eval samples.

For each condition: take all C(5,2)=10 pairs of runs, compute the oracle score of each
pair = number of eval_100 puzzles solved (test-correct, all test pairs) by EITHER run in
the pair, then average over the 10 pairs. Test-correctness only (strict). Also reports
single-run mean (oracle@1) and oracle@5 (union over all 5) for context.
"""
import json, glob, itertools, statistics as st

def all_ok(r):
    r = r or []
    return len(r) > 0 and all(x.get("correct") for x in r)

def strict_solved_set(run_dir):
    s = set()
    for it in sorted(glob.glob(run_dir + "/iteration_*")):
        for uid, t in json.load(open(it + "/solution_trees.json")).items():
            ok = any(
                not x.get("parsing_error") and all_ok(x.get("test_results"))
                for b in t["prompt_branches"].values()
                for th in b["threads"].values()
                for x in th["steps"]
            )
            if ok:
                s.add(uid)
    return s

FAM = {
    "baseline (no memory)": "eval100_baseline",
    "on-policy induced (55)": "eval100_arcmemo",
    "on-policy + reselection": "eval100_arcmemo_reselect",
    "paper lib (270, ref)": "eval100_paperlib",
}
SUFFIXES = ["", "_rep2", "_rep3", "_rep4", "_rep5"]

res = {}
for label, base in FAM.items():
    dirs = [sorted(glob.glob(f"outputs/_runs/{base}{s}/*/"))[-1].rstrip("/") for s in SUFFIXES]
    sets = [strict_solved_set(d) for d in dirs]
    singles = [len(s) for s in sets]
    summ = [json.load(open(d + "/summary.json"))["strict_solved_puzzles"] for d in dirs]
    assert singles == summ, (label, singles, summ)
    pairs = [len(sets[i] | sets[j]) for i, j in itertools.combinations(range(5), 2)]
    res[label] = {
        "oracle1_single_mean": round(st.mean(singles), 2),
        "oracle1_single_sd": round(st.stdev(singles), 2),
        "oracle2_pairs": pairs,
        "oracle2_mean": round(st.mean(pairs), 2),
        "oracle2_sd": round(st.stdev(pairs), 2),
        "oracle2_min": min(pairs),
        "oracle2_max": max(pairs),
        "oracle5_union": len(set().union(*sets)),
    }

json.dump(res, open("outputs/_runs/eval100_oracle2_stats.json", "w"), indent=2)

# compact human summary
with open("outputs/_runs/eval100_oracle2_summary.txt", "w") as f:
    f.write("condition | oracle@1 (single, mean±sd) | oracle@2 (pair mean±sd) [min,max] | oracle@5 (union)\n")
    for label, v in res.items():
        f.write(
            f"{label} | {v['oracle1_single_mean']}±{v['oracle1_single_sd']} | "
            f"{v['oracle2_mean']}±{v['oracle2_sd']} [{v['oracle2_min']},{v['oracle2_max']}] | "
            f"{v['oracle5_union']}\n"
        )
print("done")
