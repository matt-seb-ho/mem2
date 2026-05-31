"""How do scores change across retry DEPTH? (baseline vs memory)

Each eval run has iteration_1/2/3 = pass 1 (initial), passes 2-3 (retries).
A puzzle is "solved by depth k" if ANY pass <= k has a test-correct step (all test
pairs correct) — i.e. cumulative solved as retries accrue. We compute this per run,
then average over the 5 independent samples per condition (mean ± sd).

depth1 = pass-1 (first attempt only); depth3 = full run (== strict_solved_puzzles).
"""
import json, glob, statistics as st

def all_ok(r):
    r = r or []
    return len(r) > 0 and all(x.get("correct") for x in r)

def solved_by_depth(run_dir, max_depth=3):
    """Return dict k -> set(puzzles solved in any pass <= k)."""
    per_pass = {}  # pass_idx -> set solved at exactly that pass
    for it in sorted(glob.glob(run_dir + "/iteration_*")):
        k = int(it.split("iteration_")[1])
        s = set()
        for uid, t in json.load(open(it + "/solution_trees.json")).items():
            if any(not x.get("parsing_error") and all_ok(x.get("test_results"))
                   for b in t["prompt_branches"].values()
                   for th in b["threads"].values() for x in th["steps"]):
                s.add(uid)
        per_pass[k] = s
    cum = {}
    acc = set()
    for k in range(1, max_depth + 1):
        acc = acc | per_pass.get(k, set())
        cum[k] = set(acc)
    return cum

CONDS = {
    "baseline (no memory)": "eval100_baseline",
    "on-policy induced (55)": "eval100_arcmemo",
    "on-policy + reselection": "eval100_arcmemo_reselect",
    "paper lib (270, ref)": "eval100_paperlib",
}
SUFFIXES = ["", "_rep2", "_rep3", "_rep4", "_rep5"]

report = {}
for label, base in CONDS.items():
    counts_by_depth = {1: [], 2: [], 3: []}
    for suf in SUFFIXES:
        rd = sorted(glob.glob(f"outputs/_runs/{base}{suf}/*/"))[-1].rstrip("/")
        cum = solved_by_depth(rd)
        # sanity: depth-3 cumulative == reported strict_solved
        strict = json.load(open(rd + "/summary.json"))["strict_solved_puzzles"]
        assert len(cum[3]) == strict, (rd, len(cum[3]), strict)
        for k in (1, 2, 3):
            counts_by_depth[k].append(len(cum[k]))
    report[label] = {
        f"depth{k}": {
            "samples": counts_by_depth[k],
            "mean": round(st.mean(counts_by_depth[k]), 2),
            "sd": round(st.stdev(counts_by_depth[k]), 2),
        } for k in (1, 2, 3)
    }
    # marginal gains per added retry (means)
    m1, m2, m3 = (report[label][f"depth{k}"]["mean"] for k in (1, 2, 3))
    report[label]["gain_pass2"] = round(m2 - m1, 2)
    report[label]["gain_pass3"] = round(m3 - m2, 2)

json.dump(report, open("outputs/_runs/eval100_depth_curve.json", "w"), indent=2)

print(f"{'condition':26s} | depth1 (pass-1) | depth2 | depth3 (full) | +p2  +p3")
for label, r in report.items():
    d1, d2, d3 = r["depth1"], r["depth2"], r["depth3"]
    print(f"{label:26s} | {d1['mean']:5.1f}±{d1['sd']:.2f}    | "
          f"{d2['mean']:5.1f}±{d2['sd']:.2f} | {d3['mean']:5.1f}±{d3['sd']:.2f}  | "
          f"+{r['gain_pass2']:.1f} +{r['gain_pass3']:.1f}")
print("\nwrote outputs/_runs/eval100_depth_curve.json")
