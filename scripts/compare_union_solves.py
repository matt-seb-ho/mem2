"""Does memory unlock NEW solves? Compare union-of-solves (over each condition's 5
independent runs) between baseline and each memory method.

For each condition, union = puzzles solved test-correct (all test pairs) in ANY of its
5 runs (this is oracle@5 / pass@5 set). Then:
  - unlocked = solved by memory-union but NEVER by baseline (in any baseline run)
  - lost     = solved by baseline-union but NEVER by memory
  - shared   = in both unions
Test-correctness only.
"""
import json, glob

def all_ok(r):
    r = r or []
    return len(r) > 0 and all(x.get("correct") for x in r)

def union_solved(base):
    s = set()
    for suf in ["", "_rep2", "_rep3", "_rep4", "_rep5"]:
        rd = sorted(glob.glob(f"outputs/_runs/{base}{suf}/*/"))[-1].rstrip("/")
        for it in sorted(glob.glob(rd + "/iteration_*")):
            for uid, t in json.load(open(it + "/solution_trees.json")).items():
                if any(not x.get("parsing_error") and all_ok(x.get("test_results"))
                       for b in t["prompt_branches"].values()
                       for th in b["threads"].values() for x in th["steps"]):
                    s.add(uid)
    return s

CONDS = {
    "baseline": "eval100_baseline",
    "induced": "eval100_arcmemo",
    "reselect": "eval100_arcmemo_reselect",
    "paperlib": "eval100_paperlib",
}
U = {k: union_solved(v) for k, v in CONDS.items()}
base = U["baseline"]

print(f"baseline union (oracle@5): {len(base)} puzzles\n")
report = {"baseline_union_size": len(base), "baseline_union": sorted(base), "methods": {}}
for k in ["induced", "reselect", "paperlib"]:
    m = U[k]
    unlocked = sorted(m - base)   # memory solves, baseline never does
    lost = sorted(base - m)       # baseline solves, memory never does
    shared = m & base
    print(f"== {k} ==  union={len(m)}")
    print(f"  unlocked (memory-only, baseline never): {len(unlocked)} -> {unlocked}")
    print(f"  lost (baseline-only, memory never):     {len(lost)} -> {lost}")
    print(f"  shared: {len(shared)}   net = +{len(unlocked)} -{len(lost)} = {len(unlocked)-len(lost):+d}\n")
    report["methods"][k] = {
        "union_size": len(m), "unlocked": unlocked, "lost": lost,
        "shared": len(shared), "net": len(unlocked) - len(lost),
    }

# Combined: anything ANY memory method unlocks vs baseline
mem_all = U["induced"] | U["reselect"] | U["paperlib"]
unlocked_any = sorted(mem_all - base)
lost_all = sorted(base - mem_all)
print("== ANY memory method (induced ∪ reselect ∪ paperlib) vs baseline ==")
print(f"  union(any memory)={len(mem_all)}  baseline={len(base)}")
print(f"  unlocked by SOME memory method (baseline never): {len(unlocked_any)} -> {unlocked_any}")
print(f"  solved by baseline but NO memory method ever:    {len(lost_all)} -> {lost_all}")
report["any_memory"] = {
    "union_size": len(mem_all), "unlocked_by_some_memory": unlocked_any,
    "baseline_only_never_memory": lost_all,
}
json.dump(report, open("outputs/_runs/eval100_union_diff.json", "w"), indent=2)
print("\nwrote outputs/_runs/eval100_union_diff.json")
