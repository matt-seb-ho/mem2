#!/usr/bin/env python3
"""Analyze why concept hints hurt certain math problems."""
import json
from pathlib import Path
from collections import defaultdict

# ─── Config ────────────────────────────────────────────────────────────
BASELINE_EVAL = Path("/root/arc/mem2/outputs/_runs/baseline_math_eval/d6436faba6c1/eval_records.jsonl")
CONCEPT_EVAL  = Path("/root/arc/mem2/outputs/_runs/concept_math_eval/980bd5b0ad59/eval_records.jsonl")
OPT2_EVAL     = Path("/root/arc/mem2/outputs/_runs/concept_math_opt2_cues/9f411021d209/eval_records.jsonl")
COMPOSED_EVAL = Path("/root/arc/mem2/outputs/_runs/concept_math_opt123_composed/79cf3a41fc74/eval_records.jsonl")

SEL_DIR = Path("/root/arc/mem2/data/competition_math_nt_cp_l5/concept_memory/selection_v1")
SELECTED_CONCEPTS = SEL_DIR / "selected_concepts.json"
PROMPT_INFO       = SEL_DIR / "prompt_info.json"
CONCEPT_FREQ      = SEL_DIR / "concept_frequencies.json"

# Router thresholds (composite router)
MAX_CONCEPT_COUNT = 4
MAX_HINT_CHARS    = 4000

# ─── Helpers ───────────────────────────────────────────────────────────
def load_eval(path: Path) -> dict[str, bool]:
    """Return {problem_uid: solved} where solved = any attempt correct across all passes."""
    solved = defaultdict(bool)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["is_correct"]:
                solved[rec["problem_uid"]] = True
            elif rec["problem_uid"] not in solved:
                solved[rec["problem_uid"]] = False
    return dict(solved)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ─── Load data ─────────────────────────────────────────────────────────
baseline = load_eval(BASELINE_EVAL)
concept  = load_eval(CONCEPT_EVAL)
opt2     = load_eval(OPT2_EVAL)
composed = load_eval(COMPOSED_EVAL)

selected = load_json(SELECTED_CONCEPTS)   # {uid: [concept_name, ...]}
prompt   = load_json(PROMPT_INFO)          # {uid: {hint: str, ...}}
freq     = load_json(CONCEPT_FREQ)         # {concept_name: float}

# ─── Classify problems ────────────────────────────────────────────────
all_uids = sorted(set(baseline) | set(concept))
hurt   = [u for u in all_uids if baseline.get(u) and not concept.get(u)]
helped = [u for u in all_uids if not baseline.get(u) and concept.get(u)]
both   = [u for u in all_uids if baseline.get(u) and concept.get(u)]
neither= [u for u in all_uids if not baseline.get(u) and not concept.get(u)]

print("=" * 100)
print("CONCEPT HINTS IMPACT ANALYSIS: baseline vs concept_math_eval")
print("=" * 100)
print(f"Total problems:    {len(all_uids)}")
print(f"Baseline solved:   {sum(1 for u in all_uids if baseline.get(u))}")
print(f"Concept solved:    {sum(1 for u in all_uids if concept.get(u))}")
print(f"Both solved:       {len(both)}")
print(f"Hurt (B yes, C no):{len(hurt)}")
print(f"Helped (B no, C yes):{len(helped)}")
print(f"Neither solved:    {len(neither)}")
print()

# ─── Build detail rows ────────────────────────────────────────────────
def problem_detail(uid):
    concepts = selected.get(uid, [])
    n_concepts = len(concepts)
    hint_text = prompt.get(uid, {}).get("hint", "")
    hint_len = len(hint_text)
    freqs = [freq.get(c, 0.0) for c in concepts]
    avg_freq = sum(freqs) / len(freqs) if freqs else 0.0
    max_freq = max(freqs) if freqs else 0.0
    # Would the composite router gate this problem?
    gated = n_concepts > MAX_CONCEPT_COUNT or hint_len > MAX_HINT_CHARS
    return {
        "uid": uid,
        "n_concepts": n_concepts,
        "hint_len": hint_len,
        "avg_freq": avg_freq,
        "max_freq": max_freq,
        "gated": gated,
        "concepts": concepts,
        "opt2_solved": opt2.get(uid, False),
        "composed_solved": composed.get(uid, False),
    }

hurt_details   = [problem_detail(u) for u in hurt]
helped_details = [problem_detail(u) for u in helped]

# ─── Print tables ─────────────────────────────────────────────────────
def print_table(title, rows):
    print(f"\n{'─' * 100}")
    print(f"  {title}  ({len(rows)} problems)")
    print(f"{'─' * 100}")
    hdr = f"{'problem_uid':<16} {'#concepts':>9} {'hint_len':>9} {'avg_freq':>9} {'max_freq':>9} {'gated':>6} {'opt2':>5} {'comp':>5}  concepts"
    print(hdr)
    print("-" * len(hdr) + "-" * 40)
    for r in sorted(rows, key=lambda x: -x["hint_len"]):
        concepts_str = ", ".join(r["concepts"][:5])
        if len(r["concepts"]) > 5:
            concepts_str += f" ... (+{len(r['concepts'])-5})"
        print(f"{r['uid']:<16} {r['n_concepts']:>9} {r['hint_len']:>9} {r['avg_freq']:>9.3f} {r['max_freq']:>9.3f} {str(r['gated']):>6} {str(r['opt2_solved']):>5} {str(r['composed_solved']):>5}  {concepts_str}")

print_table("HURT PROBLEMS (baseline solved, concept failed)", hurt_details)
print_table("HELPED PROBLEMS (baseline failed, concept solved)", helped_details)

# ─── Summary statistics ───────────────────────────────────────────────
def summarize(label, rows):
    if not rows:
        print(f"\n  {label}: no problems")
        return
    n_concepts = [r["n_concepts"] for r in rows]
    hint_lens  = [r["hint_len"] for r in rows]
    avg_freqs  = [r["avg_freq"] for r in rows]
    n_gated    = sum(1 for r in rows if r["gated"])
    n_opt2     = sum(1 for r in rows if r["opt2_solved"])
    n_composed = sum(1 for r in rows if r["composed_solved"])
    print(f"\n  {label} ({len(rows)} problems):")
    print(f"    # concepts    — mean: {sum(n_concepts)/len(n_concepts):.1f}, min: {min(n_concepts)}, max: {max(n_concepts)}")
    print(f"    hint_len      — mean: {sum(hint_lens)/len(hint_lens):.0f}, min: {min(hint_lens)}, max: {max(hint_lens)}")
    print(f"    avg_freq      — mean: {sum(avg_freqs)/len(avg_freqs):.3f}")
    print(f"    gated by router (count>{MAX_CONCEPT_COUNT} or chars>{MAX_HINT_CHARS}): {n_gated}/{len(rows)} ({100*n_gated/len(rows):.0f}%)")
    print(f"    solved by opt2_cues:      {n_opt2}/{len(rows)} ({100*n_opt2/len(rows):.0f}%)")
    print(f"    solved by opt123_composed: {n_composed}/{len(rows)} ({100*n_composed/len(rows):.0f}%)")

print(f"\n{'=' * 100}")
print("SUMMARY STATISTICS")
print(f"{'=' * 100}")
summarize("HURT", hurt_details)
summarize("HELPED", helped_details)

# ─── Composed run analysis ─────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("AVAILABLE RUN IDS")
print(f"{'=' * 100}")
for name in ["concept_math_opt2_cues", "concept_math_opt123_composed"]:
    p = Path(f"/root/arc/mem2/outputs/_runs/{name}")
    if p.exists():
        run_ids = sorted(p.iterdir())
        print(f"  {name}/: {[r.name for r in run_ids]}")

# ─── Cross-run comparison ──────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("CROSS-RUN SOLVE RATES")
print(f"{'=' * 100}")
runs = {
    "baseline":      baseline,
    "concept_eval":  concept,
    "opt2_cues":     opt2,
    "opt123_composed": composed,
}
for name, data in runs.items():
    n = sum(1 for v in data.values() if v)
    print(f"  {name:<20}: {n}/{len(data)} solved ({100*n/len(data):.1f}%)")

# ─── Hurt problems recovered by composed? ─────────────────────────────
print(f"\n{'=' * 100}")
print("RECOVERY ANALYSIS: Did the composed/opt2 runs recover the hurt problems?")
print(f"{'=' * 100}")
for r in sorted(hurt_details, key=lambda x: x["uid"]):
    status = []
    if r["opt2_solved"]:
        status.append("opt2_RECOVERED")
    if r["composed_solved"]:
        status.append("composed_RECOVERED")
    if not status:
        status.append("STILL_FAILING")
    print(f"  {r['uid']:<16} gated={str(r['gated']):<6} {', '.join(status)}")

# ─── Gating effectiveness ──────────────────────────────────────────────
print(f"\n{'=' * 100}")
print(f"GATING EFFECTIVENESS (max_concept_count={MAX_CONCEPT_COUNT}, max_hint_chars={MAX_HINT_CHARS})")
print(f"{'=' * 100}")
all_details = [problem_detail(u) for u in all_uids]
gated_probs = [r for r in all_details if r["gated"]]
ungated_probs = [r for r in all_details if not r["gated"]]
print(f"  Total gated: {len(gated_probs)}/{len(all_details)}")
gated_hurt = [u for u in hurt if problem_detail(u)["gated"]]
gated_helped = [u for u in helped if problem_detail(u)["gated"]]
print(f"  Gated hurt problems: {len(gated_hurt)}/{len(hurt)} (would revert to baseline -> recovered)")
print(f"  Gated helped problems: {len(gated_helped)}/{len(helped)} (would lose the help)")
net = len(gated_hurt) - len(gated_helped)
print(f"  Net effect of gating: {'+' if net > 0 else ''}{net} problems")

