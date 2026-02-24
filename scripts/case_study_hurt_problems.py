#!/usr/bin/env python3
"""
Deep case study: Math problems where concepts consistently hurt.

For each of the 3 persistently hurt problems (solved by baseline, failed by ALL
concept variants): cmath_2490, cmath_5307, cmath_5439

Extracts and compares:
  - Selected concept names and hint text
  - Baseline successful solution code
  - Concept run's failed solution code
  - Diff summary of what changed
"""

import json
import difflib
import re
import textwrap
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASELINE_ATTEMPTS = Path("/root/arc/mem2/outputs/_runs/baseline_math_eval/d6436faba6c1/attempts.jsonl")
BASELINE_EVALS    = Path("/root/arc/mem2/outputs/_runs/baseline_math_eval/d6436faba6c1/eval_records.jsonl")
CONCEPT_ATTEMPTS  = Path("/root/arc/mem2/outputs/_runs/concept_math_eval/980bd5b0ad59/attempts.jsonl")
CONCEPT_EVALS     = Path("/root/arc/mem2/outputs/_runs/concept_math_eval/980bd5b0ad59/eval_records.jsonl")
PROMPT_INFO       = Path("/root/arc/mem2/data/competition_math_nt_cp_l5/concept_memory/selection_v1/prompt_info.json")
SELECTED_CONCEPTS = Path("/root/arc/mem2/data/competition_math_nt_cp_l5/concept_memory/selection_v1/selected_concepts.json")

TARGET_UIDS = ["cmath_2490", "cmath_5307", "cmath_5439"]

SEPARATOR = "=" * 100
SUB_SEP   = "-" * 80


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_code_from_completion(completion_text):
    """Extract the last Python code block from a completion string."""
    # Find all ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, completion_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback: look for ```...``` blocks
    pattern2 = r'```\s*\n(.*?)```'
    matches2 = re.findall(pattern2, completion_text, re.DOTALL)
    if matches2:
        return matches2[-1].strip()
    return "(no code block found in completion)"


def find_correct_attempt(attempts, evals, uid):
    """Find the attempt that was evaluated as correct for a given problem."""
    correct_attempts = set()
    for ev in evals:
        if ev["problem_uid"] == uid and ev["is_correct"]:
            correct_attempts.add(ev["attempt_idx"])

    for att in attempts:
        if att["problem_uid"] == uid and att["pass_idx"] in correct_attempts:
            return att
    return None


def find_failed_attempts(attempts, evals, uid):
    """Find all failed attempts for a given problem."""
    eval_map = {}
    for ev in evals:
        if ev["problem_uid"] == uid:
            eval_map[ev["attempt_idx"]] = ev

    result = []
    for att in attempts:
        if att["problem_uid"] == uid:
            ev = eval_map.get(att["pass_idx"])
            if ev and not ev["is_correct"]:
                result.append((att, ev))
    return result


def make_diff(baseline_code, concept_code):
    """Create a unified diff between baseline and concept code."""
    baseline_lines = baseline_code.splitlines(keepends=True)
    concept_lines = concept_code.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        baseline_lines, concept_lines,
        fromfile="baseline (correct)", tofile="concept (failed)",
        lineterm=""
    ))
    return "\n".join(diff) if diff else "(no diff -- code is identical)"


def summarize_diff(baseline_code, concept_code):
    """Produce a human-readable summary of what changed."""
    baseline_lines = set(baseline_code.splitlines())
    concept_lines = set(concept_code.splitlines())
    removed = baseline_lines - concept_lines
    added = concept_lines - baseline_lines

    summary_parts = []
    if removed:
        summary_parts.append(f"  Lines only in BASELINE ({len(removed)}):")
        for line in sorted(removed):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                summary_parts.append(f"    - {stripped[:120]}")
    if added:
        summary_parts.append(f"  Lines only in CONCEPT ({len(added)}):")
        for line in sorted(added):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                summary_parts.append(f"    + {stripped[:120]}")

    return "\n".join(summary_parts) if summary_parts else "  (code is identical)"


def extract_problem_statement(attempt):
    """Extract the problem statement from the prompt field."""
    prompt = attempt.get("prompt", "")
    match = re.search(r'### Problem\s*\n(.*?)### Instructions', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "(could not extract problem statement)"


def main():
    # Load data
    baseline_attempts = load_jsonl(BASELINE_ATTEMPTS)
    baseline_evals = load_jsonl(BASELINE_EVALS)
    concept_attempts = load_jsonl(CONCEPT_ATTEMPTS)
    concept_evals = load_jsonl(CONCEPT_EVALS)
    prompt_info = json.loads(PROMPT_INFO.read_text())
    selected_concepts = json.loads(SELECTED_CONCEPTS.read_text())

    print(SEPARATOR)
    print("DEEP CASE STUDY: PROBLEMS WHERE CONCEPTS CONSISTENTLY HURT")
    print("Baseline: solved | All concept variants: FAILED")
    print(SEPARATOR)
    print()

    for uid in TARGET_UIDS:
        print(SEPARATOR)
        print(f"  PROBLEM: {uid}")
        print(SEPARATOR)

        # ── 1. Problem Statement ──
        baseline_correct = find_correct_attempt(baseline_attempts, baseline_evals, uid)
        if baseline_correct:
            problem_stmt = extract_problem_statement(baseline_correct)
            print(f"\n{SUB_SEP}")
            print(f"  PROBLEM STATEMENT")
            print(f"{SUB_SEP}")
            print(textwrap.fill(problem_stmt, width=95, initial_indent="  ", subsequent_indent="  "))

        # ── 2. Selected Concepts ──
        concepts = selected_concepts.get(uid, [])
        print(f"\n{SUB_SEP}")
        print(f"  SELECTED CONCEPTS ({len(concepts)})")
        print(f"{SUB_SEP}")
        for i, c in enumerate(concepts, 1):
            print(f"  {i}. {c}")

        # ── 3. Hint Text ──
        hint_data = prompt_info.get(uid, {})
        hint_text = hint_data.get("hint", "(no hint found)")
        print(f"\n{SUB_SEP}")
        print(f"  INJECTED HINT TEXT (first 500 chars of {len(hint_text)} total)")
        print(f"{SUB_SEP}")
        print(textwrap.fill(hint_text[:500], width=95, initial_indent="  ", subsequent_indent="  "))
        if len(hint_text) > 500:
            print(f"  ... [{len(hint_text) - 500} more chars]")

        # ── 4. Baseline Successful Solution ──
        print(f"\n{SUB_SEP}")
        print(f"  BASELINE SUCCESSFUL SOLUTION CODE")
        print(f"{SUB_SEP}")
        if baseline_correct:
            baseline_code = extract_code_from_completion(baseline_correct["completion"])
            for ev in baseline_evals:
                if ev["problem_uid"] == uid and ev["is_correct"]:
                    test_details = ev.get("test_details", [])
                    if test_details:
                        output = test_details[0].get("output", "?")
                        expected = test_details[0].get("expected", "?")
                        print(f"  [EVAL] output={output}, expected={expected}, CORRECT")
            print()
            for line in baseline_code.splitlines():
                print(f"  {line}")
        else:
            baseline_code = "(no correct attempt found)"
            print(f"  {baseline_code}")

        # ── 5. Concept Run Failed Solutions ──
        failed_attempts = find_failed_attempts(concept_attempts, concept_evals, uid)
        print(f"\n{SUB_SEP}")
        print(f"  CONCEPT RUN FAILED SOLUTIONS ({len(failed_attempts)} attempts)")
        print(f"{SUB_SEP}")
        concept_codes = []
        for i, (att, ev) in enumerate(failed_attempts):
            test_details = ev.get("test_details", [])
            output = test_details[0].get("output", "?") if test_details else "?"
            expected = test_details[0].get("expected", "?") if test_details else "?"
            error = test_details[0].get("error", None) if test_details else None
            print(f"\n  --- Attempt {i} (pass_idx={att['pass_idx']}) ---")
            print(f"  [EVAL] output={output}, expected={expected}, FAILED")
            if error:
                print(f"  [ERROR] {str(error)[:200]}")
            concept_code = extract_code_from_completion(att["completion"])
            concept_codes.append(concept_code)
            print()
            for line in concept_code.splitlines():
                print(f"  {line}")

        # ── 6. Diff Summary ──
        if baseline_correct and concept_codes:
            primary_concept_code = concept_codes[0]
            print(f"\n{SUB_SEP}")
            print(f"  UNIFIED DIFF (baseline correct vs concept attempt 0 failed)")
            print(f"{SUB_SEP}")
            diff_text = make_diff(baseline_code, primary_concept_code)
            for line in diff_text.splitlines():
                print(f"  {line}")

            print(f"\n{SUB_SEP}")
            print(f"  CHANGE SUMMARY")
            print(f"{SUB_SEP}")
            summary = summarize_diff(baseline_code, primary_concept_code)
            print(summary)

            if len(concept_codes) > 1:
                print(f"\n{SUB_SEP}")
                print(f"  UNIFIED DIFF (baseline correct vs concept attempt 1 failed)")
                print(f"{SUB_SEP}")
                diff_text2 = make_diff(baseline_code, concept_codes[1])
                for line in diff_text2.splitlines():
                    print(f"  {line}")

                print(f"\n{SUB_SEP}")
                print(f"  CHANGE SUMMARY (attempt 1)")
                print(f"{SUB_SEP}")
                summary2 = summarize_diff(baseline_code, concept_codes[1])
                print(summary2)

        # ── 7. Mechanism Analysis ──
        print(f"\n{SUB_SEP}")
        print(f"  MECHANISM ANALYSIS")
        print(f"{SUB_SEP}")

        if baseline_correct and failed_attempts:
            baseline_prompt = baseline_correct.get("prompt", "")
            concept_prompt = failed_attempts[0][0].get("prompt", "")
            if "Hints" in concept_prompt and "Hints" not in baseline_prompt:
                hints_match = re.search(r'(### Hints.*?)$', concept_prompt, re.DOTALL)
                if hints_match:
                    hints_section = hints_match.group(1)
                    print(f"  Concept prompt has a '### Hints' section ({len(hints_section)} chars)")
                    print(f"  Baseline prompt does NOT have a hints section")
                    print()
                    print(f"  Full hints section injected into prompt:")
                    print(f"  {'~' * 60}")
                    for line in hints_section.splitlines()[:40]:
                        print(f"  {line}")
                    if len(hints_section.splitlines()) > 40:
                        print(f"  ... [{len(hints_section.splitlines()) - 40} more lines]")
                    print(f"  {'~' * 60}")
            else:
                print(f"  Both prompts have similar structure")

        print()
        print()

    # ── Final Summary ──
    print(SEPARATOR)
    print("  CROSS-PROBLEM SUMMARY")
    print(SEPARATOR)
    print()
    print("  Problem        | # Concepts | Hint Size | Baseline Output | Concept Output(s)")
    print("  " + "-" * 85)
    for uid in TARGET_UIDS:
        concepts = selected_concepts.get(uid, [])
        hint_text = prompt_info.get(uid, {}).get("hint", "")
        bl_output = "?"
        for ev in baseline_evals:
            if ev["problem_uid"] == uid and ev["is_correct"]:
                td = ev.get("test_details", [])
                if td:
                    bl_output = str(td[0].get("output", "?"))
        c_outputs = []
        for ev in concept_evals:
            if ev["problem_uid"] == uid:
                td = ev.get("test_details", [])
                if td:
                    out = td[0].get("output", "?")
                    err = td[0].get("error", None)
                    if err:
                        c_outputs.append(f"ERR")
                    else:
                        c_outputs.append(str(out))
        c_str = ", ".join(c_outputs) if c_outputs else "?"
        print(f"  {uid:<16}| {len(concepts):<10} | {len(hint_text):<9} | {bl_output:<15} | {c_str}")

    print()
    print(SEPARATOR)
    print("  END OF CASE STUDY")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
