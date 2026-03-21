"""
BFCL-V4 Concept Memory Pilot (I08)

Tests whether concept memory improves function calling performance.
Phase 1: Baseline — establish Flash accuracy on exec_simple + exec_multiple.

Usage:
    source .env
    python scripts/bfcl_pilot.py --condition baseline
    python scripts/bfcl_pilot.py --condition concept
    python scripts/bfcl_pilot.py --compare
"""
import ast
import asyncio
import json
import os
import re
import argparse
from pathlib import Path

from openai import AsyncOpenAI

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "bfcl_v4"
OUTPUT_DIR = BASE / "outputs" / "_runs" / "bfcl_pilot"

MODEL = "qwen/qwen3.5-flash-02-23"

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates function calls based on user queries. "
    "You have access to the following functions. Generate the appropriate function call(s) "
    "to answer the user's query. Output ONLY the function call(s), nothing else."
)

BASELINE_TEMPLATE = """\
### Available Functions
{function_defs}

### User Query
{query}

### Instructions
Generate the correct function call to answer the user's query.
- Use the exact function name and parameter names from the definitions above.
- Output the function call in Python syntax: `function_name(param1=value1, param2=value2)`
- If multiple function calls are needed, output each on a separate line.
- Output ONLY the function call(s), no explanations.
"""

CONCEPT_TEMPLATE = """\
### API Usage Patterns
Here are some patterns for using similar APIs effectively:
{concepts}

---

### Available Functions
{function_defs}

### User Query
{query}

### Instructions
Generate the correct function call to answer the user's query.
- Use the exact function name and parameter names from the definitions above.
- Output the function call in Python syntax: `function_name(param1=value1, param2=value2)`
- If multiple function calls are needed, output each on a separate line.
- Output ONLY the function call(s), no explanations.
"""


def load_bfcl_problems(splits: list[str] | None = None) -> list[dict]:
    """Load BFCL problems from JSONL files."""
    if splits is None:
        splits = ["BFCL_v3_exec_simple.json", "BFCL_v3_exec_multiple.json"]

    problems = []
    for split_file in splits:
        path = DATA_DIR / split_file
        if not path.exists():
            print(f"  Warning: {split_file} not found")
            continue
        with open(path) as f:
            for line in f:
                prob = json.loads(line.strip())
                prob["_split"] = split_file.replace(".json", "")
                problems.append(prob)
        print(f"  {split_file}: {sum(1 for p in problems if p['_split'] == prob['_split'])} problems")

    return problems


def format_function_defs(functions: list[dict]) -> str:
    """Format function definitions for the prompt."""
    parts = []
    for fn in functions:
        sig_parts = []
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        for pname, pdef in params.items():
            ptype = pdef.get("type", "any")
            pdesc = pdef.get("description", "")
            req = "(required)" if pname in required else "(optional)"
            sig_parts.append(f"  - {pname}: {ptype} {req} — {pdesc}")

        parts.append(
            f"**{fn['name']}**: {fn.get('description', '')}\n"
            f"Parameters:\n" + "\n".join(sig_parts)
        )
    return "\n\n".join(parts)


def extract_query(question: list) -> str:
    """Extract user query from BFCL nested question format."""
    # question is [[{role, content}, ...], ...]
    messages = []
    for turn in question:
        if isinstance(turn, list):
            for msg in turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    messages.append(msg["content"])
        elif isinstance(msg, dict) and msg.get("role") == "user":
            messages.append(msg["content"])
    return "\n".join(messages) if messages else str(question)


def parse_function_call(text: str) -> list[str]:
    """Extract function call strings from model output."""
    # Try to find function_name(...) patterns
    pattern = r'(\w[\w.]*\([^)]*\))'
    matches = re.findall(pattern, text)
    if matches:
        return matches

    # Fallback: try to find code blocks
    code_pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
    code_matches = re.findall(code_pattern, text, re.DOTALL)
    if code_matches:
        calls = []
        for block in code_matches:
            calls.extend(re.findall(pattern, block))
        return calls

    return [text.strip()]


def normalize_call(call_str: str) -> tuple[str, dict] | None:
    """Parse a function call string into (name, {param: value})."""
    try:
        # Try parsing as Python AST
        tree = ast.parse(call_str, mode='eval')
        if isinstance(tree.body, ast.Call):
            # Get function name
            if isinstance(tree.body.func, ast.Attribute):
                name = f"{ast.dump(tree.body.func.value)}.{tree.body.func.attr}"
                # Simplified: just use the attribute chain
                parts = []
                node = tree.body.func
                while isinstance(node, ast.Attribute):
                    parts.append(node.attr)
                    node = node.value
                if isinstance(node, ast.Name):
                    parts.append(node.id)
                name = ".".join(reversed(parts))
            elif isinstance(tree.body.func, ast.Name):
                name = tree.body.func.id
            else:
                return None

            params = {}
            for kw in tree.body.keywords:
                try:
                    params[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    params[kw.arg] = ast.dump(kw.value)

            return (name, params)
    except (SyntaxError, ValueError):
        pass
    return None


def evaluate_call(model_output: str, ground_truth: list[str]) -> dict:
    """Evaluate model's function call against ground truth."""
    model_calls = parse_function_call(model_output)

    gt_parsed = []
    for gt in ground_truth:
        parsed = normalize_call(gt)
        if parsed:
            gt_parsed.append(parsed)

    model_parsed = []
    for mc in model_calls:
        parsed = normalize_call(mc)
        if parsed:
            model_parsed.append(parsed)

    if not gt_parsed:
        return {"correct": False, "error": "could_not_parse_gt"}
    if not model_parsed:
        return {"correct": False, "error": "could_not_parse_model"}

    # Check if all ground truth calls are matched
    matched = 0
    for gt_name, gt_params in gt_parsed:
        for m_name, m_params in model_parsed:
            if gt_name == m_name:
                # Check params match
                if _params_match(gt_params, m_params):
                    matched += 1
                    break

    all_correct = matched == len(gt_parsed)
    return {
        "correct": all_correct,
        "matched": matched,
        "total_gt": len(gt_parsed),
        "total_model": len(model_parsed),
        "gt_names": [n for n, _ in gt_parsed],
        "model_names": [n for n, _ in model_parsed],
    }


def _params_match(gt_params: dict, model_params: dict) -> bool:
    """Check if model params match ground truth (with tolerance)."""
    for key, gt_val in gt_params.items():
        if key not in model_params:
            return False
        m_val = model_params[key]
        if isinstance(gt_val, float) and isinstance(m_val, (int, float)):
            if abs(gt_val - m_val) > abs(gt_val) * 0.2 + 1e-9:
                return False
        elif gt_val != m_val:
            # Try string comparison
            if str(gt_val).strip().lower() != str(m_val).strip().lower():
                return False
    return True


async def generate(client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"


async def run_condition(
    condition: str,
    problems: list[dict],
    output_path: Path,
):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(16)

    tasks = []
    for prob in problems:
        func_defs = format_function_defs(prob["function"])
        query = extract_query(prob["question"])

        if condition == "baseline":
            prompt = BASELINE_TEMPLATE.format(
                function_defs=func_defs,
                query=query,
            )
        else:
            # concept condition — will be implemented after baseline
            prompt = BASELINE_TEMPLATE.format(
                function_defs=func_defs,
                query=query,
            )

        tasks.append({
            "prob": prob,
            "prompt": prompt,
            "query": query,
        })

    print(f"Running {condition} on {len(tasks)} problems...")
    completions = await asyncio.gather(
        *[generate(client, t["prompt"], sem) for t in tasks]
    )

    print("Evaluating...")
    results = []
    for task, completion in zip(tasks, completions):
        prob = task["prob"]
        gt = prob.get("ground_truth", [])
        eval_result = evaluate_call(completion, gt)

        result = {
            "id": prob["id"],
            "split": prob["_split"],
            "condition": condition,
            "correct": eval_result["correct"],
            "matched": eval_result.get("matched", 0),
            "total_gt": eval_result.get("total_gt", 0),
            "gt_names": eval_result.get("gt_names", []),
            "model_names": eval_result.get("model_names", []),
            "error": eval_result.get("error"),
            "model_output": completion[:500],
            "ground_truth": gt,
        }
        results.append(result)

        status = "OK" if result["correct"] else "WRONG"
        print(f"  {prob['id']}: {status} (gt={gt[0][:60] if gt else '?'}, "
              f"model={eval_result.get('model_names', ['?'])})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n{condition.upper()}: {correct}/{total} ({100*correct/total:.1f}%)")

    # Per-split breakdown
    by_split = {}
    for r in results:
        by_split.setdefault(r["split"], []).append(r)
    for split in sorted(by_split):
        sr = by_split[split]
        sc = sum(1 for r in sr if r["correct"])
        print(f"  {split}: {sc}/{len(sr)} ({100*sc/len(sr):.0f}%)")

    # Error analysis
    parse_errors = sum(1 for r in results if r.get("error") == "could_not_parse_model")
    print(f"\nParse errors: {parse_errors}/{total}")

    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "concept", "compare"])
    parser.add_argument("--splits", nargs="+",
                       default=["BFCL_v3_exec_simple.json", "BFCL_v3_exec_multiple.json"])
    args = parser.parse_args()

    if args.condition == "compare":
        # TODO: implement comparison
        print("Compare not yet implemented")
        return

    print("Loading BFCL problems...")
    problems = load_bfcl_problems(args.splits)
    print(f"  Total: {len(problems)} problems")

    output_path = OUTPUT_DIR / f"{args.condition}_results.json"

    await run_condition(
        condition=args.condition,
        problems=problems,
        output_path=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
