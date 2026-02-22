#!/usr/bin/env python3
"""Offline concept selection for a problem set.

Follows arc_memo's modular pipeline:
  Step 1: extract_concepts.py  → concept memory JSON
  Step 2: select_concepts.py   → selected_concepts.json + prompt_info.json  (THIS)
  Step 3: eval with pre-computed hints

Usage:
    python scripts/select_concepts.py \
        --concept-memory data/competition_math_nt_cp_l5/concept_memory/extracted_v2.json \
        --problems outputs/_runs/concept_math_eval/c72796785b5b/problems.json \
        --domain math \
        --model qwen/qwen3.5-397b-a17b \
        --output-dir data/competition_math_nt_cp_l5/concept_memory/selection_v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mem2.concepts.domain import DomainProfile
from mem2.concepts.memory import ConceptMemory
from mem2.concepts.prompts import DOMAIN_PROMPT_MAP
from mem2.providers.llmplus_client import LLMPlusProviderClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline concept selection")
    p.add_argument("--concept-memory", type=Path, required=True,
                    help="Path to concept memory JSON (from extract_concepts.py)")
    p.add_argument("--problems", type=Path, required=True,
                    help="Path to problems.json (from a previous run output)")
    p.add_argument("--domain", choices=["arc", "math", "code"], required=True)
    p.add_argument("--model", type=str, default="qwen/qwen3.5-397b-a17b",
                    help="LLM model for selection")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for selected_concepts.json + prompt_info.json")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--show-other-concepts", action="store_true",
                    help="Include names of non-selected concepts in hint (default: off)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_profile(mem: ConceptMemory, domain: str) -> DomainProfile | None:
    """Build a rendering profile from the concept memory's actual categories."""
    if domain == "arc":
        return None  # Use hardcoded ARC rendering
    kinds = sorted(mem.categories.keys())
    if not kinds:
        return None
    return DomainProfile(
        valid_kinds=set(kinds),
        section_order=kinds,
        section_headers={k: f"## {k}" for k in kinds},
    )


def format_problem_text(problem: dict, domain: str) -> str:
    """Extract problem text for the selection prompt."""
    metadata = problem.get("metadata", {})
    if domain in ("math", "code"):
        return metadata.get("problem_text", str(metadata))
    # ARC: would need grid formatting — not implemented here
    return metadata.get("problem_text", "")


def parse_selection(completion: str, valid_names: set[str]) -> tuple[list[str], str | None]:
    """Parse YAML concept selection from LLM completion."""
    if not completion.strip():
        return [], "empty_completion"

    m = _YAML_BLOCK_RE.search(completion)
    yaml_text = m.group(1) if m else None

    if yaml_text is None:
        return [], "no_yaml_block"

    try:
        parsed = yaml.safe_load(yaml_text)
    except Exception as exc:
        return [], f"yaml_parse_error: {exc}"

    if not isinstance(parsed, list):
        return [], f"unexpected_type: {type(parsed).__name__}"

    selected: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict) and len(item) == 1:
            k, v = next(iter(item.items()))
            name = f"{v}".strip() if isinstance(v, str) else f"{k}".strip()
        else:
            continue
        if name in valid_names and name not in selected:
            selected.append(name)

    if not selected:
        return [], "no_valid_names"
    return selected, None


async def main() -> None:
    args = parse_args()

    # ── Load concept memory ──────────────────────────────────────────
    mem = ConceptMemory()
    mem.load_from_file(args.concept_memory)
    logger.info(f"Loaded {len(mem.concepts)} concepts from {args.concept_memory}")
    valid_names = set(mem.concepts.keys())

    # ── Render concept memory string ─────────────────────────────────
    profile = build_profile(mem, args.domain)
    mem_str = mem.to_string(usage_threshold=0, profile=profile)
    logger.info(f"Concept memory string: {len(mem_str)} chars")

    # ── Load problems ────────────────────────────────────────────────
    problems = json.loads(args.problems.read_text())
    pids = sorted(problems.keys())
    logger.info(f"Loaded {len(pids)} problems")

    # ── Get selection prompt template ────────────────────────────────
    select_template, hint_template = DOMAIN_PROMPT_MAP.get(
        args.domain, DOMAIN_PROMPT_MAP["arc"]
    )

    # ── Build prompts ────────────────────────────────────────────────
    prompts = []
    for pid in pids:
        problem_text = format_problem_text(problems[pid], args.domain)
        prompt = select_template.format(concepts=mem_str, puzzle=problem_text)
        prompts.append(prompt)

    logger.info(f"Built {len(prompts)} selection prompts "
                f"(avg {sum(len(p) for p in prompts) // len(prompts)} chars)")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — first prompt preview:")
        print(f"{'='*60}")
        print(prompts[0][:2000])
        print(f"... ({len(prompts[0])} chars total)")
        return

    # ── Run LLM selection ────────────────────────────────────────────
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "default_max_concurrency": args.concurrency,
    })
    gen_cfg = {"n": 1, "temperature": 0.0, "max_tokens": args.max_tokens}

    logger.info(f"Sending {len(prompts)} selection prompts to {args.model}...")
    results = await client.async_batch_generate(
        prompts=prompts, model=args.model, gen_cfg=gen_cfg,
    )

    # ── Parse selections ─────────────────────────────────────────────
    selected_concepts: dict[str, list[str]] = {}
    completions_log: dict[str, str] = {}
    parse_errors: dict[str, str] = {}
    n_ok = 0
    n_fail = 0

    for pid, result_list in zip(pids, results):
        completion = str(result_list[0]) if result_list else ""
        completions_log[pid] = completion

        names, error = parse_selection(completion, valid_names)
        if names:
            selected_concepts[pid] = names
            n_ok += 1
        else:
            parse_errors[pid] = error or "unknown"
            n_fail += 1
            logger.warning(f"[{pid}] Selection failed: {error}")

    logger.info(f"Selection: {n_ok} ok, {n_fail} failed out of {len(pids)}")

    # ── Render prompt_info (pre-computed hints) ──────────────────────
    prompt_info: dict[str, dict] = {}
    for pid, selection in selected_concepts.items():
        rendered_hint = mem.to_string(
            concept_names=selection,
            skip_parameter_description=False,
            usage_threshold=0,
            show_other_concepts=args.show_other_concepts,
            profile=profile,
        )
        # Store raw rendered concepts — the inference engine wraps with its
        # own hint template at eval time (matches arc_memo's pattern).
        prompt_info[pid] = {"hint": rendered_hint}

    # ── Save outputs ─────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "selected_concepts.json").write_text(
        json.dumps(selected_concepts, indent=2) + "\n"
    )
    (out / "prompt_info.json").write_text(
        json.dumps(prompt_info, indent=2) + "\n"
    )
    (out / "completions.json").write_text(
        json.dumps(completions_log, indent=2) + "\n"
    )
    if parse_errors:
        (out / "parse_errors.json").write_text(
            json.dumps(parse_errors, indent=2) + "\n"
        )

    # ── Summary ──────────────────────────────────────────────────────
    sizes = [len(v) for v in selected_concepts.values()]
    print(f"\n{'='*60}")
    print("Selection Summary")
    print(f"{'='*60}")
    print(f"Problems:          {len(pids)}")
    print(f"Selections OK:     {n_ok}")
    print(f"Selection failed:  {n_fail}")
    if sizes:
        print(f"Concepts per problem: min={min(sizes)}, max={max(sizes)}, "
              f"mean={sum(sizes)/len(sizes):.1f}")
    print(f"Output dir:        {out}")
    print(f"Files:")
    print(f"  selected_concepts.json  — pid → [concept_name, ...]")
    print(f"  prompt_info.json        — pid → {{hint: rendered_text}}")
    print(f"  completions.json        — pid → raw LLM response")
    if parse_errors:
        print(f"  parse_errors.json       — {n_fail} failures with reasons")

    try:
        usage = client.get_usage_snapshot()
        if usage:
            print(f"LLM usage:         {usage}")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
