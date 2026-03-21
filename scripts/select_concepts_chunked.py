#!/usr/bin/env python3
"""Chunked concept selection — splits a large concept library into smaller
batches so the selector sees ~200-300 concepts at a time (matching the scale
where selection works reliably), then merges per-problem results.

Usage:
    python scripts/select_concepts_chunked.py \
        --concept-memory data/competition_math_all_l5/concept_memory/extracted_v3a_flash.json \
        --problems outputs/_runs/build_math_l5_flash/a3d763b86ea6/problems.json \
        --domain math \
        --model qwen/qwen3.5-flash-02-23 \
        --output-dir data/competition_math_all_l5/concept_memory/selection_v3a_chunked \
        --selector-render-mode cues_only \
        --concepts-per-chunk 220 \
        --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
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
    p = argparse.ArgumentParser(description="Chunked concept selection")
    p.add_argument("--concept-memory", type=Path, required=True)
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--domain", choices=["arc", "math", "code"], required=True)
    p.add_argument("--model", type=str, default="qwen/qwen3.5-flash-02-23")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--provider", default="llmplus_openrouter")
    p.add_argument("--selector-render-mode", default="cues_only",
                    choices=["full", "cues_only", "name_only"])
    p.add_argument("--concepts-per-chunk", type=int, default=220,
                    help="Max concepts per selection chunk (default: 220)")
    p.add_argument("--max-concepts-per-problem", type=int, default=5,
                    help="Max concepts to keep per problem after merging (default: 5)")
    p.add_argument("--chunk-delay", type=float, default=3.0,
                    help="Delay between concept chunks in seconds")
    p.add_argument("--show-other-concepts", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_profile(mem: ConceptMemory, domain: str) -> DomainProfile | None:
    if domain == "arc":
        return None
    kinds = sorted(mem.categories.keys())
    if not kinds:
        return None
    return DomainProfile(
        valid_kinds=set(kinds),
        section_order=kinds,
        section_headers={k: f"## {k}" for k in kinds},
    )


def format_problem_text(problem: dict, domain: str) -> str:
    metadata = problem.get("metadata", {})
    if domain == "code":
        return metadata.get("question_content", metadata.get("problem_text", str(metadata)))
    if domain == "math":
        return metadata.get("problem_text", str(metadata))
    return metadata.get("problem_text", "")


def parse_selection(completion: str, valid_names: set[str]) -> tuple[list[str], str | None]:
    if not completion.strip():
        return [], "empty_completion"
    m = _YAML_BLOCK_RE.search(completion)
    if m is None:
        return [], "no_yaml_block"
    try:
        parsed = yaml.safe_load(m.group(1))
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


def chunk_concepts(mem: ConceptMemory, chunk_size: int) -> list[list[str]]:
    """Split concept names into chunks of roughly equal size."""
    all_names = sorted(mem.concepts.keys())
    n_chunks = math.ceil(len(all_names) / chunk_size)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(all_names))
        chunks.append(all_names[start:end])
    return chunks


def render_concept_chunk(mem: ConceptMemory, concept_names: list[str],
                         profile: DomainProfile | None, sel_flags: dict) -> str:
    """Render a subset of concepts using the same rendering logic."""
    # Build a temporary ConceptMemory with only the chunk's concepts
    chunk_mem = ConceptMemory()
    for name in concept_names:
        if name in mem.concepts:
            concept = mem.concepts[name]
            chunk_mem.concepts[name] = concept
            kind = concept.kind
            if kind not in chunk_mem.categories:
                chunk_mem.categories[kind] = []
            if name not in chunk_mem.categories[kind]:
                chunk_mem.categories[kind].append(name)

    chunk_profile = None
    if profile:
        chunk_kinds = sorted(chunk_mem.categories.keys())
        chunk_profile = DomainProfile(
            valid_kinds=set(chunk_kinds),
            section_order=chunk_kinds,
            section_headers={k: f"## {k}" for k in chunk_kinds},
        )

    return chunk_mem.to_string(
        usage_threshold=0,
        profile=chunk_profile,
        skip_cues=sel_flags["skip_cues"],
        skip_implementation=sel_flags["skip_implementation"],
        skip_parameters=sel_flags["skip_parameters"],
        skip_parameter_description=sel_flags["skip_parameter_description"],
        include_description=sel_flags["include_description"],
    )


async def main() -> None:
    args = parse_args()

    # ── Load concept memory ──────────────────────────────────────────
    mem = ConceptMemory()
    mem.load_from_file(args.concept_memory)
    logger.info(f"Loaded {len(mem.concepts)} concepts from {args.concept_memory}")
    valid_names = set(mem.concepts.keys())

    # ── Build rendering profile & flags ──────────────────────────────
    from mem2.branches.memory_retriever.ps_selector import _RENDER_PROFILES
    profile = build_profile(mem, args.domain)
    sel_flags = _RENDER_PROFILES.get(args.selector_render_mode, _RENDER_PROFILES["full"])

    # ── Chunk the concept library ────────────────────────────────────
    concept_chunks = chunk_concepts(mem, args.concepts_per_chunk)
    logger.info(f"Split {len(mem.concepts)} concepts into {len(concept_chunks)} chunks "
                f"of ~{args.concepts_per_chunk} each")

    # ── Load problems ────────────────────────────────────────────────
    problems = json.loads(args.problems.read_text())
    pids = sorted(problems.keys())
    logger.info(f"Loaded {len(pids)} problems")

    # ── Get selection prompt template ────────────────────────────────
    select_template, hint_template = DOMAIN_PROMPT_MAP.get(
        args.domain, DOMAIN_PROMPT_MAP["arc"]
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for i, chunk_names in enumerate(concept_chunks):
            mem_str = render_concept_chunk(mem, chunk_names, profile, sel_flags)
            logger.info(f"Chunk {i+1}: {len(chunk_names)} concepts, "
                        f"{len(mem_str)} chars (~{len(mem_str)//4} tokens)")
        return

    # ── Run selection per chunk ───────────────────────────────────────
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": args.provider,
        "default_max_concurrency": args.concurrency,
    })
    gen_cfg = {"n": 1, "temperature": 0.0, "max_tokens": args.max_tokens,
               "batch_size": args.concurrency}

    # Per-problem aggregated selections across chunks
    all_selections: dict[str, list[str]] = {pid: [] for pid in pids}
    all_completions: dict[str, dict[str, str]] = {}  # chunk_idx -> pid -> completion

    for chunk_idx, chunk_names in enumerate(concept_chunks):
        chunk_valid = set(chunk_names)
        mem_str = render_concept_chunk(mem, chunk_names, profile, sel_flags)
        logger.info(f"Chunk {chunk_idx+1}/{len(concept_chunks)}: "
                    f"{len(chunk_names)} concepts, {len(mem_str)} chars")

        # Build prompts for all problems against this chunk
        prompts = []
        for pid in pids:
            problem_text = format_problem_text(problems[pid], args.domain)
            prompt = select_template.format(concepts=mem_str, puzzle=problem_text)
            prompts.append(prompt)

        logger.info(f"Sending {len(prompts)} prompts to {args.model}...")
        results = await client.async_batch_generate(
            prompts=prompts, model=args.model, gen_cfg=gen_cfg,
            request_timeout=args.request_timeout,
        )

        # Parse and accumulate
        chunk_completions = {}
        n_ok = 0
        for pid, result_list in zip(pids, results):
            completion = str(result_list[0]) if result_list else ""
            chunk_completions[pid] = completion
            names, error = parse_selection(completion, chunk_valid)
            if names:
                # Add new names (deduplicated)
                for name in names:
                    if name not in all_selections[pid]:
                        all_selections[pid].append(name)
                n_ok += 1

        all_completions[f"chunk_{chunk_idx}"] = chunk_completions
        logger.info(f"Chunk {chunk_idx+1}: {n_ok}/{len(pids)} problems got selections")

        # Save intermediate state
        (out / "completions_all_chunks.json").write_text(
            json.dumps(all_completions, indent=2) + "\n"
        )

        if chunk_idx < len(concept_chunks) - 1 and args.chunk_delay > 0:
            logger.info(f"Waiting {args.chunk_delay}s before next chunk...")
            await asyncio.sleep(args.chunk_delay)

    # ── Merge and cap selections ─────────────────────────────────────
    selected_concepts: dict[str, list[str]] = {}
    parse_errors: dict[str, str] = {}
    for pid in pids:
        names = all_selections[pid][:args.max_concepts_per_problem]
        if names:
            selected_concepts[pid] = names
        else:
            parse_errors[pid] = "no_selections_across_all_chunks"

    n_ok = len(selected_concepts)
    n_fail = len(parse_errors)
    logger.info(f"Merged selection: {n_ok} ok, {n_fail} failed out of {len(pids)}")

    # ── Render prompt_info ───────────────────────────────────────────
    prompt_info: dict[str, dict] = {}
    for pid, selection in selected_concepts.items():
        rendered_hint = mem.to_string(
            concept_names=selection,
            skip_parameter_description=False,
            usage_threshold=0,
            show_other_concepts=args.show_other_concepts,
            profile=profile,
        )
        prompt_info[pid] = {"hint": rendered_hint}

    # ── Save outputs ─────────────────────────────────────────────────
    (out / "selected_concepts.json").write_text(
        json.dumps(selected_concepts, indent=2) + "\n"
    )
    (out / "prompt_info.json").write_text(
        json.dumps(prompt_info, indent=2) + "\n"
    )
    if parse_errors:
        (out / "parse_errors.json").write_text(
            json.dumps(parse_errors, indent=2) + "\n"
        )

    # ── Summary ──────────────────────────────────────────────────────
    sizes = [len(v) for v in selected_concepts.values()]
    print(f"\n{'='*60}")
    print("Chunked Selection Summary")
    print(f"{'='*60}")
    print(f"Total concepts:    {len(mem.concepts)}")
    print(f"Chunks:            {len(concept_chunks)} × ~{args.concepts_per_chunk}")
    print(f"Problems:          {len(pids)}")
    print(f"Selections OK:     {n_ok} ({100*n_ok/len(pids):.1f}%)")
    print(f"Selection failed:  {n_fail}")
    if sizes:
        print(f"Concepts/problem:  min={min(sizes)}, max={max(sizes)}, "
              f"mean={sum(sizes)/len(sizes):.1f}")
    print(f"Output dir:        {out}")


if __name__ == "__main__":
    asyncio.run(main())
