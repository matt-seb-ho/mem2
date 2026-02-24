#!/usr/bin/env python3
"""Offline concept routing — filter precomputed selections through a Router.

Stage 3 in the modular pipeline:
  Step 1: extract_concepts.py  → concept memory JSON
  Step 2: select_concepts.py   → selected_concepts.json + prompt_info.json
  Step 3: route_concepts.py    → filtered selected_concepts.json  (THIS)
  Step 4: eval with pre-computed hints

Takes the output of select_concepts.py (selected_concepts.json) and runs a
Router (NLI or LLM) over each problem's rendered hints to filter which
concepts to keep. Produces a new selection directory that can be used directly
by ps_selector via ``selected_concepts_file`` or ``prompt_info_file``.

Output goes to ``routed_<type>_<model>/`` alongside the source selection dir:
  selection_v2/selected_concepts.json  →  routed_nli_nli-deberta-v3-base/
  selection_v2/selected_concepts.json  →  routed_llm_qwen3-coder-30b-a3b-instruct/

Usage (NLI):
    python scripts/route_concepts.py \
        --concept-memory data/livecodebench_v56/concept_memory/extracted_v2.json \
        --selected-concepts data/livecodebench_v56/concept_memory/selection_v2/selected_concepts.json \
        --problems outputs/_runs/build_lcb/5b254edab37a/problems.json \
        --domain code \
        --router nli \
        --entailment-threshold 0.4
    # → data/livecodebench_v56/concept_memory/routed_nli_nli-deberta-v3-base/

Usage (LLM):
    python scripts/route_concepts.py \
        --concept-memory data/livecodebench_v56/concept_memory/extracted_v2.json \
        --selected-concepts data/livecodebench_v56/concept_memory/selection_v2/selected_concepts.json \
        --problems outputs/_runs/build_lcb/5b254edab37a/problems.json \
        --domain code \
        --router llm \
        --model qwen/qwen3-coder-30b-a3b-instruct
    # → data/livecodebench_v56/concept_memory/routed_llm_qwen3-coder-30b-a3b-instruct/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mem2.concepts.domain import DomainProfile
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Render profiles (same as ps_selector) ─────────────────────────────────
_RENDER_PROFILES = {
    "full": dict(
        skip_cues=False, skip_implementation=False, skip_parameters=False,
        skip_parameter_description=True, include_description=True,
    ),
    "cues_only": dict(
        skip_cues=False, skip_implementation=True, skip_parameters=True,
        skip_parameter_description=True, include_description=True,
    ),
    "name_only": dict(
        skip_cues=True, skip_implementation=True, skip_parameters=True,
        skip_parameter_description=True, include_description=True,
    ),
}


def _sanitize_model_name(model: str) -> str:
    """Extract short model name for directory naming.

    ``cross-encoder/nli-deberta-v3-base`` → ``nli-deberta-v3-base``
    ``qwen/qwen3-coder-30b-a3b-instruct`` → ``qwen3-coder-30b-a3b-instruct``
    """
    # Take the part after the last /
    short = model.rsplit("/", 1)[-1]
    # Replace any remaining filesystem-unfriendly chars
    return re.sub(r"[^\w\-.]", "_", short)


def _build_output_dir(args: argparse.Namespace) -> Path:
    """Auto-generate output dir as sibling of the source selection dir.

    routed_<type>_<model_short_name>/ alongside the parent of
    --selected-concepts.
    """
    parent = args.selected_concepts.resolve().parent.parent  # up from selection_vN/
    model = args.nli_model if args.router == "nli" else args.model
    model_short = _sanitize_model_name(model)
    return parent / f"routed_{args.router}_{model_short}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline concept routing")
    p.add_argument("--concept-memory", type=Path, required=True,
                    help="Path to concept memory JSON (from extract_concepts.py)")
    p.add_argument("--selected-concepts", type=Path, required=True,
                    help="Path to selected_concepts.json (from select_concepts.py)")
    p.add_argument("--problems", type=Path, required=True,
                    help="Path to problems.json (from a previous run output)")
    p.add_argument("--domain", choices=["arc", "math", "code"], required=True)
    p.add_argument("--router", choices=["nli", "llm"], required=True,
                    help="Router type: nli (cross-encoder) or llm (LLM-based)")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory (default: routed_<type>_<model>/ "
                         "alongside source selection dir)")

    # Render options
    p.add_argument("--render-mode", default="full",
                    choices=["full", "cues_only", "name_only"])
    p.add_argument("--show-other-concepts", action="store_true",
                    help="Include names of non-selected concepts in hint")

    # NLI-specific
    p.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base",
                    help="NLI cross-encoder model name")
    p.add_argument("--entailment-threshold", type=float, default=0.5,
                    help="Entailment threshold for NLI router")
    p.add_argument("--device", default="cuda",
                    help="Device for NLI model (cuda/cpu)")

    # LLM-specific
    p.add_argument("--model", default="",
                    help="LLM model for routing (required for --router llm)")
    p.add_argument("--concurrency", type=int, default=16,
                    help="Max concurrency for LLM calls")

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_profile(mem: ConceptMemory, domain: str) -> DomainProfile | None:
    """Build a rendering profile from the concept memory's actual categories."""
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


def render_hint_text(
    mem: ConceptMemory, names: list[str], render_mode: str,
    show_other_concepts: bool, profile: DomainProfile | None,
) -> str:
    """Render concept hints for a selection (same logic as ps_selector)."""
    flags = _RENDER_PROFILES.get(render_mode, _RENDER_PROFILES["full"])
    return mem.to_string(
        concept_names=names,
        skip_parameter_description=flags["skip_parameter_description"],
        skip_cues=flags["skip_cues"],
        skip_implementation=flags["skip_implementation"],
        skip_parameters=flags["skip_parameters"],
        include_description=flags["include_description"],
        usage_threshold=0,
        show_other_concepts=show_other_concepts,
        profile=profile,
    )


def build_problem_spec(pid: str, problem_data: dict) -> ProblemSpec:
    """Reconstruct a ProblemSpec from problems.json data."""
    return ProblemSpec(
        uid=pid,
        train_pairs=problem_data.get("train_pairs", []),
        test_pairs=problem_data.get("test_pairs", []),
        metadata=problem_data.get("metadata", {}),
    )


def build_retrieval_bundle(
    pid: str, names: list[str], hint_text: str,
) -> RetrievalBundle:
    """Build a RetrievalBundle mimicking ps_selector output."""
    return RetrievalBundle(
        problem_uid=pid,
        hint_text=hint_text,
        retrieved_items=[{"concept": n} for n in names],
        metadata={
            "selected_names": list(names),
            "selected_count": len(names),
            "selector_mode": "precomputed",
        },
    )


async def run_nli_routing(
    problems: dict[str, ProblemSpec],
    bundles: dict[str, RetrievalBundle],
    args: argparse.Namespace,
) -> dict[str, RetrievalBundle]:
    """Run NLI router over all bundles."""
    from mem2.branches.router.nli import NliRouter

    router = NliRouter(
        model_name=args.nli_model,
        entailment_threshold=args.entailment_threshold,
        domain=args.domain,
        device=args.device,
    )

    ctx = RunContext(
        run_id="offline_routing",
        seed=0,
        config={},
        output_dir="",
    )

    results: dict[str, RetrievalBundle] = {}
    pids = sorted(bundles.keys())
    total = len(pids)

    for i, pid in enumerate(pids):
        result = await router.route(
            ctx=ctx,
            provider=None,
            problem=problems[pid],
            retrieval=bundles[pid],
        )
        results[pid] = result

        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"NLI routing: {i + 1}/{total}")

    return results


async def run_llm_routing(
    problems: dict[str, ProblemSpec],
    bundles: dict[str, RetrievalBundle],
    args: argparse.Namespace,
) -> dict[str, RetrievalBundle]:
    """Run LLM router over all bundles with concurrency."""
    from mem2.branches.router.llm import LlmRouter
    from mem2.providers.llmplus_client import LLMPlusProviderClient

    if not args.model:
        raise ValueError("--model is required for LLM routing")

    router = LlmRouter(
        model=args.model,
        domain=args.domain,
    )

    provider = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "default_max_concurrency": args.concurrency,
    })

    ctx = RunContext(
        run_id="offline_routing",
        seed=0,
        config={},
        output_dir="",
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    results: dict[str, RetrievalBundle] = {}
    pids = sorted(bundles.keys())
    total = len(pids)
    done_count = 0

    async def route_one(pid: str) -> None:
        nonlocal done_count
        async with semaphore:
            result = await router.route(
                ctx=ctx,
                provider=provider,
                problem=problems[pid],
                retrieval=bundles[pid],
            )
            results[pid] = result
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                logger.info(f"LLM routing: {done_count}/{total}")

    await asyncio.gather(*[route_one(pid) for pid in pids])
    return results


async def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = _build_output_dir(args)
    logger.info(f"Output dir: {args.output_dir}")

    # ── Load concept memory ──────────────────────────────────────────
    mem = ConceptMemory()
    mem.load_from_file(args.concept_memory)
    logger.info(f"Loaded {len(mem.concepts)} concepts from {args.concept_memory}")
    profile = build_profile(mem, args.domain)

    # ── Load precomputed selections ───────────────────────────────────
    selected_concepts: dict[str, list[str]] = json.loads(
        args.selected_concepts.read_text()
    )
    logger.info(
        f"Loaded selections for {len(selected_concepts)} problems "
        f"from {args.selected_concepts}"
    )

    # ── Load problems ─────────────────────────────────────────────────
    problems_raw: dict[str, dict] = json.loads(args.problems.read_text())
    logger.info(f"Loaded {len(problems_raw)} problems from {args.problems}")

    # Only process problems that have both selections and problem data
    pids = sorted(
        pid for pid in selected_concepts
        if pid in problems_raw and selected_concepts[pid]
    )
    logger.info(f"Processing {len(pids)} problems (have both selection and problem data)")

    # ── Build ProblemSpecs and RetrievalBundles ────────────────────────
    problems: dict[str, ProblemSpec] = {}
    bundles: dict[str, RetrievalBundle] = {}

    for pid in pids:
        problems[pid] = build_problem_spec(pid, problems_raw[pid])
        names = selected_concepts[pid]
        hint_text = render_hint_text(
            mem, names, args.render_mode, args.show_other_concepts, profile,
        )
        bundles[pid] = build_retrieval_bundle(pid, names, hint_text)

    hint_sizes = [len(b.hint_text or "") for b in bundles.values()]
    logger.info(
        f"Rendered hints: min={min(hint_sizes)}, max={max(hint_sizes)}, "
        f"mean={sum(hint_sizes) / len(hint_sizes):.0f} chars"
    )

    if args.dry_run:
        pid = pids[0]
        print(f"\n{'='*60}")
        print(f"DRY RUN — preview for problem {pid}")
        print(f"{'='*60}")
        print(f"Selected concepts: {selected_concepts[pid]}")
        print(f"\nHint text ({len(bundles[pid].hint_text or '')} chars):")
        print((bundles[pid].hint_text or "")[:1000])
        return

    # ── Run router ────────────────────────────────────────────────────
    t0 = time.time()
    if args.router == "nli":
        results = await run_nli_routing(problems, bundles, args)
    else:
        results = await run_llm_routing(problems, bundles, args)
    elapsed = time.time() - t0
    logger.info(f"Routing complete in {elapsed:.1f}s")

    # ── Extract outputs ───────────────────────────────────────────────
    filtered_concepts: dict[str, list[str]] = {}
    routing_scores: dict[str, dict] = {}
    prompt_info: dict[str, dict] = {}
    n_kept = 0
    n_dropped = 0
    all_included: list[str] = []
    all_excluded: list[str] = []

    for pid in pids:
        result = results[pid]
        md = result.metadata

        # Extract filtered concept names from surviving items
        surviving_names = [
            item["concept"] for item in result.retrieved_items
            if "concept" in item
        ]
        filtered_concepts[pid] = surviving_names

        # Collect routing metadata
        score_entry: dict = {}
        if "routing_nli_scores" in md:
            score_entry["nli_scores"] = md["routing_nli_scores"]
        if "routing_prompt" in md:
            score_entry["prompt"] = md["routing_prompt"]
        if "routing_completion" in md:
            score_entry["completion"] = md["routing_completion"]
        if "routing_parse_failure" in md:
            score_entry["parse_failure"] = True
        score_entry["included_items"] = md.get("routing_included_items", [])
        score_entry["excluded_items"] = md.get("routing_excluded_items", [])
        score_entry["included"] = md.get("routing_included", True)
        routing_scores[pid] = score_entry

        # Re-render hint for surviving concepts
        if surviving_names:
            hint = render_hint_text(
                mem, surviving_names, args.render_mode,
                args.show_other_concepts, profile,
            )
            prompt_info[pid] = {"hint": hint}
            n_kept += 1
        else:
            n_dropped += 1

        all_included.extend(md.get("routing_included_items", []))
        all_excluded.extend(md.get("routing_excluded_items", []))

    # ── Compute concept frequencies for filtered set ──────────────────
    concept_counts: Counter = Counter()
    for names in filtered_concepts.values():
        for name in names:
            concept_counts[name] += 1
    total_problems = len(filtered_concepts)
    concept_frequencies = {
        name: count / total_problems
        for name, count in concept_counts.items()
    }

    # ── Build summary ─────────────────────────────────────────────────
    original_sizes = [len(selected_concepts[pid]) for pid in pids]
    filtered_sizes = [len(filtered_concepts[pid]) for pid in pids]

    summary = {
        "router": args.router,
        "domain": args.domain,
        "render_mode": args.render_mode,
        "total_problems": len(pids),
        "problems_with_hints": n_kept,
        "problems_dropped_all": n_dropped,
        "elapsed_seconds": round(elapsed, 1),
        "original_concepts_per_problem": {
            "min": min(original_sizes),
            "max": max(original_sizes),
            "mean": round(sum(original_sizes) / len(original_sizes), 2),
        },
        "filtered_concepts_per_problem": {
            "min": min(filtered_sizes),
            "max": max(filtered_sizes),
            "mean": round(sum(filtered_sizes) / len(filtered_sizes), 2),
        },
        "total_items_included": len(all_included),
        "total_items_excluded": len(all_excluded),
        "include_rate": round(
            len(all_included) / max(len(all_included) + len(all_excluded), 1), 3
        ),
    }
    if args.router == "nli":
        summary["nli_model"] = args.nli_model
        summary["entailment_threshold"] = args.entailment_threshold
        summary["device"] = args.device
    else:
        summary["llm_model"] = args.model
        n_parse_failures = sum(
            1 for s in routing_scores.values() if s.get("parse_failure")
        )
        summary["parse_failures"] = n_parse_failures

    # ── Save outputs ──────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "selected_concepts.json").write_text(
        json.dumps(filtered_concepts, indent=2) + "\n"
    )
    (out / "prompt_info.json").write_text(
        json.dumps(prompt_info, indent=2) + "\n"
    )
    (out / "routing_scores.json").write_text(
        json.dumps(routing_scores, indent=2) + "\n"
    )
    (out / "routing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (out / "concept_frequencies.json").write_text(
        json.dumps(concept_frequencies, indent=2) + "\n"
    )

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Routing Summary")
    print(f"{'='*60}")
    print(f"Router:            {args.router}")
    if args.router == "nli":
        print(f"NLI model:         {args.nli_model}")
        print(f"Threshold:         {args.entailment_threshold}")
    else:
        print(f"LLM model:         {args.model}")
    print(f"Problems:          {len(pids)}")
    print(f"With hints:        {n_kept}")
    print(f"All dropped:       {n_dropped}")
    print(f"Include rate:      {summary['include_rate']:.1%}")
    print(f"Concepts/problem:  "
          f"{summary['original_concepts_per_problem']['mean']:.1f} → "
          f"{summary['filtered_concepts_per_problem']['mean']:.1f}")
    print(f"Time:              {elapsed:.1f}s")
    print(f"Output dir:        {out}")
    print(f"Files:")
    print(f"  selected_concepts.json   — pid → [filtered concept names]")
    print(f"  prompt_info.json         — pid → {{hint: rendered text}}")
    print(f"  routing_scores.json      — pid → per-item scores/completions")
    print(f"  routing_summary.json     — aggregate statistics")
    print(f"  concept_frequencies.json — concept → frequency in filtered set")


if __name__ == "__main__":
    asyncio.run(main())
