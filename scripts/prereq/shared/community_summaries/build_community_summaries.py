"""Build LLM community summaries for GraphRAG and RAPTOR.

Inputs:
  data/arc_agi/concept_memory/compressed_v1.json

Output:
  data/arc_agi/concept_memory/shared/community_summaries_v1.json

The script builds Louvain communities over the existing co-activation graph,
asks DeepSeek V4 Flash for one concise report per community, and writes a
shared artifact consumed by both axis-1 community retrievers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.providers.llmplus_client import LLMPlusProviderClient


SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "community_summaries_v1.json"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28


SYSTEM_PROMPT = """You write compact ARC-AGI concept-memory community reports.

Given a cluster of concepts, summarize the shared reasoning pattern and the
operational value for solving ARC tasks. Be concrete. Do not invent concepts
outside the provided cluster.

Return plain English text only, 3 to 5 short bullets. No JSON, no code fence."""


def _render_concept(name: str, raw: dict[str, Any]) -> str:
    parts = [f"- {name} [{raw.get('kind', '?')}]"]
    desc = raw.get("description")
    if desc:
        parts.append(f"  description: {str(desc).strip()}")
    params = raw.get("parameters") or []
    if params:
        param_names = [str(p.get("name")) for p in params if isinstance(p, dict) and p.get("name")]
        if param_names:
            parts.append(f"  parameters: {', '.join(param_names[:8])}")
    cues = [str(c).strip() for c in (raw.get("cues") or []) if str(c).strip()]
    if cues:
        parts.append("  cues: " + " | ".join(cues[:4]))
    impl = [str(c).strip() for c in (raw.get("implementation") or []) if str(c).strip()]
    if impl:
        parts.append("  implementation: " + " | ".join(impl[:4]))
    return "\n".join(parts)


def _member_digest(concepts: dict[str, dict[str, Any]], members: list[str]) -> str:
    lines = []
    for name in members:
        raw = concepts[name]
        desc = str(raw.get("description") or "").replace("\n", " ").strip()
        if not desc:
            cues = [str(c).strip() for c in (raw.get("cues") or []) if str(c).strip()]
            desc = "; ".join(cues[:2])
        lines.append(f"{name}: {desc[:180]}")
    return "\n".join(lines)


def _summary_token_count(text: str) -> int:
    return len([tok for tok in text.replace("\n", " ").split(" ") if tok.strip()])


def _estimate_cost(snapshot: dict[str, dict[str, Any]]) -> float:
    usage = snapshot.get(MODEL, {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return (input_tokens / 1_000_000.0) * INPUT_COST_PER_M + (
        output_tokens / 1_000_000.0
    ) * OUTPUT_COST_PER_M


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def _build_prompt(concepts: dict[str, dict[str, Any]], community_id: str, seed: str, members: list[str]) -> str:
    rendered = "\n\n".join(_render_concept(name, concepts[name]) for name in members)
    return f"""# COMMUNITY

id: {community_id}
seed concept: {seed}
member count: {len(members)}

# MEMBER CONCEPTS

{rendered}

# TASK

Write an ARC concept-memory community report. Capture:
- what these concepts have in common
- when a solver should retrieve this community
- any important distinction between member concepts

Return only 3 to 5 concise bullets."""


async def main_async(args: argparse.Namespace) -> int:
    if OUT_FILE.exists() and not args.force:
        print(f"ERROR: output already exists: {OUT_FILE}", file=sys.stderr)
        return 2
    if not SEED_MEM.exists():
        print(f"ERROR: seed memory not found: {SEED_MEM}", file=sys.stderr)
        return 2

    seed_payload = json.loads(SEED_MEM.read_text())
    mem = ConceptMemory.from_payload(seed_payload)
    concepts: dict[str, dict[str, Any]] = seed_payload.get("concepts", {})
    if not mem.concepts:
        print("ERROR: no concepts loaded", file=sys.stderr)
        return 2

    try:
        import networkx as nx
        from networkx.algorithms import community as nx_community
    except ImportError as exc:
        raise RuntimeError("community summary build requires networkx") from exc

    graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1, load_typed_edges=False)
    G = nx.Graph()
    for name in mem.concepts:
        G.add_node(name)
    for edge in graph.edges():
        if edge.kind == "co_activation":
            G.add_edge(edge.src, edge.dst, weight=float(edge.weight or 1.0))

    raw_communities = list(nx_community.louvain_communities(G, seed=args.seed))
    communities: list[dict[str, Any]] = []
    for i, comm in enumerate(raw_communities):
        if len(comm) < args.min_community_size:
            continue
        ordered = sorted(comm, key=lambda n: (G.degree(n) if n in G else 0, n), reverse=True)
        if args.limit_communities and len(communities) >= args.limit_communities:
            break
        communities.append({
            "community_id": f"community_{i}",
            "seed_concept": ordered[0],
            "member_concepts": ordered,
            "member_digest": _member_digest(concepts, ordered),
        })

    prompts = [
        _build_prompt(concepts, c["community_id"], c["seed_concept"], c["member_concepts"])
        for c in communities
    ]
    print(f"[community_summaries] concepts={len(mem.concepts)} communities={len(communities)}")

    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "default_max_concurrency": args.concurrency,
    })
    t0 = time.monotonic()
    results = await client.async_batch_generate(
        prompts,
        MODEL,
        {
            "temperature": 0.1,
            "max_tokens": args.max_tokens,
            "batch_size": args.concurrency,
            "ignore_cache": args.ignore_cache,
        },
        request_timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()

    failures = 0
    for community, completions in zip(communities, results, strict=True):
        raw = completions[0] if completions else None
        summary = _strip_code_fence(raw or "")
        if not summary:
            failures += 1
            summary = community["member_digest"]
        community["llm_summary"] = summary
        community["summary_tokens"] = _summary_token_count(summary)

    member_counts = [len(c["member_concepts"]) for c in communities] or [0]
    out = {
        "schema_version": "1",
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "source_graph": "co_activation_louvain",
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "communities": communities,
        "stats": {
            "num_concepts": len(mem.concepts),
            "num_communities": len(communities),
            "member_count_min": min(member_counts),
            "member_count_max": max(member_counts),
            "member_count_mean": statistics.fmean(member_counts),
            "num_failures": failures,
            "wall_time_s": elapsed,
            "estimated_cost_usd": _estimate_cost(usage),
            "token_usage": usage,
        },
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[community_summaries] wrote {OUT_FILE}")
    print(f"[community_summaries] failures={failures} cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0 if failures == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-community-size", type=int, default=2)
    parser.add_argument("--limit-communities", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

