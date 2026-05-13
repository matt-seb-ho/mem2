"""Build a recursive RAPTOR tree over shared community summaries.

Input:
  data/arc_agi/concept_memory/community_summaries_v1.json

Output:
  data/arc_agi/concept_memory/raptor_tree_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.providers.llmplus_client import LLMPlusProviderClient


SOURCE = ROOT / "data" / "arc_agi" / "concept_memory" / "community_summaries_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "raptor_tree_v1.json"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _summary_tokens(node: dict[str, Any]) -> set[str]:
    return _tokens(" ".join([
        str(node.get("node_id") or ""),
        str(node.get("summary") or ""),
        " ".join(node.get("member_communities") or []),
        " ".join(node.get("member_concepts") or []),
    ]))


def _build_leaf_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for idx, raw in enumerate(data.get("communities", []) or []):
        if not isinstance(raw, dict):
            continue
        summary = str(raw.get("llm_summary") or "").strip()
        members = [m for m in (raw.get("member_concepts") or []) if isinstance(m, str)]
        if not summary or not members:
            continue
        community_id = str(raw.get("community_id") or f"community_{idx}")
        nodes.append({
            "node_id": f"rt_L0_N{idx:03d}",
            "summary": summary,
            "member_communities": [community_id],
            "member_concepts": members,
            "summary_tokens": len(_tokens(summary)),
        })
    return nodes


def _cluster_nodes(nodes: list[dict[str, Any]], target_count: int) -> list[list[dict[str, Any]]]:
    if len(nodes) <= target_count:
        return [[node] for node in nodes]
    target_count = max(2, min(target_count, len(nodes)))
    sorted_nodes = sorted(nodes, key=lambda n: len(n.get("member_concepts") or []), reverse=True)
    clusters: list[list[dict[str, Any]]] = [[node] for node in sorted_nodes[:target_count]]
    cluster_tokens = [_summary_tokens(node) for node in sorted_nodes[:target_count]]
    for node in sorted_nodes[target_count:]:
        toks = _summary_tokens(node)
        scores = [
            len(toks & ct) / max(len(toks | ct), 1)
            for ct in cluster_tokens
        ]
        best_idx = max(range(len(scores)), key=lambda i: (scores[i], -len(clusters[i])))
        clusters[best_idx].append(node)
        cluster_tokens[best_idx].update(toks)
    return clusters


def _prompt_for_cluster(level: int, cluster: list[dict[str, Any]]) -> str:
    child_text = "\n\n".join(
        "\n".join([
            f"child_id: {node['node_id']}",
            f"communities: {', '.join(node.get('member_communities') or [])}",
            f"concepts: {', '.join((node.get('member_concepts') or [])[:24])}",
            f"summary: {node.get('summary')}",
        ])
        for node in cluster
    )
    return f"""You summarize a RAPTOR parent node for ARC concept memory.

Use only the child summaries. Write 3 concise bullets describing:
- the shared solver theme
- when retrieval should descend into this parent
- key differences among the child nodes

No markdown fences. No JSON.

# Parent level
{level}

# Child nodes
{child_text}
"""


async def _summarize_clusters(
    client: LLMPlusProviderClient,
    clusters: list[list[dict[str, Any]]],
    *,
    level: int,
    args: argparse.Namespace,
) -> tuple[list[str], int]:
    prompts = [_prompt_for_cluster(level, cluster) for cluster in clusters]
    results = await client.async_batch_generate(
        prompts,
        MODEL,
        {
            "temperature": 0.1,
            "max_tokens": args.max_tokens,
            "batch_size": args.concurrency,
            "ignore_cache": args.ignore_cache,
            "extra_kwargs": {
                "extra_body": {"reasoning": {"effort": "none", "exclude": True}},
            },
        },
        request_timeout=args.timeout,
    )
    failures = 0
    summaries: list[str] = []
    for cluster, completions in zip(clusters, results, strict=True):
        raw = completions[0] if completions else ""
        summary = (raw or "").strip()
        if summary.startswith("```"):
            summary = summary.split("\n", 1)[1] if "\n" in summary else summary
            if summary.endswith("```"):
                summary = summary[: summary.rfind("```")]
        summary = summary.strip()
        if not summary:
            failures += 1
            summary = "\n".join(f"- {node.get('summary', '')[:240]}" for node in cluster)
        summaries.append(summary)
    return summaries, failures


def _parent_node(level: int, idx: int, cluster: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    member_communities: list[str] = []
    member_concepts: list[str] = []
    child_ids: list[str] = []
    for node in cluster:
        child_ids.append(str(node["node_id"]))
        member_communities.extend(node.get("member_communities") or [])
        member_concepts.extend(node.get("member_concepts") or [])
    return {
        "node_id": f"rt_L{level}_N{idx:03d}",
        "summary": summary,
        "member_communities": list(dict.fromkeys(member_communities)),
        "member_concepts": list(dict.fromkeys(member_concepts)),
        "member_node_ids": child_ids,
        "child_node_ids": child_ids,
        "summary_tokens": len(_tokens(summary)),
    }


def _estimate_cost(snapshot: dict[str, dict[str, Any]]) -> float:
    usage = snapshot.get(MODEL, {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return (input_tokens / 1_000_000.0) * INPUT_COST_PER_M + (
        output_tokens / 1_000_000.0
    ) * OUTPUT_COST_PER_M


async def main_async(args: argparse.Namespace) -> int:
    if OUT_FILE.exists() and not args.force:
        print(f"ERROR: output already exists: {OUT_FILE}", file=sys.stderr)
        return 2
    if not SOURCE.exists():
        print(f"ERROR: source summary artifact not found: {SOURCE}", file=sys.stderr)
        return 2
    source = json.loads(SOURCE.read_text())
    if source.get("schema_version") != "1":
        print("ERROR: community summaries schema_version must be '1'", file=sys.stderr)
        return 2
    level0 = _build_leaf_nodes(source)
    if len(level0) < 2:
        print("ERROR: need at least two leaf community summaries", file=sys.stderr)
        return 2

    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "default_max_concurrency": args.concurrency,
    })
    levels: list[dict[str, Any]] = [{"level": 0, "nodes": level0}]
    current = level0
    failures = 0
    t0 = time.monotonic()
    for level_idx in range(1, args.max_levels):
        if len(current) <= args.target_roots:
            break
        target = max(args.target_roots, int(round(len(current) / args.cluster_factor)))
        clusters = _cluster_nodes(current, target)
        summaries, new_failures = await _summarize_clusters(
            client, clusters, level=level_idx, args=args,
        )
        failures += new_failures
        parents = [
            _parent_node(level_idx, idx, cluster, summary)
            for idx, (cluster, summary) in enumerate(zip(clusters, summaries, strict=True))
        ]
        levels.append({"level": level_idx, "nodes": parents})
        current = parents

    usage = client.get_usage_snapshot()
    cost = _estimate_cost(usage)
    out = {
        "schema_version": "1",
        "source_seed": str(SOURCE.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "levels": levels,
        "stats": {
            "num_levels": len(levels),
            "nodes_per_level": [len(level["nodes"]) for level in levels],
            "llm_calls": sum(len(level["nodes"]) for level in levels[1:]),
            "num_failures": failures,
            "estimated_cost_usd": cost,
            "wall_time_s": time.monotonic() - t0,
            "summary_tokens_mean": statistics.fmean(
                node.get("summary_tokens", 0)
                for level in levels
                for node in level["nodes"]
            ),
            "token_usage": usage,
        },
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[raptor_tree] wrote {OUT_FILE}")
    print(
        f"[raptor_tree] levels={out['stats']['num_levels']} "
        f"nodes={out['stats']['nodes_per_level']} "
        f"calls={out['stats']['llm_calls']} cost=${cost:.4f}"
    )
    if cost > args.max_cost_usd:
        print(
            f"ERROR: estimated cost ${cost:.4f} exceeded limit ${args.max_cost_usd:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0 if len(levels) >= 2 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--target-roots", type=int, default=3)
    parser.add_argument("--cluster-factor", type=float, default=2.5)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-cost-usd", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
