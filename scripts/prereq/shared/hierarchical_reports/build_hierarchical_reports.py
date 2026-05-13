"""Build hierarchical entity-community reports for GraphRAG-style retrieval.

Inputs:
  data/arc_agi/concept_memory/shared/entity_graph_v1.json

Output:
  data/arc_agi/concept_memory/shared/hierarchical_reports_v1.json

The builder clusters the entity graph recursively with Louvain and asks the
canonical DeepSeek/OpenRouter model for concise community reports at each
level. It is intentionally a shared substrate for GraphRAG-style community
reports and RAPTOR-like multi-layer parent summaries.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.providers.llmplus_client import LLMPlusProviderClient


ENTITY_GRAPH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "entity_graph_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "hierarchical_reports_v1.json"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28


SYSTEM_PROMPT = """You write compact GraphRAG-style community reports for ARC entity graphs.

Summarize what the community represents, what ARC solver operations it supports,
and which distinctions matter for retrieval. Use only the provided entities and
child summaries. Return 3 to 5 concise bullets. No JSON. No code fences."""


def _load_entity_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(ENTITY_GRAPH.read_text())
    if data.get("schema_version") != "1":
        raise ValueError("entity graph schema_version must be '1'")
    entities = [e for e in data.get("entities", []) or [] if isinstance(e, dict)]
    edges = [e for e in data.get("edges", []) or [] if isinstance(e, dict)]
    return entities, edges, data


def _entity_line(entity: dict[str, Any]) -> str:
    attrs = entity.get("attributes") if isinstance(entity.get("attributes"), dict) else {}
    attr_text = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:4])
    attr_suffix = f"; {attr_text}" if attr_text else ""
    return (
        f"- {entity.get('mention_text', '')} [{entity.get('entity_type', 'other')}] "
        f"from {entity.get('source_concept', '?')}{attr_suffix}"
    )


def _build_graph(entities: list[dict[str, Any]], edges: list[dict[str, Any]]):
    import networkx as nx

    by_id = {e["entity_id"]: e for e in entities if isinstance(e.get("entity_id"), str)}
    G = nx.Graph()
    for entity_id, entity in by_id.items():
        G.add_node(entity_id, **entity)
    for edge in edges:
        src = edge.get("src_entity")
        dst = edge.get("dst_entity")
        if src not in by_id or dst not in by_id or src == dst:
            continue
        try:
            weight = float(edge.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += weight
        else:
            G.add_edge(src, dst, weight=weight, edge_type=edge.get("edge_type", "related_to"))
    return G


def _louvain(G, *, seed: int, min_size: int) -> list[set[str]]:
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() < min_size * 2:
        return [set(G.nodes())]
    try:
        from networkx.algorithms import community as nx_community

        communities = list(nx_community.louvain_communities(G, seed=seed, weight="weight"))
    except Exception:
        communities = [set(c) for c in __import__("networkx").connected_components(G)]
    out: list[set[str]] = []
    small: set[str] = set()
    for comm in communities:
        if len(comm) >= min_size:
            out.append(set(comm))
        else:
            small.update(comm)
    if small:
        out.append(small)
    out.sort(key=lambda c: (len(c), sorted(c)[0] if c else ""), reverse=True)
    return out


def _source_concepts(entity_ids: list[str], by_entity: dict[str, dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(
        str(by_entity[e].get("source_concept"))
        for e in entity_ids
        if e in by_entity and by_entity[e].get("source_concept")
    ))


def _member_digest(entity_ids: list[str], by_entity: dict[str, dict[str, Any]], *, max_entities: int) -> str:
    lines = [_entity_line(by_entity[e]) for e in entity_ids[:max_entities] if e in by_entity]
    if len(entity_ids) > max_entities:
        lines.append(f"- ... {len(entity_ids) - max_entities} more entities")
    return "\n".join(lines)


def _make_level_reports(
    level: int,
    partitions: list[set[str]],
    *,
    by_entity: dict[str, dict[str, Any]],
    child_map: dict[int, list[str]] | None,
    max_communities: int,
    max_entities_per_report: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for idx, comm in enumerate(partitions[:max_communities]):
        entity_ids = sorted(comm)
        reports.append({
            "community_id": f"L{level}_C{idx:03d}",
            "level": level,
            "entities": entity_ids,
            "source_concepts": _source_concepts(entity_ids, by_entity),
            "child_communities": child_map.get(idx, []) if child_map else [],
            "member_digest": _member_digest(
                entity_ids,
                by_entity,
                max_entities=max_entities_per_report,
            ),
            "llm_summary": "",
            "summary_tokens": 0,
        })
    return reports


def _supergraph(
    lower_reports: list[dict[str, Any]],
    original_graph,
):
    import networkx as nx

    entity_to_report: dict[str, int] = {}
    for idx, report in enumerate(lower_reports):
        for entity_id in report.get("entities") or []:
            entity_to_report[entity_id] = idx

    G = nx.Graph()
    for idx, report in enumerate(lower_reports):
        G.add_node(str(idx), size=len(report.get("entities") or []))
    for src, dst, data in original_graph.edges(data=True):
        a = entity_to_report.get(src)
        b = entity_to_report.get(dst)
        if a is None or b is None or a == b:
            continue
        weight = float(data.get("weight", 1.0))
        a_key, b_key = str(a), str(b)
        if G.has_edge(a_key, b_key):
            G[a_key][b_key]["weight"] += weight
        else:
            G.add_edge(a_key, b_key, weight=weight)
    return G


def _lift_partitions(
    partitions: list[set[str]],
    lower_reports: list[dict[str, Any]],
) -> tuple[list[set[str]], dict[int, list[str]]]:
    lifted: list[set[str]] = []
    child_map: dict[int, list[str]] = {}
    for idx, comm in enumerate(partitions):
        entity_ids: set[str] = set()
        child_ids: list[str] = []
        for node in sorted(comm, key=lambda n: int(n)):
            report = lower_reports[int(node)]
            child_ids.append(report["community_id"])
            entity_ids.update(report.get("entities") or [])
        lifted.append(entity_ids)
        child_map[idx] = child_ids
    return lifted, child_map


def _summary_token_count(text: str) -> int:
    return len([tok for tok in text.replace("\n", " ").split(" ") if tok.strip()])


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def _build_prompt(report: dict[str, Any], child_by_id: dict[str, dict[str, Any]]) -> str:
    child_lines: list[str] = []
    for child_id in report.get("child_communities") or []:
        child = child_by_id.get(child_id)
        if not child:
            continue
        child_lines.append(
            f"- {child_id}: {child.get('llm_summary', '')[:500]}"
        )
    children = "\n".join(child_lines) if child_lines else "(leaf community)"
    return f"""# COMMUNITY

id: {report['community_id']}
level: {report['level']}
entity count: {len(report.get('entities') or [])}
source concepts: {', '.join((report.get('source_concepts') or [])[:20])}

# ENTITY SAMPLE

{report.get('member_digest') or '(none)'}

# CHILD SUMMARIES

{children}

# TASK

Write a hierarchical ARC entity-community report for retrieval. Capture:
- shared entity/operation theme
- when a solver should retrieve this community
- important child-community distinctions, if any

Return only 3 to 5 concise bullets."""


async def _summarize_level(
    client: LLMPlusProviderClient,
    reports: list[dict[str, Any]],
    *,
    child_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    if not reports:
        return 0
    prompts = [_build_prompt(report, child_by_id) for report in reports]
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
    for report, completions in zip(reports, results, strict=True):
        raw = completions[0] if completions else ""
        summary = _strip_code_fence(raw or "")
        if not summary:
            failures += 1
            summary = report.get("member_digest") or ""
        report["llm_summary"] = summary
        report["summary_tokens"] = _summary_token_count(summary)
    return failures


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
    if not ENTITY_GRAPH.exists():
        print(f"ERROR: entity graph not found: {ENTITY_GRAPH}", file=sys.stderr)
        return 2

    entities, edges, source_graph = _load_entity_graph()
    by_entity = {e["entity_id"]: e for e in entities if isinstance(e.get("entity_id"), str)}
    graph = _build_graph(entities, edges)
    print(f"[hierarchical_reports] entities={len(by_entity)} edges={graph.number_of_edges()} model={MODEL}")

    level0_parts = _louvain(graph, seed=args.seed, min_size=args.min_community_size)
    level0 = _make_level_reports(
        0,
        level0_parts,
        by_entity=by_entity,
        child_map=None,
        max_communities=args.max_communities_per_level,
        max_entities_per_report=args.max_entities_per_report,
    )

    level1_graph = _supergraph(level0, graph)
    level1_node_parts = _louvain(level1_graph, seed=args.seed + 1, min_size=2)
    if len(level1_node_parts) <= 1 and level0:
        level1_node_parts = [set(str(i) for i in range(len(level0)))]
    level1_parts, level1_children = _lift_partitions(level1_node_parts, level0)
    level1 = _make_level_reports(
        1,
        level1_parts,
        by_entity=by_entity,
        child_map=level1_children,
        max_communities=args.max_communities_per_level,
        max_entities_per_report=args.max_entities_per_report,
    )

    level2_graph = _supergraph(level1, graph)
    level2_node_parts = _louvain(level2_graph, seed=args.seed + 2, min_size=2)
    if len(level2_node_parts) <= 1 and level1:
        level2_node_parts = [set(str(i) for i in range(len(level1)))]
    level2_parts, level2_children = _lift_partitions(level2_node_parts, level1)
    level2 = _make_level_reports(
        2,
        level2_parts,
        by_entity=by_entity,
        child_map=level2_children,
        max_communities=args.max_communities_per_level,
        max_entities_per_report=args.max_entities_per_report,
    )

    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "default_max_concurrency": args.concurrency,
    })
    t0 = time.monotonic()
    failures = 0
    failures += await _summarize_level(client, level0, child_by_id={}, args=args)
    by_report_id = {r["community_id"]: r for r in level0}
    failures += await _summarize_level(client, level1, child_by_id=by_report_id, args=args)
    by_report_id.update({r["community_id"]: r for r in level1})
    failures += await _summarize_level(client, level2, child_by_id=by_report_id, args=args)
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    cost = _estimate_cost(usage)

    hierarchy = {
        "level_0": level0,
        "level_1": level1,
        "level_2": level2,
    }
    report_counts = [len(v) for v in hierarchy.values()]
    out = {
        "schema_version": "1",
        "source_graph": str(ENTITY_GRAPH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clustering": "recursive_louvain",
        "hierarchy": hierarchy,
        "stats": {
            "num_entities": len(by_entity),
            "num_graph_edges": graph.number_of_edges(),
            "num_levels": len([v for v in hierarchy.values() if v]),
            "num_reports": sum(report_counts),
            "reports_per_level": report_counts,
            "report_entities_mean": statistics.fmean(
                len(r.get("entities") or [])
                for reports in hierarchy.values()
                for r in reports
            ) if sum(report_counts) else 0.0,
            "num_failures": failures,
            "wall_time_s": elapsed,
            "estimated_cost_usd": cost,
            "token_usage": usage,
            "source_entity_graph_stats": source_graph.get("stats") or {},
        },
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[hierarchical_reports] wrote {OUT_FILE}")
    print(
        f"[hierarchical_reports] reports={out['stats']['num_reports']} "
        f"levels={out['stats']['num_levels']} failures={failures} cost=${cost:.4f}"
    )
    if cost > args.max_cost_usd:
        print(
            f"ERROR: estimated cost ${cost:.4f} exceeded limit ${args.max_cost_usd:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0 if out["stats"]["num_levels"] >= 2 and out["stats"]["num_reports"] > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-community-size", type=int, default=4)
    parser.add_argument("--max-communities-per-level", type=int, default=28)
    parser.add_argument("--max-entities-per-report", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-cost-usd", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
