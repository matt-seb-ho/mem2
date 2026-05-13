"""Build OpenIE-style concept facts for axis-1 graph retrievers.

Inputs:
  data/arc_agi/concept_memory/compressed_v1.json

Output:
  data/arc_agi/concept_memory/shared/openie_facts_v1.json

The script asks DeepSeek V4 Flash for compact (subject, predicate, object)
facts per concept and links those facts back to neighboring concepts. The
artifact is shared by HippoRAG/PPR, PathRAG, MAGMA, LightRAG, and HippoRAG2.
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

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.providers.llmplus_client import LLMPlusProviderClient


SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "openie_facts_v1.json"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28

RELATION_KINDS = {
    "uses",
    "specializes",
    "contrasts",
    "co_occurs",
    "parameterizes",
    "same_operation",
    "other",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _concept_digest(name: str, raw: dict[str, Any]) -> str:
    parts = [f"- name: {name}", f"  kind: {raw.get('kind', '?')}"]
    desc = str(raw.get("description") or "").strip()
    if desc:
        parts.append(f"  description: {desc}")
    params = raw.get("parameters") or []
    if params:
        names = [str(p.get("name")) for p in params if isinstance(p, dict) and p.get("name")]
        if names:
            parts.append("  parameters: " + ", ".join(names[:8]))
    cues = [str(c).strip() for c in (raw.get("cues") or []) if str(c).strip()]
    if cues:
        parts.append("  cues: " + " | ".join(cues[:4]))
    impl = [str(c).strip() for c in (raw.get("implementation") or []) if str(c).strip()]
    if impl:
        parts.append("  implementation: " + " | ".join(impl[:4]))
    used_in = [str(u).strip() for u in (raw.get("used_in") or []) if str(u).strip()]
    if used_in:
        parts.append("  used_in_examples: " + ", ".join(used_in[:6]))
    return "\n".join(parts)


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _usage_count(raw: dict[str, Any]) -> int:
    used_in = raw.get("used_in") or []
    return len(used_in) if isinstance(used_in, list) else 0


def _neighbor_candidates(
    graph: ConceptGraph,
    concepts: dict[str, dict[str, Any]],
    name: str,
    *,
    max_neighbors: int,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for nbr, kind, weight in graph.neighbors(name):
        if nbr not in concepts:
            continue
        bonus = 0.5 if kind != "co_activation" else 0.0
        scored.append((float(weight or 1.0) + bonus + 0.001 * _usage_count(concepts[nbr]), nbr))
    scored.sort(reverse=True)
    out: list[str] = []
    for _, nbr in scored:
        if nbr not in out:
            out.append(nbr)
        if len(out) >= max_neighbors:
            break
    return out


def _build_prompt(
    concepts: dict[str, dict[str, Any]],
    name: str,
    neighbors: list[str],
    *,
    max_facts: int,
) -> str:
    neighbor_lines = "\n\n".join(_concept_digest(n, concepts[n]) for n in neighbors)
    if not neighbor_lines:
        neighbor_lines = "(none)"
    return f"""You extract OpenIE-style facts from ARC-AGI concept memory.

Return facts only. No prose, no markdown fences, no JSON.

Each output line must have exactly 7 fields separated by " || ":
subject || predicate || object || confidence || relation_kind || linked_concepts || supporting_text

Field rules:
- subject: short noun phrase
- predicate: short verb phrase
- object: short noun phrase
- confidence: number from 0.0 to 1.0
- relation_kind: one of uses, specializes, contrasts, co_occurs, parameterizes, same_operation, other
- linked_concepts: comma-separated concept names from the ALLOWED LINKED CONCEPTS list only
- supporting_text: exact or near-exact phrase from the concept fields

Rules:
- Extract {max_facts} or fewer facts.
- Include the source concept name in linked_concepts.
- Add 1 to 3 linked neighbor concepts only when the fact clearly relates them.
- Do not invent concept names. Use only the source concept or allowed linked concepts.
- Prefer operational ARC solver facts over generic definitions.

# SOURCE CONCEPT

{_concept_digest(name, concepts[name])}

# ALLOWED LINKED CONCEPTS

{neighbor_lines}

# OUTPUT EXAMPLE

{name} || uses || related operation || 0.9 || uses || {name} || phrase from source concept"""


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def _extract_json_array(raw: str) -> list[Any]:
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if isinstance(parsed, dict):
        for key in ("facts", "triples", "tuples"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
    if not isinstance(parsed, list):
        raise ValueError("OpenIE response was not a JSON list")
    return parsed


def _extract_line_items(raw: str) -> list[dict[str, Any]]:
    text = _strip_code_fence(raw)
    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.lstrip("-*0123456789. ").strip()
        fields = [field.strip() for field in line.split("||")]
        if len(fields) != 7:
            continue
        subject, predicate, obj, confidence, relation_kind, linked, supporting_text = fields
        linked_concepts = [c.strip() for c in linked.split(",") if c.strip()]
        items.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "relation_kind": relation_kind,
            "linked_concepts": linked_concepts,
            "supporting_text": supporting_text,
        })
    return items


def _extract_openie_items(raw: str) -> list[Any]:
    line_items = _extract_line_items(raw)
    if line_items:
        return line_items
    return _extract_json_array(raw)


def _fallback_linked(
    raw_fact: dict[str, Any],
    source: str,
    allowed: list[str],
    concepts: dict[str, dict[str, Any]],
) -> list[str]:
    text = " ".join(
        str(raw_fact.get(k) or "")
        for k in ("subject", "predicate", "object", "supporting_text")
    )
    fact_tokens = _tokenize(text)
    scored: list[tuple[int, str]] = []
    for name in allowed:
        raw = concepts[name]
        concept_text = " ".join(
            str(raw.get(k) or "")
            for k in ("name", "description", "kind", "routine_subtype", "output_typing")
        )
        score = len(fact_tokens & _tokenize(concept_text))
        if score > 0:
            scored.append((score, name))
    scored.sort(reverse=True)
    linked = [source]
    for _, name in scored[:2]:
        if name not in linked:
            linked.append(name)
    return linked


def _clean_facts(
    raw_items: list[Any],
    source: str,
    allowed: list[str],
    concepts: dict[str, dict[str, Any]],
    *,
    start_idx: int,
    max_facts: int,
) -> list[dict[str, Any]]:
    allowed_set = set(allowed)
    allowed_set.add(source)
    cleaned: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        subject = str(raw.get("subject") or "").strip()
        predicate = str(raw.get("predicate") or "").strip()
        obj = str(raw.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        linked_raw = raw.get("linked_concepts") or []
        linked = [
            str(c).strip()
            for c in linked_raw
            if isinstance(c, str) and str(c).strip() in allowed_set
        ]
        if source not in linked:
            linked.insert(0, source)
        if len(set(linked)) < 2:
            linked = _fallback_linked(raw, source, allowed, concepts)
        confidence = raw.get("confidence", 1.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 1.0
        relation_kind = str(raw.get("relation_kind") or "other").strip()
        if relation_kind not in RELATION_KINDS:
            relation_kind = "other"
        cleaned.append({
            "fact_id": f"openie_{start_idx + len(cleaned):05d}",
            "source_concept": source,
            "subject": subject[:180],
            "predicate": predicate[:120],
            "object": obj[:180],
            "confidence": confidence_value,
            "supporting_text": str(raw.get("supporting_text") or "")[:300],
            "linked_concepts": list(dict.fromkeys(linked)),
            "relation_kind": relation_kind,
        })
        if len(cleaned) >= max_facts:
            break
    return cleaned


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
    if not SEED_MEM.exists():
        print(f"ERROR: seed memory not found: {SEED_MEM}", file=sys.stderr)
        return 2

    seed_payload = json.loads(SEED_MEM.read_text())
    mem = ConceptMemory.from_payload(seed_payload)
    concepts: dict[str, dict[str, Any]] = seed_payload.get("concepts", {})
    if not mem.concepts:
        print("ERROR: no concepts loaded", file=sys.stderr)
        return 2

    graph = ConceptGraph.build_from_memory(
        mem,
        min_co_overlap=1,
        load_openie_edges=False,
    )
    names = sorted(mem.concepts.keys())
    if args.limit_concepts:
        names = names[: args.limit_concepts]
    neighbors_by_name = {
        name: _neighbor_candidates(graph, concepts, name, max_neighbors=args.max_neighbors)
        for name in names
    }
    prompts = [
        _build_prompt(
            concepts,
            name,
            neighbors_by_name[name],
            max_facts=args.max_facts_per_concept,
        )
        for name in names
    ]
    print(f"[openie_facts] concepts={len(names)} model={MODEL}")

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
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "batch_size": args.concurrency,
            "ignore_cache": args.ignore_cache,
        },
        request_timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()

    facts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    facts_per_concept: list[int] = []
    for name, completions in zip(names, results, strict=True):
        raw = completions[0] if completions else ""
        error: str | None = None
        try:
            raw_items = _extract_openie_items(raw or "")
            cleaned = _clean_facts(
                raw_items,
                name,
                neighbors_by_name[name],
                concepts,
                start_idx=len(facts),
                max_facts=args.max_facts_per_concept,
            )
        except Exception as exc:
            error = str(exc)[:240]
            cleaned = []
        if not cleaned:
            failures.append({"source_concept": name, "error": error or "no_valid_facts"})
        facts.extend(cleaned)
        facts_per_concept.append(len(cleaned))

    linked_edge_count = sum(max(0, len(set(f.get("linked_concepts", []))) - 1) for f in facts)
    out = {
        "schema_version": "1",
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "facts": facts,
        "stats": {
            "num_concepts": len(names),
            "num_facts": len(facts),
            "facts_per_concept_min": min(facts_per_concept) if facts_per_concept else 0,
            "facts_per_concept_max": max(facts_per_concept) if facts_per_concept else 0,
            "facts_per_concept_mean": statistics.fmean(facts_per_concept) if facts_per_concept else 0.0,
            "num_linked_edges": linked_edge_count,
            "num_failures": len(failures),
            "failures": failures[:20],
            "wall_time_s": elapsed,
            "estimated_cost_usd": _estimate_cost(usage),
            "token_usage": usage,
        },
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[openie_facts] wrote {OUT_FILE}")
    print(
        f"[openie_facts] facts={len(facts)} failures={len(failures)} "
        f"cost=${out['stats']['estimated_cost_usd']:.4f}"
    )
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        print(
            f"ERROR: estimated cost ${out['stats']['estimated_cost_usd']:.4f} "
            f"exceeded limit ${args.max_cost_usd:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0 if facts else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--max-facts-per-concept", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
