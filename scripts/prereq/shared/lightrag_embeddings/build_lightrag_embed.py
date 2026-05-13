"""Build LightRAG dual-graph dense embeddings for concepts and entities.

Outputs:
  data/arc_agi/concept_memory/shared/lightrag_embed_v1.npz
  data/arc_agi/concept_memory/shared/lightrag_embed_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.concepts.memory import ConceptMemory


SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
ENTITY_GRAPH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "entity_graph_v1.json"
OUT_NPZ = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "lightrag_embed_v1.npz"
OUT_META = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "lightrag_embed_v1.json"
MODEL = "text-embedding-3-small"
INPUT_COST_PER_M = 0.02


def _concept_text(name: str, raw: dict[str, Any]) -> str:
    parts = [name, str(raw.get("kind") or "")]
    for key in ("description", "routine_subtype", "output_typing"):
        if raw.get(key):
            parts.append(str(raw[key]))
    for key in ("cues", "implementation"):
        values = [str(v) for v in (raw.get(key) or [])[:6]]
        if values:
            parts.append(f"{key}: " + "; ".join(values))
    return " | ".join(parts)


def _entity_text(raw: dict[str, Any]) -> str:
    attrs = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    attr_text = "; ".join(f"{k}={v}" for k, v in list(attrs.items())[:6])
    return " | ".join([
        str(raw.get("mention_text") or ""),
        str(raw.get("entity_type") or "other"),
        str(raw.get("source_concept") or ""),
        str(raw.get("supporting_text") or ""),
        attr_text,
    ])


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _embed_texts(client: OpenAI, texts: list[str], *, batch_size: int) -> tuple[np.ndarray, int]:
    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
        if resp.usage:
            tokens += int(resp.usage.prompt_tokens or 0)
        print(f"[lightrag_embed] embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return _normalize(np.asarray(vectors, dtype=np.float32)), tokens


def main() -> int:
    args = parse_args()
    if (OUT_NPZ.exists() or OUT_META.exists()) and not args.force:
        print("ERROR: output already exists; pass --force to overwrite", file=sys.stderr)
        return 2
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY missing", file=sys.stderr)
        return 2
    seed = json.loads(SEED_MEM.read_text())
    mem = ConceptMemory.from_payload(seed)
    concepts = seed.get("concepts", {})
    concept_ids = sorted(mem.concepts.keys())
    concept_texts = [_concept_text(name, concepts[name]) for name in concept_ids]

    entity_data = json.loads(ENTITY_GRAPH.read_text())
    entities = [
        raw for raw in entity_data.get("entities", []) or []
        if isinstance(raw, dict) and raw.get("entity_id")
    ]
    entity_keys = [
        f"{raw.get('entity_id')}|{raw.get('source_concept')}|{raw.get('mention_text')}"
        for raw in entities
    ]
    entity_sources = [str(raw.get("source_concept") or "") for raw in entities]
    entity_mentions = [str(raw.get("mention_text") or "") for raw in entities]
    entity_types = [str(raw.get("entity_type") or "other") for raw in entities]
    entity_texts = [_entity_text(raw) for raw in entities]

    client = OpenAI()
    t0 = time.monotonic()
    concept_embeddings, concept_tokens = _embed_texts(
        client, concept_texts, batch_size=args.batch_size,
    )
    entity_embeddings, entity_tokens = _embed_texts(
        client, entity_texts, batch_size=args.batch_size,
    )
    dim = int(concept_embeddings.shape[1])
    if entity_embeddings.shape[1] != dim:
        print("ERROR: concept/entity embedding dims differ", file=sys.stderr)
        return 1

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        concept_embeddings=concept_embeddings,
        entity_embeddings=entity_embeddings,
    )
    total_tokens = concept_tokens + entity_tokens
    cost = (total_tokens / 1_000_000.0) * INPUT_COST_PER_M
    meta = {
        "schema_version": "1",
        "model": MODEL,
        "dim": dim,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "concept_ids": concept_ids,
        "entity_keys": entity_keys,
        "entity_sources": entity_sources,
        "entity_mentions": entity_mentions,
        "entity_types": entity_types,
        "stats": {
            "num_concepts": len(concept_ids),
            "num_entities": len(entity_keys),
            "dim": dim,
            "llm_calls": 0,
            "embedding_calls": (len(concept_texts) + args.batch_size - 1) // args.batch_size
            + (len(entity_texts) + args.batch_size - 1) // args.batch_size,
            "input_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "wall_time_s": time.monotonic() - t0,
            "faiss_index": "IndexFlatIP-compatible normalized vectors",
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[lightrag_embed] wrote {OUT_NPZ}")
    print(f"[lightrag_embed] wrote {OUT_META}")
    print(
        f"[lightrag_embed] concepts={len(concept_ids)} entities={len(entity_keys)} "
        f"dim={dim} tokens={total_tokens} cost=${cost:.4f}"
    )
    if cost > args.max_cost_usd:
        print(
            f"ERROR: estimated cost ${cost:.4f} exceeded limit ${args.max_cost_usd:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cost-usd", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
