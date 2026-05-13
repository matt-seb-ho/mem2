"""Build sentence-level concept embeddings for ColBERT-rerank (axis 1).

Why this exists
---------------
The shipped `colbert_rerank` retriever uses lowercase-token exact match
in place of BERT contextual embeddings. ColBERT's distinctive late-
interaction MaxSim mechanism degenerates to "does the query word appear
literally in this concept's text?" — equivalent to keyword overlap.

This script encodes each concept's text with a small sentence-transformer
(all-MiniLM-L6-v2, ~90MB, runs locally on CPU/MPS) and saves the dense
vectors so the retriever can do real cosine-similarity ranking.

Note: this is concept-level (one vector per concept), not full ColBERT's
token-level (multiple vectors per concept). Trade-off: simpler, faster,
no ColBERT-specific index. We lose the per-token MaxSim nuance but
recover the embedding-vs-keyword distinction the paper depends on.

Inputs
------
- mem2/data/arc_agi/concept_memory/compressed_v1.json (seed memory)

Outputs
-------
- mem2/data/arc_agi/concept_memory/concept_embeddings_v1.npz
  Schema:
    - "names": np.array of concept names (sorted alphabetically), shape (N,)
    - "embeddings": float32 np.array, shape (N, dim)
    - "model": str (HF model id)
    - "dim": int
    - "built_at": str (ISO timestamp)

- Sidecar JSON metadata: concept_embeddings_v1.meta.json

Cost / runtime
--------------
~0 (free, local). all-MiniLM-L6-v2 on Apple Silicon MPS encodes 270 short
texts in seconds. First run downloads model (~90MB).

Usage
-----
  cd mem2
  .venv/bin/python scripts/prereq/shared/colbert_rerank/build_concept_embeddings.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_NPZ = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_embeddings_v1.npz"
OUT_META = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_embeddings_v1.meta.json"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def concept_text(c: dict) -> str:
    """Compose a concept into a single text string for embedding."""
    parts = [c.get("name", "")]
    if c.get("kind"):
        parts.append(f"kind: {c['kind']}")
    if c.get("description"):
        parts.append(c["description"])
    cues = c.get("cues") or []
    if cues:
        parts.append("cues: " + "; ".join(str(x) for x in cues[:5]))
    impl = c.get("implementation") or []
    if impl:
        parts.append("uses: " + "; ".join(str(x) for x in impl[:5]))
    return " | ".join(p for p in parts if p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF sentence-transformer model id")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda; default auto")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not SEED_MEM.exists():
        print(f"ERROR: seed memory not found at {SEED_MEM}", file=sys.stderr)
        return 2

    seed = json.loads(SEED_MEM.read_text())
    concepts = seed.get("concepts", {})
    if not concepts:
        print("ERROR: no concepts in seed memory", file=sys.stderr)
        return 2

    sorted_names = sorted(concepts.keys())
    texts = [concept_text(concepts[n]) for n in sorted_names]
    print(f"[build_emb] {len(sorted_names)} concepts → preparing texts (mean len={int(np.mean([len(t) for t in texts]))} chars)")

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    print(f"[build_emb] device={device}, model={args.model}")

    # Load model
    t_load = time.monotonic()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=device)
    print(f"[build_emb] model loaded in {time.monotonic() - t_load:.1f}s")

    # Encode
    t_enc = time.monotonic()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    enc_time = time.monotonic() - t_enc
    embeddings = embeddings.astype(np.float32)
    print(f"[build_emb] encoded in {enc_time:.1f}s, shape={embeddings.shape}")

    # Save
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        names=np.array(sorted_names, dtype=object),
        embeddings=embeddings,
    )
    print(f"[build_emb] wrote {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / 1024:.1f} KB)")

    meta = {
        "schema_version": "1",
        "model": args.model,
        "dim": int(embeddings.shape[1]),
        "num_concepts": int(embeddings.shape[0]),
        "device": device,
        "encode_time_s": enc_time,
        "normalized": True,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[build_emb] meta: dim={meta['dim']}, n={meta['num_concepts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
