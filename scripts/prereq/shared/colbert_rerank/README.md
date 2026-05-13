# axis_1 / colbert_rerank — prerequisite builder

## What this folder builds

Sentence-level concept embeddings for the ColBERT-rerank retriever
(axis 1.8).

Without it, `colbert_rerank` uses lowercase-token exact match instead of
BERT contextual embeddings — its distinctive late-interaction MaxSim
mechanism collapses to keyword overlap.

With it, the retriever does cosine-similarity ranking over dense vectors.

Note: this is concept-level (one vector per concept), not full ColBERT's
token-level. Trade-off accepted for simplicity. Documented as "Reduced"
fit in doc 52.

## Files

- `build_concept_embeddings.py` — encodes all 270 concepts using a small
  sentence-transformer (default: `all-MiniLM-L6-v2`, ~90MB, runs locally
  on Apple Silicon MPS / CPU).

## Inputs

- `mem2/data/arc_agi/concept_memory/compressed_v1.json` (seed memory)

## Outputs

- `mem2/data/arc_agi/concept_memory/concept_embeddings_v1.npz`
  - `names`: sorted concept names, shape (N,)
  - `embeddings`: float32 normalized vectors, shape (N, dim)
- `mem2/data/arc_agi/concept_memory/concept_embeddings_v1.meta.json`
  (model id, dim, build timestamp, etc.)

## Cost / runtime

Free, local. First run downloads the model (~90MB from HuggingFace).
Subsequent runs are seconds. Apple Silicon MPS is auto-detected.

## Usage

```bash
cd mem2
.venv/bin/python scripts/prereq/shared/colbert_rerank/build_concept_embeddings.py
```

Optional flags: `--model`, `--device`, `--batch-size`.

## Once it lands

The `colbert_rerank` retriever needs a one-time wiring change to load
`concept_embeddings_v1.npz` and compute cosine sim against query
embeddings (encoded on the fly with the same model). Tracked as a
follow-up.
