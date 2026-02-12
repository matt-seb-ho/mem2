# Parity Baseline Freeze Procedure

This procedure freezes a reproducible baseline for ARC default migration parity.

## Inputs

- strict config: `configs/experiments/arcmemo_arc_strict.yaml`
- parity spec: `configs/parity/arcmemo_default_spec.yaml`
- prompt templates from `src/mem2/prompting/render.py`

## Command

```bash
python scripts/parity/freeze_parity_baseline.py
```

## Outputs

- `outputs/parity/parity_baseline_frozen.json`
- `configs/parity/parity_baseline_frozen.json`

Both files include:
- strict config hash,
- parity spec hash,
- prompt template hashes.
