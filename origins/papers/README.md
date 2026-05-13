# Origins Papers

This is the target home for migrated source PDFs. The actual `literature/*.pdf` move is intentionally deferred as of 2026-05-13.

## Why Deferred

- `origin/main` is still receiving adapted-memory commits from parallel panes.
- Several paper path references still sit in files owned by the adapter panes, including memory retriever files this pass must not edit.
- Moving the PDFs now would either break those panes or require edits outside this assignment's write boundary.

## Future Migration Checklist

When adapter panes are quiet:

```bash
mkdir -p origins/papers/
git mv literature/*.pdf origins/papers/
rg -n "literature/|Path\\(['\\\"]literature|literature" src scripts tests case_studies analysis origins benchmarks configs
```

Then update any source references that still point to `literature/*.pdf`, while preserving `literature/README.md` or `literature/MANIFEST.md` as breadcrumbs if they exist.
