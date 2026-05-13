# mem2 Origins

`origins/` tracks where research directions came from: Aaron's unpublished ideas, external papers, third-party repositories, prior survey insights, and future candidate threads. It is broader than `literature/`, which remains a legacy paper-file alias and must not be removed.

## What Counts as an Origin

- A paper or technical report that motivated a port or future thread.
- A third-party repository whose code or mechanism was inspected.
- Aaron's unpublished research direction or working intuition.
- A synthesis note that reframes the project, such as failure-typed querying.

## What Does Not Belong Here

- Raw run outputs, which belong in `outputs/` or `case_studies/runs/`.
- Live trace analysis, which belongs in `analysis/`.
- Active implementation code, which belongs under `src/mem2/` or scripts.

## Conventions

- Use `_index.md` as the source-tracking table.
- Put thread-level context under `threads/<thread>/`.
- Put per-paper markdown distillations under `distilled/`.
- Put notes about `third_party/<repo>/` under `external_repos/`.
- Keep `literature/` PDFs in place until a later migration explicitly moves them.
