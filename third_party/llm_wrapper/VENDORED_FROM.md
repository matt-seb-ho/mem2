# Vendored Source Note

`llmplus` in this folder is vendored from:

- local source path: `/root/workspace/llm_wrapper/llmplus`
- upstream project URL: `https://github.com/matt-seb-ho/llm_wrapper`

Additional local sync source:

- fork path: `/root/workspace/my_fork/llm_wrapper/llmplus`
- merged updates:
  - OpenAI `gpt-5` model metadata and OpenRouter `google/gemini-2.5-flash-lite-preview-09-2025`
  - robust handling of `None` completion content
  - per-request token accounting support (`include_per_request=True`)

## Pinning policy

Before release, replace this note with:

1. exact upstream commit SHA,
2. vendoring date,
3. list of local modifications (if any).

This keeps provider behavior reproducible and prevents silent drift.
