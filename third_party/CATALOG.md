# Third-Party Repo Catalog

Clone targets for Mem2 axis-candidate methods. Cloned with `git clone --depth 1` to save disk. Use these as **reference implementations** when building each axis candidate — do NOT implement from specs alone.

## Axis mapping

| Axis | Method | Local path | Upstream |
|------|--------|------------|----------|
| A. Reorg mechanism | DreamCoder wake-sleep | `third_party/dreamcoder/` | https://github.com/ellisk42/ec |
| A. Reorg mechanism | Stitch compression | `third_party/stitch/` | https://github.com/mlb2251/stitch |
| A. Reorg mechanism | LILO (DreamCoder+LLM) | `third_party/lilo/` | https://github.com/gabegrand/lilo |
| A. Reorg mechanism | ALMA (whole-arch memory search) | `third_party/alma/` | https://github.com/zksha/alma |
| B. Graph retrieval | HippoRAG / HippoRAG 2 | `third_party/hipporag/` | https://github.com/OSU-NLP-Group/HippoRAG |
| B. Graph retrieval | RAPTOR | `third_party/raptor/` | https://github.com/parthsarthi03/raptor |
| B. Graph retrieval | GraphRAG (Microsoft) | `third_party/graphrag/` | https://github.com/microsoft/graphrag |
| B. Graph retrieval | LightRAG | `third_party/lightrag/` | https://github.com/HKUDS/LightRAG |
| B. Graph retrieval | ColBERT (late interaction) | `third_party/colbert/` | https://github.com/stanford-futuredata/ColBERT |
| C. Interactive retrieval | AR-Bench (benchmark + base scaffold) | `../../RRMC/RRMC/AR-Bench/` (via RRMC workstation) | https://github.com/tmlr-group/AR-Bench |
| C. Interactive retrieval | RRMC (CEG method) | `../../RRMC/RRMC/` | internal |
| C. Interactive retrieval | MediQ (abstention-gated) | `third_party/mediq/` | https://github.com/stellalisy/mediQ |
| D. Concept format | DSPy (signature optimization) | `third_party/dspy/` | https://github.com/stanfordnlp/dspy |
| F. Architecture-edit source | ADAS (Meta Agent Search) | `third_party/adas/` | https://github.com/ShengranHu/ADAS |
| F. Architecture-edit source | ALMA (shared with axis A) | `third_party/alma/` | same |

## COLM 2026 rebuttal cycle (cloned 2026-05-27)

| Method | Local path | Upstream | Notes |
|--------|------------|----------|-------|
| Dynamic Cheatsheet (Suzgun 2025) | `third_party/dynamic-cheatsheet/` | https://github.com/suzgunmirac/dynamic-cheatsheet | DC-Cu (frozen) + DC-RS (retrieval+synthesis). Prompts at `prompts/{generator,curator_prompt_for_dc_cumulative,curator_prompt_for_dc_retrieval_synthesis}.txt`. Core at `dynamic_cheatsheet/language_model.py`. Approach names: `DynamicCheatsheet_Cumulative`, `DynamicCheatsheet_RetrievalSynthesis`, `FullHistoryAppending`, `Dynamic_Retrieval`. |
| ACE (Zhao 2025) | `third_party/ace/` | https://github.com/ace-agent/ace | 3-agent (Generator/Reflector/Curator). Bullet format `[id] helpful=X harmful=Y :: content` (see `ace/playbook_utils.py:parse_playbook_line`). Prompts at `ace/ace/prompts/{generator,reflector,curator}.py`. `EXTENDING_ACE.md` is the developer guide. |
| ReasoningBank (Ouyang 2025, Google) | `third_party/reasoning-bank/` | https://github.com/google-research/reasoning-bank | WebArena + SWE-Bench scope (NOT ARC). Core: `WebArena/induce_memory.py` (extraction), `WebArena/memory_management.py` (gemini-embedding-001 storage). Prompts: `WebArena/prompts/memory_instruction.py` (`SUCCESSFUL_SI`, `FAILED_SI` — Markdown format with `# Memory Item / ## Title / ## Description / ## Content`). |

## Not cloned (no public repo at time of setup, 2026-04-21)

These methods had no official public repo found. The mem2 copilot should try again on a future pass or implement from paper + literature/ PDF alone, with a flag that the implementation is spec-only.

- **PathRAG** (AAAI'26) — search for official release
- **G-Memory** — search for release
- **H-MEM** — search for release
- **MAGMA** (ACL'26) — search for release
- **EvolveR** — search for release
- **MemTree** — search for release
- **Memp** — search for release (paper in `literature/2508.06433_exploring_agent_procedural_memory.md`)
- **A-MEM** — search for release
- **Gödel Agent** — search for release
- **AFlow** — MetaGPT org may host it; check
- **UoT, QuestBench** — search for release
- **PARSE, FluxMem, MemAPO, GEPA** — search for release
- **TRM, HRM** — ARC Paper Prize winners; check author pages
- **SleepGate** — search for release

## Usage protocol

1. Before implementing an axis candidate, **read** its repo README + the paper PDF in `literature/<id>.pdf`.
2. Extract the core mechanism from the reference implementation (algorithm, data structures, prompt templates).
3. Adapt to mem2's branch contracts (don't copy wholesale — our architecture is different).
4. Cite the repo + paper in the branch module's docstring.
5. If a method in "Not cloned" is needed, attempt a web search for the repo before falling back to spec-only implementation. Log attempt in `docs/phase1_repo_fetch_attempts.md`.

## Disk cost

~1.4 GB total across the 12 cloned repos. LightRAG and GraphRAG are the largest. Safe to prune `--depth 1` clones after implementation lands.
