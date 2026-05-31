"""ACE (Agentic Context Engineering, Zhao 2025) - thin mem2 port.

Port choices (documented gaps vs upstream/paper):
- Generator stage: skipped. mem2's inference engine is the analog.
- Bullet retrieval: helpful_count desc + max_bullets cap (paper unspecified).
- Semantic de-duplication: NOT in this port. Upstream `BulletpointAnalyzer`
  uses sentence-transformers + faiss; we keep the port dependency-free.
  Paper's grow-and-refine claim relies on semantic dedup (arXiv 2510.04618
  section 3 "Grow-and-Refine"). This is a documented limitation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)

logger = logging.getLogger(__name__)


def parse_playbook_line(line):
    """Parse a single playbook line to extract components"""
    # Pattern: [id] helpful=X harmful=Y :: content
    pattern = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern, line.strip())
    
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
            'raw_line': line
        }
    return None

def get_next_global_id(playbook_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = playbook_text.strip().split('\n')
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r'-(\d+)$', parsed['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
    
    return max_id + 1


def format_playbook_line(bullet_id, helpful, harmful, content):
    """Format a bullet into playbook line format"""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"

def update_bullet_counts(playbook_text, bullet_tags):
    """Update helpful/harmful counts based on tags (Counter layer)"""
    lines = playbook_text.strip().split('\n')
    updated_lines = []
    
    # Create tag lookup - handle both old and new formats
    tag_map = {}
    if isinstance(bullet_tags, list) and len(bullet_tags) > 0:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                # Handle both 'id' and 'bullet' keys for backwards compatibility
                bullet_id = tag.get('id') or tag.get('bullet', '')
                tag_value = tag.get('tag', 'neutral')
                if bullet_id:
                    tag_map[bullet_id] = tag_value
    
    if not tag_map:
        print("Warning: No valid bullet tags found to update counts")
        return playbook_text
    
    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            # Preserve section headers and empty lines
            updated_lines.append(line)
            continue
            
        parsed = parse_playbook_line(line)
        if parsed and parsed['id'] in tag_map:
            tag = tag_map[parsed['id']]
            if tag == 'helpful':
                parsed['helpful'] += 1
            elif tag == 'harmful':
                parsed['harmful'] += 1
            # neutral: no change
            
            # Reconstruct line with updated counts
            new_line = format_playbook_line(
                parsed['id'], parsed['helpful'], parsed['harmful'], parsed['content']
            )
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


REFLECTOR_PROMPT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong by analyzing the gap between predicted answer and the ground truth.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently
- You will receive bulletpoints that are part of playbook that's used by the generator to answer the question.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the generator to generate the correct answer)

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
  - bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator




**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer:**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator to answer the question:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""

REFLECTOR_PROMPT_NO_GT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong when coming up the predicted answer.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently
- You will receive bulletpoints that are part of playbook that's used by the generator to answer the question.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the generator to generate the correct answer)

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
  - bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator




**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator to answer the question:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""

CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth. 

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""

CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _problem_context(problem: ProblemSpec | None) -> str:
    if problem is None:
        return ""
    if problem.metadata.get("problem_text"):
        return str(problem.metadata["problem_text"])
    return _json_dumps(
        {
            "uid": problem.uid,
            "train_pairs": problem.train_pairs,
            "test_pairs": problem.test_pairs,
            "metadata": problem.metadata,
        }
    )


def _infer_problem(ctx: RunContext, attempt: AttemptRecord) -> ProblemSpec | None:
    problems = (ctx.config or {}).get("_ace_problems")
    if isinstance(problems, dict):
        problem = problems.get(attempt.problem_uid)
        if isinstance(problem, ProblemSpec):
            return problem
    return None


def _predicted_answer(attempt: AttemptRecord) -> str:
    for key in ("predicted_answer", "answer", "final_answer"):
        if key in attempt.metadata:
            return str(attempt.metadata[key])
    return attempt.completion


def _ground_truth(eval_record: EvalRecord) -> str:
    for key in ("ground_truth", "expected", "answer", "target"):
        if key in eval_record.metadata:
            return str(eval_record.metadata[key])
    return _json_dumps(eval_record.test_details)


def _feedback_text(feedback_records: list[FeedbackRecord]) -> str:
    return "\n".join(record.content for record in feedback_records if record.content)


def _reflection_summary(reflection: dict[str, Any]) -> str:
    keep = {
        "error_identification": reflection.get("error_identification", ""),
        "root_cause_analysis": reflection.get("root_cause_analysis", ""),
        "correct_approach": reflection.get("correct_approach", ""),
        "key_insight": reflection.get("key_insight", ""),
    }
    return _json_dumps(keep)


def _playbook_stats(playbook_text: str) -> str:
    parsed = [parse_playbook_line(line) for line in playbook_text.strip().split("\n")]
    bullets = [p for p in parsed if p is not None]
    return _json_dumps(
        {
            "bullet_count": len(bullets),
            "helpful_total": sum(int(p["helpful"]) for p in bullets),
            "harmful_total": sum(int(p["harmful"]) for p in bullets),
        }
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _call_provider(provider, prompt: str, *, model: str, gen_cfg: dict[str, Any]) -> str | None:
    if provider is None:
        return None
    try:
        generate = getattr(provider, "generate")
    except AttributeError:
        logger.warning("ACE provider has no sync generate method; skipping call")
        return None
    try:
        completions = generate(prompt, model=model, **gen_cfg)
    except TypeError:
        completions = generate(prompt, model=model)
    except Exception as exc:  # pragma: no cover - provider-dependent
        logger.warning("ACE provider call failed: %s: %s", type(exc).__name__, exc)
        return None
    if not completions:
        return None
    return str(completions[0])


def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])


def _append_add_operations(playbook_text: str, operations: list[dict[str, Any]]) -> str:
    lines = [line for line in playbook_text.splitlines() if line.strip()]
    for op in operations:
        if not isinstance(op, dict) or str(op.get("type", "")).upper() != "ADD":
            continue
        content = str(op.get("content") or "").strip()
        if not content:
            continue
        slug = get_section_slug(str(op.get("section") or "general"))
        bullet_id = f"{slug}-{get_next_global_id(chr(10).join(lines)):05d}"
        lines.append(format_playbook_line(bullet_id, 0, 0, content))
    return "\n".join(lines)


def _trim_playbook(playbook_text: str, max_bullets: int) -> str:
    if max_bullets <= 0:
        return playbook_text.strip()
    lines = [line for line in playbook_text.splitlines() if line.strip()]
    parsed_rows = [(i, parse_playbook_line(line)) for i, line in enumerate(lines)]
    bullet_rows = [(i, p) for i, p in parsed_rows if p is not None]
    if len(bullet_rows) <= max_bullets:
        return "\n".join(lines)
    keep = {
        i for i, _ in sorted(
            bullet_rows,
            key=lambda row: (-int(row[1]["helpful"]), int(row[1]["harmful"]), row[0]),
        )[:max_bullets]
    }
    return "\n".join(line for i, line in enumerate(lines) if i in keep)


class AceMemoryBuilder:
    name = "ace"
    SCHEMA_NAME = "ace"

    def __init__(
        self,
        max_bullets: int = 50,
        reflector_max_lessons: int = 3,
        freeze: bool = False,
        reflector_model: str = "",
        curator_model: str = "",
        reflector_gen_cfg: dict[str, Any] | None = None,
        curator_gen_cfg: dict[str, Any] | None = None,
        playbook_token_budget: int = 80000,
        use_ground_truth: bool = True,
        num_epochs: int = 1,
        seed_memory_file: str | None = None,
    ):
        self.max_bullets = int(max_bullets)
        self.reflector_max_lessons = max(0, int(reflector_max_lessons))
        self.freeze = bool(freeze)
        self.reflector_model = reflector_model
        self.curator_model = curator_model
        self.playbook_token_budget = int(playbook_token_budget)
        self.use_ground_truth = bool(use_ground_truth)
        self.num_epochs = max(1, int(num_epochs))
        self.seed_memory_file = seed_memory_file
        self.reflector_gen_cfg = dict(
            reflector_gen_cfg or {"n": 1, "temperature": 0.0, "max_tokens": 2048}
        )
        self.curator_gen_cfg = dict(
            curator_gen_cfg or {"n": 1, "temperature": 0.0, "max_tokens": 2048}
        )

    def _select_reflector_prompt(self) -> str:
        return REFLECTOR_PROMPT if self.use_ground_truth else REFLECTOR_PROMPT_NO_GT

    def _select_curator_prompt(self) -> str:
        return CURATOR_PROMPT if self.use_ground_truth else CURATOR_PROMPT_NO_GT

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        playbook_text = ""
        seed_metadata: dict[str, Any] = {}
        if self.seed_memory_file:
            from pathlib import Path
            seed_path = Path(self.seed_memory_file).expanduser()
            if not seed_path.is_absolute():
                seed_path = Path.cwd() / seed_path
            if seed_path.exists():
                try:
                    seed_blob = json.loads(seed_path.read_text(encoding="utf-8"))
                    if isinstance(seed_blob, dict):
                        payload = seed_blob.get("payload") or {}
                        playbook_text = str(payload.get("playbook_text") or "")
                        seed_metadata["seed_source"] = str(self.seed_memory_file)
                        seed_metadata["seed_schema_version"] = str(
                            seed_blob.get("schema_version") or ""
                        )
                        # Sanity counter — number of parseable bullets in the seed.
                        parsed = [
                            parse_playbook_line(ln)
                            for ln in playbook_text.splitlines()
                            if ln.strip()
                        ]
                        seed_metadata["seed_bullet_count"] = sum(
                            1 for p in parsed if p is not None
                        )
                except Exception as exc:  # pragma: no cover - configuration fault
                    seed_metadata["seed_load_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        return MemoryState(
            schema_name="ace",
            schema_version="v1",
            payload={"playbook_text": playbook_text},
            metadata={
                "initialized_problem_count": len(problems),
                **seed_metadata,
            },
        )

    def _resolve_provider(self, ctx: RunContext, role: str):
        config = ctx.config or {}
        return (
            config.get(f"_ace_{role}_provider")
            or config.get("_ace_provider")
            or config.get("_meta_edit_provider")
        )

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        if self.freeze:
            return memory

        reflector = self._resolve_provider(ctx, "reflector")
        curator = self._resolve_provider(ctx, "curator")
        playbook_text = str(memory.payload.get("playbook_text") or "")
        errors: list[dict[str, str]] = []

        feedback_by_attempt = {
            (record.problem_uid, record.attempt_idx): record
            for record in feedback_records
        }

        total_samples = len(attempts) * self.num_epochs
        for epoch in range(self.num_epochs):
            for idx, attempt in enumerate(attempts):
                if idx >= len(eval_records):
                    continue
                eval_record = eval_records[idx]
                problem = _infer_problem(ctx, attempt)
                question = _problem_context(problem) or attempt.prompt
                feedback = _feedback_text(
                    [feedback_by_attempt[(attempt.problem_uid, eval_record.attempt_idx)]]
                    if (attempt.problem_uid, eval_record.attempt_idx) in feedback_by_attempt
                    else []
                )
                if self.use_ground_truth:
                    reflector_prompt = self._select_reflector_prompt().format(
                        question,
                        attempt.completion,
                        _predicted_answer(attempt),
                        _ground_truth(eval_record),
                        feedback,
                        playbook_text,
                    )
                else:
                    reflector_prompt = self._select_reflector_prompt().format(
                        question,
                        attempt.completion,
                        _predicted_answer(attempt),
                        feedback,
                        playbook_text,
                    )
                reflection_text = _call_provider(
                    reflector,
                    reflector_prompt,
                    model=self.reflector_model,
                    gen_cfg=self.reflector_gen_cfg,
                )
                reflection = _extract_json_object(reflection_text or "")
                if reflection is None:
                    errors.append({"stage": "reflector", "problem_uid": attempt.problem_uid})
                    continue

                bullet_tags = reflection.get("bullet_tags", [])
                if isinstance(bullet_tags, list):
                    bullet_tags = bullet_tags[: self.reflector_max_lessons]
                playbook_text = update_bullet_counts(playbook_text, bullet_tags)

                curator_prompt = self._select_curator_prompt().format(
                    token_budget=self.playbook_token_budget,
                    current_step=epoch * len(attempts) + idx + 1,
                    total_samples=total_samples,
                    playbook_stats=_playbook_stats(playbook_text),
                    recent_reflection=_reflection_summary(reflection),
                    current_playbook=playbook_text,
                    question_context=question,
                )
                curator_text = _call_provider(
                    curator,
                    curator_prompt,
                    model=self.curator_model,
                    gen_cfg=self.curator_gen_cfg,
                )
                curator_result = _extract_json_object(curator_text or "")
                if curator_result is None:
                    errors.append({"stage": "curator", "problem_uid": attempt.problem_uid})
                    continue

                operations = curator_result.get("operations", [])
                if isinstance(operations, list):
                    playbook_text = _append_add_operations(playbook_text, operations)
                    playbook_text = _trim_playbook(playbook_text, self.max_bullets)

        memory.payload["playbook_text"] = playbook_text
        if errors:
            memory.metadata["ace_parse_errors"] = errors
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        return memory
