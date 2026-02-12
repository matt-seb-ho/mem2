from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mem2.core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ArcMemoCompatRequirement:
    path: str
    expected: Any


# Inference-critical settings copied from arc_memo defaults:
# - configs/generation/gen_default.yaml
# - configs/default.yaml (retry policy defaults)
# - configs/model/gpt41.yaml
ARCMEMO_INFERENCE_REQUIREMENTS: tuple[ArcMemoCompatRequirement, ...] = (
    ArcMemoCompatRequirement("pipeline.provider", "llmplus_arcmemo_gpt41"),
    ArcMemoCompatRequirement("components.provider.profile_name", "llmplus_arcmemo_gpt41"),
    ArcMemoCompatRequirement("components.inference_engine.model", "gpt-4.1-2025-04-14"),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.n", 1),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.temperature", 0.3),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.max_tokens", 1024),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.top_p", 1.0),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.batch_size", 16),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.seed", 88),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.ignore_cache", False),
    ArcMemoCompatRequirement("components.inference_engine.gen_cfg.expand_multi", None),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.include_hint", False),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.hint_template_key", "selected"),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.require_hint_citations", False),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.instruction_key", "default"),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.system_prompt_key", "default"),
    ArcMemoCompatRequirement("run.execution_mode", "arc_batch"),
    ArcMemoCompatRequirement("run.retry_criterion", "train"),
    ArcMemoCompatRequirement("run.max_passes", 3),
    ArcMemoCompatRequirement("run.retry_policy.max_passes", 3),
    ArcMemoCompatRequirement("run.retry_policy.criterion", "train"),
    ArcMemoCompatRequirement("run.retry_policy.error_feedback", "all"),
    ArcMemoCompatRequirement("run.retry_policy.num_feedback_passes", 1),
    ArcMemoCompatRequirement("run.retry_policy.include_past_outcomes", True),
    ArcMemoCompatRequirement("components.inference_engine.num_feedback_passes", 1),
    ArcMemoCompatRequirement("components.inference_engine.error_feedback", "all"),
    ArcMemoCompatRequirement("components.inference_engine.include_past_outcomes", True),
    ArcMemoCompatRequirement("components.inference_engine.include_reselected_lessons", False),
)

# Logic-parity requirements:
# - enforce ArcMemo mechanics while allowing provider/model choice.
ARCMEMO_LOGIC_REQUIREMENTS: tuple[ArcMemoCompatRequirement, ...] = (
    ArcMemoCompatRequirement("run.execution_mode", "arc_batch"),
    ArcMemoCompatRequirement("run.retry_criterion", "train"),
    ArcMemoCompatRequirement("run.retry_policy.criterion", "train"),
    ArcMemoCompatRequirement("run.retry_policy.error_feedback", "all"),
    ArcMemoCompatRequirement("run.retry_policy.num_feedback_passes", 1),
    ArcMemoCompatRequirement("run.retry_policy.include_past_outcomes", True),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.include_hint", False),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.hint_template_key", "selected"),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.require_hint_citations", False),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.instruction_key", "default"),
    ArcMemoCompatRequirement("components.inference_engine.prompt_options.system_prompt_key", "default"),
    ArcMemoCompatRequirement("components.inference_engine.num_feedback_passes", 1),
    ArcMemoCompatRequirement("components.inference_engine.error_feedback", "all"),
    ArcMemoCompatRequirement("components.inference_engine.include_past_outcomes", True),
    ArcMemoCompatRequirement("components.inference_engine.include_reselected_lessons", False),
)


def _get_by_dotted_path(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _validate_requirements(
    *,
    config: dict[str, Any],
    enabled_flag: str,
    requirements: tuple[ArcMemoCompatRequirement, ...],
    label: str,
    disable_hint: str,
) -> None:
    enabled = bool(config.get("run", {}).get(enabled_flag, False))
    if not enabled:
        return

    mismatches: list[str] = []
    for req in requirements:
        got = _get_by_dotted_path(config, req.path)
        if got != req.expected:
            mismatches.append(f"- {req.path}: expected={req.expected!r}, got={got!r}")

    if mismatches:
        details = "\n".join(mismatches)
        raise ConfigurationError(
            f"ArcMemo {label} compatibility check failed.\n"
            "The following settings drift from ArcMemo-compatible defaults:\n"
            f"{details}\n"
            f"{disable_hint}"
        )


def validate_arcmemo_compat_config(config: dict[str, Any]) -> None:
    _validate_requirements(
        config=config,
        enabled_flag="strict_arcmemo_compat",
        requirements=ARCMEMO_INFERENCE_REQUIREMENTS,
        label="strict",
        disable_hint="Set run.strict_arcmemo_compat=false to disable this guard.",
    )


def validate_arcmemo_logic_compat_config(config: dict[str, Any]) -> None:
    _validate_requirements(
        config=config,
        enabled_flag="strict_arcmemo_logic_compat",
        requirements=ARCMEMO_LOGIC_REQUIREMENTS,
        label="logic",
        disable_hint="Set run.strict_arcmemo_logic_compat=false to disable this guard.",
    )
