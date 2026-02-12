from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mem2.core.arcmemo_compat import (
    validate_arcmemo_compat_config,
    validate_arcmemo_logic_compat_config,
)
from mem2.core.errors import ConfigurationError
from mem2.registry.artifact_sink import ARTIFACT_SINKS
from mem2.registry.benchmark import BENCHMARKS
from mem2.registry.evaluator import EVALUATORS
from mem2.registry.feedback_engine import FEEDBACK_ENGINES
from mem2.registry.inference_engine import INFERENCE_ENGINES
from mem2.registry.memory_builder import MEMORY_BUILDERS
from mem2.registry.memory_retriever import MEMORY_RETRIEVERS
from mem2.registry.provider import PROVIDERS
from mem2.registry.task_adapter import TASK_ADAPTERS
from mem2.registry.trajectory_policy import TRAJECTORY_POLICIES


@dataclass(slots=True)
class PipelineComponents:
    task_adapter: Any
    benchmark: Any
    memory_builder: Any
    memory_retriever: Any
    trajectory_policy: Any
    provider: Any
    inference_engine: Any
    feedback_engine: Any
    evaluator: Any
    artifact_sink: Any



def _build_component(registry: dict[str, Any], key: str, cfg: dict[str, Any]) -> Any:
    if key not in registry:
        known = ", ".join(sorted(registry.keys()))
        raise ConfigurationError(f"Unknown component '{key}'. Known: [{known}]")
    cls = registry[key]
    kwargs = dict(cfg)
    return cls(**kwargs)



def resolve_components(config: dict[str, Any]) -> PipelineComponents:
    validate_arcmemo_compat_config(config)
    validate_arcmemo_logic_compat_config(config)

    pipe = config.get("pipeline", {})
    comp_cfg = config.get("components", {})

    return PipelineComponents(
        task_adapter=_build_component(TASK_ADAPTERS, pipe["task_adapter"], comp_cfg.get("task_adapter", {})),
        benchmark=_build_component(BENCHMARKS, pipe["benchmark"], comp_cfg.get("benchmark", {})),
        memory_builder=_build_component(MEMORY_BUILDERS, pipe["memory_builder"], comp_cfg.get("memory_builder", {})),
        memory_retriever=_build_component(MEMORY_RETRIEVERS, pipe["memory_retriever"], comp_cfg.get("memory_retriever", {})),
        trajectory_policy=_build_component(TRAJECTORY_POLICIES, pipe["trajectory_policy"], comp_cfg.get("trajectory_policy", {})),
        provider=_build_component(PROVIDERS, pipe["provider"], comp_cfg.get("provider", {})),
        inference_engine=_build_component(INFERENCE_ENGINES, pipe["inference_engine"], comp_cfg.get("inference_engine", {})),
        feedback_engine=_build_component(FEEDBACK_ENGINES, pipe["feedback_engine"], comp_cfg.get("feedback_engine", {})),
        evaluator=_build_component(EVALUATORS, pipe["evaluator"], comp_cfg.get("evaluator", {})),
        artifact_sink=_build_component(ARTIFACT_SINKS, pipe["artifact_sink"], comp_cfg.get("artifact_sink", {})),
    )
