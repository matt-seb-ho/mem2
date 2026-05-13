from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from mem2.core.arcmemo_compat import (
    validate_arcmemo_compat_config,
    validate_arcmemo_logic_compat_config,
)
from mem2.core.contracts import (
    ArtifactSink,
    BenchmarkAdapter,
    Evaluator,
    FeedbackEngine,
    InferenceEngine,
    MemoryBuilder,
    MemoryRetriever,
    ProviderClient,
    Router,
    TaskAdapter,
    TrajectoryPolicy,
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
from mem2.registry.router import ROUTERS
from mem2.registry.trajectory_policy import TRAJECTORY_POLICIES


@dataclass(slots=True)
class PipelineComponents:
    task_adapter: TaskAdapter
    benchmark: BenchmarkAdapter
    memory_builder: MemoryBuilder
    memory_retriever: MemoryRetriever
    router: Router
    trajectory_policy: TrajectoryPolicy
    provider: ProviderClient
    inference_engine: InferenceEngine
    feedback_engine: FeedbackEngine
    evaluator: Evaluator
    artifact_sink: ArtifactSink



def _validate_memory_pairing(builder, retriever) -> None:
    builder_schema = getattr(builder, "SCHEMA_NAME", None)
    retriever_schemas = getattr(retriever, "COMPATIBLE_SCHEMAS", None)
    if builder_schema is None or retriever_schemas is None:
        return  # backward-compatible
    if builder_schema not in retriever_schemas:
        raise ConfigurationError(
            f"Memory builder '{builder.name}' produces schema '{builder_schema}', "
            f"but retriever '{retriever.name}' only supports {retriever_schemas}."
        )


def _validate_domain_components(benchmark, inference_engine, evaluator, feedback_engine) -> None:
    """Validate that all domain-specific components share the same DOMAIN_NAME."""
    domains: dict[str, str] = {}
    for label, comp in [
        ("benchmark", benchmark),
        ("inference_engine", inference_engine),
        ("evaluator", evaluator),
        ("feedback_engine", feedback_engine),
    ]:
        domain = getattr(comp, "DOMAIN_NAME", None)
        if domain is not None:
            domains[label] = domain
    if not domains:
        return  # backward-compatible: no domain declarations
    unique_domains = set(domains.values())
    if len(unique_domains) > 1:
        details = ", ".join(f"{k}='{v}'" for k, v in sorted(domains.items()))
        raise ConfigurationError(
            f"Domain mismatch across pipeline components: {details}. "
            f"All domain-specific components must share the same DOMAIN_NAME."
        )


def _build_component(registry: dict[str, Any], key: str, cfg: dict[str, Any]) -> Any:
    if key not in registry:
        known = ", ".join(sorted(registry.keys()))
        raise ConfigurationError(f"Unknown component '{key}'. Known: [{known}]")
    cls = registry[key]
    # Strip None values — YAML null means "unset this inherited key"
    kwargs = {k: v for k, v in cfg.items() if v is not None}
    # Validate kwargs against constructor signature
    sig = inspect.signature(cls.__init__)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_keyword:
        accepted = {
            name for name, p in sig.parameters.items()
            if name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        unknown = set(kwargs.keys()) - accepted
        if unknown:
            raise ConfigurationError(
                f"Unknown config params for '{key}': {sorted(unknown)}. "
                f"Accepted: {sorted(accepted)}"
            )
    return cls(**kwargs)



def resolve_components(config: dict[str, Any]) -> PipelineComponents:
    validate_arcmemo_compat_config(config)
    validate_arcmemo_logic_compat_config(config)

    pipe = config.get("pipeline", {})
    comp_cfg = config.get("components", {})

    memory_builder = _build_component(MEMORY_BUILDERS, pipe["memory_builder"], comp_cfg.get("memory_builder", {}))
    memory_retriever = _build_component(MEMORY_RETRIEVERS, pipe["memory_retriever"], comp_cfg.get("memory_retriever", {}))
    _validate_memory_pairing(memory_builder, memory_retriever)

    benchmark = _build_component(BENCHMARKS, pipe["benchmark"], comp_cfg.get("benchmark", {}))
    inference_engine = _build_component(INFERENCE_ENGINES, pipe["inference_engine"], comp_cfg.get("inference_engine", {}))
    evaluator = _build_component(EVALUATORS, pipe["evaluator"], comp_cfg.get("evaluator", {}))
    feedback_engine = _build_component(FEEDBACK_ENGINES, pipe["feedback_engine"], comp_cfg.get("feedback_engine", {}))
    _validate_domain_components(benchmark, inference_engine, evaluator, feedback_engine)

    router = _build_component(ROUTERS, pipe.get("router", "none"), comp_cfg.get("router", {}))
    provider = _build_component(PROVIDERS, pipe["provider"], comp_cfg.get("provider", {}))
    trace_dir = (
        comp_cfg.get("provider", {}).get("trace_dir")
        or config.get("case_studies", {}).get("trace_dir")
    )
    if trace_dir:
        from case_studies._tracer import TraceCollectingProviderClient

        provider = TraceCollectingProviderClient(provider, trace_dir=trace_dir)

    return PipelineComponents(
        task_adapter=_build_component(TASK_ADAPTERS, pipe["task_adapter"], comp_cfg.get("task_adapter", {})),
        benchmark=benchmark,
        memory_builder=memory_builder,
        memory_retriever=memory_retriever,
        router=router,
        trajectory_policy=_build_component(TRAJECTORY_POLICIES, pipe["trajectory_policy"], comp_cfg.get("trajectory_policy", {})),
        provider=provider,
        inference_engine=inference_engine,
        feedback_engine=feedback_engine,
        evaluator=evaluator,
        artifact_sink=_build_component(ARTIFACT_SINKS, pipe["artifact_sink"], comp_cfg.get("artifact_sink", {})),
    )
