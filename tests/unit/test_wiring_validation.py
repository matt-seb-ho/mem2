"""Tests for wiring-time validation: domain triple, memory pairing, config params."""
from __future__ import annotations

import pytest

from mem2.core.errors import ConfigurationError
from mem2.orchestrator.wiring import (
    _build_component,
    _validate_domain_components,
    _validate_memory_pairing,
)


# ---------------------------------------------------------------------------
# Stub components for domain validation
# ---------------------------------------------------------------------------
class _ArcBenchmark:
    name = "arc_agi"
    DOMAIN_NAME = "arc"


class _MathBenchmark:
    name = "competition_math_ps"
    DOMAIN_NAME = "math"


class _CodeBenchmark:
    name = "livecodebench"
    DOMAIN_NAME = "code"


class _ArcIE:
    name = "python_transform_retry"
    DOMAIN_NAME = "arc"


class _MathIE:
    name = "math_ps_solve"
    DOMAIN_NAME = "math"


class _CodeIE:
    name = "lcb_solve"
    DOMAIN_NAME = "code"


class _ArcEvaluator:
    name = "arc_exec"
    DOMAIN_NAME = "arc"


class _MathEvaluator:
    name = "math_ps_exec"
    DOMAIN_NAME = "math"


class _CodeEvaluator:
    name = "lcb_exec"
    DOMAIN_NAME = "code"


class _ArcFeedback:
    name = "gt_check"
    DOMAIN_NAME = "arc"


class _MathFeedback:
    name = "math_ps_gt"
    DOMAIN_NAME = "math"


class _CodeFeedback:
    name = "lcb_gt"
    DOMAIN_NAME = "code"


class _NoDomainComponent:
    name = "custom"
    # No DOMAIN_NAME attribute


# ---------------------------------------------------------------------------
# Domain triple validation
# ---------------------------------------------------------------------------
class TestDomainValidation:
    def test_arc_triple_passes(self):
        """Matching ARC components should pass validation."""
        _validate_domain_components(
            _ArcBenchmark(), _ArcIE(), _ArcEvaluator(), _ArcFeedback()
        )

    def test_math_triple_passes(self):
        """Matching math components should pass validation."""
        _validate_domain_components(
            _MathBenchmark(), _MathIE(), _MathEvaluator(), _MathFeedback()
        )

    def test_code_triple_passes(self):
        """Matching code components should pass validation."""
        _validate_domain_components(
            _CodeBenchmark(), _CodeIE(), _CodeEvaluator(), _CodeFeedback()
        )

    def test_mismatched_benchmark_raises(self):
        """ARC benchmark + math IE should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Domain mismatch"):
            _validate_domain_components(
                _ArcBenchmark(), _MathIE(), _ArcEvaluator(), _ArcFeedback()
            )

    def test_mismatched_evaluator_raises(self):
        """Math evaluator with ARC components should raise."""
        with pytest.raises(ConfigurationError, match="Domain mismatch"):
            _validate_domain_components(
                _ArcBenchmark(), _ArcIE(), _MathEvaluator(), _ArcFeedback()
            )

    def test_mismatched_feedback_raises(self):
        """Code feedback with ARC components should raise."""
        with pytest.raises(ConfigurationError, match="Domain mismatch"):
            _validate_domain_components(
                _ArcBenchmark(), _ArcIE(), _ArcEvaluator(), _CodeFeedback()
            )

    def test_all_different_raises(self):
        """All different domains should raise."""
        with pytest.raises(ConfigurationError, match="Domain mismatch"):
            _validate_domain_components(
                _ArcBenchmark(), _MathIE(), _CodeEvaluator(), _ArcFeedback()
            )

    def test_no_domain_declarations_passes(self):
        """Components without DOMAIN_NAME are backward-compatible."""
        _validate_domain_components(
            _NoDomainComponent(), _NoDomainComponent(),
            _NoDomainComponent(), _NoDomainComponent(),
        )

    def test_partial_domain_declarations_passes(self):
        """If only some components declare DOMAIN_NAME and they agree, passes."""
        _validate_domain_components(
            _ArcBenchmark(), _NoDomainComponent(),
            _ArcEvaluator(), _NoDomainComponent(),
        )

    def test_partial_domain_mismatch_raises(self):
        """Even with partial declarations, mismatches raise."""
        with pytest.raises(ConfigurationError, match="Domain mismatch"):
            _validate_domain_components(
                _ArcBenchmark(), _NoDomainComponent(),
                _MathEvaluator(), _NoDomainComponent(),
            )


# ---------------------------------------------------------------------------
# Memory pairing validation (existing, but test it here too)
# ---------------------------------------------------------------------------
class _PsBuilder:
    name = "arcmemo_ps"
    SCHEMA_NAME = "arcmemo_ps"


class _OeRetriever:
    name = "oe_topk"
    COMPATIBLE_SCHEMAS = {"arcmemo_oe", "none"}


class _PsRetriever:
    name = "ps_selector"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}


class TestMemoryPairingValidation:
    def test_compatible_pairing_passes(self):
        """arcmemo_ps builder + ps_selector retriever should pass."""
        _validate_memory_pairing(_PsBuilder(), _PsRetriever())

    def test_incompatible_pairing_raises(self):
        """arcmemo_ps builder + oe_topk retriever should raise."""
        with pytest.raises(ConfigurationError, match="Memory builder"):
            _validate_memory_pairing(_PsBuilder(), _OeRetriever())


# ---------------------------------------------------------------------------
# Config param validation (_build_component)
# ---------------------------------------------------------------------------
class _StrictComponent:
    """Component with no **kwargs — rejects unknown params."""
    name = "strict"

    def __init__(self, alpha: int = 1, beta: str = "x"):
        self.alpha = alpha
        self.beta = beta


class _LenientComponent:
    """Component with **kwargs — accepts anything."""
    name = "lenient"

    def __init__(self, alpha: int = 1, **kwargs):
        self.alpha = alpha
        self.extra = kwargs


_TEST_REGISTRY = {
    "strict": _StrictComponent,
    "lenient": _LenientComponent,
}


class TestBuildComponentValidation:
    def test_valid_params_pass(self):
        """Known params for a strict component should work."""
        comp = _build_component(_TEST_REGISTRY, "strict", {"alpha": 5, "beta": "y"})
        assert comp.alpha == 5
        assert comp.beta == "y"

    def test_unknown_params_raise(self):
        """Unknown params for a strict component should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown config params.*strict.*gamma"):
            _build_component(_TEST_REGISTRY, "strict", {"alpha": 1, "gamma": 99})

    def test_none_values_stripped(self):
        """None-valued params (YAML null) should be stripped before validation."""
        comp = _build_component(
            _TEST_REGISTRY, "strict", {"alpha": 5, "gamma": None}
        )
        assert comp.alpha == 5
        assert comp.beta == "x"  # default

    def test_none_values_stripped_valid_key(self):
        """None-valued valid params should also be stripped (use default)."""
        comp = _build_component(
            _TEST_REGISTRY, "strict", {"alpha": 5, "beta": None}
        )
        assert comp.alpha == 5
        assert comp.beta == "x"  # default, not None

    def test_lenient_accepts_unknown(self):
        """Components with **kwargs should still accept unknown params."""
        comp = _build_component(
            _TEST_REGISTRY, "lenient", {"alpha": 2, "gamma": 99}
        )
        assert comp.alpha == 2
        assert comp.extra == {"gamma": 99}

    def test_unknown_component_raises(self):
        """Unknown component name should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown component"):
            _build_component(_TEST_REGISTRY, "nonexistent", {})

    def test_empty_cfg_uses_defaults(self):
        """Empty config should use constructor defaults."""
        comp = _build_component(_TEST_REGISTRY, "strict", {})
        assert comp.alpha == 1
        assert comp.beta == "x"

    def test_error_lists_accepted_params(self):
        """Error message should list accepted params for debugging."""
        with pytest.raises(ConfigurationError, match="Accepted.*alpha.*beta"):
            _build_component(_TEST_REGISTRY, "strict", {"wrong": 1})

    def test_real_math_ie_rejects_prompt_options(self):
        """MathPsSolveInferenceEngine should reject prompt_options (ARC-only)."""
        from mem2.registry.inference_engine import INFERENCE_ENGINES
        with pytest.raises(ConfigurationError, match="Unknown config params.*math_ps_solve"):
            _build_component(
                INFERENCE_ENGINES, "math_ps_solve",
                {"model": "test", "prompt_options": {"include_hint": True}},
            )

    def test_real_math_ie_accepts_valid_params(self):
        """MathPsSolveInferenceEngine should accept its own params."""
        from mem2.registry.inference_engine import INFERENCE_ENGINES
        comp = _build_component(
            INFERENCE_ENGINES, "math_ps_solve",
            {"model": "test", "error_feedback": "first"},
        )
        assert comp.model == "test"
        assert comp.error_feedback == "first"

    def test_real_arc_evaluator_rejects_unknown(self):
        """ArcExecEvaluator should reject unknown params."""
        from mem2.registry.evaluator import EVALUATORS
        with pytest.raises(ConfigurationError, match="Unknown config params.*arc_exec"):
            _build_component(
                EVALUATORS, "arc_exec",
                {"timeout_s": 2.0, "nonexistent_param": True},
            )

    def test_real_lcb_evaluator_null_stripped(self):
        """LcbExecutionEvaluator with require_all_tests=null should work."""
        from mem2.registry.evaluator import EVALUATORS
        comp = _build_component(
            EVALUATORS, "lcb_exec",
            {"timeout_s": 30.0, "require_all_tests": None},
        )
        assert comp.timeout_s == 30.0


# ---------------------------------------------------------------------------
# Router default wiring
# ---------------------------------------------------------------------------
class TestRouterDefaultWiring:
    def test_router_defaults_to_none_without_key(self):
        """resolve_components defaults to router='none' when pipeline has no router key."""
        from mem2.orchestrator.wiring import resolve_components

        config = {
            "pipeline": {
                "task_adapter": "arc_grid",
                "benchmark": "arc_agi",
                "memory_builder": "none",
                "memory_retriever": "none",
                "trajectory_policy": "single_path",
                "provider": "mock",
                "inference_engine": "python_transform_retry",
                "feedback_engine": "gt_check",
                "evaluator": "arc_exec",
                "artifact_sink": "json_local",
            },
            "components": {
                "task_adapter": {"task_name": "arc_grid"},
                "benchmark": {"data_root": "/tmp/fake", "limit": 1},
                "inference_engine": {"model": "mock"},
                "provider": {"profile_name": "mock"},
            },
        }
        components = resolve_components(config)
        assert components.router.name == "none"
