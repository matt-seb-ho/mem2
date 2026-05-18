"""Regression tests for the YAML-driven axis catalog (Phase 0 refactor).

Run:
    cd mem2
    source .venv/bin/activate
    pytest tests/sweeps/test_catalog.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `scripts/sweeps/*.py` importable as modules inside the tests
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sweeps"))

from mem2.sweeps.catalog import (  # noqa: E402
    AxisCatalog,
    ConditionSpec,
    GateSpec,
    conditions_from_catalog,
    load_axis_catalog,
    load_axis_index,
)

AXES_DIR = REPO_ROOT / "configs" / "axes"


# --------------------------------------------------------------------- #
#                              YAML round-trip                           #
# --------------------------------------------------------------------- #

# Axis labels were renamed to numeric (B→1, A→2, C→3, D→4, F→5, E→6) on
# 2026-04-26 to reflect execution priority. Port IDs (e.g., "A.2") remain
# stable historical identifiers and are unchanged.

AXIS_LABELS = ["1", "2", "3", "4", "5", "6"]


def test_index_declares_six_axes():
    idx = load_axis_index(AXES_DIR)
    assert idx.order == AXIS_LABELS, (
        f"axis order unexpected: {idx.order}"
    )


@pytest.mark.parametrize("axis", AXIS_LABELS)
def test_axis_catalog_loads(axis):
    cat = load_axis_catalog(axis, AXES_DIR)
    assert cat.axis == axis
    assert cat.baseline, f"axis {axis} has no baseline"
    assert any(c.label == cat.baseline for c in cat.conditions), (
        f"axis {axis} baseline {cat.baseline!r} not among conditions"
    )


def test_all_conditions_valid_override_group():
    for axis in AXIS_LABELS:
        cat = load_axis_catalog(axis, AXES_DIR)
        for c in cat.conditions:
            assert c.override_group in {"builder", "retriever", "inference_engine", "combo"}
            if c.override_group in {"builder", "combo"}:
                assert c.builder, f"{axis}/{c.label}: builder missing"
            if c.override_group in {"retriever", "combo"}:
                assert c.retriever, f"{axis}/{c.label}: retriever missing"
            if c.override_group == "inference_engine":
                assert c.inference_engine, f"{axis}/{c.label}: inference_engine missing"


# --------------------------------------------------------------------- #
#         Pre-refactor condition-list parity (reorg axis = "2")          #
# --------------------------------------------------------------------- #

def test_axis_2_reorg_matches_prerefactor_condition_labels():
    """Axis 2 (reorganization, was A) must include the pre-refactor labels
    plus the reorg-family paper ports landed in Phase 1.A (A.2 DreamCoder,
    A.3 Stitch). Assert superset — the pre-refactor floor must still be present."""
    cat = load_axis_catalog("2", AXES_DIR)
    labels = {c.label for c in cat.conditions}
    prerefactor_floor = {
        "reorg_off",
        "accretive_prune",  # added in Batch 1.1
        "reorg_on_graph_mdl_global_plateau",
        "reorg_on_trace_mdl_accretive_everyk",
    }
    assert prerefactor_floor.issubset(labels), (
        f"missing pre-refactor labels: {prerefactor_floor - labels}"
    )
    # Phase 1.A ports (must be present after sleep cycle 2026-04-23):
    phase1a_ports = {"reorg_dreamcoder", "reorg_stitch"}
    assert phase1a_ports.issubset(labels), (
        f"missing Phase 1.A port labels: {phase1a_ports - labels}"
    )


def test_axis_2_reorg_overrides_structure():
    """Each axis 2 (reorg) condition emits `pipeline.memory_builder` + dot-path
    `components.memory_builder.<key>` overrides for every builder_cfg key."""
    cat = load_axis_catalog("2", AXES_DIR)
    for cond in cat.conditions:
        ov = cond.to_overrides()
        assert ov["pipeline.memory_builder"] == cond.builder
        for k in cond.builder_cfg.keys():
            assert f"components.memory_builder.{k}" in ov


# --------------------------------------------------------------------- #
#         Axis 4 (format) combo override + axis 6 (init) null leaves     #
# --------------------------------------------------------------------- #

def test_axis_4_format_combo_override_emits_both_builder_and_retriever():
    cat = load_axis_catalog("4", AXES_DIR)
    oe = next((c for c in cat.conditions if c.label == "arcmemo_oe"), None)
    assert oe is not None, "axis 4 missing arcmemo_oe condition"
    assert oe.override_group == "combo"
    ov = oe.to_overrides()
    # builder side
    assert ov["pipeline.memory_builder"] == "arcmemo_oe"
    assert ov["components.memory_builder.seed_memory_file"] is None
    # retriever side
    assert ov["pipeline.memory_retriever"] == "oe_topk"
    assert ov["components.memory_retriever.top_k"] == 3
    # None-preservation on inherited PS-specific retriever kwargs
    assert ov["components.memory_retriever.include_description"] is None
    assert ov["components.memory_retriever.skip_cues"] is None
    assert ov["components.memory_retriever.skip_implementation"] is None


def test_axis_6_init_empty_start_preserves_null_leaves():
    cat = load_axis_catalog("6", AXES_DIR)
    empty = next((c for c in cat.conditions if c.label == "empty_start"), None)
    assert empty is not None
    ov = empty.to_overrides()
    assert ov["pipeline.memory_builder"] == "arcmemo_ps"
    # These explicit nulls must survive into the merged config
    assert ov["components.memory_builder.seed_memory_file"] is None
    assert ov["components.memory_builder.seed_annotations_file"] is None


def test_axis_3_gepa_hsea_inference_engine_override():
    cat = load_axis_catalog("3", AXES_DIR)
    cond = next((c for c in cat.conditions if c.label == "gepa_hsea_pipeline"), None)
    assert cond is not None
    assert cond.override_group == "inference_engine"
    ov = cond.to_overrides()
    assert ov["pipeline.inference_engine"] == "gepa_hsea"
    assert ov["components.inference_engine.max_retries"] == 3
    assert ov["components.inference_engine.max_hypothesize_attempts"] == 1


def test_axis_6_init_auto_advance_flag():
    cat = load_axis_catalog("6", AXES_DIR)
    assert cat.auto_advance is True, (
        "axis 6 (init) must carry auto_advance=true per the plan (init ablation's "
        "null results are reportable)"
    )


@pytest.mark.parametrize("axis", ["1", "2", "3", "4", "5"])
def test_non_init_axes_do_not_auto_advance(axis):
    cat = load_axis_catalog(axis, AXES_DIR)
    assert cat.auto_advance is False


# --------------------------------------------------------------------- #
#                     Gate specs + stage overrides                       #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("axis", AXIS_LABELS)
def test_gate_has_stage_4a_override(axis):
    cat = load_axis_catalog(axis, AXES_DIR)
    gate_4a = cat.gate.for_stage("4a")
    assert gate_4a.type == "completion_sanity"
    assert gate_4a.params.get("min_completion_rate") == 0.80


@pytest.mark.parametrize("axis", AXIS_LABELS)
def test_gate_default_is_sigma_delta(axis):
    cat = load_axis_catalog(axis, AXES_DIR)
    gate_default = cat.gate.for_stage("4b")
    assert gate_default.type == "sigma_delta"
    assert gate_default.params.get("advance_threshold_sigma") == 1.0
    assert gate_default.params.get("ambiguous_threshold_sigma") == 0.5


# --------------------------------------------------------------------- #
#         Aggregator decision parity with committed stage-4b data        #
# --------------------------------------------------------------------- #

# Committed stage-4b sweep outputs were written under the old letter labels
# (axis_A, axis_B, etc.). Map old-letter → new-numeric so the parity test can
# still locate historical files. Tests skip when files are missing.
OLD_TO_NEW_AXIS = {"A": "2", "B": "1", "C": "3", "D": "4", "E": "6", "F": "5"}

EXPECTED_4B_DECISIONS = {
    "2": "KILL",       # was A (reorg)
    "1": "KILL",       # was B (retrieval)
    "3": "ADVANCE",    # was C (interactive retrieval)
    "4": "ADVANCE",    # was D (format)
    "6": "ADVANCE",    # was E (init, via auto_advance)
    "5": "KILL",       # was F (NASM)
}


@pytest.mark.parametrize("axis,expected", EXPECTED_4B_DECISIONS.items())
def test_stage_4b_decision_matches_committed_outputs(axis, expected, tmp_path):
    """Replay committed sweep_axis_<X>.json → aggregator → assert decision.

    Historical sweep files use old letter labels (axis_A, axis_B, etc.); look
    up the new numeric label's old letter via OLD_TO_NEW_AXIS reverse map.
    """
    from aggregate_axis import (  # type: ignore[import-not-found]
        _advance_decision, group_by_condition, load_sweep,
    )
    new_to_old = {v: k for k, v in OLD_TO_NEW_AXIS.items()}
    old_letter = new_to_old.get(axis, axis)
    sweep_file = REPO_ROOT / "outputs" / "phase1_sweeps" / "step_4b" / f"axis_{old_letter}" / f"sweep_axis_{old_letter}.json"
    if not sweep_file.exists():
        pytest.skip(f"committed stage-4b sweep file missing: {sweep_file}")
    runs = load_sweep(sweep_file)
    summaries = group_by_condition(runs)
    catalog = load_axis_catalog(axis, AXES_DIR)
    decision, reason = _advance_decision(catalog, summaries, "4b")
    assert decision == expected, (
        f"axis {axis} expected {expected}, got {decision} — reason: {reason}"
    )


# --------------------------------------------------------------------- #
#                 Pluggable-axis regression (drop-in test)               #
# --------------------------------------------------------------------- #

def test_new_axis_loadable_without_python_change(tmp_path):
    """Simulate dropping `configs/axes/Z.yaml` — loader must accept it.
    No framework change required."""
    axes_dir = tmp_path / "axes"
    axes_dir.mkdir()
    (axes_dir / "Z.yaml").write_text(
        """
axis: Z
title: "Dummy test axis"
role_tag: "test"
order_index: 99
baseline: base_cond
gate:
  type: sigma_delta
  advance_threshold_sigma: 1.0
  ambiguous_threshold_sigma: 0.5
auto_advance: false
conditions:
  - label: base_cond
    baseline: true
    override_group: builder
    builder: arcmemo_ps
    builder_cfg: {}
    candidate:
      id: Z.1
      literature_role: baseline
  - label: variant_cond
    override_group: builder
    builder: arcmemo_ps
    builder_cfg:
      max_entries: 100
    candidate:
      id: Z.2
      literature_role: ingredient_source
"""
    )
    cat = load_axis_catalog("Z", axes_dir)
    assert cat.axis == "Z"
    assert cat.baseline == "base_cond"
    conds = conditions_from_catalog(cat)
    assert len(conds) == 2


# --------------------------------------------------------------------- #
#                         Spec-only filter + flag                        #
# --------------------------------------------------------------------- #

def test_spec_only_condition_skipped_by_default(tmp_path):
    axes_dir = tmp_path / "axes"
    axes_dir.mkdir()
    (axes_dir / "Q.yaml").write_text(
        """
axis: Q
baseline: cond_real
gate:
  type: sigma_delta
auto_advance: false
conditions:
  - label: cond_real
    baseline: true
    override_group: builder
    builder: arcmemo_ps
    builder_cfg: {}
  - label: cond_spec_only
    override_group: builder
    builder: does_not_exist_yet
    builder_cfg: {}
    spec_only: true
"""
    )
    cat = load_axis_catalog("Q", axes_dir)
    default = conditions_from_catalog(cat)
    assert [lbl for lbl, _ in default] == ["cond_real"]
    with_spec = conditions_from_catalog(cat, include_spec_only=True)
    assert [lbl for lbl, _ in with_spec] == ["cond_real", "cond_spec_only"]
