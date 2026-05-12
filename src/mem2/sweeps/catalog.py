"""YAML-driven axis catalog loader for the mem2 ablation framework.

Purpose
-------
Replaces the previous hardcoded `AXIS_A_CONDITIONS`..`AXIS_F_CONDITIONS`
literals in `scripts/sweeps/ablation_matrix.py` and the hardcoded `BASELINES`
dict + gate thresholds in `scripts/sweeps/aggregate_axis.py`. Per-axis data
now lives in `configs/axes/<axis>.yaml`. Axis names are numeric strings
("1".."6", execution-priority order, 2026-04-26 rename); historically were
letters ("A".."F").

Design
------
- One `AxisCatalog` per axis file. Contains a list of `ConditionSpec`s.
- `ConditionSpec.to_overrides()` produces the dotted-path override dict
  the sweep driver's `apply_condition` already consumes — so the driver's
  downstream code is unchanged.
- `override_group`: one of {"builder", "retriever", "combo"}. The `combo`
  shape swaps both builder and retriever atomically — absorbs axis 4's
  (format; was axis D pre-rename) special `arcmemo_oe` case (PS builder →
  OE builder + ps_topk retriever → oe_topk retriever) without a Python
  dispatch branch.
- `None`-valued leaves inside builder_cfg / retriever_cfg are preserved
  (axis 6's `empty_start` (init; was axis E pre-rename) needs explicit
  `seed_memory_file: null` to strip the inherited seed from the base
  config; `_deep_set` in the sweep driver treats None as "replace, do not
  merge").
- `spec_only: true` marks candidates listed in the catalog but without a
  working local implementation (registered module absent). The sweep
  runner skips these unless `--include-spec-only` is passed; the
  aggregator tags them `[spec-only]` in reports regardless.

Adding an axis
--------------
1. Drop `configs/axes/7.yaml` with the schema below (next-available numeric).
2. Append `- "7"` (quoted YAML string) to `configs/axes/_index.yaml`'s `order` list.
3. Run `python scripts/sweeps/ablation_matrix.py --axis 7 --step 4a`.
No Python edits required.

YAML schema (abridged)
----------------------
    axis: "3"
    title: "Interactive retrieval policy"
    role_tag: "ablation_novelty"
    order_index: 3                 # matches _index.yaml's `order` position
    baseline: one_shot
    gate:
      type: sigma_delta
      advance_threshold_sigma: 1.0
      ambiguous_threshold_sigma: 0.5
      stage_overrides:
        "4a":
          type: completion_sanity
          min_completion_rate: 0.80
    auto_advance: false
    conditions:
      - label: one_shot
        baseline: true                 # optional convenience; must equal top-level `baseline`
        override_group: retriever
        retriever: ps_topk
        retriever_cfg: {top_k: 3}
        spec_only: false
        candidate: {id: "C.1", literature_role: "baseline"}

      - label: rrmc_multi_round
        override_group: retriever
        retriever: rrmc_interactive
        retriever_cfg:
          top_k: 3
          per_round_k: 2
          max_rounds: 3
        candidate:
          id: "C.2"
          arxiv: null
          repo: "../../RRMC/RRMC/"
          literature_role: "ingredient_source"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_OVERRIDE_GROUPS = {"builder", "retriever", "combo"}


@dataclass(slots=True)
class GateSpec:
    """Per-axis gate (kill/advance) criteria, with optional per-stage overrides."""

    type: str                            # "sigma_delta" | "completion_sanity" | ...
    params: dict[str, Any] = field(default_factory=dict)
    stage_overrides: dict[str, "GateSpec"] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GateSpec":
        stage_raw = raw.get("stage_overrides") or {}
        stage_overrides = {
            str(k): cls.from_dict(v) for k, v in stage_raw.items()
        }
        params = {k: v for k, v in raw.items()
                  if k not in {"type", "stage_overrides"}}
        return cls(
            type=str(raw.get("type", "sigma_delta")),
            params=params,
            stage_overrides=stage_overrides,
        )

    def for_stage(self, step: str) -> "GateSpec":
        return self.stage_overrides.get(step, self)


@dataclass(slots=True)
class CandidateMetadata:
    """Informational metadata about a catalog entry — not consumed at run time.

    Used by the aggregator's report rendering (e.g., `[spec-only]` suffix,
    arxiv/repo citations in per-axis reports).
    """

    id: str = ""
    paper: str | None = None
    arxiv: str | None = None
    repo: str | None = None
    local_repo: str | None = None
    literature_role: str | None = None
    ported_from: str | None = None
    status: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CandidateMetadata":
        if not raw:
            return cls()
        return cls(
            id=str(raw.get("id", "")),
            paper=raw.get("paper"),
            arxiv=raw.get("arxiv"),
            repo=raw.get("repo"),
            local_repo=raw.get("local_repo"),
            literature_role=raw.get("literature_role"),
            ported_from=raw.get("ported_from"),
            status=dict(raw.get("status") or {}),
        )


@dataclass(slots=True)
class ConditionSpec:
    """One ablation condition = one row in the catalog.

    The override pattern is selected by `override_group`:
      - `builder`: set pipeline.memory_builder + merge into components.memory_builder
      - `retriever`: set pipeline.memory_retriever + merge into components.memory_retriever
      - `combo`: both atomically
    """

    label: str
    override_group: str                  # "builder" | "retriever" | "combo"
    builder: str | None = None
    builder_cfg: dict[str, Any] = field(default_factory=dict)
    retriever: str | None = None
    retriever_cfg: dict[str, Any] = field(default_factory=dict)
    baseline: bool = False
    spec_only: bool = False
    candidate: CandidateMetadata = field(default_factory=CandidateMetadata)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConditionSpec":
        og = str(raw.get("override_group", "builder"))
        if og not in VALID_OVERRIDE_GROUPS:
            raise ValueError(
                f"condition '{raw.get('label', '?')}' has invalid "
                f"override_group={og!r}; expected one of {sorted(VALID_OVERRIDE_GROUPS)}"
            )
        if og in {"builder", "combo"} and not raw.get("builder"):
            raise ValueError(
                f"condition '{raw.get('label', '?')}' has override_group={og!r} "
                "but no `builder` key"
            )
        if og in {"retriever", "combo"} and not raw.get("retriever"):
            raise ValueError(
                f"condition '{raw.get('label', '?')}' has override_group={og!r} "
                "but no `retriever` key"
            )
        return cls(
            label=str(raw["label"]),
            override_group=og,
            builder=raw.get("builder"),
            builder_cfg=dict(raw.get("builder_cfg") or {}),
            retriever=raw.get("retriever"),
            retriever_cfg=dict(raw.get("retriever_cfg") or {}),
            baseline=bool(raw.get("baseline", False)),
            spec_only=bool(raw.get("spec_only", False)),
            candidate=CandidateMetadata.from_dict(raw.get("candidate")),
        )

    def to_overrides(self) -> dict[str, Any]:
        """Emit a flat dot-path → value dict consumed by the sweep driver's
        `_deep_set` (or equivalent merge helper).

        None values at leaf positions are preserved — the deep-merge helper
        is responsible for treating `None` as "replace, don't merge-dict."
        """
        out: dict[str, Any] = {}
        if self.override_group in {"builder", "combo"}:
            out["pipeline.memory_builder"] = self.builder
            for k, v in (self.builder_cfg or {}).items():
                out[f"components.memory_builder.{k}"] = v
        if self.override_group in {"retriever", "combo"}:
            out["pipeline.memory_retriever"] = self.retriever
            for k, v in (self.retriever_cfg or {}).items():
                out[f"components.memory_retriever.{k}"] = v
        return out


@dataclass(slots=True)
class AxisCatalog:
    axis: str
    title: str = ""
    role_tag: str = ""
    order_index: int = 0
    baseline: str = ""
    gate: GateSpec = field(default_factory=lambda: GateSpec(type="sigma_delta"))
    auto_advance: bool = False
    conditions: list[ConditionSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AxisCatalog":
        axis = str(raw["axis"])
        baseline = str(raw.get("baseline", ""))
        conditions = [ConditionSpec.from_dict(c) for c in (raw.get("conditions") or [])]
        if baseline and not any(c.label == baseline for c in conditions):
            raise ValueError(
                f"axis {axis}: baseline {baseline!r} does not match any condition label"
            )
        return cls(
            axis=axis,
            title=str(raw.get("title", "")),
            role_tag=str(raw.get("role_tag", "")),
            order_index=int(raw.get("order_index", 0)),
            baseline=baseline,
            gate=GateSpec.from_dict(raw.get("gate") or {}),
            auto_advance=bool(raw.get("auto_advance", False)),
            conditions=conditions,
        )


@dataclass(slots=True)
class AxisIndex:
    """Top-level axis-order declaration (configs/axes/_index.yaml)."""

    order: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AxisIndex":
        return cls(order=[str(x) for x in (raw.get("order") or [])])


# ----------------------------------------------------------------------- #
#                              Loaders                                     #
# ----------------------------------------------------------------------- #

def load_axis_catalog(axis: str, axes_dir: Path | str) -> AxisCatalog:
    """Load `<axes_dir>/<axis>.yaml` and return a validated AxisCatalog."""
    p = Path(axes_dir) / f"{axis}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"axis catalog not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if str(raw.get("axis", axis)) != axis:
        raise ValueError(
            f"axis catalog at {p} declares axis={raw.get('axis')!r} "
            f"but was loaded for axis={axis!r}"
        )
    return AxisCatalog.from_dict(raw)


def load_axis_index(axes_dir: Path | str) -> AxisIndex:
    p = Path(axes_dir) / "_index.yaml"
    if not p.exists():
        return AxisIndex()
    raw = yaml.safe_load(p.read_text()) or {}
    return AxisIndex.from_dict(raw)


def conditions_from_catalog(
    catalog: AxisCatalog,
    variants: list[str] | None = None,
    *,
    include_spec_only: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """Yield `(label, override_dict)` pairs for each condition in the catalog.

    - `variants`: if provided, filter to conditions whose label appears
      (case-sensitive). Useful for axis 4's (format; was axis D pre-rename)
      "just run the OE condition" ergonomics.
    - `include_spec_only`: default False. Spec-only conditions are omitted
      at run time; the aggregator still reports them with a `[spec-only]`
      suffix based on catalog metadata.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    variant_set = set(variants) if variants else None
    for cond in catalog.conditions:
        if variant_set is not None and cond.label not in variant_set:
            continue
        if cond.spec_only and not include_spec_only:
            continue
        out.append((cond.label, cond.to_overrides()))
    return out
