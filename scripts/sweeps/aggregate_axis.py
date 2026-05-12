"""Per-axis aggregator for Phase-1 ablation sweeps.

Reads `outputs/phase1_sweeps/step_<stage>/axis_<X>/sweep_axis_<X>.json`, groups
by condition, computes mean ± std of `official_score` across seeds, applies
the axis's kill criterion (loaded from `configs/axes/<X>.yaml`), and emits:

  - mem_devlog/docs/<NN>_phase1_step<step>_results_<date>.md  (consolidated
    stage file; hand-written notes preserved past the SENTINEL marker).

Axis names: numeric strings "1".."6" since the 2026-04-26 execution-priority
rename. Historical sweep outputs under `outputs/phase1_sweeps/step_4b/axis_<L>/`
may still use the retired letter labels (A=Reorg, B=Retrieval, C=Interactive,
D=Format, E=Init, F=Edit) — readers should consult `configs/axes/_index.yaml`
for the mapping.

Usage::

    python scripts/sweeps/aggregate_axis.py --step 4a --axis 2
    python scripts/sweeps/aggregate_axis.py --step 4b --all
    python scripts/sweeps/aggregate_axis.py --step 4b --axis 3 --axes-dir configs/axes

Gate criteria, baseline labels, auto-advance flags, and spec-only metadata
all live in `configs/axes/*.yaml` as of the Phase 0 refactor (2026-04-22).
The GATE_FUNCS registry below dispatches on `gate.type`; to add a new gate
shape, register a function here + update the axis YAML's `gate.type`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from mem2.sweeps.catalog import (
    AxisCatalog,
    GateSpec,
    load_axis_catalog,
    load_axis_index,
)


# ----------------------------------------------------------------------- #
#                     Gate function registry (plugin)                     #
# ----------------------------------------------------------------------- #
#
# Each gate function takes:
#   summaries: dict[label → ConditionSummary]
#   baseline:  str (label of the baseline condition)
#   params:    dict[str, Any] — gate-specific kwargs from the YAML
# And returns:
#   (decision, reason): ("ADVANCE" | "KILL" | "AMBIGUOUS" | "SANITY_FAIL", str)
#
# Add a new gate type by registering a callable under its YAML key here.

GATE_FUNCS: dict[str, "GateFunc"] = {}


def register_gate(name: str):
    def deco(fn: "GateFunc") -> "GateFunc":
        GATE_FUNCS[name] = fn
        return fn
    return deco


@dataclass
class ConditionSummary:
    label: str
    seeds: list[int]
    scores: list[float]
    problem_counts: list[int]
    attempt_counts: list[int]
    correct_attempts: list[int]
    errors: list[str]
    durations: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.scores) if self.scores else float("nan")

    @property
    def std(self) -> float:
        return statistics.pstdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def runs_ok(self) -> int:
        """Count of seed-runs that returned status=ok."""
        return len(self.scores)

    @property
    def runs_total(self) -> int:
        return len(self.scores) + len(self.errors)

    @property
    def completion_rate(self) -> float:
        """Fraction of (condition, seed) runs that reached status=ok.
        NOT per-problem — that would need prompt_fingerprint_count which isn't
        consistently bounded at ≤ problem_count (retry passes make it 2-3×).
        """
        if self.runs_total == 0:
            return float("nan")
        return self.runs_ok / self.runs_total

    @property
    def crashed_any(self) -> bool:
        return any(self.errors)


GateFunc = callable  # type hint alias for registry


@register_gate("completion_sanity")
def gate_completion_sanity(
    summaries: dict[str, "ConditionSummary"],
    baseline: str,
    params: dict,
) -> tuple[str, str]:
    """Stage-4a-style sanity gate: every condition must produce ≥1 ok run."""
    fails = [s for s in summaries.values() if s.runs_ok == 0 or s.crashed_any]
    if fails:
        names = ", ".join(s.label for s in fails)
        return ("SANITY_FAIL", f"conditions with no ok run or crashes: {names}")
    return ("ADVANCE", "all conditions produced ok run(s)")


@register_gate("sigma_delta")
def gate_sigma_delta(
    summaries: dict[str, "ConditionSummary"],
    baseline: str,
    params: dict,
) -> tuple[str, str]:
    """Stage-4b/4c-style score-based gate vs baseline.

    Params:
      advance_threshold_sigma: Δ/σ required for ADVANCE (default 1.0)
      ambiguous_threshold_sigma: Δ/σ required for AMBIGUOUS (default 0.5)
    """
    advance_t = float(params.get("advance_threshold_sigma", 1.0))
    ambig_t = float(params.get("ambiguous_threshold_sigma", 0.5))

    if not baseline or baseline not in summaries:
        return ("AMBIGUOUS", f"no baseline condition '{baseline}' in sweep")

    base = summaries[baseline]
    if not base.scores:
        return ("AMBIGUOUS", f"baseline {baseline!r} has no ok runs")

    best_delta = -math.inf
    best_cond = None
    for label, s in summaries.items():
        if label == baseline:
            continue
        if not s.scores:
            continue
        delta = s.mean - base.mean
        pooled_sigma = max(math.sqrt(base.std ** 2 + s.std ** 2), 1e-6)
        ratio = delta / pooled_sigma
        if ratio > best_delta:
            best_delta = ratio
            best_cond = label
    if best_cond is None:
        return ("AMBIGUOUS", "no non-baseline conditions with data to compare")

    if best_delta >= advance_t:
        return ("ADVANCE", f"{best_cond} beats baseline by ≥{advance_t}σ (Δ/σ={best_delta:+.2f})")
    elif best_delta >= ambig_t:
        return ("AMBIGUOUS", f"best {best_cond} Δ/σ={best_delta:+.2f} — targeted n=50 scaleup on this axis only")
    else:
        return ("KILL", f"no condition beats baseline by ≥{ambig_t}σ (best {best_cond} at Δ/σ={best_delta:+.2f})")


def _advance_decision(
    catalog: AxisCatalog,
    summaries: dict[str, "ConditionSummary"],
    step: str,
) -> tuple[str, str]:
    """Resolve the gate for this axis + stage using the catalog.

    - Stage-specific gate override from `catalog.gate.stage_overrides[step]` wins.
    - Otherwise uses the top-level `catalog.gate`.
    - `catalog.auto_advance=true` forces ADVANCE for score-based stages
      (still allows SANITY_FAIL for stage 4a's completion gate since that's
      an integrity check, not a signal claim).
    """
    gate: GateSpec = catalog.gate.for_stage(step)
    fn = GATE_FUNCS.get(gate.type)
    if fn is None:
        return ("AMBIGUOUS", f"unknown gate.type={gate.type!r} for axis {catalog.axis}")

    decision, reason = fn(summaries, catalog.baseline, gate.params)

    # Auto-advance is meaningful only for score gates (stage 4b/4c). Stage 4a's
    # completion_sanity still needs to catch infra failures even when the axis
    # is flagged for auto-advance.
    if catalog.auto_advance and gate.type != "completion_sanity" and decision != "ADVANCE":
        return (
            "ADVANCE",
            f"axis {catalog.axis} auto-advance per catalog ({reason})",
        )
    return decision, reason


# ----------------------------------------------------------------------- #
#                            File I/O                                     #
# ----------------------------------------------------------------------- #

def load_sweep(sweep_file: Path) -> list[dict]:
    return json.loads(sweep_file.read_text())


def group_by_condition(runs: list[dict]) -> dict[str, ConditionSummary]:
    out: dict[str, ConditionSummary] = {}
    for r in runs:
        label = r["condition"]
        cs = out.setdefault(
            label,
            ConditionSummary(
                label=label, seeds=[], scores=[], problem_counts=[],
                attempt_counts=[], correct_attempts=[], errors=[], durations=[],
            ),
        )
        if r["status"] == "ok":
            s = r.get("summary", {}) or {}
            cs.seeds.append(int(r["seed"]))
            cs.scores.append(float(s.get("official_score", 0.0)))
            cs.problem_counts.append(int(s.get("problem_count", 0)))
            cs.attempt_counts.append(int(s.get("attempt_count", 0)))
            cs.correct_attempts.append(int(s.get("correct_attempts", 0)))
            cs.durations.append(float(r.get("duration_s", 0.0)))
        else:
            cs.errors.append(str(r.get("error", ""))[:200])
    return out


# ----------------------------------------------------------------------- #
#                            Report writers                                #
# ----------------------------------------------------------------------- #

STAGE_FILENAMES = {
    "4a": "64_phase1_step4a_results_2026_04_21.md",
    "4b": "65_phase1_step4b_results_2026_04_21.md",
    "4c": "phase1_step4c_results.md",  # placeholder; 4c was dropped 2026-04-21
}


def _stage_file(step: str) -> str:
    return STAGE_FILENAMES.get(step, f"phase1_step{step}_results.md")


# --- legacy write_axis_report / update_cross_axis_table removed 2026-04-22 ---
# The consolidated-stage-file flow (run_all_axes) subsumed per-axis markdown
# writes + rolling cross-axis-table updates. See the file history for the
# pre-refactor versions if you need them.


# ----------------------------------------------------------------------- #
#                                 CLI                                     #
# ----------------------------------------------------------------------- #

def _is_spec_only(catalog: AxisCatalog, label: str) -> bool:
    for c in catalog.conditions:
        if c.label == label:
            return c.spec_only
    return False


def _axis_block(
    step: str,
    catalog: AxisCatalog,
    summaries: dict[str, ConditionSummary],
    decision: str,
    reason: str,
    sweep_file: Path,
) -> str:
    """Render one axis's section as a string (no file I/O). Used by the
    consolidated-stage-file writer."""
    import io
    axis = catalog.axis
    buf = io.StringIO()
    buf.write(f"### Axis {axis} — {catalog.title}\n\n")
    buf.write(f"**Decision:** `{decision}` — {reason}\n")
    buf.write(f"**Source:** `{sweep_file}`\n")
    buf.write(f"**Baseline condition:** `{catalog.baseline or 'n/a'}`\n\n")
    buf.write("| Condition | Seeds | Mean | Std | Completion rate | Errors | Durations (s) |\n")
    buf.write("|---|---|---|---|---|---|---|\n")
    for label in sorted(summaries):
        s = summaries[label]
        tag = " [spec-only]" if _is_spec_only(catalog, label) else ""
        seeds_str = ",".join(str(x) for x in sorted(s.seeds)) if s.seeds else "—"
        mean_str = f"{s.mean:.3f}" if s.scores else "—"
        std_str = f"{s.std:.3f}" if len(s.scores) > 1 else "—"
        cr_str = f"{s.completion_rate*100:.0f}%" if not math.isnan(s.completion_rate) else "—"
        err_str = "none" if not s.crashed_any else f"{len(s.errors)} run(s) errored"
        dur_str = ",".join(f"{d:.1f}" for d in s.durations) if s.durations else "—"
        buf.write(f"| `{label}`{tag} | {seeds_str} | {mean_str} | {std_str} | {cr_str} | {err_str} | {dur_str} |\n")
    for label in sorted(summaries):
        s = summaries[label]
        if s.errors:
            buf.write(f"\n**Errors for `{label}`:**\n")
            for e in s.errors:
                buf.write(f"- `{e}`\n")
    buf.write("\n")
    return buf.getvalue()


def run_all_axes(
    step: str,
    outputs_root: Path,
    docs_root: Path,
    axes_dir: Path | str = "configs/axes",
) -> dict[str, tuple[str, str]]:
    """Aggregate every axis listed in _index.yaml into one consolidated stage
    file. Returns per-axis (decision, reason) map.

    Output file: docs_root / _stage_file(step).
    """
    docs_root.mkdir(parents=True, exist_ok=True)
    out_path = docs_root / _stage_file(step)

    # Iterate axes in _index.yaml order — NOT hardcoded "ABCDEF".
    idx = load_axis_index(Path(axes_dir))
    axes_order = idx.order or list("ABCDEF")

    per_axis_results: dict[str, tuple[AxisCatalog, str, str, dict[str, ConditionSummary], Path]] = {}
    for axis in axes_order:
        try:
            catalog = load_axis_catalog(axis, axes_dir)
        except FileNotFoundError:
            # Axis listed in _index but YAML missing — treat as not-run.
            sweep_file = outputs_root / f"step_{step}" / f"axis_{axis}" / f"sweep_axis_{axis}.json"
            per_axis_results[axis] = (None, "NOT_RUN", f"axis catalog missing: {axis}.yaml", {}, sweep_file)
            continue
        sweep_file = outputs_root / f"step_{step}" / f"axis_{axis}" / f"sweep_axis_{axis}.json"
        if not sweep_file.exists():
            per_axis_results[axis] = (catalog, "NOT_RUN", f"sweep file not found: {sweep_file}", {}, sweep_file)
            continue
        runs = load_sweep(sweep_file)
        summaries = group_by_condition(runs)
        decision, reason = _advance_decision(catalog, summaries, step)
        per_axis_results[axis] = (catalog, decision, reason, summaries, sweep_file)

    lines: list[str] = []
    stage_title = {"4a": "Stage 4a Results (condensed)",
                   "4b": "Stage 4b Results (condensed) — first-wave pilot",
                   "4c": "Stage 4c Results (condensed)"}.get(step, f"Stage {step} Results (condensed)")
    lines.append(f"# Phase-1 {stage_title}\n")
    lines.append(f"**Auto-generated by `scripts/sweeps/aggregate_axis.py --step {step} --all`. Hand-written notes go below the sentinel at the bottom and are preserved across reruns.**\n")
    lines.append(f"**Stage:** {step}  |  **Axis order (from _index.yaml):** {' → '.join(axes_order)}\n")
    lines.append(f"**Output file:** `{out_path.name}`\n")
    lines.append("---\n")
    lines.append("## Cross-axis summary\n")
    if step == "4a":
        lines.append("| Axis | Title | Conditions (spec-only) | Best mean | Completion | Decision | Reason |")
        lines.append("|---|---|---|---|---|---|---|")
        for axis in axes_order:
            catalog, decision, reason, summaries, _ = per_axis_results[axis]
            title = catalog.title if catalog else "—"
            if not summaries:
                lines.append(f"| {axis} | {title} | — | — | — | `{decision}` | {reason} |")
                continue
            n_spec = sum(1 for c in catalog.conditions if c.spec_only) if catalog else 0
            cond_str = f"{len(summaries)}" + (f" ({n_spec} spec-only)" if n_spec else "")
            comps = [s.completion_rate for s in summaries.values() if not math.isnan(s.completion_rate)]
            comp_str = f"{min(comps)*100:.0f}-{max(comps)*100:.0f}%" if comps else "—"
            best_mean = max((s.mean for s in summaries.values() if s.scores), default=float("nan"))
            best_mean_str = f"{best_mean:.3f}" if not math.isnan(best_mean) else "—"
            lines.append(f"| {axis} | {title} | {cond_str} | {best_mean_str} | {comp_str} | `{decision}` | {reason} |")
    else:
        lines.append("| Axis | Title | Conditions (spec-only) | Baseline mean ± σ | Best non-baseline mean ± σ | Δ/σ | Decision | Reason |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for axis in axes_order:
            catalog, decision, reason, summaries, _ = per_axis_results[axis]
            title = catalog.title if catalog else "—"
            if not summaries:
                lines.append(f"| {axis} | {title} | — | — | — | — | `{decision}` | {reason} |")
                continue
            base_label = catalog.baseline if catalog else ""
            base = summaries.get(base_label)
            best_label = None
            best_mean = float("-inf")
            for lbl, s in summaries.items():
                if lbl == base_label:
                    continue
                if s.mean > best_mean:
                    best_mean = s.mean
                    best_label = lbl
            base_str = f"{base.mean:.3f} ± {base.std:.3f}" if base and base.scores else "—"
            best_str = (f"{summaries[best_label].mean:.3f} ± {summaries[best_label].std:.3f} (`{best_label}`)"
                        if best_label and summaries[best_label].scores else "—")
            delta_sigma = None
            if base and best_label and base.scores and summaries[best_label].scores:
                pooled = math.sqrt(base.std ** 2 + summaries[best_label].std ** 2) or 1e-6
                delta_sigma = (summaries[best_label].mean - base.mean) / pooled
            delta_str = f"{delta_sigma:+.2f}" if delta_sigma is not None else "—"
            n_spec = sum(1 for c in catalog.conditions if c.spec_only) if catalog else 0
            cond_str = f"{len(summaries)}" + (f" ({n_spec} spec-only)" if n_spec else "")
            lines.append(f"| {axis} | {title} | {cond_str} | {base_str} | {best_str} | {delta_str} | `{decision}` | {reason} |")
    lines.append("")
    lines.append("---\n")
    lines.append("## Per-axis detail\n")
    for axis in axes_order:
        catalog, decision, reason, summaries, sweep_file = per_axis_results[axis]
        if summaries and catalog:
            lines.append(_axis_block(step, catalog, summaries, decision, reason, sweep_file))
            lines.append("---\n")
        else:
            title = catalog.title if catalog else "—"
            lines.append(f"### Axis {axis} — {title}\n\n**Decision:** `{decision}` — {reason}\n\n---\n")

    # Preserve any hand-written notes past the marker if the file exists.
    # Long unique sentinel avoids accidental matches in body text.
    marker = "<!--SENTINEL:PHASE1_NOTES_BLOCK_f4f17b2c-->"
    note_header = "<!-- Hand-written notes preserved across aggregator runs. Write freely below. -->"
    existing_notes = ""
    if out_path.exists():
        existing_text = out_path.read_text()
        if marker in existing_text:
            # Strip the fixed header-comment line so it doesn't accumulate
            # copies across reruns.
            tail = existing_text.split(marker, 1)[1]
            existing_notes = "\n".join(
                ln for ln in tail.split("\n") if ln.strip() != note_header
            )
    lines.append(marker)
    lines.append(note_header)
    if existing_notes.strip():
        lines.append(existing_notes.lstrip("\n"))

    out_path.write_text("\n".join(lines))
    return {ax: (d, r) for ax, (_, d, r, _, _) in per_axis_results.items()}


def run_axis(
    step: str,
    axis: str,
    outputs_root: Path,
    docs_root: Path,
    axes_dir: Path | str = "configs/axes",
) -> tuple[str, str]:
    """Single-axis invocation — delegates to run_all_axes and returns the
    (decision, reason) for just the requested axis. Still regenerates the
    full consolidated stage file, since axis results are interrelated."""
    results = run_all_axes(step, outputs_root, docs_root, axes_dir=axes_dir)
    return results.get(axis, ("NOT_RUN", f"axis {axis} not processed"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True)
    p.add_argument("--axis",
                   help="Axis name (numeric string, e.g. '1'..'6'); any axis listed in --axes-dir _index.yaml")
    p.add_argument("--all", action="store_true",
                   help="Process every axis listed in --axes-dir _index.yaml")
    p.add_argument("--outputs-root", default="outputs/phase1_sweeps")
    p.add_argument("--docs-root", default="../mem_devlog/docs",
                   help="Where consolidated stage file lands. Attach-mode default: mem_devlog/docs/.")
    p.add_argument("--axes-dir", default="configs/axes",
                   help="Directory with per-axis YAML catalogs + _index.yaml")
    args = p.parse_args()

    if not args.all and not args.axis:
        p.error("pass either --axis or --all")

    if args.all:
        idx = load_axis_index(Path(args.axes_dir))
        axes = idx.order or list("ABCDEF")
    else:
        axes = [args.axis]

    for ax in axes:
        decision, reason = run_axis(
            args.step, ax, Path(args.outputs_root), Path(args.docs_root),
            axes_dir=args.axes_dir,
        )
        print(f"[axis {ax}] {decision} — {reason}")


if __name__ == "__main__":
    main()
