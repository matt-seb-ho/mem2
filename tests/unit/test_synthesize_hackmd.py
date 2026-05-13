from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from case_studies.scripts.synthesize_hackmd import synthesize_hackmd


def _write_run(run_dir: Path, *, run_id: str, mode_name: str = "error_analysis") -> None:
    (run_dir / "analyses").mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "port": "graphrag",
                "label": "unit",
                "seed": 42,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    (run_dir / "analyses" / f"{mode_name}.md").write_text("# Analysis\n", encoding="utf-8")


def test_synthesize_hackmd_links_runs_and_modes(tmp_path):
    case_root = tmp_path / "case_studies"
    run_dir = case_root / "runs" / "run1"
    out_path = case_root / "synthesis" / "out.md"
    _write_run(run_dir, run_id="run1")

    out = synthesize_hackmd(
        [run_dir],
        modes=["error_analysis"],
        title="Unit Synthesis",
        intro="Intro text",
        out_path=out_path,
        now=datetime(2026, 5, 13, 12, 0, tzinfo=UTC),
    )

    rendered = out.read_text(encoding="utf-8")
    assert "# Unit Synthesis" in rendered
    assert "Intro text" in rendered
    assert "[summary](../runs/run1/summary.md)" in rendered
    assert "[error_analysis](../runs/run1/analyses/error_analysis.md)" in rendered
