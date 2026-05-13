from __future__ import annotations

import json
from pathlib import Path

from case_studies.scripts._shared.grid_render import render_grid_ascii, render_grid_html, render_grid_markdown
from case_studies.scripts.render_markdown import write_summary


def test_grid_render_outputs_html_and_ascii():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]

    html = render_grid_html(grid, label="unit")
    ascii_grid = render_grid_ascii(grid)
    markdown = render_grid_markdown(grid, label="unit")

    assert "background:#111111" in html
    assert "background:#1E93FF" in html
    assert "unit (3x3)" in html
    assert "0 1 2\n3 4 5\n6 7 8" == ascii_grid
    assert "<details><summary>ASCII fallback</summary>" in markdown


def test_render_markdown_includes_arc_grid_section(tmp_path):
    run_dir = tmp_path / "run1"
    output_dir = tmp_path / "outputs" / "run1"
    iter_dir = run_dir / "problems" / "task1" / "iter_0"
    iter_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "run1",
                "port": "graphrag",
                "label": "unit",
                "timestamp_utc": "2026-05-13T10:30:00+00:00",
                "n_problems": 1,
                "seed": 42,
                "model": "mock",
                "output_dir": str(output_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "problems.json").write_text(
        json.dumps(
            {
                "task1": {
                    "uid": "task1",
                    "train_pairs": [
                        {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]}
                    ],
                    "test_pairs": [
                        {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]}
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_dir / "call_meta.json").write_text(json.dumps({"latency_s": 0.125}) + "\n", encoding="utf-8")
    (iter_dir / "eval.json").write_text(json.dumps({"correct": True}) + "\n", encoding="utf-8")
    (iter_dir / "retrieval_bundle.json").write_text(json.dumps({"metadata": {"scoring_mode": "unit"}}) + "\n", encoding="utf-8")

    out_path = write_summary(run_dir)
    rendered = out_path.read_text(encoding="utf-8")

    assert "## ARC grids" in rendered
    assert "train 1 input (2x2)" in rendered
    assert "background:#FFDC00" in rendered
    assert "<details><summary>ASCII fallback</summary>" in rendered
