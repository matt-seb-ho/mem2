from __future__ import annotations

from pathlib import Path
from typing import Any

from mem2.io.json_io import write_json, write_jsonl


class JsonLocalArtifactSink:
    name = "json_local"

    def __init__(self):
        pass

    def write_stage_artifact(self, ctx, stage: str, data: Any) -> str:
        stage = stage.strip("/")
        root = Path(ctx.output_dir)

        if stage in {"attempts", "eval_records", "feedback_records", "events"}:
            out = root / f"{stage}.jsonl"
            write_jsonl(out, data)
            return str(out)

        if stage.endswith("/initial") or stage.startswith("memory/"):
            out = root / f"{stage}.json"
            write_json(out, data)
            return str(out)

        if stage == "frozen_config":
            out = root / "frozen_config.json"
            write_json(out, data)
            return str(out)

        out = root / f"{stage}.json"
        write_json(out, data)
        return str(out)

    def write_run_summary(self, ctx, summary: dict[str, Any]) -> str:
        out = Path(ctx.output_dir) / "summary.json"
        write_json(out, summary)
        return str(out)
