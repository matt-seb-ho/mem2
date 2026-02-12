from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mem2.core.entities import to_primitive


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_primitive(data), indent=2, sort_keys=False) + "\n")
    return p


def write_jsonl(path: str | Path, rows: list[Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_primitive(row), sort_keys=False) + "\n")
    return p
