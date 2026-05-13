from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_STUDIES_ROOT = REPO_ROOT / "case_studies"
RUNS_ROOT = CASE_STUDIES_ROOT / "runs"

for candidate in [REPO_ROOT, REPO_ROOT / "src"]:
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "run"


def utc_run_stamp(now: datetime | None = None) -> str:
    when = now or datetime.now(UTC)
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def iter_trace_dirs(run_dir: Path) -> list[Path]:
    problems_dir = run_dir / "problems"
    if not problems_dir.exists():
        return []
    return sorted(problems_dir.glob("*/iter_*"))


def relative_link(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()
