from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (_REPO_ROOT, _REPO_ROOT / "src"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts.sweep_all_axes import main


if __name__ == "__main__":
    main()
