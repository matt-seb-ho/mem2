from __future__ import annotations

import multiprocessing as mp
import re
import traceback
from typing import Any

import numpy as np

_CODE_BLOCK_PATTERN = re.compile(r"```\n(.*?)\n```", re.DOTALL)
_PYTHON_BLOCK_PATTERN = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def extract_python_block(completion: str | None) -> tuple[str | None, str | None]:
    if completion is None:
        return None, "null completion."
    if completion == "":
        return None, "empty completion."
    m = _PYTHON_BLOCK_PATTERN.search(completion)
    if m:
        return m.group(1).strip(), None
    m = _CODE_BLOCK_PATTERN.search(completion)
    if m:
        return m.group(1).strip(), None
    return None, "no python code block found."


def _worker_exec_transform(code: str, input_grid: list[list[int]], queue: mp.Queue) -> None:
    try:
        local_ns: dict[str, Any] = {"np": np}
        exec(code, local_ns, local_ns)
        fn = local_ns.get("transform")
        if not callable(fn):
            queue.put(
                {
                    "status": "error",
                    "error": "function lookup error: expected callable `transform`.",
                    "output": None,
                }
            )
            return

        x = np.array(input_grid, dtype=int)
        y = fn(x)
        if isinstance(y, list):
            y = np.array(y, dtype=int)
        if not isinstance(y, np.ndarray):
            queue.put(
                {
                    "status": "error",
                    "error": f"return type error: function returned {type(y)}",
                    "output": None,
                }
            )
            return
        if y.ndim != 2:
            queue.put(
                {
                    "status": "error",
                    "error": f"return shape error: function returned shape {y.shape}",
                    "output": None,
                }
            )
            return
        y = np.array(y, dtype=int)
        if not np.all((0 <= y) & (y <= 9)):
            queue.put(
                {
                    "status": "error",
                    "error": "return value range error: values outside [0, 9]",
                    "output": None,
                }
            )
            return
        queue.put({"status": "ok", "error": None, "output": y.tolist()})
    except Exception as exc:  # pragma: no cover - depends on model output
        queue.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}",
                "output": None,
            }
        )


def execute_transform(
    code: str,
    input_grid: list[list[int]],
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker_exec_transform, args=(code, input_grid, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "error", "error": f"timeout error (> {timeout_s}s)", "output": None}
    if queue.empty():
        return {"status": "error", "error": "execution error: empty worker response", "output": None}
    return queue.get()

