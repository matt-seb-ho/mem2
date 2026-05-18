from __future__ import annotations

import asyncio
import contextvars
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mem2.core.entities import to_primitive

_TRACE_DIR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mem2_case_trace_dir", default=None
)
_TASK_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mem2_case_task_id", default=None
)
_ITER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mem2_case_iter_id", default=None
)


def set_trace_context(trace_dir: str | Path, task_id: str, iter_id: str | int) -> tuple:
    """Set per-problem tracing context for the active async task."""
    return (
        _TRACE_DIR.set(str(trace_dir)),
        _TASK_ID.set(str(task_id)),
        _ITER_ID.set(str(iter_id)),
    )


def reset_trace_context(tokens: tuple | None) -> None:
    if not tokens:
        return
    for var, token in zip((_TRACE_DIR, _TASK_ID, _ITER_ID), tokens):
        var.reset(token)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    primitive = to_primitive(value)
    if isinstance(primitive, (str, int, float, bool, type(None), list, dict)):
        return primitive
    return str(primitive)


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_primitive(data), indent=2, sort_keys=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _active_trace_dir(default_trace_dir: str | Path | None = None) -> Path | None:
    trace_dir = _TRACE_DIR.get() or default_trace_dir
    if not trace_dir:
        return None
    return Path(trace_dir)


def _active_iter_dir(
    default_trace_dir: str | Path | None = None,
    task_id: str | None = None,
    iter_id: str | int | None = None,
) -> Path | None:
    trace_dir = _active_trace_dir(default_trace_dir)
    task = task_id or _TASK_ID.get()
    iteration = iter_id if iter_id is not None else _ITER_ID.get()
    if trace_dir is None or task is None or iteration is None:
        return None
    return (
        trace_dir
        / "problems"
        / _safe_path_part(task)
        / f"iter_{_safe_path_part(iteration)}"
    )


def _safe_path_part(value: object) -> str:
    text = str(value)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(to_primitive(prompt), indent=2, sort_keys=False)


def _responses_to_text(responses: Any) -> str:
    if isinstance(responses, list):
        flattened: list[str] = []
        for idx, item in enumerate(responses):
            if isinstance(item, list):
                for sub_idx, sub in enumerate(item):
                    flattened.append(f"### response[{idx}][{sub_idx}]\n{sub or ''}")
            else:
                flattened.append(f"### response[{idx}]\n{item or ''}")
        return "\n\n".join(flattened)
    return str(responses)


def _numeric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_first_number(snapshot: Any, names: set[str]) -> float | None:
    if isinstance(snapshot, dict):
        for key, value in snapshot.items():
            if str(key).lower() in names:
                found = _numeric_value(value)
                if found is not None:
                    return found
            found = _find_first_number(value, names)
            if found is not None:
                return found
    elif isinstance(snapshot, list):
        for item in snapshot:
            found = _find_first_number(item, names)
            if found is not None:
                return found
    return None


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fields = [
        ("cost_usd_delta", {"cost_usd", "total_cost_usd", "total_cost", "cost"}),
        ("tokens_in_delta", {"tokens_in", "input_tokens", "prompt_tokens"}),
        ("tokens_out_delta", {"tokens_out", "output_tokens", "completion_tokens"}),
    ]
    for label, keys in fields:
        before_value = _find_first_number(before, keys)
        after_value = _find_first_number(after, keys)
        if before_value is not None and after_value is not None:
            out[label] = after_value - before_value
    return out


def _count_completions(result: Any) -> int:
    if isinstance(result, list):
        if result and all(isinstance(row, list) for row in result):
            return sum(len(row) for row in result)
        return len(result)
    return 1


class TraceCollectingProviderClient:
    """Provider wrapper that persists prompts, responses, and call metadata."""

    def __init__(self, provider: Any, trace_dir: str | Path):
        self.provider = provider
        self.trace_dir = str(trace_dir)
        self.name = getattr(provider, "name", "provider")
        self.version = getattr(provider, "version", "unknown")
        self.supports_multi_completion = bool(
            getattr(provider, "supports_multi_completion", True)
        )
        self._write_lock = asyncio.Lock()
        self._call_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    async def async_generate(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[str]:
        before = self.get_usage_snapshot()
        start = time.perf_counter()
        try:
            result = await self.provider.async_generate(prompt, model, gen_cfg)
        except Exception as exc:
            await self._write_call(
                request_kind="generate",
                prompts=[prompt],
                result=[],
                model=model,
                gen_cfg=gen_cfg,
                latency_s=time.perf_counter() - start,
                usage_before=before,
                usage_after=self.get_usage_snapshot(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        after = self.get_usage_snapshot()
        await self._write_call(
            request_kind="generate",
            prompts=[prompt],
            result=result,
            model=model,
            gen_cfg=gen_cfg,
            latency_s=time.perf_counter() - start,
            usage_before=before,
            usage_after=after,
        )
        return result

    async def async_batch_generate(
        self,
        prompts: list[str | list[dict[str, Any]]],
        model: str,
        gen_cfg: dict[str, Any],
        request_timeout: float | None = 300.0,
    ) -> list[list[str | None]]:
        before = self.get_usage_snapshot()
        start = time.perf_counter()
        try:
            result = await self.provider.async_batch_generate(
                prompts, model, gen_cfg, request_timeout=request_timeout
            )
        except TypeError:
            result = await self.provider.async_batch_generate(prompts, model, gen_cfg)
        except Exception as exc:
            await self._write_call(
                request_kind="batch_generate",
                prompts=prompts,
                result=[],
                model=model,
                gen_cfg=gen_cfg,
                latency_s=time.perf_counter() - start,
                usage_before=before,
                usage_after=self.get_usage_snapshot(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        after = self.get_usage_snapshot()
        await self._write_call(
            request_kind="batch_generate",
            prompts=prompts,
            result=result,
            model=model,
            gen_cfg=gen_cfg,
            latency_s=time.perf_counter() - start,
            usage_before=before,
            usage_after=after,
        )
        return result

    def get_usage_snapshot(self) -> dict[str, Any]:
        try:
            return to_primitive(self.provider.get_usage_snapshot())
        except Exception:
            return {}

    async def _write_call(
        self,
        *,
        request_kind: str,
        prompts: list[str | list[dict[str, Any]]],
        result: Any,
        model: str,
        gen_cfg: dict[str, Any],
        latency_s: float,
        usage_before: dict[str, Any],
        usage_after: dict[str, Any],
        error: str | None = None,
    ) -> None:
        iter_dir = _active_iter_dir(self.trace_dir)
        if iter_dir is None:
            return
        async with self._write_lock:
            call_idx = self._call_index
            self._call_index += 1
            call_dir = iter_dir / "llm_calls" / f"call_{call_idx:04d}"
            call_dir.mkdir(parents=True, exist_ok=True)

            prompt_text = "\n\n### prompt separator\n\n".join(
                _prompt_to_text(prompt) for prompt in prompts
            )
            response_text = _responses_to_text(result)
            (call_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            (call_dir / "response.txt").write_text(response_text, encoding="utf-8")

            call_meta = {
                "provider": self.name,
                "provider_version": self.version,
                "wrapped_provider": getattr(
                    self.provider, "name", type(self.provider).__name__
                ),
                "request_kind": request_kind,
                "model": model,
                "gen_cfg": to_primitive(gen_cfg or {}),
                "latency_s": latency_s,
                "prompt_count": len(prompts),
                "n_completions": _count_completions(result),
                "usage_before": usage_before,
                "usage_after": usage_after,
                **_usage_delta(usage_before, usage_after),
            }
            if error:
                call_meta["error"] = error
            _write_json(call_dir / "call_meta.json", call_meta)

            for filename, content in [
                ("prompt.txt", prompt_text),
                ("response.txt", response_text),
            ]:
                top_level = iter_dir / filename
                if not top_level.exists():
                    top_level.write_text(content, encoding="utf-8")
            top_meta = iter_dir / "call_meta.json"
            if not top_meta.exists():
                _write_json(top_meta, call_meta)


def write_retrieval_bundle(
    retrieval: Any,
    trace_dir: str | Path | None = None,
    task_id: str | None = None,
    iter_id: str | int | None = None,
) -> Path | None:
    iter_dir = _active_iter_dir(trace_dir, task_id, iter_id)
    if iter_dir is None:
        return None
    payload = to_primitive(retrieval)
    if isinstance(payload, dict):
        payload.setdefault("metadata", {})
    return _write_json(iter_dir / "retrieval_bundle.json", payload)


def write_attempt_eval_trace(
    attempts: list[Any],
    eval_records: list[Any],
    trace_dir: str | Path | None = None,
    task_id: str | None = None,
    iter_id: str | int | None = None,
) -> dict[str, Path] | None:
    iter_dir = _active_iter_dir(trace_dir, task_id, iter_id)
    if iter_dir is None:
        return None

    try:
        from mem2.utils.code_execution import extract_python_block
    except Exception:
        extract_python_block = None

    parsed_attempts = []
    for idx, attempt in enumerate(attempts):
        completion = getattr(attempt, "completion", None)
        code = None
        parsing_error = None
        if extract_python_block is not None:
            code, parsing_error = extract_python_block(completion)
        parsed_attempts.append(
            {
                "attempt_idx": idx,
                "problem_uid": getattr(attempt, "problem_uid", None),
                "pass_idx": getattr(attempt, "pass_idx", None),
                "branch_id": getattr(attempt, "branch_id", None),
                "python_block": code,
                "parsing_error": parsing_error,
                "metadata": to_primitive(getattr(attempt, "metadata", {})),
            }
        )

    eval_payload = {
        "records": to_primitive(eval_records),
        "correct": any(
            bool(getattr(record, "is_correct", False)) for record in eval_records
        ),
    }
    if attempts:
        response_text = _responses_to_text(
            [getattr(attempt, "completion", "") for attempt in attempts]
        )
        (iter_dir / "response.txt").write_text(response_text, encoding="utf-8")
    return {
        "parsed": _write_json(iter_dir / "parsed.json", {"attempts": parsed_attempts}),
        "eval": _write_json(iter_dir / "eval.json", eval_payload),
    }


def write_run_meta(trace_dir: str | Path | None, meta: dict[str, Any]) -> Path | None:
    if not trace_dir:
        return None
    meta_path = Path(trace_dir) / "meta.json"
    current = _read_json(meta_path)
    current.update(to_primitive(meta))
    return _write_json(meta_path, current)


def count_llm_calls(trace_dir: str | Path | None) -> int:
    if not trace_dir:
        return 0
    root = Path(trace_dir)
    if not root.exists():
        return 0
    return sum(
        1 for _ in root.glob("problems/*/iter_*/llm_calls/call_*/call_meta.json")
    )
