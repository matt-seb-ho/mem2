"""Sync adapter for the async provider, used by LLM-aware memory builders.

Memory builders (F.2 ALMA, F.3 ADAS, A.4 LILO, A.6 A-MEM, A.7 Memp) call the
provider with a sync signature:

    completions = provider.generate(prompt, model=...)  # -> list[str]

The underlying `LLMPlusProviderClient` only exposes `async_generate`. Builders
run inside a sync `consolidate()` that itself is called from an already-running
event loop (the orchestrator). We can't use `asyncio.run` directly (it would
conflict with the running loop). We also can't `await` from sync code.

Solution: run the coroutine on a dedicated background thread with its own
event loop, blocking the caller via `future.result()`. The thread is spun up
lazily on the first call and reused across the run.

Usage in runner.py (after build_run_context):

    if llm_aware_builder(config):
        adapter = SyncMetaEditProviderAdapter(self.components.provider, model=...)
        ctx.config["_meta_edit_provider"] = adapter
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Wall-clock timeout per meta-edit LLM call. Bounds the consolidation path
# similarly to python_transform_retry._LLM_CALL_TIMEOUT_S. Guards against
# tenacity-retry-stack stalls on the meta-edit path. RN-003 F2 follow-up.
_META_EDIT_TIMEOUT_S = 900.0  # 2026-05-29: 300->900; dsv4f long-reasoning (see client.py)


class SyncMetaEditProviderAdapter:
    """Expose `.generate(prompt, model=...)` → list[str] via a sync facade."""

    def __init__(
        self,
        async_provider,
        *,
        model: str | None = None,
        gen_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._provider = async_provider
        self.model = model or getattr(async_provider, "_model", "")
        self._gen_cfg = dict(gen_cfg or {})
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> list[str]:
        target_model = model or self.model
        loop = self._ensure_loop()
        merged_cfg = {**self._gen_cfg, **kwargs}
        n = int(merged_cfg.get("n", 1) or 1)

        async def _bounded_call() -> list[str]:
            return await asyncio.wait_for(
                self._provider.async_generate(
                    prompt=prompt,
                    model=target_model,
                    gen_cfg=merged_cfg,
                ),
                timeout=_META_EDIT_TIMEOUT_S,
            )

        fut = asyncio.run_coroutine_threadsafe(_bounded_call(), loop)
        try:
            return fut.result()
        except asyncio.TimeoutError:
            logger.warning(
                "meta_edit LLM call exceeded %.0fs wall-clock — returning %d empty completions "
                "(consolidation step skipped for this call).",
                _META_EDIT_TIMEOUT_S, n,
            )
            return [""] * n

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread and self._thread.is_alive():
                return self._loop

            loop = asyncio.new_event_loop()

            def _runner(loop_: asyncio.AbstractEventLoop) -> None:
                asyncio.set_event_loop(loop_)
                try:
                    loop_.run_forever()
                finally:
                    loop_.close()

            thread = threading.Thread(
                target=_runner, args=(loop,), daemon=True,
                name="mem2-meta-edit-provider-loop",
            )
            thread.start()
            self._loop = loop
            self._thread = thread
            return loop

    def shutdown(self) -> None:
        with self._lock:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
            self._thread = None
