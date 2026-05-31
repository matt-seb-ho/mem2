import asyncio
import dataclasses
import logging
import random
import warnings
from datetime import UTC, datetime
from pathlib import Path

import diskcache as dc
import httpx
import orjson
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from tqdm import tqdm

from .configs import GenerationConfig, RetryConfig
from .model_registry import MODEL_REGISTRY, ModelMeta, Provider
from .model_token_usage import ModelTokenUsage
from .sync_adapter import syncify
from .utils import stable_hash, transient_retry

# Libraries should ship no handlers to avoid double logging or “No handlers…” warnings.
# - if the host application calls logging.basicConfig() (or sets handlers/levels explicitly),
#   this library’s logs will propagate and honour that configuration.
# - if users do nothing, this library stays silent
# - users have full control by touching the root (or a parent) logger.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LLMClient:
    def __init__(
        self,
        provider: Provider = Provider.OPENAI,
        cache_dir: str = ".llm_cache",
        system_prompt: str | None = None,
        default_max_concurrency: int = 32,
        retry_cfg: RetryConfig | None = None,
        dotenv_path: str | Path | None = None,
    ):
        self.provider_meta = MODEL_REGISTRY[provider]
        self._client_async = AsyncOpenAI(
            api_key=self.provider_meta.api_key(dotenv_path=dotenv_path),
            base_url=self.provider_meta.base_url,
            # 300s read ceiling, 10s connect. Prevents indefinite hang when an
            # OpenRouter backend stops responding mid-stream (witnessed
            # 2026-04-30: flat_topk seed 44 iter-3 stalled 18+ min on a single
            # call). The asyncio.wait_for in async_batch_generate is
            # belt-and-suspenders; this is the suspenders.
            # 2026-05-29: raised 300->900. The 300s read ceiling (added 2026-04-30)
            # was silently killing dsv4f's long-reasoning generations mid-stream
            # (reasoning models can emit 30k-55k tokens > 300s), recorded as empty
            # completions -> 91% empty on val100 / score collapse 76->19. April's
            # 76.5 run predated this cap (openai SDK default ~600s). 900 exceeds it.
            timeout=httpx.Timeout(900.0, connect=10.0),
        )

        # default semaphore (used when caller does *not* supply one)
        self._default_sem = asyncio.Semaphore(default_max_concurrency)

        # caches
        self._resp_cache = dc.Cache(cache_dir)
        self.session_stats: dict[str, ModelTokenUsage] = {}

        # timestamps
        self._session_start = datetime.now(tz=UTC).isoformat()
        self._last_request: str | None = None

        # retry decorator
        self.retry_cfg = retry_cfg or RetryConfig()
        self._retry_deco = None
        self._configure_retry(self.retry_cfg)

        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # core async generation api
    # ------------------------------------------------------------------
    async def async_generate(
        self,
        prompt: str | list[dict],
        model: str | None = None,
        gen_cfg: GenerationConfig | None = None,
        request_sem: asyncio.Semaphore | None = None,
        **gen_kwargs,
    ) -> list[str]:
        # update generation config with kwarg overrides
        gen_cfg = gen_cfg or GenerationConfig()
        if gen_kwargs:
            gen_cfg = gen_cfg.override(**gen_kwargs)
        n = gen_cfg.n

        model = model or next(iter(self.provider_meta.models))
        assert model in self.provider_meta.models, f"Unknown model {model}"
        mmeta = self.provider_meta.models[model]

        cache_key = self._make_cache_key(prompt, model, mmeta, gen_cfg)
        cached = self._resp_cache.get(cache_key) or []
        usable_cached = [] if gen_cfg.ignore_cache else cached
        if usable_cached and len(usable_cached) >= n:
            return random.sample(usable_cached, n)

        # only fetch missing samples
        # select whether to expand multi‑sample requests if not specified
        updated_cfg = dataclasses.replace(
            gen_cfg,
            n=(n - len(usable_cached)),
            expand_multi=(
                (not self.provider_meta.supports_multi)
                if gen_cfg.expand_multi is None
                else gen_cfg.expand_multi
            ),
        )
        fetched = await self._request_completions(
            prompt=prompt,
            model_meta=mmeta,
            gen_cfg=updated_cfg,
            request_sem=request_sem,
        )
        new_nonempty_responses = [r for r in fetched if r and r.strip() != ""]
        self._resp_cache[cache_key] = cached + new_nonempty_responses
        return usable_cached + fetched

    # ------------------------------------------------------------------
    # batch helper – single semaphore governs *HTTP* concurrency
    # ------------------------------------------------------------------
    async def async_batch_generate(
        self,
        prompts: list[str | list[dict]],
        model: str | None = None,
        gen_cfg: GenerationConfig | None = None,
        progress_file: str | Path = "batch_progress.json",
        show_progress: bool = True,
        request_timeout: float | None = 300.0,
        **gen_kwargs,
    ) -> list[list[str | None]]:
        request_sem = asyncio.Semaphore(gen_cfg.batch_size)
        results: list[list[str]] = [[] for _ in prompts]
        pbar = tqdm(total=len(prompts), disable=not show_progress)
        file_path = Path(progress_file).expanduser() if progress_file else None
        file_lock = asyncio.Lock()
        num_samples = gen_kwargs.get("n", (1 if gen_cfg is None else gen_cfg.n))
        variable_num_samples = isinstance(num_samples, list)
        if variable_num_samples:
            assert len(num_samples) == len(prompts)

        async def _job(idx: int, prm: str | list[dict]):
            try:
                if variable_num_samples:
                    gen_kwargs["n"] = num_samples[idx]
                coro = self.async_generate(
                    prm,
                    model=model,
                    gen_cfg=gen_cfg,
                    request_sem=request_sem,
                    **gen_kwargs,
                )
                if request_timeout:
                    res = await asyncio.wait_for(coro, timeout=request_timeout)
                else:
                    res = await coro
                results[idx] = res
            except asyncio.TimeoutError:
                logger.warning("prompt %s timed out after %.0fs", idx, request_timeout)
                results[idx] = [None] * gen_kwargs.get("n", 1)
            except Exception as e:
                msg = f"prompt {idx} failed: type: {type(e)}, message: {e}"
                # logger.error("Prompt %s failed: %s", idx, e, exc_info=False)
                logger.exception(msg, exc_info=True)
                results[idx] = [None] * gen_kwargs.get("n", 1)
            finally:
                pbar.update()
                if file_path:
                    async with file_lock:
                        tmp = file_path.with_suffix(".tmp")
                        tmp.write_bytes(orjson.dumps(results))
                        tmp.replace(file_path)

        await asyncio.gather(*(_job(i, p) for i, p in enumerate(prompts)))
        pbar.close()
        return results

    # ------------------------------------------------------------------
    # token usage
    # ------------------------------------------------------------------
    def get_token_usage(
        self,
        model: str,
    ) -> ModelTokenUsage:
        if model not in self.session_stats:
            self.session_stats[model] = ModelTokenUsage()
        return self.session_stats[model]

    def get_token_usage_dict(self, include_per_request: bool = False) -> dict[str, dict]:
        """Get token usage statistics as a dictionary."""
        return {
            model: stats.to_dict(include_per_request=include_per_request)
            for model, stats in self.session_stats.items()
        }

    def reset_usage(self) -> None:
        """Reset the session token usage statistics."""
        self.session_stats = {}
        self._session_start = datetime.now(tz=UTC).isoformat()
        self._last_request = None

    def save_session_usage(self, path: str | Path = "session_usage.json") -> None:
        record = {
            "session_start": self._session_start,
            "session_end": self._last_request,
            "stats": self.session_stats,
        }
        Path(path).write_bytes(orjson.dumps(record))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    async def _request_completions(
        self,
        prompt: str | list[dict],
        model_meta: ModelMeta,
        gen_cfg: GenerationConfig,
        request_sem: asyncio.Semaphore | None = None,
    ) -> list[str]:
        n = gen_cfg.n
        if n <= 0:
            return []
        request_sem = request_sem or self._default_sem
        if gen_cfg.expand_multi:
            expand_multi_cfg = dataclasses.replace(gen_cfg, n=1)
            tasks = [
                self._async_request(
                    prompt=prompt,
                    model_meta=model_meta,
                    gen_cfg=expand_multi_cfg,
                    request_sem=request_sem,
                )
                for _ in range(n)
            ]
            results: list[str] = []
            for coro in asyncio.as_completed(tasks):
                try:
                    results.extend(await coro)
                except Exception as e:
                    msg = f"one sub‑request failed. type: {type(e).__name__}, message: {e}"
                    logger.exception(msg, exc_info=True)
                    warnings.warn(msg, stacklevel=2)
            return results
        else:
            try:
                return await self._async_request(
                    prompt=prompt,
                    model_meta=model_meta,
                    gen_cfg=gen_cfg,
                    request_sem=request_sem,
                )
            except Exception as e:
                msg = f"LLM request failed: {type(e).__name__}: {e}"
                logger.error(msg, exc_info=False)
                warnings.warn(msg, stacklevel=2)
                return []

    async def _async_request(
        self,
        prompt: str | list[dict],
        model_meta: ModelMeta,
        gen_cfg: GenerationConfig,
        request_sem: asyncio.Semaphore,
    ) -> list[str]:
        @self._retry_deco
        async def _send():
            async with request_sem:
                msgs = self._format_chat(prompt)
                gen_kwargs = gen_cfg.to_kwargs(model_meta)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "API request: model=%s kwargs=%s msg_count=%s",
                        model_meta.name,
                        gen_kwargs,
                        len(msgs),
                    )
                resp = await self._client_async.chat.completions.create(
                    model=model_meta.name,
                    messages=msgs,
                    timeout=900.0,  # 2026-05-29: raised 300->900 (see __init__).
                                    # explicit per-call total bound; client-level
                                    # httpx Timeout(read=900) only bounds inter-byte
                                    # gaps, not total wall time.
                    **gen_kwargs,
                )
                return resp

        resp = await _send()
        self._add_token_usage(model_meta.name, resp.usage, len(resp.choices))
        results: list[str] = []
        for i, choice in enumerate(resp.choices):
            content = choice.message.content
            if content is None:
                if choice.message.refusal:
                    logger.warning(
                        "Model refused to respond (choice %s): %s",
                        i,
                        choice.message.refusal,
                    )
                else:
                    logger.warning(
                        "Model returned None content (choice %s, finish_reason=%s)",
                        i,
                        choice.finish_reason,
                    )
                content = ""
            results.append(content)
        return results

    def _format_chat(self, inp: str | list[dict]):
        if isinstance(inp, str):
            msgs = [{"role": "user", "content": inp}]
        else:
            msgs = inp
        if self.system_prompt:
            msgs = [{"role": "system", "content": self.system_prompt}, *msgs]
        return msgs

    def _add_token_usage(
        self, model: str, usage: CompletionUsage, num_completions: int
    ):
        stats = self.get_token_usage(model)
        stats.update(usage, num_completions=num_completions)
        self._last_request = datetime.now(tz=UTC).isoformat()

    def _make_cache_key(
        self,
        prompt: str | list[dict],
        model: str,
        model_meta: ModelMeta,
        args: GenerationConfig,
    ):
        key_attributes = args.to_kwargs(model_meta)
        key_attributes.pop("n", None)
        key_attributes["model"] = model
        key_attributes["prompt"] = self._format_chat(prompt)
        return stable_hash(key_attributes)

    def _configure_retry(self, retry_cfg: RetryConfig | None = None):
        self._retry_deco = transient_retry(
            **(dataclasses.asdict(retry_cfg or self.retry_cfg))
        )

    # ------------------------------------------------------------------
    # sync facade (one-liners that call run_sync)
    # ------------------------------------------------------------------
    generate = syncify(async_generate)
    batch_generate = syncify(async_batch_generate)
