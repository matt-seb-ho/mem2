"""Preflight + concurrency probe for the official DeepSeek API (deepseek-v4-flash).

Fires a single call to confirm auth/usage accounting, then ramps concurrency to
find a safe ceiling before we launch the full vanilla solve. Tokens-only accounting.

Usage:
    python scripts/deepseek_preflight.py                # single-call preflight
    python scripts/deepseek_preflight.py --levels 8 16 32 64   # concurrency ramp
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "third_party" / "llm_wrapper"
if str(VENDORED) not in sys.path:
    sys.path.insert(0, str(VENDORED))

from llmplus import GenerationConfig, LLMClient, RetryConfig  # noqa: E402
from llmplus.model_registry import Provider  # noqa: E402

MODEL = "deepseek-v4-flash"
DOTENV = str(REPO_ROOT / ".env")


def _make_client(max_concurrency: int) -> LLMClient:
    return LLMClient(
        provider=Provider.DEEPSEEK,
        cache_dir=str(REPO_ROOT / ".llm_cache"),
        default_max_concurrency=max_concurrency,
        retry_cfg=RetryConfig(attempts=2, wait_min=1, wait_max=20),
        dotenv_path=DOTENV,
    )


async def _single(client: LLMClient) -> None:
    gen = GenerationConfig(n=1, temperature=0.0, max_tokens=64, ignore_cache=True)
    t0 = time.time()
    out = await client.async_generate(
        prompt="Reply with exactly: PONG", model=MODEL, gen_cfg=gen
    )
    dt = time.time() - t0
    print(f"[preflight] ok in {dt:.2f}s — completion: {out[0]!r}")
    snap = client.get_token_usage_dict()
    print(f"[preflight] usage snapshot: {snap}")


async def _ramp(client: LLMClient, level: int) -> tuple[int, int, float, str]:
    gen = GenerationConfig(n=1, temperature=0.3, max_tokens=128, ignore_cache=True)
    prompts = [f"Briefly: what is {i}+{i}? Answer in one short sentence." for i in range(level)]
    t0 = time.time()
    ok, err, msg = 0, 0, ""
    results = await asyncio.gather(
        *[client.async_generate(prompt=p, model=MODEL, gen_cfg=gen) for p in prompts],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            err += 1
            msg = f"{type(r).__name__}: {r}"
        else:
            ok += 1
    return ok, err, time.time() - t0, msg


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="*", default=[])
    args = ap.parse_args()

    print(f"[preflight] model={MODEL} dotenv={DOTENV}")
    client = _make_client(max_concurrency=max(args.levels or [1]))
    await _single(client)

    for level in args.levels:
        ok, err, dt, msg = await _ramp(client, level)
        rate = ok / dt if dt else 0.0
        flag = "  <-- ERRORS" if err else ""
        print(
            f"[ramp] concurrency={level:>3}  ok={ok:>3} err={err:>3}  "
            f"{dt:5.1f}s  {rate:4.1f} req/s{flag}  {msg}"
        )
        if err:
            print("[ramp] stopping ramp — hit errors (likely rate limit).")
            break

    print(f"[preflight] final usage: {client.get_token_usage_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
