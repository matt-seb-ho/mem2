import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import nest_asyncio

_T = TypeVar("_T")


def run_sync(coro: Awaitable[_T]) -> _T:
    """
    Block until *coro* is complete, even if we're already inside
    a running asyncio loop (e.g. Jupyter).

    Raises whatever the coroutine raises.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running → simplest path
        return asyncio.run(coro)

    # We *are* inside a running loop → enable nested use
    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


def syncify(async_fn: Callable[..., Awaitable[_T]]) -> Callable[..., _T]:
    """
    Decorator: wrap an *async* function so it can be called like a
    synchronous one (internally uses `run_sync`).
    """

    @wraps(async_fn)
    def _wrapper(*args: Any, **kwargs: Any) -> _T:
        return run_sync(async_fn(*args, **kwargs))

    return _wrapper
