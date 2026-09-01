"""Progress bars for long tool loops (capture, gather, drift analysis)."""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import Any, TextIO, TypeVar

T = TypeVar("T")
U = TypeVar("U")

_tty_stream: TextIO | None = None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _progress_disabled(explicit: bool) -> bool:
    return explicit or _truthy_env("XENOSITE_NO_PROGRESS")


def _progress_stream() -> TextIO:
    """Progress bars on /dev/tty when available; fallback stderr."""
    global _tty_stream
    if _truthy_env("XENOSITE_PROGRESS_STDERR"):
        return sys.stderr
    if _tty_stream is None:
        try:
            _tty_stream = open("/dev/tty", "w")
        except OSError:
            return sys.stderr
    return _tty_stream


def worker_quiet() -> None:
    """Suppress noisy library logs in worker processes (keeps tqdm readable)."""
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore")


def _tqdm(total: int, *, desc: str, unit: str):
    from tqdm import tqdm

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        file=_progress_stream(),
        disable=False,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.2,
    )


def iter_progress(
    items: Iterable[T],
    *,
    desc: str = "",
    unit: str = "it",
    total: int | None = None,
    disable: bool = False,
    note: str = "",
) -> Iterator[T]:
    """Iterate with a tqdm bar (always on unless ``XENOSITE_NO_PROGRESS=1``)."""
    if _progress_disabled(disable):
        yield from items
        return
    n = total if total is not None else (len(items) if isinstance(items, Sized) else None)
    if not n:
        yield from items
        return
    from tqdm import tqdm

    bar = tqdm(
        items,
        total=n,
        desc=desc,
        unit=unit,
        file=_progress_stream(),
        disable=False,
        leave=True,
        mininterval=0.2,
    )
    if note:
        bar.write(note)
    yield from bar


def progress_bar(
    total: int,
    *,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    """Context manager for manual updates."""
    if _progress_disabled(disable) or total == 0:
        from contextlib import nullcontext

        class _NoOp:
            def update(self, n: int = 1) -> None:
                pass

            def write(self, msg: str) -> None:
                print(msg, file=sys.stderr, flush=True)

        return nullcontext(_NoOp())
    return _tqdm(total, desc=desc, unit=unit)


def map_progress(
    fn: Callable[[T], U],
    items: list[T],
    *,
    pool,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
    note: str = "",
    on_result: Callable[[U, Any], None] | None = None,
) -> list[U]:
    """Run *fn* on *items* in *pool* with a live tqdm bar on stderr."""
    from concurrent.futures import as_completed

    total = len(items)
    if _progress_disabled(disable) or total == 0:
        return list(pool.map(fn, items))

    bar = _tqdm(total, desc=desc, unit=unit)
    out: list[U] = []
    try:
        if note:
            bar.write(note)
        futures = [pool.submit(fn, item) for item in items]
        for fut in as_completed(futures):
            result = fut.result()
            out.append(result)
            if on_result is not None:
                on_result(result, bar)
            bar.update(1)
    finally:
        bar.close()
    return out
