"""Single-line agent timing logs (decoupled from streaming parse_log_line format)."""

from __future__ import annotations

import itertools
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional

# While ``Agent.speak`` / ``TreeDebater.speak`` runs, I/O helpers reuse this ``call_id`` and
# ``speak_session`` so ``[timing-meta]``, ``[io]``, and ``log_llm_io`` stay aligned.
_speak_io_tls = threading.local()

# Monotonic correlation id per process (thread-safe enough for itertools.count in CPython GIL).
_call_id_counter = itertools.count(1)


def next_call_id() -> int:
    return next(_call_id_counter)


def set_speak_io_context(call_id: int, speak_session: str) -> None:
    """Bind ``call_id`` and ``speak_session`` for nested ``log_llm_io`` / ``post_process`` I/O."""
    _speak_io_tls.call_id = int(call_id)
    _speak_io_tls.speak_session = speak_session


def clear_speak_io_context() -> None:
    for attr in ("call_id", "speak_session"):
        if hasattr(_speak_io_tls, attr):
            delattr(_speak_io_tls, attr)


def get_speak_io_call_id() -> Optional[int]:
    return getattr(_speak_io_tls, "call_id", None)


def get_speak_io_session() -> Optional[str]:
    return getattr(_speak_io_tls, "speak_session", None)


def _fmt_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    s = str(v).replace("\n", " ").replace("\r", " ")
    if len(s) > 200:
        return s[:197] + "..."
    return s


def _timing_kv_parts(ctx: Mapping[str, Any]) -> list[str]:
    """Stable key order first, then remaining keys sorted."""
    preferred = (
        "stage",
        "side",
        "speak_session",
        "call_id",
        "pass_index",
        "add_evidence",
        "block",
        "iteration",
        "max_retry",
        "kind",
        "model",
        "cache_hit",
        "audio_duration_s",
        "n_claims",
    )
    seen = set()
    parts: list[str] = []
    for k in preferred:
        if k not in ctx:
            continue
        v = ctx[k]
        if v is None:
            continue
        parts.append(f"{k}={_fmt_val(v)}")
        seen.add(k)
    for k in sorted(ctx.keys()):
        if k in seen:
            continue
        v = ctx[k]
        if v is None:
            continue
        parts.append(f"{k}={_fmt_val(v)}")
    return parts


def format_timing_line(phase: str, duration_s: float, **ctx: Any) -> str:
    parts = ["[timing]", f"phase={phase}", f"duration_s={duration_s:.4f}", *_timing_kv_parts(ctx)]
    return " ".join(parts)


def log_timing(
    log: logging.Logger,
    phase: str,
    duration_s: float,
    *,
    level: int = logging.DEBUG,
    **ctx: Any,
) -> None:
    log.log(level, format_timing_line(phase, duration_s, **ctx))


@contextmanager
def timed_phase(
    log: logging.Logger,
    phase: str,
    *,
    log_start: bool = False,
    level: int = logging.DEBUG,
    **ctx: Any,
) -> Iterator[None]:
    if log_start:
        start_parts = ["[timing]", f"phase={phase}", "event=start", *_timing_kv_parts(ctx)]
        log.log(level, " ".join(start_parts))
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log_timing(log, phase, time.perf_counter() - t0, level=level, **ctx)


def log_io_block(
    io_log: logging.Logger,
    *,
    call_id: int,
    phase: str,
    title: str,
    body: str,
    level: int = logging.DEBUG,
    **ctx: Any,
) -> None:
    """One prompt/response block in the I/O log file (not on main debate logger)."""
    head = ["[io]", f"call_id={call_id}", f"phase={phase}", f"title={title}", *_timing_kv_parts(ctx)]
    header = " ".join(head)
    sep = "\n" + ("-" * 60) + "\n"
    io_log.log(level, header + sep + (body or "").rstrip() + "\n" + ("=" * 60))


def one_line_preview(s: str, max_len: int = 280) -> str:
    """Short single-line preview for main log when full body is in *_io.log."""
    t = (s or "").strip().replace("\n", " ||| ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 24] + "... [truncated]"


def log_llm_io(
    main_log: logging.Logger,
    *,
    phase: str,
    title: str,
    body: str,
    level: int = logging.DEBUG,
    emit_main_ref: bool = True,
    call_id: Optional[int] = None,
    io_block_phase: Optional[str] = None,
    **ctx: Any,
) -> None:
    """
    When I/O logging is enabled, write full ``body`` to *_io.log via :func:`log_io_block`.
    Otherwise emit the legacy single-line ``[{title}]`` message on ``main_log``.

    ``call_id``: reuse the id from ``[timing-meta]`` when inside ``set_speak_io_context`` or when
    passed explicitly; otherwise allocate a new id (e.g. BaselineDebater).

    ``io_block_phase``: value for ``[io] phase=...``; defaults to active ``speak_session`` from
    :func:`set_speak_io_context`, then to ``phase``. Keeps one speak turn under one session label
    (e.g. ``default_speak``) while ``title`` distinguishes Prompt vs Response vs TTS artifacts.
    """
    from utils.tool import io_logger, io_logging_enabled

    text = (body or "").strip()
    if io_logging_enabled():
        cid = call_id if call_id is not None else get_speak_io_call_id()
        if cid is None:
            cid = next_call_id()
        io_ph = io_block_phase if io_block_phase is not None else (get_speak_io_session() or phase)
        log_io_block(io_logger, call_id=cid, phase=io_ph, title=title, body=text, level=level, **ctx)
        if emit_main_ref:
            main_log.log(
                level,
                f"[io-ref] speak_session={io_ph} title={title} call_id={cid}",
            )
    else:
        main_log.log(level, f"[{title}] " + text.replace("\n", " ||| "))
