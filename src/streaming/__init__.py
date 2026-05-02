"""
Streaming ASR pipeline, debate wrappers, chunk bridges, and audio simulation for TreeDebater.

Use explicit submodule imports (recommended)::

    from streaming.env import StreamingDebateEnv, StreamingInputEnv
    from streaming.overlap import OverlappingStreamingDebateEnv

Or lazy submodules::

    import streaming
    env = streaming.env  # same as ``import streaming.env as env``

Runnable modules::

    python -m streaming.env --help
    python -m streaming.overlap --help
    python -m streaming.chunk_audio --help
    python -m streaming.run_listen_demo --help
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = frozenset({"bridges", "chunk_audio", "env", "overlap", "run_listen_demo"})
__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
