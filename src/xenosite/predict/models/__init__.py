"""Built-in models. Importing this package registers ``(name, version)`` factories."""

from __future__ import annotations

_LOADED = False


def load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import bioactivation as bioactivation  # noqa: F401
    from . import epoxidation as epoxidation  # noqa: F401
    from . import ndealk as ndealk  # noqa: F401
    from . import phase1 as phase1  # noqa: F401
    from . import quinone as quinone  # noqa: F401
    from . import reactivity as reactivity  # noqa: F401
    from . import ugt as ugt  # noqa: F401

    _LOADED = True
