"""Shared numeric comparison used by tests and adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_ATOL = 1e-4


def scores_close(a: Any, b: Any, *, atol: float = DEFAULT_ATOL, path: str = "") -> bool:
    """Return True if *a* is contained in *b* (floats compared with ``atol``).

    Pair-index dicts are sorted before comparison. Extra keys in *b* are allowed.
    """
    try:
        assert_equiv_results(a, b, path=path, atol=atol)
    except AssertionError:
        return False
    return True


def assert_equiv_results(a: Any, b: Any, path: str = "", atol: float = DEFAULT_ATOL) -> bool:
    """Assert JSON-like *a* is contained in *b*, using ``np.isclose`` for floats."""
    if type(a) in (int, float):
        assert type(b) in (int, float), path
    else:
        assert type(a) == type(b), path

    if isinstance(a, (int, str)):
        assert a == b, path

    if isinstance(a, float):
        assert np.isclose(a, b, atol=atol), f"{path}: {a} vs {b}"

    if isinstance(a, dict):
        if "pair_idx" in a:
            assert "pair" in a
            a = _canonicalize_pair_idx(a)
            b = _canonicalize_pair_idx(b)
        assert len(set(a) - set(b)) == 0, path
        for k in a:
            nxt = f"{path}{'' if path == '' else '.'}{k}"
            assert_equiv_results(a[k], b[k], nxt, atol)

    if isinstance(a, list):
        assert len(a) == len(b), path
        for i in range(len(a)):
            assert_equiv_results(a[i], b[i], f"{path}[{i}]", atol)

    return True


def _canonicalize_pair_idx(a: dict) -> dict:
    a = a.copy()
    ix = [(tuple(sorted(idxs)), x) for idxs, x in zip(a["pair_idx"], a["pair"])]
    ix.sort()
    a["pair_idx"] = [i for i, _ in ix]
    a["pair"] = [x for _, x in ix]
    return a
