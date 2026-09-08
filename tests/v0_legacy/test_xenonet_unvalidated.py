"""XenoNet import warns until golden fixtures exist."""

from __future__ import annotations

import warnings

import pytest

from xenosite.predict.v0_legacy.xenonet import _unvalidated


def test_warn_unvalidated_once(monkeypatch):
    monkeypatch.setattr(_unvalidated, "_emitted", False)
    with pytest.warns(UserWarning, match="not validated until golden"):
        _unvalidated.warn_unvalidated()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _unvalidated.warn_unvalidated()
