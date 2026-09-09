"""v0 is a deprecated wrap of v1 with legacy scoring defaults."""

from __future__ import annotations

import importlib
import sys
import warnings

from xenosite.predict.registry import ensure_builtins, load_runner, registered
from xenosite.predict.v1.legacy import LegacyRunner


def test_v0_import_emits_deprecation():
    sys.modules.pop("xenosite.predict.v0", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v0 = importlib.import_module("xenosite.predict.v0")
        assert v0.LegacyRunner is LegacyRunner
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("xenosite.predict.v0 is deprecated" in str(w.message) for w in caught)


def test_v0_legacy_alias_emits_deprecation_and_loads_v1():
    for name in list(sys.modules):
        if name == "xenosite.predict.v0_legacy" or name.startswith(
            "xenosite.predict.v0_legacy."
        ):
            sys.modules.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module("xenosite.predict.v0_legacy.models.phase1")
        assert mod.Phase1Runner.name == "phase1"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("v0_legacy has moved to xenosite.predict.v1" in str(w.message) for w in caught)


def test_version_0_runner_is_legacy_wrap():
    ensure_builtins()
    runner = load_runner("epoxidation", "0")
    assert isinstance(runner, LegacyRunner)
    assert runner.version == "0"
    names = {(i.name, i.version) for i in registered()}
    assert ("epoxidation", "0") in names
    assert ("epoxidation", "1") in names
    assert load_runner("epoxidation", "1").version == "1"
    assert not isinstance(load_runner("epoxidation", "1"), LegacyRunner)
