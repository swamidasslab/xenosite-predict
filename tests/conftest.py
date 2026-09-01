"""Pytest helpers: live skip, golden loader, float compare, env isolation."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from xenosite.predict.compare import DEFAULT_ATOL, assert_equiv_results, scores_close

from tests.support import GOLDEN, ROOT, load_golden, onnx_weights_present

COMPOSE = ROOT / "tools" / "legacy-test-api" / "compose.yml"

ENV_KEYS = (
    "XENOSITE_BACKEND",
    "XENOSITE_API_KEY",
    "XENOSITE_MODELS_WEIGHTS",
    "XENOSITE_LEGACY_TEST_URL",
    "XENOSITE_ONNX_URL",
    "XENOSITE_AUTO_DOWNLOAD",
)


@pytest.fixture(autouse=True)
def _clear_xenosite_env(monkeypatch):
    """Tests must not inherit a developer shell's XENOSITE_* variables."""
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def onnx_weights_present(model: str | None = None) -> bool:
    root = ROOT / "weights" / "onnx"
    if model:
        return any((root / model).glob("*.onnx"))
    return any(root.rglob("*.onnx"))


@pytest.fixture(scope="session")
def legacy_api_url():
    """Session fixture: reuse XENOSITE_LEGACY_TEST_URL or start compose; skip if impossible.

    Note: the autouse env clearer does not apply to this session fixture's lookup
    of XENOSITE_LEGACY_TEST_URL — we read os.environ before tests, but the
    session starts first. Tests that need the URL use this fixture, not env.
    """
    existing = os.environ.get("XENOSITE_LEGACY_TEST_URL")
    if existing:
        yield existing.rstrip("/")
        return
    if not docker_usable():
        pytest.skip("Docker is not available")
    if not COMPOSE.is_file():
        pytest.skip("legacy-test-api compose file missing")
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "up", "--build", "-d"],
            check=True,
            cwd=str(ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"could not start legacy-test-api: {exc}")
    url = "http://127.0.0.1:8099"
    deadline = time.time() + 180
    import httpx

    ok = False
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                ok = True
                break
        except httpx.HTTPError:
            time.sleep(2)
    if not ok:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "down"],
            cwd=str(ROOT),
        )
        pytest.skip("legacy-test-api did not become healthy")
    yield url
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "down"],
        cwd=str(ROOT),
    )


def load_golden():
    import json

    if not GOLDEN.is_file():
        return []
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_ATOL",
    "assert_equiv_results",
    "scores_close",
    "docker_usable",
    "onnx_weights_present",
    "legacy_api_url",
    "load_golden",
]
