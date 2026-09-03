"""Download / extract ONNX weights (no network; URL comes from env)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import httpx
import pytest

from xenosite.predict import WeightsDownloadError, download_weights, ensure_weights
from xenosite.predict.backends import BackendNotConfigured, resolve_backend
from xenosite.predict import weights as weights_mod
from xenosite.predict.weights import (
    ARCHIVE_NAME,
    ENV_ONNX_URL,
    GENERATION,
    default_cache_dir,
    extract_onnx_archive,
    generation_root,
    onnx_url,
    resolve_onnx_dir,
)


def _tiny_tarball(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        readme = tarfile.TarInfo("README.txt")
        data = b"test onnx archive\n"
        readme.size = len(data)
        tf.addfile(readme, io.BytesIO(data))
        onnx = tarfile.TarInfo("demo/head.onnx")
        blob = b"not-a-real-onnx"
        onnx.size = len(blob)
        tf.addfile(onnx, io.BytesIO(blob))
    return path


def test_generation_root_prefers_v0(tmp_path):
    parent = tmp_path / "onnx"
    v0 = parent / GENERATION
    (v0 / "epoxidation").mkdir(parents=True)
    (v0 / "epoxidation" / "bond.onnx").write_bytes(b"x")
    (parent / "quinone").mkdir()
    (parent / "quinone" / "atom.onnx").write_bytes(b"old")
    assert generation_root(parent) == v0
    assert generation_root(v0) == v0


def test_generation_root_flat_legacy(tmp_path):
    parent = tmp_path / "onnx"
    (parent / "epoxidation").mkdir(parents=True)
    (parent / "epoxidation" / "bond.onnx").write_bytes(b"x")
    assert generation_root(parent) == parent


def test_generation_root_empty_parent_is_v0(tmp_path):
    parent = tmp_path / "onnx"
    parent.mkdir()
    assert generation_root(parent) == parent / GENERATION


def test_onnx_url_requires_env():
    with pytest.raises(WeightsDownloadError, match=ENV_ONNX_URL):
        onnx_url(env={})


def test_extract_local_tarball(tmp_path, capsys):
    weights_mod._announced.clear()
    tar = _tiny_tarball(tmp_path / "xenosite_onnx.tgz")
    dest = tmp_path / "onnx"
    out = download_weights(url=tar, dest=dest, env={})
    assert out == dest
    assert (dest / "demo" / "head.onnx").is_file()
    assert (dest / "README.txt").is_file()
    err = capsys.readouterr().err
    assert "INFO: downloaded ONNX weights to" in err
    assert str(dest) in err


def test_extract_skips_path_traversal(tmp_path):
    tar = tmp_path / "bad.tgz"
    with tarfile.open(tar, "w:gz") as tf:
        evil = tarfile.TarInfo("../evil.onnx")
        blob = b"nope"
        evil.size = len(blob)
        tf.addfile(evil, io.BytesIO(blob))
        ok = tarfile.TarInfo("demo/head.onnx")
        ok.size = len(blob)
        tf.addfile(ok, io.BytesIO(blob))
    dest = tmp_path / "out"
    extract_onnx_archive(tar, dest)
    assert not (tmp_path / "evil.onnx").exists()
    assert (dest / "demo" / "head.onnx").is_file()


def test_download_skips_when_present(tmp_path, capsys):
    weights_mod._announced.clear()
    dest = tmp_path / "onnx"
    dest.mkdir()
    (dest / "demo").mkdir()
    (dest / "demo" / "head.onnx").write_bytes(b"existing")
    out = download_weights(url="https://example.invalid/missing.tgz", dest=dest, env={})
    assert out == dest
    assert (dest / "demo" / "head.onnx").read_bytes() == b"existing"
    err = capsys.readouterr().err
    assert "INFO: using ONNX weights at" in err
    assert str(dest) in err


def test_download_from_http_uses_env_url(tmp_path, monkeypatch, capsys):
    tar = _tiny_tarball(tmp_path / "src.tgz")
    payload = tar.read_bytes()

    class _Stream:
        def __init__(self, data: bytes):
            self._data = data
            self.status_code = 200

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _stream(self, method, url, **kwargs):
        assert method == "GET"
        assert url == "https://example.invalid/weights.tgz"
        return _Stream(payload)

    monkeypatch.setattr(httpx.Client, "stream", _stream)
    weights_mod._announced.clear()
    dest = tmp_path / "cache" / "onnx"
    env = {ENV_ONNX_URL: "https://example.invalid/weights.tgz"}
    out = download_weights(dest=dest, env=env)
    assert (out / "demo" / "head.onnx").is_file()
    assert (dest.parent / ARCHIVE_NAME).is_file()
    err = capsys.readouterr().err
    assert "INFO: downloading ONNX weights" in err
    assert "INFO: downloaded ONNX weights to" in err
    assert "example.invalid" not in err


def test_download_http_error_does_not_leak_url(tmp_path, monkeypatch):
    """httpx embeds the request URL in errors; we must not surface it."""
    secret = "https://secret.example/private/xenosite_onnx_v0.tgz"

    class _Stream:
        status_code = 403

        def raise_for_status(self):
            req = httpx.Request("GET", secret)
            resp = httpx.Response(403, request=req)
            raise httpx.HTTPStatusError(
                f"Client error '403 Forbidden' for url '{secret}'",
                request=req,
                response=resp,
            )

        def iter_bytes(self):
            yield b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(httpx.Client, "stream", lambda self, *a, **k: _Stream())
    dest = tmp_path / "onnx"
    with pytest.raises(WeightsDownloadError) as ei:
        download_weights(url=secret, dest=dest, env={}, force=True)
    msg = str(ei.value)
    assert "failed to download ONNX weights" in msg
    assert "403" in msg
    assert "secret.example" not in msg
    assert secret not in msg
    assert ei.value.__cause__ is None


def test_default_cache_dir_isolated_env(tmp_path):
    assert default_cache_dir(env={}) is None
    d = default_cache_dir(env={"XDG_CACHE_HOME": str(tmp_path / "xdg")})
    assert d == tmp_path / "xdg" / "xenosite" / "onnx" / GENERATION


def test_picker_uses_cache_dir(tmp_path):
    cache = tmp_path / "xdg" / "xenosite" / "onnx" / GENERATION
    cache.mkdir(parents=True)
    (cache / "epoxidation").mkdir()
    (cache / "epoxidation" / "bond.onnx").write_bytes(b"x")
    be = resolve_backend(
        env={"XDG_CACHE_HOME": str(tmp_path / "xdg")},
        cwd=tmp_path,
    )
    assert be.name == "onnx"
    assert ("epoxidation", "0") in be.available_models()


def test_picker_does_not_download_with_empty_env(tmp_path):
    with pytest.raises(BackendNotConfigured):
        resolve_backend(env={}, cwd=tmp_path)


def test_ensure_weights_from_env_url(tmp_path):
    tar = _tiny_tarball(tmp_path / "local.tgz")
    dest = tmp_path / "onnx"
    out = ensure_weights(
        dest=dest,
        env={ENV_ONNX_URL: str(tar)},
    )
    assert (out / "demo" / "head.onnx").is_file()


def test_resolve_onnx_dir_auto_download(tmp_path, monkeypatch):
    tar = _tiny_tarball(tmp_path / "src.tgz")
    dest = None

    def _dl(**kwargs):
        nonlocal dest
        dest = kwargs["dest"]
        dest.mkdir(parents=True)
        (dest / "demo").mkdir()
        (dest / "demo" / "head.onnx").write_bytes(b"x")
        return dest

    monkeypatch.setattr("xenosite.predict.weights.download_weights", _dl)
    found = resolve_onnx_dir(
        env={
            ENV_ONNX_URL: "https://example.invalid/weights.tgz",
            "XENOSITE_AUTO_DOWNLOAD": "1",
            "XDG_CACHE_HOME": str(tmp_path / "xdg"),
        },
        cwd=tmp_path,
    )
    assert found is not None
    assert (found / "demo" / "head.onnx").is_file()
