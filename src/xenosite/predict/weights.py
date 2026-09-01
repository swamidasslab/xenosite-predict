"""Locate, download, and extract ONNX inference graphs.

The published archive URL is **not** compiled into this package. Set
``XENOSITE_ONNX_URL`` (https(s) or a local ``.tgz`` path). First ``predict()``
fetches v0 graphs into the user cache when the URL is set and no local
``*.onnx`` files are found. Import does not download or open ONNX.

v0 lives under ``weights/onnx/v0`` (checkout) and
``$XDG_CACHE_HOME/xenosite/onnx/v0`` (cache). A later v1 generation uses a
sibling directory and its own tarball.
"""

from __future__ import annotations

import logging
import os
import sys
import tarfile
from pathlib import Path
from typing import Mapping, Optional, Union
from urllib.parse import urlparse

import httpx

from .errors import WeightsDownloadError, WeightsNotFound

ENV_WEIGHTS = "XENOSITE_MODELS_WEIGHTS"
ENV_ONNX_URL = "XENOSITE_ONNX_URL"
ENV_AUTO_DOWNLOAD = "XENOSITE_AUTO_DOWNLOAD"

GENERATION = "v0"
WEIGHTS_ONNX_PARENT = Path("weights/onnx")
DEFAULT_ONNX_DIR = WEIGHTS_ONNX_PARENT / GENERATION
ARCHIVE_NAME = f"xenosite_onnx_{GENERATION}.tgz"
_USER_AGENT = "xenosite-predict"

logger = logging.getLogger("xenosite.predict")
logger.addHandler(logging.NullHandler())

_announced: set[str] = set()

Source = Union[str, Path]


def onnx_dir_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.rglob("*.onnx"))


def _model_onnx_dirs(path: Path) -> bool:
    """True if ``path/<model>/*.onnx`` exists (generation leaf or flat tree)."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir() or child.name.startswith(".") or child.name == "_dump":
            continue
        if any(child.glob("*.onnx")):
            return True
    return False


def generation_root(path: Path, generation: str = GENERATION) -> Path:
    """Resolve a generation leaf: ``…/v0``, a parent containing ``v0/``, or a flat tree."""
    if path.name == generation:
        return path
    nested = path / generation
    if _model_onnx_dirs(nested):
        return nested
    if _model_onnx_dirs(path):
        return path
    return nested


def default_cache_dir(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """User-level ONNX cache. Isolated ``env`` dicts do not consult ``$HOME``."""
    e = os.environ if env is None else env
    xdg = (e.get("XDG_CACHE_HOME") or "").strip()
    if xdg:
        return Path(xdg) / "xenosite" / "onnx" / GENERATION
    if env is None:
        return Path.home() / ".cache" / "xenosite" / "onnx" / GENERATION
    home = (e.get("HOME") or "").strip()
    if home:
        return Path(home) / ".cache" / "xenosite" / "onnx" / GENERATION
    return None


def onnx_url(env: Optional[Mapping[str, str]] = None) -> str:
    """Return ``XENOSITE_ONNX_URL``, or raise if unset."""
    e = os.environ if env is None else env
    url = (e.get(ENV_ONNX_URL) or "").strip()
    if not url:
        raise WeightsDownloadError(
            f"{ENV_ONNX_URL} is not set. Point it at an ONNX weight tarball "
            "(https URL or local .tgz path)."
        )
    return url


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def auto_download_enabled(
    env: Optional[Mapping[str, str]] = None,
    *,
    auto_download: Optional[bool] = None,
) -> bool:
    """Download only when a URL is configured.

    ``env=None`` (real process env) defaults on if ``XENOSITE_ONNX_URL`` is set,
    unless ``XENOSITE_AUTO_DOWNLOAD=0``. An explicit ``env`` mapping (tests)
    defaults off unless ``XENOSITE_AUTO_DOWNLOAD`` is truthy.
    """
    e = os.environ if env is None else env
    if not (e.get(ENV_ONNX_URL) or "").strip():
        return False
    if auto_download is not None:
        return auto_download
    if ENV_AUTO_DOWNLOAD in e:
        return _truthy(e.get(ENV_AUTO_DOWNLOAD))
    return env is None


def _info(message: str) -> None:
    logger.info(message)
    print(f"INFO: {message}", file=sys.stderr)


def _announce(path: Path, *, downloaded: bool) -> None:
    key = str(path)
    if key in _announced:
        return
    _announced.add(key)
    if downloaded:
        _info(f"downloaded ONNX weights to {path}")
    else:
        _info(f"using ONNX weights at {path}")


def _safe_member(member: tarfile.TarInfo) -> bool:
    name = Path(member.name)
    if name.is_absolute() or ".." in name.parts:
        return False
    if not member.isfile():
        return False
    return (
        member.name == "README.txt"
        or name.suffix == ".onnx"
        or name.name.endswith(".meta.json")
    )


def extract_onnx_archive(tarball: Path, dest: Path) -> int:
    """Unpack a packed ONNX tarball into ``dest``. Returns files extracted."""
    if not tarball.is_file():
        raise WeightsDownloadError(f"tarball not found: {tarball}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        members = [m for m in tf.getmembers() if _safe_member(m)]
        kwargs: dict = {}
        if sys.version_info >= (3, 12):
            kwargs["filter"] = "data"
        tf.extractall(dest, members=members, **kwargs)
    return len(members)


def _local_source(source: Source) -> Optional[Path]:
    if isinstance(source, Path):
        return source
    parsed = urlparse(source)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path if parsed.scheme == "file" else source)
        if path.is_file():
            return path
    return None


def _http_get(url: str, dest: Path, *, timeout: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"user-agent": _USER_AGENT},
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with partial.open("wb") as fh:
                    for chunk in response.iter_bytes():
                        fh.write(chunk)
        partial.replace(dest)
    except httpx.HTTPError as exc:
        if partial.exists():
            partial.unlink()
        raise WeightsDownloadError("failed to download ONNX weights") from exc
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def download_weights(
    *,
    url: Optional[Source] = None,
    dest: Optional[os.PathLike[str] | str] = None,
    force: bool = False,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 120.0,
) -> Path:
    """Download (if needed) and extract ONNX graphs. Returns the dest directory.

    ``url`` defaults to ``XENOSITE_ONNX_URL``. ``dest`` defaults to
    ``XENOSITE_MODELS_WEIGHTS`` or the user cache directory.

    ``predict()`` calls this automatically; a manual call is optional.
    """
    e = os.environ if env is None else env
    source: Source = url if url is not None else onnx_url(env)
    if dest is not None:
        dest_path = Path(dest)
    else:
        configured = (e.get(ENV_WEIGHTS) or "").strip()
        dest_path = (
            generation_root(Path(configured)) if configured else default_cache_dir(env)
        )
        if dest_path is None:
            raise WeightsDownloadError(
                f"No destination: set {ENV_WEIGHTS} or XDG_CACHE_HOME / HOME."
            )

    if not force and onnx_dir_has_files(dest_path):
        _announce(dest_path, downloaded=False)
        return dest_path

    local = _local_source(source)
    archive = dest_path.parent / ARCHIVE_NAME
    if local is not None:
        archive = local
    elif force or not archive.is_file():
        if not (isinstance(source, str) and source.startswith(("http://", "https://"))):
            raise WeightsDownloadError(
                "Not an http(s) URL or local tarball."
            )
        _info("downloading ONNX weights")
        _http_get(source, archive, timeout=timeout)

    extract_onnx_archive(archive, dest_path)
    if not onnx_dir_has_files(dest_path):
        raise WeightsNotFound(
            f"Extracted archive into {dest_path} but found no *.onnx files."
        )
    _announce(dest_path, downloaded=True)
    return dest_path


def ensure_weights(
    *,
    url: Optional[Source] = None,
    dest: Optional[os.PathLike[str] | str] = None,
    force: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return a directory that contains ``*.onnx`` files, downloading if needed."""
    e = os.environ if env is None else env
    if dest is not None:
        dest_path = Path(dest)
    else:
        configured = (e.get(ENV_WEIGHTS) or "").strip()
        dest_path = (
            generation_root(Path(configured)) if configured else default_cache_dir(env)
        )
        if dest_path is None:
            raise WeightsDownloadError(
                f"No destination: set {ENV_WEIGHTS} or XDG_CACHE_HOME / HOME."
            )
    if not force and onnx_dir_has_files(dest_path):
        _announce(dest_path, downloaded=False)
        return dest_path
    return download_weights(url=url, dest=dest_path, force=force, env=env)


def resolve_onnx_dir(
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    auto_download: Optional[bool] = None,
    fallback: bool = False,
) -> Optional[Path]:
    """Pick an ONNX directory: env, ``./weights/onnx/v0``, user cache, then download."""
    e = os.environ if env is None else env
    want = auto_download_enabled(env, auto_download=auto_download)

    configured = (e.get(ENV_WEIGHTS) or "").strip()
    if configured:
        path = generation_root(Path(configured))
        if onnx_dir_has_files(path):
            _announce(path, downloaded=False)
            return path
        if want:
            return ensure_weights(env=env, dest=path)
        return path

    cwd = cwd or Path.cwd()
    local = generation_root(cwd / WEIGHTS_ONNX_PARENT)
    if onnx_dir_has_files(local):
        _announce(local, downloaded=False)
        return local

    cache = default_cache_dir(env)
    if cache is not None and onnx_dir_has_files(cache):
        _announce(cache, downloaded=False)
        return cache

    if want:
        return ensure_weights(env=env, dest=cache)

    if fallback:
        return local
    return None
