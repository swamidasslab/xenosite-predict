"""Local ONNX backend. Sessions are opened on first use of a head, not at import."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..errors import WeightsNotFound
from ..scoring import SCORING_VERSIONS

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

# Cap ORT intra-op threads when many process workers share a host.
ENV_ORT_INTRA = "XENOSITE_ORT_INTRA_OP"
# Optional ORT profile prefix. Must be a real file path (parent dir exists).
ENV_ORT_PROFILE = "XENOSITE_ORT_PROFILE"

_DEFAULT_PROFILE_PREFIX = "ort_profile"
_PROFILE_FLAGS = frozenset({"1", "0", "true", "false", "yes", "no", "on", "off"})


def _ort_profile_prefix(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return ``XENOSITE_ORT_PROFILE`` when it names a real file path.

    A real path has a file name and an existing parent directory. Flag values
    (``1``, ``true``, …) do not enable profiling.
    """
    e = os.environ if env is None else env
    raw = (e.get(ENV_ORT_PROFILE) or "").strip()
    if not raw or raw.lower() in _PROFILE_FLAGS:
        return None
    path = Path(raw).expanduser()
    if path.name in {"", ".", ".."}:
        return None
    if path.exists() and path.is_dir():
        return None
    if not path.parent.is_dir():
        return None
    return str(path)


def _session_options(env: Optional[Mapping[str, str]] = None) -> Any:
    if ort is None:
        return None
    e = os.environ if env is None else env
    opts = ort.SessionOptions()
    raw = (e.get(ENV_ORT_INTRA) or "").strip()
    if raw:
        opts.intra_op_num_threads = max(1, int(raw))
        opts.inter_op_num_threads = 1
    # Profiling dumps ``*.sess`` / JSON traces; mem-pattern persistence can
    # write ``:mem:.sess`` when the prefix is left at an internal tag.
    opts.enable_mem_pattern = False
    prefix = _ort_profile_prefix(e)
    if prefix:
        opts.enable_profiling = True
        opts.profile_file_prefix = prefix
    else:
        opts.enable_profiling = False
        opts.profile_file_prefix = _DEFAULT_PROFILE_PREFIX
    return opts


class OnnxBackend:
    """Run converted numpy-NN heads from ``weights/onnx/v0/<model>/<head>.onnx``."""

    name = "onnx"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._sessions: dict[tuple[str, str], Any] = {}

    def path_for(self, model: str, head: str) -> Path:
        return self.root / model / f"{head}.onnx"

    def has_head(self, model: str, head: str) -> bool:
        return self.path_for(model, head).is_file()

    def available_models(self) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        if not self.root.is_dir():
            return found
        for model_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if any(model_dir.glob("*.onnx")):
                for version in SCORING_VERSIONS:
                    found.append((model_dir.name, version))
        names = {n for n, _ in found}
        if "ndealk" in names and "isozyme" not in names:
            for version in SCORING_VERSIONS:
                found.append(("isozyme", version))
        return found

    def session(self, model: str, head: str):
        if ort is None:
            raise WeightsNotFound("onnxruntime is not installed")
        key = (model, head)
        if key not in self._sessions:
            path = self.path_for(model, head)
            if not path.is_file():
                raise WeightsNotFound(
                    f"Missing ONNX weights {path}. Set XENOSITE_ONNX_URL "
                    "(auto-downloaded on first predict()) or "
                    "XENOSITE_MODELS_WEIGHTS to a directory of *.onnx files."
                )
            kwargs: dict[str, Any] = {"providers": ["CPUExecutionProvider"]}
            opts = _session_options()
            if opts is not None:
                kwargs["sess_options"] = opts
            self._sessions[key] = ort.InferenceSession(str(path), **kwargs)
        return self._sessions[key]

    def run_head(self, model: str, head: str, x: np.ndarray) -> np.ndarray:
        """``x`` is ``(n_features, n_patterns)`` like the legacy NN, or ``(n, f)``.

        ONNX graphs in this package take ``X`` as float32 ``(n_patterns, n_features)``.
        """
        sess = self.session(model, head)
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected 2-D feature matrix, got {arr.shape}")
        # Accept either orientation; ONNX input is (N, F)
        inp = sess.get_inputs()[0]
        want = inp.shape  # [None, F] or similar
        f_dim = want[1] if isinstance(want[1], int) else None
        if f_dim is not None:
            if arr.shape[1] == f_dim:
                feed = arr
            elif arr.shape[0] == f_dim:
                feed = arr.T
            else:
                raise ValueError(
                    f"feature dim mismatch: got {arr.shape}, ONNX wants {want}"
                )
        else:
            feed = arr if arr.shape[0] <= arr.shape[1] else arr.T
        name = inp.name
        out = sess.run(None, {name: feed})[0]
        return np.asarray(out)

    def predict_native(self, smiles: str, model: str, version: str) -> Any:
        """Feature + two-stage logic lives in the model runner, not here."""
        raise NotImplementedError(
            "OnnxBackend.predict_native is unused; model runners call run_head()"
        )
