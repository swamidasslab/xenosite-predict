"""Local ONNX backend. Sessions are opened on first use of a head, not at import."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..errors import BackendNotConfigured, WeightsNotFound

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]


class OnnxBackend:
    """Run converted numpy-NN heads from ``weights/onnx/<model>/<head>.onnx``."""

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
                found.append((model_dir.name, "0"))
        names = {n for n, _ in found}
        if "ndealk" in names and "isozyme" not in names:
            found.append(("isozyme", "0"))
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
            self._sessions[key] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
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
