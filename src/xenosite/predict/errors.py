"""User-facing exceptions."""


class XenositePredictError(Exception):
    """Base class for ``xenosite.predict`` errors."""


class InvalidMolecule(XenositePredictError, ValueError):
    """SMILES could not be parsed or the molecule is too small."""


class UnknownModel(XenositePredictError, KeyError):
    """Requested ``(name, version)`` is not in the registry."""


class BackendNotConfigured(XenositePredictError, RuntimeError):
    """No backend could be resolved from env, weights, or an explicit argument."""


class WeightsNotFound(XenositePredictError, FileNotFoundError):
    """ONNX (or other local) weights are missing for a model head."""


class ModelNotAvailable(XenositePredictError, RuntimeError):
    """The model is registered but blocked (MOPAC, missing convert, pipeline)."""
