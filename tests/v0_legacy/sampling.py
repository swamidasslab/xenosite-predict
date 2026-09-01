"""Hypothesis sampling for large fixed corpora (ob dumps, golden suite).

Failing draws are persisted under ``.hypothesis/examples/`` and replayed on
every subsequent run without manual ``@example`` pinning.

Default ``max_examples`` equals the number of unique SMILES in the sampled list
(one draw per dump molecule; model is sampled separately). Override with
``XENOSITE_HYPOTHESIS_MAX_EXAMPLES``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import lru_cache

from hypothesis import settings
from hypothesis import strategies as st

SmilesFn = Callable[[], tuple[str, ...]]


def sample_max_examples(smiles: Sequence[str]) -> int:
    """One example per SMILES in the dump-derived list by default."""
    override = os.environ.get("XENOSITE_HYPOTHESIS_MAX_EXAMPLES")
    n = len(smiles)
    if n == 0:
        return 1
    if override is not None:
        return max(1, min(int(override), n))
    return n


def molecule_sample_settings(
    smiles: tuple[str, ...] | SmilesFn,
    **overrides,
) -> settings:
    resolved = smiles() if callable(smiles) else smiles
    if "max_examples" not in overrides:
        overrides["max_examples"] = sample_max_examples(resolved)
    return settings(deadline=None, **overrides)


@lru_cache(maxsize=1)
def _ob_dump_model_sets() -> dict[str, frozenset[str]]:
    from tests.support import load_ob_dumps

    out: dict[str, frozenset[str]] = {}
    for mol in load_ob_dumps():
        smi = mol.get("smiles") or ""
        if smi:
            out[smi] = frozenset(mol.get("models") or {})
    return out


@lru_cache(maxsize=8)
def ob_dump_smiles(models: tuple[str, ...]) -> tuple[str, ...]:
    """Unique SMILES from ob dumps that have at least one of ``models``."""
    model_sets = _ob_dump_model_sets()
    out: list[str] = []
    for smi, present in model_sets.items():
        if any(m in present for m in models):
            out.append(smi)
    return tuple(out)


def ob_dump_models_for_smiles(smiles: str, models: tuple[str, ...]) -> tuple[str, ...]:
    present = _ob_dump_model_sets().get(smiles, frozenset())
    return tuple(m for m in models if m in present)


@st.composite
def ob_dump_smiles_model(draw, models: tuple[str, ...]):
    corpus = ob_dump_smiles(models)
    if not corpus:
        raise RuntimeError(f"empty ob_dump SMILES for models={models}")
    smiles = draw(st.sampled_from(corpus))
    available = ob_dump_models_for_smiles(smiles, models)
    if not available:
        raise RuntimeError(f"no models in dump for {smiles!r}")
    model = draw(st.sampled_from(available))
    return smiles, model


@lru_cache(maxsize=8)
def ob_dump_pairs(models: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """All ``(smiles, model)`` pairs (for exhaustive ``@pytest.mark.full``)."""
    pairs: list[tuple[str, str]] = []
    for smi in ob_dump_smiles(models):
        for model in ob_dump_models_for_smiles(smi, models):
            pairs.append((smi, model))
    return tuple(pairs)


_SCORE_MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "isozyme")


@lru_cache(maxsize=1)
def equiv_smiles() -> tuple[str, ...]:
    from tests.support import onnx_weights_present

    out: list[str] = []
    for smi in ob_dump_smiles(_SCORE_MODELS):
        if equiv_models_for_smiles(smi):
            out.append(smi)
    return tuple(out)


def equiv_models_for_smiles(smiles: str) -> tuple[str, ...]:
    from tests.support import onnx_weights_present

    present = _ob_dump_model_sets().get(smiles, frozenset())
    out: list[str] = []
    for model in _SCORE_MODELS:
        if model not in present:
            continue
        weight_key = "ndealk" if model == "isozyme" else model
        if onnx_weights_present(weight_key):
            out.append(model)
    return tuple(out)


@st.composite
def equiv_smiles_model(draw):
    corpus = equiv_smiles()
    if not corpus:
        raise RuntimeError("empty equiv SMILES (ob dumps or ONNX weights missing)")
    smiles = draw(st.sampled_from(corpus))
    available = equiv_models_for_smiles(smiles)
    model = draw(st.sampled_from(available))
    return smiles, model


@lru_cache(maxsize=1)
def equiv_pairs() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for smi in equiv_smiles():
        for model in equiv_models_for_smiles(smi):
            pairs.append((smi, model))
    return tuple(pairs)


@st.composite
def principled_descriptor_smiles_model(draw, models: tuple[str, ...]):
    corpus = ob_dump_smiles(models)
    if not corpus:
        raise RuntimeError(f"empty principled descriptor SMILES for models={models}")
    smiles = draw(st.sampled_from(corpus))
    available = ob_dump_models_for_smiles(smiles, models)
    model = draw(st.sampled_from(available))
    return smiles, model


@lru_cache(maxsize=4)
def principled_descriptor_pairs(models: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for smi in ob_dump_smiles(models):
        for model in ob_dump_models_for_smiles(smi, models):
            pairs.append((smi, model))
    return tuple(pairs)


@lru_cache(maxsize=1)
def _golden_suite_rows() -> tuple[tuple[str, str], ...]:
    from tests.support import load_golden_suite, onnx_weights_present

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for g in load_golden_suite(merge_smoke=True):
        model = g.get("model") or ""
        smiles = g.get("smiles") or ""
        if model == "bioactivation" or not model or not smiles:
            continue
        key = (model, smiles)
        if key in seen:
            continue
        seen.add(key)
        weight_key = "ndealk" if model == "isozyme" else model
        if not onnx_weights_present(weight_key):
            continue
        pairs.append(key)
    return tuple(pairs)


@lru_cache(maxsize=1)
def golden_suite_smiles() -> tuple[str, ...]:
    return tuple(dict.fromkeys(smiles for _, smiles in _golden_suite_rows()))


def golden_suite_models_for_smiles(smiles: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model for model, smi in _golden_suite_rows() if smi == smiles))


@st.composite
def golden_suite_smiles_model(draw):
    corpus = golden_suite_smiles()
    if not corpus:
        raise RuntimeError("empty golden suite SMILES")
    smiles = draw(st.sampled_from(corpus))
    available = golden_suite_models_for_smiles(smiles)
    model = draw(st.sampled_from(available))
    return model, smiles


@lru_cache(maxsize=1)
def golden_suite_pairs() -> tuple[tuple[str, str], ...]:
    return _golden_suite_rows()


@lru_cache(maxsize=1)
def _golden_parity_rows() -> tuple[tuple[str, str], ...]:
    from tests.support import load_golden, onnx_model_key, onnx_weights_present

    pairs: list[tuple[str, str]] = []
    for g in load_golden():
        model = g.get("model") or ""
        smiles = g.get("smiles") or ""
        if model in ("bioactivation",) or not model or not smiles:
            continue
        if not onnx_weights_present(onnx_model_key(model)):
            continue
        pairs.append((model, smiles))
    return tuple(pairs)


@lru_cache(maxsize=1)
def golden_parity_smiles() -> tuple[str, ...]:
    return tuple(dict.fromkeys(smiles for _, smiles in _golden_parity_rows()))


def golden_parity_models_for_smiles(smiles: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model for model, smi in _golden_parity_rows() if smi == smiles))


@st.composite
def golden_parity_smiles_model(draw):
    corpus = golden_parity_smiles()
    if not corpus:
        raise RuntimeError("empty golden parity SMILES")
    smiles = draw(st.sampled_from(corpus))
    available = golden_parity_models_for_smiles(smiles)
    model = draw(st.sampled_from(available))
    return model, smiles


@lru_cache(maxsize=1)
def golden_parity_pairs() -> tuple[tuple[str, str], ...]:
    return _golden_parity_rows()
