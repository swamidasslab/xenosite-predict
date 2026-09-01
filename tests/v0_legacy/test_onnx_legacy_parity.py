"""ONNX vs legacy prediction parity, tiered by descriptor agreement.

Legacy reference scores come from ``tests/fixtures/golden_smiles.json`` (captured
from the production legacy frontend). Tier 1 is a small smoke set where internal
OpenBabel matches the py2 dump oracle; tier 2 is quinone OMP-path drift; tier 3
is every golden row with ONNX weights (bioactivation excluded).

Optional ``@pytest.mark.live`` tests hit ``legacy-test-api`` when
``XENOSITE_LEGACY_TEST_URL`` or the session Docker fixture is up (needs TF weights).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from xenosite.predict import predict
from xenosite.predict.backends.legacy import LegacyTestBackend
from xenosite.predict.backends.onnx import OnnxBackend

from tests.v0_legacy.sampling import (
    golden_parity_smiles,
    golden_parity_smiles_model,
    molecule_sample_settings,
)
from tests.support import (
    ROOT,
    assert_golden_molecule,
    assert_predictions_parity,
    descriptor_omp_only,
    descriptor_passes,
    golden_name_by_smiles,
    golden_predict_kwargs,
    load_golden,
    load_ob_dumps,
    onnx_model_key,
    onnx_weights_present,
    onnx_root,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

SMOKE_DESCRIPTOR_PASSING: tuple[tuple[str, str], ...] = (
    ("epoxidation", ASPIRIN),
    ("quinone", ASPIRIN),
    ("reactivity", ASPIRIN),
    ("ugt", ASPIRIN),
    ("ndealk", ASPIRIN),
    ("phase1", ASPIRIN),
    ("epoxidation", "CC(=Cc1ccc(CO)cc1)c1ccc2c(c1)C(C)(C)C(O)CC2(C)C"),
    ("quinone", "CC(=Cc1ccc(CO)cc1)c1ccc2c(c1)C(C)(C)C(O)CC2(C)C"),
    ("ugt", "COCCc1ccc(OCC(O)CNC(C)C)cc1"),
    ("ndealk", "COCCc1ccc(OCC(O)CNC(C)C)cc1"),
)

_OMP_FAILING_CACHE: list | None = None
_GOLDEN_PARITY_CACHE: list | None = None


def _golden_row(model: str, smiles: str) -> dict:
    rows = [
        g
        for g in load_golden()
        if g.get("model") == model and g.get("smiles") == smiles
    ]
    assert rows, f"no golden row for {model} {smiles}"
    return rows[0]


def _omp_failing_sample():
    global _OMP_FAILING_CACHE
    if _OMP_FAILING_CACHE is not None:
        return _OMP_FAILING_CACHE
    names = golden_name_by_smiles()
    out = []
    seen: set[str] = set()
    for mol in load_ob_dumps():
        smi = mol.get("smiles") or ""
        if smi in seen or "quinone" not in (mol.get("models") or {}):
            continue
        if not descriptor_omp_only(smi, "quinone"):
            continue
        seen.add(smi)
        label = names.get(smi) or smi[:24]
        out.append(pytest.param("quinone", smi, id=f"quinone:{label}"))
        if len(out) >= 8:
            break
    _OMP_FAILING_CACHE = out
    return out



def _golden_parity_params():
    global _GOLDEN_PARITY_CACHE
    if _GOLDEN_PARITY_CACHE is not None:
        return _GOLDEN_PARITY_CACHE
    params = []
    for g in load_golden():
        model = g.get("model") or ""
        smiles = g.get("smiles") or ""
        if model in ("bioactivation",):
            continue
        if not onnx_weights_present(onnx_model_key(model)):
            continue
        label = g.get("name") or smiles[:24]
        params.append(pytest.param(model, smiles, id=f"{model}:{label}"))
    _GOLDEN_PARITY_CACHE = params
    return params


def _onnx_predict(model: str, smiles: str):
    key = onnx_model_key(model)
    if not onnx_weights_present(key):
        pytest.skip(f"no ONNX for {key}")
    return predict(
        smiles,
        models=[model],
        backend=OnnxBackend(onnx_root()),
        **golden_predict_kwargs(model),
    )


@pytest.mark.parametrize("model,smiles", SMOKE_DESCRIPTOR_PASSING)
def test_onnx_matches_legacy_golden_descriptor_passing_smoke(model, smiles):
    assert descriptor_passes(smiles, model), (
        f"{model} {smiles} no longer passes descriptor dump (refresh smoke set)"
    )
    got = _onnx_predict(model, smiles)
    assert got.results
    assert_golden_molecule(got, _golden_row(model, smiles), smiles=smiles, model=model)


@pytest.mark.parametrize("model,smiles", _omp_failing_sample())
def test_onnx_runs_descriptor_non_passing(model, smiles):
    """OMP-only descriptor drift: ONNX must run; parity vs legacy is live-only."""
    assert descriptor_omp_only(smiles, model)
    got = _onnx_predict(model, smiles)
    assert got.results


def _assert_onnx_matches_legacy_golden(model: str, smiles: str) -> None:
    got = _onnx_predict(model, smiles)
    assert got.results
    assert_golden_molecule(got, _golden_row(model, smiles), smiles=smiles, model=model)


def test_golden_parity_smiles_nonempty():
    assert len(golden_parity_smiles()) >= 10


@molecule_sample_settings(golden_parity_smiles)
@given(data=st.data())
def test_onnx_matches_legacy_golden_sampled(data):
    model, smiles = data.draw(golden_parity_smiles_model())
    _assert_onnx_matches_legacy_golden(model, smiles)


@pytest.mark.full
@pytest.mark.parametrize("model,smiles", _golden_parity_params())
def test_onnx_matches_legacy_golden_full(model, smiles):
    _assert_onnx_matches_legacy_golden(model, smiles)


def _predict_pair_live(smiles: str, model: str, legacy_url: str):
    key = onnx_model_key(model)
    if not onnx_weights_present(key):
        pytest.skip(f"no ONNX for {key}")
    legacy = LegacyTestBackend(legacy_url)
    onnx = OnnxBackend(onnx_root())
    kwargs = golden_predict_kwargs(model)
    try:
        leg = predict(smiles, models=[model], backend=legacy)
    except Exception as exc:
        pytest.skip(f"legacy predict unavailable for {model} {smiles}: {exc}")
    try:
        got = predict(
            smiles,
            models=[model],
            backend=onnx,
            **kwargs,
        )
    except Exception as exc:
        pytest.fail(f"onnx predict failed for {model} {smiles}: {exc}")
    assert leg.results and got.results
    return leg, got


@pytest.mark.live
@pytest.mark.parametrize("model,smiles", SMOKE_DESCRIPTOR_PASSING)
def test_onnx_legacy_api_parity_descriptor_passing_smoke(legacy_api_url, model, smiles):
    assert descriptor_passes(smiles, model)
    leg, got = _predict_pair_live(smiles, model, legacy_api_url)
    assert_predictions_parity(leg, got, smiles=smiles, model=model)


@pytest.mark.live
@pytest.mark.parametrize("model,smiles", _omp_failing_sample())
def test_onnx_legacy_api_parity_descriptor_non_passing(legacy_api_url, model, smiles):
    """OMP ortho/meta/para: legacy API uses py2 hash BFS; ONNX uses legacy OMP mode."""
    assert descriptor_omp_only(smiles, model)
    leg, got = _predict_pair_live(smiles, model, legacy_api_url)
    assert_predictions_parity(leg, got, smiles=smiles, model=model)


@pytest.mark.live
@pytest.mark.full
@pytest.mark.parametrize("model,smiles", _golden_parity_params())
def test_onnx_legacy_api_parity_golden_full(legacy_api_url, model, smiles):
    leg, got = _predict_pair_live(smiles, model, legacy_api_url)
    assert_predictions_parity(leg, got, smiles=smiles, model=model)
