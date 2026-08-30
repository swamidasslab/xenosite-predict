"""Float-compare helper and golden-fixture loader."""

import numpy as np
import pytest

from xenosite.predict.compare import assert_equiv_results, scores_close
from tests.support import load_golden


def test_atol_1e4():
    assert scores_close(0.12345, 0.12349)
    assert not scores_close(0.12345, 0.2)


def test_pair_idx_order_independent():
    a = {"pair_idx": [(1, 0), (2, 3)], "pair": [0.2, 0.1]}
    b = {"pair_idx": [(3, 2), (0, 1)], "pair": [0.1, 0.2]}
    assert_equiv_results(a, b)


def test_golden_loader():
    data = load_golden()
    assert isinstance(data, list)
    assert data, "missing tests/fixtures/golden_smiles.json"
