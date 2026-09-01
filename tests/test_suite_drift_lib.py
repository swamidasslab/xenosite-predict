import numpy as np

from tools.suite_drift_lib import (
    classify_alignment,
    index_free_max_diff,
    marriage_assignment_max_diff,
)


def test_index_free_max_diff_detects_reordering():
    want = np.array([0.9, 0.1])
    have = np.array([0.0, 1.0])
    assert index_free_max_diff(want, have) == 0.1
    assert float(np.max(np.abs(want - have))) == 0.9


def test_marriage_assignment_max_diff_same_as_sorted_for_permutation():
    rng = np.random.default_rng(0)
    base = rng.random(12)
    perm = rng.permutation(12)
    shuffled = base[perm]
    assert marriage_assignment_max_diff(base, shuffled) == 0.0


def test_classify_alignment():
    assert classify_alignment(0.5, 0.01, 0.02) == "ordering_only"
    assert classify_alignment(0.5, 0.5, 0.02) == "real_drift"
    assert classify_alignment(0.01, 0.01, 0.02) is None
