"""Hypothesis strategies for random-vector NN tests."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Logistic / windowed nets stay finite in this band; keep examples cheap.
_ELEM = st.floats(
    min_value=-8.0,
    max_value=8.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=32,
)


def feature_matrix(n_features: int, *, min_rows: int = 1, max_rows: int = 4):
    """Draw a finite float32 matrix ``(n_patterns, n_features)``."""
    return arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(min_rows, max_rows), st.just(int(n_features))),
        elements=_ELEM,
        fill=_ELEM,
        unique=False,
    )
