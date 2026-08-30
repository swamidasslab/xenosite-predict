"""OpenBabel-like periodic-table constants used by topological descriptors.

Values follow OpenBabel's default element table for the elements that appear
in drug-like molecules. Live RDKit-vs-OpenBabel tests will flag drift.
"""

from __future__ import annotations

# atomic number -> (symbol, eneg, eaffinity, maxbonds, ionization, mass, bond_rad, vdw)
# Bond/vdW radii are the uncorrected OB values; hybridization correction is small.
_PT: dict[int, tuple[str, float, float, int, float, float, float, float]] = {
    1: ("H", 2.20, 0.754, 1, 13.598, 1.00794, 0.23, 1.20),
    6: ("C", 2.55, 1.263, 4, 11.260, 12.0107, 0.68, 1.70),
    7: ("N", 3.04, -0.07, 4, 14.534, 14.0067, 0.68, 1.55),
    8: ("O", 3.44, 1.461, 2, 13.618, 15.9994, 0.68, 1.52),
    9: ("F", 3.98, 3.401, 1, 17.423, 18.9984, 0.64, 1.47),
    15: ("P", 2.19, 0.747, 5, 10.487, 30.9738, 1.05, 1.80),
    16: ("S", 2.58, 2.077, 6, 10.360, 32.065, 1.02, 1.80),
    17: ("Cl", 3.16, 3.613, 1, 12.968, 35.453, 0.99, 1.75),
    35: ("Br", 2.96, 3.364, 1, 11.814, 79.904, 1.21, 1.85),
    53: ("I", 2.66, 3.059, 1, 10.451, 126.904, 1.40, 1.98),
}

# OpenBabel NOuterElecs table (copied from epoxidation topological_descriptors)
NOUTER: dict[int, int] = {
    1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 1, 20: 2,
    21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 30: 2,
    31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 1, 38: 2, 39: 3, 40: 4,
    41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10, 47: 11, 48: 2, 49: 3, 50: 4,
    51: 5, 52: 6, 53: 7, 54: 8,
}


def symbol(z: int) -> str:
    if z in _PT:
        return _PT[z][0]
    from rdkit.Chem import GetPeriodicTable

    return GetPeriodicTable().GetElementSymbol(z)


def electronegativity(z: int) -> float:
    return _PT[z][1] if z in _PT else 0.0


def electron_affinity(z: int) -> float:
    return _PT[z][2] if z in _PT else 0.0


def max_bonds(z: int) -> int:
    return _PT[z][3] if z in _PT else 4


def ionization(z: int) -> float:
    return _PT[z][4] if z in _PT else 0.0


def mass(z: int) -> float:
    return _PT[z][5] if z in _PT else 0.0


def corrected_bond_rad(z: int, hyb: int) -> float:
    base = _PT[z][6] if z in _PT else 0.68
    # OpenBabel CorrectedBondRad shrinks slightly for higher hybridization
    return base - 0.03 * max(hyb - 1, 0)


def corrected_vdw_rad(z: int, hyb: int) -> float:
    base = _PT[z][7] if z in _PT else 1.70
    return base


def lone_pairs(atomic_num: int, total_bond_order: int, formal_charge: int) -> int:
    v = NOUTER.get(atomic_num, 0)
    return v - total_bond_order - formal_charge
