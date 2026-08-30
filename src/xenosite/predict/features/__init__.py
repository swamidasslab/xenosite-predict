"""Internal OpenBabel descriptors. Public callers pass RDKit mols and see 0-based indices.

Importing this package does not load OpenBabel; bindings load on first feature call.
"""

from .atom import quinone_atom_rows, reactivity_atom_rows
from .bond import bond_rows, ndealk_bond_rows
from .names import load_names, matrix_from_rows, select_columns
from .two_stage import topn_site_features
from .ugt import ugt_atom_rows

__all__ = [
    "bond_rows",
    "ndealk_bond_rows",
    "ugt_atom_rows",
    "reactivity_atom_rows",
    "quinone_atom_rows",
    "load_names",
    "matrix_from_rows",
    "select_columns",
    "topn_site_features",
]
