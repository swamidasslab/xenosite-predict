"""Internal OpenBabel descriptors. Public callers pass RDKit mols and see 0-based indices.

Importing this package does not load OpenBabel; bindings load on first feature call.
"""

from .atom import quinone_atom_rows, reactivity_atom_rows
from .bond import bond_rows, ndealk_bond_rows, ndealk_row_site_key, ndealk_row_site_pair, ndealk_row_topo_gid_pair, ndealk_site_from_row_scores
from .bond_lonepair import phase1_rows
from .names import load_names, matrix_from_rows, select_columns
from .two_stage import topn_site_features
from .ugt import ugt_atom_rows, ugt_inference_rows
from .quinone import eligible_atom_rows, quinone_mol_features, quinone_pair_rows
from .phase1_mol import phase1_mol_features, phase1_site_column_names
from .reactivity_mol import reactivity_mol_features, reactivity_onnx_rows

__all__ = [
    "bond_rows",
    "ndealk_bond_rows",
    "ndealk_row_site_pair",
    "ndealk_row_site_key",
    "ndealk_row_topo_gid_pair",
    "ndealk_site_from_row_scores",
    "ugt_atom_rows",
    "ugt_inference_rows",
    "reactivity_atom_rows",
    "reactivity_onnx_rows",
    "reactivity_mol_features",
    "quinone_atom_rows",
    "eligible_atom_rows",
    "quinone_pair_rows",
    "quinone_mol_features",
    "phase1_rows",
    "load_names",
    "matrix_from_rows",
    "select_columns",
    "phase1_mol_features",
    "phase1_site_column_names",
]
