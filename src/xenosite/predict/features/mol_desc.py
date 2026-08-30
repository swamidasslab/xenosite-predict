"""RDKit stand-ins for OpenBabel ``mol.calcdesc()`` molecule descriptors.

Column names match the legacy TSV headers (``MolDesc__TPSA``, …). Live tests
compare against OpenBabel dumps; mismatches are recorded in vendored-diffs.md.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdchem

from . import graph

MOL_DESC_NAMES = [
    "atoms",
    "bonds",
    "TPSA",
    "logP",
    "MW",
    "MR",
    "HBD",
    "HBA1",
    "HBA2",
    "sbonds",
    "dbonds",
    "tbonds",
    "abonds",
]


def molecule_descriptors(mol: rdchem.Mol) -> dict[str, float]:
    """Return the pybel.calcdesc keys used by XenoSite models."""
    n_explicit = mol.GetNumAtoms()
    n_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    # OpenBabel ``atoms`` counts hydrogens after AddH in some pipelines; we
    # count implicit H on the RDKit graph without adding them.
    atoms = float(n_explicit + n_h) if all(a.GetAtomicNum() != 1 for a in mol.GetAtoms()) else float(n_explicit)
    bonds = mol.GetNumBonds()
    # If Hs are implicit, OpenBabel bond count includes X–H. Approximate:
    if all(a.GetAtomicNum() != 1 for a in mol.GetAtoms()):
        bonds = float(mol.GetNumBonds() + n_h)
    else:
        bonds = float(mol.GetNumBonds())

    sb = db = tb = ab = 0
    for b in mol.GetBonds():
        t = b.GetBondType()
        if b.GetIsAromatic() or t == Chem.rdchem.BondType.AROMATIC:
            ab += 1
        elif t == Chem.rdchem.BondType.DOUBLE:
            db += 1
        elif t == Chem.rdchem.BondType.TRIPLE:
            tb += 1
        else:
            sb += 1

    hba = float(rdMolDescriptors.CalcNumHBA(mol))
    return {
        "atoms": atoms,
        "bonds": bonds,
        "TPSA": float(Descriptors.TPSA(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "MW": float(Descriptors.MolWt(mol)),
        "MR": float(Crippen.MolMR(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA1": hba,
        "HBA2": hba,
        "sbonds": float(sb),
        "dbonds": float(db),
        "tbonds": float(tb),
        "abonds": float(ab),
    }


def prefixed_mol_desc(mol: rdchem.Mol, prefix: str = "MolDesc__") -> dict[str, float]:
    md = molecule_descriptors(mol)
    extra = {
        f"{prefix}NumRings": float(rdMolDescriptors.CalcNumRings(mol)),
        f"{prefix}hydrogens": float(sum(a.GetTotalNumHs() for a in mol.GetAtoms())),
        f"{prefix}heavy_atoms": float(len(graph.heavy_atoms(mol))),
    }
    out = {f"{prefix}{k}": float(v) for k, v in md.items()}
    out.update(extra)
    return out
