"""Temporary conjugation metabolite wrapper.

Forest ships Glucuronidation and Glutathionation. UGT and GSH/protein use those
directly.

Dummy ``*`` atoms carry a CX ``atomLabel`` (``GlcA`` / ``GSH`` / ``Protein``)
so RDKit depictions can name the conjugate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from xenosite.forest import load_ruleset
from xenosite.forest.base import can_smi
from xenosite.forest.rulesets import RuleSet


@dataclass(frozen=True)
class ConjugateHead:
    """How one predict model/head enumerates star conjugates."""

    ruleset: str
    label: str
    pathway: Optional[str] = None


HEADS: dict[str, ConjugateHead] = {
    "ugt": ConjugateHead("CJ.Glucuronidation", "GlcA"),
    "reactivity.gsh": ConjugateHead("CJ.Glutathionation", "GSH"),
    "reactivity.protein": ConjugateHead("CJ.Glutathionation", "Protein", "Protein"),
}

STAR_RULESETS: frozenset[str] = frozenset(
    {
        "CJ",
        "CJ.Glucuronidation",
        "CJ.Glutathionation",
        "CJ.Acetylation",
        "CJ.Sulfation",
    }
)


def head_for_model(model: str) -> Optional[ConjugateHead]:
    return HEADS.get(model)


def is_star_conjugate(spec: str) -> bool:
    return spec in STAR_RULESETS or spec.startswith("CJ.")


def load_conjugate_ruleset(spec: str) -> RuleSet:
    return load_ruleset(spec)


def labeled_star_mol(product: Chem.Mol, label: str) -> Chem.Mol:
    """Copy ``product`` and set CX ``atomLabel`` on dummy atoms."""
    mol = Chem.Mol(product)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp("atomLabel", label)
    return mol


def mol_to_cxsmiles(mol: Chem.Mol) -> Optional[str]:
    """Non-isomeric CXSMILES (atom labels kept). ``None`` if the mol will not write."""
    try:
        out = Chem.MolToCXSmiles(mol, False)
    except Exception:
        smiles = can_smi(rdmol=mol)
        return smiles[0] if smiles else None
    return out or None


def bare_smiles(smiles: str) -> str:
    """Drop the CXSMILES ``|...|`` block."""
    return smiles.split()[0]
