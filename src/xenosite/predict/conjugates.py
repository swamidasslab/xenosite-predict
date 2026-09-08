"""Temporary conjugation metabolite wrapper.

Forest ships Glucuronidation and Glutathionation. UGT and GSH/protein use those
directly. DNA and cyanide reuse glutathionation electrophile SMARTS (epoxide,
C-Cl, terminal alkene) and drop the thiol-disulfide rule, which is not a DNA/CN
reaction.

Dummy ``*`` atoms are written as **CXSMILES** on ``Metabolite.smiles`` (the
field is still named ``smiles``). The CX ``atomLabel`` is ``GlcA`` / ``GSH`` /
``Protein`` / ``DNA`` / ``CN`` so RDKit depictions can name the conjugate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from rdkit import Chem
from xenosite.forest import load_ruleset
from xenosite.forest.base import can_smi
from xenosite.forest.rules import Glutathionation
from xenosite.forest.rulesets import RuleSet

NO_THIOL_RULESET = "GlutathionationNoThiol"


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
    "reactivity.dna": ConjugateHead(NO_THIOL_RULESET, "DNA", "DNA"),
    "reactivity.cyanide": ConjugateHead(NO_THIOL_RULESET, "CN", "Cyanide"),
}

STAR_RULESETS: frozenset[str] = frozenset(
    {
        "CJ",
        "CJ.Glucuronidation",
        "CJ.Glutathionation",
        "CJ.Acetylation",
        "CJ.Sulfation",
        NO_THIOL_RULESET,
    }
)


def head_for_model(model: str) -> Optional[ConjugateHead]:
    return HEADS.get(model)


def is_star_conjugate(spec: str) -> bool:
    return spec in STAR_RULESETS or spec.startswith("CJ.")


class GlutathionationNoThiol(Glutathionation):
    """Epoxide, C-Cl, and terminal alkene — no substrate-thiol disulfide."""

    smarts = [s for s in Glutathionation.smarts if "[#16h1" not in s]


@lru_cache(maxsize=1)
def _no_thiol_ruleset() -> RuleSet:
    return RuleSet(
        [GlutathionationNoThiol(name="Glutathionation")],
        name=NO_THIOL_RULESET,
    )


def load_conjugate_ruleset(spec: str) -> RuleSet:
    if spec == NO_THIOL_RULESET:
        return _no_thiol_ruleset()
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
