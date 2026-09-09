"""Conjugation heads: map predict models onto forest Phase II rulesets.

Forest 0.2.3+ ships Glucuronidation, Glutathionation, and
``GlutathionationNoThiol``. UGT and GSH/protein use the built-in rulesets.
DNA and cyanide use ``GlutathionationNoThiol`` (epoxide, C-Cl, terminal
alkene; no thiol disulfide).

Dummy ``*`` atoms are written as **CXSMILES** on ``Metabolite.smiles`` (the
field is still named ``smiles``). The CX ``atomLabel`` is ``GlcA`` / ``GSH`` /
``Protein`` / ``DNA`` / ``CN`` so RDKit depictions can name the conjugate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from xenosite.forest import load_ruleset
from xenosite.forest.utils import label_star_atoms, mol_to_cxsmiles as _forest_cxsmiles

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


def load_conjugate_ruleset(spec: str):
    return load_ruleset(spec)


def labeled_star_mol(product: Chem.Mol, label: str) -> Chem.Mol:
    """Copy ``product`` and set CX ``atomLabel`` on dummy atoms."""
    return label_star_atoms(Chem.Mol(product), label)


def mol_to_cxsmiles(mol: Chem.Mol) -> Optional[str]:
    """Non-isomeric CXSMILES (atom labels kept)."""
    return _forest_cxsmiles(mol, isomericSmiles=False)


def bare_smiles(smiles: str) -> str:
    """Drop the CXSMILES ``|...|`` block."""
    return smiles.split()[0]
