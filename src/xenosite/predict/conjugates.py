"""Conjugation heads: map predict models onto forest Phase II rulesets.

Forest 0.10+ ships Glucuronidation and Reactivity adducts (GSH, Protein, DNA,
Cyanide) via ``resolve("xf:…")``. Products are already star CXSMILES with
``atomLabel`` set (``GlcA`` / ``GSH`` / ``Protein`` / ``DNA`` / ``Cyanide``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from xenosite.forest import resolve

NO_THIOL_RULESET = "xf:GlutathionationNoThiol"


@dataclass(frozen=True)
class ConjugateHead:
    """How one predict model/head enumerates star conjugates."""

    ruleset: str
    label: str
    pathway: Optional[str] = None


HEADS: dict[str, ConjugateHead] = {
    "ugt": ConjugateHead("xf:Glucuronidation", "GlcA"),
    "reactivity.gsh": ConjugateHead("xf:GSH", "GSH", "Glutathionation"),
    "reactivity.protein": ConjugateHead("xf:Protein", "Protein", "Protein"),
    "reactivity.dna": ConjugateHead("xf:DNA", "DNA", "DNA"),
    "reactivity.cyanide": ConjugateHead("xf:Cyanide", "Cyanide", "Cyanide"),
}

STAR_RULESETS: frozenset[str] = frozenset(
    {
        "xf:Glucuronidation",
        "xf:Glutathionation",
        NO_THIOL_RULESET,
        "xf:GSH",
        "xf:Protein",
        "xf:DNA",
        "xf:Cyanide",
        "xf:Acetylation",
        "xf:Sulfation",
        "xf:Reactivity",
        "xf:Reactivity/GSH",
        "xf:Reactivity/Protein",
        "xf:Reactivity/DNA",
        "xf:Reactivity/Cyanide",
    }
)


def head_for_model(model: str) -> Optional[ConjugateHead]:
    return HEADS.get(model)


def is_star_conjugate(spec: str) -> bool:
    if spec in STAR_RULESETS:
        return True
    return spec.startswith("xf:Reactivity") or spec.startswith("xf:Glutathion")


@lru_cache(maxsize=32)
def load_conjugate_ruleset(spec: str):
    """Resolve a forest ruleset CURIE / name (cached)."""
    return resolve(spec)


def bare_smiles(smiles: str) -> str:
    """Drop the CXSMILES ``|...|`` block."""
    return smiles.split()[0]
