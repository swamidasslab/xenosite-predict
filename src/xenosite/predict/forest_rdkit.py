"""RDKit 2026 valence-cache shims for ``xenosite.forest``.

Forest (and Python-2 XenoNet) assume RDKit fills implicit/explicit hydrogen
counts lazily inside ``RunReactants`` and ``MolToSmiles``. RDKit 2026 asserts
those caches already exist, so resonance copies from ``join_fragments`` abort
before a reaction runs.

This module replaces the forest call sites we hit until forest itself calls
``UpdatePropertyCache(strict=False)``. Same idea as :mod:`conjugates`: keep a
local adapter so this package can move without waiting on a forest release.
"""

from __future__ import annotations

import itertools
import re

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import Kekulize, SanitizeFlags, SanitizeMol
from xenosite.forest import base as forest_base
from xenosite.forest import rules as forest_rules
from xenosite.forest import utils as forest_utils
from xenosite.forest.base import AtomTracker, SmartsReactionRule
from xenosite.forest.rulesets import RuleSet

_APPLIED = False


def refresh_mol(mol):
    """Fill valence caches without failing on odd valences (RDKit 2026)."""
    if mol is None:
        return mol
    if isinstance(mol, (list, tuple)):
        for item in mol:
            refresh_mol(item)
        return mol
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return mol


def mol_to_smiles(mol, *, canonical: bool = True, isomericSmiles: bool = False) -> str:
    """``MolToSmiles`` after :func:`refresh_mol`."""
    refresh_mol(mol)
    return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomericSmiles)


def clean(mol):
    """Forest ``clean`` plus valence refresh on each fragment."""
    if isinstance(mol, (list, tuple)):
        return list(itertools.chain.from_iterable(clean(x) for x in mol))

    out = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        for atom in frag.GetAtoms():
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
        SanitizeMol(frag, SanitizeFlags.SANITIZE_CLEANUP, catchErrors=True)
        for atom in frag.GetAtoms():
            atom.SetNoImplicit(False)
        SanitizeMol(frag, catchErrors=True)
        try:
            Kekulize(frag, clearAromaticFlags=True)
        except ValueError:
            pass
        refresh_mol(frag)
        out.append(frag)
    return out


def can_smi(line: str = "", rdmol=None):
    """Forest ``can_smi`` with valence refresh before each ``MolToSmiles``."""
    if isinstance(rdmol, (list, tuple)):
        return [can_smi(rdmol=x) for x in rdmol]

    if rdmol:
        SanitizeMol(rdmol, catchErrors=True)
        refresh_mol(rdmol)
        line = MolToSmiles(rdmol)

    if "." in line:
        return list(itertools.chain(*[can_smi(line=x) for x in line.split(".")]))

    split = line.split()
    smi = split[0]

    parsed = MolFromSmiles(smi)
    if parsed:
        refresh_mol(parsed)
        out = MolToSmiles(parsed)
    else:
        out = smi

    out = re.sub(r"\[CH*\]", "C", out)
    out = re.sub(r"\[H\]", "", out)
    out = re.sub(r"\[C\]", "C", out)
    return [out]


def _kekulize(self, mol) -> None:
    SanitizeMol(mol, SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
    try:
        Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        pass
    refresh_mol(mol)


def smarts_metabolites(self, mol, kekulize=True, **kwargs):
    """Forest ``SmartsReactionRule.metabolites`` with a cache refresh before ``RunReactants``.

    Resonance copies (``kekulize=False``) are the usual crash: they look like
    valid mols but have an empty implicit-H cache.
    """
    if kekulize:
        self._kekulize(mol)

    self._remove_props(mol)
    refresh_mol(mol)

    for rxn_num, rxn in enumerate(self.rxns):
        for _prod_num, prod in enumerate(rxn.RunReactants([mol])):
            products = list(prod)
            site = self._get_site(products, mapid_site=self.mapid_site)
            self._copy_props(
                products, {"react_atom_idx": AtomTracker.previous_index_prop_name}
            )
            yield (self.name + "_SmartsReactionRuleRxn%d" % (rxn_num), site), clean(
                products
            )


def ruleset_metabolites(self, mol, sites=None, unique=False, strict=True, **kwargs):
    """Forest ``RuleSet.metabolites`` using :func:`mol_to_smiles` for the unique key."""
    if unique:
        seen = []

    if sites:
        if isinstance(sites, str):
            sites = self._process_sites(sites=sites)

    for rule in self.rules:
        for (rxnname, site), metabolites in rule.metabolize(
            mol, strict=strict, **kwargs
        ):
            if sites:
                if not self._site_match(sites, site, ruleset_name=rule.name):
                    continue

            if unique:
                fline = "_".join(
                    [
                        ".".join(mol_to_smiles(m) for m in metabolites),
                        rule.name,
                        str(site),
                    ]
                )
                if fline in seen:
                    continue
                seen.append(fline)

            yield (rxnname, site), metabolites


def apply_rdkit2026_patches() -> None:
    """Bind the shims onto the imported ``xenosite.forest`` classes (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    SmartsReactionRule.metabolites = smarts_metabolites
    SmartsReactionRule._kekulize = _kekulize
    RuleSet.metabolites = ruleset_metabolites
    forest_utils.clean = clean
    forest_base.clean = clean
    forest_base.can_smi = can_smi
    forest_rules.clean = clean
    _APPLIED = True


apply_rdkit2026_patches()
