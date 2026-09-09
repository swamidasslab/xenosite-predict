"""XenoNet beam / best-first expansion (PhaseOneRS + phase1 weights).

v0 heap key is ``(neg_weight, smiles, site)`` — SMILES-stable. Legacy compared
the RDKit mol object, which is not ordered and can shuffle ties. Golden graphs
use a large ``beam_width`` so ties do not drop children.
"""

from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from typing import Iterator

from ._unvalidated import warn_unvalidated

warn_unvalidated()

from rdkit import Chem
from xenosite.predict.forest import forest_site_indexing, forest_site_to_rdkit
from xenosite.predict.forest_rdkit import can_smi, clean, mol_to_smiles
from xenosite.forest import PhaseOneRS

from .graph import XenoGraph
from .score import (
    Phase1SiteTable,
    edge_weight,
    phase1_site_strings,
    phase1_site_table,
    reaction_type,
)


def canonize(mol: Chem.Mol) -> str:
    smis = can_smi(rdmol=mol)
    if not smis:
        return mol_to_smiles(mol)
    return smis[0]


def _rdkit_smi(mol: Chem.Mol) -> str:
    return mol_to_smiles(mol)


def min_heavy_atoms(start: Chem.Mol, targets: list[str]) -> int:
    n = start.GetNumHeavyAtoms()
    if targets:
        tmin = min(
            Chem.MolFromSmiles(t).GetNumHeavyAtoms() // 2
            for t in targets
            if Chem.MolFromSmiles(t) is not None
        )
        return max(2, min(n // 2, tmin))
    return max(2, n // 4)


def possible_metabolites(
    mol: Chem.Mol,
) -> Iterator[tuple[tuple[str, frozenset[int]], Chem.Mol]]:
    n_atoms = mol.GetNumAtoms()
    mode = forest_site_indexing()
    seen: set[tuple] = set()
    for (rule, site), products in PhaseOneRS.metabolites(mol, unique=True):
        rdkit_site = forest_site_to_rdkit(site, n_atoms, mode)
        for product in clean(products):
            key = (rule, tuple(sorted(rdkit_site)), _rdkit_smi(product))
            if key in seen:
                continue
            seen.add(key)
            yield (rule, rdkit_site), product


def score_child(
    table: Phase1SiteTable,
    rule: str,
    site: frozenset[int],
    scoring: str,
) -> float:
    rxn = reaction_type(rule)
    strings = phase1_site_strings(rule, site)
    return edge_weight(
        table,
        site_strings=strings,
        site_rdkit_zero=site,
        rxn_type=rxn,
        scoring=scoring,
    )


def evaluate_paths(
    start: Chem.Mol,
    *,
    backend,
    depth_limit: int,
    beam_width: int,
    targets: list[str],
    scoring: str,
    cache: dict[str, Phase1SiteTable],
) -> XenoGraph:
    root = canonize(start)
    target_mols = [Chem.MolFromSmiles(t) for t in targets]
    target_mols = [m for m in target_mols if m is not None]
    target_set = [canonize(m) for m in target_mols]
    rdkit_targets = {_rdkit_smi(m) for m in target_mols}
    graph = XenoGraph(root=root, targets=target_set)
    min_heavy = min_heavy_atoms(start, target_set)

    def table_for(mol: Chem.Mol, smi: str) -> Phase1SiteTable:
        if smi not in cache:
            cache[smi] = phase1_site_table(mol, backend)
        return cache[smi]

    def recurse(
        mol: Chem.Mol,
        product_path: list[Chem.Mol],
        canon_path: list[str],
        path_weights: list[float],
        site_path: list[tuple[str, frozenset[int]]],
        depth: int,
    ) -> None:
        if target_set and canon_path[-1] in target_set:
            graph.add_path(canon_path, path_weights, site_path)
            return

        if not target_set and depth == depth_limit:
            graph.add_path(canon_path, path_weights, site_path)
            return

        if depth >= depth_limit:
            return

        parent_smi = canon_path[-1]
        try:
            table = table_for(mol, parent_smi)
        except Exception:
            return

        buckets: dict[str, list] = defaultdict(list)
        next_targets: list[tuple[float, Chem.Mol, tuple[str, frozenset[int]]]] = []
        zero_possible = True
        seen: set[tuple] = set()

        for (rule, site), product in possible_metabolites(mol):
            smi = _rdkit_smi(product)
            key = (rule, tuple(sorted(site)), smi)
            if key in seen:
                continue
            seen.add(key)
            if product.GetNumHeavyAtoms() <= min_heavy:
                continue
            if canonize(product) in canon_path:
                continue
            zero_possible = False
            w = score_child(table, rule, site, scoring)
            neg = -w
            item = (neg, product, (rule, site))
            if target_set and smi in rdkit_targets:
                next_targets.append(item)
            heappush(buckets["all_reactions"], (neg, smi, tuple(sorted(site)), product, (rule, site)))

        for neg, product, site in next_targets:
            recurse(
                product,
                product_path + [product],
                canon_path + [canonize(product)],
                path_weights + [-neg],
                site_path + [site],
                depth + 1,
            )

        if not target_set and zero_possible:
            graph.add_path(canon_path, path_weights, site_path)

        for key in buckets:
            n = min(beam_width, len(buckets[key]))
            for _ in range(n):
                neg, _smi, _site_t, product, site = heappop(buckets[key])
                recurse(
                    product,
                    product_path + [product],
                    canon_path + [canonize(product)],
                    path_weights + [-neg],
                    site_path + [site],
                    depth + 1,
                )

    recurse(start, [start], [root], [], [], 0)
    return graph
