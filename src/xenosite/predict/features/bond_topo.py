"""Bond-level topological descriptors (epoxidation / n-dealk family).

Port of ``libridass/*/topological_descriptors/bond.py`` from OpenBabel to RDKit.
``original_atom_ordering`` swaps Atom1/Atom2 on each bond — epoxidation averages
the two site-head predictions, not the features.
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdchem

from . import graph, mol_desc, ptable

ATOM_SYMBOLS = "C N O P S F Cl Br I".split()
HYB_NAMES = {"sp1": 1, "sp2": 2, "sp3": 3}


def _smarts_atom_hits(mol: rdchem.Mol, smarts: str) -> set[int]:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return set()
    hits: set[int] = set()
    for match in mol.GetSubstructMatches(patt):
        hits.update(match)
    return hits


def _smarts_bond_hits(mol: rdchem.Mol, smarts: str) -> set[frozenset[int]]:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return set()
    hits: set[frozenset[int]] = set()
    for match in mol.GetSubstructMatches(patt):
        if len(match) >= 2:
            # all pairs in the match that are actually bonded
            sm = set(match)
            for i in match:
                for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in sm and j > i:
                        hits.add(frozenset((i, j)))
    return hits


def _topo_ids(mol: rdchem.Mol) -> dict[int, int]:
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeIsotopes=False))
    return {i: int(ranks[i]) for i in range(mol.GetNumAtoms())}


def bond_rows(
    mol: rdchem.Mol,
    *,
    original_atom_ordering: bool = True,
    max_depth: int = 5,
    overlap: bool = False,
) -> list[dict[str, float | tuple[int, int]]]:
    """One dict per heavy-atom bond. Includes ``_atoms`` as 0-based ``(a, b)``."""
    Chem.GetSSSR(mol)
    rings = graph.sssr_rings(mol)
    ecc = graph.pairwise_eccentricity(mol)
    min_ecc = min(ecc.values()) if ecc else 0
    max_ecc = max(ecc.values()) if ecc else 0
    md = mol_desc.prefixed_mol_desc(mol)
    epoxide_atoms = _smarts_atom_hits(mol, "[#6]1-O-[#6]1")
    ndealk_bonds = _smarts_bond_hits(mol, "[#6]-[#7]")
    ranks = _topo_ids(mol)

    gasteiger = True
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        gasteiger = False

    hb = graph.heavy_bonds(mol)
    rows: list[dict] = []
    bond_topo_keys: list[str] = []
    for b in hb:
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        if not original_atom_ordering:
            a1, a2 = a2, a1
        i1, i2 = a1.GetIdx(), a2.GetIdx()
        key = ".".join(str(x) for x in sorted((ranks[i1], ranks[i2])))
        bond_topo_keys.append(key)

    topo_index: dict[str, int] = {}
    n_eq = Counter()
    next_id = 0
    assigned: list[int] = []
    for key in bond_topo_keys:
        if key not in topo_index:
            next_id += 1
            topo_index[key] = next_id
        assigned.append(topo_index[key])
        n_eq[topo_index[key]] += 1

    for bi, b in enumerate(hb):
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        if not original_atom_ordering:
            a1, a2 = a2, a1
        i1, i2 = a1.GetIdx(), a2.GetIdx()
        ignore1 = [] if overlap else [i2]
        ignore2 = [] if overlap else [i1]
        depth1 = graph.neighbors_by_depth(mol, i1, max_depth, ignore1)
        depth2 = graph.neighbors_by_depth(mol, i2, max_depth, ignore2)
        row: dict[str, float | tuple[int, int]] = {
            "_atoms": (i1, i2),
            "_bond_index": bi,
        }
        row.update(md)
        _atom_block(mol, row, "Atom1_", a1, depth1, rings, ecc, min_ecc, max_ecc, epoxide_atoms, gasteiger)
        _atom_block(mol, row, "Atom2_", a2, depth2, rings, ecc, min_ecc, max_ecc, epoxide_atoms, gasteiger)

        prefix = "BondDescriptor__"
        t = b.GetBondType()
        row[prefix + "Single"] = float(t == Chem.rdchem.BondType.SINGLE and not b.GetIsAromatic())
        row[prefix + "Double"] = float(t == Chem.rdchem.BondType.DOUBLE)
        row[prefix + "Triple"] = float(t == Chem.rdchem.BondType.TRIPLE)
        row[prefix + "Aromatic"] = float(b.GetIsAromatic())
        tid = assigned[bi]
        row[prefix + "NTopologicalEquivalent"] = float(n_eq[tid])
        row[prefix + "Possible_Site_of_N_Dealkylation"] = float(
            frozenset((i1, i2)) in ndealk_bonds
        )
        rows.append(row)
    return rows


def _atom_block(
    mol: rdchem.Mol,
    row: dict,
    label: str,
    atom: rdchem.Atom,
    depths: list[set[int]],
    rings: list[set[int]],
    ecc: dict[int, int],
    min_ecc: int,
    max_ecc: int,
    epoxide_atoms: set[int],
    gasteiger: bool,
) -> None:
    idx = atom.GetIdx()
    z = atom.GetAtomicNum()
    for size in range(3, 9):
        row[f"{label}Ring{size}"] = float(any(idx in r and len(r) == size for r in rings))
    inv = [1.0 / size for size in range(3, 9) if row[f"{label}Ring{size}"]]
    row[f"{label}MaxInvRingSize"] = max(inv) if inv else 0.0
    row[f"{label}NRings"] = float(sum(1 for r in rings if idx in r))

    d = ecc.get(idx, 0)
    row[f"{label}Span"] = float(d - min_ecc)
    row[f"{label}InvSpan"] = 1.0 / (d - min_ecc + 1)
    row[f"{label}NormSpan"] = (d - min_ecc) / max(float(max_ecc - min_ecc), 1.0)

    # NHeavyNbrs in epo bond.py actually counts aromatic neighbors at depth 1
    depth1 = depths[1] if len(depths) > 1 else set()
    row[f"{label}NHeavyNbrs"] = float(
        sum(1 for j in depth1 if mol.GetAtomWithIdx(j).GetIsAromatic())
    )
    row[f"{label}AromaticNeighbors"] = row[f"{label}NHeavyNbrs"]
    row[f"{label}Aromatic"] = float(atom.GetIsAromatic())
    row[f"{label}NHydrogens"] = float(graph.explicit_hydrogens(atom))
    row[f"{label}HbondAcceptor"] = float(graph.is_hbond_acceptor(atom))
    row[f"{label}HbondDonor"] = float(graph.is_hbond_donor(atom))
    row[f"{label}TotalBondOrder"] = float(graph.total_bond_order(atom))
    row[f"{label}Rotors"] = float(sum(1 for b in atom.GetBonds() if graph.is_rotor(b)))
    row[f"{label}Within_Epoxide"] = float(idx in epoxide_atoms)

    row[f"{label}PT__ElectronNeg"] = ptable.electronegativity(z)
    row[f"{label}PT__ElectronAffinity"] = ptable.electron_affinity(z)
    row[f"{label}PT__MaxBonds"] = float(ptable.max_bonds(z))
    row[f"{label}PT__Ionization"] = ptable.ionization(z)
    row[f"{label}PT__Mass"] = ptable.mass(z)
    hyb = graph.ob_hybridization(atom)
    row[f"{label}PT__CorrectedBondRad"] = ptable.corrected_bond_rad(z, hyb)
    row[f"{label}PT__CorrectedVdwRad"] = ptable.corrected_vdw_rad(z, hyb)

    if gasteiger:
        try:
            row[f"{label}PartialCharge"] = float(atom.GetDoubleProp("_GasteigerCharge"))
        except Exception:
            row[f"{label}PartialCharge"] = 0.0
    else:
        row[f"{label}PartialCharge"] = 0.0

    # Motifs (SMARTS approximations of OB atom flags)
    row[f"{label}AlphaBetaUnsat"] = float(_has_alpha_beta_unsat(atom))
    row[f"{label}CarboxylOxygen"] = float(_is_carboxyl_oxygen(atom))
    row[f"{label}SulfateOxygen"] = float(_match_env(atom, "[OX1]=[SX4]"))
    row[f"{label}PhosphateOxygen"] = float(_match_env(atom, "[OX1]=[PX4]"))
    row[f"{label}NitroOxygen"] = float(_match_env(atom, "[OX1]=[NX3+]"))
    row[f"{label}AmideNitrogen"] = float(_is_amide_nitrogen(atom))

    max_depth = len(depths)
    for depth in range(max_depth):
        nbrs = depths[depth]
        symbols = [ptable.symbol(mol.GetAtomWithIdx(j).GetAtomicNum()) for j in nbrs]
        for sym in ATOM_SYMBOLS:
            c = symbols.count(sym)
            row[f"{label}N{sym}_{depth}"] = float(c)
            if depth != 0:
                row[f"{label}P{sym}_{depth}"] = (c / len(symbols)) if symbols and c else 0.0

    for depth in range(max(max_depth - 1, 0)):
        for atom_symbol in "C N O S ALL".split():
            for name, hyb_level in HYB_NAMES.items():
                if name == "sp1" and atom_symbol in ("O", "S"):
                    continue
                nbrs = list(depths[depth])
                if atom_symbol != "ALL":
                    nbrs = [
                        j
                        for j in nbrs
                        if ptable.symbol(mol.GetAtomWithIdx(j).GetAtomicNum()) == atom_symbol
                    ]
                    lab = f"N{atom_symbol}_"
                else:
                    lab = ""
                hybs = [graph.ob_hybridization(mol.GetAtomWithIdx(j)) for j in nbrs]
                row[f"{label}{lab}{name}_{depth}"] = float(hybs.count(hyb_level))

    for depth in range(max(max_depth - 1, 0)):
        nbrs = depths[depth]
        row[f"{label}Lone_Pair_Depth_{depth}"] = float(
            sum(
                ptable.lone_pairs(
                    mol.GetAtomWithIdx(j).GetAtomicNum(),
                    graph.total_bond_order(mol.GetAtomWithIdx(j)),
                    mol.GetAtomWithIdx(j).GetFormalCharge(),
                )
                for j in nbrs
            )
        )


def _has_alpha_beta_unsat(atom: rdchem.Atom) -> bool:
    for nbr in atom.GetNeighbors():
        for b in nbr.GetBonds():
            o = b.GetOtherAtom(nbr)
            if o.GetIdx() == atom.GetIdx():
                continue
            if b.GetBondType() in (
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
            ):
                return True
    return False


def _is_carboxyl_oxygen(atom: rdchem.Atom) -> bool:
    if atom.GetAtomicNum() != 8:
        return False
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        oxy = [a for a in nbr.GetNeighbors() if a.GetAtomicNum() == 8]
        if len(oxy) >= 2:
            return True
    return False


def _is_amide_nitrogen(atom: rdchem.Atom) -> bool:
    if atom.GetAtomicNum() != 7:
        return False
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        for b in nbr.GetBonds():
            o = b.GetOtherAtom(nbr)
            if o.GetAtomicNum() == 8 and b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                return True
    return False


def _match_env(atom: rdchem.Atom, smarts: str) -> bool:
    mol = atom.GetOwningMol()
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return False
    idx = atom.GetIdx()
    return any(idx in match for match in mol.GetSubstructMatches(patt))
