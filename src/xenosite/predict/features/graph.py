"""RDKit molecular graph helpers shared by every descriptor port."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdchem

HYB_SP1 = Chem.rdchem.HybridizationType.SP
HYB_SP2 = Chem.rdchem.HybridizationType.SP2
HYB_SP3 = Chem.rdchem.HybridizationType.SP3

# OpenBabel GetHyb(): 1=sp, 2=sp2, 3=sp3, 4=sp3d, 5=sp3d2, 0=unspecified
_RDKIT_TO_OB_HYB = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
}


def ob_hybridization(atom: rdchem.Atom) -> int:
    return _RDKIT_TO_OB_HYB.get(atom.GetHybridization(), 0)


def heavy_atoms(mol: rdchem.Mol) -> list[rdchem.Atom]:
    return [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]


def heavy_bonds(mol: rdchem.Mol) -> list[rdchem.Bond]:
    out = []
    for b in mol.GetBonds():
        if b.GetBeginAtom().GetAtomicNum() != 1 and b.GetEndAtom().GetAtomicNum() != 1:
            out.append(b)
    return out


def neighbors_by_depth(
    mol: rdchem.Mol,
    start: int,
    depth: int,
    ignore: Iterable[int] = (),
) -> list[set[int]]:
    """BFS neighborhoods. Index 0 is ``{start}`` (OpenBabel-style). Heavy atoms only."""
    ignore_s = set(ignore)
    current = {start}
    seen: set[int] = set()
    output = [set(current)]
    for _ in range(depth):
        nxt: set[int] = set()
        for c in current:
            atom = mol.GetAtomWithIdx(c)
            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                if nbr.GetAtomicNum() == 1:
                    continue
                if j in seen or j in ignore_s or j in current:
                    continue
                nxt.add(j)
        seen |= current
        current = nxt
        output.append(current)
    return output


def pairwise_eccentricity(mol: rdchem.Mol) -> dict[int, int]:
    """Max graph distance from each heavy atom (span)."""
    heavy = [a.GetIdx() for a in heavy_atoms(mol)]
    ecc = {}
    for i in heavy:
        dist = {i: 0}
        q = [i]
        for u in q:
            for v in mol.GetAtomWithIdx(u).GetNeighbors():
                j = v.GetIdx()
                if v.GetAtomicNum() == 1 or j in dist:
                    continue
                dist[j] = dist[u] + 1
                q.append(j)
        ecc[i] = max(dist.values()) if dist else 0
    return ecc


def sssr_rings(mol: rdchem.Mol) -> list[set[int]]:
    Chem.GetSSSR(mol)
    ri = mol.GetRingInfo()
    return [set(r) for r in ri.AtomRings()]


def is_rotor(bond: rdchem.Bond) -> bool:
    """Approximate OpenBabel ``IsRotor``: rotatable single non-ring bond."""
    if bond.IsInRing():
        return False
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    if a.GetDegree() == 1 or b.GetDegree() == 1:
        return False
    return True


def bond_order_int(bond: rdchem.Bond) -> int:
    t = bond.GetBondType()
    if t == Chem.rdchem.BondType.AROMATIC:
        return 1  # OB aromatic IsSingle is false; GetBondOrder often 1.5→handled separately
    if t == Chem.rdchem.BondType.TRIPLE:
        return 3
    if t == Chem.rdchem.BondType.DOUBLE:
        return 2
    return 1


def total_bond_order(atom: rdchem.Atom) -> int:
    """Sum of GetBondOrder over incident bonds (aromatic counted as 1)."""
    return sum(bond_order_int(b) for b in atom.GetBonds())


def explicit_hydrogens(atom: rdchem.Atom) -> int:
    """Match OB ``ExplicitHydrogenCount`` when the mol has no explicit H: implicit H."""
    n = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            n += 1
    if n == 0:
        n = atom.GetTotalNumHs()
    return n


def is_hbond_acceptor(atom: rdchem.Atom) -> bool:
    z = atom.GetAtomicNum()
    return z in (7, 8) and atom.GetTotalNumHs() < 3


def is_hbond_donor(atom: rdchem.Atom) -> bool:
    if atom.GetAtomicNum() not in (7, 8):
        return False
    return atom.GetTotalNumHs() > 0
