"""Atom-level topological descriptors (UGT / reactivity / quinone family).

UGT's ``TopologicalDescriptors`` (67 columns) is the smaller shared graph.
Reactivity ``AtomTD`` is a larger, distinct copy — not merged until diffs +
golden tests prove identity (they do not; see docs/vendored-diffs.md).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdchem

from . import graph, mol_desc, ptable

UGT_HEADER = [
    "MaxInvRingSize",
    "Ring3",
    "Ring4",
    "Ring5",
    "Ring6",
    "Ring7",
    "Ring8",
    "NRings",
    "Span",
    "InvSpan",
    "NormSpan",
    "NHeavyNbrs",
    "Aromatic",
    "Hydrogens",
    "Rotors",
    "sp1",
    "sp2",
    "sp3",
    "hybX",
    "NC_0",
    "PC_0",
    "NO_0",
    "PO_0",
    "NN_0",
    "PN_0",
    "NS_0",
    "PS_0",
    "NP_0",
    "PP_0",
    "NX_0",
    "PX_0",
    "NC_1",
    "PC_1",
    "NO_1",
    "PO_1",
    "NN_1",
    "PN_1",
    "NS_1",
    "PS_1",
    "NP_1",
    "PP_1",
    "NX_1",
    "PX_1",
    "NC_2",
    "PC_2",
    "NO_2",
    "PO_2",
    "NN_2",
    "PN_2",
    "NS_2",
    "PS_2",
    "NP_2",
    "PP_2",
    "NX_2",
    "PX_2",
    "NC_3",
    "PC_3",
    "NO_3",
    "PO_3",
    "NN_3",
    "PN_3",
    "NS_3",
    "PS_3",
    "NP_3",
    "PP_3",
    "NX_3",
    "PX_3",
]


def ugt_atom_rows(mol: rdchem.Mol) -> list[dict[str, float]]:
    """Port of ``ugt1.xenosite.descriptor.topo.TopologicalDescriptors``."""
    Chem.GetSSSR(mol)
    rings = graph.sssr_rings(mol)
    ecc = graph.pairwise_eccentricity(mol)
    min_ecc = min(ecc.values()) if ecc else 0
    max_ecc = max(ecc.values()) if ecc else 0
    md = mol_desc.molecule_descriptors(mol)
    heavy = graph.heavy_atoms(mol)
    rows = []
    for atom in heavy:
        idx = atom.GetIdx()
        depths = graph.neighbors_by_depth(mol, idx, 4)
        out = {h: 0.0 for h in UGT_HEADER}
        for r in rings:
            if idx in r:
                s = len(r)
                si = min(s - 3, 5) + 1  # Ring3.. columns 1..6
                key = f"Ring{min(s, 8)}" if 3 <= s <= 8 else None
                if 3 <= s <= 8:
                    out[f"Ring{s if s < 8 else 8}"] += 1.0
                out["MaxInvRingSize"] = max(out["MaxInvRingSize"], 3.0 / s)
        out["NRings"] = sum(out[f"Ring{n}"] for n in range(3, 9))
        d = ecc.get(idx, 0)
        out["Span"] = float(d - min_ecc)
        out["InvSpan"] = 1.0 / (d - min_ecc + 1)
        out["NormSpan"] = (d - min_ecc) / max(float(max_ecc - min_ecc), 1.0)
        nbrs0 = [a for a in atom.GetNeighbors() if a.GetAtomicNum() != 1]
        out["NHeavyNbrs"] = float(len(nbrs0))
        out["Aromatic"] = float(atom.GetIsAromatic())
        out["Hydrogens"] = float(graph.explicit_hydrogens(atom))
        out["Rotors"] = float(
            sum(1 for b in atom.GetBonds() if graph.is_rotor(b))
        )
        h = graph.ob_hybridization(atom)
        if h == 1:
            out["sp1"] = 1.0
        elif h == 2:
            out["sp2"] = 1.0
        elif h == 3:
            out["sp3"] = 1.0
        else:
            out["hybX"] = 1.0

        base = UGT_HEADER.index("NC_0")
        skip = 12
        for n in range(4):
            neigh = depths[n] if n < len(depths) else set()
            atoms_n = [mol.GetAtomWithIdx(j) for j in neigh]
            s = float(len(atoms_n)) or 1.0
            other = s
            checks = [
                (lambda a: a.GetAtomicNum() == 6, 0),
                (lambda a: a.GetAtomicNum() == 8, 2),
                (lambda a: a.GetAtomicNum() == 7, 4),
                (lambda a: a.GetAtomicNum() == 16, 6),
                (lambda a: a.GetAtomicNum() == 15, 8),
            ]
            for pred, off in checks:
                c = float(sum(1 for a in atoms_n if pred(a)))
                out[UGT_HEADER[base + n * skip + off]] = c
                out[UGT_HEADER[base + n * skip + off + 1]] = c / s
                other -= c
            out[UGT_HEADER[base + n * skip + 10]] = other
            out[UGT_HEADER[base + n * skip + 11]] = other / s

        out["_atom"] = float(idx)
        for k, v in md.items():
            out[k] = float(v)
        rows.append(out)
    return rows


def reactivity_atom_rows(mol: rdchem.Mol) -> list[dict[str, float]]:
    """Reactivity uses its own AtomTD copy; start from UGT-like atoms plus bond-style extras.

    Full column identity waits on TSV headers from ``make extract-weights``.
    """
    rows = ugt_atom_rows(mol)
    extra_mol = mol_desc.prefixed_mol_desc(mol)
    for row in rows:
        row.update(extra_mol)
    return rows
