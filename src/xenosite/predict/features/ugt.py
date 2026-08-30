"""OpenBabel UGT topological + molecule descriptors. Pandas-free.

Port of ``libridass.ugt1.xenosite.descriptor.topo.TopologicalDescriptors``
and ``MoleculeDescriptors``. Neighborhoods are cumulative. OpenBabel is internal.
"""

from __future__ import annotations

from collections import defaultdict

from . import _ob
from .molgraph import MolGraph

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

MOL_HEADER = [
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


def ugt_atom_rows(rdkit_mol) -> list[dict]:
    pymol = _ob.from_rdkit_mol(rdkit_mol)
    mg = MolGraph(pymol)
    rings = list(pymol.sssr)
    verts = sorted(mg.vertex)
    dist, distmap = mg.pairwise_distance()
    ecc = dist.max(axis=1)
    min_d = float(ecc.min()) if len(verts) else 0.0
    max_d = float(ecc.max()) if len(verts) else 0.0
    span = max(float(max_d - min_d), 1.0)

    neighborhood: dict[tuple[int, int], set[int]] = defaultdict(set)
    for a in verts:
        neighborhood[(a, 0)].add(a)
    for i in range(1, 4):
        for a in verts:
            for n in neighborhood[(a, i - 1)]:
                neighborhood[(a, i)].update(mg.neighbors[n])

    md = pymol.calcdesc()
    rows: list[dict] = []
    for a in verts:
        out = {h: 0.0 for h in UGT_HEADER}
        obatom = pymol.OBMol.GetAtom(a)
        for r in rings:
            if r.IsInRing(a):
                s = r.Size()
                si = min(s - 3, 5) + 1
                key = UGT_HEADER[si]
                out[key] += 1.0
                out["MaxInvRingSize"] = max(out["MaxInvRingSize"], 3.0 / s)
        out["NRings"] = sum(out[f"Ring{n}"] for n in range(3, 9))
        d = float(ecc[distmap[a]])
        out["Span"] = d - min_d
        out["InvSpan"] = 1.0 / (d - min_d + 1)
        out["NormSpan"] = (d - min_d) / span
        out["NHeavyNbrs"] = float(len(mg.neighbors[a]))
        out["Aromatic"] = float(int(obatom.IsAromatic()))
        out["Hydrogens"] = float(
            obatom.ImplicitHydrogenCount() + obatom.ExplicitHydrogenCount()
        )
        out["Rotors"] = float(
            sum(
                int(pymol.OBMol.GetBond(a, i).IsRotor())
                for i in mg.neighbors[a]
                if pymol.OBMol.GetBond(a, i) is not None
            )
        )
        h = obatom.GetHyb()
        # Legacy writes hyb into column 14+min(H,4), which overwrites Rotors when H==0.
        slot = 14 + min(h, 4)
        if 0 <= slot < len(UGT_HEADER):
            out[UGT_HEADER[slot]] = 1.0

        base = UGT_HEADER.index("NC_0")
        skip = 12
        for nhood in range(4):
            members = list(neighborhood[(a, nhood)])
            size = float(len(members)) or 1.0
            other = size
            checks = (
                (lambda at: at.IsCarbon(), 0),
                (lambda at: at.IsOxygen(), 2),
                (lambda at: at.IsNitrogen(), 4),
                (lambda at: at.IsSulfur(), 6),
                (lambda at: at.IsPhosphorus(), 8),
            )
            for pred, off in checks:
                c = float(sum(1 for i in members if pred(pymol.OBMol.GetAtom(i))))
                out[UGT_HEADER[base + nhood * skip + off]] = c
                out[UGT_HEADER[base + nhood * skip + off + 1]] = c / size
                other -= c
            out[UGT_HEADER[base + nhood * skip + 10]] = other
            out[UGT_HEADER[base + nhood * skip + 11]] = other / size

        out["_atom"] = _ob.ob_idx_to_rdkit(a)
        out["_index"] = f"1.{a}"
        for k in MOL_HEADER:
            try:
                out[k] = float(md.get(k, 0.0) or 0.0)
            except (TypeError, ValueError):
                out[k] = 0.0
        rows.append(out)
    return rows
