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


_BOOST_SMARTS = {
    "Aliphatic_hydroxyl": "[OX2H][A;!#1;!$([C,N,P,S]=O)]",
    "Aromatic_hydroxyl": "[OX2H]a",
    "Carboxylic_acid": "[OX2H1][CX3]=O",
    "NH2": "[$([NX3H2,NX4H3+;!$(NC=O)])]",
    "NH": "[$([NX3H1,NX4H2+;!$(NC=O)])]",
    "N": "[$([NX3,NX4+;!$([NX3H2,NX4H3+,NX3H1,NX4H2+])])]",
    "Sulfur": "[$([SX2H]),$([OX2H][SX4](=O)=O)]",
}

_CHANCE_SMARTS = {
    "Aliphatic_hydroxyl1": "[$([OX2H][A;!#1;!$([C,N,P,S]=O)])]",
    "Aromatic_hydroxyl1": "[$([OX2H]a)]",
    "Carboxylic_acid1": "[$([OX2H1][CX3]=O)]",
    "Nitrogen1": "[$([#7;!R0]),$([NX3,NX4]);!$([#7][C,S]=[O,S,N]);!$(N=O)]",
}

_HEURISTIC_CHANCE = {
    "Nitrogen1": 0.0846824408468,
    "Aliphatic_hydroxyl1": 0.496777658432,
    "Aromatic_hydroxyl1": 0.763663220089,
    "Carboxylic_acid1": 0.799684542587,
    "Remaining": 0.00146456745332,
}

_QUANTUM_COLS = (
    "charge",
    "elec_Dens",
    "active_Chg",
    "elec_E",
    "1_EE_rep",
    "1_EN_attr",
    "elec_res",
    "elec_xch",
    "2_EE_rep",
    "2_EN_attr",
    "NN_rep",
    "coulomb_interx",
    "elec_nuclear_E",
    "fukui",
    "nucleophil",
    "electrophil",
)


def _smarts_match_count(pymol, smarts: str) -> float:
    _, pybel = _ob.load()
    pat = pybel.Smarts(smarts)
    return float(len(pat.findall(pymol)))


def _smarts_atom_sets(pymol, smarts: str) -> set[int]:
    _, pybel = _ob.load()
    pat = pybel.Smarts(smarts)
    hits: set[int] = set()
    for match in pat.findall(pymol):
        for idx in match:
            hits.add(int(idx))
    return hits


def _ugt_topology_site(pymol, verts: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    ob, pybel = _ob.load()
    vec = ob.vectorUnsignedInt()
    pymol.OBMol.GetGIDVector(vec)
    ranks = list(vec)
    topology = {a: int(ranks[a - 1]) if a - 1 < len(ranks) else 0 for a in verts}
    site = {a: a for a in verts}
    for smarts in ("[OX1,SX1,F,I,Cl,Br]~[*]", "[OH,SH]~[C,N]", "[H]~[*]"):
        pat = pybel.Smarts(smarts)
        for match in pat.findall(pymol):
            end, base = int(match[0]), int(match[1])
            if end in site and base in site:
                site[end] = site[base]
    return topology, site


def _ugt_group_ids(topologies: list[int]) -> list[int]:
    groups = [0] * len(topologies)
    gid = 0
    while 0 in groups:
        gid += 1
        seed = groups.index(0)
        stack = [seed]
        groups[seed] = gid
        while stack:
            cur = stack.pop()
            for j, top in enumerate(topologies):
                if groups[j] == 0 and top == topologies[cur]:
                    groups[j] = gid
                    stack.append(j)
    return groups


def ugt_inference_rows(rdkit_mol, *, molnum: int = 1) -> list[dict]:
    """Topological rows + legacy boosting/chance/grouping for UGT ONNX."""
    pymol = _ob.from_rdkit_mol(rdkit_mol)
    base_rows = ugt_atom_rows(rdkit_mol)
    if not base_rows:
        return []

    verts = [int(str(r["_index"]).split(".")[-1]) for r in base_rows]
    topology, site = _ugt_topology_site(pymol, verts)
    group_ids = _ugt_group_ids([topology[v] for v in verts])

    grouped: list[dict] = []
    for row, ob_atom, grp in zip(base_rows, verts, group_ids):
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        out["_atom"] = row["_atom"]
        out["_index"] = f"{molnum}.{grp}.{ob_atom}"
        out["_ob_atom"] = ob_atom
        out["_group"] = grp
        grouped.append(out)

    group_counts: dict[int, int] = {}
    for grp in group_ids:
        group_counts[grp] = group_counts.get(grp, 0) + 1

    boost_counts = {
        name: _smarts_match_count(pymol, smarts)
        for name, smarts in _BOOST_SMARTS.items()
    }

    for out in grouped:
        out["N_in_group"] = float(group_counts[out["_group"]])
        out["Weight"] = 1.0
        if out.get("NN_0") == 1.0:
            out["Weight"] = 11.33796
        elif out.get("NO_0") == 1.0:
            out["Weight"] = 5.17997
        elif out.get("NS_0") == 1.0:
            out["Weight"] = 132.29563
        for name, count in boost_counts.items():
            out[name] = count
        for col in _QUANTUM_COLS:
            out.setdefault(col, 0.0)

    chance_flags = {name: _smarts_atom_sets(pymol, smarts) for name, smarts in _CHANCE_SMARTS.items()}
    chances = []
    for out in grouped:
        ob_atom = out["_ob_atom"]
        chance = 0.0
        matched = False
        for name, atoms in chance_flags.items():
            if ob_atom in atoms:
                chance += _HEURISTIC_CHANCE[name]
                matched = True
        if not matched:
            chance += _HEURISTIC_CHANCE["Remaining"]
        chances.append(chance)
    norm = float(sum(chances)) or 1.0
    for out, chance in zip(grouped, chances):
        out["normalized_chance"] = chance / norm

    grouped.sort(key=lambda r: (int(str(r["_index"]).split(".")[0]), int(str(r["_index"]).split(".")[1]), int(str(r["_index"]).split(".")[2])))
    return grouped
