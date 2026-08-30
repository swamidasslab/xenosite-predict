"""OpenBabel BondTD (epoxidation / n-dealk family). Pandas-free list-of-dicts.

Port of ``libridass.epoxidation1.topological_descriptors.bond.BondTD`` and the
ndealk ``bond_desc.BondTD`` ordering/prefix variant. OpenBabel is internal.
"""

from __future__ import annotations

from collections import Counter, OrderedDict

from . import _ob
from .molgraph import MolGraph

NOUTER = {
    1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 1, 20: 2,
    21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 30: 2,
    31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 1, 38: 2, 39: 3, 40: 4,
    41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10, 47: 11, 48: 2, 49: 3, 50: 4,
    51: 5, 52: 6, 53: 7, 54: 8, 55: 1, 56: 2, 57: 3, 58: 4, 59: 3, 60: 4,
    61: 5, 62: 6, 63: 7, 64: 8, 65: 9, 66: 10, 67: 11, 68: 12, 69: 13, 70: 14,
    71: 15, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11, 80: 2,
    81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8, 87: 1, 88: 2, 89: 3, 90: 4,
    91: 3, 92: 4, 93: 5, 94: 6, 95: 7, 96: 8, 97: 9, 98: 10, 99: 11, 100: 12,
    101: 13, 102: 14, 103: 15, 104: 4,
}

HYB = {"sp1": 1, "sp2": 2, "sp3": 3}
ATOM_SYMBOLS = "C N O P S F Cl Br I".split()


def _pt():
    ob, _ = _ob.load()
    return ob.OBElementTable()


class BondTD:
    def __init__(
        self,
        pymol,
        *,
        original_atom_ordering: bool = True,
        overlap: bool = False,
        molnum: int = 1,
        bond_prefix: str = "BondDescriptor__%s",
        atom_order: str = "original",
        ndealk_max_inv_scalar: bool = False,
    ) -> None:
        ob, _ = _ob.load()
        self.ob = ob
        self.pymol = pymol
        self.original_atom_ordering = original_atom_ordering
        self.overlap = overlap
        self.molnum = molnum
        self.bond_prefix = bond_prefix
        self.atom_order = atom_order
        self.ndealk_max_inv_scalar = ndealk_max_inv_scalar
        self.atom_prefixes = ["Atom1_", "Atom2_"]
        self.max_depth = 5
        self.broken = False
        self.PT = _pt()
        atoms = list(pymol.atoms)
        if {a.idx for a in atoms} == {1} or any(a.atomicnum == 0 for a in atoms):
            self.broken = True
            self.rows: list[dict] = []
            return
        self._heavy_bonds_and_atoms()
        self._order_bond_atoms()
        self._init_rows()
        self._topo_equiv()
        self.MG = MolGraph(pymol)
        self._atom_depths()

    def _heavy_bonds_and_atoms(self) -> None:
        ob = self.ob
        self.HB = [
            b
            for b in ob.OBMolBondIter(self.pymol.OBMol)
            if 1 not in (b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum())
        ]
        self.HBI = [1 + b.GetIdx() for b in self.HB]
        heavy = [a for a in ob.OBMolAtomIter(self.pymol.OBMol) if not a.IsHydrogen()]
        self.HA = OrderedDict((a.GetIdx(), a) for a in heavy)

    def _order_bond_atoms(self) -> None:
        self.HBA = []
        self.HBAI = []
        for b in self.HB:
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if self.atom_order == "ndealk":
                z1, z2 = a1.GetAtomicNum(), a2.GetAtomicNum()
                if z1 == 6 and z2 == 7:
                    first, second = a1, a2
                elif z2 == 6 and z1 == 7:
                    first, second = a2, a1
                elif a1.GetIdx() <= a2.GetIdx():
                    first, second = a1, a2
                else:
                    first, second = a2, a1
            elif self.original_atom_ordering:
                first, second = a1, a2
            else:
                first, second = a2, a1
            self.HBA.append((first, second))
            self.HBAI.append((first.GetIdx(), second.GetIdx()))
        self.HBA_atom1 = [p[0] for p in self.HBA]
        self.HBA_atom2 = [p[1] for p in self.HBA]
        self.HBAI_atom1 = [p[0] for p in self.HBAI]
        self.HBAI_atom2 = [p[1] for p in self.HBAI]
        self.BA_zips = [
            (self.atom_prefixes[0], self.HBA_atom1),
            (self.atom_prefixes[1], self.HBA_atom2),
        ]
        self.BAI_zips = [
            (self.atom_prefixes[0], self.HBAI_atom1),
            (self.atom_prefixes[1], self.HBAI_atom2),
        ]

    def _init_rows(self) -> None:
        n = len(self.HB)
        self.rows = [{} for _ in range(n)]
        self.index = [f"{self.molnum}.{x}.{y}" for x, y in self.HBAI]
        for i, (a, b) in enumerate(self.HBAI):
            self.rows[i]["_atoms"] = (_ob.ob_idx_to_rdkit(a), _ob.ob_idx_to_rdkit(b))
            self.rows[i]["_index"] = self.index[i]

    def _set(self, name: str, values) -> None:
        for row, v in zip(self.rows, values):
            try:
                row[name] = float(v)
            except (TypeError, ValueError):
                row[name] = 0.0

    def _atom_indexes_by_depth(self, start, depth, ignore=()) -> list[set[int]]:
        ignore_s = set(ignore)
        current = {start}
        seen: set[int] = set()
        out = [set(current)]
        for _ in range(depth):
            nxt: set[int] = set()
            for c in current:
                nxt |= self.MG.neighbors[c] - (seen | ignore_s | current)
            seen |= current
            current = nxt
            out.append(current)
        return out

    def _atom_depths(self) -> None:
        if self.overlap:
            a1 = [self._atom_indexes_by_depth(x, self.max_depth) for x, y in self.HBAI]
            a2 = [self._atom_indexes_by_depth(y, self.max_depth) for x, y in self.HBAI]
        else:
            a1 = [self._atom_indexes_by_depth(x, self.max_depth, [y]) for x, y in self.HBAI]
            a2 = [self._atom_indexes_by_depth(y, self.max_depth, [x]) for x, y in self.HBAI]
        self.Atom1_ex = dict(zip(self.HBI, a1))
        self.Atom2_ex = dict(zip(self.HBI, a2))
        self.nb_zips = [
            (self.atom_prefixes[0], self.Atom1_ex),
            (self.atom_prefixes[1], self.Atom2_ex),
        ]

    def _topo_equiv(self) -> None:
        ob = self.ob
        vec = ob.vectorUnsignedInt()
        self.pymol.OBMol.GetGIDVector(vec)
        ranks = list(vec)
        # GID vector is 0-based array indexed by atom idx-1
        atom_topology = {}
        for idx in self.HA:
            atom_topology[idx] = int(ranks[idx - 1]) if idx - 1 < len(ranks) else 0
        seen: dict[str, int] = {}
        nxt = 0
        self.BT: dict[str, int] = {}
        for ix, pair in zip(self.index, self.HBAI):
            key = ".".join(str(atom_topology[y]) for y in sorted(pair))
            if key not in seen:
                nxt += 1
                seen[key] = nxt
            self.BT[ix] = seen[key]

    def lone_pairs(self, atom) -> int:
        b = sum(bond.GetBondOrder() for bond in self.ob.OBAtomBondIter(atom))
        v = NOUTER.get(atom.GetAtomicNum(), 0)
        return v - b - atom.GetFormalCharge()

    def within_substructure(self, smarts: str, indexes, atom_level: bool = True):
        sp = self.ob.OBSmartsPattern()
        sp.Init(smarts)
        sp.Match(self.pymol.OBMol)
        maps = list(sp.GetMapList())
        if atom_level:
            hits = {x for y in maps for x in y}
            return [int(idx in hits) for idx in indexes]
        sets = [set(y) for y in maps]
        return [int(set(x) in sets) for x in indexes]

    def add_possible_site_of_n_dealk(self) -> None:
        self._set(
            self.bond_prefix % "Possible_Site_of_N_Dealkylation",
            self.within_substructure("[#6]-[#7]", self.HBAI, atom_level=False),
        )

    def add_atom_types_hybridization(self) -> None:
        for depth in range(self.max_depth - 1):
            for atom_symbol in "C N O S ALL".split():
                for name, level in HYB.items():
                    if name == "sp1" and atom_symbol in ("O", "S"):
                        continue
                    c1, c2 = [], []
                    for bi in self.HBI:
                        n1 = self.Atom1_ex[bi][depth]
                        n2 = self.Atom2_ex[bi][depth]
                        lab = ""
                        if atom_symbol != "ALL":
                            n1 = {x for x in n1 if self.MG.vertex.get(x) == atom_symbol}
                            n2 = {x for x in n2 if self.MG.vertex.get(x) == atom_symbol}
                            lab = "N" + atom_symbol + "_"
                        h1 = [self.HA[x].GetHyb() for x in n1]
                        h2 = [self.HA[x].GetHyb() for x in n2]
                        c1.append(h1.count(level))
                        c2.append(h2.count(level))
                    self._set(f"{self.atom_prefixes[0]}{lab}{name}_{depth}", c1)
                    self._set(f"{self.atom_prefixes[1]}{lab}{name}_{depth}", c2)

    def add_atom_types_and_percentages(self) -> None:
        for label, nd in self.nb_zips:
            for depth in range(self.max_depth):
                for sym in ATOM_SYMBOLS:
                    counts, pcts = [], []
                    for bi in self.HBI:
                        symbols = [self.MG.vertex[x] for x in nd[bi][depth]]
                        c = symbols.count(sym)
                        counts.append(c)
                        pcts.append(0.0 if not c else float(c) / len(symbols))
                    self._set(f"{label}N{sym}_{depth}", counts)
                    if depth != 0:
                        self._set(f"{label}P{sym}_{depth}", pcts)

    def add_partial_charge(self) -> None:
        for label, atoms in self.BA_zips:
            self._set(f"{label}PartialCharge", [a.GetPartialCharge() for a in atoms])

    def add_rotors(self) -> None:
        for label, atoms in self.BA_zips:
            vals = []
            for a in atoms:
                vals.append(sum(int(b.IsRotor()) for b in self.ob.OBAtomBondIter(a)))
            self._set(f"{label}Rotors", vals)

    def add_molecule_descriptors(self) -> None:
        md = self.pymol.calcdesc()
        names = [
            "atoms", "bonds", "TPSA", "logP", "MW", "MR", "HBD", "HBA1", "HBA2",
            "sbonds", "dbonds", "tbonds", "abonds",
        ]
        n = len(self.rows)
        for k in names:
            self._set(f"MolDesc__{k}", [md[k]] * n)
        self._set("MolDesc__heavy_atoms", [len(self.HA)] * n)
        self._set("MolDesc__hydrogens", [md["atoms"] - len(self.HA)] * n)
        try:
            nr = float(len(self.MG.cycles()))
        except Exception:
            self.broken = True
            nr = 0.0
        self._set("MolDesc__NumRings", [nr] * n)

    def add_pyatom_motifs(self) -> None:
        for label, atoms in self.BA_zips:
            self._set(f"{label}AlphaBetaUnsat", [int(a.HasAlphaBetaUnsat()) for a in atoms])
            self._set(f"{label}CarboxylOxygen", [int(a.IsCarboxylOxygen()) for a in atoms])
            self._set(f"{label}SulfateOxygen", [int(a.IsSulfateOxygen()) for a in atoms])
            self._set(f"{label}PhosphateOxygen", [int(a.IsPhosphateOxygen()) for a in atoms])
            self._set(f"{label}NitroOxygen", [int(a.IsNitroOxygen()) for a in atoms])
            self._set(f"{label}AmideNitrogen", [int(a.IsAmideNitrogen()) for a in atoms])

    def add_periodic_table(self) -> None:
        pt = self.PT
        for label, atoms in self.BA_zips:
            z = [a.GetAtomicNum() for a in atoms]
            hyb = [a.GetHyb() for a in atoms]
            self._set(f"{label}PT__ElectronNeg", [pt.GetElectroNeg(i) for i in z])
            self._set(f"{label}PT__ElectronAffinity", [pt.GetElectronAffinity(i) for i in z])
            self._set(f"{label}PT__MaxBonds", [pt.GetMaxBonds(i) for i in z])
            self._set(f"{label}PT__Ionization", [pt.GetIonization(i) for i in z])
            self._set(f"{label}PT__Mass", [pt.GetMass(i) for i in z])
            self._set(
                f"{label}PT__CorrectedBondRad",
                [pt.CorrectedBondRad(i, h) for i, h in zip(z, hyb)],
            )
            self._set(
                f"{label}PT__CorrectedVdwRad",
                [pt.CorrectedVdwRad(i, h) for i, h in zip(z, hyb)],
            )

    def add_aromatic(self) -> None:
        for label, atoms in self.BA_zips:
            self._set(f"{label}Aromatic", [int(a.IsAromatic()) for a in atoms])

    def add_hbond(self) -> None:
        for label, atoms in self.BA_zips:
            self._set(f"{label}HbondAcceptor", [int(a.IsHbondAcceptor()) for a in atoms])
            self._set(f"{label}HbondDonor", [int(a.IsHbondDonor()) for a in atoms])

    def add_heavy_neighbors(self) -> None:
        a1, a2 = [], []
        for bi in self.HBI:
            a1.append(sum(int(self.HA[i].IsAromatic()) for i in self.Atom1_ex[bi][1]))
            a2.append(sum(int(self.HA[i].IsAromatic()) for i in self.Atom2_ex[bi][1]))
        self._set("Atom1_NHeavyNbrs", a1)
        self._set("Atom2_NHeavyNbrs", a2)

    def add_bond_order(self) -> None:
        for label, atoms in self.BA_zips:
            self._set(
                f"{label}TotalBondOrder",
                [sum(b.GetBondOrder() for b in self.ob.OBAtomBondIter(a)) for a in atoms],
            )

    def add_hydrogens(self) -> None:
        for label, atoms in self.BA_zips:
            self._set(
                f"{label}NHydrogens",
                [self.ob.OBAtom.ExplicitHydrogenCount(a) for a in atoms],
            )

    def add_ring_sizes(self) -> None:
        for size in range(3, 9):
            for label, atoms in self.BA_zips:
                self._set(f"{label}Ring{size}", [int(a.IsInRingSize(size)) for a in atoms])
        for prefix in self.atom_prefixes:
            inv = []
            for row in self.rows:
                best = 0.0
                for size in range(3, 9):
                    if row.get(f"{prefix}Ring{size}"):
                        best = max(best, 1.0 / size)
                inv.append(best)
            if self.ndealk_max_inv_scalar and inv:
                inv = [max(inv)] * len(inv)
            self._set(f"{prefix}MaxInvRingSize", inv)

    def add_nrings(self) -> None:
        cycles = self.MG.cycles()
        for label, indexes in self.BAI_zips:
            self._set(
                f"{label}NRings",
                [sum(1 for cyc in cycles if idx in cyc) for idx in indexes],
            )

    def add_span(self) -> None:
        dist, distmap = self.MG.pairwise_distance()
        dmax = dist.max(axis=1)
        min_d, max_d = float(dmax.min()), float(dmax.max())
        span = max(float(max_d - min_d), 1.0)
        for label, indexes in self.BAI_zips:
            s, inv, norm = [], [], []
            for idx in indexes:
                v = float(dmax[distmap[idx]] - min_d)
                s.append(v)
                inv.append(1.0 / (v + 1))
                norm.append(v / span)
            self._set(f"{label}Span", s)
            self._set(f"{label}InvSpan", inv)
            self._set(f"{label}NormSpan", norm)

    def add_bond_descriptors(self) -> None:
        self._set(self.bond_prefix % "Single", [int(b.IsSingle()) for b in self.HB])
        self._set(self.bond_prefix % "Aromatic", [int(b.IsAromatic()) for b in self.HB])
        self._set(self.bond_prefix % "Double", [int(b.IsDouble()) for b in self.HB])
        self._set(self.bond_prefix % "Triple", [int(b.IsTriple()) for b in self.HB])
        counts = Counter(self.BT.values())
        self._set(
            self.bond_prefix % "NTopologicalEquivalent",
            [counts[self.BT[ix]] for ix in self.index],
        )

    def add_aromatic_neighbors(self) -> None:
        for label, nd in self.nb_zips:
            self._set(
                f"{label}AromaticNeighbors",
                [
                    sum(int(self.HA[i].IsAromatic()) for i in nd[bi][1])
                    for bi in self.HBI
                ],
            )

    def add_bond_neighbors(self) -> None:
        for label, nd in self.nb_zips:
            for depth in (1, 2):
                single, aromatic, double, triple = [], [], [], []
                for bi in self.HBI:
                    start = next(iter(nd[bi][0]))
                    ends = nd[bi][depth]
                    bonds = []
                    for end in ends:
                        path = self.MG.shortest_path(start, end)
                        if len(path) < 2:
                            continue
                        bonds.append(self.pymol.OBMol.GetBond(path[-2], path[-1]))
                    single.append(sum(1 for b in bonds if b and b.IsSingle()))
                    aromatic.append(sum(1 for b in bonds if b and b.IsAromatic()))
                    double.append(sum(1 for b in bonds if b and b.IsDouble()))
                    triple.append(sum(1 for b in bonds if b and b.IsTriple()))
                self._set(f"{label}BN_single_{depth}", single)
                self._set(f"{label}BN_aromatic_{depth}", aromatic)
                self._set(f"{label}BN_double_{depth}", double)
                self._set(f"{label}BN_triple_{depth}", triple)

    def add_lone_pairs_depth(self) -> None:
        for label, nd in self.nb_zips:
            for depth in range(self.max_depth - 1):
                self._set(
                    f"{label}Lone_Pair_Depth_{depth}",
                    [
                        sum(self.lone_pairs(self.HA[x]) for x in nd[bi][depth])
                        for bi in self.HBI
                    ],
                )

    def add_epoxide(self) -> None:
        for label, indexes in self.BAI_zips:
            self._set(
                f"{label}Within_Epoxide",
                self.within_substructure("[#6]1-O-[#6]1", indexes),
            )

    def run(self) -> list[dict]:
        if self.broken:
            return []
        self.add_possible_site_of_n_dealk()
        self.add_atom_types_hybridization()
        self.add_atom_types_and_percentages()
        self.add_partial_charge()
        self.add_rotors()
        self.add_molecule_descriptors()
        self.add_pyatom_motifs()
        self.add_periodic_table()
        self.add_aromatic()
        self.add_hbond()
        self.add_heavy_neighbors()
        self.add_bond_order()
        self.add_hydrogens()
        self.add_ring_sizes()
        self.add_nrings()
        self.add_span()
        self.add_bond_descriptors()
        self.add_aromatic_neighbors()
        self.add_bond_neighbors()
        self.add_lone_pairs_depth()
        self.add_epoxide()
        return self.rows


def bond_rows(rdkit_mol, *, original_atom_ordering: bool = True, **kwargs) -> list[dict]:
    pymol = _ob.from_rdkit_mol(rdkit_mol)
    return BondTD(
        pymol, original_atom_ordering=original_atom_ordering, **kwargs
    ).run()
