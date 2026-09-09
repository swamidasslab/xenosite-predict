"""Phase1 Bond_and_LonePair descriptors. Pandas-free. OpenBabel internal.

Port of ``topological_descriptors.bond_and_lone_pair.Bond_and_LonePairTD``.
Includes hydrogens and lone-pair pseudo-bonds. These 404 numeric columns (409
minus five training-label placeholders) feed ``site.onnx``.

``Possible_Sites`` SMARTS flags live in ``possible_site_flags`` for a future
``ReactionType`` path only: legacy ``model1`` applies them *after* the site net
as multipliers, not as NN inputs. We do not emit them from ``phase1_rows``;
dump comparisons skip them via ``POSSIBLE_SITE_COLUMNS``.
"""

from __future__ import annotations

from collections import Counter, OrderedDict

from . import _ob
from .bond import NOUTER
from .molgraph import MolGraph

ATOM_SYMBOLS = "C N O P S F Cl Br I H".split()
HYB_ORDER = (("sp1", 1), ("sp2", 2), ("sp3", 3))
MOL_DESCS = [
    "atoms", "bonds", "TPSA", "logP", "MW", "MR", "HBD", "HBA1", "HBA2",
    "sbonds", "dbonds", "tbonds", "abonds",
]
TARGET_FIELDS = (
    "StableOxygenation",
    "UnstableOxygenation",
    "Dehydrogenation",
    "Reduction",
    "Hydrolysis",
)
QUINONE = [
    "Quinone_Dehydrogenation",
    "Imine_Dehydrogenation",
    "QuinoneImine_Dehydrogenation",
    "QuinoneMethide_Dehydrogenation",
    "ImineMethide_Dehydrogenation",
]
ESTER = ["Ester_Hydrolysis", "PEster_Hydrolysis", "HalogenEster_Hydrolysis"]
SITE_SMARTS = [
    ("EpoxidationAr_StableOxygenation", "[$([c,n,s,p][c,n,s,p])]"),
    ("EpoxidationAl_StableOxygenation", "[$([C,N,S,P]=[C,N,S,P])]"),
    (
        "S_StableOxygenation",
        "[!$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]);"
        "!$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6]);$([#16])]",
    ),
    ("N_StableOxygenation", "[!$([NX3](=O)=O);!$([NX3+](=O)[O-]);$([#7])]"),
    ("C_StableOxygenation", "[$([#6;H1,H2,H3])]"),
    ("N_UnstableOxygenation", "[#7][#6]"),
    ("O_UnstableOxygenation", "[#8][#6]"),
    ("S_UnstableOxygenation", "[#16][#6]"),
    ("C_UnstableOxygenation", "[#6][#6]"),
    ("Halogen_UnstableOxygenation", "[#9,#17,#35,#53][#6]"),
    (
        "DehydrogenationAl_Dehydrogenation",
        "[$([C;H1,H2,H3]),$([N;H1,H2,H3]),$([O;H1]),$([S;H1])]",
    ),
    (
        "DehydrogenationAr_Dehydrogenation",
        "[$([c;H1,H2,H3]),$([n;H1,H2,H3]),$([o;H1]),$([s;H1])]",
    ),
    ("Quinone_Dehydrogenation", "[$([#8H]c~c~c~c[#8H]),$([#8H]c~c[#8H])]"),
    ("Imine_Dehydrogenation", "[$([#7H]c~c~c~c[#7H]),$([#7H]c~c[#7H])]"),
    (
        "QuinoneImine_Dehydrogenation",
        "[$([#7H]c~c~c~c[#8H]),$([#7H]c~c[#8H]),$([#8H]c~c~c~c[#7H]),$([#8H]c~c[#7H])]",
    ),
    (
        "QuinoneMethide_Dehydrogenation",
        "[$([#6H]c~c~c~c[#8H]),$([#6H]c~c[#8H]),$([#8H]c~c~c~c[#6H]),$([#8H]c~c[#6H])]",
    ),
    (
        "ImineMethide_Dehydrogenation",
        "[$([#6H]c~c~c~c[#7H]),$([#6H]c~c[#7H]),$([#7H]c~c~c~c[#6H]),$([#7H]c~c[#6H])]",
    ),
    (
        "Nitro_Reduction",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-]),$([#7]~[O]),$([#7+]~[O-])]",
    ),
    ("Carbonyl_Reduction", "[$([#6X3]=[OX1]),$([#6X3+]-[OX1-])]"),
    ("Sulfo_Reduction", "[$([#16]~O)]"),
    ("HalogenatedCarbon_Reduction", "[#9,#17,#35,#53][#6]"),
    ("DoubleTripleBondAl_Reduction", "[C,N,O]=,#[C,N,O]"),
    ("DoubleTripleBondAr_Reduction", "[c,n,o][c,n,o]"),
    (
        "Ether_Hydrolysis",
        "[$([OD2]([#6])[#6]);!$([OX2H0]([#6])[CX3](=O))]",
    ),
    (
        "Ester_Hydrolysis",
        "[$([#8X2H0,#16X2H0]([#6,#7,#15])[C,P,S,N](=[#8X1,S])),"
        "$([C,P,S,N](=[#8X1,#16])[#8X2H0,#16X2H0][#6,#7,#15])]",
    ),
    (
        "PEster_Hydrolysis",
        "[$([OX2H0,SX2H0]([#6,#7])P1SO1),$(P1(SO1)[OX2H0,SX2H0][#6,#7])]",
    ),
    (
        "HalogenEster_Hydrolysis",
        "[$([#9,#17,#35,#53][CX3,P,S](=[OX1])),$([CX3,P,S](=[OX1])[#9,#17,#35,#53])]",
    ),
    (
        "Amide_Hydrolysis",
        "[$([#7][CX3,P,S](=[OX1])),$([CX3,P,S](=[OX1])[#7])]",
    ),
]

# Collapsed Possible_Sites column names in the py2 dump (B_desc ‖ PS_desc).
# Not part of site.onnx input; ignored in dump parity tests until ReactionType.
POSSIBLE_SITE_COLUMNS = frozenset(
    {name for name, _ in SITE_SMARTS}
    - set(QUINONE[1:])
    - set(ESTER[1:])
)


def _pt():
    return _ob.element_table()


def _renumber_atoms(obmol, order: list[int]) -> None:
    try:
        obmol.RenumberAtoms(order)
        return
    except TypeError:
        pass
    ob, _ = _ob.load()
    vec = getattr(ob, "vectorUnsignedInt", None) or getattr(ob, "vectorInt")
    v = vec()
    for i in order:
        v.append(i)
    obmol.RenumberAtoms(v)


def phase1_pymol(rdkit_mol):
    """Match ``libridass.phase1.predictor.PyMolPredictor.read`` (add H, heavy then light)."""
    pymol = _ob.from_rdkit_mol(rdkit_mol)
    pymol.removeh()
    pymol.addh()
    pymol.convertdbonds()
    heavy = [x.idx for x in pymol.atoms if not x.OBAtom.IsHydrogen()]
    light = [x.idx for x in pymol.atoms if x.OBAtom.IsHydrogen()]
    _renumber_atoms(pymol.OBMol, heavy + light)
    _ob_mod, pybel = _ob.load()
    return pybel.readstring("sdf", pymol.write("sdf"))


class BondAndLonePairTD:
    def __init__(self, pymol, *, molnum: int = 1) -> None:
        ob, _ = _ob.load()
        self.ob = ob
        self.pymol = pymol
        self.pymol.OBMol.ConvertDativeBonds()
        self.molnum = molnum
        self.title = str(molnum)
        self.bond_prefix = "BondDesc__%s"
        self.atom_prefixes = ["Atom1_", "Atom2_"]
        self.max_depth = 5
        self.broken = False
        self.PT = _pt()
        atoms = list(pymol.atoms)
        if {a.idx for a in atoms} == {1} or any(a.atomicnum == 0 for a in atoms):
            self.broken = True
            self.rows: list[dict] = []
            return
        self._all_bonds_and_atoms()
        self._order_bonds_and_lone_pairs()
        if self.broken:
            self.rows = []
            return
        self._init_rows()
        self._topo_equiv()
        self.MG = MolGraph(pymol, hydrogens=True)
        self.MG.vertex = dict(
            zip(
                sorted(self.MG.vertex.keys()),
                [_pt().GetSymbol(x.atomicnum) for x in self.pymol.atoms],
            )
        )
        self._atom_depths()

    def _all_bonds_and_atoms(self) -> None:
        ob = self.ob
        self.HB = list(ob.OBMolBondIter(self.pymol.OBMol))
        heavy = list(ob.OBMolAtomIter(self.pymol.OBMol))
        self.HA = OrderedDict((x.GetIdx(), x) for x in heavy)

    def _order_bonds_and_lone_pairs(self) -> None:
        self.B_and_LP = []
        self.B_and_LPI = []
        for idx, atom in self.HA.items():
            z = atom.GetAtomicNum()
            if z in (7, 8, 15, 16) or (atom.IsCarbon() and atom.GetHyb() != 3):
                self.B_and_LP.append((atom, atom))
                self.B_and_LPI.append((idx, idx))
        self.HBA = []
        self.HBAI = []
        for bond in self.HB:
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            i1, i2 = a1.GetIdx(), a2.GetIdx()
            if i1 > i2:
                first, second = a2, a1
            else:
                first, second = a1, a2
            if first.IsHydrogen():
                self.broken = True
                return
            self.HBA.append((first, second))
            self.HBAI.append((first.GetIdx(), second.GetIdx()))
            self.B_and_LP.append((first, second))
            self.B_and_LPI.append((first.GetIdx(), second.GetIdx()))
        self.LP_atom1 = [p[0] for p in self.B_and_LP]
        self.LP_atom2 = [p[1] for p in self.B_and_LP]
        self.LPI_atom1 = [p[0] for p in self.B_and_LPI]
        self.LPI_atom2 = [p[1] for p in self.B_and_LPI]
        self.BA_zips = [
            (self.atom_prefixes[0], self.LP_atom1),
            (self.atom_prefixes[1], self.LP_atom2),
        ]
        self.BAI_zips = [
            (self.atom_prefixes[0], self.LPI_atom1),
            (self.atom_prefixes[1], self.LPI_atom2),
        ]

    def _init_rows(self) -> None:
        n = len(self.B_and_LPI)
        self.rows = [{} for _ in range(n)]
        self.index = [f"{self.title}.{x}.{y}" for x, y in self.B_and_LPI]
        for i, (a, b) in enumerate(self.B_and_LPI):
            self.rows[i]["_atoms"] = (a - 1, b - 1)
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
        self.Atom1_neighbor_depths = {
            x: self._atom_indexes_by_depth(x, self.max_depth)
            for x in set(self.LPI_atom1)
        }
        self.Atom2_neighbor_depths = {
            x: self._atom_indexes_by_depth(x, self.max_depth)
            for x in set(self.LPI_atom2)
        }
        self.depth_zips = [
            (self.atom_prefixes[0], self.Atom1_neighbor_depths),
            (self.atom_prefixes[1], self.Atom2_neighbor_depths),
        ]

    def _topo_equiv(self) -> None:
        vec = self.ob.vectorUnsignedInt()
        self.pymol.OBMol.GetGIDVector(vec)
        ranks = list(vec)
        atom_topology = {}
        for idx in self.HA:
            atom_topology[idx] = int(ranks[idx - 1]) if idx - 1 < len(ranks) else 0
        seen: dict[str, int] = {}
        nxt = 0
        self.BT: dict[str, int] = {}
        for ix, pair in zip(self.index, self.B_and_LPI):
            if pair[0] != pair[1]:
                key = ".".join(str(x) for x in sorted(atom_topology[y] for y in pair))
            else:
                key = str(atom_topology[pair[0]])
            if key not in seen:
                nxt += 1
                seen[key] = nxt
            self.BT[ix] = seen[key]

    def _atom_index_for_row(self, label: str, index: str) -> int:
        parts = index.split(".")
        return int(parts[1] if label == "Atom1_" else parts[2])

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

    def add_atom_types_hybridization(self) -> None:
        for depth in range(self.max_depth - 1):
            for atom_symbol in "C N O S ALL".split():
                for name, level in HYB_ORDER:
                    if name == "sp1" and atom_symbol in ("O", "S"):
                        continue
                    c1, c2 = [], []
                    for ix in self.index:
                        a1 = int(ix.split(".")[1])
                        a2 = int(ix.split(".")[2])
                        n1 = self.Atom1_neighbor_depths[a1][depth]
                        n2 = self.Atom2_neighbor_depths[a2][depth]
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
        for label, nd in self.depth_zips:
            for depth in range(self.max_depth):
                for sym in ATOM_SYMBOLS:
                    counts, pcts = [], []
                    for ix in self.index:
                        atom_index = self._atom_index_for_row(label, ix)
                        symbols = [self.MG.vertex[x] for x in nd[atom_index][depth]]
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
            vals = [
                sum(int(b.IsRotor()) for b in self.ob.OBAtomBondIter(a)) for a in atoms
            ]
            self._set(f"{label}Rotors", vals)

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
        a1, a2, ar1, ar2 = [], [], [], []
        for ix in self.index:
            i1, i2 = int(ix.split(".")[1]), int(ix.split(".")[2])
            n1 = self.Atom1_neighbor_depths[i1][1]
            n2 = self.Atom2_neighbor_depths[i2][1]
            a1.append(sum(1 - int(self.HA[idx].IsHydrogen()) for idx in n1))
            a2.append(sum(1 - int(self.HA[idx].IsHydrogen()) for idx in n2))
            ar1.append(sum(int(self.HA[idx].IsAromatic()) for idx in n1))
            ar2.append(sum(int(self.HA[idx].IsAromatic()) for idx in n2))
        self._set("Atom1_NHeavyNbrs", a1)
        self._set("Atom2_NHeavyNbrs", a2)
        self._set("Atom1_NArNbrs", ar1)
        self._set("Atom2_NArNbrs", ar2)

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
            self._set(f"{prefix}MaxInvRingSize", inv)

    def add_nrings(self) -> None:
        cycles = self.MG.dfs_cycles()
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

    def add_lone_pairs_depth(self) -> None:
        for label, nd in self.depth_zips:
            for depth in range(self.max_depth - 1):
                vals = []
                for ix in self.index:
                    atom_index = self._atom_index_for_row(label, ix)
                    vals.append(
                        sum(self.lone_pairs(self.HA[x]) for x in nd[atom_index][depth])
                    )
                self._set(f"{label}Lone_Pair_Depth_{depth}", vals)

    def add_epoxide(self) -> None:
        for label, indexes in self.BAI_zips:
            self._set(
                f"{label}Within_Epoxide",
                self.within_substructure("[#6]1-O-[#6]1", indexes),
            )

    def add_bond_neighbors(self) -> None:
        for label, nd in self.depth_zips:
            for depth in (1, 2):
                single, aromatic, double, triple = [], [], [], []
                for ix in self.index:
                    atom_index = self._atom_index_for_row(label, ix)
                    neighbors = nd[atom_index]
                    start = next(iter(neighbors[0]))
                    ends = neighbors[depth]
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

    def add_aromatic_neighbors(self) -> None:
        for label, nd in self.depth_zips:
            vals = []
            for ix in self.index:
                atom_index = self._atom_index_for_row(label, ix)
                vals.append(
                    sum(int(self.HA[idx].IsAromatic()) for idx in nd[atom_index][1])
                )
            self._set(f"{label}AromaticNeighbors", vals)

    def _organize_atom1_atom2(self) -> None:
        skip = {"_atoms", "_index"}
        cols = [k for k in self.rows[0] if k not in skip] if self.rows else []
        a1 = [c for c in cols if self.atom_prefixes[0] in c]
        a2 = [c for c in cols if self.atom_prefixes[1] in c]
        order = a1 + a2
        for row in self.rows:
            extra = {k: row[k] for k in skip if k in row}
            rebuilt = {k: row[k] for k in order}
            rebuilt.update(extra)
            row.clear()
            row.update(rebuilt)

    def _bond_order_flag(self, index: str, order: str) -> int:
        a1 = int(index.split(".")[1])
        a2 = int(index.split(".")[2])
        if (a1, a2) not in self.HBAI:
            return 0
        bond = self.pymol.OBMol.GetBond(a1, a2)
        if bond is None:
            return 0
        if order == "Aromatic":
            return int(bond.IsAromatic())
        if order == "Single":
            return int(bond.IsSingle())
        if order == "Double":
            return int(bond.IsDouble())
        if order == "Triple":
            return int(bond.IsTriple())
        if order == "Ester":
            return int(bond.IsEster())
        if order == "Amide":
            return int(bond.IsAmide())
        if order == "InRing":
            return int(bond.IsInRing())
        return 0

    def add_bond_descriptors(self) -> None:
        for name in ("Single", "Aromatic", "Double", "Triple", "InRing", "Ester", "Amide"):
            self._set(self.bond_prefix % name, [self._bond_order_flag(ix, name) for ix in self.index])
        counts = Counter(self.BT.values())
        self._set(
            self.bond_prefix % "NTopologicalEquivalent",
            [counts[self.BT[ix]] for ix in self.index],
        )

    def add_is_lone_pair(self) -> None:
        self._set(
            self.bond_prefix % "connecting_to_H",
            [int(a.IsHydrogen()) for a in self.LP_atom2],
        )
        self._set(
            self.bond_prefix % "atom_level",
            [
                int(int(ix.split(".")[1]) == int(ix.split(".")[2]))
                for ix in self.index
            ],
        )

    def add_molecule_descriptors(self) -> None:
        md = self.pymol.calcdesc()
        n = len(self.rows)
        for k in MOL_DESCS:
            self._set(f"MolDesc__{k}", [md[k]] * n)
        self._set("MolDesc__heavy_atoms", [len(self.HA)] * n)
        self._set("MolDesc__hydrogens", [md["atoms"] - len(self.HA)] * n)
        try:
            nr = float(len(self.MG.dfs_cycles()))
        except Exception:
            self.broken = True
            nr = 0.0
        self._set("MolDesc__NumRings", [nr] * n)

    def add_empty_targets(self) -> None:
        n = len(self.rows)
        for field in TARGET_FIELDS:
            self._set(field, [0.0] * n)
            self._set(f"Mol{field}", [0.0] * n)

    def _rewrite_index(self) -> None:
        new_index = []
        for ix in self.index:
            parts = ix.split(".")
            new_index.append(
                f"{self.title}.{self.BT[ix]}.{parts[1]}.{parts[2]}"
            )
        self.index = new_index
        for row, ix in zip(self.rows, self.index):
            row["_index"] = ix

    def run(self) -> list[dict]:
        if self.broken:
            return []
        self.add_atom_types_hybridization()
        self.add_atom_types_and_percentages()
        self.add_partial_charge()
        self.add_rotors()
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
        self.add_lone_pairs_depth()
        self.add_epoxide()
        self.add_bond_neighbors()
        self.add_aromatic_neighbors()
        self._organize_atom1_atom2()
        self.add_bond_descriptors()
        self.add_is_lone_pair()
        self.add_molecule_descriptors()
        self.add_empty_targets()
        self._rewrite_index()
        return self.rows


def _smarts_mol_copy(pymol):
    _, pybel = _ob.load()
    copy = pybel.readstring("sdf", pymol.write("sdf"))
    copy.OBMol.ConvertDativeBonds()
    return copy


def _smarts_hits(pymol, smarts: str) -> set[int]:
    """Match SMARTS on a copy of ``pymol`` without kekulizing (deterministic on OB 3.x)."""
    _, pybel = _ob.load()
    mol = _smarts_mol_copy(pymol)
    pattern = pybel.Smarts(smarts)
    return {int(el) for tup in pattern.findall(mol) for el in tup}


def _all_smarts_hits(pymol) -> dict[str, set[int]]:
    return {name: _smarts_hits(pymol, sm) for name, sm in SITE_SMARTS}


def _c_unstable_bond_ok(bond, a1, a2) -> bool:
    """Filter ``[#6][#6]`` false positives that OB 2.4 skipped after kekulize.

    OB 3.x without kekulize matches some aromatic 5-ring C=C pairs the dump omits;
    exclude those deterministically instead of kekulizing.
    """
    if bond.IsAromatic() and bond.GetBondOrder() == 2:
        in5 = a1.IsInRingSize(5) and a2.IsInRingSize(5)
        in6 = a1.IsInRingSize(6) and a2.IsInRingSize(6)
        if in5 and not in6:
            return False
        if in5 and in6:
            return False
    return True


def possible_site_flags(pymol) -> list[dict]:
    """Possible_Sites SMARTS flags in Bond_and_LonePair row order."""
    ob, _ = _ob.load()
    hits = _all_smarts_hits(pymol)
    ha = OrderedDict((x.GetIdx(), x) for x in ob.OBMolAtomIter(pymol.OBMol))
    hb = list(ob.OBMolBondIter(pymol.OBMol))
    names = [n for n, _ in SITE_SMARTS]
    flags: list[dict] = []
    pairs: list[tuple[int, int]] = []

    def blank():
        return {n: 0.0 for n in names}

    for idx, atom in ha.items():
        sites = blank()
        if atom.GetAtomicNum() in (7, 8, 15, 16):
            for s in ("S_StableOxygenation", "N_StableOxygenation"):
                if idx in hits[s]:
                    sites[s] = 1.0
            flags.append(sites)
            pairs.append((idx, idx))
        elif atom.IsCarbon() and atom.GetHyb() != 3:
            flags.append(sites)
            pairs.append((idx, idx))

    for bond in hb:
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        i1, i2 = a1.GetIdx(), a2.GetIdx()
        sites = blank()
        if i1 in hits["EpoxidationAr_StableOxygenation"] and i2 in hits["EpoxidationAr_StableOxygenation"]:
            sites["EpoxidationAr_StableOxygenation"] = 1.0
        if i1 in hits["EpoxidationAl_StableOxygenation"] and i2 in hits["EpoxidationAl_StableOxygenation"]:
            sites["EpoxidationAl_StableOxygenation"] = 1.0
        for s in ("S_StableOxygenation", "N_StableOxygenation", "C_StableOxygenation"):
            if i1 in hits[s] or i2 in hits[s]:
                if i1 == i2 or a1.IsHydrogen() or a2.IsHydrogen():
                    sites[s] = 1.0
        for s in (
            "N_UnstableOxygenation",
            "O_UnstableOxygenation",
            "S_UnstableOxygenation",
            "C_UnstableOxygenation",
            "Halogen_UnstableOxygenation",
            "HalogenatedCarbon_Reduction",
            "DoubleTripleBondAr_Reduction",
            "DoubleTripleBondAl_Reduction",
            "Ester_Hydrolysis",
            "PEster_Hydrolysis",
            "HalogenEster_Hydrolysis",
            "Amide_Hydrolysis",
        ):
            if (i1 in hits[s]) and (i2 in hits[s]) and (a1.IsCarbon() or a2.IsCarbon()):
                if s == "C_UnstableOxygenation" and not _c_unstable_bond_ok(bond, a1, a2):
                    continue
                sites[s] = 1.0
        for s in (
            "C_UnstableOxygenation",
            "DoubleTripleBondAr_Reduction",
            "DoubleTripleBondAl_Reduction",
        ):
            if (i1 in hits[s]) and (i2 in hits[s]):
                if s != "C_UnstableOxygenation" or _c_unstable_bond_ok(bond, a1, a2):
                    sites[s] = 1.0
        for s in (
            "DehydrogenationAl_Dehydrogenation",
            "DehydrogenationAr_Dehydrogenation",
            "Quinone_Dehydrogenation",
            "QuinoneImine_Dehydrogenation",
            "Imine_Dehydrogenation",
            "QuinoneMethide_Dehydrogenation",
            "ImineMethide_Dehydrogenation",
        ):
            if (i1 in hits[s] or i2 in hits[s]) and (a1.IsHydrogen() or a2.IsHydrogen()):
                sites[s] = 1.0
        for s in ("Nitro_Reduction", "Carbonyl_Reduction", "Sulfo_Reduction"):
            if (i1 in hits[s] or i2 in hits[s]) and (a1.IsOxygen() or a2.IsOxygen()):
                sites[s] = 1.0
        if i1 in hits["Ether_Hydrolysis"] or i2 in hits["Ether_Hydrolysis"]:
            sites["Ether_Hydrolysis"] = 1.0
        flags.append(sites)
        pairs.append(tuple(sorted((i1, i2))))
    return flags


def _collapse_sites(sites: dict) -> dict:
    out = dict(sites)
    out["Quinone_Dehydrogenation"] = max(float(sites[k]) for k in QUINONE)
    out["Ester_Hydrolysis"] = max(float(sites[k]) for k in ESTER)
    for k in QUINONE[1:] + ESTER[1:]:
        out.pop(k, None)
    return out


def phase1_rows(rdkit_mol) -> list[dict]:
    """Bond_and_LonePair rows only (site.onnx features + label placeholders)."""
    pymol = phase1_pymol(rdkit_mol)
    B = BondAndLonePairTD(pymol)
    if B.broken:
        raise ValueError("Bond_and_LonePairTD marked molecule broken")
    rows = B.run()
    if B.broken or not rows:
        raise ValueError("Bond_and_LonePairTD failed run")
    return rows
