"""OpenBabel AtomTD (quinone / reactivity). Pandas-free list-of-dicts.

Port of ``libridass.quinone1.topological_descriptors.atom.AtomTD`` and the
reactivity copy (reduced set, no EDG/EWG/OMP). OpenBabel is internal.
"""

from __future__ import annotations

from collections import OrderedDict

from . import _ob
from .bond import ATOM_SYMBOLS, HYB, NOUTER
from .molgraph import MolGraph

EDG = {
    "Phenoxide": "[$([OX1-]a)]",
    "Amine": "[$([NX3;H2,H1,H0]a)]",
    "Phenol": "[$([OX2H]a)]",
    "Ether": "[$([OX2]([#6])a)]",
    "Amides": "[$([NX3H](C=O)a)]",
    "Benzyl_Ester": "[$([OX2]([CX2](=O)[!H])a)]",
    "Alkyl": "[$([CX3;H1,H0](C)a)]",
    "Phenyl": "[$(c1ccccc1a)]",
    "Vinyl": "[$([CX3H](=C)a)]",
}

EWG = {
    "Nitro": "[$([NX3](=O)(=O)a),$([NX3+](=O)([O-])a)]",
    "Quaternary_Amine": "[$([NX4+]([#6])([#6])([#6])a)]",
    "Amonium": "[$([NX4+]([#1])([#1])([#1])a)]",
    "Sulfonate": "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])a)]",
    "Cyano": "[$([CX2](#N)a)]",
    "Acyl_Chloride": "[$([CX3](=O)(Cl)a)]",
    "Carboxylic_Acid": "[$([CX3](=O)([OX2H])a),$([CX3](=O)([OX2-])a)]",
    "Benzoate_Ester": "[$([CX3](=O)([OX2!H])a)]",
    "Ketone": "[$([CX3](=O)([#6])a)]",
    "Aldehyde": "[$([CX3H1](=O)a)]",
}

OMP = {"Ortho": 3, "Meta": 4, "Para": 5}
MOL_DESC = [
    "bonds", "TPSA", "logP", "MW", "MR", "HBD", "HBA1", "HBA2",
    "sbonds", "dbonds", "tbonds", "abonds",
]


class AtomTD:
    def __init__(
        self,
        pymol,
        *,
        molnum: int = 1,
        reduced_descriptor_set: bool = False,
        add_possible_site: bool = False,
        max_depth: int = 5,
    ) -> None:
        self.pymol = pymol
        self.molnum = molnum
        self.reduced_descriptor_set = reduced_descriptor_set
        self.add_possible_site = add_possible_site
        self.max_depth = max_depth
        self.broken = False
        atoms = list(pymol.atoms)
        if {a.idx for a in atoms} == {1} or any(a.atomicnum == 0 for a in atoms):
            self.broken = True
            self.rows: list[dict] = []
            return
        ob, _ = _ob.load()
        self.ob = ob
        self.HA = OrderedDict(
            (a.GetIdx(), a)
            for a in ob.OBMolAtomIter(pymol.OBMol)
            if not a.IsHydrogen()
        )
        self.MG = MolGraph(pymol)
        pt = _ob.element_table()
        self.MG.vertex = dict(
            zip(
                sorted(self.MG.vertex.keys()),
                [pt.GetSymbol(x.GetAtomicNum()) for x in self.HA.values()],
            )
        )
        self.NI = OrderedDict(
            (idx, self._atom_indexes_by_depth(idx, self.max_depth)) for idx in self.HA
        )
        self.NA = OrderedDict(
            (idx, [[self.HA[z] for z in layer] for layer in layers])
            for idx, layers in self.NI.items()
        )
        self.rings = list(pymol.sssr)
        sssr = list(pymol.OBMol.GetSSSR())
        self.aromatic_rings = [
            {x for x in self.HA if r.IsInRing(x)} for r in sssr if r.IsAromatic()
        ]
        self.aromatic_6_rings = [
            {x for x in self.HA if r.IsInRing(x)}
            for r in sssr
            if r.IsAromatic() and r.Size() == 6
        ]
        self._init_rows()

    def _init_rows(self) -> None:
        ob = self.ob
        vec = ob.vectorUnsignedInt()
        self.pymol.OBMol.GetGIDVector(vec)
        ranks = list(vec)
        seen: dict[int, int] = {}
        nxt = 0
        self.rows = []
        for idx in self.HA:
            raw = int(ranks[idx - 1]) if idx - 1 < len(ranks) else 0
            if raw not in seen:
                nxt += 1
                seen[raw] = nxt
            group = seen[raw]
            self.rows.append(
                {
                    "_atom": _ob.ob_idx_to_rdkit(idx),
                    "_index": f"{self.molnum}.{group}.{idx}",
                }
            )

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

    def within_substructure(self, smarts: str) -> list[int]:
        sp = self.ob.OBSmartsPattern()
        sp.Init(smarts)
        sp.Match(self.pymol.OBMol)
        hits = {x for y in sp.GetMapList() for x in y}
        return [int(idx in hits) for idx in self.HA]

    def lone_pairs(self, atom) -> float:
        if atom.GetAtomicNum() == 0:
            return float("nan")
        b = sum(bond.GetBondOrder() for bond in self.ob.OBAtomBondIter(atom))
        return NOUTER.get(atom.GetAtomicNum(), 0) - b - atom.GetFormalCharge()

    def add_ring_sizes(self) -> None:
        for size in range(3, 9):
            if size == 8:
                of_size = [r for r in self.rings if r.Size() >= size]
            else:
                of_size = [r for r in self.rings if r.Size() == size]
            self._set(
                f"Ring{size}",
                [sum(1 for r in of_size if r.IsInRing(idx)) for idx in self.HA],
            )
        inv = []
        for row in self.rows:
            best = 0.0
            for size in range(3, 9):
                if row.get(f"Ring{size}"):
                    best = max(best, 1.0 / size)
            inv.append(best)
        self._set("MaxInvRingSize", inv)
        self._set(
            "NRings",
            [sum(1 for r in self.rings if r.IsInRing(idx)) for idx in self.HA],
        )

    def add_atom_types_and_percentages(self) -> None:
        for depth in range(self.max_depth):
            for sym in ATOM_SYMBOLS:
                counts, pcts = [], []
                for layers in self.NI.values():
                    symbols = [self.MG.vertex[x] for x in layers[depth]]
                    c = symbols.count(sym)
                    counts.append(c)
                    pcts.append(0.0 if not c else float(c) / len(symbols))
                self._set(f"N{sym}_{depth}", counts)
                if depth != 0:
                    self._set(f"P{sym}_{depth}", pcts)

    def add_span(self) -> None:
        dist, distmap = self.MG.pairwise_distance()
        dmax = dist.max(axis=1)
        min_d, max_d = float(dmax.min()), float(dmax.max())
        span = max(float(max_d - min_d), 1.0)
        s, inv, norm = [], [], []
        for idx in self.HA:
            v = float(dmax[distmap[idx]] - min_d)
            s.append(v)
            inv.append(1.0 / (v + 1))
            norm.append(v / span)
        self._set("Span", s)
        self._set("InvSpan", inv)
        self._set("NormSpan", norm)

    def add_atom_types_hybridization(self) -> None:
        for depth in range(self.max_depth):
            for name, level in HYB.items():
                for atom_symbol in "C N O S ALL".split():
                    if name == "sp1" and atom_symbol in ("O", "S"):
                        continue
                    counts = []
                    for layers in self.NI.values():
                        nset = layers[depth]
                        lab = ""
                        if atom_symbol != "ALL":
                            nset = {x for x in nset if self.MG.vertex.get(x) == atom_symbol}
                            lab = "N" + atom_symbol + "_"
                        counts.append(
                            [self.HA[x].GetHyb() for x in nset].count(level)
                        )
                    self._set(f"{lab}{name}_{depth}", counts)

    def add_rotors(self) -> None:
        self._set(
            "Rotors",
            [
                sum(int(b.IsRotor()) for b in self.ob.OBAtomBondIter(a))
                for a in self.HA.values()
            ],
        )

    def add_partial_charge(self) -> None:
        self._set("PartialCharge", [a.GetPartialCharge() for a in self.HA.values()])

    def add_aromatic(self) -> None:
        self._set("Aromatic", [int(a.IsAromatic()) for a in self.HA.values()])

    def add_aromatic_neighbors(self) -> None:
        self._set(
            "AromaticNeighbors",
            [sum(int(y.IsAromatic()) for y in layer[1]) for layer in self.NA.values()],
        )

    def add_hydrogens(self) -> None:
        self._set(
            "NHydrogens",
            [
                a.ExplicitHydrogenCount() + a.ImplicitHydrogenCount()
                for a in self.HA.values()
            ],
        )

    def add_molecule_descriptors(self) -> None:
        md = self.pymol.calcdesc()
        n = len(self.rows)
        for k in MOL_DESC:
            self._set(f"MolDesc__{k}", [md.get(k, 0)] * n)
        self._set("MolDesc__heavy_atoms", [len(self.HA)] * n)
        self._set("MolDesc__hydrogens", [md.get("atoms", 0) - len(self.HA)] * n)
        try:
            nr = float(len(self.MG.cycles()))
        except Exception:
            self.broken = True
            nr = 0.0
        self._set("MolDesc__NumRings", [nr] * n)

    def add_epoxide(self) -> None:
        self._set("Within_Epoxide", self.within_substructure("[#6]1-O-[#6]1"))

    def add_within_substituted_michael_acceptor(self) -> None:
        self._set(
            "Within_Substituted_Michael_Acceptor",
            self.within_substructure("O=[#6][#6](=[#6])[#6]"),
        )

    def add_within_michael_acceptor(self) -> None:
        self._set("Within_Michael_Acceptor", self.within_substructure("O=[#6][#6](=[#6])"))

    def add_bond_neighbors(self) -> None:
        for depth in range(1, 3):
            single, aromatic, double, triple = [], [], [], []
            for layers in self.NI.values():
                start = next(iter(layers[0]))
                bonds = []
                for end in layers[depth]:
                    path = self.MG.shortest_path(start, end)
                    if len(path) < 2:
                        continue
                    bonds.append(self.pymol.OBMol.GetBond(path[-2], path[-1]))
                single.append(sum(1 for b in bonds if b and b.IsSingle()))
                aromatic.append(sum(1 for b in bonds if b and b.IsAromatic()))
                double.append(sum(1 for b in bonds if b and b.IsDouble()))
                triple.append(sum(1 for b in bonds if b and b.IsTriple()))
            self._set(f"BN_single_{depth}", single)
            self._set(f"BN_aromatic_{depth}", aromatic)
            self._set(f"BN_double_{depth}", double)
            self._set(f"BN_triple_{depth}", triple)

    def add_lone_pairs_depth(self) -> None:
        for depth in range(self.max_depth):
            self._set(
                f"Lone_Pair_Depth_{depth}",
                [
                    sum(self.lone_pairs(a) for a in layers[depth])
                    for layers in self.NA.values()
                ],
            )

    def _omp_paths(self, ends_for_start, *, site: bool) -> list[int]:
        add = []
        for paths in ends_for_start:
            if site:
                hit = any(
                    any(
                        set(path[:-1]).issubset(ring)
                        for ring in self.aromatic_rings
                        if path and path[-1] not in ring
                    )
                    for path in paths
                )
            else:
                hit = any(
                    any(
                        set(path[1:-1]).issubset(ring)
                        for ring in self.aromatic_rings
                        if path
                        and path[0] not in ring
                        and path[-1] not in ring
                    )
                    for path in paths
                )
            add.append(int(hit))
        return add

    def ortho_meta_para_to_atoms(self, prefix: str, *, site: bool = False) -> None:
        atoms = "C N O S F Cl Br I".split()
        for label, depth in OMP.items():
            if site:
                depth -= 1
            at_depth = [(idx, layers[depth]) for idx, layers in self.NI.items()]
            for sym in atoms:
                typed = [
                    (idx, [y for y in nbrs if self.MG.vertex.get(y) == sym])
                    for idx, nbrs in at_depth
                ]
                paths = [
                    [self.MG.shortest_path(start, end) for end in ends]
                    for start, ends in typed
                ]
                self._set(prefix % (label, sym), self._omp_paths(paths, site=site))

    def ortho_meta_para_to_motifs(
        self, prefix: str, motif_dict: dict[str, str], *, site: bool = False
    ) -> None:
        for name, smarts in motif_dict.items():
            flags = self.within_substructure(smarts)
            matches = {idx for idx, hit in enumerate(flags, start=1) if hit}
            for position, depth in OMP.items():
                if site:
                    depth -= 1
                at_depth = [(idx, layers[depth]) for idx, layers in self.NI.items()]
                typed = [
                    (idx, [x for x in nbrs if x in matches]) for idx, nbrs in at_depth
                ]
                paths = [
                    [self.MG.shortest_path(start, end) for end in ends]
                    for start, ends in typed
                ]
                self._set(prefix % (position, name), self._omp_paths(paths, site=site))

    def add_ortho_meta_para_to_motifs(self) -> None:
        self.ortho_meta_para_to_motifs("%s_To_EDG_%s", EDG)
        self.ortho_meta_para_to_motifs("Site_%s_To_EDG_%s", EDG, site=True)
        self.ortho_meta_para_to_motifs("%s_To_EWG_%s", EWG)
        self.ortho_meta_para_to_motifs("Site_%s_To_EWG_%s", EWG, site=True)

    def add_ortho_meta_para_to_atoms(self) -> None:
        self.ortho_meta_para_to_atoms("%s_To_%s")
        self.ortho_meta_para_to_atoms("Site_%s_To_%s", site=True)

    def add_within_electron_donating_groups(self) -> None:
        for name, smarts in EDG.items():
            self._set(f"Within_EDG_{name}", self.within_substructure(smarts))

    def add_within_electron_withdrawing_groups(self) -> None:
        for name, smarts in EWG.items():
            self._set(f"Within_EWG_{name}", self.within_substructure(smarts))

    def add_bond_order(self) -> None:
        self._set(
            "TotalBondOrder",
            [
                sum(b.GetBondOrder() for b in self.ob.OBAtomBondIter(a))
                for a in self.HA.values()
            ],
        )

    def add_pyatom_motifs(self) -> None:
        atoms = list(self.HA.values())
        self._set("AlphaBetaUnsat", [int(a.HasAlphaBetaUnsat()) for a in atoms])
        self._set("CarboxylOxygen", [int(a.IsCarboxylOxygen()) for a in atoms])
        self._set("SulfateOxygen", [int(a.IsSulfateOxygen()) for a in atoms])
        self._set("PhosphateOxygen", [int(a.IsPhosphateOxygen()) for a in atoms])
        self._set("NitroOxygen", [int(a.IsNitroOxygen()) for a in atoms])
        self._set("AmideNitrogen", [int(a.IsAmideNitrogen()) for a in atoms])

    def add_hbond(self) -> None:
        atoms = list(self.HA.values())
        self._set("HbondAcceptor", [int(a.IsHbondAcceptor()) for a in atoms])
        self._set("HbondDonor", [int(a.IsHbondDonor()) for a in atoms])

    def add_periodic_table_descriptors(self) -> None:
        pt = _ob.element_table()
        atoms = list(self.HA.values())
        z = [a.GetAtomicNum() for a in atoms]
        hyb = [a.GetHyb() for a in atoms]
        self._set("PT__ElectronNeg", [pt.GetElectroNeg(i) for i in z])
        self._set("PT__ElectronAffinity", [pt.GetElectronAffinity(i) for i in z])
        self._set("PT__MaxBonds", [pt.GetMaxBonds(i) for i in z])
        self._set("PT__Ionization", [pt.GetIonization(i) for i in z])
        self._set("PT__Mass", [pt.GetMass(i) for i in z])
        self._set("PT__CorrectedBondRad", [pt.CorrectedBondRad(i, h) for i, h in zip(z, hyb)])
        self._set("PT__CorrectedVdwRad", [pt.CorrectedVdwRad(i, h) for i, h in zip(z, hyb)])

    def add_possible_site_of_quinone_formation(self) -> None:
        adjacent = [
            self.HA[y]
            for y, layers in self.NI.items()
            if any(
                (layers[1] & ring) and y not in ring for ring in self.aromatic_6_rings
            )
        ]
        just_single = [a for a in adjacent if not a.HasNonSingleBond()]
        adj_c = [
            a
            for a in just_single
            if any(
                n.IsAromatic() and n.IsCarbon() for n in self.ob.OBAtomAtomIter(a)
            )
        ]
        correct = [a for a in adj_c if a.GetAtomicNum() in (6, 7, 8, 16)]
        candidates = [a.GetIdx() for a in correct]
        possible: list[int] = []
        for cand in candidates:
            layers = self.NI[cand]
            right = [
                x
                for d in (3, 5, 7, 9, 11)
                if d < len(layers)
                for x in layers[d]
                if self.HA[x] in adjacent
            ]
            if candidates and right:
                possible.append(cand)
        self._set(
            "Possible_Site_of_Quinone_Formation",
            [int(idx in possible) for idx in self.HA],
        )

    def run(self) -> list[dict]:
        if self.broken:
            return []
        if self.add_possible_site:
            self.add_possible_site_of_quinone_formation()
        self.add_molecule_descriptors()
        self.add_ring_sizes()
        self.add_atom_types_and_percentages()
        self.add_atom_types_hybridization()
        self.add_span()
        self.add_rotors()
        self.add_pyatom_motifs()
        self.add_periodic_table_descriptors()
        self.add_lone_pairs_depth()
        self.add_bond_order()
        self.add_bond_neighbors()
        self.add_epoxide()
        self.add_within_substituted_michael_acceptor()
        self.add_within_michael_acceptor()
        self.add_aromatic()
        self.add_aromatic_neighbors()
        self.add_hydrogens()
        self.add_hbond()
        self.add_partial_charge()
        if not self.reduced_descriptor_set:
            self.add_within_electron_withdrawing_groups()
            self.add_within_electron_donating_groups()
            self.add_ortho_meta_para_to_motifs()
            self.add_ortho_meta_para_to_atoms()
        return self.rows


def quinone_atom_rows(rdkit_mol, **kwargs) -> list[dict]:
    pymol = _ob.from_rdkit_mol(rdkit_mol)
    return AtomTD(pymol, **kwargs).run()


def reactivity_atom_rows(rdkit_mol, **kwargs) -> list[dict]:
    pymol = _ob.from_rdkit_mol(rdkit_mol)
    return AtomTD(pymol, reduced_descriptor_set=True, **kwargs).run()
