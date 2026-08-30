"""Heavy-atom graph helpers used by BondTD/AtomTD (numpy, no pandas).

Port of ``xenosite.finger.graph.UndirectedGraph`` methods BondTD actually calls:
neighbors, pairwise_distance, shortest_path. Rings are OpenBabel SSSR
(not DFS cycles). Vertex keys are 1-based OpenBabel atom indices.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class MolGraph:
    def __init__(self, pymol) -> None:
        ob, _pybel = _ob_mod()
        self.pymol = pymol
        self.vertex: dict[int, str] = {}
        self.neighbors: dict[int, set[int]] = defaultdict(set)
        for a in pymol.atoms:
            if a.atomicnum == 1:
                continue
            i = a.OBAtom.GetIdx()
            self.vertex[i] = _pt().GetSymbol(a.atomicnum)
            self.add_vertex(i)
        for b in ob.OBMolBondIter(pymol.OBMol):
            i = b.GetBeginAtom().GetIdx()
            j = b.GetEndAtom().GetIdx()
            if i not in self.vertex or j not in self.vertex:
                continue
            self.add_edge(i, j)

    def add_vertex(self, idx: int) -> None:
        self.vertex.setdefault(idx, "*")
        self.neighbors.setdefault(idx, set())

    def add_edge(self, i: int, j: int) -> None:
        self.add_vertex(i)
        self.add_vertex(j)
        self.neighbors[i].add(j)
        self.neighbors[j].add(i)

    def pairwise_distance(self) -> tuple[np.ndarray, dict[int, int]]:
        verts = list(self.vertex)
        v2i = {v: n for n, v in enumerate(verts)}
        n = len(verts)
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0.0)
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                dist[v2i[i], v2i[j]] = 1.0
        for k in range(n):
            dist = np.minimum(dist, dist[:, k][np.newaxis, :] + dist[k, :][:, np.newaxis])
        return dist, v2i

    def shortest_path(self, s: int, e: int) -> list[int]:
        if s == e:
            return [s]
        path = {s: [s]}
        q = deque([s])
        seen = {s}
        while q:
            v = q.popleft()
            for w in self.neighbors[v]:
                if w in seen:
                    continue
                seen.add(w)
                path[w] = path[v] + [w]
                if w == e:
                    return path[w]
                q.append(w)
        return []

    def cycles(self) -> list[set[int]]:
        """Smallest set of smallest rings (OpenBabel SSSR), heavy atoms only."""
        obmol = self.pymol.OBMol
        obmol.FindSSSR()
        rings: list[set[int]] = []
        for ring in obmol.GetSSSR():
            member = {idx for idx in self.vertex if ring.IsInRing(idx)}
            if member:
                rings.append(member)
        return rings


_PT = None


def _ob_mod():
    from . import _ob

    return _ob.load()


def _pt():
    global _PT
    if _PT is None:
        from . import _ob

        _PT = _ob.element_table()
    return _PT
