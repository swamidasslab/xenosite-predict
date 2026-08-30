"""Heavy-atom graph helpers used by BondTD/AtomTD (numpy, no pandas).

Port of ``xenosite.finger.graph.UndirectedGraph`` methods BondTD actually calls:
neighbors, pairwise_distance, shortest_path, DFS cycles.
Vertex keys are 1-based OpenBabel atom indices.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class MolGraph:
    def __init__(self, pymol) -> None:
        ob, _pybel = _ob_mod()
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

    def _dfs(self):
        ignore: set[int] = set()
        explored: set[int] = set()
        visited: set[int] = set()
        edge: set[frozenset[int]] = set()
        vs = [v for v in sorted(self.vertex) if v not in ignore]
        if not vs:
            return
        start = vs[0]
        visited.add(start)
        stack = [start]
        while stack:
            t = stack[-1]
            skip = False
            for n in sorted(self.neighbors[t]):
                e = frozenset((t, n))
                if e in edge:
                    continue
                if n in ignore:
                    continue
                if n not in visited and n not in explored:
                    edge.add(e)
                    visited.add(n)
                    stack.append(n)
                    yield (t, n, "t")
                    skip = True
                    break
                if n in visited:
                    edge.add(e)
                    yield (t, n, "b")
            if skip:
                continue
            explored.add(t)
            stack.pop()

    def cycles(self) -> list[set[int]]:
        walk = list(self._dfs())
        back = [n for n, (_a, _b, t) in enumerate(walk) if t == "b"]
        cycles: list[set[int]] = []
        for bi in back:
            sv = v = walk[bi][1]
            member = {v}
            i = bi
            while True:
                v = walk[i][0]
                member.add(v)
                while True:
                    i -= 1
                    if i < 0 or (walk[i][1] == v and walk[i][2] != "b"):
                        break
                if v == sv:
                    break
            cycles.append(member)
        return cycles


_PT = None


def _ob_mod():
    from . import _ob

    return _ob.load()


def _pt():
    global _PT
    if _PT is None:
        ob, _p = _ob_mod()
        _PT = ob.OBElementTable()
    return _PT
