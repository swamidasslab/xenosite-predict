"""Heavy-atom graph helpers used by BondTD/AtomTD (numpy, no pandas).

Port of ``xenosite.finger.graph.UndirectedGraph`` methods BondTD actually calls:
neighbors, pairwise_distance, shortest_path. ``cycles()`` is OpenBabel SSSR;
``dfs_cycles()`` is the legacy DFS back-edge set used by BondTD ``NRings``.
Vertex keys are 1-based OpenBabel atom indices.
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
        """One minimum-length path. Neighbors are visited in sorted index order.

        Legacy code iterated ``set`` neighbors in CPython 2.7 hash order, so
        tied BFS paths were machine-dependent. We do not replicate that.
        """
        if s == e:
            return [s]
        path = {s: [s]}
        q = deque([s])
        seen = {s}
        while q:
            v = q.popleft()
            for w in sorted(self.neighbors[v]):
                if w in seen:
                    continue
                seen.add(w)
                path[w] = path[v] + [w]
                if w == e:
                    return path[w]
                q.append(w)
        return []

    def all_shortest_paths(self, s: int, e: int) -> list[list[int]]:
        """Every minimum-length path (sorted neighbor order).

        Quinone ortho/meta/para is true if **any** of these paths lies on an
        aromatic ring, so the feature does not depend on set iteration order.
        """
        if s == e:
            return [[s]]
        dist = {s: 0}
        parents: dict[int, list[int]] = {s: []}
        q = deque([s])
        while q:
            v = q.popleft()
            dv = dist[v]
            if e in dist and dv >= dist[e]:
                continue
            for w in sorted(self.neighbors[v]):
                if w not in dist:
                    dist[w] = dv + 1
                    parents[w] = [v]
                    q.append(w)
                elif dist[w] == dv + 1:
                    parents[w].append(v)

        if e not in dist:
            return []

        found: list[list[int]] = []

        def rec(node: int) -> None:
            if node == s:
                found.append([s])
                return
            for p in parents[node]:
                before = len(found)
                rec(p)
                for i in range(before, len(found)):
                    found[i] = found[i] + [node]

        rec(e)
        return found

    def dfs(self, v: int | None = None):
        """Port of ``xenosite.finger.graph.UndirectedGraph.DFS`` (sorted neighbors)."""
        ignore: set[int] = set()
        explored: set[int] = set()
        if v is None:
            vs = [n for n in sorted(self.vertex) if n not in ignore]
            v = vs[0]
        visited = {v}
        edge: set[frozenset[int]] = set()
        stack = [v]
        while stack:
            t = stack[-1]
            skip = False
            for n in sorted(self.neighbors[t]):
                e = frozenset((t, n))
                if e in edge:
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

    def dfs_cycles(self) -> list[set[int]]:
        """DFS back-edge cycles used by BondTD ``NRings`` (legacy UndirectedGraph).

        Fusion atoms sit in extra perimeter cycles that SSSR drops; the 2.4 dump
        oracle counts those. ``cycles()`` stays OpenBabel SSSR.
        """
        walk = list(self.dfs())
        cycles: list[set[int]] = []
        back = [n for n, (_a, _b, t) in enumerate(walk) if t == "b"]
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
