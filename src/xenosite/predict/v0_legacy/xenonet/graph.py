"""XenoNet ``XenoDigraph``: multi-edges, path trim, metabolite likelihoods.

Likelihoods clone ``compute_metabolite_likelihoods(mode='max')``: root = 1.0,
multi-edges parent→child take max, drop 1-step cycles, then
``child = sum_parents parent_like * (w / sum_outgoing(parent))``.
"""

from __future__ import annotations

from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from itertools import pairwise, product
from typing import Optional

from ._unvalidated import warn_unvalidated

warn_unvalidated()


Incoming = namedtuple("incoming_edge", "weight parent")


@dataclass
class Edge:
    parent: str
    child: str
    weight: float
    rule: str
    site: tuple[int, ...]
    site_path: tuple[str, frozenset[int]] | None = None


@dataclass
class Node:
    node_id: int
    smiles: str
    metabolism_score: Optional[float] = None


@dataclass
class XenoGraph:
    root: str
    targets: list[str] = field(default_factory=list)
    nodes: dict[str, Node] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    edges: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        if self.root and self.root not in self.nodes:
            self._ensure_node(self.root)

    def _ensure_node(self, smiles: str) -> Node:
        if smiles not in self.nodes:
            self.nodes[smiles] = Node(node_id=len(self.nodes), smiles=smiles)
        return self.nodes[smiles]

    def add_edge(
        self,
        parent: str,
        child: str,
        weight: float,
        site_path: tuple[str, frozenset[int]],
    ) -> None:
        self._ensure_node(parent)
        self._ensure_node(child)
        rule, site = site_path
        site_t = tuple(sorted(int(x) for x in site))
        for existing in self.edges[parent]:
            if existing.child == child and existing.site_path == site_path:
                return
        self.edges[parent].append(
            Edge(
                parent=parent,
                child=child,
                weight=float(weight),
                rule=rule,
                site=site_t,
                site_path=site_path,
            )
        )
        self.children[parent].add(child)

    def add_path(
        self,
        smiles_path: list[str],
        weights: list[float],
        sites: list[tuple[str, frozenset[int]]],
    ) -> None:
        if len(smiles_path) < 2:
            if smiles_path:
                self._ensure_node(smiles_path[0])
            return
        for parent, child, w, site in zip(
            smiles_path, smiles_path[1:], weights, sites
        ):
            self.add_edge(parent, child, w, site)

    def is_empty(self) -> bool:
        return not any(self.edges.values())

    def all_edges(self) -> list[Edge]:
        out: list[Edge] = []
        for bucket in self.edges.values():
            out.extend(bucket)
        return out

    def find_all_paths_from_start(
        self, start: Optional[str] = None, path: Optional[list[str]] = None
    ) -> list[list[str]]:
        if not start:
            start = self.root
        if start not in self.nodes:
            return []
        path = (path or []) + [start]
        if start not in self.children or not self.children[start]:
            return [path]
        paths = [path]
        for vertex in self.children[start]:
            if vertex not in path:
                paths.extend(self.find_all_paths_from_start(vertex, path))
        return paths

    def find_all_paths(
        self,
        start: Optional[str] = None,
        end: Optional[str | list[str]] = None,
        path: Optional[list[str]] = None,
    ) -> list[list[str]]:
        if start is None:
            start = self.root
        if end is None:
            end = self.targets[0] if len(self.targets) == 1 else list(self.targets)
        if start not in self.nodes:
            return []
        path = (path or []) + [start]
        ends = end if isinstance(end, list) else [end]
        if start in ends:
            return [path]
        if start not in self.children:
            return []
        paths: list[list[str]] = []
        for vertex in self.children[start]:
            if vertex not in path:
                paths.extend(self.find_all_paths(vertex, end, path))
        return paths

    def pathway_weight_site_products(
        self, pathway: list[str]
    ) -> list[tuple[tuple[float, ...], tuple[tuple[str, frozenset[int]], ...]]]:
        weights: list[list[float]] = []
        sites: list[list[tuple[str, frozenset[int]]]] = []
        for curr, nxt in pairwise(pathway):
            w_here: list[float] = []
            s_here: list[tuple[str, frozenset[int]]] = []
            for edge in self.edges.get(curr, []):
                if edge.child == nxt:
                    w_here.append(edge.weight)
                    sp = edge.site_path or (edge.rule, frozenset(edge.site))
                    s_here.append(sp)
            weights.append(w_here)
            sites.append(s_here)
        if not weights:
            return []
        out = []
        for w_prod, s_prod in zip(product(*weights), product(*sites)):
            out.append((w_prod, s_prod))
        return out

    def trim_weights(self, threshold: float = 0.0) -> XenoGraph:
        """Keep path edges whose weights are all strictly greater than ``threshold``."""
        trimmed = XenoGraph(root=self.root, targets=list(self.targets))
        if not self.targets:
            paths = self.find_all_paths_from_start(self.root)
        elif len(self.targets) == 1:
            paths = self.find_all_paths(self.root, self.targets[0])
        else:
            paths = []
            for target in self.targets:
                paths.extend(self.find_all_paths(self.root, target))
        for path in paths:
            if len(path) < 2:
                trimmed._ensure_node(path[0] if path else self.root)
                continue
            for w_path, s_path in self.pathway_weight_site_products(pathway=path):
                if all(w > threshold for w in w_path):
                    trimmed.add_path(path, list(w_path), list(s_path))
        if not trimmed.nodes:
            trimmed._ensure_node(self.root)
        return trimmed

    def compute_metabolite_likelihoods(
        self, mode: str = "max", max_iterations: int = 100_000
    ) -> dict[str, float]:
        if self.is_empty():
            if self.root in self.nodes:
                self.nodes[self.root].metabolism_score = 1.0
            return {self.root: 1.0}

        child_to_parents: dict[str, list[Incoming]] = defaultdict(list)
        metabolite_scores: dict[str, float] = {self.root: 1.0}

        for parent_smi, edges in self.edges.items():
            for edge in edges:
                existing = child_to_parents[edge.child]
                found = False
                for idx, item in enumerate(existing):
                    if item.parent == parent_smi:
                        if mode == "sum":
                            new_weight = item.weight + edge.weight
                        else:
                            new_weight = max(item.weight, edge.weight)
                        existing[idx] = Incoming(weight=new_weight, parent=parent_smi)
                        found = True
                        break
                if not found:
                    child_to_parents[edge.child].append(
                        Incoming(parent=parent_smi, weight=edge.weight)
                    )

        path_nodes = set(self.nodes)

        for node_smi in list(path_nodes):
            if node_smi == self.root:
                continue
            for parent_edge in list(child_to_parents[node_smi]):
                parent = parent_edge.parent
                if any(x.parent == node_smi for x in child_to_parents[parent]):
                    child_to_parents[parent] = [
                        inc for inc in child_to_parents[parent] if inc.parent != node_smi
                    ]

        parent_normalization_factor: dict[str, float] = defaultdict(float)
        for child, incoming in child_to_parents.items():
            for parent_edge in incoming:
                parent_normalization_factor[parent_edge.parent] += parent_edge.weight

        remaining = max_iterations
        while len(metabolite_scores) != len(path_nodes):
            if remaining == 0:
                break
            remaining -= 1
            for node in list(path_nodes - set(metabolite_scores)):
                incoming = list(child_to_parents[node])
                if node == self.root or node in metabolite_scores:
                    continue
                if incoming and all(inc.parent in metabolite_scores for inc in incoming):
                    metric = 0.0
                    for inc in incoming:
                        denom = parent_normalization_factor[inc.parent]
                        frac = (inc.weight / denom) if denom else 0.0
                        metric += metabolite_scores[inc.parent] * frac
                    metabolite_scores[node] = metric

        for smi, score in metabolite_scores.items():
            if smi in self.nodes:
                self.nodes[smi].metabolism_score = score
        return metabolite_scores

    def to_dict(self) -> dict:
        nodes = [
            {
                "id": n.node_id,
                "smiles": n.smiles,
                "metabolism_score": n.metabolism_score,
            }
            for n in sorted(self.nodes.values(), key=lambda x: x.node_id)
        ]
        edges = [
            {
                "parent": e.parent,
                "child": e.child,
                "weight": e.weight,
                "rule": e.rule,
                "site": list(e.site),
            }
            for e in sorted(
                self.all_edges(),
                key=lambda e: (e.parent, e.child, e.rule, e.site),
            )
        ]
        return {"root": self.root, "targets": list(self.targets), "nodes": nodes, "edges": edges}
