from __future__ import annotations

from typing import Any

import networkx as nx


def build_prerequisite_dag(concepts: list[str]) -> dict[str, Any]:
    """Build a naive prerequisite DAG: earlier concepts prerequisite later ones."""
    g = nx.DiGraph()
    clean = [c.strip() for c in concepts if c and c.strip()]
    for concept in clean:
        g.add_node(concept)
    for i in range(len(clean) - 1):
        g.add_edge(clean[i], clean[i + 1])

    if not nx.is_directed_acyclic_graph(g):
        g = nx.DiGraph((u, v) for u, v in g.edges() if u != v)

    order = list(nx.topological_sort(g)) if g.number_of_nodes() else []
    return {
        "nodes": clean,
        "edges": [{"from": u, "to": v} for u, v in g.edges()],
        "topological_order": order,
    }
