"""The free energies of a network of relative transformations.

ddG of an edge A -> B
    ddG = dG(A -> B, complex) - dG(A -> B, solvent), with the standard error sqrt(se_c^2 + se_s^2) (the two
    legs are independent calculations; OpenFE adds them the same way).

Cycle closure
    For every cycle of a cycle basis of the network the sum of the ddG around it, and its standard error,
    reported as numbers without a verdict (.claude/rules/fep-invariants.md).

dG of every ligand
    The maximum likelihood estimate of Xu (J. Chem. Inf. Model. 2019, 59, 4720), by cinnabar's solver
    (``MLEEstimator.mle``, the one OpenFE's ``openfe gather --report dg`` uses): edges weighted by the inverse
    of their variance. The estimate fixes only differences; the reference is one of
        ``experiment``: the estimates shifted so that their mean over the ligands with an experimental value is
            the mean of those values (the FEP+ predictions of the public benchmark of Ross et al. 2023 and the
            OpenFE benchmark of Baumann et al. 2026 are referenced so);
        ``abfe``: absolute binding free energies of reference ligands, computed by cuFEP's ABFE, enter the same
            likelihood as harmonic wells on their nodes (cinnabar's ``node_data_label``, Xu 2019), so that the
            estimates are absolute without any experimental value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Edge:
    ligand_a: str
    ligand_b: str
    ddg: float
    se: float

    @property
    def name(self) -> str:
        return f"{self.ligand_a}~{self.ligand_b}"


def edge_ddg(dg_complex: float, se_complex: float, dg_solvent: float, se_solvent: float) -> tuple[float, float]:
    """ddG of an edge from the free energies of its two legs (each A -> B)."""
    return float(dg_complex - dg_solvent), float(math.hypot(se_complex, se_solvent))


def cycle_closures(edges: list[Edge]) -> list[dict]:
    """The sum of ddG around every cycle of a cycle basis of the network, with its standard error."""
    import networkx as nx

    graph = nx.Graph()
    signed = {}
    for e in edges:
        graph.add_edge(e.ligand_a, e.ligand_b)
        signed[(e.ligand_a, e.ligand_b)] = (e.ddg, e.se)
        signed[(e.ligand_b, e.ligand_a)] = (-e.ddg, e.se)
    out = []
    for cycle in nx.cycle_basis(graph):
        nodes = list(cycle) + [cycle[0]]
        total, var = 0.0, 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            d, s = signed[(a, b)]
            total += d
            var += s * s
        out.append({"ligands": list(cycle), "sum_kcal_mol": total, "se_kcal_mol": math.sqrt(var)})
    return out


def _graph(ligands: list[str], edges: list[Edge], anchors: dict[str, tuple[float, float]] | None = None):
    import networkx as nx

    graph = nx.DiGraph()
    for name in ligands:
        graph.add_node(name)
    for e in edges:
        if e.se <= 0.0:
            raise ValueError(f"edge {e.name} has no standard error; the estimate weights edges by it")
        graph.add_edge(e.ligand_a, e.ligand_b, f_ij=float(e.ddg), f_dij=float(e.se))
    for name, (value, se) in (anchors or {}).items():
        if se <= 0.0:
            raise ValueError(f"the reference value of {name} has no standard error")
        graph.nodes[name]["f_i"] = float(value)
        graph.nodes[name]["f_di"] = float(se)
    if not nx.is_weakly_connected(graph):
        raise ValueError("the edges do not connect all ligands")
    return graph


def mle(ligands: list[str], edges: list[Edge], anchors: dict[str, tuple[float, float]] | None = None):
    """(estimates, covariance) in the order of ``ligands``: cinnabar's maximum likelihood estimate."""
    from cinnabar.estimators import MLEEstimator

    graph = _graph(ligands, edges, anchors)
    f, cov = MLEEstimator.mle(graph, edge_data_label="f_ij", node_data_label="f_i" if anchors else None)
    order = list(graph.nodes())
    index = [order.index(name) for name in ligands]
    return np.asarray(f)[index], np.asarray(cov)[np.ix_(index, index)]


def dg_from_experiment(ligands: list[str], edges: list[Edge],
                       experiment: dict[str, float]) -> dict[str, tuple[float, float]]:
    """dG of every ligand: the estimate shifted to the mean of the experimental values it has. The standard
    error is that of the estimate of a ligand minus the mean of the estimates of the reference ligands."""
    refs = [name for name in ligands if name in experiment]
    if not refs:
        raise ValueError("no ligand of the network has an experimental value")
    f, cov = mle(ligands, edges)
    idx = [ligands.index(name) for name in refs]
    w = np.zeros(len(ligands))
    w[idx] = 1.0 / len(idx)
    shift = float(np.mean([experiment[name] for name in refs]) - w @ f)
    out = {}
    for i, name in enumerate(ligands):
        d = -w.copy()
        d[i] += 1.0
        out[name] = (float(f[i] + shift), float(math.sqrt(max(d @ cov @ d, 0.0))))
    return out


def dg_from_abfe(ligands: list[str], edges: list[Edge],
                 abfe: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """dG of every ligand from the edges and the ABFE of the reference ligands, in one likelihood."""
    if not abfe:
        raise ValueError("no reference ligand has an absolute binding free energy")
    unknown = sorted(set(abfe) - set(ligands))
    if unknown:
        raise ValueError(f"reference ligands that are not in the network: {', '.join(unknown)}")
    f, cov = mle(ligands, edges, anchors=abfe)
    return {name: (float(f[i]), float(math.sqrt(max(cov[i, i], 0.0)))) for i, name in enumerate(ligands)}
