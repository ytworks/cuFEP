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
    of their variance. The edges fix only the differences between the ligands; the one constant that is left is
    set as the relative calculations of the literature set it:
        with experimental values, the mean of the estimates of the ligands that have one is the mean of those
            values (the predictions of the public benchmark of Ross et al. 2023 and of the OpenFE benchmark
            of Baumann et al. 2026 are referenced so);
        without any, the mean of the estimates of all ligands is 0 (cinnabar's output): dG relative to the
            mean of the network, and the result says so.
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


def _graph(ligands: list[str], edges: list[Edge]):
    import networkx as nx

    graph = nx.DiGraph()
    for name in ligands:
        graph.add_node(name)
    for e in edges:
        if e.se <= 0.0:
            raise ValueError(f"edge {e.name} has no standard error; the estimate weights edges by it")
        graph.add_edge(e.ligand_a, e.ligand_b, f_ij=float(e.ddg), f_dij=float(e.se))
    if not nx.is_weakly_connected(graph):
        raise ValueError("the edges do not connect all ligands")
    return graph


def mle(ligands: list[str], edges: list[Edge]):
    """(estimates, covariance) in the order of ``ligands``: cinnabar's maximum likelihood estimate."""
    from cinnabar.estimators import MLEEstimator

    graph = _graph(ligands, edges)
    f, cov = MLEEstimator.mle(graph, edge_data_label="f_ij")
    order = list(graph.nodes())
    index = [order.index(name) for name in ligands]
    return np.asarray(f)[index], np.asarray(cov)[np.ix_(index, index)]


def network_ddg(ligands: list[str], edges: list[Edge], pairs: list[tuple[str, str]]) -> dict[tuple[str, str], tuple]:
    """ddG of every pair (A, B) from the network, dG(B) - dG(A) of the maximum likelihood estimate, with its standard
    error from the covariance of the estimate: the ddG that all edges together give a pair (an edge's own and that of
    the paths around it), as the cycle closure correction of Wang et al. (J. Chem. Theory Comput. 2013) gives every
    edge a corrected ddG. It does not depend on the constant that the edges leave."""
    f, cov = mle(ligands, edges)
    at = {name: i for i, name in enumerate(ligands)}
    out = {}
    for a, b in pairs:
        i, j = at[a], at[b]
        out[(a, b)] = (float(f[j] - f[i]), float(math.sqrt(max(cov[i, i] + cov[j, j] - 2.0 * cov[i, j], 0.0))))
    return out


def ligand_dg(ligands: list[str], edges: list[Edge], experiment: dict[str, float]) -> dict:
    """dG of every ligand from the edges: ``{"reference": ..., "values": {name: (dG, standard error)}}``.

    With experimental values the estimates are shifted so that the mean of those of the ligands with a value is
    the mean of the values (``reference``: ``"mean of the experimental values"``), and the standard error of a
    ligand is that of its estimate minus that mean; without any, the mean of all estimates is 0 (``reference``:
    ``"mean of the network"``), and the standard error is that of a ligand against that mean."""
    f, cov = mle(ligands, edges)
    refs = [name for name in ligands if name in experiment]
    members = refs or list(ligands)
    w = np.zeros(len(ligands))
    w[[ligands.index(name) for name in members]] = 1.0 / len(members)
    target = float(np.mean([experiment[name] for name in refs])) if refs else 0.0
    shift = target - float(w @ f)
    values = {}
    for i, name in enumerate(ligands):
        d = -w.copy()
        d[i] += 1.0
        values[name] = (float(f[i] + shift), float(math.sqrt(max(d @ cov @ d, 0.0))))
    return {"reference": "mean of the experimental values" if refs else "mean of the network",
            "reference_ligands": members, "values": values}
