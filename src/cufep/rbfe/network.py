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
    ``"mean of the network"``), and the standard error is that of a ligand against that mean. Every ligand its own
    compound of one state (:func:`compound_dg`)."""
    out = compound_dg(ligands, edges, {name: name for name in ligands}, experiment)
    return {"reference": out["reference"], "reference_ligands": out["reference_compounds"],
            "values": {name: (v["dg"], v["se"]) for name, v in out["compounds"].items()}}


def compound_dg(states: list[str], edges: list[Edge], compound_of: dict[str, str], experiment: dict[str, float],
                fraction: dict[str, float] | None = None, symmetry: dict[str, int] | None = None,
                kt: float = 0.5925) -> dict:
    """dG of every compound from the edges between its states (kcal/mol).

    The states of a compound are its binding modes, rotamers and protonation or tautomeric forms; each is a ligand
    of the network. With f_i the estimate of state i (:func:`mle`), P_i the population of its form in solution
    (``fraction``; 1 for a compound of one form) and s the symmetry number of the compound (``symmetry``; 1 by
    default):

        dG(compound) = -kT ln sum_i P_i exp(-f_i / kT) - kT ln s

    the binding of a ligand of several states (Azimi et al., J. Comput. Aided Mol. Des. 2022, eq 10, for protonation
    states with their populations in solution; Ross et al., Commun. Chem. 2023, Supplementary eq 2 for binding modes
    that interconvert in solution but not in the complex, P = 1, and Supplementary section 1.2.1 for a symmetric group
    that does not turn in the complex, s = 2: "lowered by kT ln(2)"). One constant is left as for the ligands of
    :func:`ligand_dg`: the mean over the compounds with an experimental value is the mean of those values (else the
    mean over all compounds is 0), and the states get the same constant. Standard errors from the covariance of the
    estimate (linearised: the weights of the states in the sum). Returns ``{"reference", "reference_compounds",
    "compounds": {name: {dg, se, weights, symmetry_number}}, "states": {name: {dg, se}}}``."""
    f, cov = mle(states, edges)
    fraction = fraction or {}
    symmetry = symmetry or {}
    compounds = list(dict.fromkeys(compound_of[s] for s in states))
    index = {s: i for i, s in enumerate(states)}
    value, gradient = {}, {}
    for c in compounds:
        members = [s for s in states if compound_of[s] == c]
        logs = np.array([math.log(float(fraction.get(s, 1.0))) - f[index[s]] / kt for s in members])
        top = float(logs.max())
        weights = np.exp(logs - top)
        total = float(weights.sum())
        sigma = int(symmetry.get(c, 1))
        value[c] = -kt * (top + math.log(total)) - kt * math.log(sigma)
        g = np.zeros(len(states))
        for s, w in zip(members, weights / total):
            g[index[s]] = w
        gradient[c] = g
    refs = [c for c in compounds if c in experiment]
    members = refs or compounds
    mean_gradient = np.mean([gradient[c] for c in members], axis=0)
    target = float(np.mean([experiment[c] for c in refs])) if refs else 0.0
    shift = target - float(np.mean([value[c] for c in members]))

    def error(d: np.ndarray) -> float:
        return float(math.sqrt(max(d @ cov @ d, 0.0)))

    out_compounds = {}
    for c in compounds:
        out_compounds[c] = {"dg": value[c] + shift, "se": error(gradient[c] - mean_gradient),
                            "weights": {s: float(gradient[c][index[s]]) for s in states if compound_of[s] == c},
                            "symmetry_number": int(symmetry.get(c, 1))}
    out_states = {}
    for s in states:
        unit_vector = np.zeros(len(states))
        unit_vector[index[s]] = 1.0
        out_states[s] = {"dg": float(f[index[s]] + shift), "se": error(unit_vector - mean_gradient)}
    return {"reference": "mean of the experimental values" if refs else "mean of the network",
            "reference_compounds": members, "compounds": out_compounds, "states": out_states}
