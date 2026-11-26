"""Does the LOMAP rule of `cufep rbfe` keep the accuracy when it takes fewer edges? A check on the published edges.

The public benchmark publishes, for every edge of its map, the ddG of a relative calculation (Bennett) with its
standard error (`benchmarks/<set>/inputs/edge_predictions_*`). Here the dG of every ligand are estimated as
`cufep rbfe` estimates them (maximum likelihood, shifted to the mean of the experimental values) from
    (a) every published edge, and
    (b) the edges that the LOMAP rule of `cufep rbfe` (lomap2's GraphGen with the defaults of `network`) keeps when it
        may choose among the published edges only (the other pairs get the score 0),
and compared with experiment: pairwise RMSE, R^2, Kendall tau and RMSE, with 95 % bootstrap intervals over ligands.
The standard errors the two networks predict are given too, from the published errors of the edges and from equal
errors (1 for every edge), and with equal errors for the network that LOMAP plans among all pairs, as `cufep rbfe`
runs it. Numbers only. CPU, inside the image:

    spikes/run.sh spikes/m39_network_reduction.py tyk2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def scores_of(ligands, pairs, settings):
    from cufep.rbfe.mapping import kartograf_mapping, lomap_score

    by_name = {lig.name: lig for lig in ligands}
    return {(a, b): lomap_score(by_name[a], by_name[b], kartograf_mapping(by_name[a], by_name[b], settings))
            for a, b in pairs}


def lomap_subset(names, scores, network):
    """The edges GraphGen keeps when the pairs of ``scores`` are the only ones it may take."""
    from lomap.graphgen import GraphGen

    index = {n: i for i, n in enumerate(names)}
    matrix = np.zeros((len(names), len(names)))
    for (a, b), s in scores.items():
        matrix[index[a], index[b]] = matrix[index[b], index[a]] = s
    graph = GraphGen(score_matrix=matrix, ids=list(range(len(names))), names=list(names),
                     max_path_length=network.max_path_length, actives=[False] * len(names), max_dist_from_active=2,
                     similarity_cutoff=1 - network.distance_cutoff,
                     require_cycle_covering=network.require_cycle_covering, radial=False, fast=False,
                     hub=None).resultGraph
    kept = {frozenset((names[i], names[j])) for i, j in graph.edges}
    return [pair for pair in scores if frozenset(pair) in kept]


def estimate(names, edges, experiment):
    from cufep.benchmarks.stats import bootstrap, metrics
    from cufep.rbfe.network import ligand_dg

    dg = ligand_dg(names, edges, experiment)["values"]
    pred, exp = [dg[n][0] for n in names], [experiment[n] for n in names]
    m, ci = metrics(pred, exp), bootstrap(pred, exp)
    keys = ("pairwise_rmse", "pearson_r2", "kendall_tau", "rmse")
    return {"edges": len(edges), "mean_predicted_se": float(np.mean([dg[n][1] for n in names])),
            **{k: m[k] for k in keys}, "intervals": {k: ci.get(k) for k in keys}}


def equal_error_se(names, pairs, experiment):
    """The mean standard error of the dG that a network gives when every edge has the error 1 (its topology alone)."""
    from cufep.rbfe.network import Edge, ligand_dg

    dg = ligand_dg(names, [Edge(a, b, 0.0, 1.0) for a, b in pairs], experiment)["values"]
    return float(np.mean([dg[n][1] for n in names]))


def main(name: str) -> None:
    from cufep.benchmarks.rbfe import published_edges, rbfe_benchmark_config
    from cufep.benchmarks.runner import read_reference
    from cufep.prep.ligand import load_ligands
    from cufep.rbfe.mapping import MappingSettings, plan_network
    from cufep.rbfe.network import Edge

    cfg = rbfe_benchmark_config(name, "full")
    ligands = load_ligands(cfg.system.ligands)
    names = [lig.name for lig in ligands]
    experiment = {k: v["exp"] for k, v in read_reference(name).items() if k in names}
    published = published_edges(name)
    m = cfg.mapping
    settings = MappingSettings(atom_max_distance=m.atom_max_distance_A,
                               map_hydrogens_on_hydrogens_only=m.map_hydrogens_on_hydrogens_only,
                               map_exact_ring_matches_only=m.map_exact_ring_matches_only,
                               allow_partial_fused_rings=m.allow_partial_fused_rings,
                               allow_bond_breaks=m.allow_bond_breaks)
    scores = scores_of(ligands, list(published), settings)
    subset = lomap_subset(names, scores, cfg.network)
    edges = {pair: Edge(pair[0], pair[1], v["bennett"], v["bennett_error"]) for pair, v in published.items()}
    planned = plan_network(ligands, settings, cfg.network.distance_cutoff, cfg.network.max_path_length,
                           cfg.network.require_cycle_covering)
    out = {"set": name, "ligands": len(names),
           "a_every_published_edge": estimate(names, list(edges.values()), experiment),
           "b_lomap_among_the_published_edges": estimate(names, [edges[p] for p in subset], experiment),
           "b_edges": [list(p) for p in subset], "scores": {f"{a}~{b}": s for (a, b), s in scores.items()},
           "equal_errors_mean_se": {"a_published_map": equal_error_se(names, list(published), experiment),
                                    "b_lomap_among_the_published_edges": equal_error_se(names, subset, experiment),
                                    "c_lomap_among_all_pairs": equal_error_se(names, planned, experiment)},
           "c_edges": [list(p) for p in planned]}
    dest = Path("spikes/out/m39")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{name}.json").write_text(json.dumps(out, indent=1))
    for key in ("a_every_published_edge", "b_lomap_among_the_published_edges"):
        r = out[key]
        ci = r["intervals"]
        print(f"{name} {key}: {r['edges']} edges; pairwise RMSE {r['pairwise_rmse']:.2f} "
              f"{np.round(ci['pairwise_rmse'], 2)}, "
              f"R^2 {r['pearson_r2']:.2f} {np.round(ci['pearson_r2'], 2)}, tau {r['kendall_tau']:.2f} "
              f"{np.round(ci['kendall_tau'], 2)}, RMSE {r['rmse']:.2f}; mean predicted SE {r['mean_predicted_se']:.3f}")
    print(f"{name} equal errors, mean SE of the dG: " + ", ".join(f"{k} {v:.3f}" for k, v in
                                                                 out["equal_errors_mean_se"].items())
          + f" ({len(planned)} edges in c)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tyk2")
