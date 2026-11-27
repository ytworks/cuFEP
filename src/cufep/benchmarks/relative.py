"""The built-in benchmarks by relative binding free energies (`cufep benchmark ... --protocol rbfe`).

A set runs with the settings of its relative example (``rbfe_example`` in the registry): everything in that
configuration but the input files, which are the set's, the ligands and their network, and the output
directory. ``full`` runs every compound of the set on the LOMAP network of ``cufep rbfe`` (``network.edges``
of the example is not used); ``smoke`` runs the first edge of the example alone, with shortened sampling (never
lengthening a setting of the example). The experimental values are those of the set's table
(``ligands.csv``), which the relative calculation also uses to set the one constant of the dG.
"""
from __future__ import annotations

import csv
from pathlib import Path

from cufep.benchmarks.runner import _shorten, data_root, load_registry, read_reference

TIERS = ("smoke", "full")
#: the shortened settings of the smoke tier (the numbers of the absolute smoke tier, for the sections a relative
#: configuration has)
SMOKE = {"sampling": {"pre_equilibration": {"complex": {"nvt_ns": 0.02, "npt_ns": 0.02, "md_ns": 0.3},
                                            "solvent": {"nvt_ns": 0.01, "npt_ns": 0.02, "md_ns": 0.05}},
                      "equilibration_ns": 0.05, "production_ns": 0.1, "repeats": 1},
         "lambda_schedule": {"adaptive": {"window_ps": 10, "equilibration_ps": 0}},
         "membrane": {"equilibration_ns": 0.5}}


def _entry(name: str) -> dict:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(registry)})")
    return registry[name]


def example_path(name: str) -> Path:
    return data_root() / _entry(name)["rbfe_example"]


def benchmark_rbfe_config(name: str, tier: str, out_dir: str | Path | None = None):
    """The configuration of a relative benchmark run (:class:`cufep.rbfe.config.RbfeConfig`)."""
    from cufep.rbfe.config import RbfeConfig, load_rbfe_config

    if tier not in TIERS:
        raise KeyError(f"unknown tier '{tier}' of a relative benchmark (available: {', '.join(TIERS)})")
    entry, root = _entry(name), data_root()
    example = load_rbfe_config(example_path(name))
    data = example.model_dump(mode="json")
    data["system"].update(protein=str(root / entry["protein"]), ligands=str(root / entry["ligands"]),
                          experiment=None)
    data["output"]["dir"] = str(out_dir or Path("runs") / f"benchmark_rbfe_{name}_{tier}")
    if tier == "smoke":
        first = [list(example.network.edges[0])] if example.network.edges else None
        if first is None:
            raise ValueError(f"{example_path(name)} gives no edge for the smoke tier")
        data["system"]["ligand_names"] = first[0]
        data["network"]["edges"] = first
        _shorten(data, SMOKE)
    else:
        data["system"]["ligand_names"] = None
        data["network"]["edges"] = None
    return RbfeConfig.model_validate(data)


def experiment(name: str) -> dict[str, float]:
    """The experimental dG (kcal/mol) of every compound of the set, from its table."""
    return {compound: ref["exp"] for compound, ref in read_reference(name).items()}


def published_edges(name: str) -> dict[frozenset, dict]:
    """The edges of the published map of the set with their FEP+ predictions (``edge_predictions_*_out.csv`` of the
    public benchmark), keyed by the pair of compounds; each with the direction it was computed in."""
    path = data_root() / _entry(name)["fepplus_edges"]
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = row["Lig 1"].strip(), row["Lig 2"].strip()
            out[frozenset((a, b))] = {
                "ligand_a": a, "ligand_b": b, "exp_ddg": float(row["Exp. ddG (kcal/mol)"]),
                "bennett_ddg": float(row["Bennett ddG (kcal/mol)"]),
                "bennett_se": float(row["Bennett std. error (kcal/mol)"]),
                "ccc_ddg": float(row["CCC ddG (kcal/mol)"]), "ccc_se": float(row["CCC std. error (kcal/mol)"])}
    return out


def oriented(value: float | None, published: dict, a: str, b: str) -> float | None:
    """A ddG of the published edge in the direction a -> b."""
    if value is None:
        return None
    return value if (published["ligand_a"], published["ligand_b"]) == (a, b) else -value


def plan_rows(name: str, rcfg) -> tuple[list, list[tuple[str, str]], str, list[dict]]:
    """The ligands, the edges, how they were obtained and, for every edge, its mapping and LOMAP score and whether
    the published map of the set has it: what `cufep benchmark plan` shows before anything runs."""
    from cufep.rbfe.run import edges_of, ligands_of, mapped_edges

    ligands = ligands_of(rcfg)
    edges, how = edges_of(rcfg, ligands, lambda _line: None)
    published = published_edges(name)
    rows = mapped_edges(rcfg, ligands, edges)
    for r in rows:
        r["in_published_map"] = frozenset((r["ligand_a"], r["ligand_b"])) in published
    return ligands, edges, how, rows


def plan_lines(name: str, tier: str, rcfg) -> list[str]:
    ligands, edges, how, rows = plan_rows(name, rcfg)
    published = published_edges(name)
    names = {lig.name for lig in ligands}
    in_map = sum(r["in_published_map"] for r in rows)
    of_these = sum(1 for key in published if key <= names)
    lines = [f"benchmark {name}, tier {tier}, relative: {len(ligands)} ligand(s), {len(edges)} edge(s) ({how})",
             f"{in_map} of the {len(edges)} edges are in the published map of the set, which has {of_these} edges "
             "between these ligands",
             "", "| edge | atoms mapped | unique to A | unique to B | LOMAP score | in the published map |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['ligand_a']} -> {r['ligand_b']} | {r['mapped']} | {r['unique_a']} | {r['unique_b']} | "
              f"{r['lomap_score']:.2f} | {'yes' if r['in_published_map'] else 'no'} |" for r in rows]
    return lines
