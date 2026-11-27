"""The built-in benchmarks by relative binding free energies (`cufep benchmark ... --protocol rbfe`)."""
import pytest
from click.testing import CliRunner

from cufep.benchmarks.relative import benchmark_rbfe_config, example_path, experiment, published_edges
from cufep.benchmarks.runner import data_root, load_registry, read_reference

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit")


def _settings(rcfg) -> dict:
    data = rcfg.model_dump(mode="json")
    for key in ("protein", "ligands", "ligand_names", "experiment"):
        data["system"].pop(key)
    data.pop("output")
    data.pop("network")
    return data


@pytest.mark.parametrize("name", ["tyk2", "a2a"])
def test_a_relative_benchmark_runs_with_the_settings_of_its_example_on_every_compound_of_the_set(name, tmp_path):
    """The full tier: the settings of the relative example, the protein and every compound of the set, the LOMAP
    network (the edges of the example are not used); the example's protein is the set's and its ligands are records
    of the set with the same poses, so the example's edges are calculations the set can have."""
    from cufep.prep.ligand import load_ligands
    from cufep.rbfe.config import load_rbfe_config

    entry, root = load_registry()[name], data_root()
    example = load_rbfe_config(example_path(name))
    full = benchmark_rbfe_config(name, "full", tmp_path / "full")
    assert _settings(full) == _settings(example)
    assert full.system.ligand_names is None and full.network.edges is None and full.system.experiment is None
    assert full.network == type(full.network)()                   # LOMAP with the defaults of docs/rbfe.md
    assert full.system.protein == root / entry["protein"] and full.system.ligands == root / entry["ligands"]
    assert str(full.output.dir) == str(tmp_path / "full")
    assert example.system.protein.read_bytes() == full.system.protein.read_bytes()
    in_set = {lig.name: lig for lig in load_ligands(full.system.ligands)}
    assert sorted(in_set) == sorted(read_reference(name))
    for lig in load_ligands(example.system.ligands):
        other = in_set[lig.name]
        assert lig.offmol.to_smiles(mapped=True) == other.offmol.to_smiles(mapped=True)
        assert abs(lig.positions_nm() - other.positions_nm()).max() < 1e-4
    assert experiment(name) == {compound: ref["exp"] for compound, ref in read_reference(name).items()}


def test_the_smoke_tier_runs_the_first_edge_of_the_example_and_never_lengthens_a_setting(tmp_path, monkeypatch):
    smoke = benchmark_rbfe_config("tyk2", "smoke", tmp_path / "s")
    assert smoke.network.edges == [("ejm_31", "ejm_46")] and smoke.system.ligand_names == ["ejm_31", "ejm_46"]
    assert smoke.sampling.production_ns == 0.1 and smoke.sampling.equilibration_ns == 0.05
    adaptive = smoke.lambda_schedule.adaptive
    assert adaptive.window("complex") == adaptive.window("solvent") == 10 and adaptive.equilibration("complex") == 0
    a2a = benchmark_rbfe_config("a2a", "smoke", tmp_path / "m")
    assert a2a.membrane.enabled is True and a2a.membrane.equilibration_ns == 0.5
    assert a2a.network.edges == [("4e", "4b")]
    # a shorter setting of the example stays
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "examples" / "rbfe_tyk2").mkdir(parents=True)
    (tmp_path / "examples" / "rbfe_tyk2" / "config.yaml").write_text(
        "system: {protein: protein.pdb, ligands: ligands.sdf}\nnetwork: {edges: [[a, b]]}\n"
        "sampling: {production_ns: 0.05}\n")
    monkeypatch.setenv("CUFEP_DATA", str(tmp_path))
    assert benchmark_rbfe_config("tyk2", "smoke", tmp_path / "s").sampling.production_ns == 0.05
    with pytest.raises(KeyError):
        benchmark_rbfe_config("tyk2", "subset")


def test_benchmark_run_with_the_relative_protocol_runs_the_relative_calculation_with_the_values_of_the_set(
        tmp_path, monkeypatch):
    from cufep import cli
    from cufep.rbfe import run as rbfe_run

    seen = {}

    def fake(rcfg, experiment=None, only_edges=None, **_kw):
        seen.update(rcfg=rcfg, experiment=experiment, only_edges=only_edges)
        return {"edges": [], "failed_in_this_run": ["ejm_31~ejm_46/r0/complex: no free energy"]}

    monkeypatch.setattr(rbfe_run, "run_rbfe", fake)
    result = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--protocol", "rbfe", "--tier", "full",
                                           "--gpus", "0,1", "-o", str(tmp_path / "out")])
    assert result.exit_code == 1, result.output
    assert seen["rcfg"].network.edges is None and seen["rcfg"].resources.gpus == [0, 1]
    assert str(seen["rcfg"].output.dir) == str(tmp_path / "out") and seen["experiment"] == experiment("tyk2")
    seen.clear()
    result = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--protocol", "rbfe", "--tier", "full",
                                           "-o", str(tmp_path / "out"), "--edge", "ejm_42~ejm_54",
                                           "--edge", "ejm_55~ejm_54"])
    assert seen["only_edges"] == ["ejm_42~ejm_54", "ejm_55~ejm_54"]
    refused = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--edge", "ejm_42~ejm_54"])
    assert refused.exit_code != 0 and "--edge is an option of a relative benchmark" in refused.output
    for args in (["--tier", "subset"], ["--repeats", "3"]):
        refused = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--protocol", "rbfe", *args])
        assert refused.exit_code != 0 and "Error" in refused.output


def test_the_published_map_is_read_in_the_direction_it_was_computed():
    tyk2 = published_edges("tyk2")
    assert len(tyk2) == 24
    edge = tyk2[frozenset(("jmc_23", "jmc_30"))]
    assert (edge["ligand_a"], edge["ligand_b"]) == ("jmc_23", "jmc_30")
    assert edge["exp_ddg"] == pytest.approx(0.76, abs=1e-3) and edge["ccc_ddg"] == pytest.approx(0.14)
    from cufep.benchmarks.relative import oriented

    assert oriented(edge["ccc_ddg"], edge, "jmc_30", "jmc_23") == pytest.approx(-0.14)
    assert len(published_edges("a2a")) == 19                # the edges between the records of the set's compounds


@pytest.mark.needs("openff.toolkit", "rdkit", "kartograf", "lomap")
def test_the_lomap_network_of_tyk2_is_that_of_the_openfe_benchmark():
    """The network the full tier runs: lomap2's defaults with Kartograf on the 16 compounds give the 22 edges with
    which the OpenFE benchmark (IndustryBenchmarks2024, jacs_set/tyk2) computed this set, in the same directions."""
    from cufep.benchmarks.relative import openfe_results
    from cufep.rbfe.run import edges_of, ligands_of

    rcfg = benchmark_rbfe_config("tyk2", "full")
    edges, how = edges_of(rcfg, ligands_of(rcfg), lambda _line: None)
    assert how == "planned by LOMAP" and len(edges) == 22
    assert sorted(edges) == sorted((e["ligand_a"], e["ligand_b"]) for e in openfe_results("tyk2")[1].values())


def _synthetic_run(tmp_path, dg: dict[str, float], edges: list[dict]):
    """A run directory as `cufep rbfe run` leaves it (summary.json, network.json), with the given dG and edges."""
    import json

    ligands = list(dg)
    summary = {"ligands": ligands, "edges": edges, "experiment_kcal_mol": experiment("tyk2"),
               "cycle_closures": [{"ligands": ligands[:3], "sum_kcal_mol": 0.25, "se_kcal_mol": 0.5}],
               "dg": {"reference": "mean of the experimental values", "reference_ligands": ligands,
                      "values": {name: {"dg": v, "se": 0.3} for name, v in dg.items()}}, "notes": []}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "network.json").write_text(json.dumps({"ligands": ligands, "edges": [[e["ligand_a"], e["ligand_b"]]
                                                                                     for e in edges],
                                                       "edges_from": "planned by LOMAP"}))
    return tmp_path


def _edge(a, b, ddg=None, backend=("cupotential", "cupotential"), missing=None):
    e = {"edge": f"{a}~{b}", "ligand_a": a, "ligand_b": b, "repeats": [{"repeat": 0, "missing": missing}]
         if missing else [{"repeat": 0}]}
    if ddg is not None:
        e.update(ddg=ddg, se=0.2, windows={"complex": [12], "solvent": [10]},
                 lowest_overlap={"complex": 0.11, "solvent": 0.2}, not_reliable=[],
                 backend={"complex": [backend[0]], "solvent": [backend[1]]})
    return e


def test_the_report_compares_every_method_on_the_same_compounds_and_edges_in_the_direction_of_the_run(tmp_path):
    """Known answers: the statistics of each method are those of its values on the compounds that every method has;
    a published ddG computed the other way round is turned to the direction of the run (OpenFE and FEP+ computed
    ejm_49 -> ejm_31, the run ejm_31 -> ejm_49); an edge without a ddG is listed with the reason, a leg on stock
    OpenMM is named, and the dG differences of the network are given next to the ddG of each edge."""
    from cufep.benchmarks.relative import centred, report
    from cufep.benchmarks.stats import metrics

    ref = read_reference("tyk2")
    dg = {"ejm_31": -9.0, "ejm_46": -11.5, "ejm_49": -8.5, "jmc_23": -11.8}
    edges = [_edge("ejm_31", "ejm_46", -2.4, backend=("cupotential", "openmm")), _edge("ejm_31", "ejm_49", 0.6),
             _edge("jmc_23", "ejm_31", 2.9), _edge("ejm_46", "ejm_49", missing=["complex"])]
    run = _synthetic_run(tmp_path, dg, edges)
    text = report("tyk2", run)
    names = list(dg)
    exp = [ref[n]["exp"] for n in names]
    openfe = {"ejm_31": -9.82880464210406, "ejm_46": -11.022554899425682, "ejm_49": -8.786546351803489,
              "jmc_23": -11.096362322919733}
    for column, values in ((1, [dg[n] for n in names]), (2, [ref[n]["fepplus"] for n in names]),
                           (3, [openfe[n] for n in names])):
        m = metrics(centred(values, exp), exp)
        row = next(line for line in text.splitlines() if line.startswith("| RMSE (kcal/mol) |"))
        assert row.split("|")[column + 1].strip().startswith(f"{m['rmse']:.2f} [")
    row = next(line for line in text.splitlines() if line.startswith("| ejm_31 -> ejm_49 |"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[2] == "+1.79" and cells[3] == "+0.60" and cells[5] == "+0.50"      # experiment, edge, network
    assert cells[6] == "+1.84" and cells[7] == "+1.86"                              # FEP+ turned round
    assert cells[8] == "+0.84 ± 0.14"                                               # OpenFE turned round
    row = next(line for line in text.splitlines() if line.startswith("| ejm_46 -> ejm_49 |"))
    assert "no ddG (no result of the complex leg)" in row and "| - | - | - |" in row
    assert "the fallback to stock OpenMM, whose reason the log of the leg gives): ejm_31~ejm_46." in text
    assert "(planned by LOMAP)" in text and "| ejm_31 -> ejm_46 -> ejm_49 | +0.25 | 0.50 |" in text
    edge_rmse = metrics([-2.4, 0.6, 2.9], [ref["ejm_46"]["exp"] - ref["ejm_31"]["exp"],
                                           ref["ejm_49"]["exp"] - ref["ejm_31"]["exp"],
                                           ref["ejm_31"]["exp"] - ref["jmc_23"]["exp"]])["rmse"]
    assert f"| all 3 edges of the run | 3 | RMSE (kcal/mol) | {edge_rmse:.2f} |" in text
    for f in ("rbfe_report.md", "rbfe_report_ligands.csv", "rbfe_report_edges.csv"):
        assert (run / f).exists()
    import csv

    with open(run / "rbfe_report_edges.csv") as fh:
        rows = {r["edge"]: r for r in csv.DictReader(fh)}
    assert float(rows["ejm_31~ejm_49"]["openfe_ddg"]) == pytest.approx(0.8430866987305023)
    assert rows["ejm_46~ejm_49"]["why_no_ddg"] == "no result of the complex leg"


def test_a_set_without_published_openfe_results_says_so(tmp_path):
    import json

    from cufep.benchmarks.relative import openfe_results, report

    assert openfe_results("a2a") == ({}, {})
    dg, of_edges = openfe_results("tyk2")
    assert len(dg) == 16 and len(of_edges) == 22
    (tmp_path / "summary.json").write_text(json.dumps({
        "ligands": ["4e", "4b"], "edges": [_edge("4b", "4e", -2.0)], "experiment_kcal_mol": experiment("a2a"),
        "cycle_closures": [], "notes": [],
        "dg": {"reference": "mean of the experimental values", "reference_ligands": ["4e", "4b"],
               "values": {"4e": {"dg": -12.0, "se": 0.2}, "4b": {"dg": -10.0, "se": 0.2}}}}))
    text = report("a2a", tmp_path)
    assert "the OpenFE benchmark published no result for this set; its columns are empty" in text
    assert "| statistic | cuFEP | FEP+ |" in text and "OpenFE's network" not in text


def test_the_published_map_of_a2a_is_read_by_compound(tmp_path):
    """Review of PR #21: the FEP+ map of A2A names its records ("4e D", 4jS ...), the run its compounds; the edge
    4e -> 4b of the smoke tier (published "4e D" -> "4b D": experiment +2.11, Bennett +1.03, cycle closure +1.02) was
    not found, and 8 of the 19 published edges between the records of the set were left out."""
    from cufep.benchmarks.relative import report

    a2a = published_edges("a2a")
    edge = a2a[frozenset(("4e", "4b"))]
    assert (edge["ligand_a"], edge["ligand_b"]) == ("4e", "4b") and edge["bennett_ddg"] == pytest.approx(1.03, abs=5e-3)
    compounds = set(read_reference("a2a"))
    assert sum(1 for key in a2a if key <= compounds) == 19
    assert frozenset(("4j", "4k")) in a2a and frozenset(("4m", "4n")) in a2a
    import json

    (tmp_path / "summary.json").write_text(json.dumps({
        "ligands": ["4e", "4b"], "edges": [_edge("4e", "4b", 1.5)], "experiment_kcal_mol": experiment("a2a"),
        "cycle_closures": [], "notes": [],
        "dg": {"reference": "mean of the experimental values", "reference_ligands": ["4e", "4b"],
               "values": {"4e": {"dg": -11.8, "se": 0.2}, "4b": {"dg": -10.3, "se": 0.2}}}}))
    row = next(line for line in report("a2a", tmp_path).splitlines() if line.startswith("| 4e -> 4b |"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[6] == "+1.03" and cells[7] == "+1.02"


def test_without_dg_the_edges_are_still_compared(tmp_path):
    """Review of PR #21: a run without dG (edges that do not connect the compounds) gave an empty row for all its
    edges, because the row also asked for the ddG of the network."""
    import json

    from cufep.benchmarks.relative import report
    from cufep.benchmarks.stats import metrics

    ref = read_reference("tyk2")
    edges = [_edge("ejm_31", "ejm_46", -1.5), _edge("ejm_46", "jmc_23", -0.5), _edge("ejm_31", "ejm_49")]
    (tmp_path / "summary.json").write_text(json.dumps({
        "ligands": ["ejm_31", "ejm_46", "ejm_49", "jmc_23"], "edges": edges, "experiment_kcal_mol": experiment("tyk2"),
        "cycle_closures": [], "dg": None, "notes": ["the edges with a ddG do not connect all ligands: no dG"]}))
    text = report("tyk2", tmp_path)
    rmse = metrics([-1.5, -0.5], [ref["ejm_46"]["exp"] - ref["ejm_31"]["exp"],
                                  ref["jmc_23"]["exp"] - ref["ejm_46"]["exp"]])["rmse"]
    assert f"| all 2 edges of the run | 2 | RMSE (kcal/mol) | {rmse:.2f} |" in text


def test_every_method_is_centred_on_the_compounds_it_is_compared_on(tmp_path):
    """Review of PR #21: only cuFEP was on the experimental mean of the compounds compared; FEP+ as published has a
    mean signed error of -0.156 over the 17 compounds of A2A (MUE 0.841 published, 0.793 centred). Every method is
    centred on the compounds compared (cinnabar's centralizing), so that RMSE and MUE compare like with like."""
    import json

    import numpy as np

    from cufep.benchmarks.relative import report

    ref = read_reference("a2a")
    names = list(ref)
    exp = np.array([ref[n]["exp"] for n in names])
    fep = np.array([ref[n]["fepplus"] for n in names])
    assert np.mean(fep - exp) == pytest.approx(-0.156, abs=1e-3)
    centred = fep - fep.mean() + exp.mean()
    mue = float(np.mean(np.abs(centred - exp)))
    assert mue == pytest.approx(0.793, abs=1e-3)
    (tmp_path / "summary.json").write_text(json.dumps({
        "ligands": names, "edges": [], "experiment_kcal_mol": experiment("a2a"), "cycle_closures": [], "notes": [],
        "dg": {"reference": "mean of the experimental values", "reference_ligands": names,
               "values": {n: {"dg": ref[n]["exp"] + 0.4 * (i % 3 - 1), "se": 0.2} for i, n in enumerate(names)}}}))
    row = next(line for line in report("a2a", tmp_path).splitlines() if line.startswith("| MUE (kcal/mol) |"))
    assert row.split("|")[3].strip().startswith(f"{mue:.2f} [")
