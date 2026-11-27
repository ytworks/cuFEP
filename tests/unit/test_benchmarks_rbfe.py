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

    def fake(rcfg, experiment=None, **_kw):
        seen.update(rcfg=rcfg, experiment=experiment)
        return {"edges": [], "failed_in_this_run": ["ejm_31~ejm_46/r0/complex: no free energy"]}

    monkeypatch.setattr(rbfe_run, "run_rbfe", fake)
    result = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--protocol", "rbfe", "--tier", "full",
                                           "--gpus", "0,1", "-o", str(tmp_path / "out")])
    assert result.exit_code == 1, result.output
    assert seen["rcfg"].network.edges is None and seen["rcfg"].resources.gpus == [0, 1]
    assert str(seen["rcfg"].output.dir) == str(tmp_path / "out") and seen["experiment"] == experiment("tyk2")
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
    assert len(published_edges("a2a")) == 46


@pytest.mark.needs("openff.toolkit", "rdkit", "kartograf", "lomap")
def test_the_lomap_network_of_tyk2_is_that_of_the_openfe_benchmark():
    """The network the full tier runs: lomap2's defaults with Kartograf on the 16 compounds give the 22 edges with
    which the OpenFE benchmark (IndustryBenchmarks2024, jacs_set/tyk2) computed this set, in the same directions."""
    from cufep.rbfe.run import edges_of, ligands_of

    rcfg = benchmark_rbfe_config("tyk2", "full")
    edges, how = edges_of(rcfg, ligands_of(rcfg), lambda _line: None)
    assert how == "planned by LOMAP"
    assert sorted(edges) == sorted([
        ("ejm_31", "ejm_42"), ("ejm_31", "ejm_44"), ("ejm_31", "ejm_45"), ("ejm_31", "ejm_46"), ("ejm_31", "ejm_48"),
        ("ejm_42", "ejm_54"), ("ejm_43", "ejm_42"), ("ejm_44", "ejm_43"), ("ejm_47", "ejm_31"), ("ejm_47", "ejm_50"),
        ("ejm_49", "ejm_31"), ("ejm_49", "ejm_50"), ("ejm_50", "ejm_42"), ("ejm_50", "ejm_48"), ("ejm_50", "ejm_55"),
        ("ejm_55", "ejm_54"), ("jmc_23", "ejm_31"), ("jmc_23", "jmc_27"), ("jmc_23", "jmc_28"), ("jmc_27", "ejm_46"),
        ("jmc_28", "jmc_30"), ("jmc_30", "ejm_46")])
