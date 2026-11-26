"""ddG of edges, cycle closures and the dG of the ligands of a network (cufep.rbfe.network), against synthetic
truth and against cinnabar's estimate."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.needs("cinnabar", "networkx")

TRUTH = {"a": -9.5, "b": -11.3, "c": -7.8, "d": -10.1}
PAIRS = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("a", "d")]


def _edges(noise: float = 0.0, se: float = 0.3, seed: int = 4):
    from cufep.rbfe.network import Edge

    rng = np.random.default_rng(seed)
    return [Edge(a, b, TRUTH[b] - TRUTH[a] + noise * rng.normal(), se) for a, b in PAIRS]


def test_the_ddg_of_an_edge_is_complex_minus_solvent_with_the_errors_added_in_quadrature():
    from cufep.rbfe.network import edge_ddg

    ddg, se = edge_ddg(-12.0, 0.3, -10.5, 0.4)
    assert ddg == pytest.approx(-1.5) and se == pytest.approx(0.5)


def test_exact_edges_give_the_true_differences_and_zero_cycle_closures():
    from cufep.rbfe.network import cycle_closures, ligand_dg, mle

    names = list(TRUTH)
    f, cov = mle(names, _edges())
    assert np.allclose(f - f[0], [TRUTH[n] - TRUTH["a"] for n in names], atol=1e-9)
    closures = cycle_closures(_edges())
    assert len(closures) == len(PAIRS) - len(names) + 1
    assert all(abs(c["sum_kcal_mol"]) < 1e-9 for c in closures)
    # every cycle of three edges of standard error 0.3 has a sum with standard error 0.3 sqrt(3)
    assert all(c["se_kcal_mol"] == pytest.approx(0.3 * math.sqrt(len(c["ligands"]))) for c in closures)
    # shifted to the mean of the experimental values the network has: two values give every ligand exactly
    dg = ligand_dg(names, _edges(), {"a": TRUTH["a"], "c": TRUTH["c"]})
    assert dg["reference"] == "mean of the experimental values" and dg["reference_ligands"] == ["a", "c"]
    assert all(dg["values"][n][0] == pytest.approx(TRUTH[n], abs=1e-9) for n in names)
    assert cov.shape == (4, 4)


def test_the_network_gives_every_pair_the_difference_of_its_estimate_with_the_error_of_the_paths():
    """The ddG the whole network gives a pair (what the cycle closure correction gives an edge): exact edges give the
    true difference also for a pair that is no edge; an edge outside every cycle keeps its own ddG and error; an edge
    of a triangle of equal errors s has the error s sqrt(2/3) (its two paths in parallel: s^2 against 2 s^2)."""
    from cufep.rbfe.network import Edge, mle, network_ddg

    names = list(TRUTH)
    exact = network_ddg(names, _edges(), [("a", "b"), ("b", "d")])
    assert exact[("b", "d")][0] == pytest.approx(TRUTH["d"] - TRUTH["b"], abs=1e-9)
    tree = [Edge("a", "b", -1.7, 0.3), Edge("b", "c", 2.9, 0.4)]
    out = network_ddg(["a", "b", "c"], tree, [("a", "b"), ("b", "c"), ("a", "c")])
    assert out[("a", "b")] == pytest.approx((-1.7, 0.3)) and out[("b", "c")] == pytest.approx((2.9, 0.4))
    assert out[("a", "c")] == pytest.approx((1.2, 0.5))
    triangle = [Edge("a", "b", -1.0, 0.3), Edge("b", "c", 2.0, 0.3), Edge("a", "c", 1.6, 0.3)]
    out = network_ddg(["a", "b", "c"], triangle, [("a", "b"), ("a", "c")])
    f, _ = mle(["a", "b", "c"], triangle)
    assert out[("a", "b")][0] == pytest.approx(f[1] - f[0]) and out[("a", "c")][0] == pytest.approx(f[2] - f[0])
    # the closure of the cycle (-1 + 2 - 1.6 = -0.6) is spread over its three edges in equal parts
    assert out[("a", "b")][0] == pytest.approx(-1.0 + 0.2) and out[("a", "c")][0] == pytest.approx(1.6 - 0.2)
    assert out[("a", "b")][1] == pytest.approx(0.3 * math.sqrt(2.0 / 3.0))


def test_the_estimate_is_cinnabars_for_noisy_edges():
    """The same numbers as cinnabar's maximum likelihood estimate through its own FEMap (Xu 2019), which OpenFE
    reports; the network fixes differences only, so both are compared after removing the mean."""
    import openff.units as off_units
    from cinnabar import FEMap, Measurement, ReferenceState

    from cufep.rbfe.network import mle

    edges = _edges(noise=0.6, se=0.25)
    names = list(TRUTH)
    f, cov = mle(names, edges)
    femap = FEMap()
    for e in edges:
        femap.add_measurement(Measurement(labelA=e.ligand_a, labelB=e.ligand_b,
                                          DG=e.ddg * off_units.unit.kilocalorie_per_mole,
                                          uncertainty=e.se * off_units.unit.kilocalorie_per_mole,
                                          computational=True, source="test"))
    femap.generate_absolute_values()
    theirs = {m.labelB: m.DG.m for m in femap if isinstance(m.labelA, ReferenceState) and m.computational
              and m.labelB in TRUTH}
    ours = dict(zip(names, f - f.mean()))
    mean = np.mean(list(theirs.values()))
    assert all(ours[n] == pytest.approx(theirs[n] - mean, abs=1e-6) for n in names)


def test_without_experimental_values_the_dg_are_relative_to_the_mean_of_the_network():
    """The edges fix the differences only: without an experimental value the mean of all ligands is 0, as cinnabar
    gives them, and the result says which reference it has; one experimental value moves all ligands by the same
    amount, and a ligand far from it in the network has the larger error."""
    from cufep.rbfe.network import ligand_dg

    names = list(TRUTH)
    mean = np.mean(list(TRUTH.values()))
    relative = ligand_dg(names, _edges(), {})
    assert relative["reference"] == "mean of the network"
    assert all(relative["values"][n][0] == pytest.approx(TRUTH[n] - mean, abs=1e-9) for n in names)
    one = ligand_dg(names, _edges(), {"b": TRUTH["b"] + 1.0})
    assert all(one["values"][n][0] == pytest.approx(TRUTH[n] + 1.0, abs=1e-9) for n in names)
    assert one["values"]["b"][1] == pytest.approx(0.0, abs=1e-9) and one["values"]["d"][1] > one["values"]["a"][1]


def test_edges_that_do_not_connect_the_ligands_or_lack_an_error_are_refused():
    from cufep.rbfe.network import Edge, mle

    with pytest.raises(ValueError, match="connect"):
        mle(["a", "b", "c"], [Edge("a", "b", 1.0, 0.2)])
    with pytest.raises(ValueError, match="standard error"):
        mle(["a", "b"], [Edge("a", "b", 1.0, 0.0)])


def test_a_run_directory_is_analysed_from_its_results_whatever_its_configuration_would_be_refused_for(tmp_path):
    """Review of PR #18 (P2): `cufep rbfe analyze` validated the whole stored configuration, and the windows of a
    precalculation that #17 accepted (0.04 ps without equilibration, a sample every 0.02 ps: 10 steps, fewer than the
    25 of the barostat) stopped it before the results of the legs were combined. It reads the number of repeats
    only; a new run is refused for these windows as before."""
    import json

    import yaml

    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import analyse_directory

    windows = {"adaptive": {"window_ps": 0.04, "equilibration_ps": 0.0, "sample_interval_ps": 0.02}}
    stored = {"system": {"protein": "p.pdb", "ligands": "l.sdf"}, "lambda_schedule": windows,
              "sampling": {"repeats": 1}}
    with pytest.raises(ValueError, match="barostat"):
        RbfeConfig.model_validate(stored)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(stored))
    (tmp_path / "network.json").write_text(json.dumps({"ligands": ["a", "b"], "edges": [["a", "b"]],
                                                       "experiment_kcal_mol": {"a": -9.0, "b": -10.0}}))
    (tmp_path / "edges" / "a~b" / "prep").mkdir(parents=True)
    (tmp_path / "edges" / "a~b" / "prep" / "prep.json").write_text(json.dumps({"made_from": {}, "legs": {}}))
    for leg, dg in (("complex", -12.0 * 4.184), ("solvent", -10.5 * 4.184)):
        leg_dir = tmp_path / "edges" / "a~b" / "repeat_0" / leg
        leg_dir.mkdir(parents=True)
        (leg_dir / "result.json").write_text(json.dumps({
            "delta_g": dg, "delta_g_error": 0.3 * 4.184, "n_states": 9, "backend": "openmm", "notes": [],
            "quality": {"overlap": {"lowest": 0.1, "ok": True}, "samples": {"counted": 100.0}},
            "run": {"hybrid_version": 2}}))
    summary = analyse_directory(tmp_path)
    edge = summary["edges"][0]
    assert edge["ddg"] == pytest.approx(-1.5) and edge["se"] == pytest.approx(0.3 * math.sqrt(2))
    assert summary["dg"]["values"]["b"]["dg"] - summary["dg"]["values"]["a"]["dg"] == pytest.approx(-1.5)


def _leg_result(dg_kcal: float, se_kcal: float | None = 0.3, temperature: float = 298.15, pressure: float = 1.0,
                overlap: float = 0.1):
    return {"delta_g": dg_kcal * 4.184, "delta_g_error": None if se_kcal is None else se_kcal * 4.184,
            "temperature_K": temperature, "pressure_bar": pressure, "lrc_method": "none",
            "prepared_systems_sha256": "abc", "n_states": 9, "backend": "openmm", "notes": [],
            "quality": {"overlap": {"lowest": overlap, "ok": overlap >= 0.03}, "samples": {"counted": 100.0}},
            "run": {"hybrid_version": 2}}


def _write_legs(out_dir, edge: str, repeat: int, complex_, solvent) -> None:
    """The results of the two legs of a repeat of an edge, with a record of its prepared systems (an edge with results
    and without one is left out)."""
    import json

    prep = out_dir / "edges" / edge / "prep" / "prep.json"
    if not prep.exists():
        prep.parent.mkdir(parents=True, exist_ok=True)
        prep.write_text(json.dumps({"made_from": {}, "legs": {}}))
    for leg, result in (("complex", complex_), ("solvent", solvent)):
        leg_dir = out_dir / "edges" / edge / f"repeat_{repeat}" / leg
        leg_dir.mkdir(parents=True)
        (leg_dir / "result.json").write_text(json.dumps(result))


def test_legs_of_other_conditions_or_without_a_standard_error_give_no_ddg_and_the_network_goes_on(tmp_path):
    """Review of PR #17: the legs of an edge run at 298 K and 310 K (two runs of --leg in one directory) were
    subtracted, and a leg whose standard error could not be determined (null) stopped the whole combination with a
    TypeError. Such an edge gets no ddG, says why, and the other edges are combined."""
    from cufep.rbfe.run import combine_network

    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    _write_legs(tmp_path, "b~c", 0, _leg_result(-8.0, temperature=310.0), _leg_result(-9.0))
    _write_legs(tmp_path, "a~c", 0, _leg_result(-7.0, se_kcal=None), _leg_result(-6.0))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], 1, {})
    edges = {e["edge"]: e for e in summary["edges"]}
    assert edges["a~b"]["ddg"] == pytest.approx(-1.5)
    assert "ddg" not in edges["b~c"] and any("temperature" in item for item in edges["b~c"]["inconsistent_legs"])
    assert "ddg" not in edges["a~c"]
    assert edges["a~c"]["repeats"][0]["no_standard_error"] == ["complex"]


def test_an_edge_without_its_own_ddg_gets_that_of_the_network(tmp_path):
    """Every edge of the network gets the ddG of the network, in the summary and in edges.csv; one whose legs failed
    gets the sum along the path the other edges make."""
    import csv

    from cufep.rbfe.run import combine_network, write_results

    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    _write_legs(tmp_path, "b~c", 0, _leg_result(-8.0), _leg_result(-9.0))
    _write_legs(tmp_path, "a~c", 0, _leg_result(-7.0, se_kcal=None), _leg_result(-6.0))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], 1, {})
    edges = {e["edge"]: e for e in summary["edges"]}
    assert "ddg" not in edges["a~c"] and edges["a~c"]["network_ddg"] == pytest.approx(-1.5 + 1.0)
    assert edges["a~b"]["network_ddg"] == pytest.approx(edges["a~b"]["ddg"])
    assert edges["a~c"]["network_se"] == pytest.approx(math.hypot(edges["a~b"]["se"], edges["b~c"]["se"]))
    write_results(tmp_path, summary)
    rows = {r["edge"]: r for r in csv.DictReader((tmp_path / "edges.csv").open())}
    assert float(rows["a~c"]["network_ddg_kcal_mol"]) == pytest.approx(-0.5) and rows["a~c"]["ddg_kcal_mol"] == ""


def test_the_free_energies_of_the_legs_in_the_table_give_its_ddg(tmp_path):
    """Review of PR #17: with two repeats (complex, solvent) = (1, 0) and (5, 0) kcal/mol the table gave the ddG of
    both (3) next to the legs of the first (1 and 0). The legs are the means over the same repeats as the ddG."""
    import csv

    from cufep.rbfe.run import combine_network, write_results

    _write_legs(tmp_path, "a~b", 0, _leg_result(1.0), _leg_result(0.0))
    _write_legs(tmp_path, "a~b", 1, _leg_result(5.0), _leg_result(0.0))
    write_results(tmp_path, combine_network(tmp_path, ["a", "b"], [("a", "b")], 2, {}))
    row = next(csv.DictReader((tmp_path / "edges.csv").open()))
    assert float(row["ddg_kcal_mol"]) == pytest.approx(3.0)
    assert float(row["complex_dg"]) - float(row["solvent_dg"]) == pytest.approx(3.0)


def test_a_repeat_whose_legs_differ_gives_no_ddg_and_keeps_its_legs(tmp_path):
    """Second review of PR #17: the edge had no ddG when its legs were made at 310 K and 298 K, but the repeat kept
    the difference of the two as its ddG in summary.json. The legs of a repeat are compared before they are
    subtracted; such a repeat has its two legs and what differs, and no ddG."""
    from cufep.rbfe.run import combine_network

    _write_legs(tmp_path, "a~b", 0, _leg_result(-8.0, temperature=310.0), _leg_result(-9.0))
    repeat = combine_network(tmp_path, ["a", "b"], [("a", "b")], 1, {})["edges"][0]["repeats"][0]
    assert "ddg" not in repeat and "se" not in repeat
    assert repeat["complex_dg"] == pytest.approx(-8.0) and repeat["solvent_dg"] == pytest.approx(-9.0)
    assert any("temperature" in item for item in repeat["inconsistent_legs"])


def test_edges_of_other_conditions_give_their_ddg_but_no_network(tmp_path):
    """Second review of PR #17: A~B run at 298 K and B~C at 310 K (--edge, one directory): each edge passed its own
    check, and the dG of A, B and C were made from both. Each edge gives its ddG; the dG and the cycle closures are
    made only from edges of one set of conditions."""
    from cufep.rbfe.run import analyse_directory, combine_network, write_run_files  # noqa: F401

    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    _write_legs(tmp_path, "b~c", 0, _leg_result(-8.0, temperature=310.0), _leg_result(-9.0, temperature=310.0))
    _write_legs(tmp_path, "a~c", 0, _leg_result(-7.0), _leg_result(-6.0))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], 1, {"a": -9.0})
    edges = {e["edge"]: e for e in summary["edges"]}
    assert edges["a~b"]["ddg"] == pytest.approx(-1.5) and edges["b~c"]["ddg"] == pytest.approx(1.0)
    assert summary["dg"] is None and summary["cycle_closures"] == []
    assert any("temperature" in item for item in summary["inconsistent_edges"])


def test_the_lowest_overlap_in_the_table_is_that_of_all_repeats_of_the_ddg(tmp_path):
    """Second review of PR #17: the table showed the lowest overlap of the first repeat (0.12) next to NOT RELIABLE
    from the second (0.001)."""
    import csv

    from cufep.rbfe.run import combine_network, write_results

    _write_legs(tmp_path, "a~b", 0, _leg_result(1.0, overlap=0.12), _leg_result(0.0))
    _write_legs(tmp_path, "a~b", 1, _leg_result(1.2, overlap=0.001), _leg_result(0.0))
    write_results(tmp_path, combine_network(tmp_path, ["a", "b"], [("a", "b")], 2, {}))
    row = next(csv.DictReader((tmp_path / "edges.csv").open()))
    assert float(row["lowest_overlap_complex"]) == pytest.approx(0.001) and row["not_reliable"] == "complex"


def test_a_run_directory_is_analysed_whatever_its_initial_windows_would_be_refused_for(tmp_path):
    """Second review of PR #17: `cufep rbfe analyze` validated the whole stored configuration, so a directory that an
    earlier version ran with nine initial windows and max_windows 2 (refused now) was not analysed although its
    legs had their results. It reads the number of repeats only; a new run is refused as before."""
    import json

    import yaml

    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import analyse_directory

    stored = {"system": {"protein": "p.pdb", "ligands": "l.sdf"},
              "lambda_schedule": {"adaptive": {"max_windows": 2}}, "sampling": {"repeats": 1}}
    with pytest.raises(ValueError, match="max_windows"):
        RbfeConfig.model_validate(stored)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(stored))
    (tmp_path / "network.json").write_text(json.dumps({"ligands": ["a", "b"], "edges": [["a", "b"]],
                                                       "experiment_kcal_mol": {}}))
    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    assert analyse_directory(tmp_path)["edges"][0]["ddg"] == pytest.approx(-1.5)


def test_a_leg_of_this_run_without_a_free_energy_is_a_failure_of_the_run():
    """Second review of PR #17: a leg whose windows were not connected has no free energy (null, no error) and the
    run ended with 0. A leg of this run without a finite free energy is a failure of the run, as one with an error;
    a leg that this run did not run is not."""
    from cufep.rbfe.run import failed_jobs

    results = [{"edge": "a~b", "leg": "solvent", "repeat": 0, "error": "boom"},
               {"edge": "a~b", "leg": "complex", "repeat": 0, "delta_g": None},
               {"edge": "b~c", "leg": "complex", "repeat": 0, "delta_g": float("nan")},
               {"edge": "b~c", "leg": "solvent", "repeat": 0, "delta_g": -3.0}]
    assert failed_jobs(results) == ["a~b/r0/solvent: boom", "a~b/r0/complex: no free energy",
                                    "b~c/r0/complex: no free energy"]



def _prepared(out_dir, edge: str, ligand_a: str, ligand_b: str) -> None:
    import json

    d = out_dir / "edges" / edge / "prep"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prep.json").write_text(json.dumps({"made_from": {"ligand_a": ligand_a, "ligand_b": ligand_b}}))


def test_a_ligand_of_one_name_in_two_edges_must_be_one_molecule_for_a_network(tmp_path):
    """Third review of PR #17: A~B was run, then B was given another protonation state under its name and B~C was
    run in the same directory: the two B were one node, and the dG of A, B and C were made from both. The records of
    the prepared edges name the molecule of each end; one name of two molecules gives the ddG of each edge and no
    network."""
    from cufep.rbfe.run import combine_network, result_lines

    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    _write_legs(tmp_path, "b~c", 0, _leg_result(-8.0), _leg_result(-9.0))
    _prepared(tmp_path, "a~b", "digest-a", "digest-b1")
    _prepared(tmp_path, "b~c", "digest-b2", "digest-c")
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {})
    assert [e.get("ddg") for e in summary["edges"]] == [pytest.approx(-1.5), pytest.approx(1.0)]
    assert summary["dg"] is None and any("ligand b" in item for item in summary["inconsistent_edges"])
    # the reason is said once (third review of PR #17: it was printed twice)
    lines = result_lines(summary)
    assert sum("NO NETWORK" in line for line in lines) == 1


def test_the_table_gives_the_overlap_among_the_samples_of_the_estimate(tmp_path):
    """Third review of PR #17: the check behind NOT RELIABLE is made on the overlap of all iterations and on that of
    the samples MBAR estimates from; with 0.12 and 0.001 the table gave 0.12 next to NOT RELIABLE. It gives both."""
    import csv

    from cufep.rbfe.run import combine_network, write_results

    complex_ = _leg_result(1.0, overlap=0.12)
    complex_["quality"]["overlap"].update(ok=False, used={"lowest": 0.001, "ok": False})
    _write_legs(tmp_path, "a~b", 0, complex_, _leg_result(0.0))
    write_results(tmp_path, combine_network(tmp_path, ["a", "b"], [("a", "b")], 1, {}))
    row = next(csv.DictReader((tmp_path / "edges.csv").open()))
    assert float(row["lowest_overlap_complex"]) == pytest.approx(0.12)
    assert float(row["lowest_overlap_used_complex"]) == pytest.approx(0.001) and row["not_reliable"] == "complex"


def test_a_compound_combines_its_states_by_their_populations_and_its_symmetry():
    """dG of a compound = -kT ln sum P exp(-dG_state/kT) - kT ln s, against synthetic truth: two binding modes (P 1),
    two protonation forms (P 0.9 and 0.1), a symmetry number 2; the constant is set by the compounds with experimental
    values, and the states get it too; the standard error is that of the weighted states (here, a tree of edges)."""
    from cufep.rbfe.network import Edge, compound_dg

    kt = 0.5925
    truth = {"x_up": -10.0, "x_down": -9.2, "y_n": -11.0, "y_d": -12.5, "z": -8.0}
    pairs = [("x_up", "x_down"), ("x_up", "y_n"), ("y_n", "y_d"), ("y_n", "z")]
    edges = [Edge(a, b, truth[b] - truth[a], 0.2) for a, b in pairs]
    compound_of = {"x_up": "x", "x_down": "x", "y_n": "y", "y_d": "y", "z": "z"}
    fraction = {"y_n": 0.9, "y_d": 0.1}
    symmetry = {"z": 2}

    def exact(c):
        members = [s for s in truth if compound_of[s] == c]
        return -kt * math.log(sum(fraction.get(s, 1.0) * math.exp(-truth[s] / kt) for s in members)) \
            - kt * math.log(symmetry.get(c, 1))

    experiment = {"x": exact("x") + 0.4, "y": exact("y") - 0.4}         # mean of x and y as their exact mean
    out = compound_dg(list(truth), edges, compound_of, experiment, fraction, symmetry, kt)
    assert out["reference"] == "mean of the experimental values" and out["reference_compounds"] == ["x", "y"]
    for c in ("x", "y", "z"):
        assert out["compounds"][c]["dg"] == pytest.approx(exact(c), abs=1e-9)
    for s, v in truth.items():
        assert out["states"][s]["dg"] == pytest.approx(v, abs=1e-9)
    w = out["compounds"]["y"]["weights"]
    assert w["y_d"] == pytest.approx(0.1 * math.exp(1.5 / kt) / (0.9 + 0.1 * math.exp(1.5 / kt)))
    assert out["compounds"]["z"]["symmetry_number"] == 2
    # the error of the compounds against the mean of x and y, by the linearised weights and the covariance
    from cufep.rbfe.network import mle

    f, cov = mle(list(truth), edges)
    names = list(truth)
    gx = np.array([out["compounds"]["x"]["weights"].get(s, 0.0) for s in names])
    gy = np.array([w.get(s, 0.0) for s in names])
    gz = np.array([1.0 if s == "z" else 0.0 for s in names])
    d = gz - (gx + gy) / 2
    assert out["compounds"]["z"]["se"] == pytest.approx(math.sqrt(d @ cov @ d))


def test_states_of_a_compound_are_read_and_their_contradictions_refused(tmp_path):
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import compound_experiment, states_of
    from cufep.workflow import WorkflowError

    class Lig:
        def __init__(self, name):
            self.name = name

    def config(text):
        (tmp_path / "states.csv").write_text(text)
        return RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf",
                                                     "states": str(tmp_path / "states.csv")}})

    good = ("ligand,compound,form,solution_fraction,symmetry_number\n"
            "a_n,a,neutral,0.8,1\na_d,a,deprotonated,0.2,1\nb,b,,1,2\nb_flip,b,,1,2\n")
    ligands = [Lig(n) for n in ("a_n", "a_d", "b")]
    states = states_of(config(good), ligands)
    assert states["compound"] == {"a_n": "a", "a_d": "a", "b": "b"} and states["symmetry"] == {"a": 1, "b": 2}
    assert states["fraction"]["a_d"] == 0.2 and states["left_out"] == {"b": ["b_flip"]}
    for text, message in ((good.replace("0.2,1", "0.3,1"), "add up"), (good.replace("b_flip,b,,1,2", "b_flip,b,,1,3"),
                                                                         "two symmetry numbers"),
                          (good.replace("b,b,,1,2\n", ""), "does not name the ligands b")):
        with pytest.raises(WorkflowError, match=message):
            states_of(config(text), ligands)
    assert compound_experiment(states, {"a_n": -9.0, "a_d": -9.0}) == {"a": -9.0}
    with pytest.raises(WorkflowError, match="other experimental values"):
        compound_experiment(states, {"a_n": -9.0, "a_d": -8.0})


def test_a_run_of_states_gives_the_dg_of_its_compounds_again_from_its_directory(tmp_path):
    """The network of a run with states records them: `cufep rbfe analyze` (analyse_directory) gives the same
    compounds as the run, and compounds.csv has them."""
    import csv
    import json

    from cufep.rbfe.run import analyse_directory, combine_network, write_results

    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    _write_legs(tmp_path, "b~c", 0, _leg_result(-8.0), _leg_result(-9.0))
    _write_legs(tmp_path, "a~c", 0, _leg_result(-7.0), _leg_result(-6.5))
    states = {"compound": {"a": "A", "b": "A", "c": "C"}, "form": {"a": "", "b": "", "c": ""},
              "fraction": {"a": 1.0, "b": 1.0, "c": 1.0}, "symmetry": {"A": 1, "C": 2}}
    experiment = {"a": -9.0, "b": -9.0, "c": -8.0}
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], 1, experiment, states,
                              300.0)
    assert set(summary["compounds"]["values"]) == {"A", "C"}
    # two states of one compound have no experimental ddG between them (+0.00 was given), the others have theirs
    edges = {e["edge"]: e for e in summary["edges"]}
    assert "exp_ddg" not in edges["a~b"] and edges["b~c"]["exp_ddg"] == pytest.approx(1.0)
    assert summary["compounds"]["values"]["C"]["symmetry_number"] == 2
    write_results(tmp_path, summary)
    (tmp_path / "resolved_config.yaml").write_text("sampling: {repeats: 1}\n")
    (tmp_path / "network.json").write_text(json.dumps({"ligands": ["a", "b", "c"],
                                                       "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                                                       "experiment_kcal_mol": experiment, "states": states,
                                                       "temperature_K": 300.0}))
    again = analyse_directory(tmp_path)
    for c in ("A", "C"):
        for key in ("dg", "se"):
            assert again["compounds"]["values"][c][key] == pytest.approx(summary["compounds"]["values"][c][key])
    rows = {r["compound"]: r for r in csv.DictReader((tmp_path / "compounds.csv").open())}
    assert float(rows["C"]["dg_kcal_mol"]) == pytest.approx(summary["compounds"]["values"]["C"]["dg"], abs=1e-3)


def test_the_experiment_of_an_edge_between_states_is_said_to_be_that_of_their_compounds(tmp_path):
    """Second review of PR #19 (P2): `cufep rbfe run` and `analyze` gave an edge between states of two compounds the
    experimental ddG of the compounds as "experiment" in edges.csv and the RESULT lines, beside the ddG of the edge,
    which is another quantity (kT ln 2 apart for two equal binding modes). It is named that of the compounds; an edge
    of a run without states keeps "experiment" of its ligands."""
    import csv

    from cufep.rbfe.run import combine_network, result_lines, write_results

    _write_legs(tmp_path, "a~c", 0, _leg_result(-7.0), _leg_result(-6.5))
    _write_legs(tmp_path, "b~c", 0, _leg_result(-8.0), _leg_result(-9.0))
    states = {"compound": {"a": "A", "b": "A", "c": "C"}, "form": {"a": "", "b": "", "c": ""},
              "fraction": {"a": 1.0, "b": 1.0, "c": 1.0}, "symmetry": {"A": 1, "C": 1}}
    experiment = {"a": -9.0, "b": -9.0, "c": -8.0}
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "c"), ("b", "c")], 1, experiment, states, 300.0)
    assert {e["edge"]: e["exp_ddg_of"] for e in summary["edges"]} == {"a~c": "the compounds", "b~c": "the compounds"}
    write_results(tmp_path, summary)
    rows = {r["edge"]: r for r in csv.DictReader((tmp_path / "edges.csv").open())}
    assert rows["a~c"]["exp_ddg_kcal_mol"] == "1.000" and rows["a~c"]["exp_ddg_of"] == "the compounds"
    line = next(x for x in result_lines(summary) if x.startswith("RESULT a~c"))
    assert "(experiment of the compounds A -> C +1.00)" in line
    plain = tmp_path / "plain"
    _write_legs(plain, "a~c", 0, _leg_result(-7.0), _leg_result(-6.5))
    summary = combine_network(plain, ["a", "c"], [("a", "c")], 1, experiment)
    assert summary["edges"][0]["exp_ddg_of"] == "the ligands"
    assert "(experiment +1.00)" in next(x for x in result_lines(summary) if x.startswith("RESULT a~c"))


def test_a_custom_core_maps_only_its_atoms_and_the_records_of_other_edges_stay_as_they_were():
    """mapping.core_smarts restricts the mapping to atoms in a match of the pattern in both ligands; without it the
    records of a leg and of a prepared edge are those of earlier versions (the settings added are left out at their
    defaults)."""
    from rdkit import Chem

    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import hybrid_settings
    from cufep.rbfe.mapping import MappingError, core_atoms, within_core

    toluene = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    ring = core_atoms(toluene, "c1ccccc1")
    assert ring == {1, 2, 3, 4, 5, 6}
    mapping = {i: i for i in range(15)}
    assert within_core(mapping, ring, ring) == {i: i for i in ring}
    with pytest.raises(MappingError, match="matches no atom"):
        core_atoms(toluene, "C(=O)N")
    with pytest.raises(MappingError, match="not a SMARTS"):
        core_atoms(toluene, "c1cc(")
    # review of PR #19: the stereochemistry of the pattern is respected; a pattern without it matches either
    same, other = (Chem.AddHs(Chem.MolFromSmiles(s)) for s in ("CC[C@@H](F)Cl", "CC[C@H](F)Cl"))
    assert core_atoms(same, "C[C@@H](F)Cl") == {1, 2, 3, 4}                # the CH2, the stereocentre, F, Cl
    with pytest.raises(MappingError, match="matches no atom"):
        core_atoms(other, "C[C@@H](F)Cl")
    assert core_atoms(other, "CC(F)Cl") == core_atoms(same, "CC(F)Cl")
    # eleventh review of PR #19: the 1296 orders of the atoms of one tert-butyl group used up the 1000 matches of the
    # search, and the other group of the ligand was left out of the core; a match is a set of atoms
    butyls = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)CCC(C)(C)C"))
    group = "C([CH3]([H])([H])[H])([CH3]([H])([H])[H])[CH3]([H])([H])[H]"
    assert len(core_atoms(butyls, group)) == 26
    # a search that reaches its limit is refused, not cut
    import cufep.rbfe.mapping as mapping_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mapping_module, "MAX_CORE_MATCHES", 2)
        assert len(core_atoms(butyls, group)) == 26                         # two matches, the limit
        with pytest.raises(MappingError, match="more than 2 sets of atoms"):  # six methyl groups
            core_atoms(butyls, "[CH3]")
    plain = RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf"}})
    recorded = hybrid_settings(plain, "solvent")["rbfe"]
    assert "core_smarts" not in recorded["mapping"] and "alchemical_water_cutoff_nm" not in recorded["alchemy"]
    custom = RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf"},
                                        "mapping": {"core_smarts": "c1ccccc1"}})
    assert hybrid_settings(custom, "solvent")["rbfe"]["mapping"]["core_smarts"] == "c1ccccc1"
    # review of PR #19: the cutoff of the alchemical water is of the preparation of an edge, never of a leg's record
    other = RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf"},
                                       "alchemy": {"alchemical_water_cutoff_nm": 1.0}})
    assert hybrid_settings(other, "solvent") == hybrid_settings(plain, "solvent")


def test_a_directory_analysed_without_states_keeps_no_compounds_of_before(tmp_path):
    """Review of PR #19: compounds.csv of an earlier run with states stayed next to results without compounds."""
    from cufep.rbfe.run import combine_network, write_results

    _write_legs(tmp_path, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    (tmp_path / "compounds.csv").write_text("compound,dg_kcal_mol\nX,-9.0\n")
    write_results(tmp_path, combine_network(tmp_path, ["a", "b"], [("a", "b")], 1, {}))
    assert not (tmp_path / "compounds.csv").exists()


def _with_moves(result: dict, mu: float | None) -> dict:
    """A complex leg with grand canonical moves of water at the excess chemical potential ``mu`` (None: without)."""
    moves = {"enabled": False} if mu is None else {"enabled": True, "excess_chemical_potential_kcal_per_mol": mu,
                                                   "standard_volume_nm3": 0.0303}
    return {**result, "run": {"grand_canonical": moves}, "gcmc": None if mu is None else {"insertions_attempted": [1]}}


def test_complex_legs_in_other_grand_canonical_ensembles_of_water_are_not_combined(tmp_path):
    """Review of PR #19 (P1): A~B run with the moves at -6 kcal/mol and B~C at -4 kcal/mol passed every check and gave
    one set of dG, though B's complex leg is of another ensemble in each. Edges of other moves of water give their
    ddG one by one, without a network; two repeats of one edge in other ensembles give no ddG; edges of one ensemble
    combine; a complex leg of an earlier version without a record of its moves and without counts made none."""
    from cufep.rbfe.run import combine_network, water_moves_of

    _write_legs(tmp_path, "a~b", 0, _with_moves(_leg_result(-12.0), -6.0), _leg_result(-10.5))
    _write_legs(tmp_path, "b~c", 0, _with_moves(_leg_result(-8.0), -4.0), _leg_result(-9.0))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {})
    assert summary["dg"] is None and any("grand canonical" in item for item in summary["inconsistent_edges"])
    assert {e["edge"]: "ddg" in e for e in summary["edges"]} == {"a~b": True, "b~c": True}
    same = tmp_path / "same"
    _write_legs(same, "a~b", 0, _with_moves(_leg_result(-12.0), -6.0), _leg_result(-10.5))
    _write_legs(same, "b~c", 0, _with_moves(_leg_result(-8.0), -6.0), _leg_result(-9.0))
    assert combine_network(same, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {})["dg"] is not None
    repeats = tmp_path / "repeats"
    _write_legs(repeats, "a~b", 0, _with_moves(_leg_result(-12.0), -6.0), _leg_result(-10.5))
    _write_legs(repeats, "a~b", 1, _with_moves(_leg_result(-11.0), -4.0), _leg_result(-10.5))
    edge = combine_network(repeats, ["a", "b"], [("a", "b")], 2, {})["edges"][0]
    assert "ddg" not in edge and any("grand canonical" in item for item in edge["inconsistent_legs"])
    assert water_moves_of(_leg_result(-1.0)) == {"enabled": False}
    assert water_moves_of({**_leg_result(-1.0), "gcmc": {"insertions_attempted": [1]}}) == {"enabled": True}


def test_a_first_repeat_of_an_earlier_version_does_not_let_others_of_other_ensembles_pass(tmp_path, monkeypatch):
    """Second review of PR #19 (P1): repeat 0 of an earlier version (counts of moves, no record of their settings),
    repeats 1 and 2 at -6 and -4 kcal/mol: the first state was unknown, so neither was compared and the three were
    averaged. All known states are compared with each other; a result of an earlier version has the settings its
    samples recorded, else at least whether it made moves."""
    from cufep.rbfe.run import combine_network, water_moves_of
    from cufep.sampling import storage

    # (an earlier version without REST2, so that its hybrid is that of this version)
    old = {**_leg_result(-12.0), "run": {"rest2": {"scale": 1.0}}, "gcmc": {"insertions_attempted": [1]}}
    _write_legs(tmp_path, "a~b", 0, old, _leg_result(-10.5))
    _write_legs(tmp_path, "a~b", 1, _with_moves(_leg_result(-11.0), -6.0), _leg_result(-10.5))
    _write_legs(tmp_path, "a~b", 2, _with_moves(_leg_result(-11.5), -4.0), _leg_result(-10.5))
    edge = combine_network(tmp_path, ["a", "b"], [("a", "b")], 3, {})["edges"][0]
    assert "ddg" not in edge and any("excess chemical potential" in item for item in edge["inconsistent_legs"])
    # without moves against with them, whatever is known of the numbers
    mixed = tmp_path / "mixed"
    _write_legs(mixed, "a~b", 0, old, _leg_result(-10.5))
    _write_legs(mixed, "a~b", 1, _with_moves(_leg_result(-11.0), None), _leg_result(-10.5))
    assert "ddg" not in combine_network(mixed, ["a", "b"], [("a", "b")], 2, {})["edges"][0]
    # the settings that the samples of a result of an earlier version recorded
    monkeypatch.setattr(storage, "stored_conditions", lambda directory: (10, {"settings": {"gcmc": {
        "excess_chemical_potential_kcal_per_mol": -6.0, "standard_volume_nm3": 0.0303}}}))
    assert water_moves_of(old, tmp_path) == {"enabled": True, "excess_chemical_potential_kcal_per_mol": -6.0,
                                             "standard_volume_nm3": 0.0303}


def test_the_experiment_of_a_state_in_ligands_csv_is_said_to_be_that_of_its_compound(tmp_path):
    """Third review of PR #19 (P3): ligands.csv gave the two states 4jS and 4jR the experimental dG of their compound
    4j (-10.887 kcal/mol) each, as if each had been measured; a statistic over the states counted one measurement
    twice. The column exp_dg_of names the compound; the ligands of a run without states keep "the ligand"."""
    import csv

    from cufep.rbfe.run import combine_network, write_results

    _write_legs(tmp_path, "4jS~4jR", 0, _leg_result(-12.0), _leg_result(-11.5))
    states = {"compound": {"4jS": "4j", "4jR": "4j"}, "form": {"4jS": "", "4jR": ""},
              "fraction": {"4jS": 1.0, "4jR": 1.0}, "symmetry": {"4j": 1}}
    experiment = {"4jS": -10.887, "4jR": -10.887}
    write_results(tmp_path, combine_network(tmp_path, ["4jS", "4jR"], [("4jS", "4jR")], 1, experiment, states))
    rows = {r["ligand"]: r for r in csv.DictReader((tmp_path / "ligands.csv").open())}
    assert {name: (r["exp_dg_kcal_mol"], r["exp_dg_of"]) for name, r in rows.items()} == {
        "4jS": ("-10.887", "the compound 4j"), "4jR": ("-10.887", "the compound 4j")}
    plain = tmp_path / "plain"
    _write_legs(plain, "a~b", 0, _leg_result(-12.0), _leg_result(-10.5))
    write_results(plain, combine_network(plain, ["a", "b"], [("a", "b")], 1, {"a": -9.0}))
    rows = {r["ligand"]: r for r in csv.DictReader((plain / "ligands.csv").open())}
    assert (rows["a"]["exp_dg_of"], rows["b"]["exp_dg_of"]) == ("the ligand", "")


def test_a_mapping_whose_dummy_atoms_join_two_mapped_atoms_is_refused(tmp_path):
    """Eleventh review of PR #19 (P1): propane and 2,2-difluoropropane with the custom core [CH3] map the two methyl
    carbons alone; the central carbon of each end, unique to it, stays bonded to both in the hybrid, and its bonds and
    angle hold them at both ends of the transformation. Such a mapping is refused (and left out of the networks LOMAP
    plans); a mapping whose unique atoms hang from one mapped atom is taken."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from rdkit.Geometry import Point3D

    from cufep.prep.ligand import load_ligands
    from cufep.rbfe.mapping import MappingError, MappingSettings, bridging_dummies, kartograf_mapping

    a = Chem.AddHs(Chem.MolFromSmiles("CCC"))
    AllChem.EmbedMolecule(a, randomSeed=3)
    b = Chem.AddHs(Chem.MolFromSmiles("CC(F)(F)C"))
    xyz = a.GetConformer().GetPositions()
    AllChem.EmbedMolecule(b, randomSeed=3, coordMap={0: Point3D(*xyz[0]), 1: Point3D(*xyz[1]), 4: Point3D(*xyz[2])})
    rdMolAlign.AlignMol(b, a, atomMap=[(0, 0), (1, 1), (4, 2)])
    writer = Chem.SDWriter(str(tmp_path / "l.sdf"))
    for name, mol in (("propane", a), ("difluoropropane", b)):
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    propane, difluoro = load_ligands(tmp_path / "l.sdf")
    with pytest.raises(MappingError, match="unique to A are bonded to the mapped atoms"):
        kartograf_mapping(propane, difluoro, MappingSettings(core_smarts="[CH3]"))
    whole = kartograf_mapping(propane, difluoro, MappingSettings())
    assert bridging_dummies(propane.to_rdkit(), difluoro.to_rdkit(), whole) == []


def test_rows_of_states_that_contradict_each_other_or_fractions_outside_the_run_are_refused(tmp_path):
    """Eleventh review of PR #19: (P2) the rows a,A / b,A / a,B became two compounds, and in the other order one: a
    state named on two rows that say other things is refused, with the lines; (P3) the forms of a compound with
    populations 0.6, 1.2 and -0.8 add up to 1 and passed with the first state alone: the population of every row of a
    compound of the run is checked."""
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import states_of
    from cufep.workflow import WorkflowError

    class Lig:
        def __init__(self, name):
            self.name = name

    def config(text):
        (tmp_path / "states.csv").write_text(text)
        return RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf",
                                                     "states": str(tmp_path / "states.csv")}})

    for text in ("ligand,compound\na,A\nb,A\na,B\n", "ligand,compound\na,B\nb,A\na,A\n"):
        with pytest.raises(WorkflowError, match="names states twice with other values: a on lines 2 and 4"):
            states_of(config(text), [Lig("a"), Lig("b")])
    assert states_of(config("ligand,compound\na,A\nb,A\na,A\n"), [Lig("a"), Lig("b")])["compound"] == {
        "a": "A", "b": "A"}                                              # the same row twice says nothing other
    forms = ("ligand,compound,form,solution_fraction\n"
             "x1,X,one,0.6\nx2,X,two,1.2\nx3,X,three,-0.8\n")
    with pytest.raises(WorkflowError, match="x2: solution_fraction 1.2 is not in"):
        states_of(config(forms), [Lig("x1")])
    # twelfth review (P3): rows that say the same in other words are one state (1 and 1.0, a blank and the default)
    for text in ("ligand,compound,solution_fraction\na,A,1\na,A,1.0\nb,B,1\n",
                 "ligand,compound,solution_fraction,symmetry_number\na,A,,\na,A,1,1\nb,B,1.0,1.0\n"):
        assert states_of(config(text), [Lig("a"), Lig("b")])["compound"] == {"a": "A", "b": "B"}
    with pytest.raises(WorkflowError, match="names states twice with other values"):
        states_of(config("ligand,compound,solution_fraction\na,A,1\na,A,0.5\nb,B,1\n"), [Lig("a"), Lig("b")])
    with pytest.raises(WorkflowError, match="is not a positive whole number"):
        states_of(config("ligand,compound,symmetry_number\na,A,1.5\nb,B,1\n"), [Lig("a"), Lig("b")])


def test_a_state_takes_the_experiment_of_its_compound_and_the_states_are_combined_at_the_temperature_of_their_legs(
        tmp_path):
    """Eleventh review of PR #19: (P2) with A1 = -10 and B1 = -9 kcal/mol the edge A2 -> B1 had no experimental ddG
    and A2 no value in ligands.csv, though compounds.csv had both: a state takes the value of its compound; (P2) the
    record of a later run at 310 K combined states whose legs were made at 298.15 K: they are combined at the
    temperature of their legs, and a note says so; (P2) an edge with results whose record of the prepared systems is
    missing is left out."""
    import csv

    from cufep.rbfe.run import combine_network, write_results

    _write_legs(tmp_path, "A2~B1", 0, _leg_result(-12.0), _leg_result(-10.5))
    _write_legs(tmp_path, "A1~A2", 0, _leg_result(-8.0), _leg_result(-8.0))
    states = {"compound": {"A1": "A", "A2": "A", "B1": "B"}, "form": {"A1": "", "A2": "", "B1": ""},
              "fraction": {"A1": 1.0, "A2": 1.0, "B1": 1.0}, "symmetry": {"A": 1, "B": 1}}
    summary = combine_network(tmp_path, ["A1", "A2", "B1"], [("A2", "B1"), ("A1", "A2")], 1,
                              {"A1": -10.0, "B1": -9.0}, states, 310.0)
    edges = {e["edge"]: e for e in summary["edges"]}
    assert edges["A2~B1"]["exp_ddg"] == pytest.approx(1.0) and "exp_ddg" not in edges["A1~A2"]
    assert summary["compounds"]["temperature_K"] == pytest.approx(298.15)
    assert any("combined at 298.15 K" in note for note in summary["notes"])
    write_results(tmp_path, summary)
    rows = {r["ligand"]: r for r in csv.DictReader((tmp_path / "ligands.csv").open())}
    assert rows["A2"]["exp_dg_kcal_mol"] == "-10.000" and rows["A2"]["exp_dg_of"] == "the compound A"
    # twelfth review (P2): legs at 298.15 and 298.15000001 K, one temperature by the rule that compares them, were two,
    # and the record of the run (310 K) was taken
    import json

    leg = tmp_path / "edges" / "A1~A2" / "repeat_0" / "complex" / "result.json"
    leg.write_text(json.dumps(_leg_result(-8.0, temperature=298.15000001)))
    (tmp_path / "edges" / "A1~A2" / "repeat_0" / "solvent" / "result.json").write_text(
        json.dumps(_leg_result(-8.0, temperature=298.15000001)))
    nearly = combine_network(tmp_path, ["A1", "A2", "B1"], [("A2", "B1"), ("A1", "A2")], 1,
                             {"A1": -10.0, "B1": -9.0}, states, 310.0)
    assert nearly["compounds"]["temperature_K"] == pytest.approx(298.15, abs=1e-6)
    assert nearly["compounds"]["values"]["A"]["dg"] == pytest.approx(summary["compounds"]["values"]["A"]["dg"],
                                                                     abs=1e-6)
    (tmp_path / "edges" / "A1~A2" / "prep" / "prep.json").unlink()
    edges = {e["edge"]: e for e in combine_network(tmp_path, ["A1", "A2", "B1"], [("A2", "B1"), ("A1", "A2")], 1,
                                                   {"A1": -10.0, "B1": -9.0}, states, 298.15)["edges"]}
    assert "ddg" not in edges["A1~A2"] and "is missing" in edges["A1~A2"]["earlier_preparation"][0]


def test_the_experiment_of_a_compound_is_taken_from_its_states_the_run_leaves_out(tmp_path):
    """Twelfth review of PR #19 (P2): A1 and A2 states of compound A, B1 of B, the experiment A1 = -10 and B1 = -9; a
    run of A2 and B1 (system.ligand_names) dropped the value of A1 before the compounds were given theirs, and A2 -> B1
    had no experimental ddG and the dG one compound of reference. The values of the states the run leaves out are
    read too."""
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import compound_experiment, experiment_of, states_of

    class Lig:
        def __init__(self, name):
            self.name, self.properties = name, {}

    (tmp_path / "states.csv").write_text("ligand,compound\nA1,A\nA2,A\nB1,B\n")
    (tmp_path / "exp.csv").write_text("ligand,dg\nA1,-10\nB1,-9\n")
    rcfg = RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf",
                                                 "states": str(tmp_path / "states.csv"),
                                                 "experiment": str(tmp_path / "exp.csv")}})
    ligands = [Lig("A2"), Lig("B1")]
    states = states_of(rcfg, ligands)
    assert states["left_out"] == {"A": ["A1"]}
    assert compound_experiment(states, experiment_of(rcfg, ligands, states)) == {"A": -10.0, "B": -9.0}


def test_the_ions_of_the_legs_are_compared_by_their_parameters_and_a_custom_core_of_rule_2_is_checked():
    """Twelfth review of PR #19: (P2) a Na+ of the system in the complex leg and the Na+ of the force field in the
    solvent leg, of the same charge, sigma and epsilon, were taken as other ions and the edge was left out; where they
    were taken from is not compared. (P1) an edge whose custom core version 2 of the rule found is taken where both
    rules give the same core of its ligands, and left out where the search of version 2 stopped before a match."""
    from rdkit import Chem

    from cufep.rbfe.prep import custom_core_rule_holds, earlier_rules_of

    na = {"name": "NA", "charge": 1.0, "sigma_nm": 0.2439, "epsilon_kj_mol": 0.3658}
    info = {"made_from": {"alchemical_water_rule": 2}, "legs": {
        "complex": {"mapping": {}, "alchemical_water": {"ion": {**na, "from": "the system"}}},
        "solvent": {"mapping": {}, "alchemical_water": {"ion": {**na, "from": "the force field"}}}}}
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert earlier_rules_of(info, (mol, mol)) == []
    info["legs"]["solvent"]["alchemical_water"]["ion"] = {**na, "sigma_nm": 0.3, "from": "the force field"}
    assert any("other ions" in why for why in earlier_rules_of(info, (mol, mol)))
    toluene = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    butyls = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)CCC(C)(C)C"))
    group = "C([CH3]([H])([H])[H])([CH3]([H])([H])[H])[CH3]([H])([H])[H]"
    assert custom_core_rule_holds({"custom_core_rule": 2, "mapping": {"core_smarts": "c1ccccc1"}}, (toluene, toluene))
    assert not custom_core_rule_holds({"custom_core_rule": 2, "mapping": {"core_smarts": group}}, (butyls, butyls))
    of_rule_2 = {"made_from": {"custom_core_rule": 2, "mapping": {"core_smarts": group}}, "legs": {}}
    assert any("earlier rule" in why for why in earlier_rules_of(of_rule_2, (butyls, butyls)))
