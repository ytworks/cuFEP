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
    for leg, dg in (("complex", -12.0 * 4.184), ("solvent", -10.5 * 4.184)):
        leg_dir = tmp_path / "edges" / "a~b" / "repeat_0" / leg
        leg_dir.mkdir(parents=True)
        (leg_dir / "result.json").write_text(json.dumps({
            "delta_g": dg, "delta_g_error": 0.3 * 4.184, "n_states": 9, "backend": "openmm", "notes": [],
            "quality": {"overlap": {"lowest": 0.1, "ok": True}, "samples": {"counted": 100.0}}}))
    summary = analyse_directory(tmp_path)
    edge = summary["edges"][0]
    assert edge["ddg"] == pytest.approx(-1.5) and edge["se"] == pytest.approx(0.3 * math.sqrt(2))
    assert summary["dg"]["values"]["b"]["dg"] - summary["dg"]["values"]["a"]["dg"] == pytest.approx(-1.5)


def _leg_result(dg_kcal: float, se_kcal: float | None = 0.3, temperature: float = 298.15, pressure: float = 1.0,
                overlap: float = 0.1):
    return {"delta_g": dg_kcal * 4.184, "delta_g_error": None if se_kcal is None else se_kcal * 4.184,
            "temperature_K": temperature, "pressure_bar": pressure, "lrc_method": "none",
            "prepared_systems_sha256": "abc", "n_states": 9, "backend": "openmm", "notes": [],
            "quality": {"overlap": {"lowest": overlap, "ok": overlap >= 0.03}, "samples": {"counted": 100.0}}}


def _write_legs(out_dir, edge: str, repeat: int, complex_, solvent) -> None:
    import json

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


RULE = "velocities_of_an_inserted_water"


def _moves(mu: float = -6.09, volume: float = 0.030345, rule: str | None = "turned with it") -> dict:
    """What a complex leg with the moves records of them: the values of the water and the rule of the mover (None:
    samples of a version before the record of the rule)."""
    return {"enabled": True, "excess_chemical_potential_kcal_per_mol": mu, "standard_volume_nm3": volume, RULE: rule}


NO_MOVES = {"enabled": False}


def _with_moves(dg_kcal: float, moves: dict | None, **more) -> dict:
    """A leg result as this version writes it: the counts of the moves (None without them) and what the samples
    were made with."""
    result = {**_leg_result(dg_kcal, **more), "gcmc": {"insertions_accepted": [3]} if (moves or {}).get("enabled")
              else None}
    if moves is not None:
        result["water_moves"] = moves
    return result


def test_complex_legs_of_other_grand_canonical_moves_of_water_are_not_combined(tmp_path):
    """Review of PR #21 (P1): `--edge a~b`, then another sampling.gcmc.excess_chemical_potential_kcal_per_mol and
    `--edge b~c` in the same directory: the samples of a~b were not looked at again, the conditions compared among
    edges were temperature, pressure and the tail correction, and the dG of a, b and c were estimated from edges of
    two activities of water. Each edge gives its ddG; no dG and no cycle closure is made of them. The same for
    edges with and without the moves, and for the repeats of one edge, which then has no ddG."""
    from cufep.rbfe.run import combine_network, result_lines

    _write_legs(tmp_path, "a~b", 0, _with_moves(-12.0, _moves()), _with_moves(-10.5, NO_MOVES))
    _write_legs(tmp_path, "b~c", 0, _with_moves(-8.0, _moves(mu=-5.0)), _with_moves(-9.0, NO_MOVES))
    _write_legs(tmp_path, "a~c", 0, _with_moves(-7.0, _moves()), _with_moves(-6.0, NO_MOVES))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], 1, {"a": -9.0})
    edges = {e["edge"]: e for e in summary["edges"]}
    assert edges["a~b"]["ddg"] == pytest.approx(-1.5) and edges["b~c"]["ddg"] == pytest.approx(1.0)
    assert edges["a~b"]["conditions"]["water_moves"] == _moves()
    assert summary["dg"] is None and summary["cycle_closures"] == []
    (differing,) = summary["inconsistent_edges"]
    assert differing == ("grand canonical moves of water in the complex leg: a~b excess chemical potential -6.09 "
                         "kcal/mol, standard volume 0.030345 nm3, b~c excess chemical potential -5 kcal/mol, standard "
                         "volume 0.030345 nm3, a~c excess chemical potential -6.09 kcal/mol, standard volume 0.030345 "
                         "nm3")
    assert any(line.startswith("note: NO NETWORK") for line in result_lines(summary))
    # with and without the moves
    other = tmp_path / "with_and_without"
    _write_legs(other, "a~b", 0, _with_moves(-12.0, _moves()), _with_moves(-10.5, NO_MOVES))
    _write_legs(other, "b~c", 0, _with_moves(-8.0, NO_MOVES), _with_moves(-9.0, NO_MOVES))
    summary = combine_network(other, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {})
    assert summary["dg"] is None and "b~c none" in summary["inconsistent_edges"][0]
    # the same moves in every edge: one network
    same = tmp_path / "same"
    _write_legs(same, "a~b", 0, _with_moves(-12.0, _moves()), _with_moves(-10.5, NO_MOVES))
    _write_legs(same, "b~c", 0, _with_moves(-8.0, _moves()), _with_moves(-9.0, NO_MOVES))
    summary = combine_network(same, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {})
    assert summary["dg"] is not None and "inconsistent_edges" not in summary and summary["notes"] == [
        "no ligand has an experimental value: the dG are relative to the mean of the network"]
    # two repeats of one edge
    repeats = tmp_path / "repeats"
    _write_legs(repeats, "a~b", 0, _with_moves(-12.0, _moves()), _with_moves(-10.5, NO_MOVES))
    _write_legs(repeats, "a~b", 1, _with_moves(-12.4, _moves(volume=0.03)), _with_moves(-10.5, NO_MOVES))
    edge = combine_network(repeats, ["a", "b"], [("a", "b")], 2, {})["edges"][0]
    assert "ddg" not in edge
    assert edge["inconsistent_legs"] == [
        "grand canonical moves of water in the complex leg: repeat 0 excess chemical potential -6.09 kcal/mol, "
        "standard volume 0.030345 nm3, repeat 1 excess chemical potential -6.09 kcal/mol, standard volume 0.03 nm3"]


def test_the_water_moves_of_a_leg_are_those_of_its_result_or_else_of_its_samples(tmp_path, monkeypatch):
    """What a complex leg was sampled with is read where it is recorded: in the result of this version; for a
    result of the versions that made the moves and did not record them (it has their counts), in the conditions of
    its samples; a result of a version before a relative leg could make them (no ``gcmc`` at all) was made
    without. A result with counts and without samples was made with the moves, with values of the water that are
    not recorded."""
    from cufep.rbfe.run import water_moves_differ, water_moves_of_leg, water_values_recorded
    from cufep.workflow import water_moves_of

    record = {"enabled": True, "interval_ps": 4.0, "excess_chemical_potential_kcal_per_mol": -6.09,
              "standard_volume_nm3": 0.030345, "cell_pressure_bar": 1.0}
    assert water_moves_of({"settings": {"gcmc": record}}) == _moves(rule=None)
    assert water_moves_of({"settings": {"gcmc": {**record, RULE: "turned with it"}}}) == _moves()
    assert water_moves_of({"settings": {"rbfe": {}}}) == NO_MOVES and water_moves_of({}) == NO_MOVES
    assert water_moves_of(None) is None
    assert water_moves_of_leg({"water_moves": _moves(mu=-5.0), "gcmc": {}}, tmp_path) == _moves(mu=-5.0)
    assert water_moves_of_leg({"delta_g": 1.0}, tmp_path) == NO_MOVES            # a version before the moves
    assert water_moves_of_leg({"gcmc": None}, tmp_path) == NO_MOVES
    made = water_moves_of_leg({"gcmc": {"insertions_accepted": [1]}}, tmp_path)   # no samples to ask
    assert made == {"enabled": True} and not water_values_recorded(made) and water_values_recorded(NO_MOVES)
    assert water_moves_differ(made, NO_MOVES) and not water_moves_differ(made, _moves())
    assert water_moves_differ(_moves(), _moves(mu=-5.0)) and not water_moves_differ(_moves(), _moves())
    import cufep.sampling.storage as storage

    monkeypatch.setattr(storage, "recorded_conditions", lambda directory: {"settings": {"gcmc": record}})
    assert water_moves_of_leg({"gcmc": {"insertions_accepted": [1]}}, tmp_path) == _moves(rule=None)
    # a result of the versions that recorded the values of the water and not the rule of the mover: the rule of
    # its samples (third review of PR #21)
    values = {key: value for key, value in _moves(mu=-5.0).items() if key != RULE}
    assert water_moves_of_leg({"water_moves": values, "gcmc": {}}, tmp_path) == _moves(mu=-5.0, rule=None)
    monkeypatch.setattr(storage, "recorded_conditions", lambda directory: {"settings": {"gcmc": {
        **record, RULE: "turned with it"}}})
    assert water_moves_of_leg({"water_moves": values, "gcmc": {}}, tmp_path) == _moves(mu=-5.0)


def test_a_leg_known_to_have_made_the_moves_is_not_combined_with_one_without_them(tmp_path):
    """Second review of PR #21 (P1): a result of an earlier version with the counts of the moves, whose samples
    are not in the directory, was made with them for certain; it was "not recorded", left out of the comparison,
    and went into one network with an edge whose complex leg made none. Whether the moves were made is compared
    for every edge; the values of the water where they are recorded. An edge with repeats that record them and
    repeats that do not keeps the latter named, with the values of the former."""
    from cufep.rbfe.run import combine_network, result_lines

    counts_only = _with_moves(-8.0, None) | {"gcmc": {"insertions_accepted": [1]}}
    _write_legs(tmp_path, "a~b", 0, counts_only, _with_moves(-9.0, NO_MOVES))
    _write_legs(tmp_path, "b~c", 0, _with_moves(-12.0, NO_MOVES), _with_moves(-10.5, NO_MOVES))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {"a": -9.0})
    assert summary["dg"] is None
    assert summary["inconsistent_edges"] == ["grand canonical moves of water in the complex leg: a~b made, the "
                                             "values of the water not recorded, b~c none"]
    # with an edge that made the moves too: one network, and the note of what was not compared
    known = tmp_path / "known"
    _write_legs(known, "a~b", 0, counts_only, _with_moves(-9.0, NO_MOVES))
    _write_legs(known, "b~c", 0, _with_moves(-12.0, _moves()), _with_moves(-10.5, NO_MOVES))
    summary = combine_network(known, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {"a": -9.0})
    assert summary["dg"] is not None and "inconsistent_edges" not in summary
    assert summary["edges"][0]["conditions"]["water_moves"] == {"enabled": True, RULE: None}
    assert summary["edges"][0]["water_values_not_recorded"] == [0]
    assert summary["edges"][1]["water_values_not_recorded"] == []
    note = ("grand canonical moves of water in the complex leg: made, and the excess chemical potential and the "
            "standard volume of the water not recorded, for a~b (repeat 0) (results of an earlier version whose "
            "samples are not there); those values were not compared for them")
    # (and the note of the rule of the mover, which the result of a~b does not record either)
    assert summary["notes"][0] == note and len(summary["notes"]) == 2 and f"note: {note}" in result_lines(summary)
    # repeats of one edge: one records the values, one does not
    mixed = tmp_path / "mixed"
    _write_legs(mixed, "a~b", 0, _with_moves(-8.0, _moves()), _with_moves(-9.0, NO_MOVES))
    _write_legs(mixed, "a~b", 1, counts_only, _with_moves(-9.0, NO_MOVES))
    _write_legs(mixed, "b~c", 0, _with_moves(-12.0, _moves(mu=-5.0)), _with_moves(-10.5, NO_MOVES))
    summary = combine_network(mixed, ["a", "b", "c"], [("a", "b"), ("b", "c")], 2, {})
    edge = summary["edges"][0]
    assert edge["ddg"] == pytest.approx(1.0) and edge["conditions"]["water_moves"] == _moves(rule=None)
    assert edge["water_values_not_recorded"] == [1]                       # still named
    assert summary["dg"] is None and "a~b excess chemical potential -6.09" in summary["inconsistent_edges"][0]
    assert any("for a~b (repeat 1)" in item for item in summary["notes"])
    # a repeat that made the moves and one that made none: no ddG
    opposite = tmp_path / "opposite"
    _write_legs(opposite, "a~b", 0, counts_only, _with_moves(-9.0, NO_MOVES))
    _write_legs(opposite, "a~b", 1, _with_moves(-8.0, NO_MOVES), _with_moves(-9.0, NO_MOVES))
    edge = combine_network(opposite, ["a", "b"], [("a", "b")], 2, {})["edges"][0]
    assert "ddg" not in edge and edge["inconsistent_legs"] == [
        "grand canonical moves of water in the complex leg: repeat 0 made, the values of the water not recorded, "
        "repeat 1 none"]


def _run_directory(tmp_path, edges: dict) -> None:
    """The record of a run (configuration, network) around legs written by ``_write_legs``."""
    import json

    import yaml

    names = sorted({name for edge in edges for name in edge.split("~")})
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump({"sampling": {"repeats": 1}}))
    (tmp_path / "network.json").write_text(json.dumps({"ligands": names, "edges": [edge.split("~") for edge in edges],
                                                       "experiment_kcal_mol": {}}))
    for edge, (complex_dg, solvent_dg) in edges.items():
        _write_legs(tmp_path, edge, 0, _with_moves(complex_dg, _moves()), _with_moves(solvent_dg, NO_MOVES))
        _prepared(tmp_path, edge, *edge.split("~"))


def test_the_results_of_a_run_are_made_again_only_from_legs_it_has_and_under_its_lock(tmp_path):
    """Second review of PR #21 (P2, twice): `cufep rbfe analyze --write` on a copy of the results without the
    directories of the edges (benchmarks/results/tyk2_rbfe) found no leg, and replaced the ddG of 22 edges and the
    dG of 16 compounds by nothing; and it wrote without the lock of the directory, so that a summary read while a
    leg was running could replace the one the run wrote when the leg had finished. With the legs, the tables are
    made again, with what this version records of an edge."""
    import json
    import shutil

    from cufep.rbfe.run import rewrite_results
    from cufep.run import lock_output_directory
    from cufep.workflow import WorkflowError

    run = tmp_path / "run"
    _run_directory(run, {"a~b": (-12.0, -10.5), "b~c": (-8.0, -9.0)})
    summary = rewrite_results(run)
    assert [e["ddg"] for e in summary["edges"]] == pytest.approx([-1.5, 1.0])
    written = json.loads((run / "summary.json").read_text())
    assert written["edges"][0]["conditions"]["water_moves"] == _moves() and (run / "edges.csv").exists()
    # the lock of the directory: a run that holds it is not written over
    held = lock_output_directory(run)
    before = (run / "summary.json").read_bytes()
    with pytest.raises(WorkflowError, match="another cufep run is using the output directory"):
        rewrite_results(run)
    held.close()
    assert (run / "summary.json").read_bytes() == before
    # a copy of the results without the legs
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in ("summary.json", "edges.csv", "ligands.csv", "network.json", "resolved_config.yaml"):
        shutil.copy(run / name, copy / name)
    before = {path.name: path.read_bytes() for path in copy.iterdir()}
    with pytest.raises(WorkflowError, match="holds the ddG of 2 edge\\(s\\) made from files that are not in the "
                                           "directory \\(a~b: repeat_0/complex/result.json, repeat_0/solvent/"
                                           "result.json, prep/prep.json; b~c: .*Nothing was written.*cufep "
                                           "benchmark report"):
        rewrite_results(copy)
    assert {path.name: path.read_bytes() for path in copy.iterdir() if path.name != ".cufep.lock"} == before
    # one edge whose legs are gone, among edges that have theirs
    shutil.rmtree(run / "edges" / "b~c")
    with pytest.raises(WorkflowError, match="holds the ddG of 1 edge\\(s\\) made from files that are not in the "
                                           "directory \\(b~c: "):
        rewrite_results(run)
    assert (run / "summary.json").read_bytes() == before["summary.json"]


def test_analyze_write_refuses_the_recorded_results_which_have_no_legs(tmp_path):
    """The reviewer's command: `cufep rbfe analyze benchmarks/results/tyk2_rbfe --write` (on a copy) is refused
    and leaves every file as it was; without --write the directory is read and nothing is written."""
    import shutil

    from click.testing import CliRunner

    from cufep import cli
    from cufep.benchmarks.runner import data_root

    copy = tmp_path / "tyk2_rbfe"
    shutil.copytree(data_root() / "benchmarks" / "results" / "tyk2_rbfe", copy)
    before = {path.name: path.read_bytes() for path in copy.iterdir()}
    result = CliRunner().invoke(cli.main, ["rbfe", "analyze", str(copy), "--write"])
    assert result.exit_code != 0
    assert "holds the ddG of 22 edge(s) made from files that are not in the directory" in result.output
    assert {path.name: path.read_bytes() for path in copy.iterdir() if path.name != ".cufep.lock"} == before
    result = CliRunner().invoke(cli.main, ["rbfe", "analyze", str(copy)])
    assert result.exit_code == 0
    assert {path.name: path.read_bytes() for path in copy.iterdir() if path.name != ".cufep.lock"} == before


def test_results_are_not_made_again_without_every_file_their_ddg_was_made_from(tmp_path):
    """Third review of PR #21 (P2): with the solvent leg of a repeat set aside after its ddG was written, the complex
    leg that was left passed the check of `cufep rbfe analyze --write`: an edge of one repeat lost its ddG, and an
    edge of two repeats got the ddG and the standard error of the repeat that was left. Every repeat the ddG of the
    summary was made from must have the result of both legs, and the edge the record of its preparation (the
    edges of a network are compared by it); else nothing is written. With every file there the results are made
    again."""
    import yaml

    from cufep.rbfe.run import rewrite_results
    from cufep.workflow import WorkflowError

    run = tmp_path / "run"
    _run_directory(run, {"a~b": (-12.0, -10.5), "b~c": (-8.0, -9.0)})
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({"sampling": {"repeats": 2}}))
    _write_legs(run, "a~b", 1, _with_moves(-12.4, _moves()), _with_moves(-10.5, NO_MOVES))    # b~c: one repeat
    made = {e["edge"]: e for e in rewrite_results(run)["edges"]}
    assert made["a~b"]["ddg"] == pytest.approx(-1.7) and made["b~c"]["ddg"] == pytest.approx(1.0)
    names = ("summary.json", "edges.csv", "ligands.csv")
    before = {name: (run / name).read_bytes() for name in names}

    def refused_without(path, said: str) -> None:
        aside = path.with_name(path.name + ".aside")
        path.rename(aside)
        with pytest.raises(WorkflowError, match="holds the ddG of 1 edge\\(s\\) made from files that are not in the "
                                               f"directory \\({said}\\).*Nothing was written"):
            rewrite_results(run)
        assert {name: (run / name).read_bytes() for name in names} == before
        aside.rename(path)

    # one leg of one of the two repeats of a~b; one leg of the only repeat of b~c; the record of the prepared edge
    refused_without(run / "edges" / "a~b" / "repeat_1" / "solvent" / "result.json",
                    "a~b: repeat_1/solvent/result.json")
    refused_without(run / "edges" / "b~c" / "repeat_0" / "complex" / "result.json",
                    "b~c: repeat_0/complex/result.json")
    refused_without(run / "edges" / "b~c" / "prep" / "prep.json", "b~c: prep/prep.json")
    rewrite_results(run)
    assert {name: (run / name).read_bytes() for name in names} == before


def test_edges_of_another_rule_of_the_mover_are_named_in_the_results(tmp_path):
    """Third review of PR #21 (P2): the rule of the mover for the velocities of an inserted water was recorded with
    the samples and did not reach the result of a leg. a~b finished by the mover before 2026-11-28 and b~c added by
    this version, with the same values of the water, gave one set of dG, and nothing in the results told which edge
    was of which mover. The rule is in what the repeat and the edge record of the moves; the edges without a
    recorded rule are named in a note of the network, also where no network is made, and a network none of whose
    edges records a rule says that; a network of this version's rule has no note (the tests of one network above).
    Fourth review (P2): an edge without a recorded rule was said to be of another rule than the others; two edges
    of this version, the samples of one set aside, were so reported. Rules that are known to differ are reported
    as that; an edge without a rule is one of which it cannot be told."""
    from cufep.rbfe.run import combine_network, result_lines

    _write_legs(tmp_path, "a~b", 0, _with_moves(-12.0, _moves(rule=None)), _with_moves(-10.5, NO_MOVES))
    _write_legs(tmp_path, "b~c", 0, _with_moves(-8.0, _moves()), _with_moves(-9.0, NO_MOVES))
    _write_legs(tmp_path, "a~c", 0, _with_moves(-7.0, NO_MOVES), _with_moves(-6.0, NO_MOVES))
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {"a": -9.0})
    edges = {e["edge"]: e for e in summary["edges"]}
    assert summary["dg"] is not None                                # both edges give the dG, as before
    assert edges["a~b"]["repeats"][0]["water_moves"][RULE] is None
    assert edges["a~b"]["conditions"]["water_moves"] == _moves(rule=None)
    assert edges["a~b"]["water_rule_not_recorded"] == [0] and edges["b~c"]["water_rule_not_recorded"] == []
    assert edges["b~c"]["conditions"]["water_moves"][RULE] == "turned with it"
    note = ("grand canonical moves of water in the complex leg: no rule for the velocities of an inserted water is "
            "recorded for a~b (until 2026-11-28 the mover left those velocities as they were); whether every edge "
            "with a ddG was sampled under one rule cannot be told (turned with it for b~c)")
    assert summary["notes"] == [note] and f"note: {note}" in result_lines(summary)
    assert not any("were not sampled under one rule" in item for item in summary["notes"])
    # with an edge without the moves there is no network; the note stays
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], 1, {"a": -9.0})
    assert summary["dg"] is None and summary["notes"][0].startswith("NO NETWORK") and summary["notes"][1] == note
    assert "water_rule_not_recorded" in summary["edges"][2] and summary["edges"][2]["water_rule_not_recorded"] == []
    # every edge of the earlier mover: each says so, and so does the network
    earlier = tmp_path / "earlier"
    _write_legs(earlier, "a~b", 0, _with_moves(-12.0, _moves(rule=None)), _with_moves(-10.5, NO_MOVES))
    _write_legs(earlier, "b~c", 0, _with_moves(-8.0, _moves(rule=None)), _with_moves(-9.0, NO_MOVES))
    summary = combine_network(earlier, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {"a": -9.0})
    assert [e["water_rule_not_recorded"] for e in summary["edges"]] == [[0], [0]]
    assert summary["notes"] == [
        "grand canonical moves of water in the complex leg: no rule for the velocities of an inserted water is "
        "recorded for any of the 2 edge(s) with a ddG whose complex legs made the moves (until 2026-11-28 the mover "
        "left those velocities as they were)"]
    # the repeats of one edge are not of one rule: named with their repeats
    repeats = tmp_path / "repeats"
    _write_legs(repeats, "a~b", 0, _with_moves(-12.0, _moves(rule=None)), _with_moves(-10.5, NO_MOVES))
    _write_legs(repeats, "a~b", 1, _with_moves(-12.4, _moves()), _with_moves(-10.5, NO_MOVES))
    summary = combine_network(repeats, ["a", "b"], [("a", "b")], 2, {})
    edge = summary["edges"][0]
    assert edge["ddg"] == pytest.approx(-1.7) and edge["conditions"]["water_moves"][RULE] is None
    assert edge["water_rule_not_recorded"] == [0]
    assert any("is recorded for a~b (repeat 0) (until" in item
               and item.endswith("cannot be told (turned with it for a~b (repeat 1))") for item in summary["notes"])
    # rules that are known to differ (another rule than this version's): reported as that, with the edges of each,
    # and the edge without a rule apart
    other = tmp_path / "other"
    _write_legs(other, "a~b", 0, _with_moves(-12.0, _moves(rule="left as they were")), _with_moves(-10.5, NO_MOVES))
    _write_legs(other, "b~c", 0, _with_moves(-8.0, _moves()), _with_moves(-9.0, NO_MOVES))
    _write_legs(other, "c~d", 0, _with_moves(-8.0, _moves(rule=None)), _with_moves(-9.0, NO_MOVES))
    summary = combine_network(other, ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")], 1, {"a": -9.0})
    assert summary["notes"] == [
        "grand canonical moves of water in the complex leg: the edges with a ddG were not sampled under one rule for "
        "the velocities of an inserted water: left as they were for a~b; turned with it for b~c; no rule recorded "
        "(until 2026-11-28 the mover left those velocities as they were) for c~d"]


def test_the_record_of_a_prepared_edge_is_needed_whatever_the_summary_holds_of_it(tmp_path):
    """The independent review of that fix: an edge of a summary of an earlier version (no LOMAP score) was made again
    without the record of its preparation, and was then left out of the comparison of the prepared systems of the
    network without a word. The record is one of the files a stored ddG needs."""
    from cufep.rbfe.run import files_gone

    _run_directory(tmp_path, {"a~b": (-12.0, -10.5)})
    earlier = {"edge": "a~b", "ddg": -1.5, "repeats": [{"repeat": 0, "ddg": -1.5}]}
    assert files_gone(tmp_path, earlier) == []
    (tmp_path / "edges" / "a~b" / "prep" / "prep.json").unlink()
    assert files_gone(tmp_path, earlier) == ["prep/prep.json"]


def test_a_result_whose_samples_were_set_aside_is_combined_from_the_record_that_is_left(tmp_path):
    """Fourth review of PR #21 (P2): both legs of an edge have their result.json; the HDF5 parts of the samples of
    its complex leg were moved away and hrex/manifest.json was left. To complete the rule of the mover of a result
    that records the values of the water and not the rule (f24342b), the samples were opened as a reader of
    samples opens them, which refuses a manifest whose parts are not there: `cufep rbfe analyze` stopped for the
    whole directory, which d605f12 combined from its results. What is said of a result is read from the record of
    its samples alone: the rule the manifest records, or none; the ddG and the dG are given."""
    import json

    from cufep.rbfe.run import combine_network, water_moves_of_leg
    from cufep.sampling.storage import MANIFEST, StorageError, stored_conditions

    values = {key: value for key, value in _moves().items() if key != RULE}        # as f24342b recorded them
    record = {"enabled": True, "interval_ps": 4.0, "excess_chemical_potential_kcal_per_mol": -6.09,
              "standard_volume_nm3": 0.030345}
    for edge, complex_dg, solvent_dg in (("a~b", -12.0, -10.5), ("b~c", -8.0, -9.0)):
        _write_legs(tmp_path, edge, 0, _with_moves(complex_dg, values), _with_moves(solvent_dg, NO_MOVES))

    def leg(edge: str):
        return tmp_path / "edges" / edge / "repeat_0" / "complex"

    def manifest_without_its_parts(edge: str, gcmc: dict) -> None:
        (leg(edge) / "hrex").mkdir(exist_ok=True)
        (leg(edge) / "hrex" / MANIFEST).write_text(json.dumps({
            "parts": [{"file": "segment_000001.h5", "rows": 10, "tail": 0, "frames": 0}],
            "attrs": {"conditions": {"settings": {"gcmc": gcmc}}}}))

    def moves(edge: str) -> dict:
        return water_moves_of_leg(json.loads((leg(edge) / "result.json").read_text()), leg(edge))

    manifest_without_its_parts("a~b", {**record, RULE: "turned with it"})
    manifest_without_its_parts("b~c", record)
    with pytest.raises(StorageError, match="names parts of the samples that are not there"):
        stored_conditions(leg("a~b") / "hrex")                       # (a reader of the samples refuses them)
    assert moves("a~b") == _moves() and moves("b~c") == _moves(rule=None)
    summary = combine_network(tmp_path, ["a", "b", "c"], [("a", "b"), ("b", "c")], 1, {"a": -9.0})
    assert [e["ddg"] for e in summary["edges"]] == pytest.approx([-1.5, 1.0]) and summary["dg"] is not None
    assert [e["water_rule_not_recorded"] for e in summary["edges"]] == [[], [0]]
    # a record that cannot be read: no rule, and the result is combined all the same
    (leg("b~c") / "hrex" / MANIFEST).write_text("{")
    assert moves("b~c") == _moves(rule=None)
    # a result of the versions before a leg recorded its moves (their counts alone): the values and the rule of the
    # record that is left
    counts_only = _with_moves(-8.0, None) | {"gcmc": {"insertions_accepted": [1]}}
    assert water_moves_of_leg(counts_only, leg("a~b")) == _moves()
    assert water_moves_of_leg(counts_only, leg("b~c")) == {"enabled": True}


def test_results_are_not_made_again_without_the_record_of_samples_that_their_summary_cites(tmp_path):
    """The independent review of the fixes of the fourth review: the complex leg of a~b has a result of a version
    before a leg recorded its moves (their counts alone), and the summary has the values of its water from the
    record of its samples; they are not those of b~c, so the summary gives no network. With the directory of
    the samples moved away the values are not recorded any more, are not compared, and `cufep rbfe analyze
    --write` put one network and its dG in the place of that summary. What the summary holds of the moves of a
    repeat must still be read, or nothing is written."""
    import json
    import shutil

    from cufep.rbfe.run import rewrite_results
    from cufep.sampling.storage import MANIFEST
    from cufep.workflow import WorkflowError

    run = tmp_path / "run"
    _run_directory(run, {"a~b": (-12.0, -10.5), "b~c": (-8.0, -9.0)})
    leg = run / "edges" / "a~b" / "repeat_0" / "complex"
    (leg / "result.json").write_text(json.dumps(_with_moves(-12.0, None) | {"gcmc": {"insertions_accepted": [1]}}))
    (leg / "hrex").mkdir()
    (leg / "hrex" / MANIFEST).write_text(json.dumps({"parts": [], "attrs": {"conditions": {"settings": {"gcmc": {
        "enabled": True, "excess_chemical_potential_kcal_per_mol": -5.0, "standard_volume_nm3": 0.030345}}}}}))
    summary = rewrite_results(run)
    assert summary["dg"] is None and summary["notes"][0].startswith("NO NETWORK")
    assert summary["edges"][0]["conditions"]["water_moves"]["excess_chemical_potential_kcal_per_mol"] == -5.0
    names = ("summary.json", "edges.csv", "ligands.csv")
    before = {name: (run / name).read_bytes() for name in names}
    shutil.move(leg / "hrex", tmp_path / "aside")
    with pytest.raises(WorkflowError, match="holds the ddG of 1 edge\\(s\\) made from files that are not in the "
                                           "directory \\(a~b: repeat_0/complex/hrex \\(the record of its samples"):
        rewrite_results(run)
    assert {name: (run / name).read_bytes() for name in names} == before
    # a summary that holds the moves of the edge alone, not of its repeats
    of_the_edge = json.loads(before["summary.json"])
    del of_the_edge["edges"][0]["repeats"][0]["water_moves"]
    (run / "summary.json").write_text(json.dumps(of_the_edge))
    with pytest.raises(WorkflowError, match="a~b: repeat_\\*/complex/hrex \\(the record of its samples"):
        rewrite_results(run)
    (run / "summary.json").write_bytes(before["summary.json"])
    shutil.move(tmp_path / "aside", leg / "hrex")
    rewrite_results(run)
    assert {name: (run / name).read_bytes() for name in names} == before
