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


def _leg_result(dg_kcal: float, se_kcal: float | None = 0.3, temperature: float = 298.15, pressure: float = 1.0):
    return {"delta_g": dg_kcal * 4.184, "delta_g_error": None if se_kcal is None else se_kcal * 4.184,
            "temperature_K": temperature, "pressure_bar": pressure, "lrc_method": "none",
            "prepared_systems_sha256": "abc", "n_states": 9, "backend": "openmm", "notes": [],
            "quality": {"overlap": {"lowest": 0.1, "ok": True}, "samples": {"counted": 100.0}}}


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
