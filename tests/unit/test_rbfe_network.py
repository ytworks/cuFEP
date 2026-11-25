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
