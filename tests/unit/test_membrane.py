"""Ions of a membrane system."""
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import pytest

from cufep.prep.membrane import add_ions


def _counts(modeller, forcefield):
    names = [r.name for r in modeller.topology.residues()]
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles()))
    return {"water": names.count("HOH"), "NA": names.count("NA"), "CL": names.count("CL"), "charge": round(charge, 6)}


@pytest.fixture()
def peptides_in_water(dipeptides_in_water):
    _system, topology, positions, _box, _second, _first = dipeptides_in_water
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    def build(extra_sodium: bool = False):
        modeller = app.Modeller(topology, positions * u.nanometer)
        if extra_sodium:                                         # a system with a net charge of +1
            ion = app.Topology()
            ion.addAtom("NA", app.Element.getBySymbol("Na"), ion.addResidue("NA", ion.addChain()))
            box = modeller.topology.getPeriodicBoxVectors()
            modeller.add(ion, [mm.Vec3(0.1, 0.1, 0.1)] * u.nanometer)
            modeller.topology.setPeriodicBoxVectors(box)
        return modeller

    return build, forcefield


def test_a_neutral_system_without_salt_keeps_all_its_water(peptides_in_water):
    """Review of PR 1: with 0 M and no net charge the function reported no ion and still made a water a Cl-."""
    build, forcefield = peptides_in_water
    modeller = build()
    before = _counts(modeller, forcefield)
    assert before["charge"] == 0.0 and before["NA"] == before["CL"] == 0 and before["water"] > 100
    info = add_ions(modeller, forcefield, 0.0, 0.0, log=lambda m: None)
    assert (info["n_sodium"], info["n_chloride"], info["net_charge_before_ions"]) == (0, 0, 0)
    assert _counts(modeller, forcefield) == before


def test_reported_ions_are_the_ions_in_the_topology(peptides_in_water):
    build, forcefield = peptides_in_water
    # salt in a neutral system: as many of each as reported, one water gone for each ion
    modeller = build()
    before = _counts(modeller, forcefield)
    info = add_ions(modeller, forcefield, 0.5, 0.0, log=lambda m: None)
    after = _counts(modeller, forcefield)
    assert info["n_sodium"] == info["n_chloride"] >= 1
    assert (after["NA"], after["CL"]) == (info["n_sodium"], info["n_chloride"])
    assert after["water"] == before["water"] - info["n_sodium"] - info["n_chloride"] and after["charge"] == 0.0
    # a net charge without salt: only the counter-ion
    modeller = build(extra_sodium=True)
    before = _counts(modeller, forcefield)
    assert before["charge"] == 1.0
    info = add_ions(modeller, forcefield, 0.0, 0.0, log=lambda m: None)
    after = _counts(modeller, forcefield)
    assert (info["n_sodium"], info["n_chloride"], info["net_charge_before_ions"]) == (0, 1, 1)
    assert (after["NA"], after["CL"], after["water"], after["charge"]) == (1, 1, before["water"] - 1, 0.0)
