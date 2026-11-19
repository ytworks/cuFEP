"""Shared fixtures. Unit tests run on the CPU with stock OpenMM (no GPU, no plugin)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def water_ligand_system():
    """A small TIP3P box in which one water molecule plays the role of the ligand.

    Returns (system, positions_nm, box_nm, ligand_atom_indices). Dispersion correction is
    off so that energies of the alchemical and the original System can be compared exactly.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(2.0, 2.0, 2.0) * u.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(False)
    # relax a little so that energies are ordinary numbers
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    integ.setRandomNumberSeed(1)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(mod.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 1)
    integ.step(200)
    st = ctx.getState(getPositions=True)
    pos = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    box = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
    del ctx
    ligand = [a.index for a in list(mod.topology.residues())[10].atoms()]
    return system, pos, box, ligand


_DIPEPTIDE_HEAVY = [("ACE", "CH3", "C", (0.00, 0.00, 0.0)), ("ACE", "C", "C", (1.25, 0.85, 0.0)),
                    ("ACE", "O", "O", (1.25, 2.08, 0.0)), ("ALA", "N", "N", (2.45, 0.20, 0.0)),
                    ("ALA", "CA", "C", (3.70, 0.95, 0.0)), ("ALA", "CB", "C", (3.70, 1.75, 1.30)),
                    ("ALA", "C", "C", (4.95, 0.10, 0.0)), ("ALA", "O", "O", (4.95, -1.13, 0.0)),
                    ("NME", "N", "N", (6.15, 0.75, 0.0)), ("NME", "C", "C", (7.40, 0.00, 0.0))]
_DIPEPTIDE_BONDS = {"ACE": [("CH3", "C"), ("C", "O")], "ALA": [("N", "CA"), ("CA", "CB"), ("CA", "C"), ("C", "O")],
                    "NME": [("N", "C")]}


@pytest.fixture(scope="session")
def dipeptides_in_water():
    return build_dipeptides_in_water()


def build_dipeptides_in_water(box_nm: float = 2.4, dispersion_correction: bool = False):
    """Two alanine dipeptides in a small TIP3P box: molecules with torsions and 1-4 pairs.

    Returns (system, topology, positions_nm, box_nm, atoms of the second peptide, atoms of the
    first). The second peptide plays the ligand, the first one the environment that can be hot.
    Dispersion correction is off by default, as in ``water_ligand_system``.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    top, pos = app.Topology(), []
    for shift in ((0.4, 0.6, 0.6), (0.4, 1.5, 1.4)):
        chain, previous_c = top.addChain(), None
        for res_name in ("ACE", "ALA", "NME"):
            res, local = top.addResidue(res_name, chain), {}
            for rn, name, element, xyz in _DIPEPTIDE_HEAVY:
                if rn == res_name:
                    local[name] = top.addAtom(name, app.Element.getBySymbol(element), res)
                    pos.append(mm.Vec3(*(0.1 * np.array(xyz) + np.array(shift))))
            for a, b in _DIPEPTIDE_BONDS[res_name]:
                top.addBond(local[a], local[b])
            if previous_c is not None:
                top.addBond(previous_c, local["N"])
            previous_c = local["C"] if res_name != "NME" else None
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    mod = app.Modeller(top, pos * u.nanometer)
    mod.addHydrogens(ff)
    mod.addSolvent(ff, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * u.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(dispersion_correction)
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    integ.setRandomNumberSeed(1)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(mod.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 300)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 1)
    integ.step(300)
    st = ctx.getState(getPositions=True)
    positions = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    box = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
    del ctx
    chains = list(mod.topology.chains())
    return (system, mod.topology, positions, box, [a.index for a in chains[1].atoms()],
            [a.index for a in chains[0].atoms()])


def reference_context(system, pos, box, platform="Reference"):
    import openmm as mm
    import openmm.unit as u

    integ = mm.VerletIntegrator(0.001 * u.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(pos)
    return ctx


def potential(ctx, groups=None) -> float:
    import openmm.unit as u

    kwargs = {} if groups is None else {"groups": groups}
    return ctx.getState(getEnergy=True, **kwargs).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
