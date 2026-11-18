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
