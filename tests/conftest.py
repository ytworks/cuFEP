"""Shared fixtures. Unit tests run on the CPU with stock OpenMM (no GPU, no plugin)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

#: the packages of the preparation of proteins and ligands; a test that uses them is marked
#: ``@pytest.mark.needs(*PREPARATION)`` (or with the ones it uses)
PREPARATION = ("openff.toolkit", "rdkit", "pdbfixer")


def _missing(packages) -> list[str]:
    import importlib.util

    out = []
    for name in packages:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def pytest_configure(config):
    config.addinivalue_line("markers", "needs(*packages): the test needs these packages")


def pytest_collection_modifyitems(config, items):
    """A test whose packages are not installed is skipped and says which ones; under CI (the variable
    CI is set) a missing package stops the run instead, so that no test is skipped there unseen
    (review of PR 1: without OpenFF, PDBFixer and RDKit 21 tests failed or errored locally)."""
    import os

    for item in items:
        for mark in item.iter_markers("needs"):
            missing = _missing(mark.args)
            if not missing:
                continue
            if os.environ.get("CI"):
                raise pytest.UsageError(f"{item.nodeid} needs {', '.join(missing)}, which this CI does not have")
            item.add_marker(pytest.mark.skip(reason=f"needs {', '.join(missing)}"))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


def lj_particles(n: int = 8, edge: float = 3.0):
    """``n`` Lennard-Jones particles in a periodic box: (System, positions, box), nm. Small enough
    for MD on the Reference platform in a unit test."""
    import openmm as mm

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(edge, 0, 0), mm.Vec3(0, edge, 0), mm.Vec3(0, 0, edge))
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.0)
    for _ in range(n):
        system.addParticle(40.0)
        nonbonded.addParticle(0.0, 0.3, 0.5)
    system.addForce(nonbonded)
    positions = np.array([[0.7 * (i % 2), 0.7 * ((i // 2) % 2), 0.7 * (i // 4)] for i in range(n)]) + 0.5
    return system, positions, edge * np.eye(3)


@pytest.fixture
def other_backend(monkeypatch):
    """A second backend for the plain MD of a leg, in the place of the cuPotential plugin: stock
    OpenMM underneath, so that it runs without a GPU. The fixture gives its class: set ``refuse``
    (it cannot run the System) or ``fail`` (it stops when it makes a Context), read ``contexts``
    (the Contexts it made), ``precisions`` (the precision of each instance) and ``asked`` (every
    name the plain MD asked a backend for)."""
    import openmm as mm

    from cufep.backends.base import BackendError
    from cufep.backends.openmm_stock import StockBackend
    from cufep.sampling import equilibrate

    class OtherBackend(StockBackend):
        name = "cupotential"
        refuse = False
        fail = False
        contexts = 0
        precisions: list = []
        asked: list = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            type(self).precisions.append(self.precision)

        def check_system(self, system):
            if type(self).refuse:
                raise BackendError("this System is not for the plugin")

        def make_context(self, system, integrator, device_index=None):
            if type(self).fail:
                raise mm.OpenMMException("the plugin stopped")
            type(self).contexts += 1
            return super().make_context(system, integrator, device_index)

    made = equilibrate.get_backend

    def get(name, **kwargs):
        OtherBackend.asked.append(name)
        if name == "cupotential":
            return OtherBackend(**kwargs)
        return made(name, **kwargs)

    monkeypatch.setattr(equilibrate, "get_backend", get)
    return OtherBackend


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
    # the same configuration in every run and on every machine. OpenMM puts a new hydrogen at a random place
    # next to its atom (Python's random) and minimises: with a fixed seed and the Reference platform that is
    # one geometry, and the MD below runs on one thread with deterministic forces. Without these every build
    # was another configuration, with another number of waters, and the GPU test that holds the two backends
    # to 0.1 kT on it met 0.007 to 0.054 kT in six builds and 0.141 in a seventh
    import random

    state = random.getstate()
    random.seed(20261121)
    try:
        mod.addHydrogens(ff, platform=mm.Platform.getPlatformByName("Reference"))
        mod.addSolvent(ff, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * u.nanometer)
    finally:
        random.setstate(state)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(dispersion_correction)
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    integ.setRandomNumberSeed(1)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"),
                     {"Threads": "1", "DeterministicForces": "true"})
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


def as_earlier_version(hrex_dir, counts: bool = True) -> Path:
    """Rewrite the samples of a leg as the versions before the manifest wrote them: one samples.h5 that
    holds everything, with extendable datasets (frames in compressed chunks of 8), its attributes as HDF5
    attributes (dicts and lists as JSON text), and the counts dataset of the previous version (``counts``)
    or none (the versions before it). The manifest and the segments are removed. Returns samples.h5."""
    import json

    import h5py
    import numpy as np

    from cufep.sampling.storage import read_samples

    d = Path(hrex_dir)
    data = read_samples(d)
    attrs = data.pop("attrs")
    for path in [*d.glob("segment_*"), d / "manifest.json", d / "samples.h5"]:
        if path.exists():
            path.unlink()
    n, k, s = data["u"].shape
    with h5py.File(d / "samples.h5", "w") as h5:
        h5.create_dataset("u", data=data["u"], maxshape=(None, k, s), chunks=(16, k, s))
        h5.create_dataset("states", data=np.asarray(data["states"], dtype="i4"), maxshape=(None, k), chunks=(256, k))
        h5.create_dataset("volume", data=data["volume"], maxshape=(None, k), chunks=(256, k))
        for name, chunk in (("lrc_iter", 1024), ("lrc_du", 1024), ("frame_iter", 256)):
            h5.create_dataset(name, data=data[name], maxshape=(None,), chunks=(chunk,))
        if "frames" in data:
            a = data["frames"].shape[1]
            h5.create_dataset("frames", data=data["frames"], maxshape=(None, a, 3), chunks=(8, a, 3),
                              compression="gzip", compression_opts=1)
            h5.create_dataset("frame_box", data=data["frame_box"], maxshape=(None, 3, 3))
        for key, value in attrs.items():
            if value is not None:
                h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        if counts:
            h5.create_dataset("committed_counts", data=np.array([n, len(data["lrc_iter"]), len(data["frame_iter"])],
                                                                dtype="i8"))
    return d / "samples.h5"


def edit_attributes(hrex_dir, set: dict | None = None, remove=()) -> None:
    """Change the recorded attributes of the samples of a leg of this version (in its manifest)."""
    from cufep.sampling.storage import _write_manifest, read_manifest

    manifest = read_manifest(hrex_dir)
    assert manifest is not None, f"{hrex_dir} has no manifest"
    attrs = dict(manifest.get("attrs") or {})
    attrs.update(set or {})
    for key in remove:
        attrs.pop(key, None)
    manifest["attrs"] = attrs
    _write_manifest(Path(hrex_dir), manifest)
