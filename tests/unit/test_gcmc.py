"""Grand canonical Monte Carlo of water (``sampling.gcmc``): a water that is switched off against a
System without it, the stand-in energy with ghosts against OpenMM, and the number of waters the
moves give against a known answer."""
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import pytest
from test_relocation_mc import ideal_waters, nonbonded, reaction_field_energy

from cufep.alchemy.factory import make_alchemical_system
from cufep.sampling.gcmc import (
    COUNTS,
    GcmcError,
    GcmcMover,
    GcmcSpec,
    WaterSwitch,
    farthest_waters,
    gcmc_spec,
    ghosts_off,
)
from cufep.sampling.relocation_mc import SCORE_RADIUS_NM, StandIn
from cufep.sampling.states import AlchemicalState, StateEnergyEvaluator


def water_box(water: str, edge: float = 2.0):
    """A box of water of the model of ``water`` with every virtual site in its place: (forcefield, topology,
    positions in nm, box in nm)."""
    forcefield = app.ForceField(water)
    model = app.Modeller(app.Topology(), [])
    model.addSolvent(forcefield, model="tip4pew" if "opc" in water else "tip3p",
                     boxSize=mm.Vec3(edge, edge, edge) * u.nanometer)
    system = forcefield.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(model.positions)
    context.applyConstraints(1e-10)
    context.computeVirtualSites()
    positions = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer))
    return forcefield, model.topology, positions, np.eye(3) * edge


def spec_of(system, topology, positions, box, ligand, ghosts: int, **more) -> GcmcSpec:
    values = dict(cutoff_nm=0.9, softcore_alpha=0.5, softcore_b=1, temperature_K=298.15, interval_ps=4.0, attempts=20,
                  batch_size=10, radius_margin_nm=0.3, excess_chemical_potential_kcal_per_mol=-6.09,
                  standard_volume_nm3=0.030345, ghost_waters=ghosts)
    return gcmc_spec(system, topology, positions, box, ligand, **{**values, **more})


def energies(context, positions, states) -> dict:
    """What is compared of a Context: the potential energy by force group, the forces on the atoms and the
    reduced potentials of the windows."""
    context.setPositions(positions)
    context.getState(getEnergy=True)
    system = context.getSystem()
    real = [i for i in range(system.getNumParticles()) if not system.isVirtualSite(i)]
    groups = [context.getState(getEnergy=True, groups=1 << g).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
              for g in range(4)]
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
    return {"groups": np.array(groups), "forces": np.asarray(forces)[real],
            "u": StateEnergyEvaluator(context, states, 298.15, None).reduced_potentials()[0]}


@pytest.mark.parametrize("water", ["amber14/tip3p.xml", "amber14/opc.xml"])
def test_a_water_that_is_switched_off_in_a_context_is_a_water_the_system_was_built_without(water):
    """The moves switch waters in a Context that runs. A Context whose waters were switched off gives, in
    every window, the energies and forces of the alchemical System of a system built with those waters off
    (:func:`ghosts_off`: what a leg with the moves is made of), and the other way round a Context of that
    System with the ghosts switched on gives those of the system with every water. With TIP3P and with OPC,
    whose fourth site, a virtual one, carries the charge that is switched."""
    forcefield, topology, positions, box = water_box(water)
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    residues = list(topology.residues())
    ligand = [a.index for a in residues[10].atoms()]                # a water stands for the ligand
    spec = spec_of(system, topology, positions, box, ligand, ghosts=5)
    assert len(spec.ghosts) == 5 and spec.waters.shape[1] == (4 if "opc" in water else 3)
    off = spec.waters[spec.ghosts].reshape(-1)
    with_ghosts = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    ghosts_off(with_ghosts, off)
    states = [AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=e, lambda_sterics=s)
              for e, s in ((1.0, 1.0), (0.3, 1.0), (0.0, 0.4))]
    platform = mm.Platform.getPlatformByName("CPU")
    properties = {"Threads": "1", "DeterministicForces": "true"}

    def context_of(plain):
        alch = make_alchemical_system(plain, ligand)
        made = mm.Context(alch.system, mm.VerletIntegrator(0.001), platform, properties)
        made.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
        return made

    all_on, built_off = context_of(system), context_of(with_ghosts)
    expected = {"on": energies(all_on, positions, states), "off": energies(built_off, positions, states)}
    assert abs(expected["on"]["groups"][0] - expected["off"]["groups"][0]) > 1.0       # kJ/mol: five waters
    switched = WaterSwitch(all_on, spec, on=np.ones(len(spec.waters), dtype=bool))
    assert switched.apply(spec.initial_on()) == 5
    back = WaterSwitch(built_off, spec)
    assert back.apply(np.ones(len(spec.waters), dtype=bool)) == 5 and back.apply(back.on) == 0
    for context, key in ((all_on, "off"), (built_off, "on")):
        got = energies(context, positions, states)
        assert np.allclose(got["groups"], expected[key]["groups"], rtol=0.0, atol=2e-3)            # kJ/mol
        assert np.abs(got["forces"] - expected[key]["forces"]).max() < 1e-2                         # kJ/mol/nm
        assert np.allclose(got["u"], expected[key]["u"], rtol=0.0, atol=1e-3)                       # kT


def test_the_stand_in_energy_leaves_out_the_waters_that_are_switched_off():
    """The trials are evaluated with the stand-in energy of the water relocation moves. A water that is off
    is not there for it: the energy of a water with everything else is what OpenMM gives with the ghosts
    made silent, for a pose at the cutoff and for the waters where they are (what a deletion needs)."""
    forcefield, topology, positions, box = water_box("amber14/opc.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True)
    ligand = [a.index for a in list(topology.residues())[10].atoms()]
    spec = spec_of(system, topology, positions, box, ligand, ghosts=40)
    on = spec.initial_on()
    active = np.ones(system.getNumParticles(), dtype=bool)
    active[spec.waters[~on].reshape(-1)] = False
    stand_in = StandIn(spec, positions, box, 1.0, 1.0, active=active)
    ghosts = set(spec.waters[~on].reshape(-1).tolist())
    which = np.flatnonzero(on)[[3, 50, 120]]
    for radius in (spec.cutoff_nm, SCORE_RADIUS_NM):
        got = stand_in.water_energies(which, radius)
        for value, water in zip(got, which):
            atoms = set(spec.waters[water].tolist())
            expected = reaction_field_energy(system, positions, box, radius, silent=ghosts) \
                - reaction_field_energy(system, positions, box, radius, silent=ghosts | atoms)
            assert value == pytest.approx(expected, rel=1e-9, abs=1e-6)
            pose = stand_in.pose_energies(positions[spec.waters[water]][None], water, radius)[0]
            assert pose == pytest.approx(expected, rel=1e-9, abs=1e-6)
    # and with every water there it is another number: the ghosts were among the neighbours
    everything = StandIn(spec, positions, box, 1.0, 1.0).water_energies(np.flatnonzero(on), spec.cutoff_nm)
    assert np.abs(everything - stand_in.water_energies(np.flatnonzero(on), spec.cutoff_nm)).max() > 1.0
    # the trials of a batch: each has the acceptance of the published expressions with that energy,
    #   insertion  ln P = B - ln(N + 1) - beta u,  u the energy of the ghost in its pose,
    #   deletion   ln P = ln N - B - beta u,       u minus the energy of the water where it is
    mover = GcmcMover(spec, None, seed=7)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    batch = mover.trials(mover.configuration(positions, box, state, on), 400)
    insert, inside = batch["insert"], mover.in_sphere(positions, box, on)
    n, beta = len(inside), 1.0 / (8.31446261815324e-3 * spec.temperature_K)
    assert n >= 2 and batch["n"] == n and 150 < insert.sum() < 250 and not on[batch["ghost"]]
    poses = batch["poses"]
    assert len(poses) == insert.sum()
    centre = positions[spec.ligand_heavy].mean(axis=0)
    assert (np.linalg.norm(poses[:, 0] - centre, axis=1) < spec.radius_nm).all()       # the oxygen, in the sphere
    shape = positions[spec.waters[batch["ghost"]]]
    for a in range(1, poses.shape[1]):                               # the ghost as one body
        assert np.allclose(np.linalg.norm(poses[:, a] - poses[:, 0], axis=1), np.linalg.norm(shape[a] - shape[0]))
    u_in = stand_in.pose_energies(poses, batch["ghost"], spec.cutoff_nm)
    assert np.allclose(batch["u"][insert], u_in, rtol=1e-9, atol=1e-9)
    assert np.allclose(batch["log_p"][insert], spec.adams - np.log(n + 1.0) - beta * u_in, rtol=1e-12, atol=1e-9)
    deleted = batch["delete"][~insert]
    assert set(deleted.tolist()) <= set(inside.tolist()) and len(set(deleted.tolist())) > 1
    u_out = -stand_in.water_energies(deleted, spec.cutoff_nm)
    assert np.allclose(batch["u"][~insert], u_out, rtol=1e-9, atol=1e-9)
    assert np.allclose(batch["log_p"][~insert], np.log(n) - spec.adams - beta * u_out, rtol=1e-12, atol=1e-9)
    assert (batch["delete"][insert] == -1).all() and ((batch["uniform"] >= 0) & (batch["uniform"] < 1)).all()


def ideal_spec(system, n: int, sites: int, radius: float, mean: float, **more) -> GcmcSpec:
    """The specification for ``n`` waters that interact with nothing, all of them ghosts at the start, with
    the Adams value whose number of waters in the sphere is ``mean`` for an ideal gas (B = ln mean)."""
    parameters = nonbonded(system)
    count = system.getNumParticles()
    values = dict(waters=np.array([[1 + sites * k + s for s in range(sites)] for k in range(n)]),
                  ligand=np.array([0]), ligand_heavy=np.array([0]), radius_nm=radius, charge=parameters[:, 0],
                  sigma=parameters[:, 1], epsilon=parameters[:, 2],
                  mass=np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(count)]),
                  cutoff_nm=0.9, softcore_alpha=0.5, softcore_b=1, temperature_K=300.0, interval_ps=4.0, attempts=12,
                  batch_size=4, excess_chemical_potential_kj_per_mol=0.0,
                  standard_volume_nm3=4.0 / 3.0 * np.pi * radius ** 3 / mean, ghosts=np.arange(n))
    spec = GcmcSpec(**{**values, **more})
    assert spec.adams == pytest.approx(np.log(mean))
    return spec


class WellMover(GcmcMover):
    """A mover whose simulation has an energy of ``well_kt`` kT less for every water that interacts and has
    its oxygen in the sphere: an energy the stand-in knows nothing of."""

    well_kt = 0.0

    def _potential(self) -> float:
        state = self.context.getState(getPositions=True)
        x = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
        box = np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
        return super()._potential() - self.well_kt / self.beta * len(self.in_sphere(x, box, self.switch.on))


@pytest.mark.parametrize(("sites", "well_kt"), [(3, 0.0), (4, 1.0)])
def test_the_number_of_waters_in_the_sphere_is_that_of_the_grand_canonical_ensemble(sites, well_kt):
    """Waters that interact with nothing, in equilibrium with a reservoir: their number in the sphere has a
    Poisson distribution with the mean exp(B), and with exp(B + 1) where every water in the sphere has an
    energy 1 kT lower. The moves alone, started from an empty sphere, give that mean and that variance. The
    energy of the well reaches the acceptance only through the potential energy of the simulation, in the
    second test of a trial (with the sign of that term turned the mean would be 1.5 in place of 10.9). With
    three sites and, where the well is, with a fourth site that is a virtual one."""
    n, edge, radius, mean = 60, 2.4, 0.6, 4.0
    system, positions, box = ideal_waters(n, edge, 0.0, radius, seed=3, sites=sites)
    spec = ideal_spec(system, n, sites, radius, mean, attempts=6, batch_size=3)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    mover = WellMover(spec, context, seed=5)
    mover.well_kt = well_kt
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    assert not mover.on.any()
    numbers, totals = [], dict.fromkeys(COUNTS, 0)
    for move in range(1500):
        counts = mover.attempt(state)
        assert counts["insertions_attempted"] + counts["deletions_attempted"] == spec.attempts
        for key in COUNTS[:4]:
            totals[key] += counts[key]
        x = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer))
        inside = mover.in_sphere(x, box)
        assert counts["waters_in_the_sphere"] == len(inside) == int(mover.on.sum())
        assert np.array_equal(mover.on, mover.switch.on)
        if move >= 100:
            numbers.append(len(inside))
    numbers = np.array(numbers)
    expected = mean * np.exp(well_kt)
    # tolerances: about four times the scatter over eight chains of this length with other seeds (the
    # means had a standard deviation of 0.08 without the well and 0.38 with it, the variances of 0.27 and
    # 1.8; the largest deviations from the exact values were 0.16 and 0.79, 0.45 and 4.1)
    assert numbers.mean() == pytest.approx(expected, abs=0.33 if well_kt == 0.0 else 1.5)
    assert numbers.var() == pytest.approx(expected, abs=1.1 if well_kt == 0.0 else 7.2)
    assert totals["insertions_accepted"] > 300 and totals["deletions_accepted"] > 300
    assert abs(totals["insertions_accepted"] - totals["deletions_accepted"]) <= n
    # an inserted water is the ghost as one body: every constrained distance and the virtual site
    x = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer))
    for atoms in spec.waters[mover.on]:
        shape = np.linalg.norm(x[atoms][:, None, :] - x[atoms][None, :, :], axis=2)
        first = np.linalg.norm(positions[atoms][:, None, :] - positions[atoms][None, :, :], axis=2)
        assert np.abs(shape - first).max() < 1e-9


def test_the_ghosts_are_the_waters_farthest_from_the_ligand_and_a_move_stops_when_none_is_left():
    """The ghosts of the start are the waters whose oxygen is farthest from the ligand, so that the sphere
    starts with the waters it has. A request that would take waters of the sphere, or more than there are,
    is refused when the leg is built. An insertion with no ghost left stops the calculation with the reason,
    as grand does, in place of going on with fewer waters than the ensemble has."""
    forcefield, topology, positions, box = water_box("amber14/tip3p.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True)
    residues = list(topology.residues())
    ligand = [a.index for a in residues[10].atoms()]
    spec = spec_of(system, topology, positions, box, ligand, ghosts=7)
    centre = positions[ligand[0]]
    delta = positions[spec.waters[:, 0]] - centre
    distance = np.linalg.norm(delta - np.rint(delta / 2.0) * 2.0, axis=1)
    assert set(spec.ghosts.tolist()) == set(np.argsort(distance)[-7:].tolist())
    assert np.array_equal(farthest_waters(spec.waters, positions, box, spec.ligand_heavy, 7), spec.ghosts)
    assert distance[spec.ghosts].min() > spec.radius_nm
    with pytest.raises(GcmcError, match="waters of the sphere"):
        spec_of(system, topology, positions, box, ligand, ghosts=len(spec.waters) - 1)
    with pytest.raises(GcmcError, match="the system has"):
        spec_of(system, topology, positions, box, ligand, ghosts=len(spec.waters) + 1)
    n, radius = 6, 0.6
    ideal, x, cell = ideal_waters(n, 2.4, 0.0, radius, seed=1)
    full = ideal_spec(ideal, n, 3, radius, mean=50.0)            # a sphere that wants more waters than there are
    context = mm.Context(ideal, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(x)
    mover = GcmcMover(full, context, seed=2)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    with pytest.raises(GcmcError, match="no ghost water is left"):
        for _ in range(200):
            mover.attempt(state)
    assert mover.switch.on.all()


def leg_with_moves(system, positions, box, ligand, **more):
    """(specification of the moves, alchemical System of the leg, tail correction) for a box of 3-site waters
    one of which is the ligand: twelve ghosts, a sphere of 0.5 nm, and a chemical potential so high that
    insertions are accepted within a few trials (a test of the bookkeeping, not of water)."""
    from cufep.alchemy.lrc import TailEnergy

    parameters, n = nonbonded(system), system.getNumParticles()
    waters = np.array([[i, i + 1, i + 2] for i in range(0, n, 3) if i not in ligand])
    values = dict(waters=waters, ligand=np.array(ligand), ligand_heavy=np.array(ligand[:1]), radius_nm=0.5,
                  charge=parameters[:, 0], sigma=parameters[:, 1], epsilon=parameters[:, 2],
                  mass=np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(n)]), cutoff_nm=0.9,
                  softcore_alpha=0.5, softcore_b=1, temperature_K=300.0, interval_ps=0.04, attempts=60, batch_size=20,
                  excess_chemical_potential_kj_per_mol=100.0, standard_volume_nm3=0.0303,
                  ghosts=farthest_waters(waters, positions, box, np.array(ligand[:1]), 12))
    spec = GcmcSpec(**{**values, **more})
    plain = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    ghosts_off(plain, spec.waters[spec.ghosts].reshape(-1))
    alch = make_alchemical_system(plain, ligand)
    tail = TailEnergy.from_system(system, ligand, alch.ligand_parameters, 0.9, 2.0)
    return spec, alch, tail


def test_a_leg_with_the_moves_stores_them_and_the_waters_of_every_replica_and_goes_on_after_a_stop(
        tmp_path, water_ligand_system):
    """Replica exchange of a small leg with the moves (stock OpenMM, CPU, no barostat): every replica has its
    own list of the waters that interact, which its worker gives the Context before the MD and takes back
    after the moves. The samples hold the counts of the moves per iteration and lambda window, the checkpoint
    the lists; over all replicas the waters that interact are those of the start plus the insertions minus the
    deletions; a leg that is resumed goes on with the lists it had; the result gives the numbers by window."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.sampling.hrex import ReplicaExchangeSampler, SamplerError
    from cufep.sampling.storage import read_samples
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import gcmc_numbers

    system, pos, box, ligand = water_ligand_system
    moves, alch, tail = leg_with_moves(system, pos, box, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 2})
    k, n_waters = len(schedule.states), len(moves.waters)
    # 10 steps of 2 fs per iteration, one move for every 0.04 ps (20 steps): every second iteration
    spec = LegSpec(name="gcmc", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=None,
                   barostat="none", timestep_fs=2.0, steps_per_iteration=10, seed=5, gcmc=moves, lrc=tail.to_dict())
    assert spec.n_switched_waters == n_waters and not spec.has_barostat

    def sampler():
        return ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=lambda m: None)

    def lists() -> np.ndarray:
        with np.load(tmp_path / "hrex" / "checkpoint.npz") as checkpoint:
            return checkpoint["water_on"]

    with sampler() as s:
        s.initialize(pos, box)
        assert np.array_equal(lists(), np.broadcast_to(moves.initial_on(), (k, n_waters)))
        s.run(4, record_stride=1, checkpoint_stride=2)
    samples = read_samples(tmp_path / "hrex")
    counts = samples["gcmc"]
    assert counts.shape == (4, k, 5) and np.isfinite(samples["u"]).all()
    trials = counts[:, :, 0] + counts[:, :, 2]
    assert trials.sum(axis=1).tolist() == [0, k * moves.attempts, 0, k * moves.attempts]
    assert (counts[::2, :, 4] == -1).all() and (counts[1::2, :, 4] >= 0).all()      # waters in the sphere
    assert (counts[:, :, 1] <= counts[:, :, 0]).all() and (counts[:, :, 3] <= counts[:, :, 2]).all()
    made = int(counts[:, :, 1].sum() - counts[:, :, 3].sum())
    assert counts[:, :, 1].sum() >= 1                             # (insertions were made)
    first = lists()
    assert first.shape == (k, n_waters) and first.dtype == bool
    assert int(first.sum()) == k * (n_waters - len(moves.ghosts)) + made
    # the tail energy of the coupled state was taken of the waters that interact
    assert np.isfinite(samples["lrc_du"]).all() and len(samples["lrc_du"]) == 4
    with sampler() as s:
        assert s.restore()
        assert np.array_equal(s.arrays.won[s.cur] > 0.5, first)
        s.run(2, record_stride=1, checkpoint_stride=2)
    samples = read_samples(tmp_path / "hrex")
    counts = samples["gcmc"]
    assert counts.shape == (6, k, 5)
    assert int(lists().sum()) == k * (n_waters - len(moves.ghosts)) + int(counts[:, :, 1].sum() - counts[:, :, 3].sum())
    numbers = gcmc_numbers(samples, 2)                            # the iterations from the third on
    assert [a + b for a, b in zip(numbers["insertions_attempted"], numbers["deletions_attempted"])] \
        == [2 * moves.attempts] * k
    assert sum(numbers["insertions_accepted"]) == int(counts[2:, :, 1].sum())
    waters = numbers["waters_in_the_sphere"]
    assert all(low <= mean <= high for low, mean, high in zip(waters["lowest"], waters["mean"], waters["highest"]))
    assert gcmc_numbers({"u": samples["u"]}, 0) is None           # samples without the moves have no such numbers
    # a checkpoint without the lists is not one of a leg with the moves
    with np.load(tmp_path / "hrex" / "checkpoint.npz") as checkpoint:
        kept = {key: checkpoint[key] for key in checkpoint.files if key != "water_on"}
    np.savez(tmp_path / "hrex" / "checkpoint.npz", **kept)
    with sampler() as s, pytest.raises(SamplerError, match="which waters of each replica interact"):
        s.restore()


def test_a_start_moved_into_the_fixed_cell_is_minimised_there_on_stock_openmm(water_ligand_system):
    """A2A 4g~4kU (2026-11-28): the end of a pre-equilibration moved into the fixed cell of the moves (z -2.0 %) went
    to the walk on cuPotential as it was, and a contract check of the plugin stopped it in its first 5 ps, with every
    worker of the run under MPS (stock OpenMM ran the same walk; minimised first, the plugin ran it too). The one
    function both protocols use minimises it in the cell on stock OpenMM whatever the backend of the leg (here the
    plugin, which this machine does not have), in the first state of the leg: lower energy and largest force, and every
    water keeps its geometry."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.engine.leg import minimised_in_cell
    from cufep.sampling.states import set_state
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import positions_in_cell

    system, pos, box, ligand = water_ligand_system
    _moves, alch, _tail = leg_with_moves(system, pos, box, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 1})
    spec = LegSpec(name="cell", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="cupotential", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=None,
                   barostat="none", timestep_fs=4.0, steps_per_iteration=10, seed=5)
    probe = mm.Context(alch.system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    cell = np.asarray(box) @ np.diag([1.0, 1.0, 0.95])
    moved = positions_in_cell(pos, box, cell, probe.getMolecules())
    out = minimised_in_cell(spec, alch.system, cell, moved, np.zeros_like(moved), None, 5)

    def energy_and_force(x) -> tuple[float, float]:
        probe.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in cell])
        probe.setPositions(x)
        set_state(probe, schedule.states[0])
        st = probe.getState(getEnergy=True, getForces=True)
        f = st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        return st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole), float(np.linalg.norm(f, axis=1).max())

    (e0, f0), (e1, f1) = energy_and_force(moved), energy_and_force(out)
    assert e1 < e0 and f1 < f0
    waters = np.arange(0, system.getNumParticles(), 3)
    for x in (moved, out):
        assert np.allclose(np.linalg.norm(x[waters + 1] - x[waters], axis=1), 0.09572, atol=1e-4)


def test_a_window_of_the_precalculation_starts_with_the_waters_of_its_configuration_and_returns_them(
        water_ligand_system):
    """A window of the precalculation of adaptive lambda scheduling makes the moves too. A configuration that
    says which of its waters interact is run with those; one that does not (the walk through the windows) has
    the ghosts of the start; the last configuration of the window says which waters it ended with."""
    from cufep.alchemy.schedule import run_window

    system, pos, box, ligand = water_ligand_system
    moves, alch, _tail = leg_with_moves(system, pos, box, ligand)
    integrator = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integrator.setRandomNumberSeed(31)
    context = mm.Context(alch.system, integrator, mm.Platform.getPlatformByName("CPU"),
                         {"Threads": "1", "DeterministicForces": "true"})
    mover = GcmcMover(moves, context, seed=4)
    velocities = np.zeros_like(pos)
    task = {"state": {"lambda_restraints": 1.0, "lambda_electrostatics": 0.5, "lambda_sterics": 1.0},
            "config": (pos, box, velocities), "stages": ["discharge"], "candidates": 5, "temperature_K": 300.0,
            "equilibration_steps": 20, "n_samples": 2, "steps_per_sample": 10,
            "leg_stages": ["discharge", "vanish"], "rest2_scale": 1.0}
    first = run_window(context, integrator, task, mover, 20)
    assert len(first["config"]) == 4 and np.isfinite(first["energies"]["discharge"]).all()
    assert first["config"][3].dtype == bool and np.array_equal(first["config"][3], mover.on)
    # a configuration with a list of its own: run with those waters, whatever the Context had
    given = moves.initial_on()
    given[moves.ghosts[:2]] = True
    mover.set_on(moves.initial_on())
    short = {**task, "equilibration_steps": 0, "n_samples": 1, "steps_per_sample": 5}
    again = run_window(context, integrator, {**short, "config": (pos, box, velocities, given)}, mover, 1000)
    assert np.array_equal(again["config"][3], given) and np.array_equal(mover.switch.on, given)   # (no move in 5 steps)
    # and a configuration without a list has the ghosts of the start
    plain = run_window(context, integrator, short, mover, 1000)
    assert np.array_equal(plain["config"][3], moves.initial_on())


BASE = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}
MOVES = {"enabled": True, "excess_chemical_potential_kcal_per_mol": -6.0, "standard_volume_nm3": 0.03}


def test_the_moves_are_a_setting_of_the_complex_leg_and_combinations_they_cannot_be_made_with_are_refused():
    """``sampling.gcmc`` is off by default. The conditions of the samples name its settings, with the two
    numbers of the water as they are used and the pressure the fixed cell was equilibrated at, for the
    complex leg of a run that makes the moves and for no other leg: a leg sampled with them is not continued
    without them, and every other leg keeps the record it has. The pre-equilibration of that leg records its
    ghosts. Refused when the configuration is read, before anything is written: the moves together with the
    water relocation moves, with the isotropic tail correction, with the complex leg in water, with a batch
    larger than a move, and ``auto`` for a water model cuFEP has no values of."""
    from cufep.config.schema import Config
    from cufep.prep.water import BULK_WATER, bulk_water_values
    from cufep.workflow import (
        SETTINGS_ADDED,
        equil_made_with,
        gcmc_in,
        gcmc_water_values,
        pressure_of_samples,
        samples_settings,
        settings_differences,
    )

    default = Config.model_validate(BASE)
    assert default.sampling.gcmc.enabled is False
    assert SETTINGS_ADDED["sampling"]["gcmc"] == default.sampling.gcmc.model_dump(mode="json")
    moving = Config.model_validate(dict(BASE, sampling={"gcmc": MOVES}))
    assert gcmc_in(moving, "complex") and not gcmc_in(moving, "solvent") and not gcmc_in(default, "complex")
    assert gcmc_water_values(moving) == (-6.0, 0.03)
    assert "gcmc" not in samples_settings(default, "complex")
    assert samples_settings(moving, "solvent") == samples_settings(default, "solvent")
    record = samples_settings(moving, "complex")
    assert record["gcmc"] == {"enabled": True, "interval_ps": 4.0, "attempts": 2500, "batch_size": 1000,
                              "radius_margin_nm": 0.3, "ghost_waters": "auto",
                              "excess_chemical_potential_kcal_per_mol": -6.0, "standard_volume_nm3": 0.03,
                              "cell_pressure_bar": 1.0}
    assert settings_differences(samples_settings(default, "complex"), record)[0].startswith("gcmc: not recorded -> ")
    # the pressure of such samples is that of the cell they are fixed in; of others that of their barostat
    assert pressure_of_samples({"pressure_bar": None, "settings": record}) == 1.0
    assert pressure_of_samples({"pressure_bar": 2.0, "settings": {}}) == 2.0
    assert pressure_of_samples({"pressure_bar": None}) is None
    assert equil_made_with(moving, "complex", True, 1)["gcmc"] == {
        "ghost_waters": "auto", "radius_margin_nm": 0.3, "standard_volume_nm3": 0.03}
    assert "gcmc" not in equil_made_with(moving, "solvent", False, 1)
    assert "gcmc" not in equil_made_with(default, "complex", True, 1)

    def problem(**sections) -> str:
        with pytest.raises(ValueError) as refused:
            Config.model_validate(dict(BASE, **sections))
        return str(refused.value)

    assert "one kind of water moves" in problem(sampling={"gcmc": MOVES, "relocation_mc": {"enabled": True}})
    assert "analysis.lrc: reweight" in problem(sampling={"gcmc": MOVES}, analysis={"lrc": "isotropic"})
    assert "membrane.complex_leg: water" in problem(
        sampling={"gcmc": MOVES}, membrane={"enabled": True, "complex_leg": "water"})
    assert "batch_size" in problem(sampling={"gcmc": dict(MOVES, attempts=10, batch_size=20)})
    # auto: the values of the water model, where cuFEP has them
    for water in ("amber14/tip3p.xml", "amber14/opc.xml", "amber14/spce.xml"):
        chosen = dict(BASE, forcefield={"water": water}, sampling={"gcmc": {"enabled": True}})
        if bulk_water_values(water) is None:
            assert "sampling.gcmc.excess_chemical_potential_kcal_per_mol is auto" in problem(**{
                key: value for key, value in chosen.items() if key != "system"})
        else:
            assert gcmc_water_values(Config.model_validate(chosen)) == BULK_WATER[water]
    assert bulk_water_values("amber14/spce.xml") is None


def test_the_system_of_a_leg_with_the_moves_has_its_ghosts_off_and_its_alchemical_run_a_fixed_cell():
    """What a leg with the moves is made of (:func:`leg_system`): the prepared System with the nonbonded
    interactions of the ghosts of the start taken out, so that the pre-equilibration settles the cell
    without them, and the specification of the moves from the System with every water. Without the moves,
    and for the solvent leg, the prepared System as it is. The tail correction of the leg is set up with
    every water (the worker leaves out the ghosts of each configuration). The alchemical run starts in the
    mean cell of the pre-equilibration, every molecule shifted whole into it; without the moves in the last
    cell, as it was."""
    from cufep.alchemy.lrc import TailEnergy
    from cufep.config.schema import Config
    from cufep.workflow import WorkflowError, alchemical_box, alchemical_leg, leg_spec, leg_system, positions_in_cell

    forcefield, topology, positions, box = water_box("amber14/opc.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    residues = list(topology.residues())
    ligand = [a.index for a in residues[10].atoms()]
    prep = {"system_xml": mm.XmlSerializer.serialize(system), "topology": topology, "positions": positions,
            "box": box, "ligand": ligand, "protein": [], "solute": ligand}
    moving = Config.model_validate(dict(BASE, forcefield={"water": "amber14/opc.xml"}, sampling={
        "gcmc": dict(MOVES, radius_margin_nm=0.5, ghost_waters=9), "rest2": {"scale": 1.0}}))
    plain, none = leg_system(Config.model_validate(BASE), prep, "complex")
    assert none is None and mm.XmlSerializer.serialize(plain) == prep["system_xml"]
    assert leg_system(moving, prep, "solvent")[1] is None
    leg, moves = leg_system(moving, prep, "complex")
    assert len(moves.ghosts) == 9 and moves.radius_nm == pytest.approx(0.5)
    assert moves.excess_chemical_potential_kj_per_mol == pytest.approx(-6.0 * 4.184)
    off = set(moves.waters[moves.ghosts].reshape(-1).tolist())
    before, after = nonbonded(system), nonbonded(leg)
    for i in range(system.getNumParticles()):
        if i in off:
            assert after[i, 0] == 0.0 and after[i, 2] == 0.0 and after[i, 1] == before[i, 1]
        else:
            assert np.array_equal(after[i], before[i])
    assert np.array_equal(nonbonded(system), np.stack([moves.charge, moves.sigma, moves.epsilon], axis=1))
    # auto: as many ghosts as the sphere would hold if it were bulk water
    auto = Config.model_validate(dict(BASE, forcefield={"water": "amber14/opc.xml"}, sampling={
        "gcmc": dict(MOVES, radius_margin_nm=0.5)}))
    assert len(leg_system(auto, prep, "complex")[1].ghosts) == int(np.ceil(4.0 / 3.0 * np.pi * 0.5 ** 3 / 0.03))
    with pytest.raises(WorkflowError, match="sampling.gcmc cannot be used with this system"):
        leg_system(Config.model_validate(dict(BASE, forcefield={"water": "amber14/opc.xml"}, sampling={
            "gcmc": dict(MOVES, radius_margin_nm=0.5, ghost_waters=10000)})), prep, "complex")
    # the leg: no barostat, the moves, and a tail correction that knows every water
    built = alchemical_leg(moving, leg, prep, box, None, True, moves)
    spec = leg_spec(moving, built, [{"lambda_restraints": 1.0, "lambda_electrostatics": 1.0, "lambda_sterics": 1.0}],
                    "isotropic", 4.0, "openmm", 1)
    assert spec.gcmc is moves and spec.barostat == "none" and spec.pressure_bar is None and not spec.has_barostat
    assert spec.conditions()["settings"]["gcmc"]["cell_pressure_bar"] == 1.0
    tail = TailEnergy.from_dict(built.tail)
    oxygens = set(moves.waters[:, 0].tolist()) - set(ligand)
    assert oxygens <= set(tail.env.tolist())                       # the ghosts of the start too
    everything = tail.energy(positions, box)
    without = tail.energy(positions, box, off=np.array(sorted(off)))
    assert everything != without and np.isfinite(without)
    silent = TailEnergy.from_system(leg, ligand, built.alch.ligand_parameters, 0.9, 2.0)
    assert without == pytest.approx(silent.energy(positions, box), rel=1e-12)
    # the cell of the alchemical run
    eq = {"box": box * 1.001, "mean_box": box * np.array([[1.002], [1.0], [0.999]])}

    class Arrays(dict):
        files = property(lambda self: list(self))

    assert np.array_equal(alchemical_box(Arrays(eq), None), eq["box"])
    assert np.array_equal(alchemical_box(Arrays(eq), moves), eq["mean_box"])
    with pytest.raises(WorkflowError, match="does not record its mean cell"):
        alchemical_box(Arrays(box=eq["box"]), moves)
    molecules = [tuple(a.index for a in residue.atoms()) for residue in residues]
    moved = positions_in_cell(positions, eq["box"], eq["mean_box"], molecules)
    for atoms in molecules[:50]:
        index = list(atoms)
        shift = moved[index] - positions[index]
        assert np.allclose(shift, shift[0], atol=1e-12)            # the molecule as one body
        centre = positions[index].mean(axis=0)
        assert np.allclose(moved[index].mean(axis=0) @ np.linalg.inv(eq["mean_box"]),
                           centre @ np.linalg.inv(eq["box"]), atol=1e-12)


def test_a_leg_whose_ghosts_run_out_stops_with_the_reason(tmp_path, water_ligand_system):
    """With no ghost left an insertion cannot be tried, and the leg stops with what it needs (as grand
    does), in place of going on with fewer waters than the ensemble has. The sampler does not take that for
    a failure of a worker: nothing is restarted and nothing is repeated."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.sampling.hrex import ReplicaExchangeSampler
    from cufep.sampling.worker import LegSpec

    system, pos, box, ligand = water_ligand_system
    waters = np.array([[i, i + 1, i + 2] for i in range(0, system.getNumParticles(), 3) if i not in ligand])
    one = farthest_waters(waters, pos, box, np.array(ligand[:1]), 1)
    moves, alch, _tail = leg_with_moves(system, pos, box, ligand, ghosts=one)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 1})
    spec = LegSpec(name="gcmc", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=None,
                   barostat="none", timestep_fs=2.0, steps_per_iteration=20, seed=5, gcmc=moves)
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        with pytest.raises(GcmcError, match="no ghost water is left"):
            s.run(40, record_stride=1, checkpoint_stride=10)
        assert s.restarts == 0


def interacting_waters(n_waters: int, edge: float, seed: int):
    """``n_waters`` rigid three-site molecules with a dipole and a Lennard-Jones centre, and one charged
    Lennard-Jones particle at the centre of a cubic cell that is the ligand: (system, positions, box). The
    molecules interact with the ligand and with each other, cut at 0.7 nm, which is not where the stand-in
    energy cuts (0.9 nm): the potential of the simulation and the stand-in differ for every trial."""
    rng = np.random.default_rng(seed)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(edge, 0, 0), mm.Vec3(0, edge, 0), mm.Vec3(0, 0, edge))
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.7)
    force.setUseDispersionCorrection(False)
    positions = [np.full(3, 0.5 * edge)]
    system.addParticle(12.0)
    force.addParticle(0.4, 0.35, 2.0)
    shape = np.array([[0.0, 0.0, 0.0], [0.0957, 0.0, 0.0], [-0.024, 0.0927, 0.0]])
    for _ in range(n_waters):
        from cufep.sampling.relocation_mc import random_rotations

        positions.extend(rng.random(3) * edge + shape @ random_rotations(rng, 1)[0].T)
        first = system.getNumParticles()
        for mass, charge, epsilon in ((15.999, -0.4, 0.6), (1.008, 0.2, 0.0), (1.008, 0.2, 0.0)):
            system.addParticle(mass)
            force.addParticle(charge, 0.3, epsilon)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            force.addException(first + a, first + b, 0.0, 0.3, 0.0)
    system.addForce(force)
    return system, np.array(positions), np.eye(3) * edge


class NoStandIn(GcmcMover):
    """A mover whose trials are evaluated with no energy at all: every trial is then one Metropolis step
    with the potential energy of the simulation alone, the plain grand canonical move."""

    def _stand_in(self, x, box, state, on):
        class Nothing:
            def pose_energies(self, poses, moved, radius):
                return np.zeros(len(poses))

            def water_energies(self, which, radius):
                return np.zeros(len(which))

        return Nothing()


def waters_in_the_sphere(mover_class, seed: int, moves: int = 1200) -> np.ndarray:
    """The number of waters in the sphere after each of ``moves`` moves of a mover of ``mover_class`` on the
    interacting molecules, started from an empty sphere."""
    n, edge, radius = 40, 2.4, 0.6
    system, positions, _box = interacting_waters(n, edge, seed=3)
    # all molecules are ghosts at the start: the System the Context is made of has them switched off
    spec = ideal_spec(system, n, 3, radius, mean=3.0, attempts=6, batch_size=3)
    leg = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    ghosts_off(leg, spec.waters.reshape(-1))
    context = mm.Context(leg, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    mover = mover_class(spec, context, seed=seed)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    return np.array([mover.attempt(state)["waters_in_the_sphere"] for _ in range(moves)])


def test_trials_evaluated_with_the_stand_in_energy_give_the_waters_of_the_plain_move():
    """The trials of a move are evaluated with the stand-in energy and tested again with the potential energy
    of the simulation. Molecules that interact with the ligand and with each other, in a simulation whose
    potential is cut elsewhere than the stand-in, so that the two energies differ: the number of waters in
    the sphere is the one a mover gives whose trials use no stand-in at all (each trial one Metropolis step
    with the potential of the simulation). The stand-in changes how many trials pass, not what is sampled:
    with its energy counted twice in the second test, or with its sign turned for deletions, the two movers
    give other numbers of waters (15 and 23 in place of 4.3: checked once on such changes of the code; the
    acceptance of the single trials is held to the published expressions in the test of the stand-in energy)."""
    with_stand_in = waters_in_the_sphere(GcmcMover, seed=5)[100:]
    plain = waters_in_the_sphere(NoStandIn, seed=6)[100:]
    assert plain.mean() > 3.6                                   # (the interactions matter: 3.0 without them)
    # tolerance: about four times the scatter of the difference of two chains, from six chains of each mover
    # with other seeds (means 4.51 and 4.49, standard deviations 0.19 and 0.24)
    assert with_stand_in.mean() == pytest.approx(plain.mean(), abs=1.2)


def test_the_windows_started_in_this_process_make_the_moves_a_worker_makes(water_ligand_system):
    """Review of PR #19 (P2): the windows that a leg does not choose itself start with a short MD
    (cufep.engine.leg.starts_from). In a pool every worker made the moves of water in it (two in 10 ps at 4 ps), in
    this process none were made: the start depended on the GPUs a leg was granted. Started in this process, every
    window makes the moves too and its configuration says which waters interact."""
    from types import SimpleNamespace

    from cufep.alchemy.schedule import fixed_schedule
    from cufep.engine.leg import starts_from
    from cufep.sampling.worker import LegSpec

    system, pos, box, ligand = water_ligand_system
    moves, alch, _tail = leg_with_moves(system, pos, box, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 2})
    spec = LegSpec(name="gcmc", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=None,
                   barostat="none", timestep_fs=2.0, steps_per_iteration=10, seed=5, gcmc=moves)
    run = SimpleNamespace(job=SimpleNamespace(inline=True, control=None, devices=[None]), device0=None, seed=5,
                          cfg=SimpleNamespace(sampling=SimpleNamespace(time_per_iteration_ps=0.02)), log=lambda m: None)
    configs = starts_from(run, spec, schedule, (pos, box, np.zeros_like(pos)), alch.system, system.getNumParticles(),
                          10, 0.08)
    assert len(configs) == len(schedule.states)
    assert all(len(c) == 4 and c[3].dtype == bool and len(c[3]) == len(moves.waters) for c in configs)


def _chain_in_a_cell(xs, edge: float = 4.0, missing_bonds=()):
    """A chain of carbon atoms at ``xs`` (nm, in this order of atoms, bonded along x, but for the pairs of atoms
    ``missing_bonds``, which the topology does not hold) through the middle of a cubic cell of ``edge`` nm, and four
    waters far from it: (system, topology, positions, box, ligand)."""
    system, top = mm.System(), app.Topology()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9)
    chain = top.addChain()
    residue = top.addResidue("LIG", chain)
    atoms, positions = [], []
    for x in xs:
        system.addParticle(12.0)
        nb.addParticle(0.0, 0.34, 0.4)
        atoms.append(top.addAtom("C", app.Element.getBySymbol("C"), residue))
        positions.append([x, 0.5 * edge, 0.5 * edge])
    order = np.argsort(xs)
    for i, j in zip(order[:-1], order[1:]):
        if tuple(sorted((int(i), int(j)))) not in {tuple(sorted(pair)) for pair in missing_bonds}:
            top.addBond(atoms[i], atoms[j])
    for k in range(4):
        water = top.addResidue("HOH", chain)
        o = [3.05, 0.1 + 0.3 * k, 0.1]
        for name, element, q, s, e, dx in (("O", "O", -0.834, 0.315, 0.636, 0.0), ("H1", "H", 0.417, 0.1, 0.0, 0.0957),
                                           ("H2", "H", 0.417, 0.1, 0.0, -0.024)):
            system.addParticle(15.999 if element == "O" else 1.008)
            nb.addParticle(q, s, e)
            top.addAtom(name, app.Element.getBySymbol(element), water)
            positions.append([o[0] + dx, o[1] + (0.0927 if name == "H2" else 0.0), o[2]])
    system.addForce(nb)
    box = np.eye(3) * edge
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    top.setPeriodicBoxVectors([mm.Vec3(*row) for row in box])
    return system, top, np.asarray(positions), box, list(range(len(xs)))


def test_the_sphere_of_a_long_ligand_is_that_of_the_molecule_whole_whatever_its_first_atom():
    """Second review of PR #19 (P2): a chain of heavy atoms from x = 0 to 2.10 nm, 0.15 nm apart, in a cubic cell of
    4 nm, margin 0.30 nm: with the atom at x = 0 first, the image nearest to it put the atom at 2.10 nm at -1.90, the
    radius was 2.225 nm and the cell was refused; with the atom at 1.05 nm first it was 1.35 nm. The molecule is taken
    whole along its bonds: 1.35 nm whatever the first atom, for the water relocation moves, for the grand canonical
    moves, and for the two end states of a relative leg in the order of its call, A then B or B then A; the centre of
    the sphere is the middle of the chain. A molecule that needs no unwrapping has the pose of before."""
    from cufep.alchemy.lrc import nearest_image
    from cufep.rbfe.leg import end_states_heavy
    from cufep.sampling.gcmc import _centre
    from cufep.sampling.relocation_mc import relocation_spec, whole_pose

    xs = 0.15 * np.arange(15)
    middle_first = np.concatenate([xs[7:], xs[:7]])
    radii = []
    for order in (xs, middle_first):
        system, top, positions, box, ligand = _chain_in_a_cell(order)
        radii.append(relocation_spec(system, top, positions, box, ligand, 0.9, 0.5, 1, 298.15, 4, None, 4.0,
                                     0.3).radius_nm)
        spec = spec_of(system, top, positions, box, ligand, 2)
        radii.append(spec.radius_nm)
        assert _centre(positions, box, spec.ligand_heavy, spec.ligand_heavy_bonds) == pytest.approx([1.05, 2.0, 2.0])
    a = dict(zip(("topology", "positions", "ligand"), _chain_in_a_cell(xs)[1:3] + (list(range(15)),)))
    b = dict(zip(("topology", "positions", "ligand"), _chain_in_a_cell(middle_first[:-1])[1:3] + (list(range(14)),)))
    system, top, positions, box, ligand = _chain_in_a_cell(xs)
    for first, second in ((a, b), (b, a)):
        points, bonds = end_states_heavy(first, second)
        radii.append(spec_of(system, top, positions, box, ligand, 2, radius_from_nm=np.asarray(points),
                             radius_from_bonds=bonds).radius_nm)
    assert radii == pytest.approx([1.35] * 6, abs=1e-9)
    compact = positions[:5]
    assert np.array_equal(whole_pose(compact, [(i, i + 1) for i in range(4)], box),
                          compact[0] + nearest_image(compact - compact[0], box))


def test_the_sphere_takes_the_bonds_the_topology_does_not_hold_and_puts_the_two_end_states_together():
    """Third review of PR #19 (P2, twice), cells of 4 nm, margin 0.30 nm:
    the core of a hybrid along x from 0 to 1.95 nm and an atom unique to B at 2.10 nm, whose bond to the core the
    topology does not hold: that atom was put at -1.90 nm, the radius was 2.225 nm and the cell refused; with the bonds
    of the System (``bonds``) it is 1.35 nm, in the check of the cell and in the moves;
    end states A (x = 0 to 1.95 nm) and B (two atoms at 2.04 and 2.19 nm), each whole: with A first B was put at the
    images nearest to the first atom of A (-1.96 and -1.81 nm), 2.255 nm; with B first 1.395 nm. The end states are
    put together by their centres: 1.395 nm both ways."""
    from cufep.rbfe.leg import end_states_heavy
    from cufep.sampling.gcmc import _centre
    from cufep.sampling.relocation_mc import relocation_spec

    xs = 0.15 * np.arange(15)
    system, top, positions, box, ligand = _chain_in_a_cell(xs, missing_bonds=[(13, 14)])
    assert relocation_spec(system, top, positions, box, ligand, 0.9, 0.5, 1, 298.15, 4, None, 4.0, 0.3,
                           bonds=[(13, 14)]).radius_nm == pytest.approx(1.35)
    spec = spec_of(system, top, positions, box, ligand, 2, bonds=[(13, 14)])
    assert spec.radius_nm == pytest.approx(1.35) and [13, 14] in spec.ligand_heavy_bonds.tolist()
    assert _centre(positions, box, spec.ligand_heavy, spec.ligand_heavy_bonds) == pytest.approx([1.05, 2.0, 2.0])
    a_xs = 0.15 * np.arange(14)
    system, top, positions, box, ligand = _chain_in_a_cell(a_xs)
    a = dict(zip(("topology", "positions", "ligand"), _chain_in_a_cell(a_xs)[1:3] + (list(range(14)),)))
    b = dict(zip(("topology", "positions", "ligand"), _chain_in_a_cell([2.04, 2.19])[1:3] + ([0, 1],)))
    radii = []
    for first, second in ((a, b), (b, a)):
        points, bonds = end_states_heavy(first, second)
        radii.append(spec_of(system, top, positions, box, ligand, 2, radius_from_nm=np.asarray(points),
                             radius_from_bonds=bonds).radius_nm)
    assert radii == pytest.approx([1.395, 1.395], abs=1e-9)


def test_a_pre_equilibration_of_an_earlier_version_is_used_with_the_ghosts_of_the_earlier_rule(tmp_path):
    """Eleventh review of PR #19 (P3): a pre-equilibration of an absolute leg of an earlier version (no record of its
    ghosts) was refused where the images nearest to the first heavy atom of the ligand break it (a chain of 2.1 nm in
    a cell of 4 nm), though the rule of those versions (radius 1.955 nm from those images, the waters farthest from
    their centre) chose the ghost of this run (radius 1.080 nm). The ghosts of that rule are computed from the
    prepared system and the settings of the moves: the same are used, others are refused. A record that cannot be
    read is not used."""
    from cufep.config.schema import Config
    from cufep.engine.leg import EQUIL_GHOSTS, NO_GHOST_RECORD, equil_ghosts_problem
    from cufep.workflow import earlier_ghosts_of

    settings = Config.model_validate(dict(BASE, sampling={"gcmc": {**MOVES, "radius_margin_nm": 0.03,
                                                                     "ghost_waters": 1}})).sampling.gcmc
    equil = tmp_path / "equil.npz"
    equil.write_bytes(b"made by a version that recorded no ghosts")
    system, top, positions, box, ligand = _chain_in_a_cell(0.15 * np.arange(15))
    spec = spec_of(system, top, positions, box, ligand, 1, radius_margin_nm=0.03)
    assert spec.radius_nm == pytest.approx(1.08) and list(spec.ghosts) == [0]
    earlier = earlier_ghosts_of({"positions": positions, "box": box}, spec, settings)
    assert list(earlier) == [0] and equil_ghosts_problem(equil, spec, earlier) is None
    # waters that the two centres (x 0.783 and 1.05 nm) put in another order: the earlier rule chose another ghost
    moved = np.array(positions, dtype=float)
    for k, oxygen in enumerate(([3.05, 2.0, 0.1], [2.75, 2.0, 0.1], [1.0, 2.0, 0.1], [1.3, 2.0, 0.1])):
        moved[15 + 3 * k:18 + 3 * k] += np.asarray(oxygen) - moved[15 + 3 * k]
    spec = spec_of(system, top, moved, box, ligand, 1, radius_margin_nm=0.03)
    earlier = earlier_ghosts_of({"positions": moved, "box": box}, spec, settings)
    assert list(spec.ghosts) == [0] and list(earlier) == [1]
    assert equil_ghosts_problem(equil, spec, earlier) == NO_GHOST_RECORD
    earlier = earlier_ghosts_of({"positions": moved, "box": box}, spec, settings)
    # a record is the list of integers it is written as (twelfth review, P3: 0.9 was taken as 0, and 1e999 stopped
    # the leg)
    for text in ("", "{}", "[0]", '{"ghosts": [0.9]}', '{"ghosts": [1e999]}', '{"ghosts": [true]}',
                 '{"ghosts": [-1]}'):
        (tmp_path / EQUIL_GHOSTS).write_text(text)
        assert "cannot be read" in equil_ghosts_problem(equil, spec, earlier)
    (tmp_path / EQUIL_GHOSTS).write_text('{"ghosts": [0]}')
    assert equil_ghosts_problem(equil, spec, earlier) is None


def test_an_absolute_leg_with_the_moves_is_told_a_way_that_keeps_its_restraint(tmp_path, monkeypatch):
    """Fourth review of PR #19 (P2): a complex leg of the ABFE with the moves whose pre-equilibration cannot be used
    was told to remove equil.npz and run --reschedule, which its restraint (chosen from the MD that started from that
    pre-equilibration) then refuses, and which leaves the archive without the files its samples were made from. The
    run refuses it before it writes anything, with a new output directory as the way on, whatever --reschedule; start
    configurations in another cell than the pre-equilibration fixes are refused without --reschedule only."""
    import json
    from types import SimpleNamespace

    from cufep import workflow
    from cufep.config.schema import Config
    from cufep.engine.leg import EQUIL_GHOSTS

    leg_dir = tmp_path / "lig" / "repeat_0" / "complex"
    leg_dir.mkdir(parents=True)
    box = np.eye(3) * 4.0
    np.savez(leg_dir / "equil.npz", mean_box=box, box=box)
    moves = SimpleNamespace(ghosts=np.arange(5), waters=np.zeros((50, 3)))
    monkeypatch.setattr(workflow, "load_prepared", lambda prep_dir, leg: {})
    monkeypatch.setattr(workflow, "leg_system", lambda cfg, prep, leg: (None, moves))
    monkeypatch.setattr(workflow, "earlier_ghosts_of", lambda *args: None)
    cfg = Config.model_validate(dict(BASE, sampling={"gcmc": MOVES}))
    for reschedule in (False, True):
        with pytest.raises(workflow.WorkflowError) as refused:
            workflow.check_pre_equilibration_of_a_leg_with_moves(cfg, tmp_path / "lig", "complex", leg_dir, reschedule)
        assert "records no ghost waters" in str(refused.value) and "run the calculation in a new output.dir" in str(
            refused.value) and "remove" not in str(refused.value)
    (leg_dir / EQUIL_GHOSTS).write_text(json.dumps({"ghosts": list(range(5))}))
    workflow.check_pre_equilibration_of_a_leg_with_moves(cfg, tmp_path / "lig", "complex", leg_dir, False)
    from cufep.alchemy.schedule import fixed_schedule

    # start configurations without windows that have them are made again: not compared (sixth review of PR #19)
    np.savez(leg_dir / "start_configs.npz", boxes=np.stack([box * 1.01] * 2))
    workflow.check_pre_equilibration_of_a_leg_with_moves(cfg, tmp_path / "lig", "complex", leg_dir, False)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 1})
    states = [state.to_dict() for state in schedule.states]
    schedule.save(leg_dir / "schedule.yaml")                         # windows that have these start configurations
    np.savez(leg_dir / "start_configs.npz", boxes=np.stack([box * 1.01] * len(states)),
             made_with=json.dumps({"states": states}))
    with pytest.raises(workflow.WorkflowError, match="in another cell than its pre-equilibration fixes"):
        workflow.check_pre_equilibration_of_a_leg_with_moves(cfg, tmp_path / "lig", "complex", leg_dir, False)
    workflow.check_pre_equilibration_of_a_leg_with_moves(cfg, tmp_path / "lig", "complex", leg_dir, True)
