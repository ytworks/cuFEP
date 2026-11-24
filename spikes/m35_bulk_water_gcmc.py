"""Bulk water with grand canonical moves in a sphere: does the sphere hold the water of the liquid?
(plan docs/plans/2026-11-24-gcmc-water.md, the check of the two numbers of a water model)

    scripts/in_docker.sh --gpus 0 python spikes/m35_bulk_water_gcmc.py --water amber14/tip3p.xml \
        --mu -6.09 --volume 0.030345

The moves of ``sampling.gcmc`` put the sphere in equilibrium with a reservoir whose water has the
excess chemical potential mu'_sol and the standard volume V° of the Adams value. If the two numbers
are those of the water model with the interactions cut as a run cuts them, a sphere in the bulk
liquid keeps, on average, V_sphere / V° waters, and the box keeps the waters it had at 1 bar; with
a mu'_sol that is too high the moves put water in until the pressure has risen by what the error
is worth (0.05 kcal/mol are about 115 bar and 0.5 % of the density).

The system: a box of water with the settings of a run (PME, cutoff 0.9 nm, dispersion correction,
rigid water) and one particle without interactions that stands for the ligand: the sphere is
centred on it and goes where it goes. First the liquid with every water at 1 bar gives the volume
per molecule with these settings (``--volume`` overrides it as the standard volume of the Adams
value). Then the ghosts of the start, the waters farthest from that particle, are switched off and
the cell is made smaller by their share, so that the waters that interact start at the density of
1 bar, and MD at constant volume alternates with moves (``--interval-ps``, ``--attempts``,
``--batch-size``). Printed: the waters in the sphere after every move against V_sphere / V° (mean,
standard error from blocks), the waters that interact in the whole cell against the start, and the
trials and the moves made. The number of waters of the whole cell fluctuates in the grand canonical
ensemble by sqrt(N kT rho kappa), about 0.7 % for 1,240 waters, and slowly: a run of a few ns
resolves the sphere better than the cell. No time is measured.
"""
from __future__ import annotations

import argparse

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from cufep.backends import get_backend
from cufep.prep.water import water_model_of
from cufep.sampling.gcmc import COUNTS, GcmcMover, gcmc_spec, ghosts_off
from cufep.sampling.states import AlchemicalState
from cufep.workflow import positions_in_cell


def build(water: str, edge: float):
    """(System with every water, topology, positions in nm, box in nm, the atom that stands for the ligand)."""
    forcefield = app.ForceField(water)
    model = app.Modeller(app.Topology(), [])
    model.addSolvent(forcefield, model=water_model_of(water).box, boxSize=mm.Vec3(edge, edge, edge) * u.nanometer)
    topology = model.topology
    chain = topology.addChain()
    centre = topology.addAtom("C1", app.Element.getBySymbol("C"), topology.addResidue("DUM", chain))
    positions = list(model.positions.value_in_unit(u.nanometer)) + [mm.Vec3(0.5 * edge, 0.5 * edge, 0.5 * edge)]
    # the particle without interactions is not in the force field: the System is made of the waters and it is added
    waters_only = app.Topology()
    waters_only.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    by_chain = {}
    for residue in topology.residues():
        if residue.name != "HOH":
            continue
        new_chain = by_chain.setdefault(residue.chain.index, waters_only.addChain())
        new_residue = waters_only.addResidue("HOH", new_chain)
        atoms = {atom: waters_only.addAtom(atom.name, atom.element, new_residue) for atom in residue.atoms()}
        for a, b in residue.internal_bonds():
            waters_only.addBond(atoms[a], atoms[b])
    system = forcefield.createSystem(waters_only, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    nonbonded.setUseDispersionCorrection(True)
    system.addParticle(12.0)
    nonbonded.addParticle(0.0, 0.3, 0.0)
    assert centre.index == system.getNumParticles() - 1
    return system, topology, np.array([[p.x, p.y, p.z] for p in positions]), np.eye(3) * edge, centre.index


def blocks(values: np.ndarray, n: int = 5) -> float:
    parts = np.array_split(np.asarray(values, dtype=float), n)
    return float(np.std([part.mean() for part in parts], ddof=1) / np.sqrt(n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--water", default="amber14/tip3p.xml")
    parser.add_argument("--mu", type=float, required=True, help="excess chemical potential, kcal/mol")
    parser.add_argument("--volume", type=float, default=None,
                        help="standard volume, nm^3 (default: the volume per water measured here at 1 bar)")
    parser.add_argument("--timestep-fs", type=float, default=4.0)
    parser.add_argument("--backend", default="cupotential")
    parser.add_argument("--box-nm", type=float, default=3.5)
    parser.add_argument("--radius-nm", type=float, default=0.8)
    parser.add_argument("--ghosts", type=int, default=150)
    parser.add_argument("--npt-ns", type=float, default=0.4)
    parser.add_argument("--ns", type=float, default=1.0, help="MD with the moves")
    parser.add_argument("--interval-ps", type=float, default=4.0)
    parser.add_argument("--attempts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    temperature, step = args.temperature * u.kelvin, args.timestep_fs * u.femtoseconds
    per_ps = int(round(1000.0 / args.timestep_fs))
    system, topology, positions, box, centre = build(args.water, args.box_nm)
    backend = get_backend(args.backend)
    n_waters = sum(1 for residue in topology.residues() if residue.name == "HOH")

    # 1. the liquid at 1 bar with every water: its volume per molecule with these settings
    npt = backend.prepare_system(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)))
    npt.addForce(backend.make_isotropic_barostat(1.0 * u.bar, temperature, 25, seed=args.seed + 1))
    context = backend.make_context(npt, backend.make_integrator(temperature, 1 / u.picosecond, step, seed=args.seed), 0)
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 500)
    context.setVelocitiesToTemperature(temperature, args.seed)
    volumes = []
    for _ in range(max(10, int(round(args.npt_ns * 1000.0)))):
        context.getIntegrator().step(per_ps)
        cell = context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        volumes.append(abs(float(np.linalg.det(cell))))
    state = context.getState(getPositions=True, getVelocities=True)
    last_box = np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
    liquid = np.array(volumes[len(volumes) // 2:]) / n_waters
    measured = float(liquid.mean())
    standard = measured if args.volume is None else args.volume
    print(f"{args.water} on {args.backend}, time step {args.timestep_fs:g} fs: {n_waters} waters; at 1 bar "
          f"{1000 * measured:.3f} ± {1000 * blocks(liquid):.3f} A3 per water (second half of {args.npt_ns:g} ns); "
          f"standard volume used {1000 * standard:.3f} A3", flush=True)
    molecules = context.getMolecules()
    end = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    velocities = state.getVelocities(asNumpy=True)
    del context

    # 2. the ghosts are switched off, and the cell is made smaller by their share: the waters that interact
    # keep the volume per molecule of the liquid at 1 bar
    moves = gcmc_spec(system, topology, end, last_box, [centre], 0.9, 0.5, 1, args.temperature, args.interval_ps,
                      args.attempts, args.batch_size, args.radius_nm, args.mu, standard, args.ghosts)
    ghosts_off(system, moves.waters[moves.ghosts].reshape(-1))
    n_on = int(moves.initial_on().sum())
    fixed_box = last_box * (n_on * measured / abs(float(np.linalg.det(last_box)))) ** (1.0 / 3.0)
    start = positions_in_cell(end, last_box, fixed_box, molecules)
    expected = 4.0 / 3.0 * np.pi * moves.radius_nm ** 3 / standard
    print(f"{len(moves.ghosts)} ghosts at the start, {n_on} waters interact in a cell of "
          f"{abs(float(np.linalg.det(fixed_box))):.3f} nm3; sphere of {moves.radius_nm:.2f} nm, B = {moves.adams:.3f}, "
          f"V_sphere / V° = {expected:.2f} waters", flush=True)
    fixed = backend.prepare_system(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)))
    integrator = backend.make_integrator(temperature, 1 / u.picosecond, step, seed=args.seed + 2)
    context = backend.make_context(fixed, integrator, 0)
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in fixed_box])
    context.setPositions(start)
    context.setVelocities(velocities)
    mover = GcmcMover(moves, context, seed=args.seed + 3)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    steps = int(round(args.interval_ps * per_ps))
    n_moves = int(round(args.ns * 1000.0 / args.interval_ps))
    in_sphere, interacting, totals = [], [], dict.fromkeys(COUNTS[:4], 0)
    for move in range(n_moves):
        integrator.step(steps)
        counts = mover.attempt(state)
        for key in totals:
            totals[key] += counts[key]
        in_sphere.append(counts[COUNTS[4]])
        interacting.append(int(mover.on.sum()))
        if (move + 1) % max(1, n_moves // 10) == 0:
            print(f"  after {move + 1} moves ({(move + 1) * args.interval_ps / 1000.0:.2f} ns): waters in the sphere "
                  f"{np.mean(in_sphere[-(n_moves // 10):]):.2f} (last tenth), waters that interact "
                  f"{interacting[-1]} ({interacting[-1] - n_on:+d})", flush=True)
    half = n_moves // 2
    sphere, box_waters = np.array(in_sphere[half:], dtype=float), np.array(interacting[half:], dtype=float)
    print(f"second half: waters in the sphere {sphere.mean():.2f} ± {blocks(sphere):.2f} (V_sphere / V° = "
          f"{expected:.2f}); waters that interact in the box {box_waters.mean():.1f} ± {blocks(box_waters):.1f} "
          f"(start {n_on}; {100.0 * (box_waters.mean() / n_on - 1.0):+.2f} %)")
    print("trials and moves: " + ", ".join(f"{key} {value}" for key, value in totals.items()))


if __name__ == "__main__":
    main()
