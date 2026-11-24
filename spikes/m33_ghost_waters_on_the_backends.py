"""Ghost waters on the two backends: can single water molecules be switched off and on in a live
Context, with the energies and forces of a System built with those waters absent?
(plan docs/plans/2026-11-24-gcmc-water.md, the risk check before the implementation)

    scripts/in_docker.sh --gpus 0 python spikes/m33_ghost_waters_on_the_backends.py

Grand canonical Monte Carlo of water in OpenMM keeps the number of particles of the System: a water
that is deleted stays in the System without nonbonded interactions (a ghost), and an insertion puts
a ghost at a new place and gives it its interactions back. In cuFEP that means, for every site of
the water, charge 0 and epsilon 0 in the NonbondedForce (on the cuPotential backend the plugin's own
force) and epsilon 0 in the soft-core force of the ligand, written into the Context while it runs
(``setParticleParameters`` and ``updateParametersInContext``). Nothing in cuFEP did that before.

The system is the one of the GPU tests of REST2: two alanine dipeptides in a 3.2 nm box of water,
one the ligand, part of the other hot, with the dispersion correction on as in a run; with TIP3P and
with OPC (four sites, the fourth a virtual site that carries the charge). Printed:

1. on stock OpenMM in double precision: a System with waters switched off when it is built against
   a System built from a topology without those waters, by force group (is a ghost an absent water?);
2. on each backend: a Context whose waters are switched off while it runs against a Context of the
   System built so, in three lambda windows: the energy by force group, the forces on the atoms, the
   reduced potentials of all windows; the same after the waters are switched on again; after random
   switches mixed with changes of the window; after a ghost is put at another place and switched on;
   after MD with ghosts;
3. each backend against stock OpenMM in double precision with an Ewald tolerance of 1e-6, with
   ghosts: the differences between the windows.

No time is measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import build_dipeptides_in_water  # noqa: E402

from cufep.alchemy.factory import SOFTCORE_FORCE_NAME, make_alchemical_system  # noqa: E402
from cufep.alchemy.schedule import fixed_schedule  # noqa: E402
from cufep.backends import get_backend  # noqa: E402
from cufep.sampling.states import StateEnergyEvaluator, set_state  # noqa: E402

KT = 8.31446261815324e-3 * 300.0
GROUPS = {"main": 0, "soft-core": 1, "restraint": 2, "REST2 torsions": 3}


def nonbonded_of(system):
    """The force of ``system`` that holds the charges and the Lennard-Jones parameters of all atoms:
    the stock NonbondedForce, or the plugin's force that took its place."""
    found = [system.getForce(i) for i in range(system.getNumForces())
             if type(system.getForce(i)).__name__ == "NonbondedForce"]
    assert len(found) == 1, [type(f) for f in found]
    return found[0]


def softcore_of(system):
    found = [system.getForce(i) for i in range(system.getNumForces())
             if isinstance(system.getForce(i), mm.CustomNonbondedForce)
             and system.getForce(i).getName() == SOFTCORE_FORCE_NAME]
    assert len(found) == 1
    return found[0]


class Switch:
    """Switches waters of a System off and on: in the System alone (before a Context is made of it)
    or in a Context made of it."""

    def __init__(self, system, original) -> None:
        self.nonbonded, self.softcore = nonbonded_of(system), softcore_of(system)
        source = nonbonded_of(original)
        self.parameters = [source.getParticleParameters(i) for i in range(original.getNumParticles())]
        names = [self.softcore.getPerParticleParameterName(k)
                 for k in range(self.softcore.getNumPerParticleParameters())]
        self.epsilon = names.index("epsilon")
        self.softcore_parameters = [list(softcore_of(original).getParticleParameters(i))
                                    for i in range(original.getNumParticles())]

    def set(self, atoms, on: bool, context=None) -> None:
        for i in (int(a) for a in atoms):
            q, sigma, epsilon = self.parameters[i]
            values = list(self.softcore_parameters[i])
            if not on:
                q, epsilon = q * 0.0, epsilon * 0.0
                values[self.epsilon] = 0.0
            self.nonbonded.setParticleParameters(i, q, sigma, epsilon)
            self.softcore.setParticleParameters(i, values)
        if context is not None:
            self.nonbonded.updateParametersInContext(context)
            self.softcore.updateParametersInContext(context)


class Leg:
    """One backend with the system: makes Contexts of the System as it is, or with waters off."""

    def __init__(self, name: str, alch_system, states, box, tight: bool = False, **backend) -> None:
        self.name, self.states, self.box = name, states, box
        self.backend = get_backend(name, **backend)
        self.source = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(alch_system))
        if tight:
            nonbonded_of(self.source).setEwaldErrorTolerance(1e-6)

    def context(self, off=()):
        """(Context, Switch, prepared System) with the atoms ``off`` switched off when it is built."""
        system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(self.source))
        Switch(system, self.source).set(off, False)
        prepared = self.backend.prepare_system(system)
        integrator = self.backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=7)
        context = self.backend.make_context(prepared, integrator, 0)
        context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in self.box])
        return context, Switch(prepared, self.source), prepared


def numbers(context, states, positions, real) -> dict:
    """What is compared of a Context at ``positions``: the energy by force group and the forces on
    the atoms in three windows, and the reduced potentials of all windows."""
    out = {"groups": [], "forces": []}
    for index in (0, len(states) // 2, len(states) - 1):
        set_state(context, states[index])
        context.setPositions(positions)
        context.getState(getEnergy=True)                    # (all groups once: the lists of the soft-core force)
        out["groups"].append([context.getState(getEnergy=True, groups=1 << g).getPotentialEnergy()
                              .value_in_unit(u.kilojoule_per_mole) for g in GROUPS.values()])
        forces = context.getState(getForces=True).getForces(asNumpy=True)
        out["forces"].append(np.asarray(forces.value_in_unit(u.kilojoule_per_mole / u.nanometer))[real])
    set_state(context, states[0])
    context.setPositions(positions)
    out["u"] = StateEnergyEvaluator(context, states, 300.0, None).reduced_potentials()[0]
    return {key: np.asarray(value) for key, value in out.items()}


def compare(label: str, a: dict, b: dict) -> None:
    groups = np.abs(a["groups"] - b["groups"]).max(axis=0)
    forces = np.linalg.norm(a["forces"] - b["forces"]) / np.linalg.norm(b["forces"])
    windows = np.abs((a["u"] - a["u"][0]) - (b["u"] - b["u"][0])).max()
    print(f"  {label}: energy by group, largest difference (kJ/mol) "
          + ", ".join(f"{name} {value:.4g}" for name, value in zip(GROUPS, groups))
          + f"; total {np.abs(a['groups'].sum(axis=1) - b['groups'].sum(axis=1)).max():.4g} of "
          f"{np.abs(b['groups'].sum(axis=1)).max():.6g}; forces, relative {forces:.2e}; "
          f"between windows {windows:.4f} kT; u of window 0 {abs(a['u'][0] - b['u'][0]):.4f} kT")


def absent_system(water: str, topology, positions, gone, ligand, other, rest2: bool, dispersion: bool):
    """The alchemical System built from the topology without the waters ``gone`` (residues), and the
    positions of its atoms."""
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", water)
    model = app.Modeller(topology, positions * u.nanometer)
    model.delete(gone)
    system = forcefield.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setUseDispersionCorrection(dispersion)
    alch = make_alchemical_system(system, ligand, rest2=rest2, rest2_atoms=other[:12] if rest2 else ())
    return alch.system, np.asarray(model.positions.value_in_unit(u.nanometer))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--toggles", type=int, default=40, help="random switches mixed with changes of the window")
    parser.add_argument("--no-rest2", action="store_true", help="the system and the windows without REST2")
    parser.add_argument("--no-dispersion-correction", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(11)
    for water in ("amber14/tip3p.xml", "amber14/opc.xml"):
        system, topology, pos, box, ligand, other = build_dipeptides_in_water(
            box_nm=3.2, dispersion_correction=not args.no_dispersion_correction, water=water)
        rest2 = not args.no_rest2
        alch = make_alchemical_system(system, ligand, rest2=rest2, rest2_atoms=other[:12] if rest2 else ())
        schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 4},
                                  **({"rest2_scale": 3.0} if rest2 else {}))
        states = schedule.states
        residues = [r for r in topology.residues() if r.name == "HOH"]
        waters = np.array([[a.index for a in r.atoms()] for r in residues])
        real = np.array([i for i in range(system.getNumParticles()) if not system.isVirtualSite(i)])
        centre = pos[ligand].mean(axis=0)
        by_distance = np.argsort(np.linalg.norm(pos[waters[:, 0]] - centre, axis=1))
        chosen = np.concatenate([by_distance[:3], by_distance[-3:]])      # three next to the ligand, three far
        off = waters[chosen].reshape(-1)
        print(f"== {water}: {len(waters)} waters of {waters.shape[1]} sites, {system.getNumParticles()} particles; "
              f"waters switched off: {chosen.tolist()}")

        # 1. is a ghost an absent water? (stock OpenMM, double precision)
        reference = Leg("openmm", alch.system, states, box, precision="double")
        built_off = numbers(reference.context(off)[0], states, pos, real)
        without, pos_without = absent_system(water, topology, pos, [residues[k] for k in chosen], ligand, other,
                                             rest2, not args.no_dispersion_correction)
        real_without = np.array([i for i in range(without.getNumParticles()) if not without.isVirtualSite(i)])
        kept = np.array([i for i in real if i not in set(off.tolist())])
        gone = Leg("openmm", without, states, box, precision="double")
        absent = numbers(gone.context()[0], states, pos_without, real_without)
        built_off_kept = dict(built_off, forces=built_off["forces"][:, np.isin(real, kept)])
        compare("stock, double: switched off when built against built without the waters", built_off_kept, absent)

        tight = Leg("openmm", alch.system, states, box, tight=True, precision="double")
        tight_off = numbers(tight.context(off)[0], states, pos, real)
        tight_on = numbers(tight.context()[0], states, pos, real)
        change = (tight_off["groups"].sum(axis=1) - tight_on["groups"].sum(axis=1))[0]
        print(f"  stock, double, tight Ewald sum: the energy changes by {change:.3f} kJ/mol when the six waters are "
              "switched off (first window)")
        for name in ("openmm", "cupotential"):
            leg = Leg(name, alch.system, states, box)
            print(f" {name}:")
            built = numbers(leg.context(off)[0], states, pos, real)
            context, switch, _prepared = leg.context()
            all_on = numbers(context, states, pos, real)
            switch.set(off, False, context)
            compare("switched off while it runs against switched off when built",
                    numbers(context, states, pos, real), built)
            windows = np.abs((built["u"] - built["u"][0]) - (tight_off["u"] - tight_off["u"][0])).max()
            by_window = (all_on["u"] - all_on["u"][0]) - (tight_on["u"] - tight_on["u"][0])
            change = (built["groups"].sum(axis=1) - all_on["groups"].sum(axis=1))[0]
            print(f"  the energy changes by {change:.3f} kJ/mol when the six waters are switched off (first window)")
            print(f"  without ghosts, against stock in double precision with a tight Ewald sum: between windows "
                  f"{np.abs(by_window).max():.4f} kT, by window {np.round(by_window, 3).tolist()}")
            print(f"  with ghosts, against stock in double precision with a tight Ewald sum: between windows "
                  f"{windows:.4f} kT")
            switch.set(off, True, context)
            compare("switched on again against never switched", numbers(context, states, pos, real), all_on)

            # random switches mixed with changes of the window
            mask = np.ones(len(waters), dtype=bool)
            for _ in range(args.toggles):
                k = int(rng.integers(len(waters)))
                mask[k] = not mask[k]
                set_state(context, states[int(rng.integers(len(states)))])
                switch.set(waters[k], bool(mask[k]), context)
                context.getState(getEnergy=True)
            now_off = waters[~mask].reshape(-1)
            compare(f"after {args.toggles} random switches ({int((~mask).sum())} waters off) against built so",
                    numbers(context, states, pos, real), numbers(leg.context(now_off)[0], states, pos, real))

            # a ghost put at another place and switched on
            ghost = int(np.flatnonzero(~mask)[0])
            moved = pos.copy()
            moved[waters[ghost]] += rng.normal(size=3) * 0.4
            context.setPositions(moved)
            switch.set(waters[ghost], True, context)
            mask[ghost] = True
            compare("a ghost at a new place, switched on, against built so",
                    numbers(context, states, moved, real),
                    numbers(leg.context(waters[~mask].reshape(-1))[0], states, moved, real))

            # MD with ghosts, then the energies of where it ended
            set_state(context, states[len(states) // 2])
            context.setPositions(pos)
            context.setVelocitiesToTemperature(300 * u.kelvin, 3)
            context.getIntegrator().step(250)
            state = context.getState(getPositions=True, getEnergy=True)
            end = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
            energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            print(f"  after 250 steps of MD with ghosts: potential energy {energy:.1f} kJ/mol, finite "
                  f"{bool(np.isfinite(energy) and np.isfinite(end).all())}")
            compare("where the MD ended, against built so",
                    numbers(context, states, end, real),
                    numbers(leg.context(waters[~mask].reshape(-1))[0], states, end, real))
            del context


if __name__ == "__main__":
    main()
