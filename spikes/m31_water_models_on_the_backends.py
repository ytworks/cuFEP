"""TIP3P and OPC (4 sites, the fourth a virtual site) on the two backends: are the energies between
states and the forces of the cuPotential plugin those of stock OpenMM, and how far is each from a
tight Ewald reference? (plan docs/plans/2026-11-24-opc-water-model.md)

    scripts/in_docker.sh --gpus 0 python spikes/m31_water_models_on_the_backends.py [--frames 10]

The system and the lambda windows are those of the GPU test ``test_rest2_backends_agree``
(``tests/gpu/test_gpu.py``): two alanine dipeptides in a 3.2 nm box of water, one the ligand, with
REST2. The first configuration is the one of that test; the others follow it by 2 ps of MD each on
stock OpenMM. For every configuration four Contexts give the reduced potential of every window:

    stock   stock OpenMM, mixed precision, Ewald tolerance 5e-4 (what a leg runs on)
    double  stock OpenMM, double precision, the same tolerance
    cupo    the cuPotential plugin
    tight   stock OpenMM, double precision, Ewald tolerance 1e-6 (the reference of docs/M0_REPORT.md, 2b)

Printed per water model: the difference between windows (relative to the first) of the plugin against
stock, which is what the test holds to 0.1 kT, and of each against the tight reference; the potential
energy and the forces of the plugin against stock in double precision; the forces on the virtual
sites. No time is measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as u

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import build_dipeptides_in_water  # noqa: E402

from cufep.alchemy.factory import make_alchemical_system  # noqa: E402
from cufep.alchemy.schedule import fixed_schedule  # noqa: E402
from cufep.backends import get_backend  # noqa: E402
from cufep.sampling.states import StateEnergyEvaluator  # noqa: E402

KJ_PER_MOL = u.kilojoule_per_mole
KT = 8.31446261815324e-3 * 300.0


def make_context(backend, system, seed: int = 1):
    integ = backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=seed)
    return backend.make_context(backend.prepare_system(system), integ, 0), integ


def load(ctx, pos, box) -> None:
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)


def measure(water: str, n_frames: int) -> None:
    system, top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2, water=water)
    alch = make_alchemical_system(system, ligand, rest2=True, rest2_atoms=other[:12])
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 4}, rest2_scale=3.0)
    n = system.getNumParticles()
    sites = np.array([i for i in range(n) if system.isVirtualSite(i)], dtype=int)
    water_atoms = np.array([a.index for a in top.atoms() if a.residue.name == "HOH" and a.element is not None])

    # configurations: the one of the test, then MD on stock OpenMM
    ctx, integ = make_context(get_backend("openmm"), alch.system)
    load(ctx, pos, box)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 7)
    frames = [np.array(pos)]
    for _ in range(n_frames):
        integ.step(1000)
        frames.append(np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)))
    del ctx, integ

    tight = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(alch.system))
    for force in tight.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setEwaldErrorTolerance(1e-6)
    made = {"stock": (get_backend("openmm"), alch.system),
            "double": (get_backend("openmm", precision="double"), alch.system),
            "cupo": (get_backend("cupotential"), alch.system),
            "tight": (get_backend("openmm", precision="double"), tight)}
    contexts = {key: make_context(backend, sys_)[0] for key, (backend, sys_) in made.items()}
    evaluators = {key: StateEnergyEvaluator(c, schedule.states, 300.0, 1.0) for key, c in contexts.items()}
    reduced = {key: [] for key in contexts}
    energy = {key: [] for key in contexts}
    forces = {key: [] for key in contexts}
    for frame in frames:
        for key, c in contexts.items():
            load(c, frame, box)
            st = c.getState(getEnergy=True, getForces=True)
            energy[key].append(st.getPotentialEnergy().value_in_unit(KJ_PER_MOL))
            forces[key].append(np.asarray(st.getForces(asNumpy=True).value_in_unit(KJ_PER_MOL / u.nanometer)))
            reduced[key].append(evaluators[key].reduced_potentials()[0])
    red = {key: np.array(v) for key, v in reduced.items()}
    between = {key: v - v[:, :1] for key, v in red.items()}          # relative to the first window, kT

    print(f"\n== {water}: {n} particles, {len(sites)} virtual sites, {len(frames)} configurations, "
          f"{len(schedule.states)} windows")
    test = np.abs(between["cupo"] - between["stock"])
    print("  between windows, plugin against stock (the quantity of the test, level 0.1 kT):")
    print(f"    configuration of the test: max {test[0].max():.4f} kT   per window "
          + " ".join(f"{x:.3f}" for x in test[0]))
    print(f"    all configurations: max {test.max():.4f}, mean of the maxima {test.max(axis=1).mean():.4f}, "
          f"maxima " + " ".join(f"{x:.3f}" for x in test.max(axis=1)))
    # every window is compared, as in the test (at a configuration of the first window the others lie
    # up to 160 kT above it); `neighbours` is the same between one window and the next
    print(f"  between windows, against the tight Ewald reference (1e-6, double), kT "
          f"(windows span {between['tight'].min():.0f} to {between['tight'].max():.0f} kT):")
    for key in ("stock", "double", "cupo"):
        err = np.abs(between[key] - between["tight"])[:, 1:]
        step = np.abs(np.diff(red[key], axis=1) - np.diff(red["tight"], axis=1))
        print(f"    {key:6s} max {err.max():.4f}  mean {err.mean():.4f}  "
              f"mean of the maxima {err.max(axis=1).mean():.4f}"
              f"   neighbours: max {step.max():.4f}  mean {step.mean():.4f}")
    step = np.abs(np.diff(red["cupo"], axis=1) - np.diff(red["stock"], axis=1))
    print(f"  between neighbouring windows, plugin against stock: max {step.max():.4f}  mean {step.mean():.4f} kT")
    e_d, e_c, e_t = (np.array(energy[k]) for k in ("double", "cupo", "tight"))

    def offset(a, b) -> str:
        return f"offset {np.mean(a - b) / KT:+.3f} kT (sd {np.std(a - b) / KT:.3f})"

    print(f"  potential energy: plugin against stock double, max relative "
          f"{np.max(np.abs(e_c - e_d) / np.abs(e_d)):.2e}; {offset(e_c, e_d)}; "
          f"stock double against tight, {offset(e_d, e_t)}; plugin against tight, {offset(e_c, e_t)}")

    def relative(a, b, atoms=None):
        out = []
        for fa, fb in zip(forces[a], forces[b]):
            if atoms is not None:
                fa, fb = fa[atoms], fb[atoms]
            out.append(np.linalg.norm(fa - fb) / np.linalg.norm(fb))
        return max(out)

    print(f"  forces, max over configurations of |F_a - F_b| / |F_b|: "
          f"plugin against stock double {relative('cupo', 'double'):.2e} "
          f"(water atoms {relative('cupo', 'double', water_atoms):.2e}, "
          f"ligand {relative('cupo', 'double', np.array(ligand)):.2e}); "
          f"stock mixed against double {relative('stock', 'double'):.2e}; "
          f"stock double against tight {relative('double', 'tight'):.2e}; "
          f"plugin against tight {relative('cupo', 'tight'):.2e}")
    if len(sites):
        on_sites = {key: max(float(np.abs(f[sites]).max()) for f in forces[key]) for key in ("double", "cupo")}
        print(f"  largest force component on a virtual site (kJ/mol/nm): stock double {on_sites['double']:.3g}, "
              f"plugin {on_sites['cupo']:.3g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=10, help="configurations of MD after the one of the test")
    args = parser.parse_args()
    for water in ("amber14/tip3p.xml", "amber14/opc.xml"):
        measure(water, args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
