"""M2 smoke test: lambda search + multi-GPU replica exchange + MBAR on the TYK2 complex.

Uses the serialized alchemical System written by spike 2 (no restraint: this only
exercises the sampling machinery, it is not a binding free energy).

Usage: spikes/run.sh spikes/m2_hrex_smoke.py [--devices 6,7] [--iterations 40] [--backend cupotential]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as u

from cufep.alchemy.schedule import ScheduleSettings, trailblaze
from cufep.analysis.mbar import analyze_leg
from cufep.backends import get_backend
from cufep.sampling.hrex import ReplicaExchangeSampler
from cufep.sampling.storage import read_samples
from cufep.sampling.worker import LegSpec

OUT = Path("spikes/out")
TAG = "tyk2_ejm_31_cube"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--backend", default="cupotential")
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--max-windows", type=int, default=48)
    args = ap.parse_args()
    devices = [int(d) for d in args.devices.split(",")]
    xml = (OUT / f"{TAG}_alch_system.xml").read_text()
    data = np.load(OUT / f"{TAG}_equil.npz")
    system = mm.XmlSerializer.deserialize(xml)
    T = 298.15

    # ---- lambda search on the first device ------------------------------------------------
    backend = get_backend(args.backend)
    prepared = backend.prepare_system(system)
    prepared.addForce(backend.make_isotropic_barostat(1 * u.bar, T * u.kelvin, 25, seed=1))
    integ = backend.make_integrator(T * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=2)
    ctx = backend.make_context(prepared, integ, devices[0])
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in data["box"]])
    ctx.setPositions(data["positions"]); ctx.setVelocities(data["velocities"])
    t0 = time.perf_counter()
    schedule, configs = trailblaze(ctx, integ, ["restraints", "electrostatics", "sterics"], T, 4.0,
                                   ScheduleSettings(target_std_kT=args.target, max_windows=args.max_windows))
    print(f"lambda search took {time.perf_counter() - t0:.0f} s; {len(schedule.states)} states")
    for st, sd in zip(schedule.states, schedule.step_std_kT):
        print(f"   elec {st.lambda_electrostatics:.4f}  sterics {st.lambda_sterics:.4f}   std to next {sd:.2f} kT")
    del ctx, integ

    # ---- replica exchange --------------------------------------------------------------------
    leg_dir = OUT / "hrex_smoke"
    shutil.rmtree(leg_dir, ignore_errors=True)
    spec = LegSpec(name="tyk2-smoke", system_xml=xml, states=[s.to_dict() for s in schedule.states],
                   n_atoms=system.getNumParticles(), backend=args.backend, temperature_K=T, pressure_bar=1.0,
                   barostat="isotropic", timestep_fs=4.0, steps_per_iteration=625, seed=11,
                   frame_atoms=[int(i) for i in data["ligand"]])
    with ReplicaExchangeSampler(spec, leg_dir, devices=devices) as sampler:
        sampler.initialize(np.array([c[0] for c in configs]), np.array([c[1] for c in configs]),
                           np.array([c[2] for c in configs]))
        t0 = time.perf_counter()
        stats = sampler.run(args.iterations, record_stride=4, checkpoint_stride=20, progress_every=10)
        wall = time.perf_counter() - t0
        print(f"{args.iterations} iterations of {len(schedule.states)} replicas on {len(devices)} GPU(s): "
              f"{wall / args.iterations:.2f} s/iteration; mean fraction of replicas changing state "
              f"{np.mean([s.swap_fraction for s in stats]):.2f}; swap acceptance "
              f"{sampler.accepted / max(1, sampler.attempted):.2f}")
    res = analyze_leg(read_samples(leg_dir), T, n_equilibration=min(8, args.iterations // 4))
    print(f"MBAR: dG(decouple) = {res.delta_g / 4.184:.2f} ± {res.delta_g_error / 4.184:.2f} kcal/mol; "
          f"min neighbour overlap {res.min_overlap:.3f}; g = {res.statistical_inefficiency:.1f}; "
          f"round trips {res.round_trips}; stages {{{', '.join(f'{k}: {v / 4.184:.2f}' for k, v in res.stage_delta_g.items())}}}")
    print("neighbour overlaps:", np.round(res.overlap_neighbour, 2).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
