"""Run only the adaptive precalculation of a finished leg with other settings.

Rebuilds the leg from the files of an earlier run and runs adaptive lambda scheduling with the
given parameters, with the windows of an iteration in a pool of workers. Prints the windows it
chooses and every iteration. Nothing is written to the run directory.

    spikes/m3_speed_mps.sh spikes/m9_als_variants.py runs/tyk2_default_5gpu ejm_31 0 complex \
        --devices 0,0,0,0,0,0,0,0 --window-ps 250 --threshold 0.04
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ligand")
    ap.add_argument("repeat", type=int)
    ap.add_argument("leg")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--window-ps", type=float, default=50.0)
    ap.add_argument("--equilibration-ps", type=float, default=0.0)
    ap.add_argument("--sample-interval-ps", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.04)
    ap.add_argument("--initial", default="3,5,9", help="initial windows for restrain,discharge,vanish")
    ap.add_argument("--serial", action="store_true", help="one window after another in a single Context")
    args = ap.parse_args()

    import openmm as mm
    import openmm.unit as u

    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.alchemy.schedule import AlsSettings, adaptive_schedule
    from cufep.backends import get_backend
    from cufep.config.schema import Config
    from cufep.mps import thread_percentage
    from cufep.sampling.hrex import WindowPool
    from cufep.sampling.states import STAGES_COMPLEX, STAGES_SOLVENT, stage_state
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import load_prepared

    run_dir = Path(args.run_dir)
    raw = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    # runs made before initial_windows became a per-leg setting; this script does not use it
    raw.get("lambda_schedule", {}).get("adaptive", {}).pop("initial_windows", None)
    cfg = Config.model_validate(raw)
    lig_dir = run_dir / args.ligand
    leg_dir = lig_dir / f"repeat_{args.repeat}" / args.leg
    prep = load_prepared(lig_dir / "prep", args.leg)
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    eq = np.load(leg_dir / "equil.npz")
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
    alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha)
    if args.leg == "complex":
        alch.system.addForce(
            BoreschRestraint.from_dict(json.loads((leg_dir / "restraint.json").read_text())["restraint"]).create_force()
        )
    stages = list(STAGES_COMPLEX if args.leg == "complex" else STAGES_SOLVENT)
    T, dt = cfg.thermodynamics.temperature_K, float(eq["timestep_fs"])
    backend_name = yaml.safe_load((leg_dir / "schedule.yaml").read_text()).get("backend", "cupotential")
    backend = get_backend(backend_name, precision="mixed")
    prepared = backend.prepare_system(alch.system)
    prepared.addForce(
        backend.make_isotropic_barostat(cfg.thermodynamics.pressure_bar * u.bar, T * u.kelvin, 25, seed=3)
    )
    integ = backend.make_integrator(
        T * u.kelvin, cfg.dynamics.friction_per_ps / u.picosecond, dt * u.femtoseconds, seed=4
    )
    devices = [int(x) for x in args.devices.split(",")]
    ctx = backend.make_context(prepared, integ, devices[0])
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
    ctx.setPositions(eq["positions"])
    ctx.setVelocities(eq["velocities"])
    r, d, v = (int(x) for x in args.initial.split(","))
    settings = AlsSettings(
        overlap_threshold=args.threshold,
        window_ps=args.window_ps,
        equilibration_ps=args.equilibration_ps,
        sample_interval_ps=args.sample_interval_ps,
        initial_windows={"restrain": r, "discharge": d, "vanish": v},
        max_windows=getattr(cfg.lambda_schedule.adaptive.max_windows, args.leg),   # the limit of that run
    )
    t0 = time.perf_counter()
    if args.serial:
        schedule, _ = adaptive_schedule(ctx, integ, stages, T, dt, settings, print)
    else:
        spec = LegSpec(name="als", system_xml=mm.XmlSerializer.serialize(alch.system),
                       states=[stage_state(stages[0], 0.0).to_dict(), stage_state(stages[-1], 1.0).to_dict()],
                       n_atoms=alch.system.getNumParticles(), backend=backend_name, temperature_K=T,
                       pressure_bar=cfg.thermodynamics.pressure_bar, barostat="isotropic", timestep_fs=dt, seed=5,
                       mps_thread_percentage=thread_percentage(max(devices.count(g) for g in set(devices))))
        with WindowPool(spec, devices, print) as pool:
            schedule, _ = adaptive_schedule(ctx, integ, stages, T, dt, settings, print, pool=pool)
    wall = time.perf_counter() - t0
    print(f"\nsettings: window {args.window_ps:g} ps, equilibration {args.equilibration_ps:g} ps, sample interval "
          f"{args.sample_interval_ps:g} ps, threshold {args.threshold:g}, initial {args.initial}, "
          f"{'serial' if args.serial else str(len(devices)) + ' workers'}; {wall:.0f} s")
    for name, vals in schedule.windows().items():
        print(f"  {name:10s} {len(vals):2d}: " + " ".join(f"{x:g}" for x in vals))
    for rec in schedule.provenance["iterations"]:
        extra = f" -> section {rec['section']}, new {rec['new_lambda']}" if rec["status"] == "inserted" else ""
        print(f"  iteration {rec['iteration']} {rec['stage']:10s} {rec['status']:10s} overlaps "
              + " ".join(f"{x:.3f}" for x in rec["superdiagonal"]) + extra)
    for note in schedule.notes:
        print("  note:", note)


# the pool starts worker processes with spawn, which import this file again
if __name__ == "__main__":
    main()
