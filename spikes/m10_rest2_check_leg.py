"""On a finished REST2 leg: are the stored-style state energies those of the Context set to each state?

Rebuilds the leg's alchemical System as the workflow does, loads the start configuration of a
window, runs a little MD in that window and compares the evaluator's reduced energies at all
windows with the energy after switching the Context to each window.

    spikes/run.sh spikes/m10_rest2_check_leg.py runs/benchmark_tyk2_subset_rest2 jmc_23 0 complex --device 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ligand")
    ap.add_argument("repeat", type=int)
    ap.add_argument("leg")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    import openmm as mm
    import openmm.unit as u

    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.alchemy.schedule import Schedule
    from cufep.backends import get_backend
    from cufep.config.schema import Config
    from cufep.sampling.states import R_GAS, StateEnergyEvaluator, set_state
    from cufep.workflow import load_prepared, rest2_region

    run_dir = Path(args.run_dir)
    cfg = Config.model_validate(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    lig_dir = run_dir / args.ligand
    leg_dir = lig_dir / f"repeat_{args.repeat}" / args.leg
    prep = load_prepared(lig_dir / "prep", args.leg)
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    eq = np.load(leg_dir / "equil.npz")
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
    rest2 = cfg.sampling.rest2
    hot = (
        rest2_region(rest2, prep["topology"], prep["positions"], prep["ligand"], prep["protein"])[0]
        if args.leg == "complex"
        else []
    )
    alch = make_alchemical_system(
        system,
        prep["ligand"],
        cfg.alchemy.softcore_alpha,
        rest2=rest2.scale > 1.0,
        rest2_atoms=hot,
        bonds=[(b[0].index, b[1].index) for b in prep["topology"].bonds()],
    )
    if args.leg == "complex":
        alch.system.addForce(
            BoreschRestraint.from_dict(json.loads((leg_dir / "restraint.json").read_text())["restraint"]).create_force()
        )
    schedule = Schedule.load(leg_dir / "schedule.yaml")
    meta = yaml.safe_load((leg_dir / "schedule.yaml").read_text())
    backend_name = args.backend or meta.get("backend", "cupotential")
    T, dt = cfg.thermodynamics.temperature_K, float(eq["timestep_fs"])
    backend = get_backend(backend_name, precision="mixed")
    prepared = backend.prepare_system(alch.system)
    integ = backend.make_integrator(
        T * u.kelvin, cfg.dynamics.friction_per_ps / u.picosecond, dt * u.femtoseconds, seed=4
    )
    ctx = backend.make_context(prepared, integ, args.device)
    start = np.load(leg_dir / "start_configs.npz")
    k = len(schedule.states) // 3 if args.window is None else args.window
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in start["boxes"][k]])
    ctx.setPositions(start["positions"][k])
    ctx.setVelocities(start["velocities"][k])
    set_state(ctx, schedule.states[k])
    integ.step(500)
    kt = R_GAS * T
    red = StateEnergyEvaluator(ctx, schedule.states, T, None).reduced_potentials(restore=schedule.states[k])[0]
    direct = []
    for state in schedule.states:
        set_state(ctx, state)
        direct.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / kt)
    direct = np.array(direct)
    err = (red - red[k]) - (direct - direct[k])
    print(
        f"{args.ligand} {args.leg}, backend {backend_name}, {len(schedule.states)} windows, configuration of window "
        f"{k} (REST2 scale {schedule.states[k].rest2_scale:.2f})"
    )
    print("  REST2 scale of the windows:   " + " ".join(f"{s.rest2_scale:7.2f}" for s in schedule.states))
    print("  u(window) - u(own), direct:   " + " ".join(f"{v:7.1f}" for v in direct - direct[k]))
    print("  evaluator - direct (kT):      " + " ".join(f"{v:7.3f}" for v in err))
    print(f"  largest difference {np.max(np.abs(err)):.3f} kT")


if __name__ == "__main__":
    main()
