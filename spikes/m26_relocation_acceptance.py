"""Acceptance and cost of the water relocation moves of the package (cufep.sampling.relocation_mc) on
a stored complex leg, for several settings of the trial poses. Reads the run; writes only under
spikes/out/m26/. Numbers only.

    scripts/in_docker.sh --gpus N python spikes/m26_relocation_acceptance.py RUN LIGAND [--window last|INDEX]
            [--moves 200] [--settings 16384:none,4096:0.5,2048:0.5] [--seed 1]

The System the leg sampled, built as the run builds it and compared with the conditions of its
samples; a Context made as a worker of the run makes its own, in the state of one window, with the
configuration that window started from. For each setting (trial poses : share of the anchors drawn
in the sphere, none = uniform in the whole cell as in the paper) ``--moves`` moves are attempted,
each from that same configuration (an accepted move is undone), so that the numbers are those of
independent attempts: the share accepted in each direction, and the processor time of the proposal
(the stand-in energies, NumPy and SciPy on one core; the two evaluations of the potential energy
of the simulation are not timed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m22_decoupled_end_md import load_leg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes/out/m26"


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    run, ligand, rest = Path(args[0]).resolve(), args[1], args[2:]
    window, n_moves, settings, seed = "last", 200, "16384:none,4096:0.5,2048:0.5", 1
    while rest:
        a = rest.pop(0)
        if a == "--window":
            window = rest.pop(0)
        elif a == "--moves":
            n_moves = int(rest.pop(0))
        elif a == "--settings":
            settings = rest.pop(0)
        elif a == "--seed":
            seed = int(rest.pop(0))
        else:
            raise SystemExit(f"unknown argument {a}")

    import openmm as mm
    import openmm.unit as u

    from cufep.sampling.relocation_mc import RelocationMover
    from cufep.sampling.states import set_state
    from cufep.sampling.storage import conditions_differences, read_samples, stored_backend, stored_conditions
    from cufep.sampling.worker import ReplicaWorker, worker_environment
    from cufep.workflow import alchemical_leg, leg_spec

    cfg, leg, info, prep, eq, restraint, schedule = load_leg(run, ligand)
    built = alchemical_leg(cfg, mm.XmlSerializer.deserialize(prep["system_xml"]), prep, eq["box"], restraint, True)
    states = [s.to_dict() for s in schedule.states]
    spec = leg_spec(cfg, built, states, info["complex_barostat"], float(eq["timestep_fs"]),
                    stored_backend(leg / "hrex"), int(cfg.sampling.seed) + 9301)
    differing = conditions_differences(stored_conditions(leg / "hrex")[1], spec.conditions())
    if differing:
        raise SystemExit("the System built here is not the one the leg sampled: " + "; ".join(differing))
    k = len(states) - 1 if window == "last" else int(window)
    replica = int(np.flatnonzero(read_samples(leg / "hrex")["states"][0] == k)[0])
    start = np.load(leg / "start_configs.npz")
    positions, box = np.asarray(start["positions"][replica]), np.asarray(start["boxes"][replica])
    velocities = np.asarray(start["velocities"][replica])

    worker_environment(int(cfg.resources.cpu_threads_per_worker))
    worker = ReplicaWorker(spec, 0, None)
    context, state = worker.context, worker.states[k]
    set_state(context, state)
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    context.setPositions(positions)
    context.setVelocities(velocities)
    x0 = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer))
    print(f"{ligand}, window {k} {states[k]}: {worker.describe()}", flush=True)

    result = {"run": str(run), "ligand": ligand, "window": k, "state": states[k], "moves": n_moves, "settings": {}}
    for item in settings.split(","):
        poses, fraction = item.split(":")
        moving = cfg.model_copy(deep=True)
        moving.sampling.relocation_mc.enabled = True
        moving.sampling.relocation_mc.trial_poses = int(poses)
        moving.sampling.relocation_mc.inner_fraction = "uniform" if fraction == "none" else float(fraction)
        relocation = alchemical_leg(moving, mm.XmlSerializer.deserialize(prep["system_xml"]), prep, eq["box"],
                                    restraint, True).relocation
        mover = RelocationMover(relocation, context, seed)
        inside = int(mover.inside(x0, box).sum())
        counts = {"in": [0, 0], "out": [0, 0]}
        seconds = []
        for _ in range(n_moves):
            # the proposal alone, timed on the processor; then the move as the package makes it
            t0 = time.process_time()
            mover.proposal(x0, box, state, bool(mover.rng.random() < 0.5))
            seconds.append(time.process_time() - t0)
            move = mover.attempt(state)
            counts[move["direction"]][0] += 1
            counts[move["direction"]][1] += int(move["accepted"])
            if move["accepted"]:
                context.setPositions(x0)
                context.setVelocities(velocities)
        row = {"waters_in_the_sphere": inside, "radius_nm": relocation.radius_nm,
               "attempted_in": counts["in"][0], "accepted_in": counts["in"][1],
               "attempted_out": counts["out"][0], "accepted_out": counts["out"][1],
               "proposal_cpu_seconds_median": float(np.median(seconds))}
        result["settings"][item] = row
        print(f"  trial poses {poses}, share in the sphere {fraction}: sphere of {relocation.radius_nm:.3f} nm with "
              f"{inside} waters; into the sphere {counts['in'][1]} of {counts['in'][0]} accepted, out of it "
              f"{counts['out'][1]} of {counts['out'][0]}; proposal {np.median(seconds):.2f} s of processor time "
              "(median)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ligand}_window{k}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
