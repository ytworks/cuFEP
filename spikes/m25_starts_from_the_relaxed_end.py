"""Start configurations for the control of hypothesis E3 (plan
docs/plans/2026-11-22-accuracy-causes-and-fixes.md): the lambda windows on the decoupled side of a
stored complex leg started from the relaxed decoupled end. Reads the run; writes only to the
output directory. Numbers only.

    scripts/in_docker.sh --gpus N python spikes/m25_starts_from_the_relaxed_end.py RUN LIGAND -o OUT
            [--ns 50] [--sterics-at-most 0.25]

The System the leg sampled, built and compared with the conditions of its samples as
spikes/m22_decoupled_end_md.py does. --ns of MD in the last window (the ligand decoupled, the
restraint on) from the start configuration that window has; then back through the windows whose
charges are off and whose lambda_sterics is at most --sterics-at-most, from the last towards the
coupled end, lambda_schedule.adaptive.walk_ps of MD in each: the walk of the precalculation, in
the other direction. OUT/start_configs.npz holds the start configurations of the leg with those
of these windows replaced; the other windows keep theirs. OUT/meta.json says which windows were
replaced and gives the potential energy along the way.

No speed or run time is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m22_decoupled_end_md import load_leg, publish_npz  # noqa: E402

EPS = 1e-9                # lambda values are stored as 1 - x: 0.2 is 0.19999999999999996
REPORT_NS = 1.0           # MD between two lines of the log


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    run, ligand, rest = Path(args[0]).resolve(), args[1], args[2:]
    ns, at_most, out = 50.0, 0.25, None
    while rest:
        a = rest.pop(0)
        if a == "--ns":
            ns = float(rest.pop(0))
        elif a == "--sterics-at-most":
            at_most = float(rest.pop(0))
        elif a == "-o":
            out = Path(rest.pop(0)).resolve()
        else:
            raise SystemExit(f"unknown argument {a}")
    if out is None:
        raise SystemExit(__doc__)

    import openmm as mm
    import openmm.unit as u

    from cufep.sampling.states import set_state
    from cufep.sampling.storage import (
        conditions_differences,
        read_samples,
        stored_backend,
        stored_conditions,
    )
    from cufep.sampling.worker import ReplicaWorker, worker_environment
    from cufep.steps import md_steps
    from cufep.workflow import alchemical_leg, leg_spec

    cfg, leg, info, prep, eq, restraint, schedule = load_leg(run, ligand)
    built = alchemical_leg(cfg, mm.XmlSerializer.deserialize(prep["system_xml"]), prep, eq["box"], restraint, True)
    states = [s.to_dict() for s in schedule.states]
    backend = stored_backend(leg / "hrex")
    timestep = float(eq["timestep_fs"])
    spec = leg_spec(cfg, built, states, info["complex_barostat"], timestep, backend, int(cfg.sampling.seed) + 9101)
    differing = conditions_differences(stored_conditions(leg / "hrex")[1], spec.conditions())
    if differing:
        raise SystemExit("the System built here is not the one the leg sampled: " + "; ".join(differing))
    k = len(states) - 1
    if not np.array_equal(read_samples(leg / "hrex")["states"][0], np.arange(k + 1)):
        raise SystemExit("the replicas of this leg did not start each in the window of its own number")
    replaced = [i for i, s in enumerate(states)
                if s["lambda_electrostatics"] <= EPS and s["lambda_sterics"] <= at_most + EPS]
    if not replaced or replaced != list(range(replaced[0], k + 1)):
        raise SystemExit(f"the windows with lambda_sterics of at most {at_most:g} are not the last ones: {replaced}")
    print(f"the System and the conditions are those of the samples of {leg} ({backend}, {k + 1} states); windows "
          f"{replaced[0]} to {k} (lambda_sterics {states[replaced[0]]['lambda_sterics']:g} to 0) are started anew",
          flush=True)

    start = np.load(leg / "start_configs.npz")
    arrays = {key: np.array(start[key]) for key in ("positions", "boxes", "velocities")}

    worker_environment(int(cfg.resources.cpu_threads_per_worker))
    worker = ReplicaWorker(spec, 0, None)
    ctx = worker.context
    set_state(ctx, worker.states[k])
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in arrays["boxes"][k]])
    ctx.setPositions(arrays["positions"][k])
    ctx.setVelocities(arrays["velocities"][k])
    print(f"Context: {worker.describe()}", flush=True)

    def take(i: int) -> float:
        st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
        energy = float(st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        if not np.isfinite(energy):
            raise SystemExit(f"the potential energy in window {i} is not finite")
        arrays["positions"][i] = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        arrays["boxes"][i] = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        arrays["velocities"][i] = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
        return energy

    pieces = int(round(ns / REPORT_NS))
    for piece in range(1, pieces + 1):
        worker.integrator.step(md_steps(REPORT_NS * 1000.0, timestep))
        energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        print(f"{piece * REPORT_NS:6.1f} ns in window {k}: potential energy {energy:.0f} kJ/mol", flush=True)
    walk_ps = float(cfg.lambda_schedule.adaptive.walk_ps)
    energies = {k: take(k)}
    for i in reversed(replaced[:-1]):
        set_state(ctx, worker.states[i])
        worker.integrator.step(md_steps(walk_ps, timestep))
        energies[i] = take(i)
        print(f"window {i} (lambda_sterics {states[i]['lambda_sterics']:g}): {walk_ps:g} ps, potential energy "
              f"{energies[i]:.0f} kJ/mol", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    made_with = {"start_configurations": f"windows {replaced[0]} to {k} from the decoupled end after {ns:g} ns of MD "
                                         f"in window {k}, {walk_ps:g} ps in each on the way back "
                                         "(spikes/m25_starts_from_the_relaxed_end.py); the others as the leg had them",
                 "walk_ps": walk_ps}
    publish_npz(out / "start_configs.npz", made_with=json.dumps(made_with), **arrays)
    meta = {"run": str(run), "ligand": ligand, "leg": str(leg), "backend": backend, "timestep_fs": timestep,
            "ns_in_the_last_window": ns, "walk_ps": walk_ps, "sterics_at_most": at_most, "replaced": replaced,
            "states": [states[i] for i in replaced],
            "potential_energy_kJ_per_mol": {str(i): energies[i] for i in replaced}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
