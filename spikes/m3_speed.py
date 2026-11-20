"""Speed measurements on a finished leg (plan step 3). Results are for internal use only.

The leg is rebuilt from the files of an earlier run (prepared system, equilibrated box,
restraint, schedule, start configurations) and driven through the real replica-exchange
sampler.

    # iteration time with a given device list; a GPU listed n times gets n workers
    spikes/run.sh spikes/m3_speed.py iter runs/tyk2_ejm_31 ejm_31 0 complex --devices 1,1,2,2

    # where the time of one replica-iteration goes (one worker, in-process)
    spikes/run.sh spikes/m3_speed.py breakdown runs/tyk2_ejm_31 ejm_31 0 complex --device 1

Workers on the same GPU run concurrently only under NVIDIA MPS (see spikes/m3_speed_mps.sh).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


def build(run_dir: Path, ligand: str, repeat: int, leg: str, backend: str | None, n_windows: int | None):
    """LegSpec and start configurations of a finished leg."""
    import openmm as mm

    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.lrc import TailEnergy
    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.config.schema import Config
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import load_prepared

    cfg = Config.model_validate(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    lig_dir = run_dir / ligand
    leg_dir = lig_dir / f"repeat_{repeat}" / leg
    prep_info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    prep = load_prepared(lig_dir / "prep", leg)
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    eq = np.load(leg_dir / "equil.npz")
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
    alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha)
    if leg == "complex":
        r = BoreschRestraint.from_dict(json.loads((leg_dir / "restraint.json").read_text())["restraint"])
        alch.system.addForce(r.create_force())
    tail = TailEnergy.from_system(system, prep["ligand"], alch.ligand_parameters, cfg.dynamics.cutoff_nm,
                                  cfg.analysis.lrc_cutoff_nm).to_dict()
    sched = yaml.safe_load((leg_dir / "schedule.yaml").read_text())
    states = sched["states"]
    start = np.load(leg_dir / "start_configs.npz")
    configs = [start["positions"], start["boxes"], start["velocities"]]
    if n_windows is not None and n_windows < len(states):
        pick = np.unique(np.round(np.linspace(0, len(states) - 1, n_windows)).astype(int))
        states = [states[i] for i in pick]
        configs = [c[pick] for c in configs]
    timestep_fs = float(eq["timestep_fs"])
    frame_atoms = prep["solute"] if leg == "complex" else prep["ligand"]
    spec = LegSpec(
        name=f"{ligand}/r{repeat}/{leg}", system_xml=mm.XmlSerializer.serialize(alch.system), states=states,
        n_atoms=alch.system.getNumParticles(), backend=backend or sched.get("backend", cfg.backend.name),
        precision="mixed", platform="CUDA", temperature_K=cfg.thermodynamics.temperature_K,
        pressure_bar=cfg.thermodynamics.pressure_bar,
        barostat=prep_info["complex_barostat"] if leg == "complex" else "isotropic",
        friction_per_ps=cfg.dynamics.friction_per_ps, timestep_fs=timestep_fs,
        steps_per_iteration=int(round(2500.0 / timestep_fs)), seed=1, cpu_threads=cfg.resources.cpu_threads_per_worker,
        lrc=tail, frame_atoms=list(frame_atoms))
    return spec, configs


def cmd_iter(args) -> dict:
    from cufep.sampling.hrex import ReplicaExchangeSampler

    spec, configs = build(Path(args.run_dir), args.ligand, args.repeat, args.leg, args.backend, args.windows)
    devices = [int(x) for x in args.devices.split(",")]
    with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
        with ReplicaExchangeSampler(spec, Path(tmp) / "hrex", devices=devices, log=lambda m: None) as s:
            s.initialize(*configs)
            s.run(args.warmup, store=False)
            t0 = time.perf_counter()
            stats = s.run(args.iterations)
            wall = time.perf_counter() - t0
            restarts = s.restarts
    per_iter = wall / args.iterations
    n = spec.n_states
    out = {"what": "iter", "leg": spec.name, "backend": spec.backend, "atoms": spec.n_atoms, "windows": n,
           "devices": devices, "gpus": len(set(devices)), "workers": len(devices), "iterations": args.iterations,
           "seconds_per_iteration": per_iter, "replica_ms": 1000.0 * per_iter * len(set(devices)) / n,
           "aggregate_ns_per_day": n * 2.5e-3 / per_iter * 86400.0,
           "median_iteration_s": float(np.median([x.seconds for x in stats])), "restarts": restarts, "mps": args.mps,
           "slowest_worker_s": float(np.mean([x.slowest_worker for x in stats])),
           "controller_s": float(np.mean([x.seconds - x.slowest_worker for x in stats])),
           "worker_busy_fraction": float(np.mean([x.all_workers / (len(devices) * x.seconds) for x in stats]))}
    return out


def cmd_breakdown(args) -> dict:
    import openmm as mm
    import openmm.unit as u

    from cufep.sampling.states import set_state
    from cufep.sampling.worker import ReplicaWorker, SharedArrays

    spec, configs = build(Path(args.run_dir), args.ligand, args.repeat, args.leg, args.backend, args.windows)
    k = spec.n_states
    arrays = SharedArrays(k, spec.n_atoms, k, fallback_dir=args.scratch)
    arrays.pos[0], arrays.box[0], arrays.vel[0] = configs[0], configs[1], configs[2]
    w = ReplicaWorker(spec, args.device, arrays, 0)
    ctx, integ, ev = w.context, w.integrator, w.evaluator
    states = w.states
    T = {x: 0.0 for x in ("load", "set_state", "md", "read_back", "sync_all_groups", "main_group", "softcore_group",
                          "restraint_volume", "write")}
    reps = list(range(k)) * args.rounds
    # warm-up
    for r in range(min(k, 4)):
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in arrays.box[0, r]])
        ctx.setPositions(arrays.pos[0, r])
        ctx.setVelocities(arrays.vel[0, r])
        set_state(ctx, states[r])
        integ.step(100)

    def tick(key, t0):
        ctx.getState()                                   # make sure queued GPU work is finished
        t1 = time.perf_counter()
        T[key] += t1 - t0
        return t1

    for r in reps:
        t = time.perf_counter()
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in arrays.box[0, r]])
        ctx.setPositions(arrays.pos[0, r])
        ctx.setVelocities(arrays.vel[0, r])
        t = tick("load", t)
        set_state(ctx, states[r])
        t = tick("set_state", t)
        integ.step(spec.steps_per_iteration)
        t = tick("md", t)
        st = ctx.getState(getPositions=True, getVelocities=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        vel = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
        box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        t = tick("read_back", t)
        if ev.has_sterics:
            ctx.getState(getEnergy=True)
        t = tick("sync_all_groups", t)
        ev.main_group()
        t = tick("main_group", t)
        ev.softcore_group()
        t = tick("softcore_group", t)
        ev.restraint_energy()
        ev.volume_nm3()
        t = tick("restraint_volume", t)
        arrays.pos[1, r], arrays.vel[1, r], arrays.box[1, r] = pos, vel, box
        t = tick("write", t)
    n = len(reps)
    ms = {key: 1000.0 * v / n for key, v in T.items()}
    total = sum(ms.values())
    out = {"what": "breakdown", "leg": spec.name, "backend": spec.backend, "atoms": spec.n_atoms, "windows": k,
           "distinct_lambda_sterics": int(len(ev._ls_unique)), "ms_per_replica_iteration": ms, "total_ms": total,  # noqa: SLF001
           "md_ns_per_day": 2.5e-3 / (ms["md"] / 1000.0) * 86400.0, "md_fraction": ms["md"] / total}
    w.close()
    arrays.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("iter", "breakdown"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        p.add_argument("ligand")
        p.add_argument("repeat", type=int)
        p.add_argument("leg")
        p.add_argument("--backend", default=None)
        p.add_argument("--windows", type=int, default=None, help="use only this many of the windows")
        p.add_argument("--scratch", default="/dev/shm")
        p.add_argument("--out", default="spikes/out/m3_speed/results.jsonl")
    it = sub.choices["iter"]
    it.add_argument("--devices", required=True)
    it.add_argument("--iterations", type=int, default=30)
    it.add_argument("--warmup", type=int, default=4)
    it.add_argument("--mps", action="store_true")
    bd = sub.choices["breakdown"]
    bd.add_argument("--device", type=int, default=0)
    bd.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    out = cmd_iter(args) if args.cmd == "iter" else cmd_breakdown(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
