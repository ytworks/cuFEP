"""Plain MD of a stored complex leg (plan docs/plans/2026-11-22-accuracy-causes-and-fixes.md,
step 3): how long the emptied pocket and the bilayer take to stop changing. Reads the run; writes
only to the output directory. Numbers only.

    scripts/in_docker.sh --gpus N python spikes/m22_decoupled_end_md.py end RUN LIGAND --ns 50 -o OUT
            [--window last|INDEX] [--from start|checkpoint]
    scripts/in_docker.sh --gpus N python spikes/m22_decoupled_end_md.py bilayer RUN LIGAND --ns 90 -o OUT

end      The System the leg sampled, built as the run builds it and compared with the conditions its
         samples record (the checksum of the System among them), in the state of one window (the
         last by default: restraint on, charges and Lennard-Jones of the ligand off, REST2 scale 1),
         in a Context made as a worker of the run makes its own. It starts from the configuration
         that window has in start_configs.npz, or (--from checkpoint) from that of the replica that
         was simulated last in the window. Every 10 ps: the box, water oxygens (by atom) and protein
         heavy atoms in the place of the ligand (the prepared pose put on the pocket of the frame,
         and the ligand as it is), the six coordinates of the restraint, the reduced potential at
         every state of the leg, and the positions of the pocket (residues within 0.8 nm of the
         ligand in the prepared pose, the ligand, the C-alpha atoms used for the superposition).
bilayer  The pre-equilibration of the complex continued from its end (equil.npz) with the settings
         of the run (the solute held or free as the run had it), in pieces of 5 ns: the XY area of
         the cell and the water molecules within 0.6 nm of the ligand every 20 ps, as the
         pre-equilibration records them.

No speed or run time is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FRAME_PS = 10.0
SAVE_EVERY = 250          # frames between two writes of the series


def publish_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def fitted(reference: np.ndarray, target: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``points`` moved by the rigid motion that puts ``reference`` on ``target`` (Kabsch)."""
    rc, tc = reference.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((reference - rc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (points - rc) @ rot.T + tc


def load_leg(run: Path, ligand: str):
    """Configuration, prepared system, pre-equilibration, restraint and schedule of the complex leg."""
    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.alchemy.schedule import Schedule
    from cufep.config.schema import Config, read_settings
    from cufep.workflow import load_prepared

    cfg = Config.model_validate(read_settings(run / "resolved_config.yaml"))
    lig_dir = run / ligand
    leg = lig_dir / "repeat_0" / "complex"
    info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    prep = load_prepared(lig_dir / "prep", "complex")
    eq = np.load(leg / "equil.npz", allow_pickle=True)
    restraint = BoreschRestraint.from_dict(json.loads((leg / "restraint.json").read_text())["restraint"])
    return cfg, leg, info, prep, eq, restraint, Schedule.load(leg / "schedule.yaml")


def end(run: Path, ligand: str, ns: float, out: Path, window: str, start_from: str) -> None:
    import openmm as mm
    import openmm.unit as u

    from cufep.alchemy.lrc import nearest_image
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
    spec = leg_spec(cfg, built, states, info["complex_barostat"], timestep, backend, int(cfg.sampling.seed) + 9001)
    _n, stored = stored_conditions(leg / "hrex")
    differing = conditions_differences(stored, spec.conditions())
    if differing:
        raise SystemExit("the System built here is not the one the leg sampled: " + "; ".join(differing))
    print(f"the System and the conditions are those of the samples of {leg} ({backend}, {len(states)} states)",
          flush=True)

    k = len(states) - 1 if window == "last" else int(window)
    rows = read_samples(leg / "hrex")["states"]
    if start_from == "start":
        source = np.load(leg / "start_configs.npz")
        replica = int(np.flatnonzero(rows[0] == k)[0])
        history = "the start of the replica-exchange run"
    else:
        source = np.load(leg / "hrex" / "checkpoint.npz")
        replica = int(np.flatnonzero(rows[-1] == k)[0])                  # simulated last in window k
        visited = rows[:, replica]
        history = (f"the end of the replica-exchange run; this replica was in windows {int(visited.min())} to "
                   f"{int(visited.max())}, in window {k} for {int((visited == k).sum())} of {len(visited)} iterations")
    positions, box = np.asarray(source["positions"][replica]), np.asarray(source["boxes"][replica])
    velocities = np.asarray(source["velocities"][replica])

    # ---- atoms that are followed -------------------------------------------------------------
    atoms = list(prep["topology"].atoms())
    heavy = lambda i: atoms[i].element is not None and atoms[i].element.symbol != "H"      # noqa: E731
    ref = np.asarray(prep["positions"], dtype=float)
    lig_heavy = np.array([i for i in prep["ligand"] if heavy(i)])
    prot_heavy = np.array([i for i in prep["protein"] if heavy(i)])
    ca = np.array([i for i in prep["protein"] if atoms[i].name == "CA"])
    # the C-alpha atoms within 1.2 nm of a ligand heavy atom in the prepared pose: the same set as
    # spikes/m21_stored_run_measurements.py puts the pose on, so that the two count the same region
    near_ca = ca[np.linalg.norm(ref[ca][:, None, :] - ref[lig_heavy][None, :, :], axis=2).min(axis=1) < 1.2]
    d0 = np.linalg.norm(ref[prot_heavy][:, None, :] - ref[lig_heavy][None, :, :], axis=2).min(axis=1)
    pocket_residues = sorted({atoms[i].residue.index for i in prot_heavy[d0 < 0.8]})
    pocket = np.array([i for i in prot_heavy if atoms[i].residue.index in set(pocket_residues)])
    water_o = np.array([a.index for a in atoms if a.residue.name == "HOH" and a.element is not None
                        and a.element.symbol == "O"])
    kept = np.concatenate([pocket, lig_heavy, near_ca])

    out.mkdir(parents=True, exist_ok=True)
    meta = {"run": str(run), "ligand": ligand, "window": k, "state": states[k], "from": start_from, "history": history,
            "replica": replica, "backend": backend, "timestep_fs": timestep, "frame_ps": FRAME_PS, "ns": ns,
            "ligand_heavy": lig_heavy.tolist(), "pocket_atoms": pocket.tolist(), "near_ca": near_ca.tolist(),
            "pocket_atom_names": [f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}" for i in pocket],
            "restraint": restraint.to_dict(), "n_states": len(states)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"window {k} {states[k]}, from {history}; {len(pocket)} pocket atoms of {len(pocket_residues)} residues, "
          f"{len(water_o)} waters", flush=True)

    worker_environment(int(cfg.resources.cpu_threads_per_worker))
    worker = ReplicaWorker(spec, 0, None)
    ctx, state = worker.context, worker.states[k]
    set_state(ctx, state)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(positions)
    ctx.setVelocities(velocities)
    print(f"Context: {worker.describe()}", flush=True)

    steps = md_steps(FRAME_PS, timestep)
    n_frames = int(round(ns * 1000.0 / FRAME_PS))
    series = {key: [] for key in ("box", "u", "boresch", "kept", "fixed_w20", "fixed_w35", "ligand_w20", "ligand_w35",
                                  "fixed_p20", "ligand_p20")}
    ids, offsets = [], [0]                        # water oxygens within 0.35 nm of the fixed region, by frame

    def measure() -> None:
        st = ctx.getState(getPositions=True)
        x = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer))
        b = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
        ghost = fitted(ref[near_ca], x[near_ca], ref[lig_heavy])                 # the prepared pose on this pocket
        centre = ghost.mean(axis=0)
        w = centre + nearest_image(x[water_o] - centre, b)                       # waters in the image of the pocket
        close = np.flatnonzero(np.linalg.norm(w - centre, axis=1) < 2.0)
        lig = centre + nearest_image(x[lig_heavy] - centre, b)

        def nearest(points, sites):
            if not len(points):
                return np.zeros(0)
            return np.linalg.norm(points[:, None, :] - sites[None, :, :], axis=2).min(axis=1)

        dw_fixed, dw_lig = nearest(w[close], ghost), nearest(w[close], lig)
        dp_fixed, dp_lig = nearest(x[pocket], ghost), nearest(x[pocket], lig)
        series["box"].append(b.astype(np.float32))
        series["u"].append(np.asarray(worker.evaluator.reduced_potentials(restore=state)[0], dtype=np.float64))
        series["boresch"].append(np.asarray(restraint.values(x[None])).reshape(-1))
        series["kept"].append(x[kept].astype(np.float32))
        series["fixed_w20"].append(int((dw_fixed < 0.20).sum()))
        series["fixed_w35"].append(int((dw_fixed < 0.35).sum()))
        series["ligand_w20"].append(int((dw_lig < 0.20).sum()))
        series["ligand_w35"].append(int((dw_lig < 0.35).sum()))
        series["fixed_p20"].append(int((dp_fixed < 0.20).sum()))
        series["ligand_p20"].append(int((dp_lig < 0.20).sum()))
        inside = water_o[close[dw_fixed < 0.35]]
        ids.extend(int(i) for i in inside)
        offsets.append(len(ids))
        if any(b[i, i] < 3.0 * float(cfg.dynamics.cutoff_nm) for i in range(3)) and backend != "openmm":
            raise SystemExit(f"a height of the cell fell below three times the cutoff ({np.diag(b)}): the plugin "
                             "cannot go on; run this window on stock OpenMM")

    def save() -> None:
        publish_npz(out / "series.npz", water_ids=np.array(ids, dtype=np.int64), water_offsets=np.array(offsets),
                    **{key: np.asarray(value) for key, value in series.items()})

    measure()                                     # frame 0: the start as it was stored
    for frame in range(1, n_frames + 1):
        worker.integrator.step(steps)
        measure()
        if frame % SAVE_EVERY == 0 or frame == n_frames:
            save()
            w, p = np.array(series["fixed_w35"][-SAVE_EVERY:]), np.array(series["fixed_p20"][-SAVE_EVERY:])
            b = np.array(series["box"][-SAVE_EVERY:])
            area = np.linalg.norm(np.cross(b[:, 0], b[:, 1]), axis=1)
            print(f"{frame * FRAME_PS / 1000.0:7.2f} ns: water within 3.5 A of the place of the ligand "
                  f"{w.mean():5.2f}, protein atoms within 2.0 A {p.mean():5.2f}, XY area {area.mean():7.3f} nm2",
                  flush=True)
    print("done", flush=True)


def bilayer(run: Path, ligand: str, ns: float, out: Path) -> None:
    import openmm as mm

    from cufep.sampling.equilibrate import unrestrained_md
    from cufep.workflow import leg_seed, plain_md_settings, pre_equilibration_parts

    cfg, _leg, info, prep, eq, _restraint, _schedule = load_leg(run, ligand)
    barostat = info["complex_barostat"]
    if barostat != "membrane":
        raise SystemExit("not a bilayer system")
    _production, held = pre_equilibration_parts(cfg, "complex", True)
    atoms = list(prep["topology"].atoms())
    restrained = [i for i in prep["solute"] if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    waters = [a.index for a in atoms if a.residue.name == "HOH" and a.element is not None and a.element.symbol == "O"]
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    seed = leg_seed(cfg, 0, "complex")
    timestep = float(eq["timestep_fs"])
    start = (np.asarray(eq["positions"]), np.asarray(eq["velocities"]), np.asarray(eq["box"]))
    before = float(eq["simulated_ns"])
    out.mkdir(parents=True, exist_ok=True)
    area = [np.atleast_1d(np.asarray(eq["area_nm2"], dtype=float))]
    water = [np.atleast_1d(np.asarray(eq["pocket_waters"], dtype=float))]
    time_ns = [(np.arange(len(area[0])) + 1) * before / max(1, len(area[0]))]
    print(f"{ligand}: the pre-equilibration of the run ended after {before:g} ns (solute "
          f"{'held' if held else 'free'}); continuing for {ns:g} ns", flush=True)
    piece, done = 5.0, 0.0
    while done < ns - 1e-9:
        length = min(piece, ns - done)
        settings = plain_md_settings(cfg, timestep, barostat, seed, production_ns=length, hold_solute=held,
                                     restrained_atoms=restrained if held else [], ligand_atoms=prep["ligand"],
                                     water_oxygens=waters, bilayer_report=False)
        n_piece = int(round(done / piece)) + 1
        result = unrestrained_md(system, start, prep["solute"], settings, timestep, 0, lambda m: print(m, flush=True),
                                 what="continued pre-equilibration", seed=seed + 300 + 2 * n_piece,
                                 barostat_seed=seed + 301 + 2 * n_piece,
                                 held_at=np.asarray(prep["positions"]) if held else None)
        start = (result.positions, result.velocities, result.box)
        area.append(np.asarray(result.area_nm2, dtype=float))
        water.append(np.asarray(result.pocket_waters, dtype=float))
        n_new = len(result.area_nm2)
        time_ns.append(before + done + (np.arange(n_new) + 1) * result.simulated_ns / n_new)
        done += result.simulated_ns
        publish_npz(out / "series.npz", time_ns=np.concatenate(time_ns), area_nm2=np.concatenate(area),
                    pocket_waters=np.concatenate(water), positions=result.positions, velocities=result.velocities,
                    box=result.box, backend=str(result.backend), notes=json.dumps(result.notes))
        print(f"{before + done:7.2f} ns: XY area {np.mean(result.area_nm2):7.3f} nm2, water within 6 A of the "
              f"ligand {np.mean(result.pocket_waters):5.2f} (means of the last {result.simulated_ns:g} ns; "
              f"{result.backend})", flush=True)
    print("done", flush=True)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 3 or args[0] not in ("end", "bilayer"):
        raise SystemExit(__doc__)
    mode, run, ligand = args[0], Path(args[1]).resolve(), args[2]
    ns, out, window, start_from = 50.0, None, "last", "start"
    rest = args[3:]
    while rest:
        a = rest.pop(0)
        if a == "--ns":
            ns = float(rest.pop(0))
        elif a == "-o":
            out = Path(rest.pop(0)).resolve()
        elif a == "--window":
            window = rest.pop(0)
        elif a == "--from":
            start_from = rest.pop(0)
        else:
            raise SystemExit(f"unknown argument {a}")
    if out is None or start_from not in ("start", "checkpoint"):
        raise SystemExit(__doc__)
    if mode == "end":
        end(run, ligand, ns, out, window, start_from)
    else:
        bilayer(run, ligand, ns, out)


if __name__ == "__main__":
    main()
