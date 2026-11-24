"""Two checks of the water relocation moves on a stored complex leg, in the states and with the
configurations of its own run. Reads the run; writes nothing. Numbers only.

    scripts/in_docker.sh --gpus N python spikes/m29_relocation_checks_on_a_leg.py RUN LIGAND [--windows 0,3,10,34]
            [--moves 10] [--seed 1]

For every window that is named, from the configuration its replica had at the end of the run:
  reverse   the proposal ratio of a move and of the move back (the same pool of poses, the chosen
            one replaced by the pose the water had): their sum is zero if every term of the ratio is
            the one the reverse move computes;
  backends  the change of the potential energy of the simulation by the proposed move, in the state
            of the window (its lambda values and REST2 scale), from the backend the leg was sampled
            with and from stock OpenMM on the same System: the acceptance uses this number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m22_decoupled_end_md import load_leg  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    run, ligand, rest = Path(args[0]).resolve(), args[1], args[2:]
    windows, n_moves, seed = None, 10, 1
    while rest:
        a = rest.pop(0)
        if a == "--windows":
            windows = [int(w) for w in rest.pop(0).split(",")]
        elif a == "--moves":
            n_moves = int(rest.pop(0))
        elif a == "--seed":
            seed = int(rest.pop(0))
        else:
            raise SystemExit(f"unknown argument {a}")

    import openmm as mm
    import openmm.unit as u

    from cufep.sampling.relocation_mc import RelocationMover
    from cufep.sampling.states import set_state
    from cufep.sampling.storage import read_samples, stored_backend
    from cufep.sampling.worker import ReplicaWorker, worker_environment
    from cufep.workflow import alchemical_leg, leg_spec

    cfg, leg, info, prep, eq, restraint, schedule = load_leg(run, ligand)
    cfg.sampling.relocation_mc.enabled = True
    built = alchemical_leg(cfg, mm.XmlSerializer.deserialize(prep["system_xml"]), prep, eq["box"], restraint, True)
    states = [s.to_dict() for s in schedule.states]
    sampled_with = stored_backend(leg / "hrex")
    worker_environment(int(cfg.resources.cpu_threads_per_worker))
    workers = {name: ReplicaWorker(leg_spec(cfg, built, states, info["complex_barostat"], float(eq["timestep_fs"]),
                                            name, int(cfg.sampling.seed) + 9301), 0, None)
               for name in dict.fromkeys((sampled_with, "openmm"))}
    for name, worker in workers.items():
        print(f"{name}: {worker.describe()}", flush=True)
    end = np.load(leg / "hrex" / "checkpoint.npz")
    last_states = read_samples(leg / "hrex")["states"][-1]
    kt = 8.31446261815324e-3 * float(cfg.thermodynamics.temperature_K)
    windows = windows if windows is not None else [0, len(states) // 2, len(states) - 1]

    def potential(worker, state, positions, box):
        context = worker.context
        set_state(context, state)
        context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
        context.setPositions(positions)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

    main_worker = workers[sampled_with]
    for k in windows:
        replica = int(np.flatnonzero(last_states == k)[0])
        x, box = np.asarray(end["positions"][replica], dtype=float), np.asarray(end["boxes"][replica], dtype=float)
        state = main_worker.states[k]
        mover = RelocationMover(built.relocation, main_worker.context, seed + k)
        print(f"window {k} {states[k]}: {int(mover.inside(x, box).sum())} waters in the sphere "
              f"(radius {built.relocation.radius_nm:.3f} nm), trial energies on {mover.engine}", flush=True)
        worst_reverse, differences = 0.0, []
        for i in range(n_moves):
            out = bool(i % 2)
            move = mover.proposal(x, box, state, out)
            if move is None:
                continue
            atoms = mover.waters[move["moved"]]
            poses, anchor_inside = move["poses"].copy(), move["anchor_inside"].copy()
            poses[move["chosen"]] = x[atoms]
            anchor_inside[move["chosen"]] = out
            back = mover.proposal(move["y"], box, state, not out, moved=move["moved"], poses=poses,
                                  anchor_inside=anchor_inside, chosen=move["chosen"])
            worst_reverse = max(worst_reverse, abs(move["log_q_ratio"] + back["log_q_ratio"]))
            change = {name: potential(w, w.states[k], move["y"], box) - potential(w, w.states[k], x, box)
                      for name, w in workers.items()}
            values = list(change.values())
            differences.append((values[0] - values[-1]) / kt)
            ratios = f"log_q_ratio {move['log_q_ratio']:+10.3f}, back {back['log_q_ratio']:+10.3f}"
            print(f"   {'out' if out else 'in '}: {ratios}; "
                  + "; ".join(f"dU {name} {v / kt:+12.3f} kT" for name, v in change.items()), flush=True)
        d = np.array(differences)
        print(f"   largest |forward + back| {worst_reverse:.2e}; dU of {sampled_with} minus stock OpenMM (kT): "
              f"mean {d.mean():+.4f}, largest {np.abs(d).max():.4f}", flush=True)


if __name__ == "__main__":
    main()
