"""M0 spike 6: several worker processes, one GPU each, inside one container.

Each worker (multiprocessing 'spawn') builds its own plugin Context from the serialized
TYK2 alchemical System written by spike 2 and runs MD plus energy evaluations at the
same time as the others. Checks that every worker starts, lands on its own GPU and
finishes, next to whatever else is on the machine (e.g. a foreign MPS daemon).

Usage: spikes/run.sh spikes/m0_03_multigpu.py [n_workers]
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("spikes/out")
TAG = "tyk2_ejm_31_cube"


def worker(device: int, backend_name: str, nsteps: int, conn) -> None:
    import os
    os.environ.setdefault("OPENMM_CPU_THREADS", "2")
    import openmm as mm
    import openmm.unit as u

    from cufep.alchemy.factory import LAMBDA_ELECTROSTATICS, LAMBDA_STERICS, set_lambdas
    from cufep.backends import get_backend

    try:
        system = mm.XmlSerializer.deserialize((OUT / f"{TAG}_alch_system.xml").read_text())
        data = np.load(OUT / f"{TAG}_equil.npz")
        backend = get_backend(backend_name)
        prepared = backend.prepare_system(system)
        integ = backend.make_integrator(298.15 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=100 + device)
        ctx = backend.make_context(prepared, integ, device)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in data["box"]])
        ctx.setPositions(data["positions"])
        ctx.setVelocities(data["velocities"])
        name = ctx.getPlatform().getPropertyValue(ctx, "DeviceName")
        index = ctx.getPlatform().getPropertyValue(ctx, "DeviceIndex")
        conn.send(("ready", device, f"{name} index={index}"))
        conn.recv()                                   # wait for the common start signal
        t0 = time.perf_counter()
        energies = []
        for it in range(8):
            integ.step(nsteps // 8)
            set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: 1.0 - it / 8.0, LAMBDA_STERICS: 1.0})
            energies.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        conn.send(("done", device, time.perf_counter() - t0, bool(np.all(np.isfinite(energies)))))
    except Exception as exc:  # noqa: BLE001
        conn.send(("error", device, f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run(n: int, backend_name: str, nsteps: int = 20000) -> bool:
    ctx = mp.get_context("spawn")
    procs, pipes = [], []
    for d in range(n):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(d, backend_name, nsteps, child), daemon=True)
        p.start()
        procs.append(p)
        pipes.append(parent)
    ok = True
    for pipe in pipes:
        if not pipe.poll(300):
            print("  a worker did not start within 300 s")
            return False
        msg = pipe.recv()
        if msg[0] != "ready":
            print("  worker failed:", msg)
            ok = False
        else:
            print(f"  worker {msg[1]} ready on {msg[2]}")
    if not ok:
        return False
    t0 = time.perf_counter()
    for pipe in pipes:
        pipe.send("go")
    times = []
    for pipe in pipes:
        if not pipe.poll(600):
            print("  a worker hung (no result within 600 s)")
            return False
        msg = pipe.recv()
        if msg[0] != "done" or not msg[3]:
            print("  worker failed:", msg)
            ok = False
        else:
            times.append(msg[2])
    wall = time.perf_counter() - t0
    for p in procs:
        p.join(30)
    if ok:
        # concurrency check only: wall time close to the slowest single worker
        print(f"  {backend_name}: {n} workers finished; wall / slowest worker = {wall / max(times):.2f} "
              f"(1.0 means fully concurrent)")
    return ok


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if not (OUT / f"{TAG}_alch_system.xml").exists():
        print("run spike 2 first (it writes the serialized System)")
        return 2
    ok = True
    for backend_name in ("cupotential", "openmm"):
        print(f"[6] {n} worker processes, backend {backend_name}")
        ok &= run(n, backend_name)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
