"""M0 spike 7: semi-isotropic pressure coupling for bilayers with the plugin force.

The plugin has no membrane barostat. This spike runs OpenMM's pre-equilibrated POPC
patch (128 lipids, Lipid21 + TIP3P) with the *stock* MonteCarloMembraneBarostat
(XY isotropic, Z free, zero surface tension) in three setups:

    stock       stock force, stock integrator            (reference)
    cupo-nv     plugin force, plugin integrator          (rung 1 of the plan's ladder)
    cupo-stock  plugin force, stock integrator           (rung 2)

and compares area per lipid, box height, volume and potential energy.

Usage:
    spikes/run.sh spikes/m0_04_membrane_barostat.py run --mode cupo-nv --seed 1 --ns 20
    spikes/run.sh spikes/m0_04_membrane_barostat.py analyze
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from cufep.backends import get_backend

OUT = Path("spikes/out/membrane")
T = 303.15 * u.kelvin          # above the POPC main transition, as in the Lipid21 validation
P = 1.0 * u.bar
N_LIPIDS_PER_LEAFLET = 64


def build(ewald_tolerance: float = 5e-4, cutoff_nm: float = 0.9, hmr: bool = True):
    path = os.path.join(os.path.dirname(app.__file__), "data", "POPC.pdb")
    pdb = app.PDBFile(path)
    ff = app.ForceField("amber19/lipid21.xml", "amber14/tip3p.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_nm * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=ewald_tolerance,
                             **({"hydrogenMass": 3 * u.amu} if hmr else {}))
    return pdb.topology, pdb.positions, system


def run(mode: str, seed: int, ns: float, device: int, cutoff_nm: float = 0.9, hmr: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tight = mode == "stock-tight"          # stock reference with a 1e-5 Ewald tolerance
    top, pos, system = build(1e-5 if tight else 5e-4, cutoff_nm, hmr)
    tag = ("" if cutoff_nm == 0.9 else f"-rc{cutoff_nm:g}") + ("" if hmr else "-2fs")
    dt_fs = 4.0 if hmr else 2.0
    stock = get_backend("openmm")
    # common start: minimise and warm up on stock with the membrane barostat
    prepared = stock.prepare_system(system)
    prepared.addForce(stock.make_membrane_barostat(P, T, 25, seed=seed))
    integ = stock.make_integrator(T, 1 / u.picosecond, 2 * u.femtoseconds, seed=seed)
    ctx = stock.make_context(prepared, integ, device)
    ctx.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 1000)
    ctx.setVelocitiesToTemperature(T, seed)
    integ.step(25000)                                      # 50 ps
    st = ctx.getState(getPositions=True, getVelocities=True)
    start = (st.getPositions(asNumpy=True), st.getVelocities(asNumpy=True), st.getPeriodicBoxVectors())
    del ctx

    backend = {"stock": stock, "stock-tight": stock,
               "cupo-nv": get_backend("cupotential", integrator="nv", box_safety=1.3),
               "cupo-stock": get_backend("cupotential", integrator="stock", box_safety=1.3)}[mode]
    prepared = backend.prepare_system(system)
    prepared.addForce(backend.make_membrane_barostat(P, T, 25, seed=seed + 1000))
    integ = backend.make_integrator(T, 1 / u.picosecond, dt_fs * u.femtoseconds, seed=seed + 2000)
    ctx = backend.make_context(prepared, integ, device)
    ctx.setPeriodicBoxVectors(*start[2])
    ctx.setPositions(start[0])
    ctx.setVelocities(start[1])

    n_rec = int(round(ns * 1000 / 10))                     # one record per 10 ps
    rec = np.zeros((n_rec, 5))
    t0 = time.perf_counter()
    for i in range(n_rec):
        integ.step(int(round(10000.0 / dt_fs)))
        s = ctx.getState(getEnergy=True)
        bv = s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        rec[i] = (bv[0][0], bv[1][1], bv[2][2], s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
                  s.getKineticEnergy().value_in_unit(u.kilojoule_per_mole))
        if not np.isfinite(rec[i]).all():
            raise RuntimeError(f"non-finite state at record {i}")
    final = ctx.getState(getPositions=True)
    np.savez(OUT / f"{mode}{tag}_seed{seed}.npz", rec=rec, mode=mode + tag, seed=seed,
             positions=final.getPositions(asNumpy=True).value_in_unit(u.nanometer),
             box=final.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
    print(f"{mode} seed {seed}: {ns} ns done in {time.perf_counter() - t0:.0f} s; "
          f"final box {rec[-1, 0]:.3f} x {rec[-1, 1]:.3f} x {rec[-1, 2]:.3f} nm")


def block_se(x: np.ndarray, nblocks: int = 5) -> float:
    blocks = np.array_split(x, nblocks)
    return float(np.std([b.mean() for b in blocks], ddof=1) / np.sqrt(nblocks))


def analyze(skip_ns: float = 4.0) -> int:
    files = sorted(glob.glob(str(OUT / "*_seed*.npz")))
    if not files:
        print("no results")
        return 2
    skip = int(skip_ns * 100)
    summary: dict[str, list] = {}
    print(f"{'run':22s} {'area/lipid [A^2]':>20s} {'box z [nm]':>16s} {'volume [nm^3]':>18s} {'U [kJ/mol]':>20s}")
    for f in files:
        d = np.load(f)
        r = d["rec"][skip:]
        apl = 100.0 * r[:, 0] * r[:, 1] / N_LIPIDS_PER_LEAFLET
        vol = r[:, 0] * r[:, 1] * r[:, 2]
        row = [(apl.mean(), block_se(apl)), (r[:, 2].mean(), block_se(r[:, 2])),
               (vol.mean(), block_se(vol)), (r[:, 3].mean(), block_se(r[:, 3]))]
        summary.setdefault(str(d["mode"]), []).append(row)
        print(f"{Path(f).stem:22s} " + " ".join(f"{m:12.3f} ± {s:6.3f}" for m, s in row))
    print()
    ref = summary.get("stock")
    ok = True
    names = ["area/lipid", "box z", "volume", "U"]
    for mode, rows in summary.items():
        means = np.array([[q[0] for q in row] for row in rows]).mean(axis=0)
        ses = np.sqrt((np.array([[q[1] for q in row] for row in rows]) ** 2).sum(axis=0)) / len(rows)
        line = f"{mode:11s} mean over {len(rows)} runs: " + ", ".join(
            f"{n}={m:.3f}±{s:.3f}" for n, m, s in zip(names, means, ses)
        )
        if ref is not None and mode != "stock":
            rmeans = np.array([[q[0] for q in row] for row in ref]).mean(axis=0)
            rses = np.sqrt((np.array([[q[1] for q in row] for row in ref]) ** 2).sum(axis=0)) / len(ref)
            z = (means - rmeans) / np.sqrt(ses ** 2 + rses ** 2)
            line += " | z vs stock: " + ", ".join(f"{n} {v:+.1f}" for n, v in zip(names, z))
            # geometry must agree within 3 combined standard errors; U differs by the known
            # constant offset between the two Ewald splittings, so it is not tested here.
            ok &= bool(np.all(np.abs(z[:3]) < 3.0))
        print(line)
    print("RESULT:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", required=True, choices=["stock", "stock-tight", "cupo-nv", "cupo-stock"])
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--ns", type=float, default=20.0)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--cutoff", type=float, default=0.9)
    r.add_argument("--no-hmr", action="store_true", help="no hydrogen mass repartitioning, 2 fs")
    a = sub.add_parser("analyze")
    a.add_argument("--skip-ns", type=float, default=4.0)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.mode, args.seed, args.ns, args.device, args.cutoff, not args.no_hmr)
        return 0
    return analyze(args.skip_ns)


if __name__ == "__main__":
    raise SystemExit(main())
