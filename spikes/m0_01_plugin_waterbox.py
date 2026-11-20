"""M0 spike 1: does the cuPotential plugin run in the container on this driver?

Checks on a TIP3P water box:
  * energy and force agreement with stock OpenMM on identical coordinates
  * NVT dynamics with the plugin force (plugin integrator and stock integrator)
  * NPT dynamics with every barostat we may need:
      nv isotropic, stock isotropic, stock MonteCarloMembraneBarostat (semi-isotropic)

Timings are printed for internal use only (NVIDIA SLA 2.12: do not publish).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from cufep.backends import get_backend

T = 298.15 * u.kelvin
P = 1.0 * u.bar
KJ = u.kilojoule_per_mole


def build_water(edge_nm: float = 4.0):
    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(edge_nm, edge_nm, edge_nm) * u.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    return mod.topology, mod.positions, system


def energy_forces(backend, system, positions, box, device=0):
    prepared = backend.prepare_system(system)
    ctx = backend.make_context(prepared, backend.make_integrator(T, 1 / u.picosecond, 0.002 * u.picoseconds), device)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(KJ)
    f = st.getForces(asNumpy=True).value_in_unit(KJ / u.nanometer)
    del ctx
    return e, np.asarray(f)


def run_md(label, backend, system, positions, box, barostat=None, steps=5000, dt_fs=2.0, device=0):
    prepared = backend.prepare_system(system)
    if barostat == "iso":
        prepared.addForce(backend.make_isotropic_barostat(P, T, 25, seed=11))
    elif barostat == "stock-iso":
        b = mm.MonteCarloBarostat(P, T, 25)
        b.setRandomNumberSeed(11)
        prepared.addForce(b)
    elif barostat == "membrane":
        prepared.addForce(backend.make_membrane_barostat(P, T, 25, seed=11))
    integ = backend.make_integrator(T, 1 / u.picosecond, dt_fs * u.femtoseconds, seed=7)
    try:
        ctx = backend.make_context(prepared, integ, device)
        ctx.setPeriodicBoxVectors(*box)
        ctx.setPositions(positions)
        ctx.setVelocitiesToTemperature(T, 3)
        integ.step(500)
        t0 = time.perf_counter()
        integ.step(steps)
        st = ctx.getState(getEnergy=True)
        dt = time.perf_counter() - t0
        e = st.getPotentialEnergy().value_in_unit(KJ)
        vol = st.getPeriodicBoxVolume().value_in_unit(u.nanometer**3)
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        ok = np.isfinite(e)
        print(f"  {label:44s} {'OK ' if ok else 'BAD'} E={e:12.1f} kJ/mol  V={vol:8.3f} nm^3 "
              f"box=({bv[0][0]:.3f},{bv[1][1]:.3f},{bv[2][2]:.3f})  {1e3 * dt / steps:.3f} ms/step")
        del ctx
        return ok, vol
    except Exception as exc:  # noqa: BLE001 - spike: report and continue
        print(f"  {label:44s} FAIL {type(exc).__name__}: {str(exc)[:160]}")
        return False, float("nan")


def main() -> int:
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("openmm", mm.__version__)
    top, pos, system = build_water(4.0)
    n = system.getNumParticles()
    print(f"water box: {n} atoms")

    stock = get_backend("openmm")
    # relax on stock first (plugin is reported to crash from unminimised coordinates)
    integ = stock.make_integrator(T, 1 / u.picosecond, 0.002 * u.picoseconds, seed=1)
    ctx = stock.make_context(stock.prepare_system(system), integ, device)
    ctx.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 500)
    ctx.setVelocitiesToTemperature(T, 1)
    integ.step(5000)
    st = ctx.getState(getPositions=True)
    pos = st.getPositions(asNumpy=True)
    box = st.getPeriodicBoxVectors()
    del ctx

    print("\n[1] energy / force agreement on identical coordinates")
    e_s, f_s = energy_forces(stock, system, pos, box, device)
    results = {}
    for kind in ("nv", "stock"):
        cupo = get_backend("cupotential", integrator=kind)
        e_c, f_c = energy_forces(cupo, system, pos, box, device)
        rel_f = np.linalg.norm(f_c - f_s) / np.linalg.norm(f_s)
        print(f"  cupotential(integrator={kind}): E={e_c:.3f}  stock E={e_s:.3f}  dE={e_c - e_s:+.4f} kJ/mol "
              f"(rel {abs(e_c - e_s) / abs(e_s):.2e})  rel force err={rel_f:.2e}")
        results[kind] = (abs(e_c - e_s) / abs(e_s), rel_f)

    print("\n[2] dynamics (2 fs), plugin force")
    ok_all = True
    vols = {}
    for label, backend, baro in [
        ("stock            / NVT", stock, None),
        ("cupo nv-integ    / NVT", get_backend("cupotential", integrator="nv"), None),
        ("cupo stock-integ / NVT", get_backend("cupotential", integrator="stock"), None),
        ("stock            / NPT stock iso", stock, "stock-iso"),
        ("cupo nv-integ    / NPT nv iso", get_backend("cupotential", integrator="nv"), "iso"),
        ("cupo nv-integ    / NPT stock iso", get_backend("cupotential", integrator="nv"), "stock-iso"),
        ("cupo stock-integ / NPT nv iso", get_backend("cupotential", integrator="stock"), "iso"),
        ("stock            / NPT membrane baro", stock, "membrane"),
        ("cupo nv-integ    / NPT membrane baro", get_backend("cupotential", integrator="nv"), "membrane"),
        ("cupo stock-integ / NPT membrane baro", get_backend("cupotential", integrator="stock"), "membrane"),
    ]:
        ok, vol = run_md(label, backend, system, pos, box, baro, steps=10000, device=device)
        vols[label] = vol
        ok_all &= ok

    tol_e, tol_f = 1e-4, 5e-3
    passed = ok_all and all(r[0] < tol_e and r[1] < tol_f for r in results.values())
    print("\nRESULT:", "PASS" if passed else "CHECK", f"(energy rel tol {tol_e}, force rel tol {tol_f})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
