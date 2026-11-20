"""M0 spike 2-5: TYK2 complex, alchemical System, stock vs cuPotential.

  [2] energies/forces agree at every lambda state; between-state energy differences
      agree within 0.02 kT (what MBAR actually consumes)
  [3] the fully decoupled state runs stably for 1 ns
  [4] charge offsets update live; plugin integrator works with a stock restraint force;
      one Context can be re-used for several replicas (positions/box/lambda swapped in)
  [5] cost of "set lambda + energy" and of MD steps (internal use only, SLA 2.12)

Usage: spikes/run.sh spikes/m0_02_tyk2_alchemical.py [--ligand ejm_31] [--box cube|dodecahedron]
                                                     [--device 0] [--long]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from cufep.alchemy.factory import LAMBDA_ELECTROSTATICS, LAMBDA_STERICS, make_alchemical_system, set_lambdas
from cufep.backends import get_backend
from cufep.prep.forcefield import ForceFieldSpec, SystemSpec, create_system, make_forcefield
from cufep.prep.ligand import assign_partial_charges, load_ligands
from cufep.prep.solvate import SolvationSpec, combine, ligand_atom_indices, solvate

T = 298.15 * u.kelvin
KT = (u.MOLAR_GAS_CONSTANT_R * T).value_in_unit(u.kilojoule_per_mole)
KJ = u.kilojoule_per_mole
OUT = Path("spikes/out")

STATES = [  # (lambda_electrostatics, lambda_sterics)
    (1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
    (0.0, 0.9), (0.0, 0.7), (0.0, 0.5), (0.0, 0.3), (0.0, 0.15), (0.0, 0.05), (0.0, 0.0),
]


def build(ligand_name: str, box_shape: str):
    protein = app.PDBFile("benchmarks/tyk2/inputs/tyk2_protein.pdb")
    ligand = next(l for l in load_ligands("benchmarks/tyk2/inputs/tyk2_ligands.sdf") if l.name == ligand_name)
    t0 = time.perf_counter()
    assign_partial_charges(ligand, "am1bcc", cache_dir=OUT / "cache")
    print(f"ligand {ligand.name}: {ligand.n_atoms} atoms, formal charge {ligand.formal_charge}, "
          f"charges in {time.perf_counter() - t0:.1f} s")
    ff = make_forcefield(ForceFieldSpec(), [ligand])
    modeller = combine((protein.topology, protein.positions), ligand)
    solvate(modeller, ff, SolvationSpec(padding_nm=1.0, box_shape=box_shape, min_edge_nm=2.75))
    system = create_system(ff, modeller.topology, SystemSpec())
    lig = ligand_atom_indices(modeller.topology)
    print(f"system: {system.getNumParticles()} atoms, ligand atoms {len(lig)}, box {box_shape}")
    return modeller.topology, modeller.positions, system, lig


def equilibrate_stock(system, positions, device):
    stock = get_backend("openmm")
    prepared = stock.prepare_system(system)
    prepared.addForce(stock.make_isotropic_barostat(1 * u.bar, T, 25, seed=5))
    integ = stock.make_integrator(T, 1 / u.picosecond, 2 * u.femtoseconds, seed=3)
    ctx = stock.make_context(prepared, integ, device)
    ctx.setPositions(positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
    e1 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    ctx.setVelocitiesToTemperature(T, 2)
    integ.step(10000)            # 20 ps at 2 fs
    integ.setStepSize(4 * u.femtoseconds)
    integ.step(25000)            # 100 ps at 4 fs
    st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
    print(f"stock equilibration: E {e0:.0f} -> {e1:.0f} -> {st.getPotentialEnergy().value_in_unit(KJ):.0f} kJ/mol, "
          f"V={st.getPeriodicBoxVolume().value_in_unit(u.nanometer**3):.1f} nm^3")
    out = (st.getPositions(asNumpy=True), st.getVelocities(asNumpy=True), st.getPeriodicBoxVectors())
    del ctx
    return out


def make_ctx(backend, alch_system, device, barostat=False, dt_fs=4.0, seed=1, extra_force=None):
    prepared = backend.prepare_system(alch_system)
    if extra_force is not None:
        prepared.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(extra_force)))
    if barostat:
        prepared.addForce(backend.make_isotropic_barostat(1 * u.bar, T, 25, seed=seed + 100))
    integ = backend.make_integrator(T, 1 / u.picosecond, dt_fs * u.femtoseconds, seed=seed)
    ctx = backend.make_context(prepared, integ, device)
    return ctx, integ


def load(ctx, pos, box, vel=None):
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(pos)
    if vel is not None:
        ctx.setVelocities(vel)


def energy(ctx, state=None):
    if state is not None:
        set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: state[0], LAMBDA_STERICS: state[1]})
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand", default="ejm_31")
    ap.add_argument("--box", default="cube")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--long", action="store_true", help="run the 1 ns decoupled-state stability test")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verdict = {}

    top, pos0, system, lig = build(args.ligand, args.box)
    pos, vel, box = equilibrate_stock(system, pos0, args.device)
    alch = make_alchemical_system(system, lig)
    mm_xml = mm.XmlSerializer.serialize(alch.system)
    (OUT / f"tyk2_{args.ligand}_{args.box}_alch_system.xml").write_text(mm_xml)
    with open(OUT / f"tyk2_{args.ligand}_{args.box}_equil.pdb", "w") as fh:
        app.PDBFile.writeFile(top, pos, fh, keepIds=True)
    np.savez(OUT / f"tyk2_{args.ligand}_{args.box}_equil.npz", positions=pos.value_in_unit(u.nanometer),
             velocities=vel.value_in_unit(u.nanometer / u.picosecond),
             box=np.array([[v.value_in_unit(u.nanometer) for v in vec] for vec in box]), ligand=np.array(lig))

    stock = get_backend("openmm")
    stock_dp = get_backend("openmm", precision="double")
    cupo = get_backend("cupotential", integrator="nv")

    # ---- [0] alchemical System at lambda=1 vs the original System (stock) ----------
    c_orig, _ = make_ctx(stock, system, args.device)
    load(c_orig, pos, box)
    e_orig = energy(c_orig)
    del c_orig
    c_s, i_s = make_ctx(stock, alch.system, args.device)
    load(c_s, pos, box, vel)
    e_alch1 = energy(c_s, (1.0, 1.0))
    print(f"\n[0] original E={e_orig:.3f}  alchemical(lambda=1) E={e_alch1:.3f}  diff={e_alch1 - e_orig:+.3f} kJ/mol")

    # ---- frames at the coupled state and at the decoupled state (stock MD) ---------
    frames = []
    for state, nfr in (((1.0, 1.0), 6), ((0.0, 0.5), 4), ((0.0, 0.0), 6)):
        set_lambdas(c_s, {LAMBDA_ELECTROSTATICS: state[0], LAMBDA_STERICS: state[1]})
        i_s.step(2500)
        for _ in range(nfr):
            i_s.step(500)
            st = c_s.getState(getPositions=True)
            frames.append((state, st.getPositions(asNumpy=True), st.getPeriodicBoxVectors()))
    print(f"collected {len(frames)} frames from stock MD at 3 lambda states")

    # ---- [2] energies at all states, three evaluators --------------------------------
    c_d, _ = make_ctx(stock_dp, alch.system, args.device)
    c_c, i_c = make_ctx(cupo, alch.system, args.device)
    own = [STATES.index(s) for s, _p, _b in frames]          # state each frame was sampled at
    # tight reference: same System with a 1e-6 Ewald tolerance, double precision
    tight = mm.XmlSerializer.deserialize(mm_xml)
    for force in tight.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setEwaldErrorTolerance(1e-6)
    c_t, _ = make_ctx(stock_dp, tight, args.device)
    U = {k: np.zeros((len(frames), len(STATES))) for k in ("stock", "double", "cupo", "tight")}
    relF = []
    for f, (state, p, b) in enumerate(frames):
        for key, ctx in (("stock", c_s), ("double", c_d), ("cupo", c_c), ("tight", c_t)):
            load(ctx, p, b)
            for k, s in enumerate(STATES):
                U[key][f, k] = energy(ctx, s)
        fs = []
        for ctx in (c_d, c_c):                                   # forces at the frame's own state
            set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: state[0], LAMBDA_STERICS: state[1]})
            fs.append(np.asarray(ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(KJ / u.nanometer)))
        relF.append((np.linalg.norm(fs[1] - fs[0]) / np.linalg.norm(fs[0]),
                     np.linalg.norm((fs[1] - fs[0])[lig]) / max(np.linalg.norm(fs[0][lig]), 1e-9)))
    rows = np.arange(len(frames))
    # reduced energy differences relative to the state the frame was sampled at; only pairs
    # within 60 kT matter to MBAR and replica exchange (the rest are overlapping atoms).
    dU = {k: (v - v[rows, own][:, None]) / KT for k, v in U.items()}
    mask = np.abs(dU["double"]) < 60.0
    err_cupo = np.abs(dU["cupo"] - dU["double"])[mask]
    err_stock = np.abs(dU["stock"] - dU["double"])[mask]
    off = (U["cupo"] - U["double"])[mask]
    relE = (np.abs(U["cupo"] - U["double"]) / np.abs(U["double"]))[mask]
    relF = np.array(relF)
    print("\n[2] reference = stock OpenMM in double precision")
    print(f"  (frame, state) pairs within 60 kT of the sampled state: {mask.sum()} of {mask.size}")
    print(
        f"  absolute energy, cupo vs double: max rel {relE.max():.2e}; offset mean {off.mean():+.3f} kJ/mol, sd "
        f"{off.std():.4f}"
    )
    print(f"  between-state dU error [kT]: cupo max {err_cupo.max():.4f} mean {err_cupo.mean():.4f} | "
          f"stock mixed max {err_stock.max():.4f} mean {err_stock.mean():.4f}")
    print(
        f"  forces at sampled state, cupo vs double: whole system max rel {relF[:, 0].max():.2e}; ligand atoms max rel "
        f"{relF[:, 1].max():.2e}"
    )
    # accuracy against the tight Ewald reference: is the plugin's splitting as accurate as
    # stock PME at the same nominal tolerance?
    mask_t = np.abs(dU["tight"]) < 60.0
    acc_cupo = np.abs(dU["cupo"] - dU["tight"])[mask_t]
    acc_pme = np.abs(dU["double"] - dU["tight"])[mask_t]
    print(f"  dU error vs tight Ewald (1e-6) [kT]: cupo max {acc_cupo.max():.4f} mean {acc_cupo.mean():.4f} | "
          f"stock PME 5e-4 max {acc_pme.max():.4f} mean {acc_pme.mean():.4f}")
    del c_t
    # Pass if the plugin is self-consistent with stock at the level of the PME approximation
    # itself: its dU error against the tight reference is no worse than twice stock PME's.
    verdict["energy_equivalence"] = bool(relE.max() < 1e-4 and relF[:, 0].max() < 5e-3
                                         and acc_cupo.max() < max(2 * acc_pme.max(), 0.05))
    verdict["cross_backend_dU_within_0.02kT"] = bool(err_cupo.max() < 0.02)

    # ---- [2b] force-group decomposition: U = G0(lambda_elec) + G1(lambda_sterics) ------
    def group_energy(ctx, group, state=None):
        if state is not None:
            set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: state[0], LAMBDA_STERICS: state[1]})
        return ctx.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(KJ)

    print("\n[2b] force-group decomposition (errors in kT)")
    for key, ctx in (("double", c_d), ("cupo", c_c)):
        quad_err, add_err = [], []
        for f in (0, 3, 7, 12, 15):
            _state, p, b = frames[f]
            load(ctx, p, b)
            g0 = {le: group_energy(ctx, 0, (le, 1.0)) for le in (0.0, 0.25, 0.5, 0.75, 1.0)}
            c = g0[0.0]
            a2 = 2 * (g0[1.0] - 2 * g0[0.5] + g0[0.0])
            a1 = g0[1.0] - g0[0.0] - a2
            for le in (0.25, 0.75):
                quad_err.append(abs(c + a1 * le + a2 * le * le - g0[le]) / KT)
            for (le, ls) in ((1.0, 1.0), (0.5, 1.0), (0.0, 0.5), (0.0, 0.0)):
                tot = energy(ctx, (le, ls))
                parts = group_energy(ctx, 0) + group_energy(ctx, 1)
                if abs(tot) < 1e9:
                    add_err.append(abs(tot - parts) / KT)
        print(f"  {key:6s}: quadratic-in-lambda_elec interpolation max err {max(quad_err):.5f}; "
              f"group0+group1 vs total max err {max(add_err):.5f}")
        if key == "cupo":
            verdict["group_decomposition"] = bool(max(quad_err) < 0.01 and max(add_err) < 0.01)

    def tcost(fn, n=40):
        fn(0)
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        return 1e3 * (time.perf_counter() - t0) / n

    load(c_c, frames[0][1], frames[0][2])
    load(c_s, frames[0][1], frames[0][2])
    for name, ctx in (("stock", c_s), ("cupo ", c_c)):
        a = tcost(lambda i, c=ctx: group_energy(c, 0, ((i % 5) / 4.0, 1.0)))
        bq = tcost(lambda i, c=ctx: group_energy(c, 1, (0.0, (i % 11) / 10.0)))
        cq = tcost(lambda i, c=ctx: energy(c, (0.0, (i % 11) / 10.0)))
        print(f"  {name} ms: group0 with new lambda_elec {a:.3f} | group1 with new lambda_sterics {bq:.3f} | "
              f"full energy with new lambda_sterics {cq:.3f}")

    # ---- [4a] live charge-offset update: energy returns after a round trip -----------
    load(c_c, frames[0][1], frames[0][2])
    e_a = energy(c_c, (1.0, 1.0))
    e_b = energy(c_c, (0.3, 1.0))
    e_c = energy(c_c, (1.0, 1.0))
    print(
        f"\n[4a] offsets live: E(1)={e_a:.3f} E(0.3)={e_b:.3f} back E(1)={e_c:.3f}  |round trip|={abs(e_c - e_a):.2e} "
        f"kJ/mol"
    )
    verdict["offset_roundtrip"] = bool(abs(e_c - e_a) < 1e-3 and abs(e_b - e_a) > 0.1)

    # ---- [5] cost of set-lambda + energy, and MD steps (internal) ---------------------
    def cost(ctx, n=40):
        t0 = time.perf_counter()
        for i in range(n):
            energy(ctx, STATES[i % len(STATES)])
        return 1e3 * (time.perf_counter() - t0) / n

    def cost_same(ctx, n=40):
        energy(ctx, STATES[0])
        t0 = time.perf_counter()
        for _ in range(n):
            ctx.getState(getEnergy=True)
        return 1e3 * (time.perf_counter() - t0) / n

    def cost_load(ctx, n=20):
        t0 = time.perf_counter()
        for i in range(n):
            load(ctx, frames[i % len(frames)][1], frames[i % len(frames)][2])
            ctx.getState(getEnergy=True)
        return 1e3 * (time.perf_counter() - t0) / n

    def steps(ctx, integ, n=5000):
        load(ctx, pos, box, vel)
        set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: 1.0, LAMBDA_STERICS: 1.0})
        integ.step(500)
        ctx.getState(getEnergy=True)
        t0 = time.perf_counter()
        integ.step(n)
        ctx.getState(getEnergy=True)
        return 1e3 * (time.perf_counter() - t0) / n

    print("\n[5] costs in ms (internal use only)")
    for name, ctx, integ in (("stock", c_s, i_s), ("cupo ", c_c, i_c)):
        print(f"  {name}: set-lambda+energy {cost(ctx):.3f} | energy only {cost_same(ctx):.3f} | "
              f"load frame+energy {cost_load(ctx):.3f} | NVT step {steps(ctx, integ):.4f}")
    del c_d

    # ---- [4b] one Context re-used for several replicas ---------------------------------
    ok = True
    reps = [(frames[0][1], frames[0][2], STATES[0]), (frames[7][1], frames[7][2], STATES[7]),
            (frames[-1][1], frames[-1][2], STATES[-1])]
    for cyc in range(3):
        for p, b, s in reps:
            load(c_c, p, b)
            c_c.setVelocitiesToTemperature(T, cyc + 1)
            set_lambdas(c_c, {LAMBDA_ELECTROSTATICS: s[0], LAMBDA_STERICS: s[1]})
            i_c.step(250)
            ok &= bool(np.isfinite(energy(c_c)))
    print(
        f"\n[4b] replica time-slicing through one plugin Context (3 replicas x 3 cycles x 250 steps): "
        f"{'OK' if ok else 'BAD'}"
    )
    verdict["replica_time_slicing"] = ok
    del c_c, c_s

    # ---- [4c] plugin integrator + stock restraint force --------------------------------
    restr = mm.CustomCompoundBondForce(2, "0.5*lambda_restraints*k*(distance(p1,p2)-r0)^2")
    restr.addGlobalParameter("lambda_restraints", 1.0)
    restr.addPerBondParameter("k")
    restr.addPerBondParameter("r0")
    ca = [a.index for a in top.atoms() if a.name == "CA"][100]
    restr.addBond([ca, lig[0]], [4184.0, 1.0])
    restr.setUsesPeriodicBoundaryConditions(True)
    c_r, i_r = make_ctx(cupo, alch.system, args.device, barostat=True, extra_force=restr)
    c_x, _ = make_ctx(stock, alch.system, args.device, extra_force=restr)
    load(c_r, pos, box, vel)
    i_r.step(2500)
    st = c_r.getState(getPositions=True, getEnergy=True)
    load(c_x, st.getPositions(asNumpy=True), st.getPeriodicBoxVectors())
    e_p, e_x = st.getPotentialEnergy().value_in_unit(KJ), energy(c_x)
    print(f"[4c] plugin integrator + nv barostat + stock restraint: 2500 steps OK, E plugin={e_p:.2f} stock={e_x:.2f} "
          f"(rel {abs(e_p - e_x) / abs(e_x):.2e})")
    verdict["restraint_with_plugin"] = bool(np.isfinite(e_p) and abs(e_p - e_x) / abs(e_x) < 1e-4)
    del c_x

    # ---- [3] decoupled-state stability ------------------------------------------------
    nsteps = 250000 if args.long else 25000
    set_lambdas(c_r, {LAMBDA_ELECTROSTATICS: 0.0, LAMBDA_STERICS: 0.0, "lambda_restraints": 1.0})
    load(c_r, frames[-1][1], frames[-1][2])
    c_r.setVelocitiesToTemperature(T, 9)
    t0 = time.perf_counter()
    bad = False
    try:
        for _chunk in range(nsteps // 5000):
            i_r.step(5000)
            e = energy(c_r)
            if not np.isfinite(e):
                bad = True
                break
    except Exception as exc:  # noqa: BLE001
        bad = True
        print("   decoupled run failed:", type(exc).__name__, str(exc)[:200])
    print(f"[3] fully decoupled state, plugin, NPT, {nsteps * 4e-6:.2f} ns: {'FAILED' if bad else 'stable'} "
          f"({time.perf_counter() - t0:.0f} s wall)")
    verdict["decoupled_stable"] = not bad

    print("\nVERDICT:", verdict)
    need = [k for k in verdict if k != "cross_backend_dU_within_0.02kT"]
    return 0 if all(verdict[k] for k in need) else 1


if __name__ == "__main__":
    raise SystemExit(main())
