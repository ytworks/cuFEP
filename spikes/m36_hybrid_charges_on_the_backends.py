"""Risk check for relative binding free energies: can the cuPotential plugin carry the charges of a
hybrid ligand as particle parameter offsets under five global parameters?

The planned layout (docs/plans/2026-11-25-rbfe.md) puts only charges on the plugin's
NonbondedForce: the charges of the hybrid ligand are offsets under three lambda parameters and the
REST2 parameter, and the hot environment atoms carry the REST2 offsets as in the ABFE. With REST2
the charge of a ligand atom is (base) * sqrt(s), so every lambda parameter is lambda * sqrt(s)
(the ABFE folds sqrt(s) into lambda_electrostatics the same way):

    core          q = (qA + lambda_core * (qB - qA)) * sqrt(s)
                    = qA + qA * rest2_dq + (qB - qA) * g_core          g_core = lambda_core * sqrt(s)
    unique old    q = qA * (1 - lambda_delete) * sqrt(s)
                    = qA + qA * rest2_dq - qA * g_delete               g_delete = lambda_delete * sqrt(s)
    unique new    q = qB * lambda_insert * sqrt(s) = qB * g_insert     g_insert = lambda_insert * sqrt(s)
    hot protein   q * sqrt(s), epsilon * s                             rest2_dq, rest2_ds

A core atom then has two charge offsets under two parameters, which the ABFE never had, and the
force has five global parameters, where the ABFE has three. The lambda functions are those of
OpenFE's default protocol: lambda_core = x, lambda_delete = min(2x, 1) (old charges off in the
first half), lambda_insert = max(2x - 1, 0) (new charges on in the second half).

The ligand of a finished TYK2 complex leg (ejm_31) is made a stand-in hybrid: the five ligand
atoms farthest from its centre are the unique old atoms, five added particles next to them the
unique new ones, the others the core with changed charges (B charges: random, the net charge 0).
All pairs inside the hybrid ligand are exceptions with no charge (their Coulomb energy goes to a
force of its own in the plan) and all ligand Lennard-Jones is off in the NonbondedForce (it goes to
the soft-core force on the stock kernels in the plan), so this checks the NonbondedForce alone.

Checks, energies of the NonbondedForce at 11 x 2 states (x = 0, 0.1, ..., 1; s = 1 and 0.6):
1. stock OpenMM with the offsets against a System with the charges of each state written in
   (same Ewald tolerance, double precision): are the offsets what they are meant to be;
2. the plugin with the offsets against the System with the charges written in, double precision
   and Ewald tolerance 1e-6 (the level of the GPU tests: 0.1 kT), as differences from the state
   x = 0, s = 1;
3. the plugin: within each half of x the energy is a quadratic in x (three evaluations give it);
4. MD on the plugin at x = 0.25, s = 0.6 with its integrator and barostat, then the parameters of
   x = 0.75 set in the running Context: energies finite, and equal to a new Context.

    spikes/run.sh spikes/m36_hybrid_charges_on_the_backends.py runs/benchmark_tyk2_full_e77a6c5 ejm_31 --device 0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

KT = 0.0083144626 * 298.15  # kJ/mol


def lambdas(x: float) -> dict[str, float]:
    return {"core": x, "delete": min(2.0 * x, 1.0), "insert": max(2.0 * x - 1.0, 0.0)}


def parameters(x: float, s: float) -> dict[str, float]:
    a = math.sqrt(s)
    lam = lambdas(x)
    return {"g_core": lam["core"] * a, "g_delete": lam["delete"] * a, "g_insert": lam["insert"] * a,
            "rest2_dq": a - 1.0, "rest2_ds": s - 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ligand")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--md-steps", type=int, default=2500)
    ap.add_argument("--no-dispersion-correction", action="store_true",
                    help="switch the long-range dispersion correction off, to see whether it explains a difference")
    ap.add_argument("--md-only", action="store_true", help="only check 4")
    args = ap.parse_args()

    import openmm as mm
    import openmm.unit as u

    from cufep.backends import get_backend
    from cufep.backends.base import clone_system, find_forces
    from cufep.workflow import load_prepared

    rng = np.random.default_rng(args.seed)
    lig_dir = Path(args.run_dir) / args.ligand
    prep = load_prepared(lig_dir / "prep", "complex")
    eq = np.load(lig_dir / "repeat_0" / "complex" / "equil.npz")
    base = mm.XmlSerializer.deserialize(prep["system_xml"])
    box = [mm.Vec3(*row) for row in eq["box"]]
    pos = np.array(eq["positions"], dtype=float)
    atoms = list(prep["topology"].atoms())
    ligand = [int(i) for i in prep["ligand"]]
    protein = set(prep["protein"])

    (_, nb0), = find_forces(base, mm.NonbondedForce)
    n0 = base.getNumParticles()
    q0 = np.array([nb0.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(n0)])
    lj_parameters = [(nb0.getParticleParameters(i)[1].value_in_unit(u.nanometer),
                      nb0.getParticleParameters(i)[2].value_in_unit(u.kilojoule_per_mole)) for i in range(n0)]
    if args.no_dispersion_correction:
        nb0.setUseDispersionCorrection(False)
    print("dispersion correction of the NonbondedForce:", nb0.getUseDispersionCorrection())
    centre = pos[ligand].mean(axis=0)
    far = sorted(ligand, key=lambda i: -np.linalg.norm(pos[i] - centre))
    unique_old = sorted(far[:5])
    core = sorted(i for i in ligand if i not in unique_old)
    n_new = 5
    unique_new = list(range(n0, n0 + n_new))
    new_pos = pos[unique_old] + np.array([0.03, 0.0, 0.0])
    positions = np.vstack([pos, new_pos])

    # B charges: core changed by up to 0.15 e, new atoms up to 0.4 e; the net charge of B is that of A
    qA = {i: q0[i] for i in ligand}
    qB = {i: q0[i] + rng.uniform(-0.15, 0.15) for i in core}
    for i in unique_new:
        qB[i] = rng.uniform(-0.4, 0.4)
    shift = (sum(qA.values()) - sum(qB.values())) / len(qB)
    qB = {i: q + shift for i, q in qB.items()}
    hybrid = sorted(set(ligand) | set(unique_new))
    hybrid_set = set(hybrid)

    # hot protein atoms: side chains of the residues with a heavy atom within 0.4 nm of the ligand
    lig_heavy = [i for i in ligand if atoms[i].element.symbol != "H"]
    near = {atoms[i].residue.index for i in protein
            if atoms[i].element is not None and atoms[i].element.symbol != "H"
            and np.min(np.linalg.norm(pos[lig_heavy] - pos[i], axis=1)) < 0.4}
    backbone = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3", "OXT", "H1", "H2", "H3"}
    hot_env = sorted(a.index for a in atoms if a.index in protein and a.residue.index in near
                     and a.name not in backbone)
    region = hybrid_set | set(hot_env)
    print(f"atoms {n0} + {n_new} new; core {len(core)}, unique old {len(unique_old)}, unique new {n_new}; "
          f"hot protein atoms {len(hot_env)} of {len(near)} residues")

    def skeleton() -> mm.System:
        """The base System with the new particles, ligand LJ off and every hybrid pair an exception
        without charge and LJ (base parameters: charges of A, new particles uncharged)."""
        out = clone_system(base)
        (_, f), = find_forces(out, mm.NonbondedForce)
        for i in unique_new:
            out.addParticle(12.0)
            f.addParticle(0.0, 0.3, 0.0)
        for i in ligand:
            q, sig, _eps = f.getParticleParameters(i)
            f.setParticleParameters(i, q, sig, 0.0)
        existing = set()
        for k in range(f.getNumExceptions()):
            i, j, _qq, sig, _eps = f.getExceptionParameters(k)
            if i in hybrid_set and j in hybrid_set:
                f.setExceptionParameters(k, i, j, 0.0, sig, 0.0)
                existing.add((min(i, j), max(i, j)))
        for a, i in enumerate(hybrid):
            for j in hybrid[a + 1:]:
                if (i, j) not in existing:
                    f.addException(i, j, 0.0, 0.3, 0.0)
        # tether each new particle to its old twin (only for the MD; no NonbondedForce term)
        tether = mm.HarmonicBondForce()
        for i, j in zip(unique_old, unique_new):
            tether.addBond(i, j, 0.03, 5000.0)
        out.addForce(tether)
        # the Lennard-Jones of the ligand with its surroundings, which the plan puts in the soft-core
        # force: here plain LJ with the parameters of A (new particles none), in group 1, so that
        # the MD has the repulsion that keeps charged ligand atoms off the water
        lj = mm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); "
                                     "epsilon=sqrt(epsilon1*epsilon2)")
        lj.addPerParticleParameter("sigma")
        lj.addPerParticleParameter("epsilon")
        for i in range(out.getNumParticles()):
            sig, eps = (0.3, 0.0) if i >= n0 else lj_parameters[i]
            lj.addParticle([sig if sig > 0 else 0.3, eps])
        lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        lj.setCutoffDistance(f.getCutoffDistance())
        lj.addInteractionGroup(set(hybrid), set(range(n0)) - hybrid_set)
        for k in range(f.getNumExceptions()):
            i, j, *_ = f.getExceptionParameters(k)
            lj.addExclusion(i, j)
        out.addForce(lj)
        for force in out.getForces():
            force.setForceGroup(0 if isinstance(force, mm.NonbondedForce) else 1)
        return out

    def with_offsets() -> mm.System:
        out = skeleton()
        (_, f), = find_forces(out, mm.NonbondedForce)
        for name in ("g_core", "g_delete", "g_insert", "rest2_dq", "rest2_ds"):
            f.addGlobalParameter(name, 0.0)
        for i in core:
            f.addParticleParameterOffset("rest2_dq", i, qA[i], 0.0, 0.0)
            f.addParticleParameterOffset("g_core", i, qB[i] - qA[i], 0.0, 0.0)
        for i in unique_old:
            f.addParticleParameterOffset("rest2_dq", i, qA[i], 0.0, 0.0)
            f.addParticleParameterOffset("g_delete", i, -qA[i], 0.0, 0.0)
        for i in unique_new:
            f.addParticleParameterOffset("g_insert", i, qB[i], 0.0, 0.0)
        for i in hot_env:
            q, _sig, eps = f.getParticleParameters(i)
            q, eps = q.value_in_unit(u.elementary_charge), eps.value_in_unit(u.kilojoule_per_mole)
            if q != 0.0:
                f.addParticleParameterOffset("rest2_dq", i, q, 0.0, 0.0)
            if eps != 0.0:
                f.addParticleParameterOffset("rest2_ds", i, 0.0, 0.0, eps)
        for k in range(f.getNumExceptions()):
            i, j, qq, _sig, eps = f.getExceptionParameters(k)
            n_hot = (i in region) + (j in region)
            qq, eps = qq.value_in_unit(u.elementary_charge ** 2), eps.value_in_unit(u.kilojoule_per_mole)
            if n_hot and (qq != 0.0 or eps != 0.0):
                f.addExceptionParameterOffset("rest2_ds" if n_hot == 2 else "rest2_dq", k, qq, 0.0, eps)
        return out

    def written(x: float, s: float) -> mm.System:
        """The System with the charges of the state written in, no offsets."""
        out = skeleton()
        (_, f), = find_forces(out, mm.NonbondedForce)
        a = math.sqrt(s)
        lam = lambdas(x)
        for i in core:
            _q, sig, eps = f.getParticleParameters(i)
            f.setParticleParameters(i, (qA[i] + lam["core"] * (qB[i] - qA[i])) * a, sig, eps)
        for i in unique_old:
            _q, sig, eps = f.getParticleParameters(i)
            f.setParticleParameters(i, qA[i] * (1.0 - lam["delete"]) * a, sig, eps)
        for i in unique_new:
            _q, sig, eps = f.getParticleParameters(i)
            f.setParticleParameters(i, qB[i] * lam["insert"] * a, sig, eps)
        for i in hot_env:
            q, sig, eps = f.getParticleParameters(i)
            f.setParticleParameters(i, q * a, sig, eps * s)
        for k in range(f.getNumExceptions()):
            i, j, qq, sig, eps = f.getExceptionParameters(k)
            n_hot = (i in region) + (j in region)
            if n_hot:
                f.setExceptionParameters(k, i, j, qq * a ** n_hot, sig, eps * a ** n_hot)
        return out

    def context(backend_name: str, system: mm.System, precision: str, ewald: float | None = None):
        sysm = clone_system(system)
        if ewald is not None:
            (_, f), = find_forces(sysm, mm.NonbondedForce)
            f.setEwaldErrorTolerance(ewald)
        backend = get_backend(backend_name, precision=precision)
        prepared = backend.prepare_system(sysm)
        integ = backend.make_integrator(298.15 * u.kelvin, 1.0 / u.picosecond, 4.0 * u.femtoseconds, seed=1)
        ctx = backend.make_context(prepared, integ, args.device)
        ctx.setPeriodicBoxVectors(*box)
        ctx.setPositions(positions)
        return ctx

    def nb_energy(ctx) -> float:
        return ctx.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

    def set_state(ctx, x: float, s: float) -> None:
        for name, value in parameters(x, s).items():
            ctx.setParameter(name, value)

    xs = [round(0.1 * k, 1) for k in range(11)]
    ss = [1.0, 0.6]
    states = [(x, s) for s in ss for x in xs]

    offsets = with_offsets()
    if not args.md_only:
        plugin = context("cupotential", offsets, "mixed")
        stock = context("openmm", offsets, "double")
        print("global parameters of the plugin Context:", sorted(plugin.getParameters().keys()))
        e_plugin, e_stock, e_written, e_reference = {}, {}, {}, {}
        for x, s in states:
            set_state(plugin, x, s)
            set_state(stock, x, s)
            nb_energy(plugin)  # a full evaluation first (neighbour lists), as the evaluator does
            e_plugin[(x, s)] = nb_energy(plugin)
            e_stock[(x, s)] = nb_energy(stock)
            sysm = written(x, s)
            e_written[(x, s)] = nb_energy(context("openmm", sysm, "double"))
            e_reference[(x, s)] = nb_energy(context("openmm", sysm, "double", ewald=1e-6))
        ref = (0.0, 1.0)
    
        print("\n1. stock OpenMM, offsets against charges written in (double precision, same Ewald tolerance), kT")
        for s in ss:
            d1 = [(e_stock[(x, s)] - e_written[(x, s)]) / KT for x in xs]
            print(f"   s = {s}: from {min(d1):.3e} to {max(d1):.3e} kT")
        d1p = [(e_plugin[k] - e_reference[k]) / KT for k in states]
        print(f"   (the plugin against the reference, absolute: from {min(d1p):.3f} to {max(d1p):.3f} kT)")
    
        print("\n2. differences from the state x = 0, s = 1, against the reference (charges written in, Ewald 1e-6), kT")
        print("   x     s     reference dE   plugin - ref   stock(5e-4) - ref")
        worst_plugin = worst_stock = 0.0
        for k in states:
            dref = (e_reference[k] - e_reference[ref]) / KT
            dp = (e_plugin[k] - e_plugin[ref]) / KT - dref
            ds = (e_written[k] - e_written[ref]) / KT - dref
            worst_plugin, worst_stock = max(worst_plugin, abs(dp)), max(worst_stock, abs(ds))
            print(f"   {k[0]:.1f}  {k[1]:.1f}  {dref:12.3f}  {dp:12.4f}  {ds:12.4f}")
        print(f"   largest: plugin {worst_plugin:.4f} kT, stock with the plugin's tolerance {worst_stock:.4f} kT")
    
        print("\n3. plugin: the energy within each half of x from three evaluations (quadratic), kT")
        worst_fit = 0.0
        for s in ss:
            for lo, mid, hi, inside in ((0.0, 0.25, 0.5, (0.1, 0.2, 0.3, 0.4)), (0.5, 0.75, 1.0, (0.6, 0.7, 0.8, 0.9))):
                pts = []
                for x in (lo, mid, hi):
                    set_state(plugin, x, s)
                    nb_energy(plugin)
                    pts.append(nb_energy(plugin))
                coef = np.polyfit([lo, mid, hi], pts, 2)
                for x in inside:
                    err = abs(np.polyval(coef, x) - e_plugin[(x, s)]) / KT
                    worst_fit = max(worst_fit, err)
        print(f"   largest error of the quadratic {worst_fit:.2e} kT")

    print("\n4. MD on the plugin at x = 0.25, s = 0.6, its integrator and barostat")
    md_system = clone_system(offsets)
    backend = get_backend("cupotential", precision="mixed")
    md_system.addForce(backend.make_isotropic_barostat(1.0 * u.bar, 298.15 * u.kelvin, 25, seed=3))
    prepared = backend.prepare_system(md_system)
    integ = backend.make_integrator(298.15 * u.kelvin, 1.0 / u.picosecond, 4.0 * u.femtoseconds, seed=5)
    ctx = backend.make_context(prepared, integ, args.device)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    set_state(ctx, 0.25, 0.6)
    ctx.setVelocitiesToTemperature(298.15 * u.kelvin, 7)
    for block in range(5):
        integ.step(args.md_steps // 5)
        st = ctx.getState(getEnergy=True)
        print(f"   after {(block + 1) * (args.md_steps // 5)} steps: potential "
              f"{st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole):.1f} kJ/mol, volume "
              f"{st.getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3):.3f} nm3")
    set_state(ctx, 0.75, 0.6)
    st = ctx.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=False)
    live = nb_energy(ctx)
    fresh = context("cupotential", offsets, "mixed")
    fresh.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    fresh.setPositions(st.getPositions())
    set_state(fresh, 0.75, 0.6)
    nb_energy(fresh)
    again = nb_energy(fresh)
    print(f"   parameters of x = 0.75 set in the running Context: {live:.3f} kJ/mol; a new Context: {again:.3f}; "
          f"difference {abs(live - again) / KT:.2e} kT; finite: {math.isfinite(live)}")
    # what MBAR uses is the difference between states for one configuration: the same in both Contexts?
    for name, c in (("running", ctx), ("new", fresh)):
        diffs = []
        for x in (0.0, 0.25, 0.5, 1.0):
            set_state(c, x, 0.6 if 0.0 < x < 1.0 else 1.0)
            nb_energy(c)
            diffs.append(nb_energy(c))
        set_state(c, 0.75, 0.6)
        nb_energy(c)
        e75 = nb_energy(c)
        print(f"   {name} Context: E(x) - E(0.75) for x = 0, 0.25, 0.5, 1: "
              + ", ".join(f"{(e - e75) / KT:.4f}" for e in diffs) + " kT")
    for name, c in (("running", ctx), ("new", fresh)):
        print(f"   {name} Context: PME parameters {c.getSystem().getForce(0) if False else ''}"
              f"box {[round(v, 4) for v in np.diag(np.array(c.getState().getPeriodicBoxVectors(asNumpy=True)._value))]}")


if __name__ == "__main__":
    main()
