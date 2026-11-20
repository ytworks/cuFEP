"""REST2 feasibility: can the NonbondedForce carry the solute scaling of protein atoms as offsets?

REST2 scales the charges of the atoms of the hot region by sqrt(s) and their Lennard-Jones
epsilon by s (s = T0 / T_hot). For protein atoms in the hot region that means particle
parameter offsets on charge and on epsilon, driven by global parameters, and for their 1-4
pairs exception parameter offsets. This script checks on a finished TYK2 complex leg that

1. stock OpenMM gives the same energies with offsets as with directly scaled parameters,
2. the cuPotential plugin accepts such a NonbondedForce and gives the same energy differences,
3. what a change of the global parameters followed by an energy evaluation costs.

    spikes/run.sh spikes/m10_rest2_offsets.py runs/tyk2_default_5gpu_v2 ejm_31 0 --device 0
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ligand")
    ap.add_argument("repeat", type=int)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cutoff-nm", type=float, default=0.5, help="residues with a heavy atom this close to the ligand")
    ap.add_argument(
        "--no-dispersion-correction",
        action="store_true",
        help="switch the long-range dispersion correction off in all Systems, to see whether it explains a difference",
    )
    args = ap.parse_args()

    import openmm as mm
    import openmm.unit as u

    from cufep.backends import get_backend
    from cufep.backends.base import clone_system, find_forces
    from cufep.workflow import load_prepared

    lig_dir = Path(args.run_dir) / args.ligand
    leg_dir = lig_dir / f"repeat_{args.repeat}" / "complex"
    prep = load_prepared(lig_dir / "prep", "complex")
    eq = np.load(leg_dir / "equil.npz")
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    box = [mm.Vec3(*row) for row in eq["box"]]
    system.setDefaultPeriodicBoxVectors(*box)
    pos = eq["positions"]
    atoms = list(prep["topology"].atoms())
    ligand = np.array(prep["ligand"], dtype=int)

    # hot region: side chains of the residues around the ligand
    heavy = np.array([a.index for a in atoms if a.element is not None and a.element.symbol != "H"])
    lig_heavy = np.array([i for i in ligand if atoms[i].element.symbol != "H"])
    protein = set(prep["protein"])
    near = set()
    for i in heavy:
        if i in protein and np.min(np.linalg.norm(pos[lig_heavy] - pos[i], axis=1)) < args.cutoff_nm:
            near.add(atoms[i].residue.index)
    backbone = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3", "OXT", "H1", "H2", "H3"}
    region = sorted(a.index for a in atoms if a.index in protein and a.residue.index in near and a.name not in backbone)
    names = sorted({f"{atoms[i].residue.name}{atoms[i].residue.id}" for i in region})
    print(f"hot protein atoms: {len(region)} side-chain atoms of {len(names)} residues: {' '.join(names)}")
    region_set = set(region)

    (_, nb), = find_forces(system, mm.NonbondedForce)
    print("long-range dispersion correction of the prepared System:", nb.getUseDispersionCorrection())
    if args.no_dispersion_correction:
        nb.setUseDispersionCorrection(False)
    params = [nb.getParticleParameters(i) for i in range(system.getNumParticles())]

    def with_offsets() -> mm.System:
        out = clone_system(system)
        (_, f), = find_forces(out, mm.NonbondedForce)
        # the base parameters stay the physical ones; the global parameters are sqrt(s) - 1 and
        # s - 1, zero in the physical state
        f.addGlobalParameter("rest2_sqrt", 0.0)      # sqrt(s) - 1: charges
        f.addGlobalParameter("rest2_scale", 0.0)     # s - 1: epsilon, pairs inside the region
        for i in region:
            q, sig, eps = params[i]
            f.addParticleParameterOffset("rest2_sqrt", i, q, 0.0, 0.0)
            f.addParticleParameterOffset("rest2_scale", i, 0.0, 0.0, eps)
        n_exc = 0
        for k in range(f.getNumExceptions()):
            i, j, qq, sig, eps = f.getExceptionParameters(k)
            n_hot = (i in region_set) + (j in region_set)
            if n_hot == 0 or (
                qq.value_in_unit(u.elementary_charge**2) == 0.0 and eps.value_in_unit(u.kilojoule_per_mole) == 0.0
            ):
                continue
            f.addExceptionParameterOffset("rest2_scale" if n_hot == 2 else "rest2_sqrt", k, qq, 0.0, eps)
            n_exc += 1
        print(f"offsets: {2 * len(region)} particle offsets, {n_exc} exception offsets")
        return out

    def scaled(a: float) -> mm.System:
        out = clone_system(system)
        (_, f), = find_forces(out, mm.NonbondedForce)
        for i in region:
            q, sig, eps = params[i]
            f.setParticleParameters(i, q * a, sig, eps * a * a)
        for k in range(f.getNumExceptions()):
            i, j, qq, sig, eps = f.getExceptionParameters(k)
            n_hot = (i in region_set) + (j in region_set)
            if n_hot:
                f.setExceptionParameters(k, i, j, qq * a ** n_hot, sig, eps * a ** n_hot)
        return out

    def context(backend_name: str, sysm: mm.System, precision: str = "mixed"):
        backend = get_backend(backend_name, precision=precision)
        prepared = backend.prepare_system(sysm)
        integ = backend.make_integrator(298.15 * u.kelvin, 1.0 / u.picosecond, 4.0 * u.femtoseconds, seed=1)
        ctx = backend.make_context(prepared, integ, args.device)
        ctx.setPeriodicBoxVectors(*box)
        ctx.setPositions(pos)
        return ctx

    def energy_forces(ctx):
        st = ctx.getState(getEnergy=True, getForces=True)
        return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
                st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))

    offsets = with_offsets()
    scales = (1.0, 0.8, 0.5, 0.3)                      # s = T0 / T_hot: 298, 373, 596, 994 K
    ref = {}
    for s in scales:
        ctx = context("openmm", scaled(math.sqrt(s)), precision="double")
        ref[s] = energy_forces(ctx)
        del ctx
    print("reference: stock OpenMM, double precision, parameters scaled directly")
    for name in ("openmm", "cupotential"):
        try:
            ctx = context(name, offsets)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: cannot build a Context with these offsets: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        e1 = None
        for s in scales:
            ctx.setParameter("rest2_sqrt", math.sqrt(s) - 1.0)
            ctx.setParameter("rest2_scale", s - 1.0)
            e, f = energy_forces(ctx)
            e1 = e if e1 is None else e1
            de, de_ref = e - e1, ref[s][0] - ref[1.0][0]
            fr = ref[s][1]
            rel = np.linalg.norm(f - fr) / np.linalg.norm(fr)
            hot = np.linalg.norm(f[region] - fr[region]) / np.linalg.norm(fr[region])
            print(
                f"  {name:11s} s = {s:.1f} (T_hot {298.15 / s:5.0f} K): E(s) - E(1) = {de:12.2f} kJ/mol, reference "
                f"{de_ref:12.2f}, "
                f"difference {de - de_ref:8.3f}; relative force difference {rel:.1e} (hot atoms {hot:.1e})"
            )
        # cost of switching the scale and evaluating the energy
        t0 = time.perf_counter()
        n = 20
        for k in range(n):
            s = 0.5 if k % 2 else 0.8
            ctx.setParameter("rest2_sqrt", math.sqrt(s) - 1.0)
            ctx.setParameter("rest2_scale", s - 1.0)
            ctx.getState(getEnergy=True)
        dt = (time.perf_counter() - t0) / n
        ctx.getState(getEnergy=True)
        t0 = time.perf_counter()
        for _k in range(n):
            ctx.getState(getEnergy=True)
        dt0 = (time.perf_counter() - t0) / n
        print(
            f"  {name:11s} energy evaluation {1000 * dt0:.1f} ms; with a change of the two parameters {1000 * dt:.1f} "
            f"ms"
        )
        del ctx


if __name__ == "__main__":
    main()
