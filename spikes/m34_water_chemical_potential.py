"""The excess chemical potential and the standard volume of water in cuFEP's own Hamiltonian, with TIP3P
and with OPC: the two numbers per water model that grand canonical Monte Carlo of water needs
(plan docs/plans/2026-11-24-gcmc-water.md).

    scripts/in_docker.sh --gpus 4,5,6,7 python spikes/m34_water_chemical_potential.py

For each water model a cubic box of pure water (``--box-nm``, 3.5 nm) is built with the force field
file of the model and ``cufep.prep.forcefield.create_system`` with the defaults of ``SystemSpec``,
which is how a run builds its Systems: PME with a cutoff of 0.9 nm and an Ewald tolerance of 5e-4,
the dispersion correction on, no switching function, rigid water, hydrogens of 3 amu. It is minimised
and equilibrated (NVT, then NPT) on stock OpenMM with fixed seeds. Everything is at 298.15 K and
1 bar (``thermodynamics``), with the integrator, the barostat (every 25 steps), the friction (1 / ps)
and the time step (4 fs) that a leg runs with.

Measured:

    V°      the mean volume per water molecule of the liquid: NPT MD of the box after 0.2 ns, the
            volume every 1 ps, on the cuPotential backend (the default backend of a run) and on stock
            OpenMM, with the standard error from 5 time blocks, and the density.
    mu_ex   the excess chemical potential: ONE water of the box is decoupled as a solvent leg decouples
            a ligand. The alchemical System is that of ``make_alchemical_system`` with the atoms of the
            water as the ligand (all four sites for OPC) and softcore_alpha 0.5; the windows are those
            of ``fixed_schedule`` with the stages discharge and vanish (``--windows``, 8 and 16
            intervals: 25 windows); the sampling is Hamiltonian replica exchange with worker processes
            (``LegSpec``, ``ReplicaExchangeSampler``; three workers per GPU under cuFEP's own NVIDIA
            MPS daemon) on the cuPotential backend, iterations of 2.5 ps, 0.5 ns of equilibration
            and ``--production-ns`` (3 ns) of production per window, every iteration stored; the
            analysis is ``leg_result_from_samples`` of the workflow with the defaults of the
            ``analysis`` section (MBAR, the tail correction by reweighting to 2.0 nm, the errors from
            time blocks). mu_ex = - dG(decouple).

Before the sampling the alchemical System is checked on stock OpenMM in double precision and on the
plugin: at the fully coupled state its potential energy is that of the original System less the part
of the analytic dispersion correction that belongs to the water (the alchemical System has the
Lennard-Jones interaction of the ligand in a soft-core force that has no such correction), and at the
decoupled state it is that of a System in which the charges and the epsilon of the water are zero.
A water model that fails this check is not sampled.

No symmetry or standard-state term is applied, and none is needed: the decoupled water is one of the
N identical molecules of the liquid and has no restraint, so the ratio of configuration integrals
that the decoupling gives, Z(N - 1) V / Z(N), is exp(mu_ex / kT) itself; the factor N of the
identical molecules and the volume belong to the ideal part of the chemical potential.

The MBAR part is the free energy with the Lennard-Jones interaction of the water cut at 0.9 nm and
no correction for what lies beyond. Printed beside it: the tail correction of the analysis (to the
full range of the interaction, by reweighting and in the homogeneous approximation), and the change of
OpenMM's analytic dispersion correction of the original System when the epsilon of the one water is
set to zero (stock OpenMM, double precision; it goes as 1 / V, and the table takes it at the mean
volume of the NPT run on the plugin), which is what a simulation with that correction has in its
energy when a water is inserted. The last lines of each model are the Adams value
B = mu_ex / kT + ln(V_sphere / V°) for a sphere of 0.8 nm, with each of the three mu_ex. Published
for TIP3P (Samways et al., J. Chem. Inf. Model. 2020, the package grand, 298 K): mu_ex -6.09 kcal/mol,
V° 30.345 A^3.

``--water`` names a water model file (repeatable; both by default), ``--gpus`` the number of GPUs
inside the container that the workers of the leg use (all by default); MD outside the leg runs on
the first.

The numbers of each water model are written as JSON to ``--out`` (``volume.json``,
``energy_check.json``, ``chemical_potential.json``, ``leg_result.json``). The equilibrated box
(``equilibrated.npz``) and the samples of the leg (``<out>/<model>/leg/hrex``) stay there, and the same
command starts from that box again and continues a leg that was stopped: MD on a GPU is not the same to
the bit in two runs, and a leg is continued only with the System its samples were made with. ``--part
analysis`` analyses the samples that are stored, also while the leg is running. No time is measured.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from cufep.alchemy.factory import make_alchemical_system
from cufep.alchemy.lrc import TailEnergy
from cufep.alchemy.schedule import fixed_schedule
from cufep.backends import get_backend
from cufep.backends.base import clone_system
from cufep.config.schema import AlchemyConfig, AnalysisConfig, DynamicsConfig, SamplingConfig, ThermodynamicsConfig
from cufep.prep.forcefield import SystemSpec, create_system
from cufep.prep.water import water_model_of
from cufep.sampling.states import set_state
from cufep.steps import md_steps

KJ = u.kilojoule_per_mole
KCAL = 4.184
R_GAS = 8.31446261815324e-3                      # kJ/mol/K
AMU_PER_NM3_IN_G_PER_CM3 = 1.66053907e-3
WATERS = ("amber14/tip3p.xml", "amber14/opc.xml")
PUBLISHED = {"amber14/tip3p.xml": {"mu_ex_kcal": -6.09, "v_A3": 30.345}}      # Samways et al. 2020
SPHERE_RADIUS_NM = 0.8
BUILD_SEED, EQUILIBRATION_SEED, VOLUME_SEED, LEG_SEED = 20261125, 11, 23, 34
VOLUME_EQUILIBRATION_NS = 0.2
VOLUME_BLOCKS = 5


def log(message: str) -> None:
    print(message, flush=True)


def settings_of_a_run() -> dict:
    """The defaults of the configuration that the measurement mirrors."""
    thermo, dyn, alchemy, sampling, analysis = (ThermodynamicsConfig(), DynamicsConfig(), AlchemyConfig(),
                                                SamplingConfig(), AnalysisConfig())
    return {"temperature_K": thermo.temperature_K, "pressure_bar": thermo.pressure_bar,
            "timestep_fs": dyn.timestep_fs, "friction_per_ps": dyn.friction_per_ps,
            "barostat_frequency": dyn.barostat_frequency, "cutoff_nm": SystemSpec().cutoff_nm,
            "ewald_tolerance": SystemSpec().ewald_tolerance, "hydrogen_mass_amu": SystemSpec().hydrogen_mass_amu,
            "softcore_alpha": alchemy.softcore_alpha, "softcore_b": alchemy.softcore_b,
            "annihilate_electrostatics": alchemy.annihilate_electrostatics,
            "time_per_iteration_ps": sampling.time_per_iteration_ps, "exchange": sampling.exchange,
            "lrc": analysis.lrc, "lrc_cutoff_nm": analysis.lrc_cutoff_nm, "min_overlap": analysis.min_overlap,
            "min_effective_samples": analysis.min_effective_samples}


def short(water: str) -> str:
    return Path(water).stem


def build(water: str, box_nm: float):
    """A cubic box of pure water and its System as a run builds one."""
    ff = app.ForceField(water)
    mod = app.Modeller(app.Topology(), [])
    state = random.getstate()
    random.seed(BUILD_SEED)
    try:
        mod.addSolvent(ff, model=water_model_of(water).box, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * u.nanometer)
    finally:
        random.setstate(state)
    system = create_system(ff, mod.topology, SystemSpec())
    pos = np.asarray(mod.positions.value_in_unit(u.nanometer), dtype=float)
    box = np.asarray(mod.topology.getPeriodicBoxVectors().value_in_unit(u.nanometer), dtype=float)
    return system, mod.topology, pos, box


def md_context(backend, system, run: dict, timestep_fs: float, seed: int, barostat: bool, device: int = 0):
    """A Context of ``backend`` as a worker of a leg makes it (ReplicaWorker.__init__)."""
    temperature = run["temperature_K"] * u.kelvin
    prepared = backend.prepare_system(system)
    if barostat:
        prepared.addForce(backend.make_isotropic_barostat(run["pressure_bar"] * u.bar, temperature,
                                                          run["barostat_frequency"], seed=seed + 1))
    integ = backend.make_integrator(temperature, run["friction_per_ps"] / u.picosecond,
                                    timestep_fs * u.femtoseconds, seed=seed)
    return backend.make_context(prepared, integ, device), integ


def load(ctx, pos, box) -> None:
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(pos)


def configuration(ctx):
    st = ctx.getState(getPositions=True)
    return (np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=float),
            np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer), dtype=float))


def equilibrate(system, pos, box, run: dict):
    """Minimise, 20 ps at constant volume with 2 fs, 200 ps at constant pressure with the time step of a
    run; stock OpenMM, fixed seeds."""
    stock = get_backend("openmm")
    temperature = run["temperature_K"] * u.kelvin
    ctx, integ = md_context(stock, system, run, 2.0, EQUILIBRATION_SEED, False)
    load(ctx, pos, box)
    ctx.applyConstraints(1e-6)
    ctx.computeVirtualSites()
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
    ctx.setVelocitiesToTemperature(temperature, EQUILIBRATION_SEED)
    integ.step(md_steps(20.0, 2.0))
    pos, box = configuration(ctx)
    del ctx, integ
    ctx, integ = md_context(stock, system, run, run["timestep_fs"], EQUILIBRATION_SEED + 2, True)
    load(ctx, pos, box)
    ctx.setVelocitiesToTemperature(temperature, EQUILIBRATION_SEED + 2)
    integ.step(md_steps(200.0, run["timestep_fs"]))
    pos, box = configuration(ctx)
    del ctx, integ
    return pos, box


# ------------------------------------------------------------------------------------------- V°
def measure_volume(system, n_waters: int, pos, box, backend_name: str, ns: float, run: dict) -> dict:
    """NPT MD of the box on one backend: the mean volume per water and the density."""
    mass = sum(system.getParticleMass(i).value_in_unit(u.amu) for i in range(system.getNumParticles()))
    dt = run["timestep_fs"]
    ctx, integ = md_context(get_backend(backend_name), system, run, dt, VOLUME_SEED, True)
    load(ctx, pos, box)
    ctx.setVelocitiesToTemperature(run["temperature_K"] * u.kelvin, VOLUME_SEED)
    integ.step(md_steps(VOLUME_EQUILIBRATION_NS * 1000.0, dt))
    per_ps = md_steps(1.0, dt)
    volumes = np.empty(int(round(ns * 1000.0)))
    for i in range(volumes.size):                                    # one volume per ps
        integ.step(per_ps)
        volumes[i] = ctx.getState().getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3)
    del ctx, integ
    if not np.all(np.isfinite(volumes)):
        raise FloatingPointError(f"the volume of the box became non-finite on {backend_name} with {dt:g} fs")
    blocks = volumes[: volumes.size // VOLUME_BLOCKS * VOLUME_BLOCKS].reshape(VOLUME_BLOCKS, -1).mean(axis=1)
    per_water = 1000.0 * blocks / n_waters                           # A^3
    density = mass / blocks * AMU_PER_NM3_IN_G_PER_CM3
    return {"backend": backend_name, "ns": float(ns), "samples": int(volumes.size), "blocks": VOLUME_BLOCKS,
            "volume_nm3": float(volumes.mean()), "volume_sd_nm3": float(volumes.std(ddof=1)),
            "v_A3": float(1000.0 * volumes.mean() / n_waters),
            "v_se_A3": float(per_water.std(ddof=1) / math.sqrt(VOLUME_BLOCKS)),
            "v_blocks_A3": [float(x) for x in per_water],
            "density_g_cm3": float(mass / volumes.mean() * AMU_PER_NM3_IN_G_PER_CM3),
            "density_se_g_cm3": float(density.std(ddof=1) / math.sqrt(VOLUME_BLOCKS))}


# ----------------------------------------------------------------------- the alchemical System
def changed(system, atoms, charge: bool = False, epsilon: bool = False, dispersion: bool | None = None):
    """A copy of ``system`` with the charges and/or the epsilons of ``atoms`` set to zero, and with the
    dispersion correction as given."""
    out = clone_system(system)
    nb = next(f for f in out.getForces() if isinstance(f, mm.NonbondedForce))
    for i in atoms:
        q, sigma, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0 if charge else q, sigma, 0.0 if epsilon else eps)
    if dispersion is not None:
        nb.setUseDispersionCorrection(dispersion)
    return out


def potential_energy(backend, system, pos, box, run: dict, state=None) -> float:
    ctx, integ = md_context(backend, system, run, run["timestep_fs"], 1, False)
    load(ctx, pos, box)
    if state is not None:
        set_state(ctx, state)
    energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    del ctx, integ
    return float(energy)


def check_alchemical_system(system, alch, schedule, ligand, pos, box, run: dict) -> dict:
    """The energies of the alchemical System at its two ends against those of plain Systems, and the
    part of the analytic dispersion correction that belongs to the water (kJ/mol)."""
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    nb_alch = next(f for f in alch.system.getForces() if isinstance(f, mm.NonbondedForce))
    coupled, decoupled = schedule.states[0], schedule.states[-1]
    double = get_backend("openmm", precision="double")
    correction = {}
    for name, sys_ in (("all", system), ("without", changed(system, ligand, epsilon=True))):
        correction[name] = (potential_energy(double, changed(sys_, (), dispersion=True), pos, box, run)
                            - potential_energy(double, changed(sys_, (), dispersion=False), pos, box, run))
    of_the_water = correction["all"] - correction["without"]
    out = {"exceptions_original": nb.getNumExceptions(), "exceptions_alchemical": nb_alch.getNumExceptions(),
           "virtual_sites_in_the_ligand": [int(i) for i in ligand if system.isVirtualSite(i)],
           "ligand_parameters": [list(p) for p in alch.ligand_parameters],
           "volume_nm3": float(abs(np.linalg.det(box))),
           "dispersion_correction_kJ": correction["all"], "dispersion_correction_of_the_water_kJ": of_the_water,
           "backends": {}, "passed": True}
    zero = changed(system, ligand, charge=True, epsilon=True)
    for name, backend, level in (("openmm double", double, 1e-7), ("cupotential", get_backend("cupotential"), 1e-5)):
        e = {"original": potential_energy(backend, system, pos, box, run),
             "alchemical_coupled": potential_energy(backend, alch.system, pos, box, run, coupled),
             "alchemical_decoupled": potential_energy(backend, alch.system, pos, box, run, decoupled),
             "water_zero": potential_energy(backend, zero, pos, box, run)}
        # relative to the size of the energy: double precision, and the single precision of the plugin
        e["coupled_minus_original_less_the_correction"] = e["alchemical_coupled"] - (e["original"] - of_the_water)
        e["decoupled_minus_water_zero"] = e["alchemical_decoupled"] - e["water_zero"]
        e["level_relative"] = level
        e["passed"] = bool(max(abs(e["coupled_minus_original_less_the_correction"]),
                               abs(e["decoupled_minus_water_zero"])) < level * abs(e["original"]))
        out["passed"] = out["passed"] and e["passed"]
        out["backends"][name] = e
    return out


# ------------------------------------------------------------------------------------- the leg
def run_leg(alch, schedule, tail: dict, pos, box, leg_dir: Path, run: dict, args, devices: list[int],
            thread_percentage: int | None) -> None:
    """Hamiltonian replica exchange of the windows, as ``run_leg`` of the workflow runs a leg."""
    from cufep.sampling.hrex import ReplicaExchangeSampler
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import analysis_record, planned_iterations

    plan = planned_iterations(run["time_per_iteration_ps"], run["timestep_fs"], args.equilibration_ns,
                              args.production_ns)
    total = plan.n_equilibration + plan.n_production
    spec = LegSpec(name=leg_dir.parent.name, system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend="cupotential", precision="mixed", temperature_K=run["temperature_K"],
                   pressure_bar=run["pressure_bar"], barostat="isotropic",
                   barostat_frequency=run["barostat_frequency"], exchange=run["exchange"],
                   friction_per_ps=run["friction_per_ps"], timestep_fs=run["timestep_fs"],
                   steps_per_iteration=plan.steps_per_iteration, seed=LEG_SEED, lrc=tail,
                   mps_thread_percentage=thread_percentage)

    def on_iteration(sampler, _stats) -> None:
        if sampler.iteration % 100 == 0:
            log(f"[{spec.name}] iteration {sampler.iteration} of {total} "
                f"({plan.n_equilibration} of them equilibration)")

    with ReplicaExchangeSampler(spec, leg_dir / "hrex", devices=devices, log=log) as sampler:
        if not sampler.restore():
            sampler.initialize(pos, box)
        else:
            log(f"[{spec.name}] resumed at iteration {sampler.iteration}")
        sampler.storage.set_plan(plan.n_equilibration, total, analysis_record(AnalysisConfig()))
        if total > sampler.iteration:
            sampler.run(total - sampler.iteration, record_stride=1, checkpoint_stride=100, on_iteration=on_iteration)


def analyse_leg(water: str, leg_dir: Path, run: dict, check: dict) -> dict:
    """The result of the leg as the workflow makes it, and the numbers of this measurement from it."""
    from cufep.jsonio import clean, dumps
    from cufep.workflow import leg_result_from_samples

    record = {"ligand": f"one water ({short(water)})", "leg": "solvent", "repeat": 0, "backend": "cupotential",
              "timestep_fs": run["timestep_fs"], "notes": []}
    result = leg_result_from_samples(leg_dir, AnalysisConfig(), record)
    (leg_dir.parent / "leg_result.json").write_text(dumps(result, indent=1))
    a = result["analysis"]
    kt = R_GAS * result["temperature_K"]
    decouple, mbar, tail = result["delta_g_decouple"], result["delta_g_mbar"], result["lrc"]
    of_the_water = check["dispersion_correction_of_the_water_kJ"]

    def kcal(x):
        return None if x is None else float(x) / KCAL

    # the error of the MBAR part alone from time blocks: the largest over the numbers of blocks, which is
    # the rule of the error of dG(decouple) (LegAnalysis.delta_g_error_from)
    mbar_blocks = [v for v in (a.get("delta_g_mbar_error_blocks") or {}).values() if v is not None]
    out = {
        "water": water, "backend": result["backend"], "windows": result["n_states"],
        "lambda_windows": result["lambda_windows"], "production_ns": result["production_ns"],
        "iterations_stored": result["n_iterations_stored"],
        "equilibration_iterations": result["n_equilibration_iterations"],
        "kT_kcal": kt / KCAL,
        # the free energy of decoupling that a leg reports: MBAR minus the tail correction
        "dG_decouple_kcal": kcal(decouple), "dG_decouple_se_kcal": kcal(result["delta_g_error"]),
        "dG_error_from": a.get("delta_g_error_from"), "dG_error_blocks": a.get("delta_g_error_n_blocks"),
        "dG_mbar_kcal": kcal(mbar), "dG_mbar_se_kcal": kcal(result["delta_g_mbar_error"]),
        "dG_mbar_error_blocks_kcal": {k: kcal(v) for k, v in (a.get("delta_g_mbar_error_blocks") or {}).items()},
        "dG_mbar_se_time_blocks_kcal": kcal(max(mbar_blocks)) if mbar_blocks else None,
        "dG_error_blocks_kcal": {k: kcal(v) for k, v in (a.get("delta_g_error_blocks") or {}).items()},
        "tail_part_of_dG_kcal": kcal(-tail), "tail_se_kcal": kcal(result["lrc_error"]),
        "tail_reweight_kcal": kcal(result["lrc_reweight"]), "tail_isotropic_kcal": kcal(result["lrc_isotropic"]),
        "stage_dG_kcal": {k: kcal(v) for k, v in (a.get("stage_delta_g") or {}).items()},
        "bar_dG_kcal": kcal(a.get("bar_delta_g")), "first_half_kcal": [kcal(x) for x in a.get("first_half") or []],
        "second_half_kcal": [kcal(x) for x in a.get("second_half") or []],
        "lowest_overlap": a.get("min_overlap"), "lowest_overlap_of_the_samples_used": a.get("min_overlap_used"),
        "overlap_neighbour": a.get("overlap_neighbour"),
        "fewest_independent_samples": a.get("effective_samples_per_state"),
        "fewest_samples_given_to_mbar": min(a.get("window_kept_samples") or [0]),
        "round_trips": a.get("round_trips"), "not_determined": a.get("not_determined"),
        "quality": result.get("quality"),
        # mu_ex = - dG(decouple): with the tail correction of the analysis, without any correction
        # beyond the cutoff, and with the part of OpenMM's analytic dispersion correction of the water
        "mu_ex_with_tail_kcal": kcal(-decouple), "mu_ex_mbar_only_kcal": kcal(-mbar),
        "dispersion_correction_of_the_water_kcal": kcal(of_the_water),
        "volume_of_the_dispersion_correction_nm3": check["volume_nm3"],
        "mu_ex_with_the_analytic_correction_kcal": kcal(-mbar + of_the_water),
        "mu_ex_se_kcal": kcal(result["delta_g_error"]),
    }
    return clean(out)


# ----------------------------------------------------------------------------------- per model
def measure(water: str, args, run: dict, devices: list[int], thread_percentage: int | None) -> None:
    out = Path(args.out) / short(water)
    out.mkdir(parents=True, exist_ok=True)
    parts = ("volume", "mu") if args.part == "all" else (args.part,)
    system, topology, pos, box = build(water, args.box_nm)
    waters = list(topology.residues())
    n_sites = sum(1 for i in range(system.getNumParticles()) if system.isVirtualSite(i))
    log(f"\n== {water}: {len(waters)} waters, {system.getNumParticles()} particles, {n_sites} virtual sites, "
        f"box {box[0, 0]:.3f} nm as built")
    if args.part == "analysis":
        check = json.loads((out / "energy_check.json").read_text())
        (out / "chemical_potential.json").write_text(json.dumps(analyse_leg(water, out / "leg", run, check), indent=1))
        return
    start = out / "equilibrated.npz"
    earlier = start.exists()
    if earlier:
        # MD on a GPU is not the same to the bit in two runs, and a leg is continued only with the System
        # (its box) that its samples were made with: a later run in this directory starts where the first did
        with np.load(start) as data:
            pos, box = np.array(data["positions"]), np.array(data["box"])
    else:
        pos, box = equilibrate(system, pos, box, run)
        np.savez(start, positions=pos, box=box)
    log(f"   equilibrated{' (by an earlier run in this directory)' if earlier else ''}: box {box[0, 0]:.4f} nm, "
        f"{1000.0 * abs(np.linalg.det(box)) / len(waters):.3f} A^3 per water")

    if "volume" in parts:
        volumes = {}
        for backend_name in ("cupotential", "openmm"):
            volumes[backend_name] = v = measure_volume(system, len(waters), pos, box, backend_name, args.volume_ns,
                                                       run)
            log(f"   V° on {backend_name:11s} {v['v_A3']:.3f} +- {v['v_se_A3']:.3f} A^3 ({v['ns']:g} ns in "
                f"{v['blocks']} blocks), density {v['density_g_cm3']:.4f} +- {v['density_se_g_cm3']:.4f} g/cm3")
        (out / "volume.json").write_text(json.dumps(
            {"water": water, "waters": len(waters), "box_nm_built": args.box_nm, "settings": run,
             "equilibration_ns": VOLUME_EQUILIBRATION_NS, **volumes}, indent=1))

    if "mu" in parts:
        ligand = [a.index for a in waters[len(waters) // 2].atoms()]
        system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])       # as alchemical_leg does
        alch = make_alchemical_system(system, ligand, run["softcore_alpha"], softcore_b=run["softcore_b"],
                                      annihilate_electrostatics=run["annihilate_electrostatics"])
        schedule = fixed_schedule(["discharge", "vanish"], {"discharge": args.windows[0], "vanish": args.windows[1]})
        check = check_alchemical_system(system, alch, schedule, ligand, pos, box, run)
        (out / "energy_check.json").write_text(json.dumps({"water": water, "ligand_atoms": ligand, **check}, indent=1))
        log(f"   the water that is decoupled: atoms {ligand}, virtual sites {check['virtual_sites_in_the_ligand']}; "
            f"exceptions {check['exceptions_original']} -> {check['exceptions_alchemical']}")
        log(f"   analytic dispersion correction of the System {check['dispersion_correction_kJ']:.3f} kJ/mol, of it "
            f"the water's {check['dispersion_correction_of_the_water_kJ']:.4f} kJ/mol "
            f"({check['dispersion_correction_of_the_water_kJ'] / KCAL:.4f} kcal/mol) at {check['volume_nm3']:.3f} nm3")
        for name, e in check["backends"].items():
            log(f"   {name:13s} original {e['original']:.3f} kJ/mol; coupled - (original - the water's correction) "
                f"{e['coupled_minus_original_less_the_correction']:+.5f}; decoupled - (water zero) "
                f"{e['decoupled_minus_water_zero']:+.5f}; level {e['level_relative']:g} of the energy: "
                f"{'met' if e['passed'] else 'NOT MET'}")
        if not check["passed"]:
            log(f"   {water}: the alchemical System of one water does not have the energies of the plain Systems; "
                "this water model is not sampled")
            return
        tail = TailEnergy.from_system(system, ligand, alch.ligand_parameters, run["cutoff_nm"],
                                      run["lrc_cutoff_nm"]).to_dict()
        (out / "leg").mkdir(exist_ok=True)
        schedule.save(out / "leg" / "schedule.yaml")
        log(f"   {len(schedule.states)} windows (discharge {args.windows[0]}, vanish {args.windows[1]} intervals), "
            f"{args.equilibration_ns:g} ns + {args.production_ns:g} ns per window, {run['timestep_fs']:g} fs, "
            f"workers on devices {devices}")
        run_leg(alch, schedule, tail, pos, box, out / "leg", run, args, devices, thread_percentage)
        result = analyse_leg(water, out / "leg", run, check)
        (out / "chemical_potential.json").write_text(json.dumps(result, indent=1))


def table(args, waters) -> None:
    """The numbers of every water model that the output directory holds."""
    sphere_A3 = 4.0 / 3.0 * math.pi * (10.0 * SPHERE_RADIUS_NM) ** 3

    def read(water: str, name: str) -> dict | None:
        path = Path(args.out) / short(water) / name
        return json.loads(path.read_text()) if path.exists() else None

    def num(value, digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    def pm(value, error, digits: int = 3) -> str:
        return f"{num(value, digits)} +- {num(error, digits)}"

    log("\n== the numbers (kcal/mol, A^3, g/cm3); mu_ex and its parts from the cuPotential backend")
    for water in waters:
        vol, mu, pub = read(water, "volume.json"), read(water, "chemical_potential.json"), PUBLISHED.get(water, {})
        log(f"{short(water)}:")
        if vol:
            for name in ("cupotential", "openmm"):
                v = vol[name]
                log(f"   V° {name:11s} {pm(v['v_A3'], v['v_se_A3'])} A^3 ({vol['waters']} waters, {v['ns']:g} ns), "
                    f"density {pm(v['density_g_cm3'], v['density_se_g_cm3'], 4)}"
                    + (f"   [published V° {pub['v_A3']}]" if pub else ""))
        if mu:
            # the errors: of the sum as the analysis gives it, of the MBAR part from time blocks where it has them
            se, se_mbar = mu["dG_decouple_se_kcal"], mu["dG_mbar_se_time_blocks_kcal"] or mu["dG_mbar_se_kcal"]
            # OpenMM's correction goes as 1 / V: taken at the mean volume of the NPT run where there is one
            v_check = mu["volume_of_the_dispersion_correction_nm3"]
            v_mean = vol["cupotential"]["volume_nm3"] if vol else v_check
            of_the_water = mu["dispersion_correction_of_the_water_kcal"] * v_check / v_mean
            with_analytic = None if mu["mu_ex_mbar_only_kcal"] is None else mu["mu_ex_mbar_only_kcal"] + of_the_water
            log(f"   dG(decouple) {pm(mu['dG_decouple_kcal'], se)} = MBAR {pm(mu['dG_mbar_kcal'], se_mbar)}"
                f" + tail {pm(mu['tail_part_of_dG_kcal'], mu['tail_se_kcal'], 4)}  (error of the sum from "
                f"{mu['dG_error_from']}, {mu['dG_error_blocks']} blocks; "
                + ", ".join(f"{k} {num(v)}" for k, v in mu["stage_dG_kcal"].items()) + ")")
            halves = [(mu[key] or [None])[0] for key in ("first_half_kcal", "second_half_kcal")]
            log(f"   {mu['windows']} windows, {mu['production_ns']:g} ns per window; lowest overlap "
                f"{num(mu['lowest_overlap'])} (of the samples used {num(mu['lowest_overlap_of_the_samples_used'])}); "
                f"fewest independent samples in a window {num(mu['fewest_independent_samples'], 0)}; halves "
                f"{num(halves[0])} and {num(halves[1])}; BAR {num(mu['bar_dG_kcal'])}")
            log(f"   mu_ex with the tail of the analysis        {pm(mu['mu_ex_with_tail_kcal'], se)}"
                + (f"   [published {pub['mu_ex_kcal']}]" if pub else ""))
            log(f"   mu_ex, MBAR alone (nothing beyond 0.9 nm)  {pm(mu['mu_ex_mbar_only_kcal'], se_mbar)}")
            log(f"   mu_ex with OpenMM's analytic correction    {pm(with_analytic, se_mbar)}   (the water's part of "
                f"the correction {num(of_the_water, 4)} at {v_mean:.2f} nm3, "
                f"{num(mu['dispersion_correction_of_the_water_kcal'], 4)} at the {v_check:.2f} nm3 of the check; "
                f"the tail of the analysis in the homogeneous approximation {num(mu['tail_isotropic_kcal'], 4)}, "
                f"by reweighting {num(mu['tail_reweight_kcal'], 4)})")
            if vol and with_analytic is not None:
                v0 = vol["cupotential"]["v_A3"]
                for label, value in (("with the tail", mu["mu_ex_with_tail_kcal"]),
                                     ("MBAR alone", mu["mu_ex_mbar_only_kcal"]),
                                     ("analytic correction", with_analytic)):
                    log(f"   Adams B, sphere of {SPHERE_RADIUS_NM:g} nm, V° of the plugin, mu_ex {label:19s} "
                        f"{value / mu['kT_kcal'] + math.log(sphere_A3 / v0):.3f}")
        if pub and vol:
            kt = R_GAS * vol["settings"]["temperature_K"] / KCAL
            adams = pub["mu_ex_kcal"] / kt + math.log(sphere_A3 / pub["v_A3"])
            log(f"   Adams B of the published TIP3P values: {adams:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--water", action="append", default=None, help="water model file (repeatable; default both)")
    parser.add_argument("--production-ns", type=float, default=3.0, help="production per lambda window")
    parser.add_argument("--equilibration-ns", type=float, default=0.5, help="equilibration per lambda window")
    parser.add_argument("--volume-ns", type=float, default=2.0, help="NPT MD for V° on each backend, after 0.2 ns")
    parser.add_argument("--box-nm", type=float, default=3.5, help="edge of the cubic box as it is built")
    parser.add_argument("--windows", type=int, nargs=2, default=(8, 16), metavar=("DISCHARGE", "VANISH"),
                        help="intervals of the two stages")
    parser.add_argument("--out", default="spikes/out/m34")
    parser.add_argument("--gpus", type=int, default=None, help="number of GPUs inside the container (default all)")
    parser.add_argument("--workers-per-gpu", type=int, default=3,
                        help="workers of the leg on one GPU (more than 1 needs NVIDIA MPS, which is started)")
    parser.add_argument("--part", choices=("all", "volume", "mu", "analysis"), default="all",
                        help="volume: V° only; mu: the leg only; analysis: the stored samples of the leg")
    args = parser.parse_args()
    waters = tuple(args.water) if args.water else WATERS
    run = settings_of_a_run()

    from cufep import mps
    from cufep.scheduler import detect_gpus

    n_gpus, daemon, factor = 0, None, 1
    if args.part != "analysis":                       # the analysis needs no GPU
        n_gpus = args.gpus if args.gpus is not None else max(1, len(detect_gpus(max_utilization=101)))
    if args.part in ("all", "mu") and args.workers_per_gpu > 1:
        daemon = mps.start(log)                       # before the first CUDA Context of this process
        factor = args.workers_per_gpu if daemon is not None else 1
    devices = [g for g in range(n_gpus) for _ in range(factor)]
    try:
        for water in waters:
            measure(water, args, run, devices, mps.thread_percentage(factor))
    finally:
        if daemon is not None:
            daemon.stop()
    table(args, waters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
