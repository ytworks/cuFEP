"""One leg of an edge: the hybrid System of its end states, sampled by Hamiltonian replica exchange.

The steps of an ABFE leg (``cufep.workflow.run_leg``) without the restraint, and with the functions it shares:
the pre-equilibration of the hybrid System at lambda 0 (ligand A, the unique atoms of B dummies), the lambda
windows, the replica exchange (:func:`cufep.workflow.sample_leg`), the fallback to stock OpenMM, and the
analysis of the leg (:func:`cufep.workflow.leg_result_from_samples`).

The lambda windows (``lambda_schedule``)
    adaptive, ``legs: each`` (default): each leg runs its own precalculation of adaptive lambda scheduling, the
        windows of an iteration at the same time on the GPUs it is granted (as the ABFE).
    adaptive, ``legs: solvent``: the solvent leg runs the precalculation and the complex leg takes its windows, as
        Midgley et al. 2025 decide the windows of a relative transformation from the ligand in solution.
    fixed: the windows of ``windows``.
    Windows that a leg does not choose itself start, all of them, from the end of its pre-equilibration, minimised in
    their own state (OpenFE minimises every replica, 5000 steps) and 10 ps of MD in it, all windows at the same time
    on the GPUs of the leg; the equilibration of the replica exchange then relaxes them further.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cufep.workflow import KCAL, LegJob, WorkflowError

#: the stage of the one coordinate of a relative transformation
STAGES = ["morph"]
#: how long the complex leg waits between two looks for the windows of the solvent leg
WAIT_SECONDS = 20.0
#: how the windows that a leg does not choose itself get the configurations of their replicas: from the end of the
#: pre-equilibration, minimised in their own state (OpenFE's 5000 steps of minimisation) and a short MD in it
PRE_EQUILIBRATION_START = "the end of the pre-equilibration, minimised in each window"
MINIMIZE_STEPS = 5000
START_MD_PS = 10.0


@dataclass
class HybridLegJob(LegJob):
    """A leg of an edge: ``ligand`` is the name of the edge, ``lig_dir`` its directory, ``config`` the
    configuration of the relative run."""

    @property
    def runner(self):
        return run_hybrid_leg_safe


def run_hybrid_leg_safe(job: HybridLegJob) -> dict:
    """:func:`run_hybrid_leg` that records a failure instead of raising (for the job processes)."""
    import traceback

    from cufep.workflow import begin_leg, write_leg_error

    begin_leg(job.leg_dir, job)
    try:
        return run_hybrid_leg(job)
    except Exception as exc:  # noqa: BLE001
        err = write_leg_error(job.leg_dir, f"{type(exc).__name__}: {exc}", job, traceback.format_exc(limit=12))
        print(f"[{job.tag}] FAILED: {err['error']}", flush=True)
        return err


def hybrid_topology(top_a, top_b, unique_b: list[int]):
    """The topology of the hybrid System: that of A, and the unique atoms of B in a residue LIG of a chain of their
    own (their bonds among themselves; those to the core are terms of the System)."""
    import openmm.app as app

    out = app.Topology()
    out.setPeriodicBoxVectors(top_a.getPeriodicBoxVectors())
    atoms = {}
    for chain in top_a.chains():
        c = out.addChain(chain.id)
        for res in chain.residues():
            r = out.addResidue(res.name, c, res.id, res.insertionCode)
            for atom in res.atoms():
                atoms[("a", atom.index)] = out.addAtom(atom.name, atom.element, r, atom.id)
    for bond in top_a.bonds():
        out.addBond(atoms[("a", bond[0].index)], atoms[("a", bond[1].index)])
    if unique_b:
        b_atoms = list(top_b.atoms())
        r = out.addResidue("LIG", out.addChain("X"))
        for j in unique_b:
            atoms[("b", j)] = out.addAtom(b_atoms[j].name + "B", b_atoms[j].element, r)
        wanted = set(unique_b)
        for bond in top_b.bonds():
            i, j = bond[0].index, bond[1].index
            if i in wanted and j in wanted:
                out.addBond(atoms[("b", i)], atoms[("b", j)])
    return out


@dataclass
class HybridLeg:
    hybrid: object                # cufep.rbfe.hybrid.HybridSystem
    xml: str
    prep: dict                    # what the shared functions read of a prepared system (cufep.workflow.load_prepared)
    hot_atoms: list
    hot_residues: list
    barostat: str
    membrane_info: dict
    info: dict


def build_leg(rcfg, cfg, edge_dir: Path, leg: str) -> HybridLeg:
    """The hybrid System of a leg from its prepared end states, and a prepared system of it."""
    import openmm as mm

    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.prep import load_end_states
    from cufep.workflow import rest2_region

    end = load_end_states(edge_dir, leg)
    a, b, info = end["a"], end["b"], end["info"]
    rest2 = cfg.sampling.rest2
    rest2_on = rest2.scale > 1.0
    hot_atoms, hot_residues = [], []
    if rest2_on and leg == "complex":
        hot_atoms, hot_residues = rest2_region(rest2, a["topology"], a["positions"], a["ligand"], a["protein"],
                                               info.get("input_chains"))
    hybrid = make_hybrid_system(mm.XmlSerializer.deserialize(a["system_xml"]), a["positions"], a["ligand"],
                                mm.XmlSerializer.deserialize(b["system_xml"]), b["positions"], b["ligand"],
                                end["core_map"], softcore_alpha=rcfg.alchemy.softcore_alpha, rest2=rest2_on,
                                rest2_atoms=hot_atoms)
    unique_b = sorted(j for j in b["ligand"] if j not in set(end["core_map"].values()))
    topology = hybrid_topology(a["topology"], b["topology"], unique_b)
    xml = mm.XmlSerializer.serialize(hybrid.system)
    prep = {"input_chains": info.get("input_chains"), "system_xml": xml, "topology": topology,
            "positions": hybrid.positions, "box": a["box"], "ligand": list(hybrid.ligand_atoms),
            "solute": sorted(set(a["solute"]) | set(hybrid.unique_new)), "protein": list(a["protein"])}
    leg_info = info["legs"][leg]
    return HybridLeg(hybrid=hybrid, xml=xml, prep=prep, hot_atoms=hot_atoms, hot_residues=hot_residues,
                     barostat=leg_info["barostat"], membrane_info=leg_info.get("membrane_info") or {}, info=info)


def hybrid_settings(rcfg) -> dict:
    """The settings of a relative run that the samples of a leg depend on beyond its System."""
    return {"rbfe": {"alchemy": rcfg.alchemy.model_dump(mode="json"), "mapping": rcfg.mapping.model_dump(mode="json")},
            "sampling.rest2": rcfg.sampling.rest2.model_dump(mode="json")}


def _schedule_from_solvent(rcfg, edge_dir: Path, repeat: int, log, control=None):
    """The windows of the solvent leg of the edge, when it has them; waits for them."""
    from cufep.alchemy.schedule import Schedule

    solvent = Path(edge_dir) / f"repeat_{repeat}" / "solvent"
    told = False
    while True:
        if (solvent / "schedule.yaml").exists() and (solvent / "start_configs.npz").exists():
            schedule = Schedule.load(solvent / "schedule.yaml")
            schedule.notes.append("the windows of the solvent leg (lambda_schedule.adaptive.legs: solvent)")
            return schedule
        if (solvent / "error.json").exists():
            raise WorkflowError("the solvent leg of this edge failed before it had its windows "
                                f"({solvent / 'error.json'})")
        if not told:
            log("waiting for the lambda windows of the solvent leg (lambda_schedule.adaptive.legs: solvent)")
            told = True
            if control is not None:
                control.serial()
        time.sleep(WAIT_SECONDS)


def run_hybrid_leg(job: HybridLegJob) -> dict:
    """Run (or resume, or extend) one leg of an edge and return its result."""
    import openmm as mm
    import openmm.unit as u

    from cufep.alchemy.schedule import AlsSettings, Schedule, adaptive_schedule, schedule_from_windows
    from cufep.mps import thread_percentage
    from cufep.rbfe.config import RbfeConfig
    from cufep.sampling.hrex import WindowPool
    from cufep.sampling.storage import stored_backend, stored_conditions
    from cufep.sampling.worker import LegSpec, backend_of
    from cufep.workflow import (
        _choose_backend,
        _Log,
        _precision,
        archive_leg,
        backend_fallback_refusal,
        check_equil_file,
        leg_seed,
        leg_with_samples_stays_on_its_backend,
        md_work,
        planned_iterations,
        publish_windows,
        run_pre_equilibration,
        sample_leg,
        set_aside_failed_backend,
        write_leg_result,
    )

    rcfg = RbfeConfig.model_validate(job.config)
    cfg = rcfg                                  # the sections the engine of a leg reads (CommonConfig)
    leg_dir = job.leg_dir
    leg_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(leg_dir / "log.txt", prefix=f"[{job.tag}] ")
    if job.reschedule:
        previous = archive_leg(leg_dir)
        if previous is not None:
            log(f"the lambda windows are determined again; what this leg had is in {previous.name}/")
    edge_dir = Path(job.lig_dir)
    built = build_leg(rcfg, cfg, edge_dir, job.leg)
    hybrid, prep, barostat = built.hybrid, built.prep, built.barostat
    is_complex = job.leg == "complex"
    in_membrane = is_complex and barostat == "membrane"
    T = cfg.thermodynamics.temperature_K
    seed = leg_seed(cfg, job.repeat, job.leg)
    device0 = job.devices[0] if job.devices else None
    rest2 = cfg.sampling.rest2
    rest2_scale = rest2.scale if rest2.scale > 1.0 else 1.0
    notes: list[str] = []
    timing: dict = {"phases": {}, "started": time.time()}

    def timed(name: str, t_start: float) -> None:
        timing["phases"][name] = timing["phases"].get(name, 0.0) + (time.perf_counter() - t_start)

    log(f"hybrid: {len(hybrid.core)} core atoms, {len(hybrid.unique_old)} unique to "
        f"{built.info['ligand_a']}, {len(hybrid.unique_new)} unique to {built.info['ligand_b']}; "
        + (f"REST2 on the ligand and {len(built.hot_atoms)} atoms of {len(built.hot_residues)} residues "
           f"({' '.join(built.hot_residues)}), {rest2_scale:g} times the temperature in the middle"
           if rest2_scale > 1.0 else "REST2 off"))

    # ---- 1. pre-equilibration of the hybrid System at lambda 0 ---------------------------------
    equil_file = leg_dir / "equil.npz"
    if not equil_file.exists():
        t0 = time.perf_counter()
        run_pre_equilibration(cfg, job.leg, prep, hybrid.system, barostat, built.membrane_info, seed, equil_file,
                              device0, log)
        timed("pre_equilibration", t0)
    eq = np.load(equil_file)
    check_equil_file(eq, equil_file, cfg, job.leg, in_membrane, seed, log)
    timestep_fs = float(eq["timestep_fs"])

    # ---- 2. backend -------------------------------------------------------------------------------
    backend_name, bnotes = _choose_backend(cfg, hybrid.system, log)
    notes += bnotes
    made_with = stored_backend(leg_dir / "hrex")
    if made_with is not None and made_with != backend_name:
        log(f"this leg goes on with {made_with}, the backend its samples were made with")
        notes.append(f"the leg was continued on {made_with}, the backend its samples were made with")
        backend_name = made_with
    pressure = cfg.thermodynamics.pressure_bar
    iteration_ps = cfg.sampling.time_per_iteration_ps
    plan = planned_iterations(iteration_ps, timestep_fs, cfg.sampling.equilibration(job.leg), job.production_ns)
    n_equil, n_prod = plan.n_equilibration, plan.n_production
    ckpt_stride = max(1, int(round(cfg.sampling.checkpoint_interval_ps / iteration_ps)))
    atoms = list(prep["topology"].atoms())
    frame_atoms = prep["solute"] if is_complex else list(hybrid.ligand_atoms)
    where = {a: i for i, a in enumerate(frame_atoms)}
    frame_ligand = [where[i] for i in hybrid.atoms_a if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    frame_ca = [where[i] for i in prep["protein"] if atoms[i].name == "CA" and i in where] if is_complex else []
    settings = hybrid_settings(rcfg)

    def spec_of(states: list[dict], backend: str, **more) -> LegSpec:
        return LegSpec(name=job.tag, system_xml=built.xml, states=states, n_atoms=hybrid.system.getNumParticles(),
                       backend=backend, precision=_precision(cfg, backend), platform=cfg.backend.platform,
                       temperature_K=T, pressure_bar=pressure, barostat=barostat,
                       barostat_frequency=cfg.dynamics.barostat_frequency, exchange=cfg.sampling.exchange,
                       friction_per_ps=cfg.dynamics.friction_per_ps, timestep_fs=timestep_fs,
                       steps_per_iteration=plan.steps_per_iteration, cpu_threads=cfg.resources.cpu_threads_per_worker,
                       box_safety=1.3 if barostat == "membrane" else None, settings=settings, protocol=HYBRID,
                       mps_thread_percentage=thread_percentage(job.oversubscription), **more)

    from cufep.engine.protocol import HYBRID, stage_state

    pool_spec_states = [stage_state("morph", 0.0).to_dict(), stage_state("morph", 1.0).to_dict()]

    def from_pre_equilibration(schedule) -> list:
        """The start of every window: the end of the pre-equilibration, minimised in the state of the window
        (OpenFE's minimisation, ``MINIMIZE_STEPS``) and ``START_MD_PS`` of MD in it, all windows at the same time on
        the GPUs the leg is granted (in this process one after another where it has no workers). The atoms unique
        to the other ligand do not interact at lambda 0, so water can sit where they are: a window in which they
        interact starts from that only after its minimisation. The minimisation and this MD run on stock OpenMM, as
        the minimisation, NVT and NPT of every pre-equilibration do (cufep.sampling.equilibrate): OpenMM's
        minimiser moves to a Context of the CPU platform where the forces are very large, and the cuPotential
        plugin has no kernels there (the ejm_31 -> ejm_49 complex leg of the second run of the TYK2 example
        failed so). Only the start configurations come from it; the samples are of the backend of the leg."""
        from cufep.alchemy.schedule import run_window
        from cufep.steps import md_steps

        start = (np.asarray(eq["positions"]), np.asarray(eq["box"]), np.asarray(eq["velocities"]))
        tasks = [{"state": state.to_dict(), "config": start, "stages": [], "candidates": 2, "temperature_K": T,
                  "equilibration_steps": md_steps(START_MD_PS, timestep_fs), "n_samples": 0, "steps_per_sample": 1,
                  "minimize_steps": MINIMIZE_STEPS, "protocol": HYBRID} for state in schedule.states]
        start_spec = spec_of(pool_spec_states, "openmm", seed=seed + 29)
        devices = list(job.devices)
        if job.control is not None:
            work = md_work(START_MD_PS, hybrid.system.getNumParticles(), iteration_ps)
            granted = job.control.request_devices(
                len(tasks), len(tasks) * work / max(1, job.oversubscription), pinned=True,
                what=f"the start of {len(tasks)} lambda windows")
            devices = granted or devices
        if not job.inline and len(devices) >= 2:
            with WindowPool(start_spec, devices[:len(tasks)], log) as pool:
                done = pool.run(tasks)
        else:
            ctx, integ = precalculation_context(start_spec)
            done = [run_window(ctx, integ, task) for task in tasks]
            del ctx, integ
        return [d["config"] for d in done]

    def precalculation_context(pool_spec):
        """A Context of this process for the windows: the backend of the spec, its barostat, the end of the
        pre-equilibration."""
        back = backend_of(pool_spec)
        prepared = back.prepare_system(hybrid.system)
        if barostat == "membrane":
            prepared.addForce(back.make_membrane_barostat(pressure * u.bar, T * u.kelvin,
                                                          cfg.dynamics.barostat_frequency, seed=seed + 5))
        else:
            prepared.addForce(back.make_isotropic_barostat(pressure * u.bar, T * u.kelvin,
                                                           cfg.dynamics.barostat_frequency, seed=seed + 5))
        integ = back.make_integrator(T * u.kelvin, cfg.dynamics.friction_per_ps / u.picosecond,
                                     timestep_fs * u.femtoseconds, seed=seed + 6)
        ctx = back.make_context(prepared, integ, device0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
        ctx.setPositions(eq["positions"])
        ctx.setVelocities(eq["velocities"])
        return ctx, integ

    def run_with(backend: str):

        hrex_dir = leg_dir / "hrex"
        sched_file, start_file = leg_dir / "schedule.yaml", leg_dir / "start_configs.npz"
        ls = rcfg.lambda_schedule
        # ---- 3. lambda windows ---------------------------------------------------------------
        if sched_file.exists() and start_file.exists():
            schedule = Schedule.load(sched_file)
            with np.load(start_file) as start:
                configs = (start["positions"], start["boxes"], start["velocities"])
            if len(configs[0]) != len(schedule.states):
                raise WorkflowError(f"{start_file} does not hold one start configuration per window of {sched_file}")
            log(f"lambda schedule ({schedule.mode}): the {len(schedule.states)} windows this leg already has")
        else:
            t0 = time.perf_counter()
            pool_spec = spec_of(pool_spec_states, backend, seed=seed + 23)
            a = ls.adaptive
            own_precalculation = ls.mode == "adaptive" and (a.legs == "each" or job.leg == "solvent")
            ctx = integ = None
            if own_precalculation:
                # the Context of this process, which walks through the windows of the first iteration
                ctx, integ = precalculation_context(pool_spec)
            if ls.mode == "fixed":
                values = ls.windows
                schedule = schedule_from_windows(STAGES, {"morph": values},
                                                 {"morph": values if isinstance(values, int) else len(values)},
                                                 rest2_scale)
                cfgs = from_pre_equilibration(schedule)
                how = {"start_configurations": PRE_EQUILIBRATION_START, "start_md_ps": START_MD_PS,
                       "minimize_steps": MINIMIZE_STEPS, "start_backend": "openmm",
                       "windows": "given"}
            elif own_precalculation:
                als = AlsSettings(overlap_threshold=a.overlap_threshold, overlap_matrix=a.overlap_matrix,
                                  candidates=a.candidates, initial_windows={"morph": a.initial_windows},
                                  window_ps=a.window(job.leg), equilibration_ps=a.equilibration(job.leg),
                                  sample_interval_ps=a.sample_interval_ps, start_configurations=a.start_configurations,
                                  walk_ps=a.walk_ps, max_windows=a.max_windows, rest2_scale=float(rest2_scale))
                from cufep.alchemy.candidates import initial_window_indices

                n_first = len(initial_window_indices(a.initial_windows, a.candidates))
                devices = list(job.devices)
                if job.control is not None:
                    work = md_work(a.window(job.leg) + a.equilibration(job.leg), hybrid.system.getNumParticles(),
                                   iteration_ps)
                    granted = job.control.request_devices(
                        n_first, n_first * work / max(1, job.oversubscription), pinned=True,
                        what=f"the precalculation of the lambda schedule ({n_first} lambda windows)")
                    devices = granted or devices
                if job.inline or len(devices) < 2:
                    schedule, cfgs = adaptive_schedule(ctx, integ, STAGES, T, timestep_fs, als, log)
                else:
                    with WindowPool(pool_spec, devices[:n_first], log) as pool:
                        schedule, cfgs = adaptive_schedule(ctx, integ, STAGES, T, timestep_fs, als, log, pool=pool)
                how = {"start_configurations": als.start_configurations,
                       **({"walk_ps": float(als.walk_ps)} if als.start_configurations == "walk" else {}),
                       "windows": "chosen by this precalculation"}
            else:
                schedule = _schedule_from_solvent(rcfg, edge_dir, job.repeat, log, job.control)
                cfgs = from_pre_equilibration(schedule)
                how = {"start_configurations": PRE_EQUILIBRATION_START, "start_md_ps": START_MD_PS,
                       "minimize_steps": MINIMIZE_STEPS, "start_backend": "openmm",
                       "windows": "those of the solvent leg"}
            del ctx, integ
            record = {**how, "settings": settings, "states": [s.to_dict() for s in schedule.states]}
            configs = publish_windows(leg_dir, schedule, cfgs, record, leg=job.leg, edge=job.ligand,
                                      repeat=job.repeat, seed=seed, backend=backend)
            timed("lambda_schedule", t0)
            log(f"lambda schedule ({schedule.mode}): {len(schedule.states)} windows in {time.perf_counter() - t0:.0f} s"
                f" ({how['windows']})")
        if abs(schedule.rest2_scale - rest2_scale) > 1e-9:
            raise WorkflowError("the lambda schedule of this leg was made with a REST2 scale of "
                                f"{schedule.rest2_scale:g} but sampling.rest2.scale is {rest2.scale:g}")
        # ---- 4. replica exchange ----------------------------------------------------------------
        spec = spec_of([s.to_dict() for s in schedule.states], backend, seed=seed + 17, frame_atoms=frame_atoms,
                       frame_ligand=frame_ligand, frame_ca=frame_ca)
        return sample_leg(spec, hrex_dir, job, configs, n_equil, n_equil + n_prod, cfg, log, timing, timed,
                          ckpt_stride), schedule

    had_samples = stored_conditions(leg_dir / "hrex")[0] > 0
    try:
        samples, schedule = run_with(backend_name)
    except Exception as exc:  # noqa: BLE001  classified here; everything that is not repeated is raised
        refusal = backend_fallback_refusal(exc, backend_name, cfg.backend.fallback, had_samples)
        if refusal is not None:
            if refusal == "the leg had samples from an earlier run":
                raise WorkflowError(leg_with_samples_stays_on_its_backend(exc)) from exc
            raise
        kept = set_aside_failed_backend(leg_dir)
        log(f"cuPotential run failed ({str(exc)[:300]}); repeating this leg on stock OpenMM; what the failed run "
            f"left is kept in {kept.name}/")
        notes.append("the cuPotential run failed; the whole leg was repeated on stock OpenMM "
                     f"(the failed run is kept in {kept.name}/)")
        backend_name = "openmm"
        samples, schedule = run_with(backend_name)

    # ---- 5. analysis ---------------------------------------------------------------------------------
    t_analysis = time.perf_counter()
    run_record = {"ligand": job.ligand, "leg": job.leg, "repeat": job.repeat, "backend": backend_name,
                  "timestep_fs": timestep_fs, "notes": list(notes), "origin_not_recorded": [],
                  "rest2": {"scale": float(rest2_scale), "hot_protein_atoms": len(built.hot_atoms),
                            "hot_residues": built.hot_residues, "torsion_terms_scaled": hybrid.rest2_torsions,
                            "scale_of_window": [float(st.rest2_scale) for st in schedule.states]}}
    result = hybrid_leg_result(leg_dir, cfg, run_record, built, samples=samples, schedule=schedule)
    timed("analysis", t_analysis)
    timing.update(finished=time.time(), backend=backend_name, atoms=hybrid.system.getNumParticles(),
                  ligand=job.ligand, edge=job.ligand, leg=job.leg, repeat=job.repeat)
    from cufep.files import publish_text

    publish_text(leg_dir / "timing.json", json.dumps(timing, indent=1))
    write_leg_result(leg_dir, result)
    a = result["analysis"]
    log(f"dG(A -> B, {job.leg}) = {result['delta_g'] / KCAL:.2f} ± {result['delta_g_error'] / KCAL:.2f} kcal/mol; "
        f"min overlap {a['min_overlap']:.3f}, fewest independent samples in a window "
        f"{a['effective_samples_per_state']:.0f}, round trips {a['round_trips']}")
    return result


def hybrid_leg_result(leg_dir: Path, cfg, run_record: dict, built: HybridLeg | None = None, samples=None,
                      schedule=None) -> dict:
    """The result of a leg of an edge: that of an ABFE leg (:func:`cufep.workflow.leg_result_from_samples`, the
    same analysis), with the free energy named for what it is (A -> B) and without the terms of a restraint."""
    from cufep.workflow import leg_result_from_samples

    result = leg_result_from_samples(leg_dir, cfg, run_record, samples=samples, schedule=schedule)
    result["delta_g"] = result.pop("delta_g_decouple")
    for key in ("restraint_choice", "restraint_release", "restraint_release_analytical", "annihilate_electrostatics",
                "ligand_symmetry_number", "relocation_mc", "gcmc"):
        result.pop(key, None)
    result["edge"] = run_record.get("ligand")
    result["notes"] = [str(n).replace("cufep run CONFIG --ligand NAME", "cufep rbfe run CONFIG --edge NAME")
                       for n in result.get("notes", [])]
    if built is not None:
        h = built.hybrid
        result["hybrid"] = {"ligand_a": built.info["ligand_a"], "ligand_b": built.info["ligand_b"],
                            "core_atoms": len(h.core), "unique_old_atoms": len(h.unique_old),
                            "unique_new_atoms": len(h.unique_new), "lomap_score": built.info.get("lomap_score")}
    return result
