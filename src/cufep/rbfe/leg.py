"""One leg of an edge: the hybrid System of its end states, sampled by Hamiltonian replica exchange.

A leg of the protocol ``hybrid`` (:mod:`cufep.engine.protocol`) on the engine of a leg (:mod:`cufep.engine.leg`):
the pre-equilibration of the hybrid System at lambda 0 (ligand A, the unique atoms of B dummies), the lambda
windows, the replica exchange (:func:`cufep.workflow.sample_leg`), the fallback to stock OpenMM, and the analysis
of the samples (:func:`cufep.engine.leg.path_result`): the free energy of transforming A into B in the leg.

The lambda windows (``lambda_schedule``)
    adaptive, ``legs: each`` (default): each leg runs its own precalculation of adaptive lambda scheduling, the
        windows of an iteration at the same time on the GPUs it is granted (as the ABFE).
    adaptive, ``legs: solvent``: the solvent leg runs the precalculation and the complex leg takes its windows, as
        Midgley et al. 2025 decide the windows of a relative transformation from the ligand in solution.
    fixed: the windows of ``windows``.
    Windows that a leg does not choose itself start, all of them, from the end of its pre-equilibration, minimised in
    their own state (OpenFE minimises every replica, 5000 steps) and 10 ps of MD in it, all windows at the same time
    on the GPUs of the leg; the equilibration of the replica exchange then relaxes them further.

Grand canonical moves of water (``sampling.gcmc``, on by default in the complex leg)
    As in the ABFE (:mod:`cufep.sampling.gcmc`): the waters farthest from the ligand are ghosts from the
    pre-equilibration on, which has its barostat; the alchemical run has none and stays in the mean cell of the
    pre-equilibration. The sphere is around the heavy atoms of the whole hybrid ligand (both end states, so that A ->
    B and B -> A have one sphere; cuFEP's own). In the stand-in energy of the trials the ligand interacts with the
    parameters of both ends at full strength in every window, and the test with the full potential of the window
    corrects it (the scheme of the module: what U' is changes how many trials pass, not what is sampled).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from cufep.engine.jobs import Job
from cufep.workflow import KCAL, WorkflowError, gcmc_in

#: the stage of the one coordinate of a relative transformation
STAGES = ["morph"]
#: how long the complex leg waits between two looks for the windows of the solvent leg
WAIT_SECONDS = 20.0
#: how the windows that a leg does not choose itself get the configurations of their replicas: from the end of the
#: pre-equilibration, minimised in their own state (OpenFE's 5000 steps of minimisation) and a short MD in it
PRE_EQUILIBRATION_START = "the end of the pre-equilibration, minimised in each window"
MINIMIZE_STEPS = 5000
START_MD_PS = 10.0
#: how the result of a leg tells the user to run it again
RBFE_COMMAND = "cufep rbfe run CONFIG --edge NAME"


@dataclass(kw_only=True)
class EdgeLegJob(Job):
    """A leg of an edge: ``edge_dir`` is the directory of the edge, ``config`` the configuration of the relative
    run."""

    edge: str
    edge_dir: str

    unit_key: ClassVar[str] = "edge"

    @property
    def name(self) -> str:
        return self.edge

    @property
    def unit_dir(self) -> Path:
        return Path(self.edge_dir)

    def runner(self):
        return run_hybrid_leg_safe


def run_hybrid_leg_safe(job: EdgeLegJob) -> dict:
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


def hybrid_topology(top_a, top_b, unique_b: list[int], alchemical_water=()):
    """The topology of the hybrid System: that of A, and the unique atoms of B in a residue LIG of a chain of their
    own (their bonds among themselves; those to the core are terms of the System). The residue of the alchemical
    water of an edge that changes the charge is named ``ALCHEMICAL_WATER_RESNAME`` (cufep.rbfe.charge), so that the
    grand canonical moves, which take the waters named HOH, leave it out."""
    import openmm.app as app

    from cufep.rbfe.charge import ALCHEMICAL_WATER_RESNAME

    alchemical = {int(i) for i in alchemical_water}
    out = app.Topology()
    out.setPeriodicBoxVectors(top_a.getPeriodicBoxVectors())
    atoms = {}
    for chain in top_a.chains():
        c = out.addChain(chain.id)
        for res in chain.residues():
            name = ALCHEMICAL_WATER_RESNAME if {a.index for a in res.atoms()} & alchemical else res.name
            r = out.addResidue(name, c, res.id, res.insertionCode)
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
    gcmc: object = None           # cufep.sampling.gcmc.GcmcSpec of a leg with grand canonical moves of water


def hybrid_gcmc(rcfg, hybrid, topology, box, end_heavy_nm=None, end_heavy_bonds=None):
    """The grand canonical moves of water of a hybrid System (all its waters interacting): their specification
    (:func:`cufep.sampling.gcmc.gcmc_spec`), with the ligand of the stand-in energy at fixed lambdas, its atoms with
    the parameters of A and those unique to B with the parameters of B (the module's docstring; the alchemical water
    enters it as the NonbondedForce has it, its charges of A and no Lennard-Jones). The test with the full potential
    corrects what the stand-in leaves out. The radius of the sphere is from the heavy atoms of both end states in
    their poses (``end_heavy_nm``), so that A -> B and B -> A have one sphere (review of PR #19: from the hybrid, which
    holds the mapped atoms where A has them), each molecule whole along its bonds (``end_heavy_bonds``)."""
    import dataclasses

    from cufep.sampling.gcmc import GcmcError, gcmc_spec
    from cufep.workflow import gcmc_water_values

    g = rcfg.sampling.gcmc
    mu, volume = gcmc_water_values(rcfg)
    try:
        spec = gcmc_spec(hybrid.system, topology, hybrid.positions, np.asarray(box, dtype=float),
                         list(hybrid.ligand_atoms), rcfg.dynamics.cutoff_nm, rcfg.alchemy.softcore_alpha, 1,
                         rcfg.thermodynamics.temperature_K, g.interval_ps, g.attempts, g.batch_size,
                         g.radius_margin_nm, mu, volume, None if g.ghost_waters == "auto" else int(g.ghost_waters),
                         radius_from_nm=end_heavy_nm, radius_from_bonds=end_heavy_bonds, bonds=hybrid.bonds)
    except GcmcError as exc:
        raise WorkflowError(f"sampling.gcmc cannot be used with this system: {exc}") from exc
    charge, sigma, epsilon = (np.array(v, dtype=float) for v in (spec.charge, spec.sigma, spec.epsilon))
    for h in hybrid.ligand_atoms:
        q, s, e = hybrid.parameters_a[h] if h in hybrid.parameters_a else hybrid.parameters_b[h]
        charge[h], sigma[h], epsilon[h] = q, s, e
    return dataclasses.replace(spec, charge=charge, sigma=sigma, epsilon=epsilon, stand_in_lambdas=(1.0, 1.0))


def molecules_of(system, topology, bonds) -> list[list[int]]:
    """The molecules of a hybrid System (atoms joined by bonds of the topology, ``bonds`` and constraints, a virtual
    site with its parents), for moving a configuration into another cell as a barostat moves it."""
    parent = list(range(system.getNumParticles()))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        parent[root(int(i))] = root(int(j))

    for a, b in topology.bonds():
        join(a.index, b.index)
    for a, b in bonds:
        join(a, b)
    for k in range(system.getNumConstraints()):
        i, j, _d = system.getConstraintParameters(k)
        join(i, j)
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            site = system.getVirtualSite(i)
            for k in range(site.getNumParticles()):
                join(i, site.getParticle(k))
    groups: dict[int, list[int]] = {}
    for i in range(len(parent)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def perturbed_region(edge_dir: Path, a: dict, b: dict, core_map: dict[int, int], unique_b: list[int]) -> list[int]:
    """The hot atoms of the hybrid ligand by the rule of :mod:`cufep.rbfe.hot_region`, from the two prepared ligands
    (their bonds and bond orders; atom i of a ligand is atom ``ligand[i]`` of its System)."""
    from openff.toolkit import Molecule

    from cufep.rbfe.hot_region import perturbed_hot_atoms

    prep = Path(edge_dir) / "prep"
    molecules = [Molecule.from_json((prep / f"ligand_{end}.json").read_text()).to_rdkit() for end in ("a", "b")]
    n_a = len(a["positions"])
    b_to_hybrid = {j: i for i, j in core_map.items()}
    b_to_hybrid.update({j: n_a + k for k, j in enumerate(unique_b)})
    to_a = {i: int(atom) for i, atom in enumerate(a["ligand"])}
    to_b = {i: b_to_hybrid[int(atom)] for i, atom in enumerate(b["ligand"])}
    return sorted(perturbed_hot_atoms(molecules[0], molecules[1], to_a, to_b, set(core_map)))


def end_states_heavy(a: dict, b: dict) -> tuple[list, list]:
    """The heavy atoms of the ligands of the two prepared end states, A then B (positions, nm), and the bonds between
    them (pairs of positions in that list): what the radius of the sphere of the moves is taken from, each molecule
    whole along its bonds (second review of PR #19)."""
    from cufep.sampling.relocation_mc import heavy_bonds

    points, bonds = [], []
    for end in (a, b):
        atoms = list(end["topology"].atoms())
        heavy = [i for i in end["ligand"] if atoms[i].element is not None and atoms[i].element.symbol != "H"]
        bonds += [(len(points) + int(i), len(points) + int(j)) for i, j in heavy_bonds(end["topology"], heavy)]
        points += [np.asarray(end["positions"], dtype=float)[i] for i in heavy]
    return points, bonds


def terms_left_unscaled_by_version_1(edge_dir: Path, leg: str, region: str) -> int | None:
    """The terms of B of the proper torsions of the core of the hybrid of a leg of a prepared edge that version 1 of
    the hybrid left unscaled with REST2 (:func:`cufep.rbfe.hybrid.terms_version_1_left_unscaled`), the hot region of
    the ligand by ``region``; None where the prepared systems of the edge cannot be read. They are terms of the two
    ligands and their mapping: counted on the end states of the solvent leg where the leg has its mapping."""
    import openmm as mm

    from cufep.rbfe.hybrid import make_hybrid_system, terms_version_1_left_unscaled
    from cufep.rbfe.prep import load_end_states

    try:
        info = json.loads((Path(edge_dir) / "prep" / "prep.json").read_text())
        of = "solvent" if info["legs"]["solvent"]["mapping"] == info["legs"][leg]["mapping"] else leg
        end = load_end_states(edge_dir, of)
        a, b = end["a"], end["b"]
        unique_b = sorted(j for j in b["ligand"] if j not in set(end["core_map"].values()))
        hot = perturbed_region(edge_dir, a, b, end["core_map"], unique_b) if region == "perturbed" else None
        water = (info["legs"][of].get("alchemical_water") or {}).get("atoms_a", [])
        hybrid = make_hybrid_system(mm.XmlSerializer.deserialize(a["system_xml"]), a["positions"], a["ligand"],
                                    mm.XmlSerializer.deserialize(b["system_xml"]), b["positions"], b["ligand"],
                                    end["core_map"], rest2=True, alchemical_water=water, rest2_ligand_atoms=hot)
    except (OSError, KeyError, ValueError):
        # a file of the prepared edge is missing or cannot be read: not known for this leg alone, and the analysis of
        # the other edges goes on (third review of PR #19: FileNotFoundError stopped `cufep rbfe analyze`)
        return None
    return terms_version_1_left_unscaled(hybrid.system)


def build_leg(rcfg, edge_dir: Path, leg: str) -> HybridLeg:
    """The hybrid System of a leg from its prepared end states, and a prepared system of it."""
    import openmm as mm

    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.prep import load_end_states
    from cufep.workflow import rest2_region

    end = load_end_states(edge_dir, leg)
    a, b, info = end["a"], end["b"], end["info"]
    rest2 = rcfg.sampling.rest2
    rest2_on = rest2.scale > 1.0
    hot_atoms, hot_residues = [], []
    if rest2_on and leg == "complex":
        hot_atoms, hot_residues = rest2_region(rest2, a["topology"], a["positions"], a["ligand"], a["protein"],
                                               info.get("input_chains"))
    water = (info["legs"][leg].get("alchemical_water") or {}).get("atoms_a", [])
    unique_b = sorted(j for j in b["ligand"] if j not in set(end["core_map"].values()))
    hot_ligand = perturbed_region(edge_dir, a, b, end["core_map"], unique_b) \
        if rest2_on and rest2.region == "perturbed" else None
    hybrid = make_hybrid_system(mm.XmlSerializer.deserialize(a["system_xml"]), a["positions"], a["ligand"],
                                mm.XmlSerializer.deserialize(b["system_xml"]), b["positions"], b["ligand"],
                                end["core_map"], softcore_alpha=rcfg.alchemy.softcore_alpha, rest2=rest2_on,
                                rest2_atoms=hot_atoms, alchemical_water=water, rest2_ligand_atoms=hot_ligand)
    topology = hybrid_topology(a["topology"], b["topology"], unique_b, water)
    gcmc = None
    if gcmc_in(rcfg, leg):
        from cufep.sampling.gcmc import ghosts_off

        gcmc = hybrid_gcmc(rcfg, hybrid, topology, a["box"], *end_states_heavy(a, b))
        # every MD of the leg is of the System with the ghosts of the start switched off
        ghosts_off(hybrid.system, gcmc.waters[gcmc.ghosts].reshape(-1))
    xml = mm.XmlSerializer.serialize(hybrid.system)
    prep = {"input_chains": info.get("input_chains"), "system_xml": xml, "topology": topology,
            "positions": hybrid.positions, "box": a["box"], "ligand": list(hybrid.ligand_atoms),
            "solute": sorted(set(a["solute"]) | set(hybrid.unique_new)), "protein": list(a["protein"])}
    leg_info = info["legs"][leg]
    return HybridLeg(hybrid=hybrid, xml=xml, prep=prep, hot_atoms=hot_atoms, hot_residues=hot_residues,
                     barostat=leg_info["barostat"], membrane_info=leg_info.get("membrane_info") or {}, info=info,
                     gcmc=gcmc)


#: settings added after the first relative calculations, with their defaults: a record of the samples or the windows
#: of a leg leaves them out at these values, so that the legs of earlier versions are taken up as they were
SETTINGS_ADDED = {"core_smarts": None}
#: settings of the preparation of an edge alone, never of the record of a leg (the prepared edge records them where
#: they act, cufep.rbfe.prep.edge_made_from; review of PR #19: the cutoff of the alchemical water entered the record of
#: the legs of every edge when it was not 0.8)
PREPARATION_ONLY = ("alchemical_water_cutoff_nm",)


def settings_as_recorded(section) -> dict:
    """A section of the configuration as the records of a leg hold it (:data:`SETTINGS_ADDED`,
    :data:`PREPARATION_ONLY`)."""
    return {k: v for k, v in section.model_dump(mode="json").items()
            if k not in PREPARATION_ONLY and (k not in SETTINGS_ADDED or v != SETTINGS_ADDED[k])}


def water_moves_record(gcmc: dict | None) -> dict:
    """What of the grand canonical moves of water of a leg its thermodynamic state is: whether it makes them, and the
    excess chemical potential and the standard volume of water they are made with (the Adams value of Ross et al.
    2017); the other settings of the moves change how they sample, not the ensemble."""
    if gcmc is None:
        return {"enabled": False}
    return {"enabled": True,
            "excess_chemical_potential_kcal_per_mol": float(gcmc["excess_chemical_potential_kcal_per_mol"]),
            "standard_volume_nm3": float(gcmc["standard_volume_nm3"])}


def hybrid_settings(rcfg, leg: str | None = None) -> dict:
    """The settings of a relative run that the samples of a leg depend on beyond its System; the grand canonical
    moves of water for the leg that makes them (written only there, so that the solvent leg and the legs of
    earlier versions keep the record they had)."""
    settings = {"rbfe": {"alchemy": settings_as_recorded(rcfg.alchemy),
                         "mapping": settings_as_recorded(rcfg.mapping)},
                "sampling.rest2": rcfg.sampling.rest2.model_dump(mode="json")}
    if gcmc_in(rcfg, leg):
        from cufep.workflow import gcmc_record

        settings["gcmc"] = gcmc_record(rcfg)
    return settings


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


def run_hybrid_leg(job: EdgeLegJob) -> dict:
    """Run (or resume, or extend) one leg of an edge and return its result."""
    from cufep.alchemy.schedule import AlsSettings, schedule_from_windows
    from cufep.engine.leg import (
        adaptive_windows,
        check_equil_ghosts,
        check_kept_windows,
        chosen_backend,
        finish_leg,
        goes_on_from_its_samples,
        kept_start_record,
        leg_context,
        open_leg,
        pre_equilibrated,
        record_of_kept_windows,
        refuse_samples_without_windows,
        starts_from,
        with_fallback,
    )
    from cufep.engine.protocol import HYBRID, stage_state
    from cufep.mps import thread_percentage
    from cufep.rbfe.config import RbfeConfig
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import (
        _precision,
        _schedule_of_this_leg,
        planned_iterations,
        publish_windows,
        sample_leg,
        stored_windows,
    )

    rcfg = RbfeConfig.model_validate(job.config)
    edge_dir = Path(job.edge_dir)
    built = build_leg(rcfg, edge_dir, job.leg)
    # a pre-equilibration this leg cannot use is refused before --reschedule sets anything aside (third review of
    # PR #19: the leg was refused after its result had been moved, and the command it was told to run could not go on)
    check_equil_ghosts(job.leg_dir / "equil.npz", built.gcmc)
    run = open_leg(job, rcfg)
    leg_dir, log, seed = run.leg_dir, run.log, run.seed
    hybrid, prep, barostat = built.hybrid, built.prep, built.barostat
    n_atoms = hybrid.system.getNumParticles()
    is_complex = job.leg == "complex"
    T = rcfg.thermodynamics.temperature_K
    rest2 = rcfg.sampling.rest2
    rest2_scale = rest2.scale if rest2.scale > 1.0 else 1.0
    log(f"hybrid: {len(hybrid.core)} core atoms, {len(hybrid.unique_old)} unique to "
        f"{built.info['ligand_a']}, {len(hybrid.unique_new)} unique to {built.info['ligand_b']}; "
        + (f"REST2 on {len(hybrid.rest2_ligand_atoms)} of the {len(hybrid.ligand_atoms)} atoms of the ligand "
           f"(sampling.rest2.region: {rest2.region}) and {len(built.hot_atoms)} atoms of {len(built.hot_residues)} "
           f"residues ({' '.join(built.hot_residues)}), {rest2_scale:g} times the temperature in the middle"
           if rest2_scale > 1.0 else "REST2 off")
        + ("" if not hybrid.alchemical_water else
           f"; the net charge changes: water {hybrid.alchemical_water[0]} becomes "
           f"{built.info['legs'][job.leg]['alchemical_water']['ion']['name']} (an alchemical water)"))
    gcmc = built.gcmc
    if gcmc is not None:
        mu = gcmc.excess_chemical_potential_kj_per_mol / KCAL
        log(f"grand canonical moves of water: sphere of {10 * gcmc.radius_nm:.1f} A around the hybrid ligand, Adams "
            f"value B = {gcmc.adams:.3f} (excess chemical potential {mu:.3f} kcal/mol, standard volume "
            f"{1000 * gcmc.standard_volume_nm3:.3f} A^3), {len(gcmc.ghosts)} of the "
            f"{len(gcmc.waters)} waters are ghosts at the start (those farthest from the ligand); the alchemical run "
            "has no barostat")

    # ---- 1. pre-equilibration of the hybrid System at lambda 0 ---------------------------------
    # the ghosts of a leg of an earlier version are not known (its sphere was of the hybrid, 2ecbcd2 of both end
    # states, then of each molecule whole): such a pre-equilibration is refused
    eq, _recorded = pre_equilibrated(run, prep, hybrid.system, barostat, built.membrane_info, gcmc)
    timestep_fs = float(eq["timestep_fs"])
    start = (np.asarray(eq["positions"]), np.asarray(eq["box"]), np.asarray(eq["velocities"]))
    cell = None
    if gcmc is not None:
        # the cell is fixed from here on (cufep.workflow.alchemical_box), the configuration moved into it as a
        # barostat moves it
        from cufep.workflow import alchemical_box, positions_in_cell

        cell = alchemical_box(eq, gcmc)
        start = (positions_in_cell(start[0], start[1], cell,
                                   molecules_of(hybrid.system, prep["topology"], hybrid.bonds)), cell, start[2])

    # ---- 2. backend -------------------------------------------------------------------------------
    backend_name = chosen_backend(run, hybrid.system)
    iteration_ps = rcfg.sampling.time_per_iteration_ps
    plan = planned_iterations(iteration_ps, timestep_fs, rcfg.sampling.equilibration(job.leg), job.production_ns)
    n_equil, n_prod = plan.n_equilibration, plan.n_production
    ckpt_stride = max(1, int(round(rcfg.sampling.checkpoint_interval_ps / iteration_ps)))
    atoms = list(prep["topology"].atoms())
    frame_atoms = prep["solute"] if is_complex else list(hybrid.ligand_atoms)
    where = {a: i for i, a in enumerate(frame_atoms)}
    frame_ligand = [where[i] for i in hybrid.atoms_a if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    frame_ca = [where[i] for i in prep["protein"] if atoms[i].name == "CA" and i in where] if is_complex else []
    # the settings the samples and the windows of the leg depend on beyond its System
    settings = hybrid_settings(rcfg, job.leg)
    ends = [stage_state(STAGES[0], 0.0).to_dict(), stage_state(STAGES[-1], 1.0).to_dict()]

    def spec_of(states: list[dict], backend: str, **more) -> LegSpec:
        return LegSpec(name=job.tag, system_xml=built.xml, states=states, n_atoms=n_atoms, protocol=HYBRID,
                       backend=backend, precision=_precision(rcfg, backend), platform=rcfg.backend.platform,
                       temperature_K=T, pressure_bar=None if gcmc is not None else rcfg.thermodynamics.pressure_bar,
                       barostat="none" if gcmc is not None else barostat,
                       barostat_frequency=rcfg.dynamics.barostat_frequency, exchange=rcfg.sampling.exchange,
                       friction_per_ps=rcfg.dynamics.friction_per_ps, timestep_fs=timestep_fs,
                       steps_per_iteration=plan.steps_per_iteration, cpu_threads=rcfg.resources.cpu_threads_per_worker,
                       box_safety=1.3 if barostat == "membrane" and gcmc is None else None, settings=settings,
                       mps_thread_percentage=thread_percentage(job.oversubscription), gcmc=gcmc, **more)

    def from_pre_equilibration(schedule) -> list:
        """The start of every window that the leg does not choose itself: the end of the pre-equilibration,
        minimised in the state of the window (OpenFE's minimisation, ``MINIMIZE_STEPS``) and ``START_MD_PS`` of MD
        in it (:func:`cufep.engine.leg.starts_from`). The atoms unique to the other ligand do not interact at lambda
        0, so water can sit where they are: a window in which they interact starts from that only after its
        minimisation. The minimisation and this MD run on stock OpenMM, as the minimisation, NVT and NPT of every
        pre-equilibration do (cufep.sampling.equilibrate): OpenMM's minimiser moves to a Context of the CPU platform
        where the forces are very large, and the cuPotential plugin has no kernels there (the ejm_31 -> ejm_49
        complex leg of the second run of the TYK2 example failed so). Only the start configurations come from it;
        the samples are of the backend of the leg."""
        return starts_from(run, spec_of(ends, "openmm", seed=seed + 29), schedule, start, hybrid.system, n_atoms,
                           MINIMIZE_STEPS, START_MD_PS)

    def check_scale(schedule) -> None:
        if abs(schedule.rest2_scale - rest2_scale) > 1e-9:
            raise WorkflowError("the lambda schedule of this leg was made with a REST2 scale of "
                                f"{schedule.rest2_scale:g} but sampling.rest2.scale is {rest2.scale:g}")

    def run_with(backend: str):
        ls = rcfg.lambda_schedule
        # ---- 3. lambda windows ---------------------------------------------------------------
        stored = stored_windows(leg_dir, job.leg, STAGES, settings, log, cell)
        if stored is not None:
            schedule, configs, _made = stored
            log(f"lambda schedule ({schedule.mode}): the {len(schedule.states)} windows this leg already has")
        else:
            t0 = time.perf_counter()
            a = ls.adaptive
            starting = {"start_configurations": PRE_EQUILIBRATION_START, "start_md_ps": START_MD_PS,
                        "minimize_steps": MINIMIZE_STEPS, "start_backend": "openmm"}
            if (leg_dir / "schedule.yaml").exists():
                # the windows this leg has, without start configurations of them (none, or of other windows): its
                # samples are of these windows, so they are kept and only their start configurations are made
                # again, whatever lambda_schedule asks now (review of PR #18). A schedule that is not one of this
                # leg stops the run here, before anything of the leg is written
                schedule = _schedule_of_this_leg(leg_dir / "schedule.yaml", job.leg, STAGES)
                check_scale(schedule)
                production = spec_of([st.to_dict() for st in schedule.states], backend, seed=seed + 17,
                                     frame_atoms=frame_atoms, frame_ligand=frame_ligand, frame_ca=frame_ca)
                check_kept_windows(leg_dir, schedule, production.conditions())
                record_of_kept_windows(leg_dir, schedule, settings, log)   # the settings the windows were chosen with
                log(f"lambda schedule ({schedule.mode}): the {len(schedule.states)} windows this leg has, without "
                    "the start configurations of them; they are made again for these windows (the settings of "
                    "lambda_schedule are not applied to them; cufep rbfe run --reschedule determines the windows "
                    "again)")
                cfgs = from_pre_equilibration(schedule)
                # samples taken up as they were made keep their start (unknown without a record of these windows),
                # and how the file was made now is said apart; samples started anew start from these (reviews of
                # PR #18)
                how = kept_start_record(leg_dir, schedule, settings, {**starting, "windows": "those this leg had"},
                                        log)
            elif goes_on_from_its_samples(leg_dir):
                refuse_samples_without_windows(leg_dir)
            elif ls.mode == "fixed":
                values = ls.windows
                schedule = schedule_from_windows(STAGES, {"morph": values},
                                                 {"morph": values if isinstance(values, int) else len(values)},
                                                 rest2_scale)
                cfgs = from_pre_equilibration(schedule)
                how = {**starting, "windows": "given"}
            elif a.legs == "each" or job.leg == "solvent":
                als = AlsSettings(overlap_threshold=a.overlap_threshold, overlap_matrix=a.overlap_matrix,
                                  candidates=a.candidates, initial_windows={"morph": a.initial_windows},
                                  window_ps=a.window(job.leg), equilibration_ps=a.equilibration(job.leg),
                                  sample_interval_ps=a.sample_interval_ps, start_configurations=a.start_configurations,
                                  walk_ps=a.walk_ps, max_windows=a.max_windows, rest2_scale=float(rest2_scale))
                pool_spec = spec_of(ends, backend, seed=seed + 23)
                # the Context of this process, which walks through the windows of the first iteration
                ctx, integ = leg_context(pool_spec, hybrid.system, start[1], start[0], start[2], run.device0, seed)

                def inline_mover():
                    """The moves of water of the windows that run in this process, on its Context."""
                    if gcmc is None:
                        return None, 0
                    from cufep.sampling.gcmc import GcmcMover
                    from cufep.sampling.worker import GCMC_STREAM, stream_seed
                    from cufep.steps import md_steps

                    return (GcmcMover(gcmc, ctx, stream_seed(seed + 23, 0, 0, GCMC_STREAM)),
                            md_steps(gcmc.interval_ps, timestep_fs))

                schedule, cfgs = adaptive_windows(run, pool_spec, ctx, integ, STAGES, als, n_atoms, inline_mover)
                del ctx, integ
                how = {"start_configurations": als.start_configurations,
                       **({"walk_ps": float(als.walk_ps)} if als.start_configurations == "walk" else {}),
                       "windows": "chosen by this precalculation"}
            else:
                schedule = _schedule_from_solvent(rcfg, edge_dir, job.repeat, log, job.control)
                cfgs = from_pre_equilibration(schedule)
                how = {**starting, "windows": "those of the solvent leg"}
            record = {**how, "settings": settings, "states": [s.to_dict() for s in schedule.states]}
            configs = publish_windows(leg_dir, schedule, cfgs, record, leg=job.leg, edge=job.edge,
                                      repeat=job.repeat, seed=seed, backend=backend)
            run.timed("lambda_schedule", t0)
            log(f"lambda schedule ({schedule.mode}): {len(schedule.states)} windows in {time.perf_counter() - t0:.0f} s"
                f" ({how['windows']})")
        check_scale(schedule)
        # ---- 4. replica exchange ----------------------------------------------------------------
        spec = spec_of([s.to_dict() for s in schedule.states], backend, seed=seed + 17, frame_atoms=frame_atoms,
                       frame_ligand=frame_ligand, frame_ca=frame_ca)
        return sample_leg(spec, leg_dir / "hrex", job, configs, n_equil, n_equil + n_prod, rcfg, log, run.timing,
                          run.timed, ckpt_stride), schedule

    (samples, schedule), backend_name = with_fallback(run, backend_name, run_with)

    # ---- 5. analysis ---------------------------------------------------------------------------------
    t_analysis = time.perf_counter()
    from cufep.rbfe.hybrid import HYBRID_VERSION
    from cufep.workflow import record_digest

    run_record = {"edge": job.edge, "leg": job.leg, "repeat": job.repeat, "backend": backend_name,
                  "timestep_fs": timestep_fs, "notes": list(run.notes), "origin_not_recorded": [],
                  # which hybrid System the leg ran (cufep.rbfe.run.legs_of_version_1; second review of PR #19)
                  "hybrid_version": HYBRID_VERSION,
                  # the prepared systems of the edge, which the legs of one ddG must share (edge_result)
                  "prepared_systems_sha256": record_digest(built.info.get("made_from")),
                  # the thermodynamic state of the grand canonical moves of water, which the complex legs of one network
                  # must share (cufep.rbfe.run.water_moves_of; review of PR #19)
                  "grand_canonical": water_moves_record(settings.get("gcmc")),
                  "rest2": {"scale": float(rest2_scale), "hot_protein_atoms": len(built.hot_atoms),
                            "hot_residues": built.hot_residues, "torsion_terms_scaled": hybrid.rest2_torsions,
                            "scale_of_window": [float(st.rest2_scale) for st in schedule.states]}}
    result = hybrid_leg_result(leg_dir, rcfg, run_record, built, samples=samples, schedule=schedule)
    run.timed("analysis", t_analysis)
    finish_leg(run, result, backend_name, n_atoms, {"edge": job.edge})
    a = result["analysis"]
    log(f"dG(A -> B, {job.leg}) = {result['delta_g'] / KCAL:.2f} ± {result['delta_g_error'] / KCAL:.2f} kcal/mol; "
        f"min overlap {a['min_overlap']:.3f}, fewest independent samples in a window "
        f"{a['effective_samples_per_state']:.0f}, round trips {a['round_trips']}")
    return result


def hybrid_leg_result(leg_dir: Path, analysis, run_record: dict, built: HybridLeg | None = None, samples=None,
                      schedule=None) -> dict:
    """The result of a leg of an edge: the analysis of its samples (:func:`cufep.engine.leg.path_result`), whose
    free energy along the path (``delta_g``) is that of transforming A into B in the leg, and what the hybrid
    topology is made of."""
    from cufep.engine.leg import path_result
    from cufep.workflow import gcmc_numbers

    result, of = path_result(leg_dir, analysis, run_record, {"edge": run_record.get("edge")}, samples=samples,
                             schedule=schedule, command=RBFE_COMMAND)
    # the grand canonical moves of water of the production that is analysed, by lambda window (None: none)
    result["gcmc"] = gcmc_numbers(of["samples"], of["first"])
    if built is not None:
        h = built.hybrid
        result["hybrid"] = {"ligand_a": built.info["ligand_a"], "ligand_b": built.info["ligand_b"],
                            "core_atoms": len(h.core), "unique_old_atoms": len(h.unique_old),
                            "unique_new_atoms": len(h.unique_new), "lomap_score": built.info.get("lomap_score")}
    return result
