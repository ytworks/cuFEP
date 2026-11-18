"""The ABFE workflow: system preparation, one leg of the cycle, and the final combination.

Directory layout under ``output.dir``::

    resolved_config.yaml, auto_report.md
    <ligand>/prep/                    prepared, solvated systems shared by all repeats
    <ligand>/repeat_<n>/<leg>/        equil.npz, restraint.json, schedule.yaml, hrex/, result.json
    <ligand>/result.json              binding free energy over the repeats

Thermodynamic cycle (double decoupling)::

    dG_bind = dG_decouple(solvent) - dG_decouple(complex, incl. restraint on) - dG_release

where ``dG_release`` (negative) is the free energy of releasing the Boresch restraint on
the decoupled ligand into the standard-state volume.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cufep.config.schema import Config

KCAL = 4.184
R_GAS = 8.31446261815324e-3


class WorkflowError(RuntimeError):
    """A step of the workflow cannot continue; the message says what to change."""


class _Log:
    def __init__(self, path: Path | None, prefix: str = "") -> None:
        self.path, self.prefix = path, prefix
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        if self.prefix and message.startswith(self.prefix):
            message = message[len(self.prefix):]
        line = f"{time.strftime('%H:%M:%S')} {self.prefix}{message}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name.strip()) or "ligand"


# =============================================================================== preparation
POSE_CONTACT_NM = 0.45
POSE_MIN_CONTACT_FRACTION = 0.5
POSE_CLASH_NM = 0.20
WATER_CLASH_NM = 0.22
METALS = {"Zn", "Mg", "Mn", "Fe", "Cu", "Co", "Ni", "Ca", "Cd", "Hg"}


def check_ligand(ligand, protein_topology, protein_positions_nm: np.ndarray) -> list[str]:
    """Stop on inputs that would silently give a wrong answer; return warnings otherwise."""
    from cufep.prep.ligand import LigandError
    from cufep.prep.protein import WATER

    if ligand.formal_charge != 0:
        raise LigandError(
            f"ligand '{ligand.name}' has formal charge {ligand.formal_charge:+d}. Charged ligands need a "
            "finite-size/co-alchemical ion treatment that cuFEP 0.1 does not have. Provide a neutral "
            "protonation state, or wait for charged-ligand support.")
    heavy = np.array([a.index for a in protein_topology.atoms()
                      if a.element is not None and a.element.symbol != "H" and a.residue.name not in WATER])
    lig = ligand.positions_nm()[ligand.heavy_atom_indices()]
    d = np.linalg.norm(lig[:, None, :] - protein_positions_nm[heavy][None, :, :], axis=2)
    nearest = d.min(axis=1)
    contact = float((nearest < POSE_CONTACT_NM).mean())
    if contact < POSE_MIN_CONTACT_FRACTION:
        raise LigandError(
            f"ligand '{ligand.name}' does not sit in a pocket of the protein: only {100 * contact:.0f}% of its "
            f"heavy atoms are within {10 * POSE_CONTACT_NM:.1f} A of the protein. The SDF must contain the bound "
            "pose in the coordinate frame of the PDB.")
    if nearest.min() < POSE_CLASH_NM:
        i, j = np.unravel_index(np.argmin(d), d.shape)
        atom = list(protein_topology.atoms())[int(heavy[j])]
        raise LigandError(
            f"ligand '{ligand.name}' is {10 * nearest.min():.2f} A from {atom.residue.name}{atom.residue.id}:"
            f"{atom.name}. That is a severe clash or a covalent bond; neither is supported.")
    warnings = []
    for a in protein_topology.atoms():
        if a.element is not None and a.element.symbol in METALS:
            dm = float(np.linalg.norm(lig - protein_positions_nm[a.index], axis=1).min())
            if dm < 0.30:
                raise LigandError(
                    f"ligand '{ligand.name}' is {10 * dm:.1f} A from the metal ion {a.residue.name}{a.residue.id}. "
                    "Metal coordination is not described by the default force fields; cuFEP stops here.")
            if dm < 0.6:
                warnings.append(f"metal ion {a.residue.name}{a.residue.id} is {10 * dm:.1f} A from the ligand")
    return warnings


def _remove_clashing_waters(modeller, ligand_xyz_nm: np.ndarray) -> int:
    import openmm.unit as unit

    from cufep.prep.protein import WATER

    pos = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    doomed = []
    for res in modeller.topology.residues():
        if res.name not in WATER:
            continue
        o = next((a for a in res.atoms() if a.element is not None and a.element.symbol == "O"), None)
        if o is not None and np.linalg.norm(ligand_xyz_nm - pos[o.index], axis=1).min() < WATER_CLASH_NM:
            doomed.append(res)
    if doomed:
        modeller.delete(doomed)
    return len(doomed)


def _write_system(prep_dir: Path, name: str, modeller, system, ligand_atoms: list[int]) -> None:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from cufep.prep.ligand import LIGAND_RESNAME
    from cufep.prep.protein import AMINO_ACIDS, CAPS
    from cufep.prep.topology_io import topology_to_dict

    top = modeller.topology
    pos = np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=float)
    box = np.array([[float(v) for v in vec.value_in_unit(unit.nanometer)] for vec in top.getPeriodicBoxVectors()])
    protein = [a.index for a in top.atoms() if a.residue.name in AMINO_ACIDS or a.residue.name in CAPS]
    solute = sorted(set(protein) | set(ligand_atoms))
    (prep_dir / f"{name}_system.xml").write_text(mm.XmlSerializer.serialize(system))
    (prep_dir / f"{name}_topology.json").write_text(json.dumps(topology_to_dict(top)))
    np.savez(prep_dir / f"{name}.npz", positions=pos, box=box, ligand=np.array(ligand_atoms, dtype=int),
             solute=np.array(solute, dtype=int), protein=np.array(protein, dtype=int))
    with open(prep_dir / f"{name}.pdb", "w") as fh:
        app.PDBFile.writeFile(top, modeller.positions, fh, keepIds=True)
    _ = LIGAND_RESNAME


def load_prepared(prep_dir: Path, name: str) -> dict:
    from cufep.prep.topology_io import topology_from_dict

    data = np.load(prep_dir / f"{name}.npz")
    return {"system_xml": (prep_dir / f"{name}_system.xml").read_text(),
            "topology": topology_from_dict(json.loads((prep_dir / f"{name}_topology.json").read_text())),
            "positions": data["positions"], "box": data["box"], "ligand": [int(i) for i in data["ligand"]],
            "solute": [int(i) for i in data["solute"]], "protein": [int(i) for i in data["protein"]]}


def prepare_ligand(cfg: Config, ligand, protein, lig_dir: Path, membrane: bool, box_shape: str, log=print) -> dict:
    """Build and save the solvated complex and the solvated ligand for one ligand."""
    from cufep.backends.cupotential import MIN_BOX_CUTOFF_RATIO
    from cufep.prep.forcefield import ForceFieldSpec, SystemSpec, create_system, make_forcefield
    from cufep.prep.ligand import assign_partial_charges
    from cufep.prep.solvate import SolvationSpec, combine, ligand_atom_indices, solvate

    prep_dir = lig_dir / "prep"
    done = prep_dir / "prep.json"
    if done.exists():
        return json.loads(done.read_text())
    prep_dir.mkdir(parents=True, exist_ok=True)
    warnings = check_ligand(ligand, protein.topology, protein.positions_nm())
    t0 = time.perf_counter()
    assign_partial_charges(ligand, cfg.forcefield.ligand_charges, cache_dir=cfg.output.dir / "cache")
    log(f"{ligand.name}: partial charges ({cfg.forcefield.ligand_charges}) in {time.perf_counter() - t0:.0f} s")
    ffspec = ForceFieldSpec(protein=cfg.forcefield.protein, water=cfg.forcefield.water,
                            lipid=cfg.forcefield.lipid, ligand=cfg.forcefield.ligand)
    forcefield = make_forcefield(ffspec, [ligand], include_lipid=membrane)
    sysspec = SystemSpec(cutoff_nm=cfg.dynamics.cutoff_nm, hydrogen_mass_amu=cfg.dynamics.hmr_mass_amu,
                         constraints="hbonds" if cfg.dynamics.constraints in ("auto", "hbonds") else "none")
    # The cuPotential plugin needs every box height >= 3 x cutoff, also after the box has
    # relaxed under NPT; keep an 8 % margin. Stock OpenMM has no such rule.
    plugin = cfg.backend.name == "cupotential"
    min_edge = 1.08 * MIN_BOX_CUTOFF_RATIO * cfg.dynamics.cutoff_nm if plugin else 0.0
    # For the small ligand-only box that rule decides the size, and a cube is then the
    # smallest valid box (a dodecahedron's shortest height is only 71 % of its edge).
    solvent_shape = "cube" if plugin else box_shape
    ionic = cfg.thermodynamics.ionic_strength_M

    # ---- complex ------------------------------------------------------------------------
    modeller = combine((protein.topology, protein.positions), ligand)
    n_removed = _remove_clashing_waters(modeller, ligand.positions_nm()[ligand.heavy_atom_indices()])
    if n_removed:
        warnings.append(f"{n_removed} water(s) overlapping the ligand were removed")
    if membrane:
        from cufep.prep.membrane import add_membrane
        add_membrane(modeller, forcefield, cfg, log)
        barostat = "membrane"
    else:
        solvate(modeller, forcefield, SolvationSpec(padding_nm=cfg.solvation.padding_complex_nm, box_shape=box_shape,
                                                    ionic_strength_M=ionic, min_edge_nm=min_edge))
        barostat = "isotropic"
    system = create_system(forcefield, modeller.topology, sysspec)
    lig_idx = ligand_atom_indices(modeller.topology)
    _write_system(prep_dir, "complex", modeller, system, lig_idx)
    n_complex = system.getNumParticles()

    # ---- ligand in solvent -----------------------------------------------------------------
    modeller = combine(None, ligand)
    solvate(modeller, forcefield, SolvationSpec(padding_nm=cfg.solvation.padding_solvent_nm, box_shape=solvent_shape,
                                                ionic_strength_M=ionic, min_edge_nm=min_edge))
    system = create_system(forcefield, modeller.topology, sysspec)
    _write_system(prep_dir, "solvent", modeller, system, ligand_atom_indices(modeller.topology))
    (prep_dir / "ligand.json").write_text(ligand.offmol.to_json())
    info = {"ligand": ligand.name, "n_atoms_complex": n_complex, "n_atoms_solvent": system.getNumParticles(),
            "complex_barostat": barostat, "membrane": membrane, "box_shape": box_shape,
            "solvent_box_shape": solvent_shape, "warnings": warnings}
    done.write_text(json.dumps(info, indent=1))
    log(f"{ligand.name}: complex {n_complex} atoms, solvent leg {system.getNumParticles()} atoms"
        + (f"; warnings: {'; '.join(warnings)}" if warnings else ""))
    return info


# =============================================================================== one leg
@dataclass
class LegJob:
    ligand: str
    leg: str                       # complex | solvent
    repeat: int
    lig_dir: str
    devices: list
    config: dict
    production_ns: float
    reschedule: bool = False
    inline: bool = False

    @property
    def leg_dir(self) -> Path:
        return Path(self.lig_dir) / f"repeat_{self.repeat}" / self.leg

    @property
    def tag(self) -> str:
        return f"{self.ligand}/r{self.repeat}/{self.leg}"


def _choose_backend(cfg: Config, system, log) -> tuple[str, list[str]]:
    """Backend for a leg: the configured one if it can run this System, else the fallback."""
    notes: list[str] = []
    name = cfg.backend.name
    if name == "cupotential":
        from cufep.backends.cupotential import CuPotentialBackend
        ok, why = CuPotentialBackend.available() if cfg.backend.platform == "CUDA" else (False, "platform is not CUDA")
        if ok:
            try:
                CuPotentialBackend.check_system(system)
            except Exception as exc:  # noqa: BLE001
                ok, why = False, str(exc)
        if not ok:
            if cfg.backend.fallback == "none":
                raise WorkflowError(f"the cuPotential backend cannot run this leg ({why}) and no fallback is allowed")
            notes.append(f"cuPotential not used: {why}; running on stock OpenMM")
            log(notes[-1])
            name = "openmm"
    return name, notes


def _precision(cfg: Config, backend: str) -> str:
    if backend == "cupotential" and cfg.backend.precision == "double":
        return "mixed"
    return cfg.backend.precision


def run_leg(job: LegJob) -> dict:
    """Run (or resume, or extend) one leg and return its result dictionary."""
    import openmm as mm
    import openmm.unit as u

    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.lrc import TailEnergy
    from cufep.alchemy.restraints import (BoreschRestraint, RestraintSearchSettings, find_boresch_restraint,
                                          ligand_symmetry_number)
    from cufep.alchemy.schedule import Schedule, ScheduleSettings, refine, trailblaze
    from cufep.analysis.mbar import analyze_leg
    from cufep.backends import get_backend
    from cufep.sampling.equilibrate import EquilibrationSettings, pre_equilibrate
    from cufep.sampling.hrex import ReplicaExchangeSampler, SamplerError
    from cufep.sampling.states import STAGES_COMPLEX, STAGES_SOLVENT
    from cufep.sampling.storage import read_samples
    from cufep.sampling.worker import LegSpec

    cfg = Config.model_validate(job.config)
    leg_dir = job.leg_dir
    leg_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(leg_dir / "log.txt", prefix=f"[{job.tag}] ")
    is_complex = job.leg == "complex"
    lig_dir = Path(job.lig_dir)
    prep_info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    prep = load_prepared(lig_dir / "prep", job.leg)
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    T = cfg.thermodynamics.temperature_K
    seed = int(cfg.sampling.seed) + 7919 * job.repeat + (0 if is_complex else 104729)
    device0 = job.devices[0] if job.devices else None
    barostat = prep_info["complex_barostat"] if is_complex else "isotropic"
    notes: list[str] = []
    timestep_cfg = None if cfg.dynamics.timestep_fs == "auto" else float(cfg.dynamics.timestep_fs)

    # ---- 1. pre-equilibration of the fully interacting system (stock OpenMM) -------------
    equil_file = leg_dir / "equil.npz"
    if not equil_file.exists():
        s = cfg.sampling
        settings = EquilibrationSettings(
            temperature_K=T, pressure_bar=cfg.thermodynamics.pressure_bar, friction_per_ps=cfg.dynamics.friction_per_ps,
            timestep_fs=timestep_cfg, nvt_ns=s.nvt_ns if is_complex else s.solvent_nvt_ns,
            npt_ns=s.npt_ns if is_complex else s.solvent_npt_ns,
            production_ns=s.pre_equilibration_ns if is_complex else s.solvent_pre_equilibration_ns,
            barostat=barostat, platform=cfg.backend.platform,
            precision="mixed" if cfg.backend.precision == "single" else cfg.backend.precision, seed=seed)
        t0 = time.perf_counter()
        eq = pre_equilibrate(system, prep["positions"], prep["box"], prep["solute"], settings, device0, log)
        tmp = leg_dir / "equil.tmp.npz"
        np.savez(tmp, positions=eq.positions, velocities=eq.velocities, box=eq.box, frames=eq.frames,
                 frame_boxes=eq.frame_boxes, frame_atoms=np.array(eq.frame_atoms), timestep_fs=eq.timestep_fs,
                 notes=json.dumps(eq.notes))
        tmp.replace(equil_file)
        log(f"pre-equilibration done in {time.perf_counter() - t0:.0f} s (time step {eq.timestep_fs:g} fs)")
    eq = np.load(equil_file)
    timestep_fs = float(eq["timestep_fs"])
    notes += json.loads(str(eq["notes"]))

    # ---- 2. restraint (complex only) --------------------------------------------------------
    restraint = None
    release_dg = release_dg_analytical = 0.0
    symmetry = 1
    if is_complex:
        from openff.toolkit import Molecule
        rdmol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
        symmetry = ligand_symmetry_number(rdmol)
        rfile = leg_dir / "restraint.json"
        if rfile.exists():
            restraint = BoreschRestraint.from_dict(json.loads(rfile.read_text())["restraint"])
        else:
            rc = cfg.restraints
            k_r, k_a = rc.k_r_kcal_per_A2 * KCAL * 100.0, rc.k_angle_kcal_per_rad2 * KCAL
            if rc.selection == "manual":
                if not (rc.host_atoms and rc.guest_atoms):
                    raise WorkflowError("restraints.selection is 'manual' but host_atoms/guest_atoms are not given")
                last = np.zeros((system.getNumParticles(), 3))
                last[eq["frame_atoms"]] = eq["frames"][-1]
                v = BoreschRestraint(tuple(rc.host_atoms), tuple(rc.guest_atoms), 0, 0, 0, 0, 0, 0).values(last)
                restraint = BoreschRestraint(tuple(rc.host_atoms), tuple(rc.guest_atoms), *[float(x) for x in v],
                                             k_r, k_a, k_a, k_a, k_a, k_a)
                info = {"selection": "manual"}
            else:
                full = np.zeros((eq["frames"].shape[0], system.getNumParticles(), 3))
                full[:, eq["frame_atoms"], :] = eq["frames"]
                settings = RestraintSearchSettings(k_r=k_r, k_angle=k_a, rmsf_cutoff_nm=rc.rmsf_cutoff_nm,
                                                   host_min_distance_nm=rc.host_min_distance_nm,
                                                   host_max_distance_nm=rc.host_max_distance_nm,
                                                   dssp_filter=rc.dssp_filter)
                found = find_boresch_restraint(prep["topology"], full, prep["ligand"], rdmol, T, settings)
                restraint = found.restraint
                info = {"selection": "auto", "score": found.score, "fluctuations": found.fluctuations,
                        "ligand_rmsd_A": 10 * found.ligand_rmsd_nm, "candidates_tested": found.n_candidates,
                        "notes": found.notes}
                atoms = list(prep["topology"].atoms())
                label = lambda i: f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}"   # noqa: E731
                log("restraint: host " + ", ".join(label(i) for i in restraint.host_atoms) + " | guest "
                    + ", ".join(label(i) for i in restraint.guest_atoms)
                    + f" | r0 {10 * restraint.r0:.2f} A, ligand RMSD {10 * found.ligand_rmsd_nm:.2f} A")
            rfile.write_text(json.dumps({"restraint": restraint.to_dict(), "info": info,
                                         "ligand_symmetry_number": symmetry}, indent=1))
        release_dg = restraint.release_free_energy(T, "numerical")
        release_dg_analytical = restraint.release_free_energy(T, "analytical")
        if symmetry > 1:
            notes.append(f"the ligand has {symmetry} heavy-atom graph symmetries; no symmetry correction is applied")

    # ---- 3. alchemical System ------------------------------------------------------------------
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])      # the equilibrated box
    alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha)
    if restraint is not None:
        alch.system.addForce(restraint.create_force())
    alch_xml = mm.XmlSerializer.serialize(alch.system)
    tail = None
    if cfg.analysis.lrc == "reweight":
        tail = TailEnergy.from_system(system, prep["ligand"], alch.ligand_parameters, cfg.dynamics.cutoff_nm,
                                      cfg.analysis.lrc_cutoff_nm).to_dict()
    backend_name, bnotes = _choose_backend(cfg, alch.system, log)
    notes += bnotes
    stages = list(STAGES_COMPLEX if is_complex else STAGES_SOLVENT)
    max_windows = cfg.lambda_schedule.max_windows.complex if is_complex else cfg.lambda_schedule.max_windows.solvent
    pressure = cfg.thermodynamics.pressure_bar
    steps_per_iteration = max(1, int(round(cfg.sampling.iteration_ps * 1000.0 / timestep_fs)))
    n_equil = max(1, int(round(cfg.sampling.equilibration_ns * 1000.0 / cfg.sampling.iteration_ps)))
    n_prod = max(4, int(round(job.production_ns * 1000.0 / cfg.sampling.iteration_ps)))
    ckpt_stride = max(1, int(round(cfg.sampling.checkpoint_interval_ps / cfg.sampling.iteration_ps)))

    def run_with(backend_name: str) -> tuple[dict, Schedule]:
        hrex_dir = leg_dir / "hrex"
        sched_file = leg_dir / "schedule.yaml"
        start_file = leg_dir / "start_configs.npz"
        # ---- 4. lambda schedule ----------------------------------------------------------------
        if job.reschedule and not (hrex_dir / "samples.h5").exists():
            sched_file.unlink(missing_ok=True)
        if cfg.lambda_schedule.mode == "fixed" and cfg.lambda_schedule.file is not None and not sched_file.exists():
            shutil.copy(cfg.lambda_schedule.file, sched_file)
        if sched_file.exists() and start_file.exists():
            schedule = Schedule.load(sched_file)
            start = np.load(start_file)
            configs = (start["positions"], start["boxes"], start["velocities"])
        else:
            backend = get_backend(backend_name, precision=_precision(cfg, backend_name), platform=cfg.backend.platform)
            prepared = backend.prepare_system(alch.system)
            if barostat == "membrane":
                prepared.addForce(backend.make_membrane_barostat(pressure * u.bar, T * u.kelvin, 25, seed=seed + 5))
            else:
                prepared.addForce(backend.make_isotropic_barostat(pressure * u.bar, T * u.kelvin, 25, seed=seed + 5))
            integ = backend.make_integrator(T * u.kelvin, cfg.dynamics.friction_per_ps / u.picosecond,
                                            timestep_fs * u.femtoseconds, seed=seed + 6)
            ctx = backend.make_context(prepared, integ, device0)
            ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
            ctx.setPositions(eq["positions"])
            ctx.setVelocities(eq["velocities"])
            t0 = time.perf_counter()
            if sched_file.exists():                        # fixed schedule: only make start configurations
                schedule = Schedule.load(sched_file)
                cfgs = _walk_schedule(ctx, integ, schedule, cfg.lambda_schedule.pilot_ps_per_window, timestep_fs)
            else:
                schedule, cfgs = trailblaze(
                    ctx, integ, stages, T, timestep_fs,
                    ScheduleSettings(target_std_kT=cfg.lambda_schedule.target_std_kT, max_windows=max_windows,
                                     pilot_ps=cfg.lambda_schedule.pilot_ps_per_window, seed=seed), log)
                schedule.save(sched_file, leg=job.leg, ligand=job.ligand, repeat=job.repeat, seed=seed,
                              backend=backend_name)
            del ctx, integ
            configs = (np.array([c[0] for c in cfgs]), np.array([c[1] for c in cfgs]), np.array([c[2] for c in cfgs]))
            np.savez(start_file, positions=configs[0], boxes=configs[1], velocities=configs[2])
            log(f"lambda schedule: {len(schedule.states)} states in {time.perf_counter() - t0:.0f} s")

        # ---- 5. replica exchange -----------------------------------------------------------------
        for attempt in range(3):
            spec = LegSpec(
                name=job.tag, system_xml=alch_xml, states=[s.to_dict() for s in schedule.states],
                n_atoms=alch.system.getNumParticles(), backend=backend_name,
                precision=_precision(cfg, backend_name), platform=cfg.backend.platform, temperature_K=T,
                pressure_bar=pressure, barostat=barostat, friction_per_ps=cfg.dynamics.friction_per_ps,
                timestep_fs=timestep_fs, steps_per_iteration=steps_per_iteration, seed=seed + 17,
                cpu_threads=cfg.resources.cpu_threads_per_worker, lrc=tail,
                frame_atoms=prep["solute"] if is_complex else prep["ligand"],
                box_safety=1.3 if barostat == "membrane" else None)
            changed = None
            with ReplicaExchangeSampler(spec, hrex_dir, devices=job.devices, inline=job.inline,
                                        exchange=cfg.sampling.exchange == "hrex", log=log) as sampler:
                if not sampler.restore():
                    sampler.initialize(*configs)
                else:
                    log(f"resumed at iteration {sampler.iteration}")
                progress = max(1, (n_equil + n_prod) // 20)
                if sampler.iteration < n_equil:
                    sampler.run(n_equil - sampler.iteration, checkpoint_stride=ckpt_stride, progress_every=progress)
                    sampler.storage.set_meta(n_equilibration_iterations=n_equil)
                    if cfg.lambda_schedule.mode == "adaptive" and attempt < 2:
                        eq_samples = read_samples(hrex_dir)
                        try:
                            a = analyze_leg(eq_samples, T, n_equilibration=n_equil // 2, n_fractions=1)
                            changed = refine(schedule, a.overlap_neighbour, cfg.lambda_schedule.min_overlap, max_windows)
                        except Exception as exc:  # noqa: BLE001 - refinement is best effort
                            log(f"overlap check after equilibration skipped: {exc}")
                    if changed is not None:
                        pos, box, state_of = sampler.current_configurations()
                        vel = sampler.arrays.vel[sampler.cur].copy()
                        replica_of_state = np.argsort(state_of)
                if changed is None:
                    remaining = n_equil + n_prod - sampler.iteration
                    if remaining > 0:
                        sampler.run(remaining, checkpoint_stride=ckpt_stride, progress_every=progress)
                    sampler.storage.set_meta(n_equilibration_iterations=n_equil)
            if changed is None:
                break
            schedule, picks = changed
            order = replica_of_state[np.array(picks)]
            configs = (pos[order], box[order], vel[order])
            shutil.rmtree(hrex_dir, ignore_errors=True)
            schedule.save(sched_file, leg=job.leg, ligand=job.ligand, repeat=job.repeat, seed=seed, backend=backend_name)
            np.savez(start_file, positions=configs[0], boxes=configs[1], velocities=configs[2])
            log(f"lambda schedule refined to {len(schedule.states)} states; equilibrating again")
        return read_samples(hrex_dir), schedule

    try:
        samples, schedule = run_with(backend_name)
    except (SamplerError, mm.OpenMMException) as exc:
        if backend_name != "cupotential" or cfg.backend.fallback == "none":
            raise
        log(f"cuPotential run failed ({str(exc)[:300]}); repeating this leg on stock OpenMM")
        notes.append("the cuPotential run failed; the whole leg was repeated on stock OpenMM "
                     "(samples from the two backends are never mixed)")
        failed = leg_dir / "hrex_failed_cupotential"
        shutil.rmtree(failed, ignore_errors=True)
        if (leg_dir / "hrex").exists():
            (leg_dir / "hrex").rename(failed)
        for f in ("schedule.yaml", "start_configs.npz"):
            (leg_dir / f).unlink(missing_ok=True)
        backend_name = "openmm"
        samples, schedule = run_with(backend_name)

    # ---- 6. analysis ------------------------------------------------------------------------------
    analysis = analyze_leg(samples, T)
    total = analysis.delta_g - analysis.lrc_delta_g        # decoupling with the full LJ range
    result = {
        "ligand": job.ligand, "leg": job.leg, "repeat": job.repeat, "backend": backend_name,
        "temperature_K": T, "timestep_fs": timestep_fs, "n_states": analysis.n_states,
        "production_ns": float(analysis.n_iterations_used * cfg.sampling.iteration_ps / 1000.0),
        "delta_g_decouple": total, "delta_g_mbar": analysis.delta_g, "delta_g_mbar_error": analysis.delta_g_error,
        "lrc": analysis.lrc_delta_g, "lrc_error": analysis.lrc_error,
        "restraint_release": release_dg, "restraint_release_analytical": release_dg_analytical,
        "ligand_symmetry_number": symmetry, "analysis": analysis.to_dict(), "notes": notes, "units": "kJ/mol",
    }
    ok, reasons = sampling_sufficient(result, cfg)
    result["sampling_sufficient"] = ok
    result["sampling_reasons"] = reasons
    (leg_dir / "result.json").write_text(json.dumps(result, indent=1))
    log(f"dG(decouple) = {total / KCAL:.2f} kcal/mol (MBAR {analysis.delta_g / KCAL:.2f} ± "
        f"{analysis.delta_g_error / KCAL:.2f}, tail {-analysis.lrc_delta_g / KCAL:+.2f}); "
        f"min overlap {analysis.min_overlap:.3f}, round trips {analysis.round_trips}, "
        f"N_eff/state {analysis.effective_samples_per_state:.0f}")
    return result


def _walk_schedule(context, integrator, schedule, pilot_ps: float, timestep_fs: float) -> list:
    """Start configurations for a given schedule: a short run in each state, in order."""
    import openmm.unit as u

    from cufep.sampling.states import set_state

    steps = max(1, int(round(pilot_ps * 1000.0 / timestep_fs)))
    out = []
    for state in schedule.states:
        set_state(context, state)
        integrator.step(steps)
        st = context.getState(getPositions=True, getVelocities=True)
        out.append((st.getPositions(asNumpy=True).value_in_unit(u.nanometer),
                    st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer),
                    st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)))
    return out


def sampling_sufficient(result: dict, cfg: Config) -> tuple[bool, list[str]]:
    """Criteria for stopping at a pre-declared length. None of them looks at the estimate."""
    a = result["analysis"]
    reasons = []
    if a["effective_samples_per_state"] < cfg.sampling.min_effective_samples:
        reasons.append(f"effective samples per state {a['effective_samples_per_state']:.0f} "
                       f"< {cfg.sampling.min_effective_samples}")
    if cfg.sampling.exchange == "hrex" and a["round_trips"] < cfg.sampling.min_round_trips:
        reasons.append(f"replica round trips {a['round_trips']} < {cfg.sampling.min_round_trips}")
    if a["min_overlap"] < cfg.lambda_schedule.min_overlap:
        reasons.append(f"minimum neighbour overlap {a['min_overlap']:.3f} < {cfg.lambda_schedule.min_overlap}")
    return (not reasons), reasons


def run_leg_safe(job: LegJob) -> dict:
    """``run_leg`` that records failures instead of raising (for worker processes)."""
    try:
        return run_leg(job)
    except Exception as exc:  # noqa: BLE001
        err = {"ligand": job.ligand, "leg": job.leg, "repeat": job.repeat, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(limit=12)}
        job.leg_dir.mkdir(parents=True, exist_ok=True)
        (job.leg_dir / "error.json").write_text(json.dumps(err, indent=1))
        print(f"[{job.tag}] FAILED: {err['error']}", flush=True)
        return err


# =============================================================================== combination
def combine_ligand(lig_dir: Path, ligand_name: str, repeats: int) -> dict:
    """Binding free energy from the finished legs of one ligand."""
    per_repeat = []
    warnings: list[str] = []
    for r in range(repeats):
        files = {leg: lig_dir / f"repeat_{r}" / leg / "result.json" for leg in ("complex", "solvent")}
        if not all(f.exists() for f in files.values()):
            warnings.append(f"repeat {r} is incomplete")
            continue
        c, s = (json.loads(files[leg].read_text()) for leg in ("complex", "solvent"))
        dg = s["delta_g_decouple"] - c["delta_g_decouple"] - c["restraint_release"]
        err = float(np.hypot(c["delta_g_mbar_error"], s["delta_g_mbar_error"]))
        per_repeat.append({
            "repeat": r, "dg_bind": dg, "mbar_error": err, "dg_complex": c["delta_g_decouple"],
            "dg_solvent": s["delta_g_decouple"], "restraint_release": c["restraint_release"],
            "lrc_complex": c["lrc"], "lrc_solvent": s["lrc"], "backend_complex": c["backend"],
            "backend_solvent": s["backend"], "stage_complex": c["analysis"]["stage_delta_g"],
            "stage_solvent": s["analysis"]["stage_delta_g"],
            "sampling_sufficient": bool(c["sampling_sufficient"] and s["sampling_sufficient"]),
            "production_ns": {"complex": c["production_ns"], "solvent": s["production_ns"]}})
        for leg_result in (c, s):
            warnings += [f"repeat {r} {leg_result['leg']}: {n}" for n in leg_result.get("notes", [])]
            if not leg_result["sampling_sufficient"]:
                warnings.append(f"repeat {r} {leg_result['leg']}: " + "; ".join(leg_result["sampling_reasons"]))
    out: dict = {"ligand": ligand_name, "units": "kcal/mol", "n_repeats": len(per_repeat), "warnings": warnings}
    if per_repeat:
        vals = np.array([p["dg_bind"] for p in per_repeat]) / KCAL
        out["dg_bind"] = float(vals.mean())
        out["dg_bind_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        out["dg_bind_mbar_error"] = float(np.sqrt(np.mean([(p["mbar_error"] / KCAL) ** 2 for p in per_repeat])))
        energy_keys = ("dg_bind", "mbar_error", "dg_complex", "dg_solvent", "restraint_release", "lrc_complex",
                       "lrc_solvent")
        converted = []
        for p in per_repeat:
            q = dict(p)
            for k in energy_keys:
                q[k] = p[k] / KCAL
            for k in ("stage_complex", "stage_solvent"):
                q[k] = {name: value / KCAL for name, value in p[k].items()}
            converted.append(q)
        out["repeats"] = converted
    (lig_dir / "result.json").write_text(json.dumps(out, indent=1))
    return out
