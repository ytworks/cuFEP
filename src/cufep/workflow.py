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

from cufep.config.schema import Config, WindowsConfig

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
    membrane_info: dict = {}
    if membrane:
        from cufep.prep.membrane import add_membrane
        membrane_info = add_membrane(modeller, forcefield, cfg, log)
        warnings += [f"membrane: {w}" for w in membrane_info.pop("warnings", [])]
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
            "solvent_box_shape": solvent_shape, "membrane_info": membrane_info, "warnings": warnings}
    done.write_text(json.dumps(info, indent=1))
    log(f"{ligand.name}: complex {n_complex} atoms, solvent leg {system.getNumParticles()} atoms"
        + (f"; warnings: {'; '.join(warnings)}" if warnings else ""))
    return info


# =============================================================================== one leg
class JobControl:
    """A job's end of the pipe to the scheduler (cufep.scheduler.run_jobs)."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def _send(self, message: dict) -> None:
        try:
            self.conn.send(message)
        except (BrokenPipeError, OSError):
            pass

    def request_devices(self, replicas: int, work: float, timeout: float = 300.0) -> list | None:
        """Tell the scheduler that sampling starts; returns the GPU of every worker to use."""
        self._send({"event": "sampling", "replicas": int(replicas), "work": float(work)})
        try:
            if self.conn.poll(timeout):
                return list(self.conn.recv()["devices"])
        except (EOFError, OSError, KeyError):
            pass
        return None

    def poll_devices(self) -> list | None:
        """A new device list, if the scheduler has sent one since the last call."""
        latest = None
        try:
            while self.conn.poll():
                latest = self.conn.recv().get("devices", latest)
        except (EOFError, OSError):
            pass
        return list(latest) if latest else None

    def progress(self, work: float) -> None:
        self._send({"event": "progress", "work": float(work)})

    def serial(self) -> None:
        """Sampling is over: the job needs a single worker at most from now on."""
        self._send({"event": "serial"})


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
    #: size of the system, and how many workers of this job one GPU runs at the same time
    n_atoms: int = 0
    oversubscription: int = 1
    #: set in the job's own process when a scheduler is in charge of the GPUs
    control: JobControl | None = None

    @property
    def leg_dir(self) -> Path:
        return Path(self.lig_dir) / f"repeat_{self.repeat}" / self.leg

    @property
    def tag(self) -> str:
        return f"{self.ligand}/r{self.repeat}/{self.leg}"


#: what a leg directory holds of its lambda windows and its replica-exchange run
LEG_RUN_FILES = ("schedule.yaml", "start_configs.npz", "hrex", "hrex_failed_cupotential", "result.json", "error.json",
                 "timing.json")


def archive_leg(leg_dir: Path) -> Path | None:
    """Move the lambda windows, samples and result of a leg into ``previous_<n>/`` in its directory.

    The leg then starts anew from its pre-equilibration and restraint, which stay where they
    are. Nothing is deleted. Returns the new directory, or None if the leg had none of these.
    """
    present = [leg_dir / name for name in LEG_RUN_FILES if (leg_dir / name).exists()]
    if not present:
        return None
    n = 1
    while (leg_dir / f"previous_{n}").exists():
        n += 1
    dest = leg_dir / f"previous_{n}"
    dest.mkdir()
    for f in present:
        f.rename(dest / f.name)
    return dest


def window_stage(lambda_windows: dict, k: int) -> tuple[str, float, float] | None:
    """Stage and the two lambda values of the pair of windows ``k`` and ``k + 1`` of a leg.

    ``lambda_windows`` maps the stages, in order, to their lambda values; the last window of a
    stage is the first of the next.
    """
    first = 0
    for stage, values in lambda_windows.items():
        if first <= k < first + len(values) - 1:
            return stage, float(values[k - first]), float(values[k - first + 1])
        first += len(values) - 1
    return None


def read_result(job: LegJob) -> dict:
    """The result of a finished job, or its error."""
    for name in ("result.json", "error.json"):
        f = job.leg_dir / name
        if f.exists():
            return json.loads(f.read_text())
    return {"ligand": job.ligand, "leg": job.leg, "repeat": job.repeat,
            "error": "the job process ended without writing a result (killed?)"}


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
                                          ligand_symmetry_number, with_force_constants_from_fluctuations)
    from cufep.alchemy.schedule import AlsSettings, Schedule, adaptive_schedule, schedule_from_windows
    from cufep.analysis.mbar import analyze_leg, quality_checks
    from cufep.analysis.structure import structural_analysis
    from cufep.backends import get_backend
    from cufep.mps import thread_percentage
    from cufep.sampling.equilibrate import EquilibrationSettings, pre_equilibrate
    from cufep.sampling.hrex import ReplicaExchangeSampler, SamplerError, WindowPool
    from cufep.sampling.states import STAGES_COMPLEX, STAGES_SOLVENT, stage_state
    from cufep.sampling.storage import read_samples
    from cufep.sampling.worker import LegSpec

    cfg = Config.model_validate(job.config)
    leg_dir = job.leg_dir
    leg_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(leg_dir / "log.txt", prefix=f"[{job.tag}] ")
    if job.reschedule:
        previous = archive_leg(leg_dir)
        if previous is not None:
            log(f"the lambda windows are determined again; the windows and samples this leg had are in {previous.name}/")
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
    timing: dict = {"phases": {}, "started": time.time()}

    def timed(name: str, t_start: float) -> None:
        timing["phases"][name] = timing["phases"].get(name, 0.0) + (time.perf_counter() - t_start)

    timestep_cfg = float(cfg.dynamics.timestep_fs)
    pre = cfg.sampling.pre_equilibration.complex if is_complex else cfg.sampling.pre_equilibration.solvent

    # ---- 1. pre-equilibration: plain MD of the fully interacting system -------------------
    equil_file = leg_dir / "equil.npz"
    if not equil_file.exists():
        in_membrane = is_complex and barostat == "membrane"
        production = pre.md_ns
        restrained: list[int] = []
        wall_atoms: list[int] = []
        wall_half = 0.0
        if in_membrane:
            # While the trimmed bilayer closes up, bulk water is kept out of the hydrophobic core.
            plane = float(prep_info.get("membrane_info", {}).get("phosphate_plane_nm", 0.0))
            wall_half = max(0.0, plane - 0.6)
            if wall_half > 0:
                from scipy.spatial import cKDTree
                heavy_solute = [i for i in prep["solute"]
                                if (el := list(prep["topology"].atoms())[i].element) is not None and el.symbol != "H"]
                tree = cKDTree(prep["positions"][heavy_solute])
                for a_ in prep["topology"].atoms():
                    if a_.residue.name == "HOH" and a_.element is not None and a_.element.symbol == "O":
                        xyz = prep["positions"][a_.index]
                        if abs(xyz[2]) > wall_half and tree.query(xyz)[0] > 0.6:
                            wall_atoms.append(a_.index)
            # A freshly built bilayer needs longer: hold the protein and ligand heavy atoms while the
            # lipids and water settle (NVT, NPT), then run unrestrained for membrane.equilibration_ns.
            production = max(production, cfg.membrane.equilibration_ns)
            atoms = list(prep["topology"].atoms())
            restrained = [i for i in prep["solute"] if atoms[i].element is not None and atoms[i].element.symbol != "H"]
        settings = EquilibrationSettings(
            temperature_K=T, pressure_bar=cfg.thermodynamics.pressure_bar, friction_per_ps=cfg.dynamics.friction_per_ps,
            timestep_fs=timestep_cfg, nvt_ns=pre.nvt_ns, npt_ns=2.0 * pre.npt_ns if in_membrane else pre.npt_ns,
            production_ns=production, barostat=barostat, platform=cfg.backend.platform,
            precision="mixed" if cfg.backend.precision == "single" else cfg.backend.precision, seed=seed,
            md_backend=cfg.backend.name,
            restrained_atoms=restrained, wall_atoms=wall_atoms, wall_half_width_nm=wall_half, wall_centre_nm=0.0,
            ligand_atoms=prep["ligand"] if in_membrane else [],
            water_oxygens=[a.index for a in prep["topology"].atoms()
                           if a.residue.name == "HOH" and a.element is not None and a.element.symbol == "O"]
            if in_membrane else [])
        t0 = time.perf_counter()
        eq = pre_equilibrate(system, prep["positions"], prep["box"], prep["solute"], settings, device0, log)
        tmp = leg_dir / "equil.tmp.npz"
        # the restraint search uses the last pre_equilibration md_ns of the trajectory
        n_keep = max(2, int(round(pre.md_ns * 1000.0 / settings.frame_interval_ps)))
        np.savez(tmp, positions=eq.positions, velocities=eq.velocities, box=eq.box, frames=eq.frames[-n_keep:],
                 frame_boxes=eq.frame_boxes[-n_keep:], frame_atoms=np.array(eq.frame_atoms),
                 timestep_fs=eq.timestep_fs, notes=json.dumps(eq.notes), simulated_ns=eq.simulated_ns,
                 pocket_waters=eq.pocket_waters, area_nm2=eq.area_nm2)
        tmp.replace(equil_file)
        timed("pre_equilibration", t0)
        log(f"pre-equilibration done in {time.perf_counter() - t0:.0f} s (time step {eq.timestep_fs:g} fs)")
    eq = np.load(equil_file)
    timestep_fs = float(eq["timestep_fs"])
    notes += json.loads(str(eq["notes"]))

    # ---- 2. restraint (complex only) --------------------------------------------------------
    t_restraint = time.perf_counter()
    restraint = None
    release_dg = release_dg_analytical = 0.0
    symmetry = 1
    if is_complex:
        from openff.toolkit import Molecule
        rdmol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
        symmetry = ligand_symmetry_number(rdmol)
        rfile = leg_dir / "restraint.json"
        if rfile.exists():
            stored = json.loads(rfile.read_text())
            restraint = BoreschRestraint.from_dict(stored["restraint"])
            notes.extend(stored.get("info", {}).get("notes", []))
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
                                                   dssp_filter=rc.dssp_filter,
                                                   pose_rmsd_warning_nm=rc.pose_rmsd_warning_nm)
                found = find_boresch_restraint(prep["topology"], full, prep["ligand"], rdmol, T, settings)
                restraint = found.restraint
                if rc.force_constants == "auto":
                    restraint = with_force_constants_from_fluctuations(restraint, found.fluctuations, T)
                notes.extend(found.notes)
                info = {"selection": "auto", "force_constants": rc.force_constants, "score": found.score,
                        "fluctuations": found.fluctuations,
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

    timed("restraint", t_restraint)

    # ---- 3. alchemical System ------------------------------------------------------------------
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])      # the equilibrated box
    rest2 = cfg.sampling.rest2
    rest2_on = rest2.scale > 1.0
    hot_atoms, hot_residues = ([], [])
    if rest2_on and is_complex:
        hot_atoms, hot_residues = rest2_region(rest2, prep["topology"], prep["positions"], prep["ligand"],
                                               prep["protein"])
    alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha, rest2=rest2_on,
                                  rest2_atoms=hot_atoms,
                                  bonds=[(b[0].index, b[1].index) for b in prep["topology"].bonds()])
    if rest2_on:
        log(f"REST2: hot region = ligand ({len(prep['ligand'])} atoms)"
            + (f" and {len(hot_atoms)} atoms of {len(hot_residues)} protein residues ({' '.join(hot_residues)})"
               if hot_atoms else "")
            + f"; {alch.rest2_torsions} torsion terms scaled; {rest2.scale:g} times the temperature in the middle of the leg")
    if restraint is not None:
        alch.system.addForce(restraint.create_force())
    alch_xml = mm.XmlSerializer.serialize(alch.system)
    tail = None
    if cfg.analysis.lrc in ("reweight", "isotropic"):
        tail = TailEnergy.from_system(system, prep["ligand"], alch.ligand_parameters, cfg.dynamics.cutoff_nm,
                                      cfg.analysis.lrc_cutoff_nm).to_dict()
    backend_name, bnotes = _choose_backend(cfg, alch.system, log)
    notes += bnotes
    stages = list(STAGES_COMPLEX if is_complex else STAGES_SOLVENT)
    pressure = cfg.thermodynamics.pressure_bar
    iteration_ps = cfg.sampling.time_per_iteration_ps
    steps_per_iteration = max(1, int(round(iteration_ps * 1000.0 / timestep_fs)))
    n_equil = max(1, int(round(cfg.sampling.equilibration(job.leg) * 1000.0 / iteration_ps)))
    n_prod = max(4, int(round(job.production_ns * 1000.0 / iteration_ps)))
    ckpt_stride = max(1, int(round(cfg.sampling.checkpoint_interval_ps / iteration_ps)))

    frame_atoms = prep["solute"] if is_complex else prep["ligand"]
    where = {a: i for i, a in enumerate(frame_atoms)}
    top_atoms = list(prep["topology"].atoms())
    frame_ligand = [where[i] for i in prep["ligand"] if top_atoms[i].element is not None
                    and top_atoms[i].element.symbol != "H"]
    frame_ca = [where[i] for i in prep["protein"] if top_atoms[i].name == "CA" and i in where] if is_complex else []

    def run_with(backend_name: str) -> tuple[dict, Schedule]:
        hrex_dir = leg_dir / "hrex"
        sched_file = leg_dir / "schedule.yaml"
        start_file = leg_dir / "start_configs.npz"
        # ---- 4. lambda schedule ----------------------------------------------------------------
        ls = cfg.lambda_schedule
        if ls.file is not None and not sched_file.exists():
            # one file for every leg, or a directory of frozen schedules: <ligand>/<leg>_r<repeat>.yaml
            src = Path(ls.file)
            if src.is_dir():
                src = src / safe_name(job.ligand) / f"{job.leg}_r{job.repeat}.yaml"
            if not src.exists():
                raise WorkflowError(f"lambda_schedule.file is set but there is no schedule at {src}")
            shutil.copy(src, sched_file)
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
            if sched_file.exists() or ls.mode == "fixed":
                # the windows are given: only make a start configuration for each of them
                if sched_file.exists():
                    schedule = Schedule.load(sched_file)
                else:
                    schedule = schedule_from_windows(stages, ls.windows.for_leg(job.leg),
                                                     WindowsConfig().for_leg(job.leg), rest2.scale)
                cfgs = _walk_schedule(ctx, integ, schedule, ls.adaptive.window_ps, timestep_fs)
            else:
                a_ = ls.adaptive
                als = AlsSettings(overlap_threshold=a_.overlap_threshold, candidates=a_.candidates,
                                  initial_windows=a_.initial_windows.for_leg(job.leg), window_ps=a_.window_ps,
                                  equilibration_ps=a_.equilibration_ps, sample_interval_ps=a_.sample_interval_ps,
                                  walk_ps=a_.walk_ps,
                                  max_windows=a_.max_windows.complex if is_complex else a_.max_windows.solvent,
                                  rest2_scale=rest2.scale)
                # the windows of an iteration run at the same time on the GPUs the scheduler grants
                n_first = sum(als.initial_windows[st] for st in stages) - (len(stages) - 1)
                devices = list(job.devices)
                if job.control is not None:
                    granted = job.control.request_devices(
                        n_first, n_first * (als.window_ps + als.equilibration_ps) / max(1, job.oversubscription))
                    devices = granted or devices
                if job.inline or len(devices) < 2:
                    schedule, cfgs = adaptive_schedule(ctx, integ, stages, T, timestep_fs, als, log)
                else:
                    pool_spec = LegSpec(
                        name=job.tag, system_xml=alch_xml, states=[stage_state(stages[0], 0.0).to_dict(),
                                                                   stage_state(stages[-1], 1.0).to_dict()],
                        n_atoms=alch.system.getNumParticles(), backend=backend_name,
                        precision=_precision(cfg, backend_name), platform=cfg.backend.platform, temperature_K=T,
                        pressure_bar=pressure, barostat=barostat, friction_per_ps=cfg.dynamics.friction_per_ps,
                        timestep_fs=timestep_fs, steps_per_iteration=steps_per_iteration, seed=seed + 23,
                        cpu_threads=cfg.resources.cpu_threads_per_worker,
                        box_safety=1.3 if barostat == "membrane" else None,
                        mps_thread_percentage=thread_percentage(job.oversubscription))
                    with WindowPool(pool_spec, devices[:n_first], log) as pool:
                        schedule, cfgs = adaptive_schedule(ctx, integ, stages, T, timestep_fs, als, log, pool=pool)
            schedule.save(sched_file, leg=job.leg, ligand=job.ligand, repeat=job.repeat, seed=seed,
                          backend=backend_name)
            del ctx, integ
            configs = (np.array([c[0] for c in cfgs]), np.array([c[1] for c in cfgs]), np.array([c[2] for c in cfgs]))
            np.savez(start_file, positions=configs[0], boxes=configs[1], velocities=configs[2])
            timed("lambda_schedule", t0)
            log(f"lambda schedule ({schedule.mode}): {len(schedule.states)} windows in {time.perf_counter() - t0:.0f} s: "
                + "; ".join(f"{name} {len(v)}" for name, v in schedule.windows().items()))

        if abs(schedule.rest2_scale - (rest2.scale if rest2_on else 1.0)) > 1e-9:
            raise WorkflowError(
                f"the lambda schedule of this leg was made with a REST2 scale of {schedule.rest2_scale:g} but "
                f"sampling.rest2.scale is {rest2.scale:g}; use the same value, or remove {sched_file} to make a new one")

        # ---- 5. replica exchange -----------------------------------------------------------------
        spec = LegSpec(
            name=job.tag, system_xml=alch_xml, states=[s.to_dict() for s in schedule.states],
            n_atoms=alch.system.getNumParticles(), backend=backend_name,
            precision=_precision(cfg, backend_name), platform=cfg.backend.platform, temperature_K=T,
            pressure_bar=pressure, barostat=barostat, friction_per_ps=cfg.dynamics.friction_per_ps,
            timestep_fs=timestep_fs, steps_per_iteration=steps_per_iteration, seed=seed + 17,
            cpu_threads=cfg.resources.cpu_threads_per_worker, lrc=tail,
            frame_atoms=frame_atoms, frame_ligand=frame_ligand, frame_ca=frame_ca,
            box_safety=1.3 if barostat == "membrane" else None,
            mps_thread_percentage=thread_percentage(job.oversubscription))
        n_windows, total = len(schedule.states), n_equil + n_prod
        control = job.control
        # seconds one replica needs per iteration: a guess until the first iterations were timed
        replica_seconds = [0.05 * (1.0 + alch.system.getNumParticles() / 50000.0)]

        def remaining_work(done: int) -> float:
            return n_windows * max(0, total - done) * replica_seconds[0] / max(1, job.oversubscription)

        t_sampling = time.perf_counter()
        with ReplicaExchangeSampler(spec, hrex_dir, devices=job.devices, inline=job.inline,
                                    exchange=cfg.sampling.exchange == "hrex", log=log) as sampler:
            if not sampler.restore():
                sampler.initialize(*configs)
            else:
                log(f"resumed at iteration {sampler.iteration}")
            if control is not None and total > sampler.iteration:
                devices = control.request_devices(n_windows, remaining_work(sampler.iteration))
                if devices:
                    sampler.set_devices(devices)

            def on_iteration(smp, stats) -> None:
                if control is None:
                    return
                per_worker = stats.all_workers / max(1, len(smp.workers))
                per_replica = per_worker / max(1.0, n_windows / max(1, len(smp.workers)))
                replica_seconds[0] = 0.8 * replica_seconds[0] + 0.2 * per_replica if smp.iteration > 3 else per_replica
                devices = control.poll_devices()
                if devices:
                    smp.set_devices(devices)
                if smp.iteration % 25 == 0:
                    control.progress(remaining_work(smp.iteration))

            remaining = total - sampler.iteration
            first = sampler.iteration
            if remaining > 0:
                sampler.run(remaining, checkpoint_stride=ckpt_stride, progress_every=max(1, total // 20),
                            on_iteration=on_iteration)
            sampler.storage.set_meta(n_equilibration_iterations=n_equil)
            timing["sampling"] = {
                "lambda_windows": n_windows, "iterations": sampler.iteration - first,
                "seconds_per_iteration": (time.perf_counter() - t_sampling) / max(1, sampler.iteration - first),
                "worker_busy_fraction": sampler.busy_seconds / sampler.worker_seconds_available
                if sampler.worker_seconds_available > 0 else None,
                "workers": [[it, list(devs)] for it, devs in sampler.device_history],
                "oversubscription_factor": job.oversubscription}
        if control is not None:
            control.serial()
        timed("sampling", t_sampling)
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
    t_analysis = time.perf_counter()
    analysis = analyze_leg(samples, T, n_fractions=cfg.analysis.convergence_points)
    correction = {"reweight": analysis.lrc_delta_g, "isotropic": analysis.lrc_isotropic, "none": 0.0}[cfg.analysis.lrc]
    total = analysis.delta_g - correction                  # decoupling with the full LJ range
    structure = structural_analysis(samples, int(samples["attrs"].get("n_equilibration_iterations", 0)))
    if structure is not None:
        notes += structure["warnings"]
    result = {
        "ligand": job.ligand, "leg": job.leg, "repeat": job.repeat, "backend": backend_name,
        "temperature_K": T, "timestep_fs": timestep_fs, "n_states": analysis.n_states,
        "production_ns": float(analysis.n_iterations_used * iteration_ps / 1000.0),
        "delta_g_decouple": total, "delta_g_mbar": analysis.delta_g, "delta_g_mbar_error": analysis.delta_g_error,
        "lrc": correction, "lrc_method": cfg.analysis.lrc, "lrc_reweight": analysis.lrc_delta_g,
        "lrc_isotropic": analysis.lrc_isotropic, "lrc_error": analysis.lrc_error, "structure": structure,
        "restraint_release": release_dg, "restraint_release_analytical": release_dg_analytical,
        "ligand_symmetry_number": symmetry, "analysis": analysis.to_dict(), "notes": notes, "units": "kJ/mol",
    }
    result["quality"] = quality_checks(result["analysis"], cfg.analysis.min_overlap,
                                       cfg.analysis.convergence_tolerance_sigma)
    result["lambda_windows"] = schedule.windows()
    result["notes"] = notes + list(schedule.notes) + quality_warnings(result, cfg)
    # windows that adaptive lambda scheduling added to its start, per stage (None: fixed windows)
    start = schedule.provenance.get("initial_windows") if schedule.mode == "adaptive" else None
    result["lambda_windows_added"] = (
        {name: len(values) - int(start[name]) for name, values in schedule.windows().items() if name in start}
        if start else None)
    result["rest2"] = {"scale": float(rest2.scale) if rest2_on else 1.0, "hot_protein_atoms": len(hot_atoms),
                       "hot_residues": hot_residues, "torsion_terms_scaled": alch.rest2_torsions,
                       "scale_of_window": [float(st.rest2_scale) for st in schedule.states]}
    timed("analysis", t_analysis)
    timing.update(finished=time.time(), backend=backend_name, atoms=alch.system.getNumParticles(),
                  ligand=job.ligand, leg=job.leg, repeat=job.repeat)
    (leg_dir / "timing.json").write_text(json.dumps(timing, indent=1))
    (leg_dir / "result.json").write_text(json.dumps(result, indent=1))
    log(f"dG(decouple) = {total / KCAL:.2f} kcal/mol (MBAR {analysis.delta_g / KCAL:.2f} ± "
        f"{analysis.delta_g_error / KCAL:.2f}, tail {-correction / KCAL:+.2f}); "
        f"min overlap {analysis.min_overlap:.3f}, round trips {analysis.round_trips}, "
        f"N_eff/state {analysis.effective_samples_per_state:.0f}")
    return result


def rest2_region(rest2, topology, positions_nm, ligand, protein) -> tuple[list[int], list[str]]:
    """Protein atoms of the REST2 hot region (whole residues) and the names of their residues.

    ``region: pocket`` takes the residues with a heavy atom within ``pocket_cutoff_nm`` of a
    heavy atom of the ligand in the prepared structure; ``residues`` adds residues by number.
    """
    atoms = list(topology.atoms())
    protein = set(int(i) for i in protein)
    positions = np.asarray(positions_nm)

    def heavy(i: int) -> bool:
        return atoms[i].element is not None and atoms[i].element.symbol != "H"

    chosen: set[int] = set()
    if rest2.region == "pocket":
        ligand_heavy = positions[[i for i in ligand if heavy(i)]]
        for i in sorted(protein):
            if heavy(i) and atoms[i].residue.index not in chosen:
                if np.min(np.linalg.norm(ligand_heavy - positions[i], axis=1)) < rest2.pocket_cutoff_nm:
                    chosen.add(atoms[i].residue.index)
    protein_residues = {atoms[i].residue.index: atoms[i].residue for i in protein}
    for spec in rest2.residues:
        chain, _, number = str(spec).rpartition(":")
        found = [r.index for r in protein_residues.values()
                 if str(r.id).strip() == number.strip() and (not chain or r.chain.id == chain)]
        if not found:
            raise WorkflowError(f"sampling.rest2.residues: no protein residue '{spec}'")
        chosen.update(found)
    hot = sorted(i for i in protein if atoms[i].residue.index in chosen)
    names = [f"{protein_residues[k].name}{protein_residues[k].id}" for k in sorted(chosen)]
    return hot, names


def _walk_schedule(context, integrator, schedule, window_ps: float, timestep_fs: float) -> list:
    """Start configurations for given windows: a short run in each window, in order."""
    import openmm.unit as u

    from cufep.sampling.states import set_state

    steps = max(1, int(round(window_ps * 1000.0 / timestep_fs)))
    out = []
    for state in schedule.states:
        set_state(context, state)
        integrator.step(steps)
        st = context.getState(getPositions=True, getVelocities=True)
        out.append((st.getPositions(asNumpy=True).value_in_unit(u.nanometer),
                    st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer),
                    st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)))
    return out


def quality_warnings(result: dict, cfg: Config) -> list[str]:
    """Warnings about the sampling of a finished leg, from ``result["quality"]``
    (:func:`cufep.analysis.mbar.quality_checks`). None of them changes what is run.
    """
    from cufep.analysis.mbar import quality_checks

    a = result["analysis"]
    q = result.get("quality") or quality_checks(a, cfg.analysis.min_overlap, cfg.analysis.convergence_tolerance_sigma)
    sigma = cfg.analysis.convergence_tolerance_sigma
    out = []
    if not q["overlap"]["ok"]:
        k = q["overlap"]["windows"][0]
        where = window_stage(result.get("lambda_windows") or {}, k)
        stage = "" if where is None else f" ({where[0]} stage, lambda {where[1]:g} and {where[2]:g})"
        out.append(f"NOT RELIABLE: the overlap between lambda windows {k} and {k + 1}{stage} is "
                   f"{q['overlap']['lowest']:.3f}, below {cfg.analysis.min_overlap:g}. Give that stage more lambda "
                   f"windows (lambda_schedule.adaptive.initial_windows, or lambda_schedule.windows with mode: fixed) "
                   f"and run this leg again: cufep run CONFIG --ligand NAME --leg {result.get('leg', 'LEG')} --reschedule")
    conv = q["convergence"]
    if conv["ok"] is False:
        whole = f"{conv['final'] / KCAL:.2f} ± {a['delta_g_error'] / KCAL:.2f}"
        if np.isfinite(conv["first_half"]) and np.isfinite(conv["second_half"]):
            halves = (f"the first half of the production gives {conv['first_half'] / KCAL:.2f} kcal/mol and the "
                      f"second half {conv['second_half'] / KCAL:.2f}")
        else:
            halves = "half of the production does not determine the free energy"
        out.append(f"NOT CONVERGED: {halves}, the whole {whole}; both halves should lie within {sigma:g} standard "
                   f"errors of the whole. Run this leg longer; it continues from where it stopped: cufep run CONFIG "
                   f"--ligand NAME --leg {result.get('leg', 'LEG')} --production-ns NS (or sampling.production_ns)")
    est = q["estimators"]
    if est["ok"] is False:
        out.append(f"BAR between neighbouring lambda windows gives {est['bar'] / KCAL:.2f} kcal/mol and MBAR "
                   f"{est['mbar'] / KCAL:.2f}; they should agree within {sigma:g} standard errors. The windows are "
                   f"too far apart or too little sampled")
    return out


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
    # legs whose analysis fails a check of cufep.analysis.mbar.quality_checks
    flagged: dict[str, list[str]] = {"overlap": [], "convergence": [], "estimators": []}
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
            "lambda_windows": {"complex": c["n_states"], "solvent": s["n_states"]},
            "lambda_windows_added": {"complex": c.get("lambda_windows_added"), "solvent": s.get("lambda_windows_added")},
            "min_overlap": {"complex": c["analysis"]["min_overlap"], "solvent": s["analysis"]["min_overlap"]},
            "production_ns": {"complex": c["production_ns"], "solvent": s["production_ns"]}})
        for leg_result in (c, s):
            warnings += [f"repeat {r} {leg_result['leg']}: {n}" for n in leg_result.get("notes", [])]
            where = f"{leg_result['leg']} leg" + (f" of repeat {r}" if repeats > 1 else "")
            quality = leg_result.get("quality") or {}
            for check, legs in flagged.items():
                if (quality.get(check) or {}).get("ok") is False:
                    legs.append(where)
    out: dict = {"ligand": ligand_name, "units": "kcal/mol", "n_repeats": len(per_repeat), "warnings": warnings,
                 "not_reliable": flagged["overlap"], "not_converged": flagged["convergence"],
                 "estimators_disagree": flagged["estimators"]}
    if per_repeat:
        vals = np.array([p["dg_bind"] for p in per_repeat]) / KCAL
        out["dg_bind"] = float(vals.mean())
        out["dg_bind_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        out["dg_bind_mbar_error"] = float(np.sqrt(np.mean([(p["mbar_error"] / KCAL) ** 2 for p in per_repeat])))
        out["min_overlap"] = float(min(min(p["min_overlap"].values()) for p in per_repeat))
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
