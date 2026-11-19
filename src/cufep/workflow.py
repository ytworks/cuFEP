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

from cufep import jsonio
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


def settings_differences(stored, current, prefix: str = "") -> list[str]:
    """Where two nested records of settings differ, as ``"a.b: old -> new"``."""
    from cufep.sampling.storage import _same

    if isinstance(stored, dict) and isinstance(current, dict):
        out: list[str] = []
        for key in sorted(set(stored) | set(current)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored or key not in current:
                out.append(f"{name}: {stored.get(key, 'not recorded')} -> {current.get(key, 'not given')}")
            else:
                out += settings_differences(stored[key], current[key], name)
        return out
    return [] if _same(stored, current) else [f"{prefix}: {stored} -> {current}"]


def check_made_from(what: str, where: Path, stored: dict | None, current: dict, log=print,
                    contradictions: list[str] | tuple = ()) -> bool:
    """Refuse to reuse a file of an earlier run that was made from other input or settings.

    cuFEP keeps what it has computed (the prepared systems, the pre-equilibration, the
    restraint) and uses it again when a calculation is run again in its output directory. That
    is only right while the input and the settings it came from are the same.

    ``stored`` is the record in the file of what it was made from. A file of a version that kept
    no such record (None) is refused if what it does hold contradicts this run
    (``contradictions``, as ``"item: in the file -> in this run"``); otherwise it is used and
    False is returned, so that the result can say that its origin could not be checked.
    """
    if stored is None:
        if contradictions:
            raise WorkflowError(
                f"{what} in {where} cannot be used again: the file is of a version that did not record what it "
                f"was made from, and what it does hold is not what this run asks for ({'; '.join(contradictions)}). "
                "Nothing was changed. A calculation with other input or settings needs a new output.dir.")
        log(f"{what}: {where} was written by a version that did not record what it was made from; nothing it "
            "holds contradicts this run and it is used as it is")
        return False
    differing = settings_differences(stored, current)
    if differing:
        raise WorkflowError(
            f"{what} in {where} cannot be used again: this run gives other input or settings than the ones "
            f"that went into it ({'; '.join(differing[:8])}{' ...' if len(differing) > 8 else ''}). Nothing was "
            "changed. A calculation with other input or settings needs a new output.dir.")
    return True


def prep_contradictions(info: dict, ligand_name: str, membrane: bool, box_shape: str) -> list[str]:
    """Where a ``prep.json`` without a record of its origin says something else than this run."""
    expected = {"ligand": ligand_name, "membrane": bool(membrane),
                "complex_barostat": "membrane" if membrane else "isotropic", "box_shape": box_shape}
    return [f"{key}: {info[key]} -> {value}" for key, value in expected.items() if key in info and info[key] != value]


def equil_in_membrane(eq) -> bool | None:
    """Whether a stored pre-equilibration was one of a system in a bilayer; None if the file does not tell.

    The water molecules near the ligand are counted only for a bilayer system. The area of the
    box is recorded for every system and says nothing.
    """
    if "pocket_waters" not in eq.files:
        return None
    return bool(np.asarray(eq["pocket_waters"]).size > 0)


def equil_contradictions(eq, cfg: Config, leg: str, in_membrane: bool) -> list[str]:
    """Where an ``equil.npz`` without a record of its origin says something else than this run.

    The time step in the file is the one that was used, which is smaller than the configured
    one after the automatic fall-back to 2 fs: only a larger one is a contradiction. The MD of a
    bilayer system could be extended by earlier versions, so there only a shorter one is.
    """
    pre = cfg.sampling.pre_equilibration.complex if leg == "complex" else cfg.sampling.pre_equilibration.solvent
    out = []
    files = set(eq.files)
    if "timestep_fs" in files and float(eq["timestep_fs"]) > float(cfg.dynamics.timestep_fs) + 1e-9:
        out.append(f"time step: {float(eq['timestep_fs']):g} fs -> {float(cfg.dynamics.timestep_fs):g} fs")
    if "simulated_ns" in files:
        asked = max(pre.md_ns, cfg.membrane.equilibration_ns) if in_membrane else pre.md_ns
        done = float(eq["simulated_ns"])
        if done < asked - 1e-6 or (not in_membrane and done > asked + 1e-6):
            out.append(f"length of the MD: {done:g} ns -> {asked:g} ns")
    has_bilayer = equil_in_membrane(eq)
    if has_bilayer is not None and has_bilayer != bool(in_membrane):
        out.append(f"bilayer: {'yes' if has_bilayer else 'no'} -> {'yes' if in_membrane else 'no'}")
    return out


def restraint_contradictions(stored: dict, cfg: Config) -> list[str]:
    """Where a ``restraint.json`` without a record of its origin says something else than this run."""
    rc, info, restraint = cfg.restraints, stored.get("info") or {}, stored.get("restraint") or {}
    out = []
    if "selection" in info and info["selection"] != rc.selection:
        out.append(f"restraints.selection: {info['selection']} -> {rc.selection}")
    if "force_constants" in info and info["force_constants"] != rc.force_constants:
        out.append(f"restraints.force_constants: {info['force_constants']} -> {rc.force_constants}")
    if rc.selection == "manual":
        for key, asked in (("host_atoms", rc.host_atoms), ("guest_atoms", rc.guest_atoms)):
            if asked and key in restraint and list(restraint[key]) != list(asked):
                out.append(f"restraints.{key}: {list(restraint[key])} -> {list(asked)}")
    # the six force constants, where the run sets them (not taken from the fluctuations): the file
    # holds them in kJ/mol/nm^2 and kJ/mol/rad^2
    if rc.force_constants != "auto" or rc.selection == "manual":
        from cufep.sampling.storage import _same
        if "k_r" in restraint and not _same(float(restraint["k_r"]), rc.k_r_kcal_per_A2 * KCAL * 100.0):
            out.append(f"restraints.k_r_kcal_per_A2: {float(restraint['k_r']) / (KCAL * 100.0):g} -> {rc.k_r_kcal_per_A2:g}")
        angles = {key: float(restraint[key]) / KCAL for key in ("k_theta_a", "k_theta_b", "k_phi_a", "k_phi_b", "k_phi_c")
                  if key in restraint}
        other = {key: value for key, value in angles.items() if not _same(value, float(rc.k_angle_kcal_per_rad2))}
        if other:
            out.append("restraints.k_angle_kcal_per_rad2: " + ", ".join(f"{key} {value:g}" for key, value in other.items())
                       + f" -> {rc.k_angle_kcal_per_rad2:g}")
    return out


def leg_seed(cfg: Config, repeat: int, leg: str) -> int:
    """The seed of a leg: every repeat and each of its two legs has its own."""
    return int(cfg.sampling.seed) + 7919 * int(repeat) + (0 if leg == "complex" else 104729)


def samples_settings(cfg: Config) -> dict:
    """The settings that the samples of a leg depend on and that their other recorded conditions
    (states, temperature, pressure, dynamics, System checksum) do not name in the words of the
    configuration: the soft-core, REST2, the length of an iteration and the tail correction."""
    return {"alchemy": cfg.alchemy.model_dump(mode="json"), "rest2": cfg.sampling.rest2.model_dump(mode="json"),
            "time_per_iteration_ps": float(cfg.sampling.time_per_iteration_ps),
            "tail": None if cfg.analysis.lrc == "none" else {"cutoff_nm": float(cfg.analysis.lrc_cutoff_nm)}}


def alchemical_leg(cfg: Config, system, prep: dict, box, restraint, is_complex: bool):
    """The alchemical System of a leg as its samples are made with it, and what belongs to it.

    One function for the run of a leg (:func:`run_leg`) and for the check before a run
    (:func:`check_leg_reuse`), so that both get the same System to the byte of its XML, which is
    what the checksum in the conditions of the samples is taken of. The order is part of that:
    the equilibrated ``box`` is set on the prepared ``system`` of the leg (which is changed), the
    alchemical System is made from it, the force of the ``restraint`` (None for the solvent leg)
    is added, and the result is serialised. Needs no GPU.

    Returns a namespace: ``alch`` (:func:`cufep.alchemy.factory.make_alchemical_system`), ``xml``,
    ``tail`` (the record of the tail correction, None with ``analysis.lrc: none``), ``rest2_on``,
    ``hot_atoms`` and ``hot_residues`` (the protein part of the REST2 region).
    """
    from types import SimpleNamespace

    import openmm as mm

    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.lrc import TailEnergy

    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])      # the equilibrated box
    rest2 = cfg.sampling.rest2
    rest2_on = rest2.scale > 1.0
    hot_atoms, hot_residues = ([], [])
    if rest2_on and is_complex:
        hot_atoms, hot_residues = rest2_region(rest2, prep["topology"], prep["positions"], prep["ligand"],
                                               prep["protein"])
    alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha, rest2=rest2_on,
                                  rest2_atoms=hot_atoms,
                                  bonds=[(b[0].index, b[1].index) for b in prep["topology"].bonds()])
    if restraint is not None:
        alch.system.addForce(restraint.create_force())
    xml = mm.XmlSerializer.serialize(alch.system)
    tail = None
    if cfg.analysis.lrc in ("reweight", "isotropic"):
        tail = TailEnergy.from_system(system, prep["ligand"], alch.ligand_parameters, cfg.dynamics.cutoff_nm,
                                      cfg.analysis.lrc_cutoff_nm).to_dict()
    return SimpleNamespace(alch=alch, xml=xml, tail=tail, rest2_on=rest2_on, hot_atoms=hot_atoms,
                           hot_residues=hot_residues)


def leg_spec(cfg: Config, built, states: list[dict], barostat: str, timestep_fs: float, backend_name: str, seed: int,
             **more):
    """What the workers of a leg are given (:class:`cufep.sampling.worker.LegSpec`), from the
    configuration and the alchemical System of the leg (:func:`alchemical_leg`).

    Its ``conditions()`` are what the samples record, so the run of a leg and the check before a
    run make it here, in one place. ``timestep_fs`` is the time step that was used in the
    pre-equilibration, ``more`` what only a run needs (name, frame atoms, MPS share).
    """
    from cufep.sampling.worker import LegSpec

    iteration_ps = cfg.sampling.time_per_iteration_ps
    return LegSpec(
        name=more.pop("name", "leg"), system_xml=built.xml, states=states, n_atoms=built.alch.system.getNumParticles(), backend=backend_name,
        precision=_precision(cfg, backend_name), platform=cfg.backend.platform,
        temperature_K=cfg.thermodynamics.temperature_K, pressure_bar=cfg.thermodynamics.pressure_bar,
        barostat=barostat, friction_per_ps=cfg.dynamics.friction_per_ps, timestep_fs=timestep_fs,
        steps_per_iteration=max(1, int(round(iteration_ps * 1000.0 / timestep_fs))), seed=seed,
        cpu_threads=cfg.resources.cpu_threads_per_worker, lrc=built.tail, settings=samples_settings(cfg),
        box_safety=1.3 if barostat == "membrane" else None, **more)


def conditions_of_this_run(cfg: Config, lig_dir: Path, leg: str, repeat: int, prep_info: dict, backend_name: str,
                           n_stored: int) -> dict:
    """The conditions under which this run would sample a leg that has files, made from those
    files as :func:`run_leg` makes them, without a GPU and without writing anything.

    For samples that record their conditions and not the settings behind them (one version of
    cuFEP wrote such): nothing but the checksum of the System says what soft-core, restraint or
    REST2 region they were made with, so the System of this run is built to compare it. Raises
    ConditionsError if a file that is needed for that is not there.
    """
    import openmm as mm

    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.alchemy.schedule import Schedule
    from cufep.sampling.storage import ConditionsError

    leg_dir = lig_dir / f"repeat_{repeat}" / leg
    is_complex = leg == "complex"
    needed = {"the prepared system": lig_dir / "prep" / f"{leg}_system.xml",
              "the pre-equilibration": leg_dir / "equil.npz", "the lambda schedule": leg_dir / "schedule.yaml"}
    if is_complex:
        needed["the restraint"] = leg_dir / "restraint.json"
    missing = [f"{name} ({path.name})" for name, path in needed.items() if not path.exists()]
    if missing:
        raise ConditionsError(
            f"the {n_stored} iterations of samples in {leg_dir / 'hrex'} were written by a version of cuFEP that "
            "recorded a checksum of their System and not the settings behind it. To see whether this run would "
            f"continue them under the same conditions the System has to be built again, and {', '.join(missing)} "
            "of this leg is not there. Nothing was changed. The samples can be analysed (cufep analyze); to sample "
            "the leg anew, use --reschedule.")
    prep = load_prepared(lig_dir / "prep", leg)
    eq = np.load(leg_dir / "equil.npz")
    restraint = BoreschRestraint.from_dict(json.loads((leg_dir / "restraint.json").read_text())["restraint"]) \
        if is_complex else None
    built = alchemical_leg(cfg, mm.XmlSerializer.deserialize(prep["system_xml"]), prep, eq["box"], restraint, is_complex)
    states = [state.to_dict() for state in Schedule.load(leg_dir / "schedule.yaml").states]
    barostat = prep_info["complex_barostat"] if is_complex else "isotropic"
    return leg_spec(cfg, built, states, barostat, float(eq["timestep_fs"]), backend_name, 0).conditions()


def check_leg_reuse(cfg: Config, lig_dir: Path, leg: str, repeat: int, prep_info: dict, production_ns: float,
                    reschedule: bool = False, log=print) -> None:
    """Before a run writes or samples anything: what the leg directory holds must be usable by it.

    The same checks are made again where the files are used (:func:`run_leg`); here they are made
    for all legs of a run first, from the files alone, so that a run that will be refused leaves
    the output directory, and its record of the configuration, as they were. Raises WorkflowError
    or ConditionsError.
    """
    from cufep.alchemy.schedule import Schedule
    from cufep.sampling.storage import _same, conditions_differences, other_conditions, stored_conditions

    leg_dir = lig_dir / f"repeat_{repeat}" / leg
    is_complex = leg == "complex"
    in_membrane = is_complex and prep_info.get("complex_barostat") == "membrane"
    quiet = lambda message: None                                    # noqa: E731  run_leg logs it when it uses the file
    equil_file = leg_dir / "equil.npz"
    if equil_file.exists():
        eq = np.load(equil_file)
        check_made_from("the pre-equilibration of this leg", equil_file,
                        json.loads(str(eq["made_with"])) if "made_with" in eq.files else None,
                        equil_made_with(cfg, leg, in_membrane, leg_seed(cfg, repeat, leg)), quiet,
                        contradictions=equil_contradictions(eq, cfg, leg, in_membrane))
    rfile = leg_dir / "restraint.json"
    if is_complex and rfile.exists():
        stored = _read_json(rfile) or {}
        check_made_from("the restraint of this leg", rfile, stored.get("made_with"), restraint_made_with(cfg), quiet,
                        contradictions=restraint_contradictions(stored, cfg))
    if reschedule:
        return                           # the windows and samples the leg has are set aside; it starts anew
    rest2_scale = cfg.sampling.rest2.scale if cfg.sampling.rest2.scale > 1.0 else 1.0
    sched_file = leg_dir / "schedule.yaml"
    if sched_file.exists():
        schedule = Schedule.load(sched_file)
        if abs(schedule.rest2_scale - rest2_scale) > 1e-9:
            raise WorkflowError(
                f"the lambda schedule in {sched_file} was made with a REST2 scale of {schedule.rest2_scale:g} but "
                f"sampling.rest2.scale is {rest2_scale:g}. Nothing was changed. Use the same value, or start the "
                "leg anew with --reschedule.")
    n_stored, conditions = stored_conditions(leg_dir / "hrex")
    if n_stored == 0:
        return
    if conditions is None:
        from cufep.sampling.storage import stored_attributes

        used_timestep = float(np.load(equil_file)["timestep_fs"]) if equil_file.exists() else float(cfg.dynamics.timestep_fs)
        states = [state.to_dict() for state in Schedule.load(sched_file).states] if sched_file.exists() else None
        problems = unrecorded_samples_problems(stored_attributes(leg_dir / "hrex")[1], n_stored, leg_dir, cfg, leg,
                                               production_ns, states, used_timestep)
        if problems:
            refuse_unrecorded_samples(leg_dir, n_stored, problems)
        return
    current = {"temperature_K": float(cfg.thermodynamics.temperature_K),
               "pressure_bar": float(cfg.thermodynamics.pressure_bar),
               "friction_per_ps": float(cfg.dynamics.friction_per_ps)}
    differing = [f"{key}: {conditions[key]} -> {value}" for key, value in current.items()
                 if key in conditions and not _same(conditions[key], value)]
    if isinstance(conditions.get("settings"), dict):
        differing += settings_differences(conditions["settings"], samples_settings(cfg))
    elif not differing:
        # samples that record their conditions without the settings behind them: all their conditions,
        # with the checksum of the System, against those this run would sample under
        current = conditions_of_this_run(cfg, lig_dir, leg, repeat, prep_info,
                                         str(conditions.get("backend") or cfg.backend.name), n_stored)
        differing = conditions_differences(conditions, current)
    if differing:
        raise other_conditions(n_stored, leg_dir / "hrex", differing)


def check_prepared(cfg: Config, ligand, lig_dir: Path, membrane: bool, box_shape: str, log=print) -> dict | None:
    """The record of the prepared systems of a ligand if they exist and this run can use them (None
    if there are none yet); raises WorkflowError if they were made from other input or settings."""
    done = lig_dir / "prep" / "prep.json"
    if not done.exists():
        return None
    info = json.loads(done.read_text())
    check_made_from(f"the prepared systems of {ligand.name}", lig_dir / "prep", info.get("made_from"),
                    prep_made_from(cfg, ligand, membrane, box_shape), log,
                    contradictions=prep_contradictions(info, ligand.name, membrane, box_shape))
    return info


def equil_notes(eq, cfg: Config, in_membrane: bool) -> list[str]:
    """The notes of a stored pre-equilibration, with the verdict on the bilayer made anew.

    Whether the bilayer has settled is judged with the tolerances of this run from the series the
    file holds (area in the membrane plane, water molecules near the ligand), not taken from the
    sentence the run wrote that made the file: a tolerance is a level for a note and can be changed
    without making the pre-equilibration another one.
    """
    from cufep.sampling.equilibrate import BILAYER_NOTE, bilayer_note, membrane_settled

    notes = json.loads(str(eq["notes"])) if "notes" in eq.files else []
    if not in_membrane or "area_nm2" not in eq.files:
        return notes
    notes = [n for n in notes if not n.startswith(BILAYER_NOTE)]
    waters = np.asarray(eq["pocket_waters"]) if "pocket_waters" in eq.files else np.zeros(0)
    verdict = membrane_settled(np.asarray(eq["area_nm2"]), waters, cfg.membrane.area_tolerance,
                               cfg.membrane.pocket_water_tolerance)
    note = bilayer_note(verdict, float(eq["simulated_ns"]) if "simulated_ns" in eq.files else float("nan"))
    return notes + ([note] if note else [])


def restraint_notes(stored: dict, cfg: Config) -> list[str]:
    """The notes of a stored restraint, with the note about the pose made anew.

    The file holds how far the ligand was from the pose it was given; that is compared with
    ``restraints.pose_rmsd_warning_nm`` of this run. A file of a version that measured the
    distance from the first frame of the MD, or that holds none, cannot be judged again: its
    note is kept and the leg says so.
    """
    from cufep.alchemy.restraints import POSE_NOTE, pose_note

    info = stored.get("info") or {}
    notes = list(info.get("notes", []))
    if info.get("selection") == "manual":
        return notes
    if info.get("ligand_rmsd_reference") == "input pose" and info.get("ligand_rmsd_A") is not None:
        notes = [n for n in notes if not n.startswith(POSE_NOTE)]
        note = pose_note(float(info["ligand_rmsd_A"]) / 10.0, cfg.restraints.pose_rmsd_warning_nm)
        return notes + ([note] if note else [])
    return notes + ["the distance of the ligand from its input pose was measured by an earlier version of cuFEP "
                    "in another way (or not stored) and could not be compared with restraints.pose_rmsd_warning_nm "
                    "of this run"]


def prep_made_from(cfg: Config, ligand, membrane: bool, box_shape: str) -> dict:
    """The input and the settings from which the prepared systems of a ligand are built."""
    import hashlib

    data = cfg.model_dump(mode="json")
    protein = Path(cfg.system.protein)
    return {
        "protein_sha256": hashlib.sha256(protein.read_bytes()).hexdigest() if protein.exists() else "file not found",
        "ligand": ligand.digest(),                      # chemistry and coordinates of the SDF record
        "cofactors": data["system"]["cofactors"],
        "forcefield": data["forcefield"],
        "dynamics": {k: data["dynamics"][k] for k in ("cutoff_nm", "hmr_mass_amu", "constraints")},
        "solvation": dict(data["solvation"], box_shape=box_shape),
        "thermodynamics": {k: data["thermodynamics"][k] for k in ("ionic_strength_M", "ph")},
        "membrane": dict({k: data["membrane"][k] for k in ("orientation", "lipid", "padding_nm")} if membrane else {},
                         enabled=bool(membrane)),
    }


def equil_made_with(cfg: Config, leg: str, in_membrane: bool, seed: int) -> dict:
    """The settings that the pre-equilibration of a leg depends on."""
    pre = cfg.sampling.pre_equilibration.complex if leg == "complex" else cfg.sampling.pre_equilibration.solvent
    return {"temperature_K": cfg.thermodynamics.temperature_K, "pressure_bar": cfg.thermodynamics.pressure_bar,
            "friction_per_ps": cfg.dynamics.friction_per_ps, "timestep_fs": float(cfg.dynamics.timestep_fs),
            "pre_equilibration": pre.model_dump(mode="json"),
            "membrane_equilibration_ns": cfg.membrane.equilibration_ns if in_membrane else None,
            "backend": cfg.backend.name, "precision": cfg.backend.precision, "seed": int(seed)}


def restraint_made_with(cfg: Config) -> dict:
    """The settings that the choice of the Boresch restraint depends on."""
    data = cfg.restraints.model_dump(mode="json")
    data.pop("pose_rmsd_warning_nm", None)          # a level for a warning; it does not change the restraint
    return data


def prepare_ligand(cfg: Config, ligand, protein, lig_dir: Path, membrane: bool, box_shape: str, log=print) -> dict:
    """Build and save the solvated complex and the solvated ligand for one ligand."""
    from cufep.backends.cupotential import MIN_BOX_CUTOFF_RATIO
    from cufep.prep.forcefield import ForceFieldSpec, SystemSpec, create_system, make_forcefield
    from cufep.prep.ligand import assign_partial_charges
    from cufep.prep.solvate import SolvationSpec, combine, ligand_atom_indices, solvate

    prep_dir = lig_dir / "prep"
    done = prep_dir / "prep.json"
    made_from = prep_made_from(cfg, ligand, membrane, box_shape)
    info = check_prepared(cfg, ligand, lig_dir, membrane, box_shape, log)
    if info is not None:
        return info
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
            "solvent_box_shape": solvent_shape, "membrane_info": membrane_info, "warnings": warnings,
            # the cuPotential plugin needs a larger box; a system built for it also runs on stock OpenMM
            "box_built_for_backend": cfg.backend.name, "made_from": made_from}
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
                 "result.previous.json", "timing.json")


#: where the samples of a cuPotential run that failed are kept: the first of these names that is free
FAILED_BACKEND_DIR = "hrex_failed_cupotential"


def set_aside_failed_backend(leg_dir: Path) -> Path:
    """Move the samples, lambda windows and start configurations of a leg whose cuPotential run
    failed into ``hrex_failed_cupotential/`` (or ``hrex_failed_cupotential_<n>/`` if that is
    taken), so that the leg can start anew on stock OpenMM. Nothing is deleted."""
    dest, n = leg_dir / FAILED_BACKEND_DIR, 1
    while dest.exists():
        n += 1
        dest = leg_dir / f"{FAILED_BACKEND_DIR}_{n}"
    if (leg_dir / "hrex").exists():
        (leg_dir / "hrex").rename(dest)
    dest.mkdir(exist_ok=True)
    for name in ("schedule.yaml", "start_configs.npz"):
        if (leg_dir / name).exists():
            (leg_dir / name).rename(dest / name)
    return dest


def backend_fallback_refusal(exc: BaseException, backend_name: str, fallback: str, had_samples: bool) -> str | None:
    """Whether a leg whose run raised ``exc`` is repeated on stock OpenMM: None if it is, else why not.

    Only a failure of the backend is a reason: a worker that failed, hung or did not start
    (:class:`cufep.sampling.hrex.WorkerError`), or an error of OpenMM in the controller. A leg
    that already had samples before this run is not repeated from its start automatically:
    that would set its finished samples aside.
    """
    import openmm as mm

    from cufep.sampling.hrex import WorkerError

    if not isinstance(exc, (WorkerError, mm.OpenMMException)):
        return "not a failure of the backend"
    if backend_name != "cupotential":
        return "the leg runs on stock OpenMM already"
    if fallback == "none":
        return "backend.fallback is none"
    if had_samples:
        return "the leg had samples from an earlier run"
    return None


def archive_leg(leg_dir: Path) -> Path | None:
    """Move the lambda windows, samples and result of a leg into ``previous_<n>/`` in its directory.

    The leg then starts anew from its pre-equilibration and restraint, which stay where they
    are. Nothing is deleted. Returns the new directory, or None if the leg had none of these.
    """
    present = [leg_dir / name for name in LEG_RUN_FILES if (leg_dir / name).exists()]
    present += sorted(path for path in leg_dir.glob(f"{FAILED_BACKEND_DIR}_*") if path.is_dir())
    if not present:
        return None
    n = 1
    while (leg_dir / f"previous_{n}").exists():
        n += 1
    dest = leg_dir / f"previous_{n}"
    dest.mkdir()
    for f in present:
        f.rename(dest / f.name)
    # a leg that is being run again has had its result set aside by begin_leg: in the archive
    # it is the result of those windows and samples again
    if (dest / "result.previous.json").exists() and not (dest / "result.json").exists():
        (dest / "result.previous.json").rename(dest / "result.json")
    return dest


#: written when a leg starts and removed when it has written its result or its error: a leg that
#: holds only this file was interrupted (the process was killed, the machine stopped, Ctrl-C)
STARTED_FILE = "started.json"


def _write_json(path: Path, data: dict) -> None:
    """Write a file in one step, so that a process that dies while writing leaves no half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(jsonio.dumps(data, indent=1))
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _leg_identity(leg_dir: Path, job=None) -> dict:
    """Ligand, repeat and leg of a leg directory ``<ligand>/repeat_<n>/<leg>``."""
    if job is not None:
        return {"ligand": job.ligand, "leg": job.leg, "repeat": job.repeat}
    repeat = leg_dir.parent.name.removeprefix("repeat_")
    return {"ligand": leg_dir.parent.parent.name, "leg": leg_dir.name, "repeat": int(repeat) if repeat.isdigit() else 0}


def begin_leg(leg_dir: Path, job=None) -> None:
    """Called before a leg runs (again): from here on it has a result only if this run finishes.

    The result the leg had is kept as ``result.previous.json``: what it gave before this run,
    for instance at 10 ns when it is continued to 20 ns. It is a record and is never used for
    a binding free energy: if this run fails, the leg counts as failed. A marker says that the
    leg has started; it goes when the leg has written its result or its error.
    """
    leg_dir.mkdir(parents=True, exist_ok=True)
    result = leg_dir / "result.json"
    if result.exists():
        result.replace(leg_dir / "result.previous.json")
    (leg_dir / "error.json").unlink(missing_ok=True)
    _write_json(leg_dir / STARTED_FILE, dict(_leg_identity(leg_dir, job), started=time.strftime("%Y-%m-%d %H:%M:%S")))


def write_leg_result(leg_dir: Path, result: dict) -> None:
    """The result of a leg that has finished."""
    _write_json(leg_dir / "result.json", result)
    (leg_dir / STARTED_FILE).unlink(missing_ok=True)


def write_leg_error(leg_dir: Path, message: str, job=None, trace: str | None = None) -> dict:
    """Record that a leg has failed; returns what was written to ``error.json``."""
    leg_dir.mkdir(parents=True, exist_ok=True)
    err = dict(_leg_identity(leg_dir, job), error=message)
    if trace:
        err["traceback"] = trace
    before = _read_json(leg_dir / "result.previous.json")
    if before is not None:
        err["previous_result"] = {"file": "result.previous.json", "production_ns": before.get("production_ns"),
                                  "delta_g_decouple_kJ_mol": before.get("delta_g_decouple")}
    (leg_dir / "result.json").unlink(missing_ok=True)       # no result of a run that failed
    _write_json(leg_dir / "error.json", err)
    (leg_dir / STARTED_FILE).unlink(missing_ok=True)
    return err


def leg_state(leg_dir: Path) -> tuple[str, dict | None]:
    """What a leg directory holds: ("finished", result), ("failed", error) or ("missing", None).

    A leg that has started and written neither a result nor an error was interrupted and is
    "failed"; so is one whose result file cannot be read. A directory written before results
    were set aside at the start of a run can hold a result and a later error; the later file
    counts.
    """
    result, error = leg_dir / "result.json", leg_dir / "error.json"
    if result.exists() and error.exists():
        newer = result if result.stat().st_mtime >= error.stat().st_mtime else error
    else:
        newer = result if result.exists() else error if error.exists() else None
    if newer is not None:
        data = _read_json(newer)
        if data is None:
            return "failed", dict(_leg_identity(leg_dir), error=f"{newer.name} of this leg cannot be read (the file "
                                                                 "is incomplete)")
        return ("finished" if newer is result else "failed"), data
    started = _read_json(leg_dir / STARTED_FILE) if (leg_dir / STARTED_FILE).exists() else None
    if (leg_dir / STARTED_FILE).exists():
        when = "" if not started or not started.get("started") else f" (it had started at {started['started']})"
        return "failed", dict(_leg_identity(leg_dir), interrupted=True,
                              error=f"the run of this leg was interrupted before it wrote a result{when}")
    return "missing", None


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
    state, data = leg_state(job.leg_dir)
    if data is not None:
        return dict(_leg_identity(job.leg_dir, job), **data) if state == "failed" else data
    return {"ligand": job.ligand, "leg": job.leg, "repeat": job.repeat,
            "error": "the job process ended without writing a result (killed?)"}


def record_ended_job(job: LegJob, exit_code: int | None) -> dict:
    """What a job process that has ended left behind; if it left neither a result nor an error,
    that it died is written to ``error.json`` (with its exit code), so that the files of the leg
    say the same as the scheduler that saw it."""
    state, data = leg_state(job.leg_dir)
    if state == "finished":
        return data
    if not (job.leg_dir / "error.json").exists():
        signal = "" if exit_code is None or exit_code >= 0 else f", killed by signal {-exit_code}"
        return write_leg_error(job.leg_dir, f"the job process ended without a result (exit code {exit_code}{signal})",
                               job)
    return read_result(job)


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

    from cufep.alchemy.restraints import (BoreschRestraint, RestraintSearchSettings, find_boresch_restraint,
                                          ligand_symmetry_number, with_force_constants_from_fluctuations)
    from cufep.alchemy.schedule import AlsSettings, Schedule, adaptive_schedule, schedule_from_windows
    from cufep.backends import get_backend
    from cufep.mps import thread_percentage
    from cufep.sampling.equilibrate import EquilibrationSettings, pre_equilibrate
    from cufep.sampling.hrex import ReplicaExchangeSampler, WindowPool
    from cufep.sampling.states import STAGES_COMPLEX, STAGES_SOLVENT, stage_state
    from cufep.sampling.storage import read_samples, stored_backend, stored_conditions
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
    seed = leg_seed(cfg, job.repeat, job.leg)
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
    in_membrane = is_complex and barostat == "membrane"
    equil_from = equil_made_with(cfg, job.leg, in_membrane, seed)
    if not equil_file.exists():
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
            area_tolerance=cfg.membrane.area_tolerance, water_tolerance=cfg.membrane.pocket_water_tolerance,
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
                 pocket_waters=eq.pocket_waters, area_nm2=eq.area_nm2, made_with=json.dumps(equil_from),
                 membrane_settled=json.dumps(eq.settled))
        tmp.replace(equil_file)
        timed("pre_equilibration", t0)
        log(f"pre-equilibration done in {time.perf_counter() - t0:.0f} s (time step {eq.timestep_fs:g} fs)")
    eq = np.load(equil_file)
    # files of this leg that an earlier version wrote without a record of what they were made from
    origin_not_recorded: list[str] = [] if prep_info.get("made_from") is not None else ["the prepared systems"]
    if not check_made_from("the pre-equilibration of this leg", equil_file,
                           json.loads(str(eq["made_with"])) if "made_with" in eq.files else None, equil_from, log,
                           contradictions=equil_contradictions(eq, cfg, job.leg, in_membrane)):
        origin_not_recorded.append("the pre-equilibration")
    timestep_fs = float(eq["timestep_fs"])
    # the notes of the pre-equilibration and of the restraint are made from their files when the
    # result is made (notes_of_leg_files), with the levels of the run that makes it

    # ---- 2. restraint (complex only) --------------------------------------------------------
    t_restraint = time.perf_counter()
    restraint = None
    symmetry = 1
    if is_complex:
        from openff.toolkit import Molecule
        rdmol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
        symmetry = ligand_symmetry_number(rdmol)
        rfile = leg_dir / "restraint.json"
        restraint_from = restraint_made_with(cfg)
        if rfile.exists():
            stored = json.loads(rfile.read_text())
            if not check_made_from("the restraint of this leg", rfile, stored.get("made_with"), restraint_from, log,
                                   contradictions=restraint_contradictions(stored, cfg)):
                origin_not_recorded.append("the restraint")
            restraint = BoreschRestraint.from_dict(stored["restraint"])
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
                # the pose is compared with the prepared input: the frames kept for the search begin
                # only after the first part of the pre-equilibration
                found = find_boresch_restraint(prep["topology"], full, prep["ligand"], rdmol, T, settings,
                                               input_pose_nm=prep["positions"], frame_boxes_nm=eq["frame_boxes"],
                                               input_box_nm=prep["box"])
                restraint = found.restraint
                if rc.force_constants == "auto":
                    restraint = with_force_constants_from_fluctuations(restraint, found.fluctuations, T)
                info = {"selection": "auto", "force_constants": rc.force_constants, "score": found.score,
                        "fluctuations": found.fluctuations,
                        "ligand_rmsd_A": 10 * found.ligand_rmsd_nm, "ligand_rmsd_reference": "input pose",
                        "candidates_tested": found.n_candidates, "notes": found.notes}
                atoms = list(prep["topology"].atoms())
                label = lambda i: f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}"   # noqa: E731
                log("restraint: host " + ", ".join(label(i) for i in restraint.host_atoms) + " | guest "
                    + ", ".join(label(i) for i in restraint.guest_atoms)
                    + f" | r0 {10 * restraint.r0:.2f} A, ligand RMSD {10 * found.ligand_rmsd_nm:.2f} A")
            rfile.write_text(json.dumps({"restraint": restraint.to_dict(), "info": info,
                                         "ligand_symmetry_number": symmetry, "made_with": restraint_from}, indent=1))
        if symmetry > 1:
            notes.append(f"the ligand has {symmetry} heavy-atom graph symmetries; no symmetry correction is applied")

    timed("restraint", t_restraint)

    # ---- 3. alchemical System ------------------------------------------------------------------
    built = alchemical_leg(cfg, system, prep, eq["box"], restraint, is_complex)
    alch, alch_xml, tail = built.alch, built.xml, built.tail
    rest2 = cfg.sampling.rest2
    rest2_on, hot_atoms, hot_residues = built.rest2_on, built.hot_atoms, built.hot_residues
    if rest2_on:
        log(f"REST2: hot region = ligand ({len(prep['ligand'])} atoms)"
            + (f" and {len(hot_atoms)} atoms of {len(hot_residues)} protein residues ({' '.join(hot_residues)})"
               if hot_atoms else "")
            + f"; {alch.rest2_torsions} torsion terms scaled; {rest2.scale:g} times the temperature in the middle of the leg")
    backend_name, bnotes = _choose_backend(cfg, alch.system, log)
    notes += bnotes
    # samples of two backends are never mixed: a leg that has samples goes on with their backend
    made_with = stored_backend(leg_dir / "hrex")
    if made_with is not None and made_with != backend_name:
        log(f"this leg goes on with {made_with}, the backend its samples were made with "
            f"(this run would have chosen {backend_name})")
        notes.append(f"the leg was continued on {made_with}, the backend its samples were made with")
        backend_name = made_with
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
            _schedule_of_this_leg(src, job.leg, stages)          # before it is taken over
            shutil.copy(src, sched_file)
        if sched_file.exists() and start_file.exists():
            schedule = _schedule_of_this_leg(sched_file, job.leg, stages)
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
                    schedule = _schedule_of_this_leg(sched_file, job.leg, stages)
                else:
                    schedule = schedule_from_windows(stages, ls.windows.for_leg(job.leg),
                                                     WindowsConfig().for_leg(job.leg), rest2.scale)
                cfgs = _walk_schedule(ctx, integ, schedule, ls.fixed_walk_ps, timestep_fs)
            else:
                a_ = ls.adaptive
                als = AlsSettings(overlap_threshold=a_.overlap_threshold, candidates=a_.candidates,
                                  initial_windows=a_.initial_windows.for_leg(job.leg), window_ps=a_.window(job.leg),
                                  equilibration_ps=a_.equilibration(job.leg), sample_interval_ps=a_.sample_interval_ps,
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
        spec = leg_spec(cfg, built, [s.to_dict() for s in schedule.states], barostat, timestep_fs, backend_name,
                        seed + 17, name=job.tag, frame_atoms=frame_atoms, frame_ligand=frame_ligand, frame_ca=frame_ca,
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
            if sampler.unrecorded:
                # samples of an earlier version, whose conditions are not recorded: a leg that has
                # all its iterations is analysed as it is; nothing is added to such samples
                problems = unrecorded_samples_problems(
                    sampler.storage.attributes(), sampler.storage.n_iterations, leg_dir, cfg, job.leg,
                    job.production_ns, [s.to_dict() for s in schedule.states], timestep_fs)
                if problems:
                    refuse_unrecorded_samples(leg_dir, sampler.storage.n_iterations, problems)
                log(f"the {sampler.storage.n_iterations} iterations of samples of this leg were written by a "
                    "version that did not record their conditions; they are analysed as they are")
                if "the samples" not in origin_not_recorded:
                    origin_not_recorded.append("the samples")
                timed("sampling", t_sampling)
                return read_samples(hrex_dir), schedule
            if not sampler.restore():
                sampler.initialize(*configs)
            else:
                log(f"resumed at iteration {sampler.iteration}")
            # before the first iteration: what an analysis of an interrupted run needs to know
            sampler.storage.set_plan(n_equil, total, analysis_record(cfg))
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
            # the result is for the length that was asked for, and every iteration is stored once
            if sampler.storage.n_iterations != sampler.iteration or sampler.iteration < total:
                raise WorkflowError(
                    f"the leg holds {sampler.storage.n_iterations} iterations of samples after {sampler.iteration} "
                    f"iterations were run and {total} were asked for; its samples are left as they are")
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

    # a leg that has samples from an earlier run is continued, never repeated from its start
    had_samples = stored_conditions(leg_dir / "hrex")[0] > 0
    try:
        samples, schedule = run_with(backend_name)
    except Exception as exc:  # noqa: BLE001  classified here; everything that is not repeated is raised
        refusal = backend_fallback_refusal(exc, backend_name, cfg.backend.fallback, had_samples)
        if refusal is not None:
            if refusal == "the leg had samples from an earlier run":
                raise WorkflowError(
                    f"the cuPotential run of this leg failed ({str(exc)[:300]}). The leg had samples from an earlier "
                    "run, so it is not repeated on stock OpenMM from its start: its samples and checkpoint are left "
                    "in hrex/. Run the leg again to continue it on cuPotential, or start it anew on stock OpenMM "
                    "with backend.name: openmm and --reschedule (its samples are then kept in previous_<n>/)") from exc
            raise
        kept = set_aside_failed_backend(leg_dir)
        log(f"cuPotential run failed ({str(exc)[:300]}); repeating this leg on stock OpenMM; what the failed run "
            f"left is kept in {kept.name}/")
        notes.append("the cuPotential run failed; the whole leg was repeated on stock OpenMM "
                     f"(samples from the two backends are never mixed; the failed run is kept in {kept.name}/)")
        backend_name = "openmm"
        samples, schedule = run_with(backend_name)

    # ---- 6. analysis ------------------------------------------------------------------------------
    t_analysis = time.perf_counter()
    run_record = {
        "ligand": job.ligand, "leg": job.leg, "repeat": job.repeat, "backend": backend_name, "timestep_fs": timestep_fs,
        "ligand_symmetry_number": symmetry, "notes": list(notes), "origin_not_recorded": list(origin_not_recorded),
        "prepared_systems_sha256": record_digest(prep_info.get("made_from")),
        "rest2": {"scale": float(rest2.scale) if rest2_on else 1.0, "hot_protein_atoms": len(hot_atoms),
                  "hot_residues": hot_residues, "torsion_terms_scaled": alch.rest2_torsions,
                  "scale_of_window": [float(st.rest2_scale) for st in schedule.states]}}
    result = leg_result_from_samples(leg_dir, cfg, run_record, samples=samples, schedule=schedule)
    analysis_dict, correction, total = result["analysis"], result["lrc"], result["delta_g_decouple"]
    timed("analysis", t_analysis)
    timing.update(finished=time.time(), backend=backend_name, atoms=alch.system.getNumParticles(),
                  ligand=job.ligand, leg=job.leg, repeat=job.repeat)
    (leg_dir / "timing.json").write_text(json.dumps(timing, indent=1))
    write_leg_result(leg_dir, result)

    def number(value, digits: int = 2, scale: float = 1.0) -> str:
        return "none" if value is None else f"{value / scale:.{digits}f}"

    slowest = analysis_dict["slowest_window"]
    blocks = analysis_dict.get("delta_g_error_n_blocks")
    error_from = f"from {blocks} time blocks" if analysis_dict.get("delta_g_error_from") == "time blocks" else "analytical"
    log(f"dG(decouple) = {number(total, 2, KCAL)} ± {number(analysis_dict['delta_g_error'], 2, KCAL)} kcal/mol "
        f"(standard error {error_from}; MBAR {number(analysis_dict['delta_g_mbar'], 2, KCAL)} ± "
        f"{number(analysis_dict['delta_g_mbar_error'], 2, KCAL)}, tail {number(-correction, 2, KCAL)}); "
        f"min overlap {number(analysis_dict['min_overlap'], 3)} (among the samples of the estimate "
        f"{number(analysis_dict['min_overlap_used'], 3)}), round trips {analysis_dict['round_trips']}, fewest "
        f"independent samples in a window {analysis_dict['effective_samples_per_state']:.0f}, of which MBAR gets "
        f"{analysis_dict['window_kept_samples'][slowest]} (window {slowest})")
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


def _schedule_of_this_leg(path: Path, leg: str, stages: list[str]):
    """Load a lambda schedule and make sure it is one of this leg (see ``Schedule.check_leg``)."""
    from cufep.alchemy.schedule import Schedule

    try:
        schedule = Schedule.load(path)
        schedule.check_leg(leg, stages)
    except ValueError as exc:
        raise WorkflowError(f"the lambda schedule {path} cannot be used for the {leg} leg: {exc}") from exc
    return schedule


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


def leg_quality(analysis: dict, cfg) -> dict:
    """The checks of :func:`cufep.analysis.mbar.quality_checks` with the levels of the configuration
    (the whole configuration or its ``analysis`` section)."""
    from cufep.analysis.mbar import quality_checks

    a = getattr(cfg, "analysis", cfg)
    return quality_checks(analysis, a.min_overlap, a.min_effective_samples, a.halves_tolerance_kT,
                          a.estimators_tolerance_kcal_mol)


def analysis_settings(source):
    """The settings with which the result of a leg is made from its samples and files.

    They are the ``analysis`` section and three levels for notes that are made from the files of
    the leg: ``membrane.area_tolerance`` and ``membrane.pocket_water_tolerance`` (whether the
    bilayer had settled) and ``restraints.pose_rmsd_warning_nm`` (whether the ligand left its
    pose). The run of a leg, a result that is made again and ``cufep analyze`` use the same
    ones. ``source`` is a whole configuration (returned as it is), an object with these three
    sections, or an ``analysis`` section alone, which gets the default levels.
    """
    if all(hasattr(source, section) for section in ("analysis", "membrane", "restraints")):
        return source
    settings = analysis_settings_of(None)
    settings.analysis = getattr(source, "analysis", source)
    return settings


def analysis_record(cfg) -> dict:
    """The settings the result of a leg is made with (:func:`analysis_settings`) and the version of
    the method, as the result records them."""
    from cufep.analysis.mbar import ANALYSIS_VERSION

    settings = analysis_settings(cfg)
    a = settings.analysis
    return {"version": ANALYSIS_VERSION, "lrc": a.lrc, "min_overlap": float(a.min_overlap),
            "min_effective_samples": float(a.min_effective_samples), "halves_tolerance_kT": float(a.halves_tolerance_kT),
            "estimators_tolerance_kcal_mol": float(a.estimators_tolerance_kcal_mol),
            "convergence_points": int(a.convergence_points), "convergence_blocks": int(a.convergence_blocks),
            "error_blocks": [int(count) for count in a.error_blocks], "part_error_blocks": int(a.part_error_blocks),
            "membrane_area_tolerance": float(settings.membrane.area_tolerance),
            "membrane_pocket_water_tolerance": float(settings.membrane.pocket_water_tolerance),
            "pose_rmsd_warning_nm": float(settings.restraints.pose_rmsd_warning_nm)}


def analysis_settings_of(record: dict | None):
    """The settings that a stored :func:`analysis_record` describes (:func:`analysis_settings`;
    defaults for what it lacks)."""
    from types import SimpleNamespace

    from cufep.config.schema import AnalysisConfig, MembraneConfig, RestraintConfig

    record = record or {}
    membrane, restraints = MembraneConfig(), RestraintConfig()
    return SimpleNamespace(
        analysis=AnalysisConfig(**{key: value for key, value in record.items() if key in AnalysisConfig.model_fields}),
        membrane=SimpleNamespace(
            area_tolerance=float(record.get("membrane_area_tolerance", membrane.area_tolerance)),
            pocket_water_tolerance=float(record.get("membrane_pocket_water_tolerance", membrane.pocket_water_tolerance))),
        restraints=SimpleNamespace(
            pose_rmsd_warning_nm=float(record.get("pose_rmsd_warning_nm", restraints.pose_rmsd_warning_nm))))


def notes_of_leg_files(leg_dir: Path, settings, conditions: dict) -> tuple[list[str], tuple[str, ...]]:
    """The notes that the pre-equilibration and the restraint of a leg carry, from the files of the
    leg, with the verdicts on the bilayer and on the pose made with the levels of ``settings``.

    Returns the notes and how those begin whose verdict was made here: such notes of an earlier
    run are replaced by them. A file that is not there gives no notes and replaces none.
    ``conditions`` are those of the samples (the barostat says whether the leg is in a bilayer;
    samples that do not record it leave that to the pre-equilibration file).
    """
    from cufep.alchemy.restraints import POSE_NOTE
    from cufep.sampling.equilibrate import BILAYER_NOTE

    notes: list[str] = []
    judged: list[str] = []
    equil_file, restraint_file = leg_dir / "equil.npz", leg_dir / "restraint.json"
    if equil_file.exists():
        eq = np.load(equil_file)
        barostat = conditions.get("barostat")
        in_membrane = barostat == "membrane" if barostat is not None else bool(equil_in_membrane(eq))
        notes += equil_notes(eq, settings, in_membrane)
        judged.append(BILAYER_NOTE)
    stored = _read_json(restraint_file) if restraint_file.exists() else None
    if stored:
        notes += [n for n in restraint_notes(stored, settings) if n not in notes]
        judged.append(POSE_NOTE)
    return notes, tuple(judged)


#: how the notes begin that the analysis of a leg writes; a result that is made again from its
#: samples gets them anew
_ANALYSIS_NOTES = ("NOT RELIABLE", "NOT CONVERGED", "convergence not assessed", "few independent samples",
                   "the standard error of this leg is the one MBAR gives", "the standard error of this leg is the "
                   "analytical one", "parts of the production without an estimate",
                   "the two halves of the production", "half of the production does not",
                   "BAR between neighbouring lambda windows", "written by an earlier version of cuFEP without a record")
#: what the run of a leg knows and its samples do not hold
_RUN_KEYS = ("ligand", "leg", "repeat", "backend", "timestep_fs", "ligand_symmetry_number", "rest2",
             "origin_not_recorded", "prepared_systems_sha256")


def restraint_release_of(leg_dir: Path, attrs: dict, temperature_K: float) -> tuple[float, float]:
    """Free energy of releasing the Boresch restraint of a leg (numerical, analytical; kJ/mol), from
    its ``restraint.json`` at the given temperature; (0, 0) for a leg without a restrain stage.

    It is computed where the result is made and at the temperature of the samples, so that it
    cannot be taken at another temperature than the free energy it is added to. A leg whose
    stored states have a restrain stage must have the file.
    """
    from cufep.alchemy.restraints import BoreschRestraint

    states = attrs.get("states") if isinstance(attrs.get("states"), list) else []
    restrained = any(float(state.get("lambda_restraints", 1.0)) < 1.0 for state in states)
    # the directory of an earlier set of windows of the leg (previous_<n>/) has it one level up
    candidates = [leg_dir / "restraint.json"] + ([leg_dir.parent / "restraint.json"]
                                                 if leg_dir.name.startswith("previous_") else [])
    found = next((f for f in candidates if f.exists()), None)
    if found is None:
        if restrained:
            raise WorkflowError(f"the samples in {leg_dir} are of a leg with a restrained ligand, but its "
                                "restraint.json is not there: the release of the restraint cannot be computed")
        return 0.0, 0.0
    if not restrained and states:
        return 0.0, 0.0
    restraint = BoreschRestraint.from_dict((_read_json(found) or {}).get("restraint") or {})
    return (float(restraint.release_free_energy(temperature_K, "numerical")),
            float(restraint.release_free_energy(temperature_K, "analytical")))


def unrecorded_samples_problems(attrs: dict, n_stored: int, leg_dir: Path, cfg: Config, leg: str, production_ns: float,
                                states: list[dict] | None, timestep_fs: float) -> list[str]:
    """Why samples whose conditions an earlier version did not record cannot stand for what this
    run asks for; empty if they can.

    Such samples are never added to, so they must be what the run asks for as they are: what
    their file does hold (temperature, lambda states, length of an iteration, where the
    production begins) must agree with the run, and the production they hold, counted with their
    own iteration length from their own boundary, must be as long as the one asked for.
    ``states`` are those of the leg's lambda schedule (None if there is none), ``timestep_fs``
    the time step the leg uses.
    """
    from cufep.sampling.states import AlchemicalState
    from cufep.sampling.storage import _same, equilibration_boundary

    def normal(sequence):
        return [AlchemicalState(**state).to_dict() for state in sequence]

    problems = []
    if "temperature_K" in attrs and not _same(float(attrs["temperature_K"]), float(cfg.thermodynamics.temperature_K)):
        problems.append(f"temperature: {float(attrs['temperature_K']):g} K -> {float(cfg.thermodynamics.temperature_K):g} K")
    if states is not None and isinstance(attrs.get("states"), list) and not _same(normal(attrs["states"]), normal(states)):
        problems.append("the lambda states")
    if "iteration_ps" not in attrs:
        return problems + ["the file does not record the length of an iteration"]
    stored_ps = float(attrs["iteration_ps"])
    # compared in whole steps: 2.5 ps at 3 fs are 833 steps, 2.499 ps
    asked_ps = max(1, int(round(cfg.sampling.time_per_iteration_ps * 1000.0 / timestep_fs))) * timestep_fs * 1e-3
    if abs(stored_ps - asked_ps) > 1e-6:
        problems.append(f"length of an iteration: {stored_ps:g} ps -> {asked_ps:g} ps")
    boundary = equilibration_boundary(attrs, leg_dir)
    if boundary is None:
        return problems + ["the file does not record where its production begins (the run that wrote it did not "
                           "finish); cufep analyze --equilibration-ns reads such samples"]
    stored_equil, asked_equil = boundary * stored_ps / 1000.0, cfg.sampling.equilibration(leg)
    if abs(boundary - max(1, int(round(asked_equil * 1000.0 / stored_ps)))) > 0:
        problems.append(f"equilibration per lambda window: {stored_equil:g} ns -> {asked_equil:g} ns")
    held = max(0, n_stored - boundary) * stored_ps / 1000.0
    if held < production_ns - 1e-9:
        problems.append(f"production per lambda window: the samples hold {held:g} ns and {production_ns:g} ns are asked "
                        "for (samples of unrecorded conditions are not added to)")
    return problems


def refuse_unrecorded_samples(leg_dir: Path, n_stored: int, problems: list[str]) -> None:
    from cufep.sampling.storage import ConditionsError

    raise ConditionsError(
        f"the {n_stored} iterations of samples in {leg_dir / 'hrex'} were written by a version of cuFEP that did not "
        f"record the conditions they were made under, and they are not what this run asks for ({'; '.join(problems)}). "
        "They are left as they are and can be analysed (cufep analyze). With these settings the leg has to be "
        "started anew with --reschedule (the samples it has are kept in previous_<n>/), or the calculation needs a "
        "new output.dir.")


def run_record_of(result: dict | None, leg_dir: Path | None = None) -> dict:
    """The record of the run in a leg's result; for a result of a version that kept none, what its
    fields give (its notes without those of the analysis); for no result, what the path gives."""
    if result and isinstance(result.get("run"), dict):
        return result["run"]
    record = dict(_leg_identity(leg_dir)) if leg_dir is not None else {}
    if result:
        structure = set((result.get("structure") or {}).get("warnings", []))
        record.update({key: result[key] for key in _RUN_KEYS if key in result})
        record["notes"] = [n for n in result.get("notes", []) if not n.startswith(_ANALYSIS_NOTES) and n not in structure]
    return record


def leg_result_from_samples(leg_dir: Path, analysis, run: dict, samples: dict | None = None, schedule=None,
                            n_equilibration: int | None = None) -> dict:
    """The result of a leg from the samples in its directory. Needs no GPU.

    ``analysis`` gives the settings of :func:`analysis_settings` (a whole configuration, or its
    ``analysis`` section alone), and nothing else of a configuration enters: temperature, length
    of an iteration and the end of the equilibration are those of the samples, so that a leg
    cannot be analysed as if it had been run under the settings of another run. ``run`` holds
    what only the run of the leg knows (:data:`_RUN_KEYS` and its notes). The notes of the
    pre-equilibration and of the restraint are made from the files of the leg
    (:func:`notes_of_leg_files`). The same function serves a leg that has just been sampled,
    ``cufep analyze``, and a result that is made again because the method or the settings of
    the analysis have changed.
    """
    from cufep.alchemy.schedule import Schedule
    from cufep.analysis.mbar import NoCorrection, analyze_leg
    from cufep.analysis.structure import structural_analysis
    from cufep.sampling.storage import equilibration_boundary, read_samples

    settings = analysis_settings(analysis)
    a = settings.analysis
    if samples is None:
        samples = read_samples(leg_dir / "hrex")
    attrs = samples["attrs"]
    if "temperature_K" not in attrs or "iteration_ps" not in attrs:
        raise WorkflowError(f"the samples in {leg_dir} do not record their temperature and the length of an iteration")
    temperature, iteration_ps = float(attrs["temperature_K"]), float(attrs["iteration_ps"])
    if n_equilibration is None:
        n_equilibration = equilibration_boundary(attrs, leg_dir)
        if n_equilibration is None:
            raise WorkflowError(f"the samples in {leg_dir} are of a run that did not finish and do not record where "
                                "its production begins")
    if a.lrc != "none" and "lrc_s6" not in attrs:
        raise WorkflowError(f"the samples in {leg_dir} were made without the data for a correction of the dispersion "
                            f"tail (analysis.lrc: none) and cannot be analysed with analysis.lrc: {a.lrc}")
    if schedule is None and (leg_dir / "schedule.yaml").exists():
        schedule = Schedule.load(leg_dir / "schedule.yaml")
    try:
        res = analyze_leg(samples, temperature, n_equilibration=n_equilibration, n_fractions=a.convergence_points,
                          n_blocks=a.convergence_blocks, error_blocks=a.error_blocks,
                          part_error_blocks=a.part_error_blocks, tail=a.lrc, min_overlap=a.min_overlap)
    except NoCorrection as exc:
        raise WorkflowError(f"the samples in {leg_dir}: {exc}") from exc
    correction = res.correction
    structure = structural_analysis(samples, n_equilibration)
    conditions = attrs.get("conditions") if isinstance(attrs.get("conditions"), dict) else {}
    # the notes of the pre-equilibration and the restraint, with the verdicts on the bilayer and the
    # pose by the levels of these settings; then what the run noted besides
    notes, judged = notes_of_leg_files(leg_dir, settings, conditions)
    notes += [n for n in (run.get("notes") or []) if n not in notes and not (judged and n.startswith(judged))]
    notes += [n for n in (structure or {}).get("warnings", []) if n not in notes]
    origin_not_recorded = list(run.get("origin_not_recorded") or [])
    # the pressure is the one the samples were made under; samples that do not record it have none
    pressure = conditions.get("pressure_bar")
    release, release_analytical = restraint_release_of(leg_dir, attrs, temperature)
    # the outer radius to which the tail energy of these samples was recorded; the energies cannot
    # be turned into those of another radius
    tail = conditions.get("lrc") if isinstance(conditions.get("lrc"), dict) else {}
    result = {
        "ligand": run.get("ligand"), "leg": run.get("leg"), "repeat": run.get("repeat"), "backend": run.get("backend"),
        "temperature_K": temperature, "pressure_bar": pressure,
        "prepared_systems_sha256": run.get("prepared_systems_sha256"),
        "timestep_fs": run.get("timestep_fs"), "n_states": res.n_states,
        "production_ns": float(res.n_iterations_used * iteration_ps / 1000.0),
        # decoupling with the full LJ range (MBAR minus the tail correction of the same iterations),
        # its standard error (cufep.analysis.mbar.TimeRangeEstimator), and the MBAR part alone
        "delta_g_decouple": res.delta_g, "delta_g_error": res.delta_g_error,
        "delta_g_mbar": res.delta_g_mbar, "delta_g_mbar_error": res.delta_g_mbar_error,
        "lrc": correction, "lrc_method": a.lrc, "lrc_long_cutoff_nm": tail.get("long_cutoff_nm"),
        "lrc_reweight": res.lrc_reweight,
        "lrc_isotropic": res.lrc_isotropic, "lrc_error": res.correction_error, "structure": structure,
        "restraint_release": release, "restraint_release_analytical": release_analytical,
        "ligand_symmetry_number": run.get("ligand_symmetry_number", 1), "analysis": res.to_dict(), "units": "kJ/mol",
        "analysis_record": analysis_record(settings), "run": run,
    }
    result["quality"] = leg_quality(result["analysis"], a)
    result["lambda_windows"] = schedule.windows() if schedule is not None else {}
    # what this leg used although an earlier version had written it without saying what it was made from
    result["origin_not_recorded"] = origin_not_recorded
    if origin_not_recorded:
        notes.append("written by an earlier version of cuFEP without a record of what it was made from, and used "
                     f"as it is: {', '.join(origin_not_recorded)}. Nothing in these files contradicts this run; "
                     "that they were made with its settings could not be checked")
    if schedule is not None:
        notes += [n for n in schedule.notes if n not in notes]
    result["notes"] = notes + quality_warnings(result, a)
    # windows that adaptive lambda scheduling added to its start, per stage (None: fixed windows)
    start = schedule.provenance.get("initial_windows") if schedule is not None and schedule.mode == "adaptive" else None
    result["lambda_windows_added"] = (
        {name: len(values) - int(start[name]) for name, values in schedule.windows().items() if name in start}
        if start else None)
    result["rest2"] = run.get("rest2")
    return result


def quality_warnings(result: dict, cfg) -> list[str]:
    """Warnings about a finished leg, from its checks. None of them changes what is run.

    The overlap of neighbouring lambda windows comes first: below the level, the free energy is
    NOT RELIABLE. The other messages give a number and the level the literature names for it;
    they say that more sampling is advisable, not that the result is wrong.
    """
    cfg = getattr(cfg, "analysis", cfg)            # the whole configuration or its analysis section
    a = result["analysis"]
    q = result.get("quality")
    if not q or "halves" not in q:                 # none, or the checks of an older version
        q = leg_quality(a, cfg)
    leg = result.get("leg", "LEG")
    windows = result.get("lambda_windows") or {}
    longer = f"cufep run CONFIG --ligand NAME --leg {leg} --production-ns NS (or sampling.production_ns)"

    def place(k: int) -> str:
        where = window_stage(windows, min(k, max(0, a.get("n_states", k + 2) - 2)))
        return "" if where is None else f" ({where[0]} stage)"

    def pair(k: int) -> str:
        where = window_stage(windows, k)
        return "" if where is None else f" ({where[0]} stage, lambda {where[1]:g} and {where[2]:g})"

    out = []
    ov = q["overlap"]
    all_ok = ov.get("all_ok", ov["ok"])            # the matrix of all iterations, equal numbers per window
    more_windows = (f"lambda_schedule.adaptive.initial_windows, or lambda_schedule.windows with mode: fixed) "
                    f"and run this leg again: cufep run CONFIG --ligand NAME --leg {leg} --reschedule")
    if not all_ok:
        k = ov["windows"][0]
        lowest = "not defined" if ov["lowest"] is None else f"{ov['lowest']:.3f}"
        out.append(f"NOT RELIABLE: the overlap between lambda windows {k} and {k + 1}{pair(k)} is "
                   f"{lowest}, below {cfg.min_overlap:g}. Give that stage more lambda "
                   f"windows ({more_windows}")
    used = ov.get("used")
    if used and used["ok"] is False and not (not all_ok and used["windows"] == ov["windows"]):
        k = used["windows"][0]
        counts = "" if not used.get("samples") else (f"; window {k} gives it {used['samples'][0]} samples and window "
                                                     f"{k + 1} gives {used['samples'][1]}")
        out.append(f"NOT RELIABLE: among the samples the free energy is estimated from, the overlap between lambda "
                   f"windows {k} and {k + 1}{pair(k)} is {used['lowest']:.3f}, below {cfg.min_overlap:g}"
                   f"{counts}. A longer production gives a slow window more samples: {longer}. More lambda windows "
                   f"in that stage also raise it: change lambda_schedule and run the leg again with --reschedule")
    smp = q["samples"]
    if smp["ok"] is False:
        towards = {-1: "previous", 1: "next"}.get(smp.get("neighbour"), "next")
        kept = "" if smp.get("kept") is None else f" and gives MBAR {smp['kept']} samples"
        ratio = smp.get("error_ratio")
        left = ""
        if ratio is not None and ratio > 1.0:
            # the windows are decorrelated one by one; what is left is the correlation between them
            left = (f" The standard error from time blocks is {ratio:.1f} times the one MBAR gives for these samples: "
                    f"they are correlated between the lambda windows and count as {smp['independent']:.0f} "
                    f"independent ones.")
        out.append(f"few independent samples: lambda window {smp['window']}{place(smp['window'])} has "
                   f"{smp['fewest']:.0f}{kept} (its energy difference to the {towards} window is correlated over "
                   f"{smp['statistical_inefficiency']:.0f} iterations); {smp['threshold']:g} are recommended for MBAR."
                   f"{left} The free energy, its uncertainty and the comparison of the halves rest on these samples. "
                   f"A longer production gives more: {longer}")
    if a.get("delta_g_error_note"):
        out.append("the standard error of this leg is the analytical one of MBAR and the tail correction, which takes "
                   f"the samples as independent of each other ({a['delta_g_error_note']}). A longer production allows "
                   f"the error from time blocks: {longer}")
    if a.get("parts_not_determined"):
        shown = a["parts_not_determined"]
        out.append(f"parts of the production without an estimate ({len(shown)}; they are not shown in analysis.png): "
                   + "; ".join(shown[:3]) + ("; ..." if len(shown) > 3 else ""))
    halves = q["halves"]
    if halves["ok"] is False:
        if halves.get("difference") is not None:
            in_sigma = halves.get("difference_sigma")
            sigma = "" if in_sigma is None else f", {in_sigma:.1f} standard errors of the difference"
            error = "" if q.get("uncertainty") is None else f" ± {q['uncertainty'] / KCAL:.2f}"
            out.append(f"the two halves of the production differ by {halves['difference'] / KCAL:.2f} kcal/mol "
                       f"({halves['difference_kT']:.1f} kT{sigma}; first half {halves['first_half'] / KCAL:.2f}, "
                       f"second half {halves['second_half'] / KCAL:.2f}, all of it {halves['final'] / KCAL:.2f}"
                       f"{error}). More than {halves['tolerance_kT']:g} kT indicates poor "
                       f"convergence; a longer production shows whether it is: {longer}")
        else:
            out.append(f"half of the production does not determine the free energy of this leg; run it longer: {longer}")
    elif halves["ok"] is None and halves.get("reason"):
        out.append(f"the two halves of the production could not be compared: {halves['reason']}. A longer "
                   f"production gives the halves enough samples: {longer}")
    est = q["estimators"]
    if est["ok"] is False:
        out.append(f"BAR between neighbouring lambda windows gives {est['bar'] / KCAL:.2f} kcal/mol and MBAR "
                   f"{est['mbar'] / KCAL:.2f}; a difference above {cfg.estimators_tolerance_kcal_mol:g} "
                   f"kcal/mol indicates windows that are too far apart or too little sampled")
    return out


def run_leg_safe(job: LegJob) -> dict:
    """``run_leg`` that records failures instead of raising (for worker processes)."""
    begin_leg(job.leg_dir, job)
    try:
        return run_leg(job)
    except Exception as exc:  # noqa: BLE001
        err = write_leg_error(job.leg_dir, f"{type(exc).__name__}: {exc}", job, traceback.format_exc(limit=12))
        print(f"[{job.tag}] FAILED: {err['error']}", flush=True)
        return err


# =============================================================================== combination
#: What the legs of a binding free energy must have in common: the key in a leg's result and how
#: a message names it. A complex and a solvent leg, and the repeats of a ligand, are legs of one
#: thermodynamic cycle only under the same temperature and with the same correction for the
#: dispersion tail. Backend, time step, number of lambda windows, lengths, REST2 and the kind of
#: barostat may differ between the legs and are not listed.
CYCLE_CONDITIONS = (("temperature_K", "temperature (K)"), ("pressure_bar", "pressure (bar)"),
                    ("prepared_systems_sha256", "prepared systems (input, force field, solvation)"),
                    ("lrc_method", "correction for the dispersion tail"),
                    ("lrc_long_cutoff_nm", "outer radius of the tail correction (nm)"))
#: conditions that count only for some legs: the outer radius enters the correction by
#: reweighting alone (the isotropic correction and no correction do not depend on it)
CYCLE_CONDITION_APPLIES = {"lrc_long_cutoff_nm": lambda result: result.get("lrc_method") == "reweight"}


def record_digest(record: dict | None) -> str | None:
    """A short digest of a record of settings (None for none), to compare two of them."""
    import hashlib

    if record is None:
        return None
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]


def refresh_leg_result(leg_dir: Path, analysis, log=print) -> str:
    """Make the result of a finished leg again from its samples if it was analysed otherwise than
    this run analyses: by another version of the method or with other settings.

    A leg that this run did not sample keeps the result of an earlier run, and that must not be
    combined with a leg analysed in another way. Returns "current", "reanalysed", "stale" (the
    result could not be made again; it is left as it is and :func:`combine_ligand` says so) or
    "not finished". The result that is replaced is kept as ``result.replaced.json``.
    """
    state, data = leg_state(leg_dir)
    if state != "finished":
        return "not finished"
    if data.get("analysis_record") == analysis_record(analysis):
        return "current"
    where = "/".join(leg_dir.parts[-3:])
    try:
        if not (leg_dir / "hrex" / "samples.h5").exists():
            raise WorkflowError("its samples are not there")
        result = leg_result_from_samples(leg_dir, analysis, run_record_of(data, leg_dir))
    except Exception as exc:  # noqa: BLE001  whatever keeps the samples from being analysed
        log(f"{where}: the result of this leg is of another analysis ({analysis_differences(data, analysis)}) and "
            f"could not be made again from its samples: {type(exc).__name__}: {exc}")
        return "stale"
    (leg_dir / "result.json").replace(leg_dir / "result.replaced.json")
    _write_json(leg_dir / "result.json", result)
    log(f"{where}: the result of this leg was of another analysis ({analysis_differences(data, analysis)}); it has "
        "been made again from the samples, and the one it replaces is kept as result.replaced.json")
    return "reanalysed"


def analysis_differences(result: dict, analysis) -> str:
    """How the analysis behind a leg's result differs from the one of this run, in words ("" if not)."""
    from cufep.analysis.mbar import ANALYSIS_VERSION

    stored, current = result.get("analysis_record"), analysis_record(analysis)
    if stored == current:
        return ""
    if not stored:
        return f"made before the version of the analysis was recorded; this run uses version {ANALYSIS_VERSION}"
    if stored.get("version") != current["version"]:
        return f"version {stored.get('version')} of the analysis; this run uses version {current['version']}"
    return "; ".join(settings_differences({k: v for k, v in stored.items() if k != "version"},
                                          {k: v for k, v in current.items() if k != "version"}))


def check_partner_legs(lig_dir: Path, ligand_name: str, selected: set, repeats: int, cfg: Config) -> None:
    """Before a leg is sampled: the finished legs it will be combined with and that this run does
    not sample must have been made under the temperature and pressure this run uses, and, with
    the tail correction by reweighting, with the outer radius this run uses.

    ``selected`` holds the (repeat, leg) of this ligand that the run samples. Raises WorkflowError
    with what differs, before any time is spent on a leg whose result could not be used.
    """
    from cufep.sampling.storage import _same, stored_conditions

    asked = {"temperature_K": float(cfg.thermodynamics.temperature_K), "pressure_bar": float(cfg.thermodynamics.pressure_bar)}
    names = dict(CYCLE_CONDITIONS)
    differing = []
    for r in range(repeats):
        for leg in ("complex", "solvent"):
            if (r, leg) in selected:
                continue
            leg_dir = lig_dir / f"repeat_{r}" / leg
            state, data = leg_state(leg_dir)
            if state != "finished":
                continue
            where = f"the {leg} leg of repeat {r}, which is finished and not part of this run,"
            for key, value in asked.items():
                if data.get(key) is not None and not _same(data[key], value):
                    differing.append(f"{names[key]}: {where} was made at {data[key]}; this run uses {value}")
            if cfg.analysis.lrc == "reweight":
                # what its samples hold, whatever its result says
                tail = (stored_conditions(leg_dir / "hrex")[1] or {}).get("lrc") or {}
                radius = tail.get("long_cutoff_nm")
                if radius is not None and not _same(float(radius), float(cfg.analysis.lrc_cutoff_nm)):
                    differing.append(f"{names['lrc_long_cutoff_nm']}: {where} recorded its tail energies to "
                                     f"{float(radius):g}; this run uses {float(cfg.analysis.lrc_cutoff_nm):g}")
    if differing and selected:
        raise WorkflowError(
            f"{ligand_name}: the legs this run would sample could not be combined with the legs the ligand already "
            f"has ({'; '.join(differing)}). Nothing was run. Use the conditions of the finished legs, or run all "
            "legs of the ligand under the new ones in a new output.dir.")


def error_from_time_blocks(result: dict) -> bool:
    """Whether the standard error of a leg's result is the one from time blocks."""
    a = result.get("analysis") or {}
    if "delta_g_error_from" in a:
        return a["delta_g_error_from"] == "time blocks"
    return isinstance(result.get("delta_g_error"), (int, float)) and bool(a.get("delta_g_error_n_blocks"))  # version 4


def cycle_differences(legs: list[tuple[str, dict]]) -> tuple[list[str], list[str]]:
    """What differs among leg results that are to be combined, and what could not be compared.

    ``legs`` holds (name of the leg in a message, its result). Returns two lists of sentences;
    the second names conditions that a result of an earlier version does not record. A condition
    of :data:`CYCLE_CONDITION_APPLIES` is compared among the legs it counts for.
    """
    from cufep.sampling.storage import _same

    differing, not_compared = [], []
    for key, name in CYCLE_CONDITIONS:
        applies = CYCLE_CONDITION_APPLIES.get(key, lambda _result: True)
        concerned = [(where, result) for where, result in legs if applies(result)]
        known = [(where, result[key]) for where, result in concerned if result.get(key) is not None]
        missing = [where for where, result in concerned if result.get(key) is None]
        if any(not _same(value, known[0][1]) for _where, value in known[1:]):
            differing.append(f"{name}: " + ", ".join(f"{where} {value}" for where, value in known))
        if missing:
            not_compared.append(f"{name}: not recorded for the {', the '.join(missing)}")
    return differing, not_compared


def combine_ligand(lig_dir: Path, ligand_name: str, repeats: int, analysis=None) -> dict:
    """Binding free energy from the finished legs of one ligand.

    The legs of all complete repeats must have been made under the same conditions
    (:data:`CYCLE_CONDITIONS`); if they were not, the ligand gets no binding free energy and the
    result says what differs. ``analysis`` is the ``analysis`` section of the run: a leg whose
    result is of another analysis (:func:`refresh_leg_result` could not make it again) is combined
    as it is and named under ``old_analysis`` with the checks its result does not hold.
    """
    per_repeat = []
    complete: list[tuple[str, dict]] = []      # the legs of the complete repeats
    warnings: list[str] = []
    # legs whose analysis does not meet a level of cufep.analysis.mbar.quality_checks
    flagged: dict[str, list[str]] = {"overlap": [], "samples": [], "halves": [], "estimators": []}
    not_assessed: list[str] = []           # legs too short to compare their halves
    failed_legs: list[str] = []
    for r in range(repeats):
        states = {leg: leg_state(lig_dir / f"repeat_{r}" / leg) for leg in ("complex", "solvent")}
        for leg, (state, data) in states.items():
            if state == "failed":
                where = f"{leg} leg" + (f" of repeat {r}" if repeats > 1 else "")
                failed_legs.append(where)
                kept = " The result this leg had before is kept in result.previous.json and is not used." \
                    if data.get("previous_result") else ""
                warnings.append(f"repeat {r} {leg}: FAILED: {str(data.get('error', ''))[:300]}.{kept}")
        for leg, (state, data) in states.items():
            # a finished leg whose analysis gave no number (windows without any overlap)
            if state == "finished" and not all(isinstance(data.get(key), (int, float)) and np.isfinite(data[key])
                                               for key in ("delta_g_decouple", "delta_g_mbar_error")):
                where = f"{leg} leg" + (f" of repeat {r}" if repeats > 1 else "")
                failed_legs.append(where)
                warnings.append(f"repeat {r} {leg}: FAILED: the analysis gave no free energy for this leg")
                states[leg] = ("failed", data)
        if not all(state == "finished" for state, _ in states.values()):
            warnings.append(f"repeat {r} is incomplete")
            continue
        c, s = states["complex"][1], states["solvent"][1]
        complete += [(f"{leg} leg" + (f" of repeat {r}" if repeats > 1 else ""), states[leg][1])
                     for leg in ("complex", "solvent")]
        dg = s["delta_g_decouple"] - c["delta_g_decouple"] - c["restraint_release"]
        # the standard error of a leg (from time blocks since version 4 of the analysis; a result of
        # an earlier version holds MBAR's alone) and the one MBAR gives
        reported = [leg["delta_g_error"] if isinstance(leg.get("delta_g_error"), (int, float)) else leg["delta_g_mbar_error"]
                    for leg in (c, s)]
        per_repeat.append({
            "repeat": r, "dg_bind": dg, "error": float(np.hypot(*reported)),
            "mbar_error": float(np.hypot(c["delta_g_mbar_error"], s["delta_g_mbar_error"])),
            "error_from_time_blocks": all(error_from_time_blocks(leg) for leg in (c, s)),
            "dg_complex": c["delta_g_decouple"],
            "dg_solvent": s["delta_g_decouple"], "restraint_release": c["restraint_release"],
            "lrc_complex": c["lrc"], "lrc_solvent": s["lrc"], "backend_complex": c["backend"],
            "backend_solvent": s["backend"], "stage_complex": c["analysis"]["stage_delta_g"],
            "stage_solvent": s["analysis"]["stage_delta_g"],
            "lambda_windows": {"complex": c["n_states"], "solvent": s["n_states"]},
            "lambda_windows_added": {"complex": c.get("lambda_windows_added"), "solvent": s.get("lambda_windows_added")},
            "min_overlap": {"complex": c["analysis"]["min_overlap"], "solvent": s["analysis"]["min_overlap"]},
            # among the samples the free energy is estimated from (None in results that lack it)
            "min_overlap_used": {"complex": c["analysis"].get("min_overlap_used"),
                                 "solvent": s["analysis"].get("min_overlap_used")},
            "production_ns": {"complex": c["production_ns"], "solvent": s["production_ns"]}})
        for leg_result in (c, s):
            warnings += [f"repeat {r} {leg_result['leg']}: {n}" for n in leg_result.get("notes", [])]
            where = f"{leg_result['leg']} leg" + (f" of repeat {r}" if repeats > 1 else "")
            quality = leg_result.get("quality") or {}
            if "halves" not in quality:            # written before these checks: only the overlap is known
                quality = {"overlap": quality.get("overlap")}
            for check, legs in flagged.items():
                if (quality.get(check) or {}).get("ok") is False:
                    legs.append(where)
            if (quality.get("halves") or {}).get("reason"):
                not_assessed.append(where)
    inconsistent, not_compared = cycle_differences(complete)
    if inconsistent:
        # these legs are not legs of one thermodynamic cycle: no number is made from them
        warnings.insert(0, "NO BINDING FREE ENERGY: the legs were not made under the same conditions ("
                        + "; ".join(inconsistent) + "). Run the legs that differ again under the conditions of "
                        "the others, in a new output.dir if their samples are to be kept")
        per_repeat = []
    warnings += [f"could not be compared between the legs: {item}" for item in not_compared]
    # legs that used files of an earlier version whose origin is not recorded
    unrecorded = [f"{where}: {', '.join(result['origin_not_recorded'])}" for where, result in complete
                  if result.get("origin_not_recorded")]
    # legs whose result is of another analysis than this run's and could not be made again
    old_analysis = []
    if analysis is not None:
        checks = {"samples": "independent samples", "halves": "halves of the production", "estimators": "BAR against MBAR"}
        for where, result in complete:
            differs = analysis_differences(result, analysis)
            if differs:
                quality = result.get("quality") or {}
                not_made = [name for key, name in checks.items()
                            if "halves" not in quality or (quality.get(key) or {}).get("ok") is None]
                old_analysis.append(f"{where}: {differs}" + (f"; not checked: {', '.join(not_made)}" if not_made else ""))
        warnings += [f"the result of the {item}. Its free energy and uncertainty are combined as they are" for item in old_analysis]
    out: dict = {"ligand": ligand_name, "units": "kcal/mol", "n_repeats": len(per_repeat), "warnings": warnings,
                 "failed_legs": failed_legs, "inconsistent_legs": inconsistent, "not_compared": not_compared,
                 "origin_not_recorded": unrecorded, "old_analysis": old_analysis,
                 "not_reliable": flagged["overlap"], "few_samples": flagged["samples"],
                 "halves_differ": flagged["halves"], "halves_not_compared": not_assessed,
                 "estimators_disagree": flagged["estimators"]}
    if per_repeat:
        vals = np.array([p["dg_bind"] for p in per_repeat]) / KCAL
        out["dg_bind"] = float(vals.mean())
        out["dg_bind_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        # the standard error of one calculation (root mean square over the repeats, if there are several),
        # and whether it is from time blocks in every leg
        out["dg_bind_error"] = float(np.sqrt(np.mean([(p["error"] / KCAL) ** 2 for p in per_repeat])))
        out["dg_bind_error_from_time_blocks"] = all(p["error_from_time_blocks"] for p in per_repeat)
        out["dg_bind_mbar_error"] = float(np.sqrt(np.mean([(p["mbar_error"] / KCAL) ** 2 for p in per_repeat])))
        out["min_overlap"] = float(min(min(p["min_overlap"].values()) for p in per_repeat))
        used = [v for p in per_repeat for v in p["min_overlap_used"].values() if v is not None]
        out["min_overlap_used"] = float(min(used)) if used else None
        energy_keys = ("dg_bind", "error", "mbar_error", "dg_complex", "dg_solvent", "restraint_release", "lrc_complex",
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
    (lig_dir / "result.json").write_text(jsonio.dumps(out, indent=1))
    return out
