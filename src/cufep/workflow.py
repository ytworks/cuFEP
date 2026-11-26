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
from typing import ClassVar

import numpy as np

from cufep import jsonio
from cufep.config.schema import CommonConfig, Config, WindowsConfig
from cufep.engine.jobs import Job
from cufep.files import publish, publish_npz, publish_text, publish_with

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


#: what cuFEP itself writes directly under ``output.dir``; no ligand directory may take one of
#: these names (a source scan in the tests keeps the list complete)
OUTPUT_ROOT_NAMES = frozenset({
    "cache", "cufep.log", "resolved_config.yaml", "resolved_config.refused.yaml", "auto_report.md",
    "protein_prepared.pdb", "summary.json", "summary.csv", "timing.json", "timing_rerun.json",
    "benchmark_report.md", "benchmark_report.csv", "benchmark_report.png", "benchmark_report_ligands.png",
    "frozen_schedules", ".cufep.lock",
    # the output of a relative run (cufep.rbfe.run) and its benchmark report (cufep.benchmarks.rbfe)
    "edges", "network.json", "edges.csv", "ligands.csv", "rbfe_report.md", "rbfe_report_ligands.csv",
    "rbfe_report_edges.csv", "rbfe_report.png", "rbfe_report_edges.png"})


def ligand_dir_name(name: str) -> str:
    """The name of the directory of a ligand under ``output.dir``: its SDF title with every
    character other than letters, digits, ``-``, ``_`` and ``.`` replaced by ``_``. A name that
    would not be a directory of its own there (``.``, ``..``, a hidden name, a name cuFEP uses at
    the root of the output) is refused: ``.`` wrote the ligand's files into the output root and
    ``..`` into its parent (review of PR 1)."""
    safe = safe_name(name)
    if safe in ("", ".", "..") or safe.startswith(".") or safe in OUTPUT_ROOT_NAMES:
        raise WorkflowError(f"the ligand {name!r} cannot have a directory of its own under output.dir (its name "
                            f"would be {safe!r}); give its SDF record another title line")
    return safe


def ligand_directories(out_dir: Path | str, names) -> dict[str, Path]:
    """The directory of every ligand under ``out_dir`` (:func:`ligand_dir_name`). Two ligands
    whose names give the same directory are refused, before anything is written."""
    seen: dict[str, str] = {}
    out: dict[str, Path] = {}
    for name in names:
        d = ligand_dir_name(name)
        if d in seen:
            raise WorkflowError(f"the ligands {seen[d]!r} and {name!r} would share the directory {d}/ under "
                                "output.dir; give every SDF record a title line of its own")
        seen[d] = name
        out[name] = Path(out_dir) / d
    return out


def select_names(available, wanted, source: str) -> list[str]:
    """The ``wanted`` names in the order of ``available``; all of ``available`` if none is wanted.
    A wanted name that is not available is an error that names ``source`` (the setting or the
    option it came from). One rule for every command that picks ligands."""
    available = list(available)
    wanted = list(wanted or [])
    if not wanted:
        return available
    missing = [n for n in wanted if n not in available]
    if missing:
        raise WorkflowError(f"{source}: not among the ligands of this calculation: {', '.join(missing)} "
                            f"(there are {', '.join(available[:12])}{' ...' if len(available) > 12 else ''})")
    return [n for n in available if n in set(wanted)]


def select_ligands(ligands, wanted, source: str) -> list:
    """The ligands whose names are ``wanted`` (:func:`select_names`), in the order of the SDF."""
    keep = set(select_names([lig.name for lig in ligands], wanted, source))
    return [lig for lig in ligands if lig.name in keep]


def restraint_anchor_problems(host_atoms, guest_atoms, prep: dict) -> list[str]:
    """Why Boresch anchors cannot restrain the ligand to the protein in the prepared complex
    ``prep`` (:func:`load_prepared`): the host atoms must be atoms of the protein and the guest
    atoms atoms of the ligand. With six protein atoms the restraint does not move when the
    ligand does, and its release would still be added to the binding free energy (review of
    PR 1)."""
    out = []
    for key, atoms, members, what in (("host_atoms", host_atoms, prep["protein"], "the protein"),
                                      ("guest_atoms", guest_atoms, prep["ligand"], "the ligand")):
        atoms = [int(i) for i in (atoms or [])]
        if len(atoms) != 3 or len(set(atoms)) != 3:
            out.append(f"restraints.{key} must be three distinct atoms, not {atoms}")
        outside = [i for i in atoms if i not in set(int(m) for m in members)]
        if outside:
            out.append(f"restraints.{key}: {outside} are not atoms of {what} in the prepared complex (atoms are "
                       "numbered from 0 as in prep/complex.pdb)")
    return out


def solute_correspondence(prep_from: dict, prep_to: dict) -> dict[int, int]:
    """For two prepared systems of one complex (``topology``, ``solute``, ``protein`` and
    ``ligand`` as :func:`load_prepared` gives them): the atom of ``prep_to`` for every atom of the
    solute of ``prep_from``, protein and ligand.

    The two are built from the same pose, and a builder only removes or appends molecules, so the
    atoms of the solute come in the same order in both; their numbers need not be the same (only
    the build of a bilayer can remove a crystal water or an ion, which come before the ligand).
    Raises WorkflowError where the solutes are not the same atoms with the same bonds: name,
    element, residue (name, number, insertion code), chain and whether protein or ligand, atom by
    atom (the chain: two chains of one sequence and numbering differ in nothing else, and the
    builders of OpenMM keep the names of the chains they are given). Exact; nothing is compared
    within a tolerance."""
    def of(prep):
        atoms = list(prep["topology"].atoms())
        solute = [int(i) for i in prep["solute"]]
        protein, ligand = set(int(i) for i in prep["protein"]), set(int(i) for i in prep["ligand"])
        rank = {i: r for r, i in enumerate(solute)}
        described = [(atoms[i].name, None if atoms[i].element is None else atoms[i].element.symbol,
                      atoms[i].residue.name, str(atoms[i].residue.id),
                      (getattr(atoms[i].residue, "insertionCode", "") or "").strip(), i in protein, i in ligand,
                      str(atoms[i].residue.chain.id))
                     for i in solute]
        bonds = {tuple(sorted((rank[a.index], rank[b.index]))) for a, b in prep["topology"].bonds()
                 if a.index in rank and b.index in rank}
        return solute, described, bonds

    (solute_a, atoms_a, bonds_a), (solute_b, atoms_b, bonds_b) = of(prep_from), of(prep_to)
    problem = None
    if len(solute_a) != len(solute_b):
        problem = f"{len(solute_a)} atoms of protein and ligand in one and {len(solute_b)} in the other"
    else:
        other = next((r for r, (a, b) in enumerate(zip(atoms_a, atoms_b)) if a != b), None)
        if other is not None:
            name = lambda d: f"{d[2]}{d[3]}{d[4]}:{d[0]} of chain {d[7]}"                   # noqa: E731
            problem = (f"atom {other} of protein and ligand is {name(atoms_a[other])} in one and "
                       f"{name(atoms_b[other])} in the other")
        elif bonds_a != bonds_b:
            first = sorted(bonds_a ^ bonds_b)[0]
            problem = f"the bond between atoms {first[0]} and {first[1]} of protein and ligand is in one of them only"
    if problem:
        raise WorkflowError("the two prepared systems of the complex are not of the same protein and ligand: "
                            f"{problem}. A restraint chosen in one cannot be used in the other; nothing was written")
    return dict(zip(solute_a, solute_b))


def restraint_in_system(restraint, atom_map: dict[int, int]):
    """A Boresch restraint with its six anchors as another prepared system of the same solute
    numbers them (``atom_map`` of :func:`solute_correspondence`). The reference values and force
    constants are those of the atoms and do not change."""
    from dataclasses import replace

    return replace(restraint, host_atoms=tuple(atom_map[int(i)] for i in restraint.host_atoms),
                   guest_atoms=tuple(atom_map[int(i)] for i in restraint.guest_atoms))


def check_before_jobs(cfg: Config, lig_dir: Path, legs) -> None:
    """Refuse, before any job starts and before an earlier result of a leg is set aside, what the job
    of a complex leg would refuse from the prepared systems and the files of its leg: manual restraint
    anchors that are not atoms of the protein and the ligand (:func:`check_manual_anchors`), REST2
    residues that a preparation of an earlier version cannot name (:func:`rest2_region`), and a
    manual restraint that would be measured from a pre-equilibration without the boxes of its frames
    (:func:`manual_boresch_restraint`). ``legs`` are the (repeat, leg) of this run. Review of PR 1:
    these were refused only inside each job, after its result had been set aside and its
    pre-equilibration run."""
    complex_repeats = sorted(int(r) for r, leg in legs if leg == "complex")
    if not complex_repeats:
        return
    check_manual_anchors(cfg, lig_dir)
    stored = all((Path(lig_dir) / f"repeat_{r}" / "complex" / "restraint.json").exists() for r in complex_repeats)
    if cfg.restraints.selection == "manual" and not stored:
        # anchors whose theta_A or theta_B is within 10 kT of a straight line already in the prepared pose,
        # from which the pre-equilibration starts, are refused before it (OpenFE's check_angle_not_flat, a
        # rule of every frame). The rule of a trajectory (on a line in every frame) and the reference frame
        # are checked when the restraint is made from the frames of the pre-equilibration (review of
        # 7b49066). A run whose legs all have their restraint is not checked here (review of PR 1)
        from cufep.alchemy.restraints import (
            K_ANGLE_DEFAULT,
            BoreschRestraint,
            anchor_geometry_problem,
            as_the_force_measures,
            flat_margin_of,
        )

        prep = load_prepared(Path(lig_dir) / "prep", "complex")
        rc = cfg.restraints
        host, guest = tuple(int(i) for i in rc.host_atoms), tuple(int(i) for i in rc.guest_atoms)
        # the six anchors as the force of the restraint sees them
        pose = as_the_force_measures(np.asarray(prep["positions"], dtype=float)[[*reversed(host), *guest]][None],
                                     np.asarray(prep["box"], dtype=float))
        # with force constants from the fluctuations the constant is not known yet: a pose that breaks the rule
        # with the strongest one it can get breaks it with every one; the leg asks again with the one it gets
        k_angle = 4.0 * K_ANGLE_DEFAULT if rc.force_constants == "auto" else rc.k_angle_kcal_per_rad2 * KCAL
        problem = anchor_geometry_problem(BoreschRestraint((2, 1, 0), (3, 4, 5), 0, 0, 0, 0, 0, 0), pose,
                                          flat_margin_of(k_angle, cfg.thermodynamics.temperature_K),
                                          trajectory=False)
        if problem:
            raise WorkflowError(f"restraints.host_atoms {list(host)} / guest_atoms {list(guest)} in the prepared pose: "
                                f"{problem}. Choose other anchors; nothing was changed")
    rest2 = cfg.sampling.rest2
    if rest2.scale > 1.0 and rest2.residues:
        prep = load_prepared(Path(lig_dir) / "prep", "complex")
        rest2_region(
            rest2, prep["topology"], prep["positions"], prep["ligand"], prep["protein"], prep.get("input_chains")
        )
    if cfg.restraints.selection == "manual":
        for r in complex_repeats:
            leg_dir = Path(lig_dir) / f"repeat_{r}" / "complex"
            equil = leg_dir / "equil.npz"
            if equil.exists() and not (leg_dir / "restraint.json").exists():
                with np.load(equil) as eq:
                    if "frame_boxes" not in eq:
                        raise WorkflowError(f"{equil} holds no boxes of its frames (an earlier version wrote it): the "
                                            "reference values of a manual restraint cannot be measured across the "
                                            "periodic boundary. Nothing was changed; run the calculation in a new "
                                            "output.dir")


def ligands_of_run(cfg: Config, only_ligands=()) -> list:
    """The ligands of a calculation, read from the SDF and checked before anything is written: those
    of ``system.ligand_names`` (all if none), with distinct titles, each with a directory of its own
    under ``output.dir`` (:func:`ligand_directories`), and the names of ``--ligand`` among them."""
    from cufep.prep.ligand import load_ligands

    ligands = select_ligands(load_ligands(cfg.system.ligand), cfg.system.ligand_names, "system.ligand_names")
    names = [lig.name for lig in ligands]
    if len(set(names)) != len(names):
        raise WorkflowError("ligand names in the SDF are not unique; give every record a distinct title line")
    ligand_directories(cfg.output.dir, names)
    select_names(names, only_ligands, "--ligand")
    return ligands


def check_manual_anchors(cfg: Config, lig_dir: Path) -> None:
    """Refuse a manual restraint whose anchors are not atoms of the protein and the ligand of
    this ligand's prepared complex. Called before any MD: after the preparation, by the run
    before it starts its jobs and by the job of the complex leg."""
    rc = cfg.restraints
    if rc.selection != "manual":
        return
    problems = restraint_anchor_problems(
        rc.host_atoms, rc.guest_atoms, load_prepared(Path(lig_dir) / "prep", "complex")
    )
    if problems:
        raise WorkflowError("; ".join(problems))


# =============================================================================== preparation
#: crystal waters that overlap the ligand are removed (a rule of building the system, not a verdict)
WATER_CLASH_NM = 0.22


def check_ligand(ligand) -> None:
    """Stop on a ligand that cuFEP cannot treat (a net charge).

    Earlier versions also stopped on a ligand with less than half of its heavy atoms near the protein,
    one closer than 0.2 nm to a protein atom or 0.3 nm to a metal ion, and warned of a metal ion within
    0.6 nm: levels of cuFEP's own, removed by the owner's order of 2026-11-20 (no criterion of its own).
    """
    from cufep.prep.ligand import LigandError

    if ligand.formal_charge != 0:
        raise LigandError(
            f"ligand '{ligand.name}' has formal charge {ligand.formal_charge:+d}. Charged ligands need a "
            "finite-size/co-alchemical ion treatment that cuFEP 0.1 does not have. Provide a neutral "
            "protonation state, or wait for charged-ligand support.")


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


def _solute_of(topology, ligand_atoms) -> dict:
    """The atoms of the protein, of the ligand and of both (the solute) in a topology, as a
    prepared system records them (:func:`_write_system`, :func:`load_prepared`)."""
    from cufep.prep.protein import AMINO_ACIDS, CAPS

    protein = [a.index for a in topology.atoms() if a.residue.name in AMINO_ACIDS or a.residue.name in CAPS]
    ligand = [int(i) for i in ligand_atoms]
    return {"topology": topology, "protein": protein, "ligand": ligand, "solute": sorted(set(protein) | set(ligand))}


def _write_system(prep_dir: Path, name: str, modeller, system, ligand_atoms: list[int]) -> None:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from cufep.prep.ligand import LIGAND_RESNAME
    from cufep.prep.topology_io import topology_to_dict

    top = modeller.topology
    pos = np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=float)
    box = np.array([[float(v) for v in vec.value_in_unit(unit.nanometer)] for vec in top.getPeriodicBoxVectors()])
    members = _solute_of(top, ligand_atoms)
    protein, solute = members["protein"], members["solute"]
    publish_text(prep_dir / f"{name}_system.xml", mm.XmlSerializer.serialize(system))
    publish_text(prep_dir / f"{name}_topology.json", json.dumps(topology_to_dict(top)))
    publish_npz(prep_dir / f"{name}.npz", positions=pos, box=box, ligand=np.array(ligand_atoms, dtype=int),
                solute=np.array(solute, dtype=int), protein=np.array(protein, dtype=int))
    publish_with(prep_dir / f"{name}.pdb", lambda fh: app.PDBFile.writeFile(top, modeller.positions, fh, keepIds=True))
    _ = LIGAND_RESNAME


def load_prepared(prep_dir: Path, name: str) -> dict:
    from cufep.prep.topology_io import topology_from_dict

    data = np.load(prep_dir / f"{name}.npz")
    info_file = Path(prep_dir) / "prep.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    return {"input_chains": info.get("input_chains"),       # None for a preparation of an earlier version
            "system_xml": (prep_dir / f"{name}_system.xml").read_text(),
            "topology": topology_from_dict(json.loads((prep_dir / f"{name}_topology.json").read_text())),
            "positions": data["positions"], "box": data["box"], "ligand": [int(i) for i in data["ligand"]],
            "solute": [int(i) for i in data["solute"]], "protein": [int(i) for i in data["protein"]]}


def settings_differences(stored, current, prefix: str = "") -> list[str]:
    """Where two nested records of settings differ, as ``"a.b: old -> new"``."""
    if isinstance(stored, dict) and isinstance(current, dict):
        out: list[str] = []
        for key in sorted(set(stored) | set(current)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored or key not in current:
                out.append(f"{name}: {stored.get(key, 'not recorded')} -> {current.get(key, 'not given')}")
            else:
                out += settings_differences(stored[key], current[key], name)
        return out
    return [] if _the_same_setting(stored, current) else [f"{prefix}: {stored} -> {current}"]


def _the_same_setting(a, b) -> bool:
    """Whether two values of a record of settings are the same: numbers exactly. A setting is read
    from the YAML and written to the record as it is, so the same setting is the same number; and
    two settings that differ in the tenth digit can give another restraint (a threshold of 319 or
    320 frames, a rank that turns over at ``stable_structure_fraction``, a frame every 2500 or
    2501 steps; reviews of eb6f89f and 877ab88: with a tolerance of 1e-9 such records were one).
    The conditions of stored samples, which are computed, keep their tolerance
    (:func:`cufep.sampling.storage._same`)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_the_same_setting(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_the_same_setting(a[k], b[k]) for k in a)
    return a == b


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


def prep_contradictions(info: dict, ligand_name: str, membrane: bool, box_shape: str,
                        complex_leg: str = "bilayer") -> list[str]:
    """Where a ``prep.json`` without a record of its origin says something else than this run.
    ``complex_leg`` is ``membrane.complex_leg``: with a bilayer, what the complex of the alchemical
    run is in (a file without a record of its origin has the bilayer complex alone)."""
    in_leg = bool(membrane) and complex_leg == "bilayer"
    expected = {"ligand": ligand_name, "membrane": bool(membrane),
                "complex_barostat": "membrane" if in_leg else "isotropic", "box_shape": box_shape}
    return [f"{key}: {info[key]} -> {value}" for key, value in expected.items() if key in info and info[key] != value]


def pre_equilibration_parts(cfg: CommonConfig, leg: str, in_membrane: bool) -> tuple[float, bool]:
    """``(the MD in ns that follows NVT and NPT in the pre-equilibration of a leg, whether the solute
    is held by position restraints through all of it)``.

    The MD is ``sampling.pre_equilibration.<leg>.md_ns``, and for a bilayer at least
    ``membrane.equilibration_ns``; nothing is held in it. A leg whose configuration holds its solute
    (:meth:`cufep.config.schema.CommonConfig.holds_solute`) is different: the heavy atoms of protein
    and ligand are held through NVT and NPT while the solvent and the box settle, and no free MD
    follows. In the ABFE that is a complex leg whose restraint is chosen from interactions
    (``restraints.selection: interactions``): its trajectories start from the pose that was given, as
    in the method of Ding et al., which starts them from the minimised input, and are that MD; in a
    relative calculation the complex leg, as OpenFE starts its relative legs from the minimised input
    pose. A bilayer that was just built still needs ``membrane.equilibration_ns`` to close around the
    protein, and the solute is held through it (the owner's decision of 2026-11-21, after two runs
    of the A2A example chose their restraints from two different states the ligand had reached in
    10 ns of free MD)."""
    pre = cfg.sampling.pre_equilibration.complex if leg == "complex" else cfg.sampling.pre_equilibration.solvent
    if cfg.holds_solute(leg):
        return (float(cfg.membrane.equilibration_ns) if in_membrane else 0.0), True
    return float(max(pre.md_ns, cfg.membrane.equilibration_ns) if in_membrane else pre.md_ns), False


def equil_lengths(eq, cfg: Config, leg: str, in_membrane: bool) -> tuple[float, float, float]:
    """(asked, integrated, tolerance) in ns of the unrestrained MD of a stored pre-equilibration:
    the length this run asks for, the MD the file holds, and half an MD step. A file of a version
    before revision 15 records the length that was asked for, of which only whole frame intervals
    were integrated (review of PR 1: 40 ps of 50 ps)."""
    from cufep.sampling.equilibrate import EquilibrationSettings
    from cufep.steps import integrated_before_revision_15_ps

    asked = pre_equilibration_parts(cfg, leg, in_membrane)[0]
    files = set(eq.files)
    done = float(eq["simulated_ns"])
    dt = float(eq["timestep_fs"]) if "timestep_fs" in files else float(cfg.dynamics.timestep_fs)
    if "simulated_ns_is_integrated" not in files:
        done = integrated_before_revision_15_ps(done * 1000.0, EquilibrationSettings.frame_interval_ps, dt) / 1000.0
    return float(asked), done, 0.5 * dt * 1.0e-6


def check_equil_file(eq, equil_file: Path, cfg: Config, leg: str, in_membrane: bool, seed: int, log=print) -> bool:
    """Refuse a stored pre-equilibration this run cannot use (:func:`check_made_from` with its record
    or, for a file without one, :func:`equil_contradictions`), and one that holds less MD than it was
    made for, whatever its record says (:func:`equil_lengths`). Returns whether its origin is recorded."""
    stored = json.loads(str(eq["made_with"])) if "made_with" in eq.files else None
    current = equil_made_with(cfg, leg, in_membrane, seed)
    if stored is not None and "platform" not in stored:
        current.pop("platform")                  # a file of a version that did not record its platform
        log(f"the pre-equilibration in {equil_file} was written by a version that did not record its platform: "
            f"whether it ran on {cfg.backend.platform} is not known, and it is used as it is")
    if stored is not None and stored.get("solute_held") and current.get("solute_held"):
        # the first two commits that held the solute (720cdb1, 48213b7) recorded md_ns, the free MD such a
        # pre-equilibration does not have (equil_made_with): their files are of this pre-equilibration (the
        # fifth pass of Codex: the two example runs of 720cdb1 were refused for "md_ns: 5.0 -> not given")
        stored = {**stored, "pre_equilibration": {key: value for key, value in stored["pre_equilibration"].items()
                                                  if key != "md_ns"}}
    recorded = check_made_from("the pre-equilibration of this leg", equil_file, stored, current, log,
                               contradictions=equil_contradictions(eq, cfg, leg, in_membrane))
    if "simulated_ns" in eq.files:
        asked, done, tolerance = equil_lengths(eq, cfg, leg, in_membrane)
        if done < asked - tolerance:
            raise WorkflowError(
                f"the pre-equilibration of this leg in {equil_file} cannot be used again: it holds {done * 1000:g} ps "
                f"of unrestrained MD where {asked * 1000:g} ps were asked for (versions before revision 15 integrated "
                "whole frame intervals of 20 ps only). Nothing was changed; run the calculation in a new output.dir")
    return recorded


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
    out = []
    files = set(eq.files)
    if "timestep_fs" in files and float(eq["timestep_fs"]) > float(cfg.dynamics.timestep_fs) + 1e-9:
        out.append(f"time step: {float(eq['timestep_fs']):g} fs -> {float(cfg.dynamics.timestep_fs):g} fs")
    if "simulated_ns" in files:
        asked, done, tolerance = equil_lengths(eq, cfg, leg, in_membrane)
        if done < asked - tolerance or (not in_membrane and done > asked + tolerance):
            out.append(f"length of the MD: {done:g} ns -> {asked:g} ns")
    has_bilayer = equil_in_membrane(eq)
    if has_bilayer is not None and has_bilayer != bool(in_membrane):
        out.append(f"bilayer: {'yes' if has_bilayer else 'no'} -> {'yes' if in_membrane else 'no'}")
    if pre_equilibration_parts(cfg, leg, in_membrane)[1]:
        # a file without a record is of a version before the solute was ever held: its MD was free, also the
        # 10 ns of a bilayer, whose length is the one that is asked for now (the fourth pass of Codex)
        out.append("the solute during the pre-equilibration: free -> held (restraints.selection: interactions)")
    return out


def manual_constants_of_an_earlier_version(stored: dict, cfg: Config) -> bool:
    """Whether a stored restraint is one of manual anchors made by a version that ignored
    ``restraints.force_constants: auto`` for them, while this run asks for auto: it holds the
    configured constants, not constants from the fluctuations. Such a file records no
    ``force_constants`` in its info (versions before revision 14); its record of origin, if it has
    one, says auto all the same."""
    rc, info = cfg.restraints, stored.get("info") or {}
    return (rc.selection == "manual" and rc.force_constants == "auto" and info.get("selection") == "manual"
            and "force_constants" not in info)


#: what the info of a restraint says of how its anchors were held to OpenFE's check_angle_not_flat: with the
#: force constants the restraint has, of the frames, the start and its own reference angles. A restraint with
#: constants from fluctuations that does not say so was held to the rule with the configured constants
FLAT_ANGLE_RULE = "with the force constants of the restraint"


def check_restraint_file(stored: dict, rfile: Path, cfg: Config, log=print) -> bool:
    """Refuse a stored restraint this run cannot use (:func:`check_made_from` with its record or, for a
    file without one, :func:`restraint_contradictions`), and a manual restraint of a version that
    ignored ``force_constants: auto`` when auto is asked, whatever its record says
    (:func:`manual_constants_of_an_earlier_version`; review of PR 1: such files had a record and were
    used, with the configured constants under a run that records auto). One rule for the check
    before any job and for the leg. Returns whether its origin is recorded."""
    recorded = check_made_from(
        "the restraint of this leg",
        rfile,
        stored.get("made_with"),
        restraint_made_with(cfg),
        log,
        contradictions=restraint_contradictions(stored, cfg),
    )
    info = stored.get("info") or {}
    if info.get("force_constants") == "auto" and info.get("flat_angle_rule") != FLAT_ANGLE_RULE:
        raise WorkflowError(
            f"the restraint of this leg in {rfile} cannot be used again: its force constants come from the "
            "fluctuations of the anchors (restraints.force_constants: auto) and it was made by a version that held "
            "the anchors to OpenFE's rule of the flat angle with the configured constants, not with those. Nothing "
            "was changed; run the calculation in a new output.dir")
    if recorded and manual_constants_of_an_earlier_version(stored, cfg):
        raise WorkflowError(
            f"the restraint of this leg in {rfile} cannot be used again: "
            "it is a manual restraint of a version that used "
            "the configured force constants where restraints.force_constants: auto asks for constants from the "
            "fluctuations of the anchors. Nothing was changed; run the calculation in a new output.dir"
        )
    return recorded


def restraint_contradictions(stored: dict, cfg: Config) -> list[str]:
    """Where a ``restraint.json`` without a record of its origin says something else than this run."""
    rc, info, restraint = cfg.restraints, stored.get("info") or {}, stored.get("restraint") or {}
    out = []
    if "selection" in info and info["selection"] != rc.selection:
        out.append(f"restraints.selection: {info['selection']} -> {rc.selection}")
    if "force_constants" in info and info["force_constants"] != rc.force_constants:
        out.append(f"restraints.force_constants: {info['force_constants']} -> {rc.force_constants}")
    elif manual_constants_of_an_earlier_version(stored, cfg):
        # a manual restraint of a version that ignored force_constants: auto holds the configured constants
        out.append("restraints.force_constants: the configured constants (a manual restraint of an earlier version) "
                   "-> auto")
    if rc.selection == "manual":
        for key, asked in (("host_atoms", rc.host_atoms), ("guest_atoms", rc.guest_atoms)):
            if asked and key in restraint and list(restraint[key]) != list(asked):
                out.append(f"restraints.{key}: {list(restraint[key])} -> {list(asked)}")
    # the six force constants, where the run sets them (not taken from the fluctuations, for chosen
    # and for manual anchors alike): the file holds them in kJ/mol/nm^2 and kJ/mol/rad^2
    if rc.force_constants != "auto":
        from cufep.sampling.storage import _same
        if "k_r" in restraint and not _same(float(restraint["k_r"]), rc.k_r_kcal_per_A2 * KCAL * 100.0):
            out.append(
                f"restraints.k_r_kcal_per_A2: {float(restraint['k_r']) / (KCAL * 100.0):g} -> {rc.k_r_kcal_per_A2:g}"
            )
        angles = {
            key: float(restraint[key]) / KCAL
            for key in ("k_theta_a", "k_theta_b", "k_phi_a", "k_phi_b", "k_phi_c")
            if key in restraint
        }
        other = {key: value for key, value in angles.items() if not _same(value, float(rc.k_angle_kcal_per_rad2))}
        if other:
            out.append(
                "restraints.k_angle_kcal_per_rad2: "
                + ", ".join(f"{key} {value:g}" for key, value in other.items())
                + f" -> {rc.k_angle_kcal_per_rad2:g}"
            )
    return out


def iterations_for(ns: float, iteration_ps: float, least: int) -> int:
    """The number of iterations a run makes for ``ns`` per lambda window with iterations of
    ``iteration_ps`` (the length the configuration asks for): the one rule of :func:`run_leg`."""
    return max(int(least), int(round(float(ns) * 1000.0 / float(iteration_ps))))


def planned_iterations(iteration_ps: float, timestep_fs: float, equilibration_ns: float, production_ns: float):
    """What a run of a leg plans: MD steps per iteration (whole steps of ``timestep_fs``), the
    length of an iteration that gives (2.5 ps at 3 fs are 833 steps, 2.499 ps), and the numbers of
    equilibration and production iterations. Used by the run of a leg and by every check of
    stored samples against a run, so that the two cannot round differently."""
    from types import SimpleNamespace

    from cufep.steps import md_steps

    steps = md_steps(iteration_ps, timestep_fs)
    return SimpleNamespace(steps_per_iteration=steps, iteration_ps=steps * float(timestep_fs) * 1e-3,
                           n_equilibration=iterations_for(equilibration_ns, iteration_ps, 1),
                           n_production=iterations_for(production_ns, iteration_ps, 4))


def leg_seed(cfg: Config, repeat: int, leg: str) -> int:
    """The seed of a leg: every repeat and each of its two legs has its own."""
    return int(cfg.sampling.seed) + 7919 * int(repeat) + (0 if leg == "complex" else 104729)


#: settings of ``alchemy`` that were added after samples were first recorded, with the value that
#: gives the System those samples were made with. A setting at that value is not written to the
#: conditions of the samples, so that the samples of every earlier run are still continued and
#: analysed (recorded whole, the alchemy section would differ from theirs by a key, and every
#: stored leg would be refused); at another value it is written, and differs from samples that
#: do not have it.
ALCHEMY_SETTINGS_ADDED = {"softcore_b": 1, "annihilate_electrostatics": False}

#: settings of ``analysis`` that were added after results were first recorded, with the value that
#: gives the analysis those results were made with: at that value a setting is not written to the
#: record of an analysis (:func:`analysis_record`)
ANALYSIS_SETTINGS_ADDED = {"detect_equilibration": False}

#: settings of ``membrane`` that were added after preparations were first recorded, with the value
#: that gives the prepared systems those were: at that value a setting is not written to the record
#: of a preparation (:func:`prep_made_from`)
MEMBRANE_SETTINGS_ADDED = {"complex_leg": "bilayer", "water": None}

#: the settings added later, by section of the configuration: a record of a configuration that an
#: earlier version wrote is read with these values where it lacks them (``write_run_record``)
SETTINGS_ADDED = {"alchemy": ALCHEMY_SETTINGS_ADDED, "analysis": ANALYSIS_SETTINGS_ADDED,
                  "membrane": MEMBRANE_SETTINGS_ADDED,
                  # no water relocation moves and no grand canonical moves of water: the sections as
                  # they are by default
                  "sampling": {"relocation_mc": {"enabled": False, "interval_ps": 4.0, "trial_poses": 16384,
                                                 "inner_fraction": "uniform", "radius_margin_nm": 0.3},
                               "gcmc": {"enabled": False, "interval_ps": 4.0, "attempts": 2500, "batch_size": 1000,
                                        "radius_margin_nm": 0.3, "ghost_waters": "auto",
                                        "excess_chemical_potential_kcal_per_mol": "auto",
                                        "standard_volume_nm3": "auto"}}}


def gcmc_in(cfg: Config, leg: str | None) -> bool:
    """Whether the leg makes grand canonical moves of water (``sampling.gcmc``): the complex leg, if
    they are enabled. Such a leg has ghost waters from its pre-equilibration on and no barostat in
    its alchemical run."""
    return leg == "complex" and bool(cfg.sampling.gcmc.enabled)


def gcmc_water_values(cfg: Config) -> tuple[float, float]:
    """(excess chemical potential of water in kcal/mol, standard volume in nm^3) of the moves: the
    numbers of the configuration, or with ``auto`` those of the water model
    (:func:`cufep.prep.water.bulk_water_values`; a model without them is refused when the
    configuration is read)."""
    from cufep.prep.water import bulk_water_values

    g = cfg.sampling.gcmc
    known = bulk_water_values(cfg.forcefield.water) or (None, None)
    mu = known[0] if g.excess_chemical_potential_kcal_per_mol == "auto" else g.excess_chemical_potential_kcal_per_mol
    volume = known[1] if g.standard_volume_nm3 == "auto" else g.standard_volume_nm3
    return float(mu), float(volume)


def gcmc_record(cfg: Config) -> dict:
    """What of ``sampling.gcmc`` the files of a leg with the moves depend on, the two numbers of the
    water as they are used (``auto`` resolved)."""
    mu, volume = gcmc_water_values(cfg)
    return {**cfg.sampling.gcmc.model_dump(mode="json"), "excess_chemical_potential_kcal_per_mol": mu,
            # the pressure at which the cell the leg is fixed in was equilibrated
            "cell_pressure_bar": float(cfg.thermodynamics.pressure_bar),
            "standard_volume_nm3": volume}


def leg_system(cfg: Config, prep: dict, leg: str):
    """(System, specification of the grand canonical moves of water or None) of a leg: the prepared
    System as every MD of the leg and its alchemical System are made of it.

    With the moves (:func:`gcmc_in`) the waters that are ghosts at the start have no nonbonded
    interactions in it (:func:`cufep.sampling.gcmc.ghosts_off`): the pre-equilibration, which has a
    barostat, settles the cell without them, and the alchemical run, which has none, starts with
    the waters it then has. One function for the run of a leg and the check before a run."""
    import openmm as mm

    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    if not gcmc_in(cfg, leg):
        return system, None
    from cufep.sampling.gcmc import GcmcError, gcmc_spec, ghosts_off

    g = cfg.sampling.gcmc
    mu, volume = gcmc_water_values(cfg)
    try:
        spec = gcmc_spec(system, prep["topology"], prep["positions"], np.asarray(prep["box"], dtype=float),
                         prep["ligand"], cfg.dynamics.cutoff_nm, cfg.alchemy.softcore_alpha, cfg.alchemy.softcore_b,
                         cfg.thermodynamics.temperature_K, g.interval_ps, g.attempts, g.batch_size,
                         g.radius_margin_nm, mu, volume, None if g.ghost_waters == "auto" else int(g.ghost_waters))
    except GcmcError as exc:
        raise WorkflowError(f"sampling.gcmc cannot be used with this system: {exc}") from exc
    ghosts_off(system, spec.waters[spec.ghosts].reshape(-1))
    return system, spec


def alchemy_record(cfg: Config) -> dict:
    """The ``alchemy`` section as the conditions of samples record it (ALCHEMY_SETTINGS_ADDED)."""
    record = cfg.alchemy.model_dump(mode="json")
    return {key: value for key, value in record.items()
            if key not in ALCHEMY_SETTINGS_ADDED or value != ALCHEMY_SETTINGS_ADDED[key]}


def electrostatics_annihilated(conditions: dict | None) -> bool:
    """Whether samples with these recorded conditions were made with the electrostatics inside
    the ligand annihilated (``alchemy.annihilate_electrostatics``); samples that do not say so
    were made with them kept, the only treatment there was."""
    settings = (conditions or {}).get("settings")
    alchemy = settings.get("alchemy") if isinstance(settings, dict) else None
    return bool(alchemy.get("annihilate_electrostatics", False)) if isinstance(alchemy, dict) else False


def alchemical_box(eq, gcmc) -> np.ndarray:
    """The cell in which the alchemical run of a leg starts, from its pre-equilibration (the arrays
    of ``equil.npz``): the cell of its last configuration. A leg with grand canonical moves of water
    has no barostat after that and stays in one cell: the mean cell of the second half of the
    pre-equilibration (``mean_box``; cuFEP's own choice: grand's examples go on from the last frame
    of an MD at constant pressure, whose volume is one value of a fluctuating one)."""
    if gcmc is None:
        return np.asarray(eq["box"], dtype=float)
    if "mean_box" not in eq.files:
        raise WorkflowError("the pre-equilibration of this leg does not record its mean cell, which a leg with "
                            "sampling.gcmc is fixed in: it was written by a version without it. Nothing was "
                            "changed; run the calculation in a new output.dir")
    return np.asarray(eq["mean_box"], dtype=float)


def positions_in_cell(positions, box, new_box, molecules) -> np.ndarray:
    """A configuration moved into another cell as a barostat moves it: every molecule is shifted as
    a whole, so that its centre keeps its fractional coordinates. No distance within a molecule
    changes (constraints, virtual sites)."""
    x = np.array(positions, dtype=float)
    transform = np.linalg.inv(np.asarray(box, dtype=float)) @ np.asarray(new_box, dtype=float)
    for atoms in molecules:
        index = np.fromiter(atoms, dtype=np.int64)
        centre = x[index].mean(axis=0)
        x[index] += centre @ transform - centre
    return x


def samples_settings(cfg: Config, leg: str | None = None) -> dict:
    """The settings that the samples of a leg depend on and that their other recorded conditions
    (states, temperature, pressure, dynamics, System checksum) do not name in the words of the
    configuration: the soft-core and the electrostatics inside the ligand, REST2, the length of an
    iteration and the tail correction; and, for the ``leg`` that has them (the complex leg), the
    water relocation moves. The moves are written only where they are made, so that the samples
    of every leg without them, also those of earlier versions, have the record they had."""
    settings = {"alchemy": alchemy_record(cfg), "rest2": cfg.sampling.rest2.model_dump(mode="json"),
                "time_per_iteration_ps": float(cfg.sampling.time_per_iteration_ps),
                "tail": None if cfg.analysis.lrc == "none" else {"cutoff_nm": float(cfg.analysis.lrc_cutoff_nm)}}
    if leg == "complex" and cfg.sampling.relocation_mc.enabled:
        settings["relocation_mc"] = cfg.sampling.relocation_mc.model_dump(mode="json")
    if gcmc_in(cfg, leg):
        settings["gcmc"] = gcmc_record(cfg)
    return settings


def windows_settings(cfg: Config) -> dict:
    """The settings of the System that the start configurations of the lambda windows of a leg
    were made with, and its windows too where the same precalculation chose them: the soft-core
    and the electrostatics inside the ligand, and REST2. (What else the System depends on is
    recorded and checked where it is made: the prepared systems, the pre-equilibration, the
    restraint.) The settings of ``lambda_schedule`` are not among them: a leg that has its windows
    keeps them, and its log says so every time."""
    return {"alchemy": alchemy_record(cfg), "rest2": cfg.sampling.rest2.model_dump(mode="json")}


def start_record(cfg: Config, schedule, how: dict) -> dict:
    """What is written with the start configurations of the lambda windows of a leg (``made_with``
    of ``start_configs.npz``): how they were made (``how``: the way, and whether this
    precalculation chose the windows or they were given), the settings of the System they were
    made with (:func:`windows_settings`) and the windows they are of."""
    return {**how, "settings": windows_settings(cfg), "states": [state.to_dict() for state in schedule.states]}


def start_record_of(start_file: Path) -> dict:
    """The record written with the start configurations in ``start_file`` (:func:`start_record`);
    empty for a file of a version that wrote none."""
    with np.load(start_file) as data:
        made = json.loads(str(data["made_with"])) if "made_with" in data.files else {}
    return made if isinstance(made, dict) else {}


def starts_are_of(made: dict, schedule) -> bool:
    """Whether start configurations with the record ``made`` are those of the windows of
    ``schedule``. A record that does not name its windows (an earlier version) does not say they
    are others."""
    return made.get("states") is None or made["states"] == [state.to_dict() for state in schedule.states]


def check_windows_settings(made: dict, settings: dict, where: Path, log=print) -> bool:
    """Refuse lambda windows and start configurations that were made with another System than this
    run gives (review of dd075d6: a leg that had its windows and no samples yet took them over
    after ``alchemy.annihilate_electrostatics`` was changed, and its result described windows that
    adaptive lambda scheduling had chosen with the other System). Files of a version that did not
    record the settings are used as they are, and False is returned: a run that samples such
    windows for the first time says in its result that their origin could not be checked. The
    one check for the check before a run and for the run of a leg. ``settings``: those of this run
    (:func:`windows_settings` for the ABFE; a relative leg has its own)."""
    stored = made.get("settings")
    if stored is None:
        log(f"the lambda windows of this leg and their start configurations: {where} was written by a version that "
            "did not record the settings they were made with; they are used as they are")
        return False
    differing = settings_differences(stored, settings)
    if differing:
        raise WorkflowError(
            f"the lambda windows of this leg and their start configurations ({where}) were made with other settings "
            f"than this run gives ({'; '.join(differing[:8])}{' ...' if len(differing) > 8 else ''}). Nothing was "
            "changed. Use the same settings, or start the leg anew with --reschedule.")
    return True


def alchemical_leg(cfg: Config, system, prep: dict, box, restraint, is_complex: bool, gcmc=None):
    """The alchemical System of a leg as its samples are made with it, and what belongs to it.

    One function for the run of a leg (:func:`run_leg`) and for the check before a run
    (:func:`check_leg_reuse`), so that both get the same System to the byte of its XML, which is
    what the checksum in the conditions of the samples is taken of. The order is part of that:
    the equilibrated ``box`` is set on the prepared ``system`` of the leg (which is changed), the
    alchemical System is made from it, the force of the ``restraint`` (None for the solvent leg)
    is added, and the result is serialised. Needs no GPU. ``system`` and ``gcmc`` are those of
    :func:`leg_system`: with grand canonical moves of water the System has its ghosts switched off,
    and ``box`` is the cell the leg is fixed in (:func:`alchemical_box`).

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
                                               prep["protein"], prep.get("input_chains"))
    alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha, rest2=rest2_on,
                                  rest2_atoms=hot_atoms,
                                  bonds=[(b[0].index, b[1].index) for b in prep["topology"].bonds()],
                                  annihilate_electrostatics=cfg.alchemy.annihilate_electrostatics,
                                  softcore_b=cfg.alchemy.softcore_b)
    if restraint is not None:
        alch.system.addForce(restraint.create_force())
    xml = mm.XmlSerializer.serialize(alch.system)
    tail = None
    if cfg.analysis.lrc in ("reweight", "isotropic"):
        every_water = system
        if gcmc is not None:
            # the tail correction counts the waters of each configuration that interact (the worker
            # tells it which do not): it is set up with all of them
            from cufep.sampling.gcmc import every_water_on

            every_water = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
            every_water_on(every_water, gcmc)
        tail = TailEnergy.from_system(every_water, prep["ligand"], alch.ligand_parameters, cfg.dynamics.cutoff_nm,
                                      cfg.analysis.lrc_cutoff_nm).to_dict()
    # water relocation moves (the complex leg): what a mover needs of the fully interacting system
    relocation = None
    moves = cfg.sampling.relocation_mc
    if is_complex and moves.enabled:
        from cufep.sampling.relocation_mc import RelocationError, relocation_spec

        try:
            relocation = relocation_spec(
                system, prep["topology"], prep["positions"], np.asarray(box, dtype=float), prep["ligand"],
                cfg.dynamics.cutoff_nm, cfg.alchemy.softcore_alpha, cfg.alchemy.softcore_b,
                cfg.thermodynamics.temperature_K, moves.trial_poses,
                None if moves.inner_fraction == "uniform" else float(moves.inner_fraction), moves.interval_ps,
                moves.radius_margin_nm)
        except RelocationError as exc:
            raise WorkflowError(f"sampling.relocation_mc cannot be used with this system: {exc}") from exc
    return SimpleNamespace(alch=alch, xml=xml, tail=tail, rest2_on=rest2_on, hot_atoms=hot_atoms,
                           hot_residues=hot_residues, relocation=relocation, gcmc=gcmc,
                           leg="complex" if is_complex else "solvent")


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
    # a leg with grand canonical moves of water has no barostat: its cell is fixed (alchemical_box)
    fixed_cell = getattr(built, "gcmc", None) is not None
    return LegSpec(
        name=more.pop("name", "leg"),
        system_xml=built.xml,
        states=states,
        n_atoms=built.alch.system.getNumParticles(),
        backend=backend_name,
        precision=_precision(cfg, backend_name),
        platform=cfg.backend.platform,
        temperature_K=cfg.thermodynamics.temperature_K,
        pressure_bar=None if fixed_cell else cfg.thermodynamics.pressure_bar,
        barostat="none" if fixed_cell else barostat,
        barostat_frequency=cfg.dynamics.barostat_frequency,
        exchange=cfg.sampling.exchange,
        friction_per_ps=cfg.dynamics.friction_per_ps,
        timestep_fs=timestep_fs,
        steps_per_iteration=planned_iterations(iteration_ps, timestep_fs, 0.0, 0.0).steps_per_iteration,
        seed=seed,
        cpu_threads=cfg.resources.cpu_threads_per_worker,
        lrc=built.tail,
        settings=samples_settings(cfg, getattr(built, "leg", None)),
        relocation=getattr(built, "relocation", None),
        gcmc=getattr(built, "gcmc", None),
        box_safety=1.3 if barostat == "membrane" and not fixed_cell else None,
        **more,
    )


def conditions_of_this_run(cfg: Config, lig_dir: Path, leg: str, repeat: int, prep_info: dict, backend_name: str,
                           n_stored: int) -> dict:
    """The conditions under which this run would sample a leg that has files, made from those
    files as :func:`run_leg` makes them, without a GPU and without writing anything.

    For samples that record their conditions and not the settings behind them (one version of
    cuFEP wrote such): nothing but the checksum of the System says what soft-core, restraint or
    REST2 region they were made with, so the System of this run is built to compare it. Raises
    ConditionsError if a file that is needed for that is not there.
    """

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
    system, gcmc = leg_system(cfg, prep, leg)
    built = alchemical_leg(
        cfg, system, prep, alchemical_box(eq, gcmc), restraint, is_complex, gcmc
    )
    states = [state.to_dict() for state in Schedule.load(leg_dir / "schedule.yaml").states]
    barostat = prep_info["complex_barostat"] if is_complex else "isotropic"
    return leg_spec(cfg, built, states, barostat, float(eq["timestep_fs"]), backend_name, 0).conditions()


def leg_is_complete(leg_hrex: Path, total: int) -> bool:
    """Whether the samples in ``leg_hrex`` are complete as they are: their checkpoint names ``total``
    or more rows (those it records, or for a checkpoint of an earlier version, which did not record
    them, its iteration) and the stored samples are exactly those. Samples that hold rows their
    checkpoint does not name are not complete: they are resumed, and the resume cuts them to the
    checkpoint. The one rule of run_leg and of the check before a run."""
    from cufep.sampling.storage import stored_attributes

    path = Path(leg_hrex) / "checkpoint.npz"
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        extra = json.loads(str(data["extra"])) if "extra" in data.files else {}
        iteration = int(data["iteration"])
    stored, attrs = stored_attributes(leg_hrex)
    if "rows" in extra:
        return int(extra["rows"]) >= total and stored == int(extra["rows"])
    # a checkpoint of an earlier version does not say which rows are its own: after a warm-up that was
    # not stored, its iteration and the rows stored after it can add up to the same number (review of
    # PR 1). Such a leg is complete only if the result of its finished run was made from exactly the
    # rows it holds (result.previous.json once a run of the leg has begun)
    for name in ("result.json", "result.previous.json"):
        result_path = Path(leg_hrex).parent / name
        if result_path.exists():
            try:
                data = json.loads(result_path.read_text())
            except ValueError:
                return False
            used = data.get("n_iterations_stored")
            production = (data.get("analysis") or {}).get("n_iterations_used")
            if used is None and isinstance(production, int):
                used = production + int(attrs.get("n_equilibration_iterations", 0))
            return used == stored == iteration and iteration >= total
    return False


def take_complete_leg(storage, n_equilibration: int, total: int, cfg, log=print) -> None:
    """A leg that holds all its iterations (:func:`leg_is_complete`) is analysed as it is, with the
    boundary of the equilibration of this run, which can differ from that of the run that made its
    samples (review of PR 1: 20 + 100 iterations asked for as 40 + 80 were analysed with 20). The
    plan of this run is recorded with the samples, so that a later analysis takes the same."""
    log(f"the leg holds all its {total} iterations; its samples are analysed as they are")
    before = storage.attributes().get("n_equilibration_iterations")
    if before is not None and int(before) != int(n_equilibration):
        log(f"the first {n_equilibration} of its iterations are equilibration in this run ({before} in the run that "
            "made them)")
    storage.set_plan(n_equilibration, total, analysis_record(cfg))


def old_checkpoint_refusal(leg_hrex: Path) -> str | None:
    """Why the checkpoint in ``leg_hrex`` cannot be continued from, or None: one of an earlier
    version does not record how many of its iterations were stored, and iterations that were run
    without being stored (a warm-up) cannot be told apart from stored ones (review of PR 1)."""
    path = Path(leg_hrex) / "checkpoint.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        extra = json.loads(str(data["extra"])) if "extra" in data.files else {}
    if "rows" in extra:
        return None
    return (f"its checkpoint ({path}) was written by an earlier version of cuFEP and does not record how many of "
            "its iterations were stored; iterations run without being stored cannot be told from stored ones, so "
            "the leg is not continued from it. Nothing was changed. Its samples can be analysed as they are "
            "(cufep analyze); --reschedule starts the leg anew and keeps them in previous_<n>/")


def check_leg_reuse(cfg: Config, lig_dir: Path, leg: str, repeat: int, prep_info: dict, production_ns: float,
                    reschedule: bool = False, log=print) -> None:
    """Before a run writes or samples anything: what the leg directory holds must be usable by it.

    The same checks are made again where the files are used (:func:`run_leg`); here they are made
    for all legs of a run first, from the files alone, so that a run that will be refused leaves
    the output directory, and its record of the configuration, as they were. Raises WorkflowError
    or ConditionsError.
    """
    from cufep.alchemy.schedule import Schedule
    from cufep.sampling.storage import (
        ConditionsError,
        _same,
        conditions_differences,
        other_conditions,
        stored_conditions,
    )

    leg_dir = lig_dir / f"repeat_{repeat}" / leg
    is_complex = leg == "complex"
    in_membrane = is_complex and prep_info.get("complex_barostat") == "membrane"
    quiet = lambda message: None                                    # noqa: E731  run_leg logs it when it uses the file
    # a leg whose archive (--reschedule) was begun and not completed is not run as it is: part of
    # what it had is set aside, the rest is still in place (review of 2861b93: a run without
    # --reschedule made new start configurations and analysed the samples that were left, with the
    # record of the new ones)
    unfinished = sorted(leg_dir.glob(".previous_*.partial"))
    if unfinished and not reschedule:
        raise WorkflowError(
            f"the {leg} leg of repeat {repeat} was being started anew (--reschedule) when a run stopped: "
            f"{unfinished[0]} holds a part of what the leg had. Nothing was changed. Run this leg with "
            "--reschedule again to complete that (--ligand and --leg restrict a run to it).")
    # the schedule the leg is to be given, if it will take one (lambda_schedule.file)
    check_given_schedule(cfg, lig_dir, leg, repeat, reschedule)
    equil_file = leg_dir / "equil.npz"
    if equil_file.exists():
        eq = np.load(equil_file)
        check_equil_file(eq, equil_file, cfg, leg, in_membrane, leg_seed(cfg, repeat, leg), quiet)
    # the system on which the MD for a restraint is made and the pre-equilibration it starts from: those of
    # the leg, or the complex in the bilayer of a leg whose alchemical run is in water (restraint_system)
    stage = restraint_system(prep_info) if is_complex else None
    md_equil, md_in_membrane = equil_file, in_membrane
    if stage is not None:
        md_equil, md_in_membrane = leg_dir / RESTRAINT_EQUIL, stage.get("barostat") == "membrane"
        check_restraint_equil(cfg, leg_dir, prep_info, leg_seed(cfg, repeat, leg), quiet)
    rfile = leg_dir / "restraint.json"
    if is_complex and rfile.exists():
        check_restraint_file(_read_json(rfile) or {}, rfile, cfg, quiet)
        check_restraint_of_leg(_read_json(rfile) or {}, rfile, leg_dir)
    elif is_complex and cfg.restraints.selection == "interactions":
        # the trajectories from which the restraint is still to be chosen: those the leg has must be
        # ones of this run (the same settings, and the pre-equilibration they started from)
        from cufep.sampling.restraint_md import trajectory_file

        stored = [k for k in range(int(cfg.restraints.interactions.trajectories))
                  if trajectory_file(leg_dir, k).exists()]
        if stored and not md_equil.exists():
            raise WorkflowError(f"{trajectory_file(leg_dir, stored[0]).parent} holds MD for the restraint of this leg "
                                f"but {md_equil}, which it started from, is not there. Nothing was changed; run the "
                                "calculation in a new output.dir")
        if stored:
            made_with = restraint_md_made_with(cfg, leg_seed(cfg, repeat, leg), md_in_membrane, np.load(md_equil))
            for k in stored:
                check_restraint_md_file(trajectory_file(leg_dir, k), k, made_with, quiet)
    if reschedule:
        return                           # the windows and samples the leg has are set aside; it starts anew
    rest2_scale = cfg.sampling.rest2.scale if cfg.sampling.rest2.scale > 1.0 else 1.0
    sched_file = leg_dir / "schedule.yaml"
    # samples (or the checkpoint of their start) need the windows they are of: the leg's schedule, or the one it is
    # given; checked here, before begin_leg sets the result of the leg aside (fifth review of PR #18), after the
    # checks of the samples below, whose refusals say more of them (the windows of samples that are refused for
    # their conditions do not matter)
    from cufep.engine.leg import check_samples_have_their_windows

    src = given_schedule_file(cfg, lig_dir.name, leg, repeat)
    windows = sched_file if sched_file.exists() else (src if src is not None and src.exists() else None)

    def samples_have_their_windows() -> None:
        check_samples_have_their_windows(leg_dir, None if windows is None
                                         else [state.to_dict() for state in Schedule.load(windows).states])

    if sched_file.exists():
        schedule = Schedule.load(sched_file)
        if abs(schedule.rest2_scale - rest2_scale) > 1e-9:
            raise WorkflowError(
                f"the lambda schedule in {sched_file} was made with a REST2 scale of {schedule.rest2_scale:g} but "
                f"sampling.rest2.scale is {rest2_scale:g}. Nothing was changed. Use the same value, or start the "
                "leg anew with --reschedule.")
        start_file = leg_dir / "start_configs.npz"
        if start_file.exists():
            made = start_record_of(start_file)
            if starts_are_of(made, schedule):         # start configurations of other windows are made again
                check_windows_settings(made, windows_settings(cfg), start_file, quiet)
    n_stored, conditions = stored_conditions(leg_dir / "hrex")
    if n_stored == 0:
        samples_have_their_windows()                 # a checkpoint of the start without samples
        return
    # a leg that is to be continued from a checkpoint of an earlier version (run_leg analyses a leg
    # that has all its iterations as it is, without its checkpoint)
    used_timestep = (
        float(np.load(equil_file)["timestep_fs"]) if equil_file.exists() else float(cfg.dynamics.timestep_fs)
    )
    plan = planned_iterations(cfg.sampling.time_per_iteration_ps, used_timestep, cfg.sampling.equilibration(leg),
                              production_ns)
    refusal = old_checkpoint_refusal(leg_dir / "hrex")
    if refusal and conditions is not None and not leg_is_complete(leg_dir / "hrex",
                                                                  plan.n_equilibration + plan.n_production):
        raise ConditionsError(f"the {leg} leg of repeat {repeat}: {refusal}")
    if conditions is None:
        from cufep.sampling.storage import stored_attributes

        used_timestep = (
            float(np.load(equil_file)["timestep_fs"]) if equil_file.exists() else float(cfg.dynamics.timestep_fs)
        )
        # the windows the leg will use: its own, or those it is given (sixth review of PR #18)
        states = None if windows is None else [state.to_dict() for state in Schedule.load(windows).states]
        problems = unrecorded_samples_problems(stored_attributes(leg_dir / "hrex")[1], n_stored, leg_dir, cfg, leg,
                                               production_ns, states, used_timestep)
        if problems:
            refuse_unrecorded_samples(leg_dir, n_stored, problems)
        samples_have_their_windows()
        return
    current = {"temperature_K": float(cfg.thermodynamics.temperature_K),
               "pressure_bar": None if gcmc_in(cfg, leg) else float(cfg.thermodynamics.pressure_bar),
               "friction_per_ps": float(cfg.dynamics.friction_per_ps)}
    differing = [f"{key}: {conditions[key]} -> {value}" for key, value in current.items()
                 if key in conditions and not _same(conditions[key], value)]
    if isinstance(conditions.get("settings"), dict):
        differing += settings_differences(conditions["settings"], samples_settings(cfg, leg))
    elif not differing:
        # samples that record their conditions without the settings behind them: all their conditions,
        # with the checksum of the System, against those this run would sample under
        current = conditions_of_this_run(cfg, lig_dir, leg, repeat, prep_info,
                                         str(conditions.get("backend") or cfg.backend.name), n_stored)
        differing = conditions_differences(conditions, current)
    if differing:
        raise other_conditions(n_stored, leg_dir / "hrex", differing)
    samples_have_their_windows()


def check_prepared(cfg: Config, ligand, lig_dir: Path, membrane: bool, box_shape: str, log=print) -> dict | None:
    """The record of the prepared systems of a ligand if they exist and this run can use them (None
    if there are none yet); raises WorkflowError if they were made from other input or settings."""
    done = lig_dir / "prep" / "prep.json"
    if not done.exists():
        return None
    info = json.loads(done.read_text())
    check_made_from(f"the prepared systems of {ligand.name}", lig_dir / "prep", info.get("made_from"),
                    prep_made_from(cfg, ligand, membrane, box_shape), log,
                    contradictions=prep_contradictions(info, ligand.name, membrane, box_shape,
                                                       cfg.membrane.complex_leg))
    from cufep.prep.solvate import box_height_problem

    # a box an earlier version made too small for the cutoff: refused before any MD
    for leg in ("complex", "solvent", BILAYER_COMPLEX):
        stored = lig_dir / "prep" / f"{leg}.npz"
        if stored.exists():
            problem = box_height_problem(np.load(stored)["box"], cfg.dynamics.cutoff_nm)
            if problem:
                raise WorkflowError(f"the prepared {leg} system of {ligand.name} ({stored}) cannot be used: {problem}. "
                                    "Nothing was changed; prepare it again in a new output.dir with a larger padding")
    return info


def equil_notes(eq) -> list[str]:
    """The notes of a stored pre-equilibration, without the verdicts that earlier versions wrote into
    them with levels of cuFEP's own (whether the bilayer had settled: :data:`_ANALYSIS_NOTES`). How
    the bilayer changed is a number of the result (:func:`bilayer_numbers`)."""
    notes = json.loads(str(eq["notes"])) if "notes" in eq.files else []
    return [n for n in notes if not str(n).startswith(_ANALYSIS_NOTES)]


def bilayer_numbers(eq) -> dict | None:
    """How the bilayer of a stored pre-equilibration changed between the last two quarters of its
    equilibration (:func:`cufep.sampling.equilibrate.membrane_drift`), as numbers without a verdict;
    None for a system without a bilayer."""
    from cufep.sampling.equilibrate import membrane_drift

    if not equil_in_membrane(eq) or "area_nm2" not in eq.files:
        return None
    waters = np.asarray(eq["pocket_waters"]) if "pocket_waters" in eq.files else np.zeros(0)
    return membrane_drift(np.asarray(eq["area_nm2"]), waters)


def restraint_notes(stored: dict) -> list[str]:
    """The notes of a stored restraint, without the verdicts that earlier versions wrote into them
    with levels of cuFEP's own (whether the ligand had left its pose: :data:`_ANALYSIS_NOTES`). The
    distance of the ligand from its input pose is a number of the result (``ligand_rmsd_from_input_A``)."""
    info = stored.get("info") or {}
    return [n for n in info.get("notes", []) if not str(n).startswith(_ANALYSIS_NOTES)]


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
        # membrane.complex_leg and membrane.water were added later: bilayer, and a bilayer in forcefield.water,
        # are what every preparation before them did and are not recorded, so that those preparations are
        # still used; a second prepared complex (water) and a water model of the bilayer's own are
        "membrane": dict({k: data["membrane"][k] for k in ("orientation", "lipid", "padding_nm")} if membrane else {},
                         enabled=bool(membrane),
                         **({"complex_leg": data["membrane"]["complex_leg"]}
                            if membrane and data["membrane"]["complex_leg"] != MEMBRANE_SETTINGS_ADDED["complex_leg"]
                            else {}),
                         **({"water": cfg.bilayer_water}
                            if membrane and cfg.bilayer_water != cfg.forcefield.water else {})),
    }


def equil_made_with(cfg: Config, leg: str, in_membrane: bool, seed: int) -> dict:
    """The settings that the pre-equilibration of a leg depends on."""
    pre = cfg.sampling.pre_equilibration.complex if leg == "complex" else cfg.sampling.pre_equilibration.solvent
    out = {"temperature_K": cfg.thermodynamics.temperature_K, "pressure_bar": cfg.thermodynamics.pressure_bar,
           "friction_per_ps": cfg.dynamics.friction_per_ps, "timestep_fs": float(cfg.dynamics.timestep_fs),
           "pre_equilibration": pre.model_dump(mode="json"),
           "membrane_equilibration_ns": cfg.membrane.equilibration_ns if in_membrane else None,
           "backend": cfg.backend.name, "precision": cfg.backend.precision, "seed": int(seed)}
    # every version before dynamics.barostat_frequency existed used 25 steps and recorded nothing:
    # a record without the key was made with 25, so only another value is recorded
    if int(cfg.dynamics.barostat_frequency) != BAROSTAT_FREQUENCY_BEFORE_THE_SETTING:
        out["barostat_frequency"] = int(cfg.dynamics.barostat_frequency)
    # backend.fallback decides whether this MD may be repeated on stock OpenMM. Every version before that
    # repeated it there whatever the setting said and recorded nothing: a record without the key was made
    # with the fallback, so only another value is recorded (review of f5a77ce: a pre-equilibration made with
    # the fallback was used again by a run with backend.fallback: none)
    if cfg.backend.fallback != FALLBACK_BEFORE_IT_DECIDED_PLAIN_MD:
        out["fallback"] = cfg.backend.fallback
    # the precision the plain MD runs in, where earlier versions ran it in another one and recorded the one
    # asked: "single" ran in mixed precision, and "double" asked of the plugin, which has none, ran on stock
    # OpenMM in double precision (the plugin runs it in mixed precision now). A record of such a setting
    # without this key is of an MD in the other precision, not of this run's
    from cufep.backends.base import backend_precision

    runs_in = backend_precision(cfg.backend.name, cfg.backend.precision)
    if cfg.backend.precision == "single" or runs_in != cfg.backend.precision:
        out["plain_md_precision"] = runs_in
    # the platform, as the trajectories for a restraint record it. Earlier versions recorded none: of such a
    # file the platform is not known, and it is used as it was before (check_equil_file)
    out["platform"] = cfg.backend.platform
    # a pre-equilibration that holds the solute in the pose that was given (pre_equilibration_parts) is another
    # one than the free MD of the other selections and of every earlier version, whose records have no such key
    if pre_equilibration_parts(cfg, leg, in_membrane)[1]:
        out["solute_held"] = True
        # md_ns is the free MD of the other selections, which this pre-equilibration does not have: a change of
        # it is no other pre-equilibration (the fourth pass of Codex: the run was refused for it)
        out["pre_equilibration"] = {key: value for key, value in out["pre_equilibration"].items() if key != "md_ns"}
    # a leg with grand canonical moves of water is pre-equilibrated with its ghost waters switched off (they are
    # chosen from the prepared pose by these settings): a record without the key is of the system with every water
    if gcmc_in(cfg, leg):
        moves = gcmc_record(cfg)
        out["gcmc"] = {key: moves[key] for key in ("ghost_waters", "radius_margin_nm", "standard_volume_nm3")}
    return out


#: the barostat frequency of every version before ``dynamics.barostat_frequency``
BAROSTAT_FREQUENCY_BEFORE_THE_SETTING = 25
#: what every version did before ``backend.fallback`` decided the plain MD of a leg
FALLBACK_BEFORE_IT_DECIDED_PLAIN_MD = "openmm"


def manual_boresch_restraint(host_atoms, guest_atoms, ligand_atoms, eq, n_atoms: int, k_r: float, k_angle: float,
                             where: Path | str = "equil.npz", force_constants: str = "openfe",
                             temperature_K: float = 298.15, input_pose=None, input_box=None):
    """The Boresch restraint of ``restraints.selection: manual`` and the fluctuations of its six
    coordinates: the given anchors, with the reference values of the last frame of the
    pre-equilibration.

    The anchors are measured by :func:`cufep.alchemy.restraints.measure_anchors`, as anchors chosen
    from interactions are: every frame with the ligand in the periodic image nearest to the host
    anchors (from the raw coordinates a ligand one cell away gave r0 = 5.4 nm in a 5 nm cell, and
    the restraint held the reference configuration with 52,300 kJ/mol, review of PR 1), and the
    anchors must define the six coordinates in every frame
    (:func:`cufep.alchemy.restraints.anchor_geometry_problem`, the rule of the search). With
    ``force_constants: auto`` the constants come from the fluctuations of the six coordinates over
    the frames, as for anchors the search chose (review of PR 1: they were ignored for manual
    anchors). ``n_atoms`` is not read any more. Returns (restraint, fluctuations).
    """
    from cufep.alchemy.restraints import (
        BoreschRestraint,
        constants_from_fluctuations,
        coordinate_fluctuations,
        flat_rule_problem,
        measure_anchors,
    )

    if "frame_boxes" not in eq:
        raise WorkflowError(f"{where} holds no boxes of its frames (an earlier version wrote it): the reference "
                            "values of a manual restraint cannot be measured across the periodic boundary. Nothing "
                            "was changed; run the calculation in a new output.dir")
    frame_atoms = np.asarray(eq["frame_atoms"], dtype=int)
    missing = sorted(set(int(i) for i in (*host_atoms, *guest_atoms)) - set(frame_atoms.tolist()))
    if missing:
        raise WorkflowError(
            f"restraints.host_atoms / guest_atoms {missing} are not among the atoms whose positions the "
            f"pre-equilibration kept ({where}); the anchors must be atoms of the protein and the ligand"
        )
    host, guest = tuple(int(i) for i in host_atoms), tuple(int(i) for i in guest_atoms)
    # the positions the alchemical run starts from, of which the last frame is a copy in single precision
    has_start = "positions" in eq and "box" in eq
    measured = measure_anchors(host, guest, ligand_atoms, eq["frames"], frame_atoms, eq["frame_boxes"], None,
                               start=eq["positions"] if has_start else None,
                               start_box=eq["box"] if has_start else None)
    fluctuations = coordinate_fluctuations(measured.values)
    restraint = BoreschRestraint(host, guest, *[float(x) for x in measured.values[-1]], k_r, k_angle, k_angle,
                                 k_angle, k_angle, k_angle)
    how = ""
    if force_constants == "auto":
        restraint, _flat = constants_from_fluctuations(restraint, measured.values, measured.held_theta, temperature_K)
        how = " with the force constants of their fluctuations (restraints.force_constants: auto)"
    # OpenFE's rule of the flat angle, asked once: with the force constants the restraint has, of every frame
    # (the last one gives the reference values) and of the pose the pre-equilibration started from
    theta = measured.held_theta
    if input_pose is not None and input_box is not None:
        pose = measure_anchors(host, guest, ligand_atoms, np.asarray(input_pose, dtype=float)[None],
                               np.arange(len(input_pose)), np.asarray(input_box, dtype=float)[None], None)
        theta = np.concatenate([theta, pose.held_theta])
    problem = flat_rule_problem(restraint.k_theta_a, restraint.k_theta_b, theta, temperature_K) or measured.problem
    if problem:
        raise WorkflowError(f"restraints.host_atoms {list(host)} / guest_atoms {list(guest)}{how}: {problem}. "
                            "Choose other anchors; nothing was sampled")
    return restraint, fluctuations


def restraint_made_with(cfg: Config) -> dict:
    """The settings that the choice of the Boresch restraint depends on: the section ``restraints``.
    Its part ``interactions`` belongs to ``selection: interactions`` alone and is recorded only
    then: a restraint of another selection was made without it, and the record of one made by a
    version before that selection existed does not have it and stays usable."""
    data = cfg.restraints.model_dump(mode="json")
    if cfg.restraints.selection != "interactions":
        data.pop("interactions", None)
    return data


def residue_names(topology, input_chains: dict | None = None) -> dict[int, str]:
    """What the log and the results call every residue, by its index in the topology: name, number
    and insertion code as the input has them (``VAL981``). Where two residues of the protein would
    share that name (the two chains of a dimer with one numbering), each is followed by its chain
    (``VAL981.B``, as ProLIF writes a residue): the chain of the input, which ``input_chains``
    gives for a chain of the prepared topology (the preparation makes a chain of every piece
    between chain breaks and names them anew: the two pieces of one chain of the A2A example were
    named ``ALA59.A`` and ``HIS278.B``, review of eb6f89f); where that does not tell them apart
    either, also the number of the prepared chain among those of the protein (``VAL981.X#2``)."""
    from cufep.prep.protein import AMINO_ACIDS, CAPS

    chains, plain = [], {}
    for residue in topology.residues():
        plain[residue.index] = f"{residue.name}{residue.id}{str(getattr(residue, 'insertionCode', '') or '').strip()}"
        if (residue.name in AMINO_ACIDS or residue.name in CAPS) and residue.chain not in chains:
            chains.append(residue.chain)
    number = {chain.index: k + 1 for k, chain in enumerate(chains)}
    given = {chain.index: str((input_chains or {}).get(chain.id, chain.id) or "").strip() for chain in chains}
    names = dict(plain)
    for behind in (given, number):
        shared: dict[str, set[int]] = {}
        for residue in topology.residues():
            if residue.chain.index in number:
                shared.setdefault(names[residue.index], set()).add(residue.index)
        for residue in topology.residues():
            index = residue.index
            if residue.chain.index in number and len(shared[names[index]]) > 1 and behind[residue.chain.index]:
                # the chain of the input behind a dot; the number of the piece behind a "#", which no chain id is
                # read as (three pieces named X, X and 1 would otherwise be VAL981.1, VAL981.2 and VAL981.1)
                chain = residue.chain.index
                names[index] += f".{given[chain]}" if behind is given else f"#{number[chain]}"
    return names


def atom_label(atoms: list, index: int, residues: dict[int, str]) -> str:
    """An atom of the System as the log and the results name it: its residue (:func:`residue_names`)
    and its name."""
    atom = atoms[int(index)]
    return f"{residues[atom.residue.index]}:{atom.name}"


def check_interaction_inputs(cfg: Config, ligands) -> None:
    """Refuse, before anything is written, a run with ``restraints.selection: interactions`` that
    cannot choose a restraint: ProLIF or MDAnalysis cannot be imported, or a ligand has no three
    bonded heavy atoms that can be guest anchors (L1 with two heavy atoms bonded to it, L2 and L3
    no chain ends), which neither the interactions nor the distance method can do without and
    which the SDF tells before any MD is run. ``ligands`` are those whose complex leg is to run:
    without one there is no restraint to choose and nothing is asked."""
    if cfg.restraints.selection != "interactions" or not ligands:
        return
    from cufep.alchemy.interaction_anchors import LigandGraph
    from cufep.analysis.interactions import packages_problem, with_rdkit_aromaticity

    problem = packages_problem()
    if problem:
        raise WorkflowError(problem)
    without = [lig.name for lig in ligands
               if not LigandGraph.from_rdkit(with_rdkit_aromaticity(lig.to_rdkit())).has_guest_anchors()]
    if without:
        raise WorkflowError(
            f"restraints.selection: interactions cannot give {', '.join(without)} a restraint: the ligand has no "
            "atom with two heavy atoms bonded to it from which two more bonded heavy atoms that are no chain ends "
            "lead on (L1-L2-L3). Give the anchors (restraints.selection: manual) or use restraints.selection: auto")


def check_restraint_start(stored: dict, rfile: Path, eq, name: str = "equil.npz") -> None:
    """Refuse a restraint chosen from interactions that is not of this leg. Its record holds the
    configuration its trajectories started from, the end of the pre-equilibration ``eq`` (the file
    ``name`` of the leg), as every trajectory does (``start``). The settings alone do not tell the
    restraint of another repeat, or of a pre-equilibration that was made again, from this leg's
    (review of eb6f89f). A restraint of another selection has no such record. The legs ask through
    :func:`check_restraint_of_leg`, which knows which pre-equilibration the MD started from."""
    from cufep.sampling.restraint_md import start_digest

    recorded = (stored.get("info") or {}).get("start")
    if recorded is not None and recorded != start_digest(eq["positions"], eq["velocities"], eq["box"]):
        raise WorkflowError(
            f"the restraint in {rfile} cannot be used again: it was chosen from MD that started from another "
            f"configuration than the pre-equilibration of this leg ends in ({name}). Nothing was changed. A "
            "calculation with other input or settings needs a new output.dir.")


def restraint_md_made_with(cfg: Config, seed: int, in_membrane: bool, eq) -> dict:
    """What a trajectory for the restraint of a leg is made from: its length and frame interval,
    the settings of every plain MD of the leg, the seed of the leg and the configuration it starts
    from (the end of the pre-equilibration ``eq``). Not the number of trajectories (more of them
    use those that are there) and not ``restraints.interactions.discard_ns``, which only the
    analysis reads."""
    from cufep.sampling.restraint_md import start_digest

    ri = cfg.restraints.interactions
    return {"length_ns": float(ri.length_ns), "frame_interval_ps": float(ri.frame_interval_ps),
            "temperature_K": float(cfg.thermodynamics.temperature_K),
            "pressure_bar": float(cfg.thermodynamics.pressure_bar),
            "friction_per_ps": float(cfg.dynamics.friction_per_ps),
            "barostat_frequency": int(cfg.dynamics.barostat_frequency),
            "barostat": "membrane" if in_membrane else "isotropic", "timestep_fs": float(eq["timestep_fs"]),
            "backend": cfg.backend.name, "fallback": cfg.backend.fallback, "precision": cfg.backend.precision,
            "platform": cfg.backend.platform, "seed": int(seed),
            "start": start_digest(eq["positions"], eq["velocities"], eq["box"])}


def check_restraint_md_file(path: Path, index: int, made_with: dict, log=print) -> None:
    """Refuse a stored trajectory for the restraint that this run cannot use: one made from other
    settings or another start (:func:`check_made_from`), and one that holds less MD than asked."""
    with np.load(path) as data:
        stored = json.loads(str(data["made_with"])) if "made_with" in data.files else None
        done = float(data["simulated_ns"]) if "simulated_ns" in data.files else 0.0
        timestep = float(data["timestep_fs"]) if "timestep_fs" in data.files else float(made_with["timestep_fs"])
    check_made_from(f"MD {index} for the restraint of this leg", path, stored, dict(made_with, trajectory=int(index)),
                    log, contradictions=["what it was made from: not recorded -> " + json.dumps(made_with)[:80]])
    from cufep.steps import md_steps

    # in whole steps, as the MD counts them: a length that is half a step more than a whole number of steps is
    # rounded by one rule (0.400002 ns at 4 fs is 100000 steps, which a comparison of the two times refused)
    if md_steps(done * 1000.0, timestep) < md_steps(float(made_with["length_ns"]) * 1000.0, timestep):
        raise WorkflowError(f"MD {index} for the restraint of this leg in {path} cannot be used again: it holds "
                            f"{done * 1000:g} ps where {float(made_with['length_ns']) * 1000:g} ps are asked for. "
                            "Nothing was changed; run the calculation in a new output.dir")


def restraint_trajectories(cfg: Config, job: "LegJob", leg_dir: Path, prep: dict, eq, in_membrane: bool, seed: int,
                           log=print):
    """The trajectories from which the restraint of a complex leg is chosen
    (:mod:`cufep.sampling.restraint_md`): those the leg has are checked and kept, the others are
    run, at the same time on the GPUs the scheduler grants. Returns them as the analysis reads them.

    The job asks for GPUs only if a trajectory is missing. When they are done it goes on with the
    first GPU of the grant, which is the one the scheduler leaves it when it says that it needs one
    worker again: ``job.devices`` is set to it, and the caller must use it from then on.
    """
    from cufep.mps import thread_percentage
    from cufep.sampling.restraint_md import TrajectoryTask, read_trajectories, run_trajectories, trajectory_file

    ri = cfg.restraints.interactions
    made_with = restraint_md_made_with(cfg, seed, in_membrane, eq)
    missing = []
    for k in range(int(ri.trajectories)):
        if trajectory_file(leg_dir, k).exists():
            check_restraint_md_file(trajectory_file(leg_dir, k), k, made_with, log)
        else:
            missing.append(k)
    if missing:
        settings = plain_md_settings(cfg, float(eq["timestep_fs"]), "membrane" if in_membrane else "isotropic", seed,
                                     production_ns=float(ri.length_ns), frame_interval_ps=float(ri.frame_interval_ps),
                                     bilayer_report=False)
        devices = list(job.devices)
        granted = None
        if job.control is not None and not job.inline and len(missing) > 1:
            work = md_work(ri.length_ns * 1000.0, len(prep["positions"]), cfg.sampling.time_per_iteration_ps)
            granted = job.control.request_devices(
                len(missing), len(missing) * work / max(1, job.oversubscription), pinned=True,
                what=f"{len(missing)} trajectories for the restraint")
            devices = granted or devices
        tasks = [TrajectoryTask(index=k, path=str(trajectory_file(leg_dir, k)), system_xml=prep["system_xml"],
                                positions=np.asarray(eq["positions"]), velocities=np.asarray(eq["velocities"]),
                                box=np.asarray(eq["box"]), frame_atoms=[int(i) for i in prep["solute"]],
                                settings=settings, timestep_fs=float(eq["timestep_fs"]),
                                made_with=dict(made_with, trajectory=k),
                                cpu_threads=int(cfg.resources.cpu_threads_per_worker),
                                mps_thread_percentage=thread_percentage(job.oversubscription))
                 for k in missing]
        on_gpus = [str(d) for d in devices if d is not None]             # a run without GPUs has the device None
        log(f"MD for the restraint: {len(missing)} of {ri.trajectories} trajectories of {ri.length_ns:g} ns to run "
            f"on {len(devices)} worker(s)" + (f" (GPU {', '.join(on_gpus)})" if on_gpus else ""))
        try:
            for done in run_trajectories(tasks, devices, log, inline=job.inline):
                log(f"MD {done['index']} for the restraint: {done['frames']} frames on {done['backend']} "
                    f"(time step {done['timestep_fs']:g} fs)" + "".join(f"; {note}" for note in done["notes"]))
        finally:
            if granted:
                # the job is serial again and stays on the first GPU of the grant, the one the scheduler keeps for it
                job.devices = [granted[0]]
                job.control.serial()
    return read_trajectories(leg_dir, int(ri.trajectories))


def interaction_restraint(cfg: Config, prep: dict, rdmol, eq, trajectories, temperature_K: float, start=None):
    """The Boresch restraint of ``restraints.selection: interactions`` from the trajectories of the
    leg: ``(restraint, info, report)``. ``info`` is what ``restraint.json`` records of the choice,
    ``report`` the numbers of the analysis for ``interactions.json``.

    ``start`` is the configuration the alchemical run starts from, as ``(positions of the atoms of
    prep, box)``, where that is not the end of the pre-equilibration ``eq`` from which the
    trajectories started (``membrane.complex_leg: water``: the trajectories are of the complex in
    the bilayer, the alchemical run of the complex in water). Every candidate is held to the rules
    there, as it is at the start in every other case.

    The frames later than ``discard_ns`` are analysed with the ligand in the periodic image
    nearest to the protein. ProLIF gives the interactions of every frame
    (:mod:`cufep.analysis.interactions`), the rule ranks the candidates
    (:mod:`cufep.alchemy.interaction_anchors`), and the first of them that passes the rules every
    restraint of cuFEP is held to is taken (:func:`cufep.alchemy.restraints.measure_anchors`, with
    the configuration the alchemical run starts from among the frames). The reference values are
    those of the histograms (:func:`cufep.alchemy.restraints.reference_values_from_histograms`).
    """
    from cufep.alchemy import interaction_anchors as rule
    from cufep.alchemy.restraints import (
        COORDINATES,
        BoreschRestraint,
        RestraintError,
        backbone_hosts,
        constants_from_fluctuations,
        coordinate_fluctuations,
        flat_rule_problem,
        ligand_next_to_protein,
        measure_anchors,
        reference_values_from_histograms,
        reference_values_problem,
        stable_structure_fractions,
    )
    from cufep.analysis.interactions import (
        interactions_of_frames,
        occupancies,
        protein_atoms_of,
        versions,
        with_rdkit_aromaticity,
    )

    rc, ri = cfg.restraints, cfg.restraints.interactions
    analysed = trajectories.after(ri.discard_ns)
    n = int(analysed.frames.shape[0])
    if n == 0:
        raise WorkflowError(f"no frame of the MD for the restraint is later than restraints.interactions.discard_ns = "
                            f"{ri.discard_ns:g} ns: there is nothing to choose a restraint from")
    topology = prep["topology"]
    atoms = list(topology.atoms())
    frame_atoms = np.asarray(analysed.frame_atoms, dtype=int)
    row = {int(a): i for i, a in enumerate(frame_atoms)}
    ligand = [int(i) for i in prep["ligand"]]
    absent = [i for i in ligand if i not in row]
    if absent:
        raise WorkflowError(f"the MD for the restraint kept no positions of the ligand atoms {absent[:5]}")
    # the residues that can hold host anchors, among the atoms of the frames
    hosts = {r: trio for r, trio in backbone_hosts(topology).items() if all(a in row for a in trio)}
    if not hosts:
        raise WorkflowError("the frames of the MD for the restraint hold no protein backbone")
    # a threshold of no frame is refused when the configuration is read; an MD that was repeated at 2 fs can keep
    # fewer frames than were planned, and the same holds for the frames there are (review of 877ab88)
    for name, fraction in (("min_frame_fraction", ri.min_frame_fraction),
                           ("fallback_frame_fraction", ri.fallback_frame_fraction)):
        if rule.frames_needed(n, fraction) < 1:
            raise WorkflowError(f"restraints.interactions: {n} frame(s) are analysed and {name} = {fraction:g} of "
                                "them is less than one frame: an interaction in no frame would be kept. Nothing was "
                                "sampled; run longer trajectories or keep frames more often")
    # whole molecules with the ligand in the image next to the protein: ProLIF and the angles of the rule know no
    # box. Next to its nearest protein atom, whatever atom that is: with the C-alpha atoms alone, a ligand held by a
    # side chain at the edge of a tight cell was nearer to a C-alpha of the image behind it (review of 877ab88)
    protein_rows = np.array([row[int(i)] for i in protein_atoms_of(topology, among=row)], dtype=int)
    xyz = ligand_next_to_protein(analysed.frames, np.array([row[i] for i in ligand], dtype=int), protein_rows,
                                 np.asarray(analysed.boxes, dtype=float))
    records = interactions_of_frames(topology, xyz, frame_atoms, ligand, rdmol)
    stable = stable_structure_fractions(topology, xyz, frame_atoms)
    # the backbone atoms of the protein residues that are no hosts (caps), which the distance method looks at too
    from cufep.prep.protein import AMINO_ACIDS, CAPS

    others: dict[int, tuple[int, ...]] = {}
    for atom in atoms:
        residue = atom.residue
        if (residue.index not in hosts and (residue.name in AMINO_ACIDS or residue.name in CAPS)
                and atom.name in ("N", "CA", "C") and atom.index in row):
            others[residue.index] = (*others.get(residue.index, ()), atom.index)
    frames = rule.Frames(xyz=xyz, atoms=frame_atoms, ligand=ligand, hosts=hosts, stable_fraction=stable,
                         boxes=np.asarray(analysed.boxes, dtype=float), trajectory=analysed.trajectory,
                         other_backbone=others)
    graph = rule.LigandGraph.from_rdkit(with_rdkit_aromaticity(rdmol))
    settings = rule.InteractionSettings(
        min_frame_fraction=ri.min_frame_fraction, angle_range_deg=tuple(ri.angle_range_deg),
        stable_structure_fraction=ri.stable_structure_fraction, second_interaction_ratio=ri.second_interaction_ratio,
        fallback_frame_fraction=ri.fallback_frame_fraction, fallback_cutoff_nm=ri.fallback_cutoff_nm)
    k_r, k_a = rc.k_r_kcal_per_A2 * KCAL * 100.0, rc.k_angle_kcal_per_rad2 * KCAL

    begin, begin_box = (eq["positions"], eq["box"]) if start is None else start

    def measured(host, guest):
        return measure_anchors(host, guest, ligand, analysed.frames, frame_atoms, analysed.boxes, None,
                               start=begin, start_box=begin_box, start_among_frames=True)

    def restraint_of(host, guest):
        """``(restraint, measurement, why it cannot be used or None)`` for a candidate: the restraint it
        would be, with reference values from the histograms and the force constants of the configuration
        (or of its fluctuations), held to the rules every restraint of cuFEP is held to. OpenFE's rule of
        the flat angle is asked once, with those force constants, of the frames, of the configuration the
        run starts from and of the reference angles themselves (the centre of a bin is not an angle of a
        frame)."""
        m = measured(host, guest)
        problem = m.problem or reference_values_problem(m.values, ri.angle_range_deg)
        if problem:
            return None, m, problem
        reference = reference_values_from_histograms(m.values, ri.angle_range_deg)
        made = BoreschRestraint(tuple(int(i) for i in host), tuple(int(i) for i in guest),
                                *[float(x) for x in reference], k_r, k_a, k_a, k_a, k_a, k_a)
        if rc.force_constants == "auto":
            made, _flat = constants_from_fluctuations(made, m.values, m.held_theta, temperature_K)
        theta = np.concatenate([m.held_theta, [[made.theta_a0, made.theta_b0]]])
        return made, m, flat_rule_problem(made.k_theta_a, made.k_theta_b, theta, temperature_K)

    try:
        choice = rule.choose_anchors(frames, graph, records, settings,
                                     lambda host, guest: restraint_of(host, guest)[2])
        restraint, values, _none = restraint_of(choice.host, choice.guest)
    except RestraintError as exc:
        raise WorkflowError(f"{exc}. Nothing was sampled") from exc
    reference = [restraint.r0, restraint.theta_a0, restraint.theta_b0, restraint.phi_a0, restraint.phi_b0,
                 restraint.phi_c0]
    fluctuations = coordinate_fluctuations(values.values)

    names = residue_names(topology, prep.get("input_chains"))

    def residue_label(index: int) -> str:
        return names[int(index)]

    def described(candidate) -> dict:
        out = candidate.to_dict()
        out.update(host=[atom_label(atoms, i, names) for i in candidate.host],
                   guest=[atom_label(atoms, i, names) for i in candidate.guest],
                   host_atoms=[int(i) for i in candidate.host], guest_atoms=[int(i) for i in candidate.guest],
                   residue=residue_label(candidate.residue), residue_index=int(candidate.residue))
        return out

    chosen = choice.chosen
    refused = [described(c) for ranked in choice.ranked.values() for c in ranked if c.refused]
    # the interactions the chosen anchors rest on: every kept type of their ligand atom and residue. The
    # candidate holds the frames with any of them and is named after the type of most weight, whose own
    # frames can be fewer (review of eb6f89f: "Anionic in 400 of 500 frames" for an Anionic of 200)
    # A contact of the distance method is no interaction of ProLIF: it has its frames and no types
    l1 = ligand.index(int(choice.guest[0]))
    by_interaction = choice.method == "interactions"
    kept_types = {kind: int(len(choice.table.frames_of[(atom, residue, kind)]))
                  for atom, residue, kind in choice.table.kept
                  if atom == l1 and residue == chosen.residue} if by_interaction else None
    of_type = kept_types.get(chosen.interaction, int(chosen.n_frames)) if by_interaction else int(chosen.n_frames)
    in_degrees = lambda v: [float(v[0]), *[float(np.degrees(x)) for x in v[1:]]]             # noqa: E731
    info = {
        "selection": "interactions", "force_constants": rc.force_constants, "fluctuations": fluctuations,
        "flat_angle_rule": FLAT_ANGLE_RULE,
        "method": choice.method,
        "anchors": {"host": [atom_label(atoms, i, names) for i in choice.host],
                    "guest": [atom_label(atoms, i, names) for i in choice.guest]},
        "interaction": {"residue": residue_label(chosen.residue), "type": chosen.interaction,
                        "frames": of_type, "of_frames": n,
                        "frames_with_any_kept_type": int(chosen.n_frames) if by_interaction else None,
                        "kept_types": kept_types},
        # tested: held to the rules, in the order of the ranking, until one passed; ranked: all there were
        "score": chosen.score, "candidates_tested": len(refused) + 1,
        "candidates_ranked": sum(len(v) for v in choice.ranked.values()), "candidates_refused": len(refused),
        "start_values": dict(zip(COORDINATES, in_degrees(values.start_values))),
        "start_values_units": {"r": "nm", "angles": "degrees"},          # the restraint itself is in nm and radians
        "md": {"trajectories": int(ri.trajectories), "frames_analysed": n, "backends": list(trajectories.backends),
               "timesteps_fs": [float(x) for x in trajectories.timesteps_fs]},
        "versions": versions(), "notes": list(trajectories.notes)}
    rows = choice.table.rows(frames)
    for entry in rows:
        entry.update(ligand_atom_name=atom_label(atoms, entry["ligand_atom"], names),
                     residue_name=residue_label(entry["residue"]))
    by_residue = occupancies(records)
    for entry in by_residue:
        entry["residue_name"] = residue_label(entry["residue"])
    series = np.asarray(values.values, dtype=float)
    report = {
        "frames_analysed": n, "frames_needed": int(choice.table.needed),
        "frames_needed_distance": rule.frames_needed(n, ri.fallback_frame_fraction),
        "trajectory_of_frame": [int(k) for k in analysed.trajectory],
        "interactions": rows, "residues": by_residue,
        # by the index of the residue: two residues can have one name
        "stable_structure_fraction": [{"residue": int(r), "residue_name": residue_label(r), "fraction": float(x)}
                                      for r, x in sorted(stable.items())
                                      if r in {c.residue for ranked in choice.ranked.values() for c in ranked}
                                      | {int(entry["residue"]) for entry in (*rows, *by_residue)}],
        "method": choice.method,
        "candidates": {method: [described(c) for c in ranked[:200]] for method, ranked in choice.ranked.items()},
        "candidates_in_all": {method: len(ranked) for method, ranked in choice.ranked.items()},
        "refused_candidates": refused,           # every one that was ranked before the chosen one and passed over
        "chosen": described(chosen),
        "reference_values": dict(zip(COORDINATES, in_degrees(reference))),
        "coordinates": {name: [float(x) for x in (series[:, i] if i == 0 else np.degrees(series[:, i]))]
                        for i, name in enumerate(COORDINATES)},
        "units": {"r": "nm", "angles": "degrees"},
        "md": info["md"], "versions": info["versions"]}
    return restraint, info, report


def restraint_from_interactions(cfg: Config, job: "LegJob", leg_dir: Path, prep: dict, rdmol, eq, in_membrane: bool,
                                seed: int, n_atoms: int, log=print, start=None):
    """The restraint stage of a complex leg with ``restraints.selection: interactions``:
    ``(restraint, info, seconds of the MD)``.

    The trajectories the leg does not have are run (:func:`restraint_trajectories`), their
    interactions are analysed and the restraint is chosen (:func:`interaction_restraint`), and
    the numbers of the analysis are written to ``interactions.json`` and ``interactions.csv`` in
    the directory of the leg. The caller writes ``restraint.json``, which is what says that the
    stage is done: a leg that stopped before that runs the stage again, with the trajectories it
    has, and gets the same restraint. ``n_atoms`` is the number of particles of the System.
    ``start`` is passed on to :func:`interaction_restraint`.
    """
    from cufep.alchemy.restraints import ligand_pose_rmsd
    from cufep.analysis.interactions import packages_problem

    problem = packages_problem()
    if problem:
        raise WorkflowError(problem)
    t_md = time.perf_counter()
    trajectories = restraint_trajectories(cfg, job, leg_dir, prep, eq, in_membrane, seed, log)
    md_seconds = time.perf_counter() - t_md
    restraint, info, report = interaction_restraint(cfg, prep, rdmol, eq, trajectories,
                                                    cfg.thermodynamics.temperature_K, start=start)
    # how far the ligand moved from the pose it was given, measured as for every selection, over the free MD
    # of this one: the analysed frames of the trajectories (at most 200 of them, evenly spaced, as whole systems)
    analysed = trajectories.after(cfg.restraints.interactions.discard_ns)
    some = np.unique(np.linspace(0, analysed.frames.shape[0] - 1, min(200, analysed.frames.shape[0])).astype(int))
    full = np.zeros((len(some), int(n_atoms), 3), dtype=np.float32)
    full[:, np.asarray(analysed.frame_atoms, dtype=int), :] = analysed.frames[some]
    rmsd = ligand_pose_rmsd(prep["topology"], full, prep["ligand"], rdmol, prep["positions"],
                            frame_boxes_nm=np.asarray(analysed.boxes)[some], input_box_nm=prep["box"],
                            host_max_distance_nm=cfg.restraints.host_max_distance_nm, second_half=False)
    del full
    from cufep.sampling.restraint_md import start_digest

    info.update(ligand_rmsd_A=10 * rmsd, ligand_rmsd_reference="input pose",
                start=start_digest(eq["positions"], eq["velocities"], eq["box"]))
    report["made_with"] = restraint_made_with(cfg)
    report["start"] = info["start"]              # with the anchors, what ties this analysis to the restraint of the leg
    publish_text(leg_dir / "interactions.json", json.dumps(report, indent=1))
    publish_text(leg_dir / "interactions.csv", interactions_csv(report))
    kept = sum(1 for row in report["interactions"] if row["kept"])
    log(f"interactions over {report['frames_analysed']} frames: {len(report['interactions'])} of ligand atoms with "
        f"residues, {kept} of them in {report['frames_needed']} frames or more")
    several = info["interaction"]["frames_with_any_kept_type"] not in (None, info["interaction"]["frames"])
    log(f"restraint ({info['method']}: {info['interaction']['type']} with {info['interaction']['residue']} in "
        f"{info['interaction']['frames']} of {info['interaction']['of_frames']} frames"
        + (f", {info['interaction']['frames_with_any_kept_type']} with the other kept interactions of that atom and "
           "residue" if several else "") + "): host "
        + ", ".join(info["anchors"]["host"]) + " | guest " + ", ".join(info["anchors"]["guest"])
        + f" | r0 {10 * restraint.r0:.2f} A, ligand RMSD {10 * rmsd:.2f} A"
        + (f"; {info['candidates_refused']} candidate(s) ranked before it were refused"
           if info["candidates_refused"] else ""))
    return restraint, info, md_seconds


def restraint_from_the_bilayer(cfg: Config, job: "LegJob", leg_dir: Path, prep: dict, eq, stage: dict, rdmol,
                               seed: int, device, log=print):
    """The restraint of a complex leg whose alchemical run is made in water, chosen in the bilayer
    (``membrane.complex_leg: water``): ``(restraint, info, seconds of the MD, seconds of the
    pre-equilibration in the bilayer)``.

    ``prep`` and ``eq`` are the prepared complex in water and its pre-equilibration, from whose end
    the alchemical run starts; ``stage`` is the record of the complex in the bilayer
    (:func:`restraint_system`). That complex is pre-equilibrated as the bilayer system of a leg is
    (:func:`run_pre_equilibration`, to :data:`RESTRAINT_EQUIL`), the trajectories start from there
    and the restraint is chosen in it (:func:`restraint_from_interactions`), every candidate held
    to the rules at the start of the alchemical run too. The anchors are then named as the system
    of the leg numbers them (:func:`solute_correspondence`, :func:`restraint_in_system`), and the
    record keeps them as they were chosen (``md_system``) and the start of the alchemical run
    (``alchemical_start``): :func:`check_restraint_of_leg` holds the restraint to both
    pre-equilibrations. The caller writes ``restraint.json``.
    """
    from cufep.sampling.restraint_md import start_digest

    bilayer = load_prepared(Path(job.lig_dir) / "prep", stage["name"])
    atom_map = solute_correspondence(bilayer, prep)
    equil_file = leg_dir / RESTRAINT_EQUIL
    in_membrane = stage.get("barostat") == "membrane"
    seconds = 0.0
    if not equil_file.exists():
        t0 = time.perf_counter()
        run_pre_equilibration(cfg, "complex", bilayer, None, stage["barostat"], stage.get("membrane_info") or {},
                              seed, equil_file, device, log,
                              what="pre-equilibration of the complex in the bilayer, for the restraint,")
        seconds = time.perf_counter() - t0
    eq_bilayer = np.load(equil_file)
    check_equil_file(eq_bilayer, equil_file, cfg, "complex", in_membrane, seed, log)
    # the configuration the alchemical run starts from, as the system of the bilayer numbers its atoms: the
    # solute where the complex in water has it (the anchors are atoms of the solute; no other row is read)
    begin = np.full((len(bilayer["positions"]), 3), np.nan)
    begin[list(atom_map)] = np.asarray(eq["positions"], dtype=float)[list(atom_map.values())]
    restraint, info, md_seconds = restraint_from_interactions(
        cfg, job, leg_dir, bilayer, rdmol, eq_bilayer, in_membrane, seed, len(bilayer["positions"]), log,
        start=(begin, np.asarray(eq["box"], dtype=float)))
    info["md_system"] = {"name": stage["name"], "host_atoms": [int(i) for i in restraint.host_atoms],
                         "guest_atoms": [int(i) for i in restraint.guest_atoms]}
    info["alchemical_start"] = start_digest(eq["positions"], eq["velocities"], eq["box"])
    log("the restraint was chosen from MD of the complex in the bilayer; the alchemical run is made on the complex "
        "in water (membrane.complex_leg: water)")
    return restraint_in_system(restraint, atom_map), info, md_seconds, seconds


def interactions_csv(report: dict) -> str:
    """The interactions of ``interactions.json`` as a table: one line per ligand atom, residue and type."""
    trajectories = sorted({k for row in report["interactions"] for k in row.get("frames_of_trajectory", {})})
    lines = [",".join(["ligand_atom", "residue", "interaction", "frames", "fraction", "kept",
                       *[f"frames_of_trajectory_{k}" for k in trajectories]])]
    for row in report["interactions"]:
        lines.append(",".join([row["ligand_atom_name"], row["residue_name"], row["interaction"], str(row["frames"]),
                               f"{row['fraction']:.4f}", "yes" if row["kept"] else "no",
                               *[str(row.get("frames_of_trajectory", {}).get(k, 0)) for k in trajectories]]))
    return "\n".join(lines) + "\n"


def _check_box(modeller, cutoff_nm: float, what: str) -> None:
    """WorkflowError if no backend could run the system that ``modeller`` holds with this cutoff
    (:func:`cufep.prep.solvate.box_height_problem`), before it is written."""
    import openmm.unit as unit

    from cufep.prep.solvate import box_height_problem

    box = [[float(v) for v in vec.value_in_unit(unit.nanometer)] for vec in modeller.topology.getPeriodicBoxVectors()]
    problem = box_height_problem(box, cutoff_nm)
    if problem:
        raise WorkflowError(f"{what}: {problem}. Give a larger padding or a smaller dynamics.cutoff_nm")


def prepare_ligand(cfg: Config, ligand, protein, lig_dir: Path, membrane: bool, box_shape: str, log=print) -> dict:
    """Build and save the solvated complex and the solvated ligand for one ligand."""
    from cufep.prep.forcefield import ForceFieldSpec, SystemSpec, create_system, make_forcefield
    from cufep.prep.ligand import assign_partial_charges
    from cufep.prep.solvate import SolvationSpec, combine, ligand_atom_indices, min_box_height_nm, solvate

    prep_dir = lig_dir / "prep"
    done = prep_dir / "prep.json"
    made_from = prep_made_from(cfg, ligand, membrane, box_shape)
    info = check_prepared(cfg, ligand, lig_dir, membrane, box_shape, log)
    if info is not None:
        return info
    prep_dir.mkdir(parents=True, exist_ok=True)
    check_ligand(ligand)
    warnings: list[str] = []
    t0 = time.perf_counter()
    assign_partial_charges(ligand, cfg.forcefield.ligand_charges, cache_dir=cfg.output.dir / "cache")
    log(f"{ligand.name}: partial charges ({cfg.forcefield.ligand_charges}) in {time.perf_counter() - t0:.0f} s")
    ffspec = ForceFieldSpec(protein=cfg.forcefield.protein, water=cfg.forcefield.water,
                            lipid=cfg.forcefield.lipid, ligand=cfg.forcefield.ligand)
    forcefield = make_forcefield(ffspec, [ligand], include_lipid=membrane)
    sysspec = SystemSpec(cutoff_nm=cfg.dynamics.cutoff_nm, hydrogen_mass_amu=cfg.dynamics.hmr_mass_amu,
                         constraints="hbonds" if cfg.dynamics.constraints in ("auto", "hbonds") else "none")
    # Every box height is at least 3 x cutoff for the cuPotential plugin and 2 x cutoff for stock
    # OpenMM, also after the box has relaxed under NPT: an 8 % margin over either rule
    plugin = cfg.backend.name == "cupotential"
    min_edge = min_box_height_nm(cfg.dynamics.cutoff_nm, cfg.backend.name)
    # For the small ligand-only box that rule decides the size, and a cube is then the
    # smallest valid box (a dodecahedron's shortest height is only 71 % of its edge).
    solvent_shape = "cube" if plugin else box_shape
    ionic = cfg.thermodynamics.ionic_strength_M

    # ---- complex ------------------------------------------------------------------------
    def complex_in(bilayer: bool) -> dict:
        """The complex of the pose that was given, in a bilayer or in water: the Modeller, the System,
        the ligand atoms, the barostat it runs under and the numbers of its bilayer."""
        modeller = combine((protein.topology, protein.positions), ligand)
        n_removed = _remove_clashing_waters(modeller, ligand.positions_nm()[ligand.heavy_atom_indices()])
        notes = [f"{n_removed} water(s) overlapping the ligand were removed"] if n_removed else []
        numbers: dict = {}
        if bilayer:
            from dataclasses import replace

            from cufep.prep.membrane import add_membrane

            # the water of the bilayer system: forcefield.water, or membrane.water where the bilayer is the
            # system of the MD of the restraint alone. A bilayer is built with a 3-site water whatever the
            # model: add_membrane asks for that force field when the model has more sites
            water = cfg.bilayer_water
            field = forcefield if water == cfg.forcefield.water else make_forcefield(
                replace(ffspec, water=water), [ligand], include_lipid=True)
            numbers = add_membrane(
                modeller, field, cfg, log, water=water,
                forcefield_with_water=lambda file: make_forcefield(replace(ffspec, water=file), [ligand],
                                                                   include_lipid=True))
            notes += [f"membrane: {w}" for w in numbers.pop("warnings", [])]
        else:
            # a complex in water is built without the lipid file, as a run without a bilayer builds it
            field = forcefield if not membrane else make_forcefield(ffspec, [ligand], include_lipid=False)
            solvate(modeller, field, SolvationSpec(padding_nm=cfg.solvation.padding_complex_nm, box_shape=box_shape,
                                                   water=cfg.forcefield.water,
                                                   ionic_strength_M=ionic, min_edge_nm=min_edge))
        _check_box(modeller, cfg.dynamics.cutoff_nm, f"{ligand.name}: the solvated complex")
        return {"modeller": modeller, "system": create_system(field, modeller.topology, sysspec),
                "ligand": ligand_atom_indices(modeller.topology), "barostat": "membrane" if bilayer else "isotropic",
                "membrane_info": numbers, "notes": notes}

    # With membrane.complex_leg: water the bilayer is the system of the MD from which the restraint is chosen
    # and the alchemical run is made on the complex in water: two systems of the same pose. The bilayer is
    # built first (its orientation can stop the run), and nothing is written unless the two are of the same
    # protein and ligand, atom by atom
    in_leg = membrane and cfg.membrane.complex_leg == "bilayer"
    for_restraint = complex_in(True) if membrane and not in_leg else None
    built = complex_in(in_leg)
    restraint_record = None
    if for_restraint is not None:
        solute_correspondence(_solute_of(for_restraint["modeller"].topology, for_restraint["ligand"]),
                              _solute_of(built["modeller"].topology, built["ligand"]))
        _write_system(prep_dir, BILAYER_COMPLEX, for_restraint["modeller"], for_restraint["system"],
                      for_restraint["ligand"])
        restraint_record = {"name": BILAYER_COMPLEX, "barostat": for_restraint["barostat"],
                            "n_atoms": for_restraint["system"].getNumParticles(),
                            "membrane_info": for_restraint["membrane_info"]}
        warnings += [note for note in for_restraint["notes"] if note.startswith("membrane: ")]
    warnings += built["notes"]
    barostat, membrane_info = built["barostat"], built["membrane_info"]
    _write_system(prep_dir, "complex", built["modeller"], built["system"], built["ligand"])
    n_complex = built["system"].getNumParticles()

    # ---- ligand in solvent -----------------------------------------------------------------
    modeller = combine(None, ligand)
    solvate(modeller, forcefield, SolvationSpec(padding_nm=cfg.solvation.padding_solvent_nm, box_shape=solvent_shape,
                                                water=cfg.forcefield.water,
                                                ionic_strength_M=ionic, min_edge_nm=min_edge))
    _check_box(modeller, cfg.dynamics.cutoff_nm, f"{ligand.name}: the solvated ligand")
    system = create_system(forcefield, modeller.topology, sysspec)
    _write_system(prep_dir, "solvent", modeller, system, ligand_atom_indices(modeller.topology))
    publish_text(prep_dir / "ligand.json", ligand.offmol.to_json())
    info = {"ligand": ligand.name, "n_atoms_complex": n_complex, "n_atoms_solvent": system.getNumParticles(),
            "complex_barostat": barostat, "membrane": membrane, "box_shape": box_shape,
            "solvent_box_shape": solvent_shape, "membrane_info": membrane_info, "warnings": warnings,
            # the cuPotential plugin needs a larger box; a system built for it also runs on stock OpenMM
            "box_built_for_backend": cfg.backend.name, "made_from": made_from,
            # the chain of the input PDB of every chain of the prepared protein
            "input_chains": dict(protein.input_chains)}
    if restraint_record is not None:
        # the system of the MD from which the restraint is chosen, where it is not the one of the alchemical run
        info["restraint_system"] = restraint_record
    publish_text(done, json.dumps(info, indent=1))       # last: the mark that the preparation is complete
    log(f"{ligand.name}: complex {n_complex} atoms, solvent leg {system.getNumParticles()} atoms"
        + (f"; complex in the bilayer, for the MD of the restraint, {restraint_record['n_atoms']} atoms"
           if restraint_record is not None else "")
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

    def request_devices(self, replicas: int, work: float, timeout: float = 300.0, pinned: bool = False,
                        what: str | None = None) -> list | None:
        """Tell the scheduler that sampling starts; returns the GPU of every worker to use. With
        ``pinned`` the job keeps these GPUs until it asks again or goes back to serial work: it cannot
        move its workers (the precalculation of adaptive lambda scheduling, the trajectories for a
        restraint), so the scheduler must not count them as moved (review of PR 1). ``work`` is what
        the workers still have to do (:func:`md_work`); ``what`` is what the log of the scheduler
        calls it, where it is not the replica exchange of ``replicas`` lambda windows."""
        self._send({"event": "sampling", "replicas": int(replicas), "work": float(work), "pinned": bool(pinned),
                    "what": what})
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
        """The part that used several workers is over: the job needs a single worker at most from now on."""
        self._send({"event": "serial"})


@dataclass(kw_only=True)
class LegJob(Job):
    """A leg of a ligand of the ABFE: ``lig_dir`` is the directory of the ligand."""

    ligand: str
    lig_dir: str

    unit_key: ClassVar[str] = "ligand"

    @property
    def name(self) -> str:
        return self.ligand

    @property
    def unit_dir(self) -> Path:
        return Path(self.lig_dir)

    def runner(self):
        return run_leg_safe


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


def leg_with_samples_stays_on_its_backend(exc: BaseException) -> str:
    """What a leg says whose cuPotential run failed and which had samples from an earlier run
    (:func:`backend_fallback_refusal`): it is not repeated on stock OpenMM from its start. What
    keeps it on cuPotential is its samples, which were made there (a leg with samples goes on with
    the backend they were made with). Its pre-equilibration can have run on stock OpenMM alone: that
    of a soluble complex whose restraint is chosen from interactions ends with the NPT that holds
    the solute (the sixth pass of Codex: the message said the pre-equilibration was made on
    cuPotential, where ``equil.npz`` of the TYK2 example names stock OpenMM)."""
    return (f"the cuPotential run of this leg failed ({str(exc)[:300]}). The leg had samples from an earlier "
            "run, so it is not repeated on stock OpenMM from its start: its samples and checkpoint are left "
            "in hrex/. Run the leg again to continue it on cuPotential, or run the calculation on stock "
            "OpenMM (backend.name: openmm) in a new output.dir: the samples of this leg were made on "
            "cuPotential, and samples of the two backends are never mixed")


#: the prepared complex in the bilayer of a ligand whose complex leg has its alchemical run in water
#: (``membrane.complex_leg: water``): the system of the MD from which the restraint is chosen
BILAYER_COMPLEX = "complex_bilayer"
#: the pre-equilibration of that system in the directory of the leg: where the MD for the restraint starts
RESTRAINT_EQUIL = "restraint_equil.npz"
#: the files of a leg that its samples were made from and that a new set of lambda windows uses again
LEG_INPUT_FILES = ("equil.npz", "restraint.json", RESTRAINT_EQUIL)


def restraint_system(prep_info: dict) -> dict | None:
    """The record of the prepared system in which the restraint of a complex leg is chosen where
    that is another one than the system of its alchemical run (``membrane.complex_leg: water``:
    name of the prepared system, barostat, atoms, numbers of the bilayer); None where the two are
    one system, as in every preparation without that setting."""
    record = prep_info.get("restraint_system")
    return record if isinstance(record, dict) and record.get("name") else None


def check_restraint_equil(cfg: Config, leg_dir: Path, prep_info: dict, seed: int, log=print) -> None:
    """Refuse the pre-equilibration of the system in which the restraint of a complex leg is chosen
    (:data:`RESTRAINT_EQUIL`, where that is another system than the one of the alchemical run) if
    this run cannot use it: made with other settings, or holding less MD than asked
    (:func:`check_equil_file`). One rule for the check before any job and for the leg, whether the
    leg has its restraint or not: a leg with a restraint used the file unasked when its prepared
    systems were made again, so that no check before the jobs was made (review of the first
    version of this setting: a restraint from a bilayer equilibrated for another
    ``membrane.equilibration_ns`` than the run records). Nothing to ask where there is no such
    system or no such file yet."""
    stage = restraint_system(prep_info)
    equil = Path(leg_dir) / RESTRAINT_EQUIL
    if stage is not None and equil.exists():
        check_equil_file(np.load(equil), equil, cfg, "complex", stage.get("barostat") == "membrane", seed, log)


def check_restraint_of_leg(stored: dict, rfile: Path, sample_dir: Path) -> None:
    """Refuse a restraint chosen from interactions that is not the one of the leg whose files are in
    ``sample_dir`` (a leg directory, or a ``previous_<n>/`` in it): one rule for the run of a leg,
    for the check before any job and for every reading of its samples (review of 877ab88: ``cufep
    analyze`` gave the release of the restraint of another repeat).

    The record of the restraint holds the configuration its MD started from (``start``,
    :func:`check_restraint_start`): the pre-equilibration of the leg, or, where the MD was made in
    another system than the alchemical run (``md_system``), the pre-equilibration of that system
    (:data:`RESTRAINT_EQUIL`). In that case it also holds the configuration the alchemical run
    starts from (``alchemical_start``), to which the anchors were held. A file that is needed for
    the comparison and is not there is a refusal too: a pre-equilibration that is made again is
    another start. A restraint of another selection has no such record."""
    from cufep.sampling.restraint_md import start_digest

    info = stored.get("info") or {}
    files = leg_files(sample_dir)

    def needed(name: str):
        if files[name] is None:
            raise WorkflowError(
                f"the restraint in {rfile} was chosen from MD that started from the pre-equilibration of its leg, "
                f"and {name} is not there: that the restraint is the one of this leg cannot be checked (a "
                "pre-equilibration that is made again is another start). Nothing was changed; run the calculation "
                "in a new output.dir")
        return np.load(files[name])

    if info.get("start") is not None:
        name = RESTRAINT_EQUIL if info.get("md_system") else "equil.npz"
        check_restraint_start(stored, rfile, needed(name), name)
    if info.get("alchemical_start") is not None:
        eq = needed("equil.npz")
        if info["alchemical_start"] != start_digest(eq["positions"], eq["velocities"], eq["box"]):
            raise WorkflowError(
                f"the restraint in {rfile} cannot be used again: its anchors were held to the rules at another "
                "configuration than the pre-equilibration of this leg ends in (equil.npz), from which the "
                "alchemical run starts. Nothing was changed. A calculation with other input or settings needs a "
                "new output.dir.")


def leg_files(sample_dir: Path) -> dict[str, Path | None]:
    """The pre-equilibration and the restraint (:data:`LEG_INPUT_FILES`) that belong to the samples
    of ``sample_dir``: a leg directory, or a ``previous_<n>/`` in it. Such an archive holds copies
    of them since this version; one of an earlier version has them in the leg directory above it,
    which the leg used again. None for a file that is not there."""
    out = {}
    for name in LEG_INPUT_FILES:
        candidates = [sample_dir / name] + (
            [sample_dir.parent / name] if sample_dir.name.startswith("previous_") else []
        )
        out[name] = next((f for f in candidates if f.exists()), None)
    return out


def archive_leg(leg_dir: Path) -> Path | None:
    """Move the lambda windows, samples and result of a leg into ``previous_<n>/`` in its directory.

    The leg then starts anew from its pre-equilibration and restraint, which stay where they
    are. Nothing is deleted. Returns the new directory, or None if the leg had none of these.
    """

    present = [leg_dir / name for name in LEG_RUN_FILES if (leg_dir / name).exists()]
    present += sorted(path for path in leg_dir.glob(f"{FAILED_BACKEND_DIR}_*") if path.is_dir())
    # an archive that an interrupted call began is completed, not begun a second time
    staging = sorted(leg_dir.glob(".previous_*.partial"))
    if not present and not staging:
        return None
    if staging:
        stage = staging[0]
        n = int(stage.name.split("_")[1].split(".")[0])
    else:
        n = 1
        while (leg_dir / f"previous_{n}").exists():
            n += 1
        stage = leg_dir / f".previous_{n}.partial"
        stage.mkdir()
    dest = leg_dir / f"previous_{n}"
    # the files these samples were made from stay with the leg, which uses them again; the archive
    # gets copies (leg_files), each written whole under another name first
    for name in LEG_INPUT_FILES:
        if (leg_dir / name).exists() and not (stage / name).exists():
            publish(stage / name, lambda tmp, source=leg_dir / name: shutil.copy2(source, tmp))
    for f in present:
        f.rename(stage / f.name)
    # a leg that is being run again has had its result set aside by begin_leg: in the archive
    # it is the result of those windows and samples again
    if (stage / "result.previous.json").exists() and not (stage / "result.json").exists():
        (stage / "result.previous.json").rename(stage / "result.json")
    stage.rename(dest)                                  # the archive appears complete, in one step
    return dest


#: written when a leg starts and removed when it has written its result or its error: a leg that
#: holds only this file was interrupted (the process was killed, the machine stopped, Ctrl-C)
STARTED_FILE = "started.json"


def _write_json(path: Path, data: dict) -> None:
    """Write a file in one step, so that a process that dies while writing leaves no half of it."""
    publish_text(path, jsonio.dumps(data, indent=1))


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _leg_identity(leg_dir: Path, job=None) -> dict:
    """What a leg is of, its leg and its repeat: those of its job (:meth:`cufep.engine.jobs.Job.identity`), else
    those of a leg directory ``<ligand>/repeat_<n>/<leg>`` of the ABFE."""
    if job is not None:
        return job.identity()
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
    return {**job.identity(), "error": "the job process ended without writing a result (killed?)"}


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


def plain_md_settings(cfg: Config, timestep_fs: float, barostat: str, seed: int, **parts):
    """The settings of a plain MD of the fully interacting system as the configuration gives them:
    temperature and pressure (``thermodynamics``), friction and the frequency of the barostat
    (``dynamics``), the backend, its fallback, platform and precision (``backend``), the threads of a
    Context of the CPU platform (``resources.cpu_threads_per_worker``). ``timestep_fs`` is the
    time step to begin with, ``barostat`` that of the leg (isotropic | membrane); ``parts`` are the
    lengths and the atoms that only one kind of MD has. The pre-equilibration and the MD from which
    a restraint is chosen take their settings here, so that neither can differ from the YAML.

    The precision is the one of the YAML, single too (the owner's decision of 2026-11-21). Earlier
    versions ran plain MD in mixed precision where single was asked, a rule of their own without a
    source that the log did not say (:func:`equil_made_with` tells their files apart)."""
    from cufep.sampling.equilibrate import EquilibrationSettings

    return EquilibrationSettings(
        temperature_K=cfg.thermodynamics.temperature_K, pressure_bar=cfg.thermodynamics.pressure_bar,
        friction_per_ps=cfg.dynamics.friction_per_ps, timestep_fs=timestep_fs, barostat=barostat,
        barostat_frequency=cfg.dynamics.barostat_frequency, platform=cfg.backend.platform,
        precision=cfg.backend.precision, seed=seed,
        md_backend=cfg.backend.name, fallback=cfg.backend.fallback,
        cpu_threads=int(cfg.resources.cpu_threads_per_worker), **parts)


def md_work(ps: float, n_atoms: int, iteration_ps: float) -> float:
    """The work of ``ps`` of MD of a system of ``n_atoms`` on one worker, as a job tells it to the
    scheduler: a guess of its seconds, the one replica exchange starts from before its iterations
    (of ``iteration_ps`` each) are timed. Every part of a leg that asks for GPUs tells its work in
    it, so that the shares of the jobs are in proportion to the MD they still have to run (review
    of f5a77ce: the trajectories for a restraint and the precalculation of the lambda schedule told
    picoseconds, replica exchange seconds). Only ratios matter, and no result depends on it."""
    return float(ps) / float(iteration_ps) * 0.05 * (1.0 + int(n_atoms) / 50000.0)


def _precision(cfg: Config, backend: str) -> str:
    from cufep.backends.base import backend_precision

    return backend_precision(backend, cfg.backend.precision)


def precision_that_ran(cfg: Config, backend: str) -> str:
    """The precision a plain MD of a leg ran in on ``backend``, as ``equil.npz`` records it: the one of
    :func:`_precision` where the platform takes a precision (CUDA, OpenCL). The Reference and the CPU
    platform have their own whatever is asked, and the record says that instead (the fourth pass of
    Codex: a run on the Reference platform, which is double, was recorded as mixed)."""
    if cfg.backend.platform in ("CUDA", "OpenCL"):
        return _precision(cfg, backend)
    return f"that of the {cfg.backend.platform} platform"


def run_pre_equilibration(cfg: Config, leg: str, prep: dict, system, barostat: str, membrane_info: dict, seed: int,
                          equil_file: Path, device, log=print, what: str = "pre-equilibration") -> None:
    """Make the pre-equilibration of a prepared system (:func:`load_prepared`) and write it to
    ``equil_file``: plain MD of the fully interacting system, by the settings of ``leg``
    (:func:`pre_equilibration_parts`, :func:`equil_made_with`).

    One function for the system of a leg and for the complex in the bilayer in which the restraint
    of a leg in water is chosen (``membrane.complex_leg: water``): ``barostat`` says which the
    system is (``membrane`` for a bilayer), ``membrane_info`` holds the numbers of its bilayer,
    and ``what`` is how the log names the MD. ``system`` None reads the System of ``prep``."""
    from cufep.sampling.equilibrate import pre_equilibrate
    from cufep.steps import frames_of

    if system is None:
        import openmm as mm

        system = mm.XmlSerializer.deserialize(prep["system_xml"])

    in_membrane = leg == "complex" and barostat == "membrane"
    pre = cfg.sampling.pre_equilibration.complex if leg == "complex" else cfg.sampling.pre_equilibration.solvent
    production, held = pre_equilibration_parts(cfg, leg, in_membrane)
    atoms = list(prep["topology"].atoms())
    heavy_solute = [i for i in prep["solute"] if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    # A freshly built bilayer needs longer: the heavy atoms of protein and ligand are held while the lipids and
    # the water settle (NVT, NPT), then it runs for membrane.equilibration_ns (free, or with the solute still
    # held: pre_equilibration_parts)
    restrained = heavy_solute if held or in_membrane else []
    wall_atoms: list[int] = []
    wall_half = 0.0
    if in_membrane:
        # While the trimmed bilayer closes up, bulk water is kept out of the hydrophobic core.
        plane = float((membrane_info or {}).get("phosphate_plane_nm", 0.0))
        wall_half = max(0.0, plane - 0.6)
        if wall_half > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(prep["positions"][heavy_solute])
            for a_ in atoms:
                if a_.residue.name == "HOH" and a_.element is not None and a_.element.symbol == "O":
                    xyz = prep["positions"][a_.index]
                    if abs(xyz[2]) > wall_half and tree.query(xyz)[0] > 0.6:
                        wall_atoms.append(a_.index)
    settings = plain_md_settings(
        cfg, float(cfg.dynamics.timestep_fs), barostat, seed,
        nvt_ns=pre.nvt_ns, npt_ns=2.0 * pre.npt_ns if in_membrane else pre.npt_ns, production_ns=production,
        hold_solute=held, restrained_atoms=restrained, wall_atoms=wall_atoms, wall_half_width_nm=wall_half,
        wall_centre_nm=0.0,
        ligand_atoms=prep["ligand"] if in_membrane else [],
        water_oxygens=[a.index for a in atoms
                       if a.residue.name == "HOH" and a.element is not None and a.element.symbol == "O"]
        if in_membrane else [])
    t0 = time.perf_counter()
    eq = pre_equilibrate(system, prep["positions"], prep["box"], prep["solute"], settings, device, log)
    # the restraint search uses the last pre_equilibration md_ns of the trajectory
    n_keep = max(2, frames_of(pre.md_ns * 1000.0, settings.frame_interval_ps, eq.timestep_fs))
    # simulated_ns is the MD integrated (simulated_ns_is_integrated): earlier versions recorded the
    # length asked for and integrated whole frame intervals of it (equil_contradictions)
    publish_npz(equil_file, positions=eq.positions, velocities=eq.velocities, box=eq.box,
                frames=eq.frames[-n_keep:], frame_boxes=eq.frame_boxes[-n_keep:],
                frame_atoms=np.array(eq.frame_atoms), timestep_fs=eq.timestep_fs, notes=json.dumps(eq.notes),
                simulated_ns=eq.simulated_ns, simulated_ns_is_integrated=True,
                backend=eq.backend,               # what ran the unrestrained MD (made_with: what was asked)
                backend_precision=precision_that_ran(cfg, eq.backend),
                pocket_waters=eq.pocket_waters, area_nm2=eq.area_nm2,
                made_with=json.dumps(equil_made_with(cfg, leg, in_membrane, seed)),
                membrane_drift=json.dumps(eq.drift),
                # the mean cell of the second half of the MD: the cell a leg without a barostat is fixed in
                # (alchemical_box)
                **({} if eq.mean_box is None else {"mean_box": np.asarray(eq.mean_box, dtype=float)}))
    log(f"{what} done in {time.perf_counter() - t0:.0f} s (time step {eq.timestep_fs:g} fs)")


def sample_leg(spec, hrex_dir: Path, job, configs, n_equil: int, total: int, cfg, log, timing: dict, timed,
               ckpt_stride: int, unrecorded=None) -> dict:
    """The replica exchange of a leg (step 5 of :func:`run_leg`, and of the legs of a relative calculation,
    :mod:`cufep.rbfe.leg`): continue the samples in ``hrex_dir`` or start them from ``configs``, until ``total``
    iterations of which the first ``n_equil`` are the equilibration, on the GPUs the scheduler grants; return
    the samples. ``unrecorded`` is called with the sampler for samples of an earlier version that does not record
    their conditions (they are analysed as they are); without it such samples are refused."""
    from cufep.sampling.hrex import ReplicaExchangeSampler
    from cufep.sampling.storage import read_samples

    n_windows = len(spec.states)
    control = job.control
    # seconds one replica needs per iteration: a guess until the first iterations were timed
    replica_seconds = [md_work(cfg.sampling.time_per_iteration_ps, spec.n_atoms, cfg.sampling.time_per_iteration_ps)]

    def remaining_work(done: int) -> float:
        return n_windows * max(0, total - done) * replica_seconds[0] / max(1, job.oversubscription)

    t_sampling = time.perf_counter()
    # the method of exchange is the one of the spec (leg_spec: sampling.exchange)
    with ReplicaExchangeSampler(spec, hrex_dir, devices=job.devices, inline=job.inline, log=log) as sampler:
        if sampler.unrecorded:
            if unrecorded is None:
                raise WorkflowError(f"the samples in {hrex_dir} do not record their conditions")
            unrecorded(sampler)
            timed("sampling", t_sampling)
            return read_samples(hrex_dir)
        if not sampler.storage.n_iterations - sampler.storage.n_stored and leg_is_complete(hrex_dir, total):
            # the leg has all its iterations: nothing to continue, its samples are analysed as they are
            # (whatever version wrote its checkpoint)
            take_complete_leg(sampler.storage, n_equil, total, cfg, log)
            timed("sampling", t_sampling)
            return read_samples(hrex_dir)
        if not sampler.restore():
            # the samples start from these configurations: their record says how they were made. After the
            # replica exchange accepted the leg (a refusal keeps the start of earlier samples, sixth review of
            # PR #18), after the samples it had were emptied (a failure there keeps them and their start, eighth
            # review), and before the checkpoint of the start is written, from which a retry restores without
            # coming here again (seventh review)
            from cufep.engine.leg import record_the_start_used

            sampler.initialize(*configs, on_start=lambda: record_the_start_used(Path(hrex_dir).parent))
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
    return read_samples(hrex_dir)


def run_leg(job: LegJob) -> dict:
    """Run (or resume, or extend) one leg and return its result dictionary."""
    import openmm as mm

    from cufep.alchemy.restraints import (
        BoreschRestraint,
        RestraintSearchSettings,
        find_boresch_restraint,
        ligand_symmetry_number,
        with_force_constants_from_fluctuations,
    )
    from cufep.alchemy.schedule import Schedule, schedule_from_windows
    from cufep.engine.leg import (
        adaptive_windows,
        check_kept_windows,
        chosen_backend,
        finish_leg,
        kept_start_record,
        leg_context,
        open_leg,
        pre_equilibrated,
        refuse_samples_without_windows,
        with_fallback,
    )
    from cufep.mps import thread_percentage
    from cufep.sampling.states import STAGES_COMPLEX, STAGES_SOLVENT, stage_state
    from cufep.sampling.storage import stored_conditions
    from cufep.sampling.worker import LegSpec

    cfg = Config.model_validate(job.config)
    run = open_leg(job, cfg)
    leg_dir, log, seed, notes, timing, timed = run.leg_dir, run.log, run.seed, run.notes, run.timing, run.timed
    is_complex = job.leg == "complex"
    lig_dir = Path(job.lig_dir)
    prep_info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    prep = load_prepared(lig_dir / "prep", job.leg)
    if is_complex:
        check_manual_anchors(cfg, lig_dir)
    system, gcmc = leg_system(cfg, prep, job.leg)
    if gcmc is not None:
        # every MD of this leg is of the system with the ghosts of the start switched off
        prep = {**prep, "system_xml": mm.XmlSerializer.serialize(system)}
        log(f"grand canonical moves of water: sphere of {10 * gcmc.radius_nm:.1f} A around the ligand, Adams value "
            f"B = {gcmc.adams:.3f} (excess chemical potential {gcmc.excess_chemical_potential_kj_per_mol / KCAL:.3f} "
            f"kcal/mol, standard volume {1000 * gcmc.standard_volume_nm3:.3f} A^3), {len(gcmc.ghosts)} of the "
            f"{len(gcmc.waters)} waters are ghosts at the start (those farthest from the ligand); the alchemical "
            "run has no barostat")
    T = cfg.thermodynamics.temperature_K
    device0 = run.device0
    barostat = prep_info["complex_barostat"] if is_complex else "isotropic"

    # ---- 1. pre-equilibration: plain MD of the fully interacting system -------------------
    equil_file = leg_dir / "equil.npz"
    in_membrane = is_complex and barostat == "membrane"
    eq, equil_recorded = pre_equilibrated(run, prep, system, barostat, prep_info.get("membrane_info", {}))
    # files of this leg that an earlier version wrote without a record of what they were made from
    origin_not_recorded: list[str] = [] if prep_info.get("made_from") is not None else ["the prepared systems"]
    if not equil_recorded:
        origin_not_recorded.append("the pre-equilibration")
    elif "platform" not in json.loads(str(eq["made_with"])):
        origin_not_recorded.append("the platform of the pre-equilibration")
    timestep_fs = float(eq["timestep_fs"])
    # the notes of the pre-equilibration and of the restraint are made from their files when the
    # result is made (notes_of_leg_files), with the levels of the run that makes it

    # ---- 2. restraint (complex only) --------------------------------------------------------
    t_restraint = time.perf_counter()
    md_seconds = 0.0                       # of the MD from which the restraint is chosen: a phase of its own
    restraint = None
    symmetry = 1
    if is_complex:
        from openff.toolkit import Molecule
        rdmol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
        symmetry = ligand_symmetry_number(rdmol)
        rfile = leg_dir / "restraint.json"
        restraint_from = restraint_made_with(cfg)
        # the pre-equilibration of another system in which the restraint is chosen, if the leg has one
        check_restraint_equil(cfg, leg_dir, prep_info, seed, log)
        if rfile.exists():
            stored = json.loads(rfile.read_text())
            if not check_restraint_file(stored, rfile, cfg, log):
                origin_not_recorded.append("the restraint")
            check_restraint_of_leg(stored, rfile, leg_dir)
            restraint = BoreschRestraint.from_dict(stored["restraint"])
            problems = restraint_anchor_problems(restraint.host_atoms, restraint.guest_atoms, prep)
            if problems:
                raise WorkflowError(f"the restraint in {rfile} cannot be used: " + "; ".join(problems))
        else:
            rc = cfg.restraints
            k_r, k_a = rc.k_r_kcal_per_A2 * KCAL * 100.0, rc.k_angle_kcal_per_rad2 * KCAL
            if rc.selection == "manual":
                if not (rc.host_atoms and rc.guest_atoms):
                    raise WorkflowError("restraints.selection is 'manual' but host_atoms/guest_atoms are not given")
                restraint, fluctuations = manual_boresch_restraint(
                    rc.host_atoms, rc.guest_atoms, prep["ligand"], eq, system.getNumParticles(), k_r, k_a, equil_file,
                    force_constants=rc.force_constants, temperature_K=T, input_pose=prep["positions"],
                    input_box=prep["box"])
                # how far the ligand moved from the pose it was given, measured as for anchors the search
                # chooses (review of PR 1: with manual anchors a ligand that left the site gave no warning)
                from cufep.alchemy.restraints import ligand_pose_rmsd

                full = np.zeros((eq["frames"].shape[0], system.getNumParticles(), 3))
                full[:, eq["frame_atoms"], :] = eq["frames"]
                rmsd = ligand_pose_rmsd(prep["topology"], full, prep["ligand"], rdmol, prep["positions"],
                                        frame_boxes_nm=eq["frame_boxes"], input_box_nm=prep["box"],
                                        host_max_distance_nm=rc.host_max_distance_nm)
                info = {"selection": "manual", "force_constants": rc.force_constants, "fluctuations": fluctuations,
                        "flat_angle_rule": FLAT_ANGLE_RULE,
                        "ligand_rmsd_A": 10 * rmsd, "ligand_rmsd_reference": "input pose", "notes": []}
                log(f"restraint (manual anchors): r0 {10 * restraint.r0:.2f} A, ligand RMSD {10 * rmsd:.2f} A")
            elif rc.selection == "interactions":
                stage = restraint_system(prep_info)
                if stage is None:
                    restraint, info, md_seconds = restraint_from_interactions(
                        cfg, job, leg_dir, prep, rdmol, eq, in_membrane, seed, system.getNumParticles(), log)
                else:
                    # the MD for the restraint is made on another prepared system than the alchemical run (the
                    # complex in the bilayer, membrane.complex_leg: water), which is pre-equilibrated first
                    restraint, info, md_seconds, equil_seconds = restraint_from_the_bilayer(
                        cfg, job, leg_dir, prep, eq, stage, rdmol, seed, device0, log)
                    timing["phases"]["restraint_equilibration"] = \
                        timing["phases"].get("restraint_equilibration", 0.0) + equil_seconds
                device0 = job.devices[0] if job.devices else None       # the GPU the job goes on with
                timing["phases"]["restraint_md"] = timing["phases"].get("restraint_md", 0.0) + md_seconds
                if stage is not None:
                    md_seconds += equil_seconds          # the phase "restraint" is the choice without any MD
            elif rc.selection == "auto":
                full = np.zeros((eq["frames"].shape[0], system.getNumParticles(), 3))
                full[:, eq["frame_atoms"], :] = eq["frames"]
                settings = RestraintSearchSettings(k_r=k_r, k_angle=k_a, rmsf_cutoff_nm=rc.rmsf_cutoff_nm,
                                                   host_min_distance_nm=rc.host_min_distance_nm,
                                                   host_max_distance_nm=rc.host_max_distance_nm,
                                                   dssp_filter=rc.dssp_filter,
                                                   constants_from_fluctuations=rc.force_constants == "auto")
                # the pose is compared with the prepared input: the frames kept for the search begin
                # only after the first part of the pre-equilibration
                found = find_boresch_restraint(
                    prep["topology"],
                    full,
                    prep["ligand"],
                    rdmol,
                    T,
                    settings,
                    input_pose_nm=prep["positions"],
                    frame_boxes_nm=eq["frame_boxes"],
                    input_box_nm=prep["box"],
                    start_nm=eq["positions"],
                    start_box_nm=eq["box"],
                )
                restraint = found.restraint
                if rc.force_constants == "auto":
                    restraint = with_force_constants_from_fluctuations(restraint, found.fluctuations, T)
                info = {"selection": "auto", "force_constants": rc.force_constants, "score": found.score,
                        "flat_angle_rule": FLAT_ANGLE_RULE,
                        "fluctuations": found.fluctuations,
                        "ligand_rmsd_A": 10 * found.ligand_rmsd_nm, "ligand_rmsd_reference": "input pose",
                        "candidates_tested": found.n_candidates, "notes": found.notes}
                atoms = list(prep["topology"].atoms())
                label = lambda i: f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}"   # noqa: E731
                log("restraint: host " + ", ".join(label(i) for i in restraint.host_atoms) + " | guest "
                    + ", ".join(label(i) for i in restraint.guest_atoms)
                    + f" | r0 {10 * restraint.r0:.2f} A, ligand RMSD {10 * found.ligand_rmsd_nm:.2f} A")
            else:
                raise WorkflowError(f"restraints.selection: {rc.selection} is not a selection this version knows")
            publish_text(rfile, json.dumps({"restraint": restraint.to_dict(), "info": info,
                                            "ligand_symmetry_number": symmetry, "made_with": restraint_from}, indent=1))
        if symmetry > 1:
            notes.append(f"the ligand has {symmetry} heavy-atom graph symmetries; no symmetry correction is applied")

    # the restraint without the MD it was chosen from, which is the phase restraint_md
    timing["phases"]["restraint"] = timing["phases"].get("restraint", 0.0) \
        + (time.perf_counter() - t_restraint) - md_seconds

    # ---- 3. alchemical System ------------------------------------------------------------------
    built = alchemical_leg(cfg, system, prep, alchemical_box(eq, gcmc), restraint, is_complex, gcmc)
    alch, alch_xml = built.alch, built.xml
    rest2 = cfg.sampling.rest2
    rest2_on, hot_atoms, hot_residues = built.rest2_on, built.hot_atoms, built.hot_residues
    if rest2_on:
        log(
            f"REST2: hot region = ligand ({len(prep['ligand'])} atoms)"
            + (
                f" and {len(hot_atoms)} atoms of {len(hot_residues)} protein residues ({' '.join(hot_residues)})"
                if hot_atoms
                else ""
            )
            + f"; {alch.rest2_torsions} torsion terms scaled; "
            f"{rest2.scale:g} times the temperature in the middle of the leg"
        )
    # samples of two backends are never mixed: a leg that has samples goes on with their backend
    backend_name = chosen_backend(run, alch.system)
    stages = list(STAGES_COMPLEX if is_complex else STAGES_SOLVENT)
    pressure = cfg.thermodynamics.pressure_bar
    iteration_ps = cfg.sampling.time_per_iteration_ps
    plan = planned_iterations(iteration_ps, timestep_fs, cfg.sampling.equilibration(job.leg), job.production_ns)
    steps_per_iteration, n_equil, n_prod = plan.steps_per_iteration, plan.n_equilibration, plan.n_production
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

        def check_scale(schedule) -> None:
            if abs(schedule.rest2_scale - (rest2.scale if rest2_on else 1.0)) > 1e-9:
                raise WorkflowError(
                    f"the lambda schedule of this leg was made with a REST2 scale of {schedule.rest2_scale:g} but "
                    f"sampling.rest2.scale is {rest2.scale:g}; use the same value, or remove {sched_file} to make a "
                    "new one")

        # ---- 4. lambda schedule ----------------------------------------------------------------
        ls = cfg.lambda_schedule
        found_in_leg = sched_file.exists()             # windows the leg had before this run
        given = None                                   # windows the leg is given by a file, if it has none
        if ls.file is not None and not found_in_leg:
            src = given_schedule_file(cfg, ligand_dir_name(job.ligand), job.leg, job.repeat)
            if not src.exists():
                raise WorkflowError(f"lambda_schedule.file is set but there is no schedule at {src}")
            given = given_schedule(src, job.leg, stages, rest2.scale if rest2_on else 1.0)
        stored = stored_windows(leg_dir, job.leg, stages, windows_settings(cfg), log)
        if stored is not None:
            schedule, configs, made = stored
            # windows without a record of their settings and without samples are sampled by this run
            # as they are: they get its record, which says that their origin is not known. With
            # samples the leg goes on in the states its samples were made in, under the settings
            # they record, which this run was held to (check_leg_reuse)
            if "settings" not in made and not stored_conditions(hrex_dir)[0]:
                made = adopt_windows(leg_dir, schedule, configs, cfg, made)
            windows_origin = made.get("origin_not_recorded")
            # said every time: a change of lambda_schedule in the YAML does not reach a leg that has its windows
            log(f"lambda schedule ({schedule.mode}): the {len(schedule.states)} windows this leg already has "
                "(schedule.yaml); the settings of lambda_schedule are not applied to them, "
                "cufep run --reschedule determines the windows again")
        else:
            # what a Context of this leg is made with, for the workers of the precalculation and for
            # the Context of this process, which walks through the windows and runs them itself
            # when the leg has no workers for that
            pool_spec = LegSpec(
                name=job.tag,
                system_xml=alch_xml,
                states=[stage_state(stages[0], 0.0).to_dict(), stage_state(stages[-1], 1.0).to_dict()],
                n_atoms=alch.system.getNumParticles(),
                backend=backend_name,
                precision=_precision(cfg, backend_name),
                platform=cfg.backend.platform,
                temperature_K=T,
                pressure_bar=None if gcmc is not None else pressure,
                barostat="none" if gcmc is not None else barostat,
                barostat_frequency=cfg.dynamics.barostat_frequency,
                friction_per_ps=cfg.dynamics.friction_per_ps,
                timestep_fs=timestep_fs,
                steps_per_iteration=steps_per_iteration,
                seed=seed + 23,
                cpu_threads=cfg.resources.cpu_threads_per_worker,
                box_safety=1.3 if barostat == "membrane" and gcmc is None else None,
                mps_thread_percentage=thread_percentage(job.oversubscription),
                relocation=built.relocation,
                gcmc=built.gcmc,
            )
            # (with grand canonical moves the cell is fixed, alchemical_box, and the spec has no barostat)
            start_box = alchemical_box(eq, gcmc)
            ctx, integ = leg_context(
                pool_spec, alch.system, start_box,
                eq["positions"] if gcmc is None else
                (lambda c: positions_in_cell(eq["positions"], eq["box"], start_box, c.getMolecules())),
                eq["velocities"], device0, seed)
            t0 = time.perf_counter()
            # a replica exchange that goes on from samples (or the checkpoint of their start) keeps their windows:
            # checked before anything is written, as the replica exchange would refuse them only afterwards
            # (reviews of PR #18, the class of the relative leg's)
            def production_conditions(windows) -> dict:
                return leg_spec(cfg, built, [s.to_dict() for s in windows.states], barostat, timestep_fs,
                                backend_name, seed + 17, name=job.tag, frame_atoms=frame_atoms,
                                frame_ligand=frame_ligand, frame_ca=frame_ca).conditions()

            if given is None and not sched_file.exists():
                refuse_samples_without_windows(leg_dir)
            keeps = False                               # the leg keeps windows of samples it may have
            if given is not None or sched_file.exists() or ls.mode_of(job.leg) == "fixed":
                # the windows are given: only make a start configuration for each of them
                if given is not None:
                    schedule = given
                    check_kept_windows(leg_dir, schedule, production_conditions(schedule), refuse_unrecorded=False)
                    keeps = True
                elif sched_file.exists():
                    kept = _schedule_of_this_leg(sched_file, job.leg, stages)
                    check_scale(kept)
                    check_kept_windows(leg_dir, kept, production_conditions(kept), refuse_unrecorded=False)
                    schedule = own_schedule(leg_dir, job.leg, stages, ligand=job.ligand, repeat=job.repeat,
                                            seed=seed, backend=backend_name)
                    keeps = True
                else:
                    schedule = schedule_from_windows(stages, ls.windows.for_leg(job.leg),
                                                     WindowsConfig().for_leg(job.leg), rest2.scale)
                check_scale(schedule)                      # before anything is written for these windows
                cfgs = _walk_schedule(ctx, integ, schedule, ls.fixed_walk_ps, timestep_fs)
                made_with = given_windows_record(ls.fixed_walk_ps, schedule, found_in_leg)
                if keeps:
                    # samples taken up as they were made keep their start (unknown without a record of these
                    # windows); how the file was made now is said apart (reviews of PR #18)
                    made_with = kept_start_record(leg_dir, schedule, windows_settings(cfg), made_with, log)
            else:
                als = als_settings(cfg, job.leg, rest2.scale)

                def inline_mover():
                    """The water moves of a window made by a mover on this Context, for the windows that run in
                    this process."""
                    mover, interval_steps = None, 0
                    if built.relocation is not None:
                        from cufep.sampling.relocation_mc import RelocationMover
                        from cufep.sampling.worker import RELOCATION_STREAM, stream_seed
                        from cufep.steps import md_steps

                        mover = RelocationMover(built.relocation, ctx, stream_seed(seed + 23, 0, 0, RELOCATION_STREAM))
                        interval_steps = md_steps(built.relocation.interval_ps, timestep_fs)
                    if built.gcmc is not None:                 # (and the grand canonical moves of water)
                        from cufep.sampling.gcmc import GcmcMover
                        from cufep.sampling.worker import GCMC_STREAM, stream_seed
                        from cufep.steps import md_steps

                        mover = GcmcMover(built.gcmc, ctx, stream_seed(seed + 23, 0, 0, GCMC_STREAM))
                        interval_steps = md_steps(built.gcmc.interval_ps, timestep_fs)
                    return mover, interval_steps

                # the windows of an iteration run at the same time on the GPUs the scheduler grants
                schedule, cfgs = adaptive_windows(run, pool_spec, ctx, integ, stages, als,
                                                  alch.system.getNumParticles(), inline_mover)
                made_with = {"start_configurations": als.start_configurations,
                             **({"walk_ps": float(als.walk_ps)} if als.start_configurations == "walk" else {}),
                             "windows": "chosen by this precalculation"}
            del ctx, integ
            configs = publish_windows(leg_dir, schedule, cfgs, start_record(cfg, schedule, made_with), leg=job.leg,
                                      ligand=job.ligand, repeat=job.repeat, seed=seed, backend=backend_name)
            windows_origin = made_with.get("origin_not_recorded")
            # how the file was made now (fifth review of PR #18: KeyError for samples whose start is not known)
            made_now = made_with.get("start_configurations_made_again") or made_with
            timed("lambda_schedule", t0)
            log(
                f"lambda schedule ({schedule.mode}): {len(schedule.states)} windows "
                f"in {time.perf_counter() - t0:.0f} s: "
                + "; ".join(f"{name} {len(v)}" for name, v in schedule.windows().items())
                + f"; start configurations: {made_now['start_configurations']}"
                + (f" ({made_now['walk_ps']:g} ps in each window)" if "walk_ps" in made_now else "")
            )
        # what of the windows has no record of its origin, as their files say it now (not what an
        # attempt on another backend found before its windows were set aside)
        origin_not_recorded[:] = [item for item in origin_not_recorded if item not in WINDOWS_ORIGINS]
        if windows_origin:
            origin_not_recorded.append(windows_origin)
        check_scale(schedule)

        # ---- 5. replica exchange -----------------------------------------------------------------
        spec = leg_spec(cfg, built, [s.to_dict() for s in schedule.states], barostat, timestep_fs, backend_name,
                        seed + 17, name=job.tag, frame_atoms=frame_atoms, frame_ligand=frame_ligand, frame_ca=frame_ca,
                        mps_thread_percentage=thread_percentage(job.oversubscription))

        def unrecorded(sampler) -> None:
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

        samples = sample_leg(spec, hrex_dir, job, configs, n_equil, n_equil + n_prod, cfg, log, timing, timed,
                             ckpt_stride, unrecorded)
        # what the record of the start configurations says now (the replica exchange may have started from start
        # configurations made again, whose start is known: seventh review of PR #18)
        origin_not_recorded[:] = [item for item in origin_not_recorded if item not in WINDOWS_ORIGINS]
        if windows_origin_of(leg_dir):
            origin_not_recorded.append(windows_origin_of(leg_dir))
        return samples, schedule

    (samples, schedule), backend_name = with_fallback(run, backend_name, run_with)

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
    finish_leg(run, result, backend_name, alch.system.getNumParticles(), {"ligand": job.ligand})

    def number(value, digits: int = 2, scale: float = 1.0) -> str:
        return "none" if value is None else f"{value / scale:.{digits}f}"

    slowest = analysis_dict["slowest_window"]
    blocks = analysis_dict.get("delta_g_error_n_blocks")
    error_from = (f"from {blocks} time blocks" if analysis_dict.get("delta_g_error_from") == "time blocks"
                  else analysis_dict.get("delta_g_error_from") or "analytical")
    log(f"dG(decouple) = {number(total, 2, KCAL)} ± {number(analysis_dict['delta_g_error'], 2, KCAL)} kcal/mol "
        f"(standard error {error_from}; MBAR {number(analysis_dict['delta_g_mbar'], 2, KCAL)} ± "
        f"{number(analysis_dict['delta_g_mbar_error'], 2, KCAL)}, tail {number(-correction, 2, KCAL)}); "
        f"min overlap {number(analysis_dict['min_overlap'], 3)} (among the samples of the estimate "
        f"{number(analysis_dict['min_overlap_used'], 3)}), round trips {analysis_dict['round_trips']}, fewest "
        f"independent samples in a window {analysis_dict['effective_samples_per_state']:.0f}, of which MBAR gets "
        f"{analysis_dict['window_kept_samples'][slowest]} (window {slowest})")
    return result


def rest2_region(rest2, topology, positions_nm, ligand, protein,
                 input_chains: dict[str, str] | None = None) -> tuple[list[int], list[str]]:
    """Protein atoms of the REST2 hot region (whole residues) and the names of their residues.

    ``region: pocket`` takes the residues with a heavy atom within ``pocket_cutoff_nm`` of a
    heavy atom of the ligand in the prepared structure; ``residues`` adds residues named as in
    the input PDB: ``"960"``, ``"B:960"`` or with an insertion code ``"B:898A"``. The
    preparation gives every peptide segment a chain of its own, named A, B, ... in order, so the
    chain of the input is found through ``input_chains`` (prepared chain -> input chain, recorded
    in prep.json); a number without a chain must name one residue of the input. A preparation of
    an earlier version recorded no such map, and residues are then refused rather than guessed
    (review of PR 1: with only chain B in the input, "B:960" found nothing, and with several
    chains it could take a residue of another chain).
    """
    import re

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
    if rest2.residues and input_chains is None:
        raise WorkflowError("sampling.rest2.residues names residues of the input PDB, and the prepared systems of this "
                            "ligand were made by an earlier version that did not record the chains of the input. "
                            "Nothing was changed; run the calculation in a new output.dir")
    for spec in rest2.residues:
        chain, _, number = str(spec).strip().rpartition(":")
        m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", number.strip())
        if m is None:
            raise WorkflowError(f"sampling.rest2.residues: '{spec}' is not a residue number of the input PDB "
                                "(\"960\", \"B:960\" or \"B:898A\")")
        found = {}
        for r in protein_residues.values():
            source = input_chains.get(r.chain.id)
            code = (getattr(r, "insertionCode", "") or "").strip()
            if str(r.id).strip() == m[1] and code == m[2] and (not chain or source == chain.strip()):
                found[r.index] = source
        if not found:
            raise WorkflowError(f"sampling.rest2.residues: no protein residue '{spec}' in the input PDB")
        if len(set(found.values())) > 1:
            raise WorkflowError(f"sampling.rest2.residues: '{spec}' is a residue of the chains "
                                f"{', '.join(sorted(set(found.values())))} of the input PDB; give the chain, e.g. "
                                f"'{sorted(set(found.values()))[0]}:{number.strip()}'")
        chosen.update(found)
    hot = sorted(i for i in protein if atoms[i].residue.index in chosen)
    names = [f"{protein_residues[k].name}{protein_residues[k].id}"
             f"{(getattr(protein_residues[k], 'insertionCode', '') or '').strip()}" for k in sorted(chosen)]
    return hot, names


def als_settings(cfg, leg: str, rest2_scale: float = 1.0):
    """The settings of the precalculation of adaptive lambda scheduling for one leg, from
    ``lambda_schedule.adaptive``: a setting that is given per leg is taken for this leg. The one
    place where the configuration becomes what the precalculation is run with."""
    from cufep.alchemy.schedule import AlsSettings

    a = cfg.lambda_schedule.adaptive
    return AlsSettings(overlap_threshold=a.threshold(leg), overlap_matrix=a.matrix(leg),
                       candidates=a.candidates_of(leg), initial_windows=a.initial_windows.for_leg(leg),
                       window_ps=a.window(leg), equilibration_ps=a.equilibration(leg),
                       sample_interval_ps=a.sample_interval_ps,
                       start_configurations=a.start_configurations, walk_ps=a.walk_ps,
                       max_windows=int(getattr(a.max_windows, leg)), rest2_scale=float(rest2_scale))


def _schedule_of_this_leg(path: Path, leg: str, stages: list[str]):
    """Load a lambda schedule and make sure it is one of this leg (see ``Schedule.check_leg``)."""
    from cufep.alchemy.schedule import Schedule

    try:
        schedule = Schedule.load(path)
        schedule.check_leg(leg, stages)
    except ValueError as exc:
        raise WorkflowError(f"the lambda schedule {path} cannot be used for the {leg} leg: {exc}") from exc
    return schedule


def given_schedule_file(cfg: Config, ligand_dir: str, leg: str, repeat: int) -> Path | None:
    """The file from which a leg is given its lambda windows (``lambda_schedule.file``: one file
    for every leg, or a directory of frozen schedules, ``<ligand>/<leg>_r<repeat>.yaml``); None if
    the configuration names none."""
    if cfg.lambda_schedule.file is None:
        return None
    src = Path(cfg.lambda_schedule.file)
    return src / ligand_dir / f"{leg}_r{repeat}.yaml" if src.is_dir() else src


def check_given_schedule_scale(src: Path, rest2_scale: float) -> None:
    """Refuse a given lambda schedule whose REST2 scale is not that of the run, before the leg
    takes it over (review of 2861b93: it was refused after its windows and their start
    configurations had been written with the settings of the run, and the run with the scale
    corrected to that of the schedule was then refused for those start configurations)."""
    from cufep.alchemy.schedule import Schedule

    try:
        given = float(Schedule.load(src).rest2_scale)
    except ValueError as exc:
        raise WorkflowError(f"the lambda schedule {src} cannot be read: {exc}") from exc
    if abs(given - rest2_scale) > 1e-9:
        raise WorkflowError(
            f"the lambda schedule {src} (lambda_schedule.file) was made with a REST2 scale of {given:g} but this "
            f"leg runs with {rest2_scale:g} (sampling.rest2.scale). Nothing of the windows of the leg was written. "
            "Use the same value, or a schedule made with it.")


def check_given_schedule(cfg: Config, lig_dir: Path, leg: str, repeat: int, reschedule: bool = False) -> None:
    """Before a run writes or samples anything: the lambda schedule a leg is to be given
    (``lambda_schedule.file``) must be there, one of this leg (its stages and its two end states)
    and of the REST2 scale of the run. A leg takes a given schedule when it has no windows, or
    when it is started anew (``--reschedule``); for a leg that keeps its windows nothing is
    checked. From the configuration and the files alone, also for a ligand that has no prepared
    systems yet (review of 3897311: the scale was checked for legs with prepared systems and
    without ``--reschedule`` only, the leg not at all; a leg started anew with a schedule it could
    not take had its samples set aside and then failed). Raises WorkflowError;
    :func:`given_schedule` checks again when the leg takes the schedule."""
    from cufep.sampling.states import STAGES_COMPLEX, STAGES_SOLVENT

    given = given_schedule_file(cfg, Path(lig_dir).name, leg, repeat)
    leg_dir = Path(lig_dir) / f"repeat_{repeat}" / leg
    if given is None or ((leg_dir / "schedule.yaml").exists() and not reschedule):
        return
    inside = given.resolve().relative_to(leg_dir.resolve()).parts \
        if given.resolve().is_relative_to(leg_dir.resolve()) else ()
    moved = inside and (inside[0] in LEG_RUN_FILES or inside[0].startswith(FAILED_BACKEND_DIR)
                        or (inside[0].startswith(".previous_") and inside[0].endswith(".partial")))
    if reschedule and moved:
        # what archive_leg moves or renames, not an archive that is complete (previous_<n>/).
        # (Reviews of 8dc042e, 6b1a609 and 93d42ee: the schedule of the leg itself passed the check,
        # was set aside with the leg and was gone when the leg came to take it, after its samples
        # and its result had been set aside too; the first form of this refusal also refused the
        # schedule in previous_1/, which stays where it is; and a file in an archive that was begun
        # and not completed is renamed with it)
        raise WorkflowError(f"lambda_schedule.file names {given}, a file of the leg that --reschedule sets aside. "
                            "Nothing was changed. Give a copy of it, or the one in an archive of the leg "
                            "(previous_<n>/).")
    if not given.exists():
        raise WorkflowError(f"lambda_schedule.file is set but there is no schedule at {given}")
    _schedule_of_this_leg(given, leg, list(STAGES_COMPLEX if leg == "complex" else STAGES_SOLVENT))
    check_given_schedule_scale(given, cfg.sampling.rest2.scale if cfg.sampling.rest2.scale > 1.0 else 1.0)


def given_schedule(src: Path, leg: str, stages: list[str], rest2_scale: float):
    """The lambda schedule in ``src`` (``lambda_schedule.file``) for a leg that has no windows, if
    it is one of this leg and of the REST2 scale the leg runs with; raises WorkflowError if not.

    Nothing is written: the leg walks through the given windows and publishes its start
    configurations and then its ``schedule.yaml`` together (:func:`publish_windows`), as it does
    for windows of its own. So a leg has a schedule only with the start configurations made for
    it, wherever a run of given windows is stopped, and its ``schedule.yaml`` names this leg.
    (Reviews of b9e7be2, 8dc042e and 6b1a609: while the file was copied into the leg first, start
    configurations of a stopped precalculation were taken for given windows of the same values;
    a run stopped during the walk left windows that a later run could not tell from those of an
    earlier version; and one stopped between the two writes kept the record of the leg the file
    came from.)"""
    schedule = _schedule_of_this_leg(src, leg, stages)
    check_given_schedule_scale(src, rest2_scale)
    return schedule


#: what the result of a leg names when the origin of its lambda windows is not recorded
#: (``origin_not_recorded``): windows and start configurations of a version that wrote no record,
#: sampled for the first time by a later one; windows that adaptive lambda scheduling chose in an
#: earlier run which left no start configurations with them
WINDOWS_ORIGINS = ("the lambda windows and their start configurations", "the lambda windows")


def own_schedule(leg_dir: Path, leg: str, stages: list[str], **origin):
    """The schedule a leg has without start configurations, with the record of this leg written
    to it before the walk through its windows (``origin``: ligand, repeat, seed, backend).

    Such a schedule can be the copy of a given one that an earlier version put into the leg
    before its walk: it names the leg it came from. The record is set here, before the start
    configurations are published, so that the leg cannot end with start configurations of its
    own under a schedule that names another ligand (review of 93d42ee: a run stopped between the
    two writes of :func:`publish_windows` left that, and the next run took both files)."""
    schedule = _schedule_of_this_leg(Path(leg_dir) / "schedule.yaml", leg, stages)
    schedule.save(Path(leg_dir) / "schedule.yaml", leg=leg, **origin)
    return schedule


def given_windows_record(walk_ps: float, schedule, found_in_leg: bool) -> dict:
    """How start configurations were made by the walk through windows that the precalculation of
    this run did not choose (:func:`start_record`): the windows of the configuration (mode
    fixed), a schedule this run took from ``lambda_schedule.file``, or a schedule the leg had
    before the run without start configurations (``found_in_leg``).

    Windows of the last kind that adaptive lambda scheduling chose come from an earlier run of
    the leg which left no start configurations with them, a version that wrote the schedule
    first: the settings they were chosen with are not recorded, and the record says so, whatever
    ``lambda_schedule.file`` names now (review of 8dc042e: with a file named, which a leg that has
    windows does not take, the note was left out)."""
    record = {"start_configurations": "walk of the given windows", "walk_ps": float(walk_ps), "windows": "given"}
    if found_in_leg and schedule.mode == "adaptive":
        record["origin_not_recorded"] = WINDOWS_ORIGINS[1]
    return record


def windows_origin_of(leg_dir: Path) -> str | None:
    """What the record written with the start configurations of a leg says is not recorded of the
    origin of its windows (one of :data:`WINDOWS_ORIGINS`), or None."""
    file = Path(leg_dir) / "start_configs.npz"
    if not file.exists():
        return None
    try:
        origin = start_record_of(file).get("origin_not_recorded")
    except (OSError, ValueError, KeyError):
        return None
    return origin if origin in WINDOWS_ORIGINS else None


def adopt_windows(leg_dir: Path, schedule, configs: tuple, cfg: Config, made: dict) -> dict:
    """Give start configurations that an earlier version wrote without a record of their settings
    the record of the run that samples them first: its settings, the windows, what the file said
    of how they were made, and that their origin is not recorded. Returns the record.

    The result of the leg then says so in this run and in every later one (review of 2861b93: it
    was said until the first checkpoint, after which the leg had samples and went on without the
    note), and a run with other settings is refused as for any leg that has its windows."""
    how = {key: made[key] for key in ("start_configurations", "walk_ps") if key in made}
    record = dict(start_record(cfg, schedule, how), origin_not_recorded=WINDOWS_ORIGINS[0])
    publish_npz(Path(leg_dir) / "start_configs.npz", positions=configs[0], boxes=configs[1], velocities=configs[2],
                made_with=json.dumps(record), **({"water_on": configs[3]} if len(configs) > 3 else {}))
    return record


def stored_windows(leg_dir: Path, leg: str, stages: list[str], settings: dict, log=print):
    """The lambda windows a leg has with the start configurations of their replicas, if this run
    can go on from them: ``(schedule, (positions, boxes, velocities), the record written with the
    start configurations)``, the record empty for files of a version that wrote none. None if the
    leg does not have both files, or has start configurations that are not those of its windows
    (other states, or another number of them): they are made again for the windows the leg has.
    Raises WorkflowError if the files were made with another System
    (:func:`check_windows_settings`)."""
    sched_file, start_file = Path(leg_dir) / "schedule.yaml", Path(leg_dir) / "start_configs.npz"
    if not (sched_file.exists() and start_file.exists()):
        return None
    schedule = _schedule_of_this_leg(sched_file, leg, stages)
    made = start_record_of(start_file)
    with np.load(start_file) as start:
        configs = (start["positions"], start["boxes"], start["velocities"],
                   *((start["water_on"],) if "water_on" in start.files else ()))
    if not starts_are_of(made, schedule) or len(configs[0]) != len(schedule.states):
        log(f"the start configurations in {start_file} are not those of the lambda windows in {sched_file} "
            f"({len(configs[0])} configurations, {len(schedule.states)} windows): they are made again for these "
            "windows")
        return None
    check_windows_settings(made, settings, start_file, log)
    return schedule, configs, made


def publish_windows(leg_dir: Path, schedule, cfgs: list, record: dict, **origin) -> tuple:
    """Write the start configurations of the lambda windows of a leg (``cfgs``: positions, box and
    velocities per window) with their record (:func:`start_record`), then its schedule, and return
    the configurations as three arrays.

    The start configurations first: a leg that has a schedule of its own then has the start
    configurations that were made with it. (Written the other way round, a leg that was stopped
    between the two had its windows and made their start configurations anew by the walk of given
    windows, under a schedule that recorded another way.) ``origin`` is what the schedule records
    of the leg it is of (:meth:`cufep.alchemy.schedule.Schedule.save`)."""
    configs = (np.array([c[0] for c in cfgs]), np.array([c[1] for c in cfgs]), np.array([c[2] for c in cfgs]))
    # with grand canonical moves of water: which waters interact in each (cufep.sampling.gcmc)
    if cfgs and all(len(c) > 3 for c in cfgs):
        configs += (np.array([c[3] for c in cfgs], dtype=bool),)
    publish_npz(Path(leg_dir) / "start_configs.npz", positions=configs[0], boxes=configs[1], velocities=configs[2],
                made_with=json.dumps(record), **({"water_on": configs[3]} if len(configs) > 3 else {}))
    schedule.save(Path(leg_dir) / "schedule.yaml", **origin)
    return configs


def _walk_schedule(context, integrator, schedule, window_ps: float, timestep_fs: float) -> list:
    """Start configurations for given windows: a short run in each window, in order."""
    import openmm.unit as u

    from cufep.sampling.states import set_state
    from cufep.steps import md_steps

    steps = md_steps(window_ps, timestep_fs)
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
    return quality_checks(analysis, a.min_overlap, a.min_effective_samples)


def analysis_settings(source):
    """The settings with which the result of a leg is made from its samples and files.

    They are the ``analysis`` section: the run of a leg, a result that is made again and ``cufep
    analyze`` use the same ones. ``source`` is a whole configuration (returned as it is), an object
    with an ``analysis`` section, or an ``analysis`` section alone. (Earlier versions added levels of
    cuFEP's own for notes on the bilayer and the pose; they are gone, the owner 2026-11-20.)
    """
    if hasattr(source, "analysis"):
        return source
    settings = analysis_settings_of(None)
    settings.analysis = source
    return settings


#: the version of the structure numbers of a result (:func:`cufep.analysis.structure.structural_analysis`).
#: 2: the ligand is taken in the image of its nearest protein atom; before (no version recorded) in the image
#: nearest to the centre of the C-alpha atoms, which is another one for a ligand farther from that centre than
#: half a cell. A result of an earlier version is made again from its samples (review of eb6f89f: its numbers
#: stayed as "current"). The free energies are those of ANALYSIS_VERSION and do not change with it.
STRUCTURE_VERSION = 2


def analysis_record(cfg) -> dict:
    """The settings the result of a leg is made with (:func:`analysis_settings`) and the version of
    the method, as the result records them."""
    from cufep.analysis.mbar import ANALYSIS_VERSION

    settings = analysis_settings(cfg)
    a = settings.analysis
    record = {"version": ANALYSIS_VERSION, "structure": STRUCTURE_VERSION, "lrc": a.lrc,
              "lrc_cutoff_nm": float(a.lrc_cutoff_nm),
              "min_overlap": float(a.min_overlap),
              "min_effective_samples": float(a.min_effective_samples),
              "convergence_points": int(a.convergence_points), "convergence_blocks": int(a.convergence_blocks),
              "error_blocks": [int(count) for count in a.error_blocks], "part_error_blocks": int(a.part_error_blocks)}
    # a setting that was added later is recorded only away from the value that gives what the
    # versions before it did: a result made that way has the record every earlier result has, and
    # is not made again for a key
    record.update({key: getattr(a, key) for key, value in ANALYSIS_SETTINGS_ADDED.items() if getattr(a, key) != value})
    return record


def production_start(samples: dict, n_equilibration: int, analysis) -> tuple[int, dict | None]:
    """Where the analysis of stored samples begins, and what automatic equilibration detection
    found (None with ``analysis.detect_equilibration`` off): the end of the equilibration of the
    run, ``n_equilibration``, and with the detection the iterations it leaves out after that
    (:func:`cufep.analysis.mbar.detect_equilibration`, on the iterations from the end of the
    equilibration on). The one place for the result of a leg and for the check of what an
    analysis needs, so that both take the same production."""
    from cufep.analysis.mbar import detect_equilibration

    a = getattr(analysis, "analysis", analysis)
    if not a.detect_equilibration:
        return int(n_equilibration), None
    found = detect_equilibration(samples["u"][n_equilibration:], samples["states"][n_equilibration:])
    if found is None:                      # too few iterations, or an energy that does not change
        return int(n_equilibration), {"iterations": 0, "statistical_inefficiency": None, "effective_samples": None}
    return int(n_equilibration) + int(found["iterations"]), found


def analyse_leg_directory(directory, config=None, equilibration_ns: float | None = None,
                          equilibration_iterations: int | None = None) -> dict:
    """The result of the samples in a leg directory (or its hrex/, or an archive previous_<n>/), as
    ``cufep analyze`` gives it and with the settings it uses: those of ``config`` (a whole
    configuration) if given, else those the leg's result records, else those its run stored with
    the samples before it began, else those of the result the leg had before, else the defaults.
    Returns {"result", "samples", "n_equilibration", "source", "directory"}. Raises WorkflowError
    (and NoProduction, NoCorrection through it) with the reason. The one way for every tool that
    analyses a leg directory, so that they cannot differ in settings or in the equilibration."""
    from cufep.analysis.mbar import NoProduction
    from cufep.sampling.storage import equilibration_boundary, read_samples

    if equilibration_ns is not None and equilibration_iterations is not None:
        raise WorkflowError("give the equilibration either as a length or as a number of iterations, not both")
    if equilibration_ns is not None and not (np.isfinite(equilibration_ns) and equilibration_ns >= 0):
        raise WorkflowError(f"the length of the equilibration must be a number of nanoseconds of 0 or more, not "
                            f"{equilibration_ns}")
    if equilibration_iterations is not None and int(equilibration_iterations) < 0:
        raise WorkflowError(f"the number of equilibration iterations must be 0 or more, not {equilibration_iterations}")
    d = Path(directory)
    if d.name == "hrex" and (d / "samples.h5").exists():
        d = d.parent
    hrex = d / "hrex"
    if not (hrex / "samples.h5").exists():
        raise WorkflowError(f"{d}: no samples (hrex/samples.h5) in this directory")
    from cufep.sampling.storage import StorageError

    try:
        samples = read_samples(hrex)
    except StorageError as exc:                    # a damaged leg is said as such, not as a traceback
        raise WorkflowError(f"{d}: the samples cannot be read: {exc}") from exc
    attrs = samples["attrs"]
    if equilibration_iterations is not None:
        n_equilibration = int(equilibration_iterations)            # as given, exactly
    elif equilibration_ns is not None:
        # the rule of the run (iterations_for), with the length of an iteration its configuration asked
        # for where the samples record it, else the length an iteration had (which can differ from the
        # one asked for by a fraction of a step: then --equilibration-iterations gives the boundary
        # exactly); an equilibration of 0 is none
        recorded = ((attrs.get("conditions") or {}).get("settings") or {}).get("time_per_iteration_ps") \
            if isinstance(attrs.get("conditions"), dict) else None
        if recorded is None and "iteration_ps" not in attrs:
            raise WorkflowError(f"{hrex}: the samples do not record the length of an iteration")
        n_equilibration = iterations_for(equilibration_ns, float(recorded or attrs["iteration_ps"]),
                                         1 if equilibration_ns > 0 else 0)
    else:
        n_equilibration = equilibration_boundary(attrs, d)
        if n_equilibration is None:
            raise WorkflowError(
                f"{hrex}: these samples are of a run that did not finish and that was made by a version of cuFEP "
                "which recorded the end of the equilibration only then. Give the length of the equilibration per "
                "lambda window with --equilibration-ns (sampling.equilibration_ns of that run; the default was 1).")
    results = {}
    for name in ("result.json", "result.previous.json"):
        if (d / name).exists():
            try:
                results[name] = json.loads((d / name).read_text())
            except json.JSONDecodeError:
                continue
    earlier = results.get("result.json") or results.get("result.previous.json")
    of_the_run = attrs.get("analysis_record") if isinstance(attrs.get("analysis_record"), dict) else None
    if config is not None:
        settings, source = config, "the configuration given"
    elif (results.get("result.json") or {}).get("analysis_record"):
        settings = analysis_settings_of(results["result.json"]["analysis_record"], attrs)
        source = "the settings the leg's result records"
    elif of_the_run:
        settings = analysis_settings_of(of_the_run, attrs)
        source = "the settings of the run that made the samples, which has no result (yet)"
    elif (results.get("result.previous.json") or {}).get("analysis_record"):
        settings = analysis_settings_of(results["result.previous.json"]["analysis_record"], attrs)
        source = "the settings the result records that the leg had before"
    else:
        tail = "reweight" if "lrc_s6" in attrs else "none"
        if earlier is not None and earlier.get("lrc_method") in ("reweight", "isotropic", "none"):
            tail = earlier["lrc_method"]
        # the defaults, with the outer radius the samples recorded their tail energies to
        settings = analysis_settings_of({"lrc": tail}, attrs)
        source = "the defaults: the leg has no record of the settings it was analysed with"
    try:
        result = leg_result_from_samples(d, settings, run_record_of(earlier, d), samples=samples,
                                         n_equilibration=n_equilibration)
    except NoProduction as exc:
        raise WorkflowError(f"{hrex}: no production yet: {exc}") from exc
    return {"result": result, "samples": samples, "n_equilibration": n_equilibration, "source": source,
            "directory": d}


def recorded_tail_radius(attrs: dict) -> float | None:
    """The outer radius (nm) to which the samples with these attributes recorded their tail
    energies, or None if they do not say."""
    conditions = attrs.get("conditions") if isinstance(attrs.get("conditions"), dict) else {}
    tail = conditions.get("lrc") if isinstance(conditions.get("lrc"), dict) else {}
    radius = tail.get("long_cutoff_nm")
    return None if radius is None else float(radius)


def analysis_problems(attrs: dict, analysis) -> list[str]:
    """What the samples with these attributes lack for the analysis that ``analysis`` asks for
    (empty if nothing): the data of the tail correction, and for the correction by reweighting
    the outer radius the tail energies were recorded to, which cannot be changed afterwards.

    The one check of this for every way to a result: :func:`leg_result_from_samples` (the run of
    a leg, the refresh of a leg that a run does not sample, ``cufep analyze`` with and without
    ``--config``) and the check before a run of the finished legs it will be combined with.
    """
    from cufep.sampling.storage import _same

    a = getattr(analysis, "analysis", analysis)
    problems = []
    if a.lrc in ("reweight", "isotropic") and "lrc_s6" not in attrs:
        problems.append(f"they hold no data for the tail correction (they were made with analysis.lrc: none) and "
                        f"analysis.lrc is {a.lrc}")
    radius = recorded_tail_radius(attrs)
    if a.lrc == "reweight" and radius is not None and not _same(radius, float(a.lrc_cutoff_nm)):
        problems.append(f"their tail energies were recorded to an outer radius of {radius:g} nm and "
                        f"analysis.lrc_cutoff_nm is {float(a.lrc_cutoff_nm):g} (the radius cannot be changed after the "
                        "sampling)")
    return problems


def analysis_settings_of(record: dict | None, attrs: dict | None = None):
    """The settings that a stored :func:`analysis_record` describes (:func:`analysis_settings`;
    defaults for what it lacks, except the outer radius of the tail correction, which a record of
    an earlier version lacks and the samples ``attrs`` hold)."""
    from types import SimpleNamespace

    from cufep.config.schema import AnalysisConfig, renamed_settings

    record, _renamed = renamed_settings("analysis", dict(record or {}))
    if "lrc_cutoff_nm" not in record and attrs is not None and recorded_tail_radius(attrs) is not None:
        record["lrc_cutoff_nm"] = recorded_tail_radius(attrs)
    return SimpleNamespace(
        analysis=AnalysisConfig(**{key: value for key, value in record.items() if key in AnalysisConfig.model_fields}))


def notes_of_leg_files(leg_dir: Path) -> tuple[list[str], dict]:
    """The notes that the pre-equilibration and the restraint of a leg carry, from the files of the
    leg, without verdicts of cuFEP's own (:func:`equil_notes`, :func:`restraint_notes`), and what
    those files say as numbers: how the bilayer changed at the end of its equilibration (``bilayer``),
    how far the ligand was from its input pose (``ligand_rmsd_from_input_A``) and what a restraint
    chosen from interactions rests on (``restraint_choice``, :func:`restraint_choice`). A file that
    is not there gives no notes and no number.
    """
    notes: list[str] = []
    numbers: dict = {"bilayer": None, "ligand_rmsd_from_input_A": None, "restraint_choice": None}
    files = leg_files(leg_dir)
    equil_file, restraint_file = files["equil.npz"], files["restraint.json"]
    if equil_file is not None:
        eq = np.load(equil_file)
        notes += equil_notes(eq)
        numbers["bilayer"] = bilayer_numbers(eq)
    if files[RESTRAINT_EQUIL] is not None:
        # a leg whose alchemical run is made in water and whose restraint was chosen in the bilayer: the bilayer
        # is the one of that MD, and what its pre-equilibration notes (a recovery) is said of it
        eq_md = np.load(files[RESTRAINT_EQUIL])
        notes += [note for note in (f"pre-equilibration of the complex in the bilayer, for the restraint: {n}"
                                    for n in equil_notes(eq_md)) if note not in notes]
        if numbers["bilayer"] is None:
            numbers["bilayer"] = bilayer_numbers(eq_md)
    stored = _read_json(restraint_file) if restraint_file is not None else None
    if stored:
        notes += [n for n in restraint_notes(stored) if n not in notes]
        info = stored.get("info") or {}
        if info.get("ligand_rmsd_reference") == "input pose" and info.get("ligand_rmsd_A") is not None:
            numbers["ligand_rmsd_from_input_A"] = float(info["ligand_rmsd_A"])
        numbers["restraint_choice"] = restraint_choice(info)
    return notes, numbers


def restraint_choice(info: dict) -> dict | None:
    """What a restraint chosen from interactions (``restraints.selection: interactions``) rests on,
    as its file records it: the method (interactions, or the distance method), the anchors, the
    interaction with its residue and in how many of the analysed frames it is there, and the MD
    (trajectories, analysed frames, the backend that ran each). Numbers, without a verdict. None
    for a restraint of another selection."""
    if info.get("selection") != "interactions":
        return None
    choice = {key: info.get(key) for key in ("method", "anchors", "interaction", "score", "candidates_tested",
                                             "candidates_refused", "md")}
    if info.get("md_system"):
        # the MD was made on another prepared system than the alchemical run (membrane.complex_leg: water: the
        # complex in the bilayer); said only then, so that every other result stays as it was
        choice["md_system"] = info["md_system"]["name"]
    return choice


#: how the notes begin that the analysis of a leg writes; a result that is made again from its
#: samples gets them anew
_ANALYSIS_NOTES = (
    "NOT RELIABLE",
    "NO FREE ENERGY",
    "NOT CONVERGED",
    "convergence not assessed",
    "few independent samples",
    "the standard error of this leg is the one MBAR gives",
    "the standard error of this leg is the analytical one",
    "parts of the production without an estimate",
    "parts of the production whose lambda windows overlap below",
    "the comparison of the two halves",
    "the two halves of the production",
    "half of the production does not",
    "BAR between neighbouring lambda windows",
    "written by an earlier version of cuFEP without a record",
    # verdicts of earlier versions with levels of cuFEP's own, no longer made (the owner, 2026-11-20)
    "the bilayer had not settled after",
    "the ligand moved away from its",
    "the distance of the ligand from its input pose was measured by an earlier version",
    "the ligand moved ",
    "the ligand's centre of mass drifted",
    "bilayer area or pocket hydration still drifting",
)
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
    found = leg_files(leg_dir)["restraint.json"]
    if found is None:
        if restrained:
            raise WorkflowError(f"the samples in {leg_dir} are of a leg with a restrained ligand, but its "
                                "restraint.json is not there: the release of the restraint cannot be computed")
        return 0.0, 0.0
    if not restrained and states:
        return 0.0, 0.0
    stored = _read_json(found) or {}
    # a restraint chosen from interactions belongs to the pre-equilibration its MD started from: the run of a
    # leg refuses another one, and so does every reading of its samples
    check_restraint_of_leg(stored, found, leg_dir)
    restraint = BoreschRestraint.from_dict(stored.get("restraint") or {})
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
    from cufep.engine.protocol import state_from_dict
    from cufep.sampling.storage import _same, equilibration_boundary

    def normal(sequence):
        return [state_from_dict(state).to_dict() for state in sequence]

    problems = []
    if "temperature_K" in attrs and not _same(float(attrs["temperature_K"]), float(cfg.thermodynamics.temperature_K)):
        problems.append(
            f"temperature: {float(attrs['temperature_K']):g} K -> {float(cfg.thermodynamics.temperature_K):g} K"
        )
    # a setting of alchemy that came after such samples: they were made with the one treatment
    # their version had (review of 2861b93: complete samples of that kind were given back for a run
    # that asked for annihilated electrostatics or the squared soft-core)
    for key, value in ALCHEMY_SETTINGS_ADDED.items():
        if getattr(cfg.alchemy, key) != value:
            problems.append(f"alchemy.{key}: {value} (all the version that made the samples had) -> "
                            f"{getattr(cfg.alchemy, key)}")
    if (
        states is not None
        and isinstance(attrs.get("states"), list)
        and not _same(normal(attrs["states"]), normal(states))
    ):
        problems.append("the lambda states")
    if "iteration_ps" not in attrs:
        return problems + ["the file does not record the length of an iteration"]
    stored_ps = float(attrs["iteration_ps"])
    # the run's own plan (planned_iterations): iterations of whole steps, and numbers of iterations
    plan = planned_iterations(cfg.sampling.time_per_iteration_ps, timestep_fs, cfg.sampling.equilibration(leg),
                              production_ns)
    same_length = abs(stored_ps - plan.iteration_ps) <= 1e-6
    if not same_length:
        problems.append(f"length of an iteration: {stored_ps:g} ps -> {plan.iteration_ps:g} ps")
    boundary = equilibration_boundary(attrs, leg_dir)
    if boundary is None:
        return problems + ["the file does not record where its production begins (the run that wrote it did not "
                           "finish); cufep analyze --equilibration-ns reads such samples"]
    # counted in the iterations of the samples: those of this run where the length is the same (the
    # run's own plan), else iterations of their own length by the same rule
    asked_equil = cfg.sampling.equilibration(leg)
    n_equil = plan.n_equilibration if same_length else iterations_for(asked_equil, stored_ps, 1)
    n_prod = plan.n_production if same_length else iterations_for(production_ns, stored_ps, 4)
    if boundary != n_equil:
        problems.append(
            f"equilibration per lambda window: {boundary} iterations ({boundary * stored_ps / 1000.0:g} ns) "
            f"-> {n_equil} ({asked_equil:g} ns)"
        )
    held = max(0, n_stored - boundary)
    if held < n_prod:
        problems.append(
            f"production per lambda window: the samples hold {held} iterations ({held * stored_ps / 1000.0:g} "
            f"ns) and {production_ns:g} ns are {n_prod} iterations (samples of unrecorded conditions are "
            "not added to)"
        )
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
        record["notes"] = [
            n for n in result.get("notes", []) if not n.startswith(_ANALYSIS_NOTES) and n not in structure
        ]
    return record


def leg_result_from_samples(leg_dir: Path, analysis, run: dict, samples: dict | None = None, schedule=None,
                            n_equilibration: int | None = None) -> dict:
    """The result of a leg of the ABFE from the samples in its directory. Needs no GPU.

    The analysis of the samples is that of every leg (:func:`cufep.engine.leg.path_result`): the free energy
    along the path of the leg, here the free energy of decoupling (``delta_g_decouple``), with its checks.
    ``analysis`` gives the settings of :func:`analysis_settings` (a whole configuration, or its ``analysis``
    section alone) and nothing else of a configuration enters; ``run`` holds what only the run of the leg knows
    (:data:`_RUN_KEYS` and its notes). The notes of the pre-equilibration and of the restraint are made from the
    files of the leg (:func:`notes_of_leg_files`). The ABFE adds what its restraint and its moves of water give:
    the release of the restraint, the end the ligand is decoupled to, the symmetry number, the numbers of the
    moves. The same function serves a leg that has just been sampled, ``cufep analyze``, and a result that is
    made again because the method or the settings of the analysis have changed.
    """
    from cufep.engine.leg import path_result

    path, of = path_result(leg_dir, analysis, run, {"ligand": run.get("ligand")}, samples=samples,
                           schedule=schedule, n_equilibration=n_equilibration, files=notes_of_leg_files)
    release, release_analytical = restraint_release_of(leg_dir, of["attrs"], of["temperature"])
    result = {}
    for key, value in path.items():
        # the free energy along the path of a leg of the ABFE: decoupling with the full LJ range
        result["delta_g_decouple" if key == "delta_g" else key] = value
        if key == "restraint_choice":
            result.update({
                "restraint_release": release, "restraint_release_analytical": release_analytical,
                # the end the ligand is decoupled to: with the electrostatics inside it kept, or annihilated
                "annihilate_electrostatics": electrostatics_annihilated(of["conditions"]),
                "ligand_symmetry_number": run.get("ligand_symmetry_number", 1)})
        elif key == "start_configurations":
            # the water relocation moves and the grand canonical moves of water of the production that is
            # analysed, by lambda window (None: none)
            result["relocation_mc"] = relocation_numbers(of["samples"], of["first"])
            result["gcmc"] = gcmc_numbers(of["samples"], of["first"])
    return result


def start_configurations_of(leg_dir: Path) -> str | None:
    """How the lambda windows of a leg got the configurations its replicas started from, as the
    record says that is written with the start configurations (``start_configs.npz``). None for a
    leg of an earlier version, whose files do not say it: the record of an adaptive schedule
    tells whether its precalculation had workers (one: every window from the end of the one
    before it; several: a walk), but not whether the start configurations the leg has were made
    by that precalculation. A schedule that was given (``lambda_schedule.file``, also one frozen
    from another run, which keeps the record of that run) got them from a walk through its
    windows (review of dd075d6: such a leg was said to be "walk" or "sequential")."""
    file = Path(leg_dir) / "start_configs.npz"
    if file.exists():
        try:
            made = start_record_of(file)
            if "start_configurations" in made:
                walk = f", {made['walk_ps']:g} ps in each window" if "walk_ps" in made else ""
                return f"{made['start_configurations']}{walk}"
        except (OSError, ValueError, KeyError):
            pass
    return None


def overlap_below_level(result: dict, cfg) -> list[dict]:
    """Where the overlap of neighbouring lambda windows of a leg is below ``analysis.min_overlap``,
    the check behind NOT RELIABLE (:func:`leg_quality`): for all iterations and for the samples the
    free energy is estimated from (``of``, as the check's ``below`` names them), the overlap, the
    pair of windows with the lowest one, its stage and its two lambda values. Empty if the leg
    meets the level. The summary of a ligand keeps it, so that the benchmark report can say where."""
    ov = leg_quality(result["analysis"], cfg)["overlap"]
    used = ov.get("used")
    checks = [("all iterations", ov, not ov.get("all_ok", ov["ok"])),
              ("the samples of the estimate", used, bool(used) and used["ok"] is False)]
    windows = result.get("lambda_windows") or {}
    out = []
    for of, check, below in checks:
        if not below:
            continue
        pair = check.get("windows")
        where = None if not pair else window_stage(windows, pair[0])
        out.append({"of": of, "overlap": check["lowest"], "windows": pair,
                    "stage": None if where is None else where[0],
                    "lambda": None if where is None else [where[1], where[2]]})
    return out


def quality_warnings(result: dict, cfg, command: str = "cufep run CONFIG --ligand NAME") -> list[str]:
    """Warnings about a finished leg, from its checks. None of them changes what is run.

    The verdicts are the two checks of :func:`cufep.analysis.mbar.quality_checks`, made again from
    the numbers of the analysis with the levels of this run, whatever version wrote the result:
    the overlap of neighbouring lambda windows comes first (below the level, the free energy is
    NOT RELIABLE), then the number of uncorrelated samples. The other messages are notes on the
    analysis, not verdicts.
    """
    cfg = getattr(cfg, "analysis", cfg)            # the whole configuration or its analysis section
    a = result["analysis"]
    q = leg_quality(a, cfg)
    leg = result.get("leg", "LEG")
    windows = result.get("lambda_windows") or {}
    longer = f"{command} --leg {leg} --production-ns NS (or sampling.production_ns)"

    def place(k: int) -> str:
        where = window_stage(windows, min(k, max(0, a.get("n_states", k + 2) - 2)))
        return "" if where is None else f" ({where[0]} stage)"

    def pair(k: int) -> str:
        where = window_stage(windows, k)
        return "" if where is None else f" ({where[0]} stage, lambda {where[1]:g} and {where[2]:g})"

    out = []
    if a.get("not_determined"):
        # the leg has no free energy; the ligand counts it as failed (combine_ligand)
        out.append(f"NO FREE ENERGY: {a['not_determined']}. Give the lambda windows more overlap and run this "
                   f"leg again: {command} --leg {leg} --reschedule")
    ov = q["overlap"]
    all_ok = ov.get("all_ok", ov["ok"])            # the matrix of all iterations, equal numbers per window
    more_windows = (f"lambda_schedule.adaptive.initial_windows, or lambda_schedule.windows with mode: fixed) "
                    f"and run this leg again: {command} --leg {leg} --reschedule")
    if not all_ok:
        lowest = "not defined" if ov["lowest"] is None else f"{ov['lowest']:.3f}"
        if ov.get("windows"):
            k = ov["windows"][0]
            between = f"between lambda windows {k} and {k + 1}{pair(k)}"
        else:                                      # a result that records the lowest overlap only
            between = "between two neighbouring lambda windows"
        out.append(f"NOT RELIABLE: the overlap {between} is {lowest}, below {cfg.min_overlap:g}. Give that stage "
                   f"more lambda windows ({more_windows}")
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
        out.append(f"few independent samples: lambda window {smp['window']}{place(smp['window'])} has "
                   f"{smp['fewest']:.0f}{kept} (its energy difference to the {towards} window is correlated over "
                   f"{smp['statistical_inefficiency']:.0f} iterations); {smp['threshold']:g} are recommended for MBAR "
                   f"(Mey et al. 2020). The free energy and its uncertainty rest on these samples. A longer production "
                   f"gives more: {longer}")
    if a.get("delta_g_error_note"):
        out.append("the standard error of this leg is the analytical one of MBAR and the tail correction, which takes "
                   f"the samples as independent of each other ({a['delta_g_error_note']}). A longer production allows "
                   f"the error from time blocks: {longer}")
    if a.get("parts_not_determined"):
        shown = a["parts_not_determined"]
        out.append(f"parts of the production without an estimate ({len(shown)}; they are not shown in analysis.png): "
                   + "; ".join(shown[:3]) + ("; ..." if len(shown) > 3 else ""))
    if a.get("parts_below_level"):
        shown = a["parts_below_level"]
        out.append(f"parts of the production whose lambda windows overlap below the level of a reliable estimate "
                   f"({len(shown)}; their estimates are shown in analysis.png and rest on too little overlap): "
                   + "; ".join(shown[:3]) + ("; ..." if len(shown) > 3 else ""))
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
#: dispersion tail, and only if the ligand they decouple ends as the same molecule: with the
#: electrostatics inside it kept in both, or annihilated in both. Backend, time step, number of
#: lambda windows, lengths, REST2, the soft-core (which acts between the ends only) and the kind
#: of barostat may differ between the legs and are not listed.
CYCLE_CONDITIONS = (("temperature_K", "temperature (K)"), ("pressure_bar", "pressure (bar)"),
                    ("prepared_systems_sha256", "prepared systems (input, force field, solvation)"),
                    ("lrc_method", "correction for the dispersion tail"),
                    ("lrc_long_cutoff_nm", "outer radius of the tail correction (nm)"),
                    ("annihilate_electrostatics", "electrostatics inside the ligand annihilated"))
#: what a result that does not record a condition was made with: the only treatment there was
#: before the condition existed. Such a result is compared by it, not left uncompared (a leg
#: with annihilated electrostatics and an earlier result of the other leg would else be combined
#: with a warning)
CYCLE_CONDITION_DEFAULTS = {"annihilate_electrostatics": False}
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
    # the result in place stays valid until the new one is complete: it is copied aside, and the new
    # one replaces it in one step (review of PR 1: moved aside first, a failed write left the leg with
    # no result, and the next run read it as not finished)
    publish(leg_dir / "result.replaced.json", lambda tmp: shutil.copy2(leg_dir / "result.json", tmp))
    _write_json(leg_dir / "result.json", result)
    log(f"{where}: the result of this leg was of another analysis ({analysis_differences(data, analysis)}); it has "
        "been made again from the samples, and the one it replaces is kept as result.replaced.json")
    return "reanalysed"


def analysis_differences(result: dict, analysis) -> str:
    """How the analysis behind a leg's result differs from the one of this run, in words ("" if not)."""
    return record_differences(result.get("analysis_record"), analysis_record(analysis))


def record_differences(stored: dict | None, current: dict) -> str:
    """How the analysis record ``stored`` of a leg's result differs from ``current``, the record of
    the run that combines it, in words ("" if not)."""
    if stored == current:
        return ""
    if not stored:
        return f"made before the version of the analysis was recorded; this run uses version {current.get('version')}"
    if stored.get("version") != current.get("version"):
        return f"version {stored.get('version')} of the analysis; this run uses version {current['version']}"
    return "; ".join(settings_differences({k: v for k, v in stored.items() if k != "version"},
                                          {k: v for k, v in current.items() if k != "version"}))


def samples_analysis_problems(leg_dir: Path, analysis) -> list[str]:
    """Why the samples of a leg directory cannot give a result with the settings ``analysis``,
    found by the functions of the analysis itself (:func:`analysis_problems`, then the production
    from the stored boundary to the stored end and the tail correction of it) without solving
    MBAR; empty if they can."""
    from cufep.analysis.mbar import MIN_INDEPENDENT_SAMPLES, NoCorrection, tail_series
    from cufep.sampling.storage import equilibration_boundary, read_samples

    samples = read_samples(leg_dir / "hrex")
    attrs = samples["attrs"]
    problems = analysis_problems(attrs, analysis)
    if problems:
        return problems
    boundary = equilibration_boundary(attrs, leg_dir)
    if boundary is None:
        return ["they do not record where their production begins"]
    boundary = production_start(samples, boundary, analysis)[0]
    n = int(samples["u"].shape[0]) - boundary
    if n < MIN_INDEPENDENT_SAMPLES:
        return [f"they hold {max(0, n)} iterations of production"]
    a = getattr(analysis, "analysis", analysis)
    try:
        series = tail_series(samples, a.lrc, boundary, n, float(attrs.get("temperature_K", 298.15)))
    except NoCorrection as exc:
        return [str(exc)]
    why = series.estimate(np.ones(series.values.size, dtype=bool))[3]
    return [f"the tail correction of their production cannot be estimated: {why}"] if why else []


def with_cycle_conditions(leg_dir: Path, data: dict) -> dict:
    """The result ``data`` of a leg with the condition of the cycle that it lacks and that its
    samples record: whether the electrostatics inside the ligand were annihilated.

    A result has the key since the setting exists; one written without it is of a version that
    had only the electrostatics kept, unless that version analysed again the samples of a later
    one (review of 2861b93: such a result of annihilated samples was read as "kept", and combined
    with a leg that had kept them). So the samples are asked where the result does not say, by
    the check before a run and by the combination alike; a leg whose samples are gone, or record
    no conditions, was made with the electrostatics kept (:data:`CYCLE_CONDITION_DEFAULTS`)."""
    from cufep.sampling.storage import StorageError, stored_conditions

    if data.get("annihilate_electrostatics") is not None:
        return data
    try:
        n_stored, conditions = stored_conditions(Path(leg_dir) / "hrex")
    except StorageError:
        return data
    if n_stored == 0 or conditions is None:
        return data
    return dict(data, annihilate_electrostatics=electrostatics_annihilated(conditions))


def combined_record(leg_dir: Path, data: dict, cfg: Config) -> tuple[dict, list[str]]:
    """What a finished leg that a run does not sample will be combined with at the end of that run,
    and why it could not be: its result as it is, if it was analysed as this run analyses or its
    samples are gone; else the result its samples will give when they are analysed again
    (:func:`refresh_leg_result`): temperature and pressure of the samples, the correction of this
    run, the outer radius the samples recorded."""
    from cufep.sampling.storage import stored_attributes

    n_stored, attrs = stored_attributes(leg_dir / "hrex")
    if data.get("analysis_record") == analysis_record(cfg) or n_stored == 0:
        return with_cycle_conditions(leg_dir, data), []
    conditions = attrs.get("conditions") if isinstance(attrs.get("conditions"), dict) else {}
    record = {key: data.get(key) for key, _name in CYCLE_CONDITIONS}
    record.update(temperature_K=attrs.get("temperature_K"), pressure_bar=pressure_of_samples(conditions),
                  lrc_method=cfg.analysis.lrc, lrc_long_cutoff_nm=recorded_tail_radius(attrs),
                  annihilate_electrostatics=electrostatics_annihilated(conditions))
    return record, samples_analysis_problems(leg_dir, cfg)


def check_partner_legs(lig_dir: Path, ligand_name: str, selected: set, repeats: int, cfg: Config,
                       prep_info: dict | None = None) -> None:
    """Before a leg is sampled: the finished legs it will be combined with and that this run does
    not sample must be legs of the same thermodynamic cycle as the ones this run makes.

    Each of them is taken as it will be combined (:func:`combined_record`: its result, or what its
    samples give when they are analysed again), and compared with what the legs of this run will
    record by the rule of the combination itself (:func:`cycle_differences`); samples that cannot
    be analysed with the settings of this run are named too. ``selected`` holds the (repeat, leg)
    of this ligand that the run samples, ``prep_info`` the record of its prepared systems if they
    exist. Raises WorkflowError with what differs, before any time is spent on a leg whose result
    could not be used.
    """
    own = {
        "temperature_K": float(cfg.thermodynamics.temperature_K),
        "pressure_bar": float(cfg.thermodynamics.pressure_bar),
        "prepared_systems_sha256": record_digest((prep_info or {}).get("made_from")),
        "lrc_method": cfg.analysis.lrc,
        "lrc_long_cutoff_nm": float(cfg.analysis.lrc_cutoff_nm),
        "annihilate_electrostatics": bool(cfg.alchemy.annihilate_electrostatics),
    }
    legs, differing = [("the legs this run samples", own)], []
    for r in range(repeats):
        for leg in ("complex", "solvent"):
            if (r, leg) in selected:
                continue
            leg_dir = lig_dir / f"repeat_{r}" / leg
            state, data = leg_state(leg_dir)
            if state != "finished":
                continue
            where = f"the {leg} leg of repeat {r} (finished, not part of this run)"
            record, problems = combined_record(leg_dir, data, cfg)
            legs.append((where, record))
            differing += [f"{where} has samples that cannot be analysed with the settings of this run: {problem}"
                          for problem in problems]
    differing = cycle_differences(legs)[0] + differing
    if differing and selected and len(legs) > 1:
        raise WorkflowError(
            f"{ligand_name}: the legs this run would sample could not be combined with the legs the ligand already "
            f"has ({'; '.join(differing)}). Nothing was run. Use the settings of the finished legs, or run all "
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
        values = [(where, result.get(key) if result.get(key) is not None else CYCLE_CONDITION_DEFAULTS.get(key))
                  for where, result in concerned]
        known = [(where, value) for where, value in values if value is not None]
        missing = [where for where, value in values if value is None]
        if any(not _same(value, known[0][1]) for _where, value in known[1:]):
            differing.append(f"{name}: " + ", ".join(f"{where} {value}" for where, value in known))
        if missing:
            not_compared.append(f"{name}: not recorded for the {', the '.join(missing)}")
    return differing, not_compared


def leg_digest(leg_dir: Path) -> str | None:
    """A digest of what a leg directory holds for the combination (:func:`leg_state`): the state of
    the leg and its result or error. Two digests differ whenever the leg would be combined otherwise."""
    state, data = leg_state(Path(leg_dir))
    return record_digest({"state": state, "data": data})


def combine_ligand(lig_dir: Path, ligand_name: str, repeats: int, analysis=None, publish: bool = True,
                   record: dict | None = None) -> dict:
    """Binding free energy from the finished legs of one ligand.

    The legs of all complete repeats must have been made under the same conditions
    (:data:`CYCLE_CONDITIONS`); if they were not, the ligand gets no binding free energy and the
    result says what differs. ``analysis`` is the ``analysis`` section of the run: a leg whose
    result is of another analysis (:func:`refresh_leg_result` could not make it again) is combined
    as it is and named under ``old_analysis`` with the checks its result does not hold. ``record``
    is an analysis record (:func:`analysis_record`) in place of ``analysis``: the one of the run
    that combined a ligand before (:func:`stored_ligand_result`).

    The result records what it was made of: the repeats it covers (``repeats_planned``), the
    analysis record of the run that combined it, and a digest of every leg (:func:`leg_digest`),
    so that a later run can see whether the legs are still the ones it was combined from. It is
    written to ``<ligand>/result.json`` unless ``publish`` is False (a ligand that is not part of
    a run is combined again without being written to).
    """
    run_record = record if record is not None else (analysis_record(analysis) if analysis is not None else None)
    digests = {f"{r}/{leg}": leg_digest(lig_dir / f"repeat_{r}" / leg)
               for r in range(repeats) for leg in ("complex", "solvent")}
    per_repeat = []
    complete: list[tuple[str, dict]] = []      # the legs of the complete repeats
    warnings: list[str] = []
    # legs whose analysis does not meet a level of cufep.analysis.mbar.quality_checks; the checks and their
    # messages are made again from the numbers of every leg with the levels of the run that combines, so that
    # a verdict of an earlier version (of a check that is no longer made) never reaches the result
    from cufep.analysis.mbar import QUALITY_CHECKS

    levels = analysis_settings_of(run_record) if run_record is not None else analysis_settings_of(None)
    flagged: dict[str, list[str]] = {check: [] for check in QUALITY_CHECKS}
    overlap_places: list[dict] = []            # where the legs flagged NOT RELIABLE are below the level
    failed_legs: list[str] = []
    failed_leg_ids: list[list] = []        # [repeat, leg] of every failed leg and every leg without a free energy
    for r in range(repeats):
        states = {leg: leg_state(lig_dir / f"repeat_{r}" / leg) for leg in ("complex", "solvent")}
        for leg, (state, data) in states.items():
            if state == "failed":
                where = f"{leg} leg" + (f" of repeat {r}" if repeats > 1 else "")
                failed_legs.append(where)
                failed_leg_ids.append([r, leg])
                kept = " The result this leg had before is kept in result.previous.json and is not used." \
                    if data.get("previous_result") else ""
                warnings.append(f"repeat {r} {leg}: FAILED: {str(data.get('error', ''))[:300]}.{kept}")
        for leg, (state, data) in states.items():
            if state != "finished":
                continue
            # a finished leg whose analysis gave no number (windows without any overlap), or no standard
            # error of the number it reports (version 5 on: delta_g_error)
            needed = ("delta_g_decouple", "delta_g_error") if "delta_g_error" in data else \
                ("delta_g_decouple", "delta_g_mbar_error")
            if not all(isinstance(data.get(key), (int, float)) and np.isfinite(data[key]) for key in needed):
                where = f"{leg} leg" + (f" of repeat {r}" if repeats > 1 else "")
                failed_legs.append(where)
                failed_leg_ids.append([r, leg])
                warnings.append(f"repeat {r} {leg}: FAILED: the analysis gave no free energy for this leg")
                states[leg] = ("failed", data)
        if not all(state == "finished" for state, _ in states.values()):
            warnings.append(f"repeat {r} is incomplete")
            continue
        c, s = states["complex"][1], states["solvent"][1]
        complete += [(f"{leg} leg" + (f" of repeat {r}" if repeats > 1 else ""),
                      with_cycle_conditions(lig_dir / f"repeat_{r}" / leg, states[leg][1]))
                     for leg in ("complex", "solvent")]
        dg = s["delta_g_decouple"] - c["delta_g_decouple"] - c["restraint_release"]
        # the standard error of a leg (from time blocks since version 4 of the analysis; a result of
        # an earlier version holds MBAR's alone) and the one MBAR gives
        reported = [
            leg["delta_g_error"] if isinstance(leg.get("delta_g_error"), (int, float)) else leg["delta_g_mbar_error"]
            for leg in (c, s)
        ]
        # MBAR's own error is shown next to the reported one; a leg without it (pymbar gave none) is
        # combined all the same, and the ligand has none (review of PR 1: np.hypot(None, ...) stopped it)
        own = [leg.get("delta_g_mbar_error") for leg in (c, s)]
        own = [float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None for x in own]
        per_repeat.append(
            {
                "repeat": r,
                "dg_bind": dg,
                "error": float(np.hypot(*reported)),
                "mbar_error": float(np.hypot(*own)) if None not in own else None,
                "error_from_time_blocks": all(error_from_time_blocks(leg) for leg in (c, s)),
                "dg_complex": c["delta_g_decouple"],
                "dg_solvent": s["delta_g_decouple"],
                "restraint_release": c["restraint_release"],
                # what a restraint chosen from interactions rests on (None for another selection, and
                # for the result of a leg that an earlier version made)
                "restraint_choice": c.get("restraint_choice"),
                "lrc_complex": c["lrc"],
                "lrc_solvent": s["lrc"],
                "backend_complex": c["backend"],
                "backend_solvent": s["backend"],
                "stage_complex": c["analysis"]["stage_delta_g"],
                "stage_solvent": s["analysis"]["stage_delta_g"],
                "lambda_windows": {"complex": c["n_states"], "solvent": s["n_states"]},
                "lambda_windows_added": {
                    "complex": c.get("lambda_windows_added"),
                    "solvent": s.get("lambda_windows_added"),
                },
                "min_overlap": {"complex": c["analysis"]["min_overlap"], "solvent": s["analysis"]["min_overlap"]},
                # among the samples the free energy is estimated from (None in results that lack it)
                "min_overlap_used": {
                    "complex": c["analysis"].get("min_overlap_used"),
                    "solvent": s["analysis"].get("min_overlap_used"),
                },
                "production_ns": {"complex": c["production_ns"], "solvent": s["production_ns"]},
            }
        )
        for leg_result in (c, s):
            notes = [n for n in leg_result.get("notes", []) if not str(n).startswith(_ANALYSIS_NOTES)]
            notes += quality_warnings(leg_result, levels)
            warnings += [f"repeat {r} {leg_result['leg']}: {n}" for n in notes]
            where = f"{leg_result['leg']} leg" + (f" of repeat {r}" if repeats > 1 else "")
            quality = leg_quality(leg_result["analysis"], levels)
            for check, legs in flagged.items():
                if (quality.get(check) or {}).get("ok") is False:
                    legs.append(where)
            overlap_places += [dict(leg=where, **place) for place in overlap_below_level(leg_result, levels)]
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
    if run_record is not None:
        for where, result in complete:
            differs = record_differences(result.get("analysis_record"), run_record)
            if differs:
                # what the numbers of such a result cannot decide (it lacks those of the samples)
                quality = leg_quality(result["analysis"], levels)
                not_made = [] if quality["samples"]["ok"] is not None else ["independent samples"]
                old_analysis.append(
                    f"{where}: {differs}" + (f"; not checked: {', '.join(not_made)}" if not_made else "")
                )
        warnings += [
            f"the result of the {item}. Its free energy and uncertainty are combined as they are"
            for item in old_analysis
        ]
    # the version of the analysis behind every leg that is combined (None: not recorded)
    versions = sorted({(result.get("analysis_record") or {}).get("version") for _where, result in complete},
                      key=lambda v: -1 if v is None else v)
    out: dict = {
        "ligand": ligand_name,
        "units": "kcal/mol",
        "n_repeats": len(per_repeat),
        "warnings": warnings,
        "analysis_versions": versions,
        "failed_legs": failed_legs,
        "failed_leg_ids": failed_leg_ids,
        "inconsistent_legs": inconsistent,
        "not_compared": not_compared,
        "origin_not_recorded": unrecorded,
        "old_analysis": old_analysis,
        "not_reliable": flagged["overlap"],
        "overlap_below_level": overlap_places,
        "few_samples": flagged["samples"],
        "repeats_planned": int(repeats),
        "analysis_record": run_record,
        "leg_digests": digests,
        # the conditions of the cycle that this combination compared among its legs
        "cycle_conditions": [key for key, _name in CYCLE_CONDITIONS],
    }
    if per_repeat:
        vals = np.array([p["dg_bind"] for p in per_repeat]) / KCAL
        out["dg_bind"] = float(vals.mean())
        out["dg_bind_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        # the standard error of one calculation (root mean square over the repeats, if there are several),
        # and whether it is from time blocks in every leg
        out["dg_bind_error"] = float(np.sqrt(np.mean([(p["error"] / KCAL) ** 2 for p in per_repeat])))
        out["dg_bind_error_from_time_blocks"] = all(p["error_from_time_blocks"] for p in per_repeat)
        out["dg_bind_mbar_error"] = (float(np.sqrt(np.mean([(p["mbar_error"] / KCAL) ** 2 for p in per_repeat])))
                                     if all(p["mbar_error"] is not None for p in per_repeat) else None)
        out["min_overlap"] = float(min(min(p["min_overlap"].values()) for p in per_repeat))
        used = [v for p in per_repeat for v in p["min_overlap_used"].values() if v is not None]
        out["min_overlap_used"] = float(min(used)) if used else None
        energy_keys = ("dg_bind", "error", "mbar_error", "dg_complex", "dg_solvent", "restraint_release", "lrc_complex",
                       "lrc_solvent")
        converted = []
        for p in per_repeat:
            q = dict(p)
            for k in energy_keys:
                q[k] = None if p[k] is None else p[k] / KCAL
            for k in ("stage_complex", "stage_solvent"):
                q[k] = {name: value / KCAL for name, value in p[k].items()}
            converted.append(q)
        out["repeats"] = converted
    if publish:
        publish_text(lig_dir / "result.json", jsonio.dumps(out, indent=1))
    return out


def stored_ligand_result(lig_dir: Path, name: str) -> dict:
    """The result of a ligand that is not part of this run (``--ligand``), as its own legs give it.

    The ligand keeps the repeats and the analysis settings of the run that combined it: a run of
    one ligand with one repeat must not combine another ligand of three repeats as one (review of
    PR 1). Its ``result.json`` is used as it is while the legs it was made from are unchanged. A
    leg can change afterwards (interrupted, run again, analysed again) without the ligand being
    combined again; then the ligand is combined again here, read-only, over its own repeats and
    with the analysis record of the run that combined it, and the result says so (review of PR 1:
    the mean of three repeats and success were shown after the last complex leg had been
    interrupted). Nothing in the ligand's directory is written. A result of an earlier version
    records neither its repeats nor its legs: the repeats are those it names, and it is combined
    again and kept if that gives the same numbers.

    The verdicts of a stored result are those of the version of the analysis that combined it;
    of another version (one that judged the halves or BAR, for example) it is combined again too,
    read-only, with its repeats and analysis settings, so that only the checks of this version
    reach the summary (review of 7b49066: a ligand combined by analysis version 8 kept its
    verdicts on the halves and BAR in a run of another ligand).
    """
    from cufep.analysis.mbar import ANALYSIS_VERSION

    def verdicts_current(result: dict) -> bool:
        # of this version of the analysis, and combined under the conditions of the cycle this
        # version compares (review of 2861b93: a ligand that a version before
        # annihilate_electrostatics had combined from legs that differ in it was shown again as it was)
        return ((result.get("analysis_record") or {}).get("version") == ANALYSIS_VERSION
                and result.get("cycle_conditions") == [key for key, _name in CYCLE_CONDITIONS])

    lig_dir = Path(lig_dir)
    path = lig_dir / "result.json"
    stored = _read_json(path) if path.exists() else None
    present = 0
    while (lig_dir / f"repeat_{present}").is_dir():
        present += 1
    if stored is None:
        if present == 0:
            return {"ligand": name, "units": "kcal/mol", "n_repeats": 0, "not_part_of_this_run": True,
                    "repeats_planned": 0,
                    "warnings": ["not part of this run, and no result of an earlier run"]}
        why = ("its result.json cannot be read" if path.exists() else
               "the run that made its legs did not combine them (no result.json of the ligand)")
        out = combine_ligand(lig_dir, name, present, publish=False)
        out["warnings"].insert(0, f"not part of this run; {why}: shown as its legs give it now, over the "
                                  f"{present} repeat(s) it has")
        return dict(out, not_part_of_this_run=True, combined_again_from_legs=True)
    if isinstance(stored.get("repeats_planned"), int) and isinstance(stored.get("leg_digests"), dict):
        repeats, recorded = int(stored["repeats_planned"]), stored["leg_digests"]
        changed = [(r, leg) for r in range(repeats) for leg in ("complex", "solvent")
                   if recorded.get(f"{r}/{leg}") != leg_digest(lig_dir / f"repeat_{r}" / leg)]
        if not changed and verdicts_current(stored):
            return dict(stored, not_part_of_this_run=True)
        out = combine_ligand(lig_dir, name, repeats, publish=False, record=stored.get("analysis_record"))
        if not changed:                         # the same legs; the verdicts of this version
            return dict(out, not_part_of_this_run=True)
        legs = ", ".join(f"the {leg} leg of repeat {r}" for r, leg in changed)
        out["warnings"].insert(0, f"not part of this run; its legs changed after its result was combined ({legs}): "
                                  "shown as its legs give it now, with the repeats and the analysis settings of the "
                                  "run that combined it; its result.json is unchanged")
        return dict(out, not_part_of_this_run=True, combined_again_from_legs=True)
    # a result of an earlier version: its repeats are those it names
    repeats = _repeats_named(stored)
    repeats = present if repeats is None else repeats
    out = combine_ligand(lig_dir, name, repeats, publish=False)
    if _same_combination(stored, out):
        # the same numbers: the stored result, unless its verdicts are of another version
        return dict(stored if verdicts_current(stored) else out, not_part_of_this_run=True, repeats_planned=repeats)
    out["warnings"].insert(0, f"not part of this run; its legs no longer give the result it has (made by an earlier "
                              f"version of cuFEP): shown as its legs give it now, over the {repeats} repeat(s) its "
                              "result names; its result.json is unchanged")
    return dict(out, not_part_of_this_run=True, combined_again_from_legs=True)


def _repeats_named(stored: dict) -> int | None:
    """How many repeats a ligand result of an earlier version covers: one more than the highest
    repeat it names (in its repeats, its failed legs or its warnings), None if it names none."""
    import re

    named = [int(p["repeat"]) for p in stored.get("repeats") or [] if isinstance(p, dict) and "repeat" in p]
    named += [int(item[0]) for item in stored.get("failed_leg_ids") or [] if isinstance(item, (list, tuple)) and item]
    for warning in stored.get("warnings") or []:
        match = re.match(r"repeat (\d+) ", str(warning))
        if match:
            named.append(int(match.group(1)))
    return max(named) + 1 if named else None


def _same_combination(stored: dict, again: dict) -> bool:
    """Whether a ligand result of an earlier version says what its legs give now: the number of
    complete repeats, the binding free energy, the free energies of every repeat, and the failed
    and inconsistent legs where it records them."""
    def rounded(x):
        if isinstance(x, float):
            return None if not np.isfinite(x) else float(f"{x:.9g}")
        return x

    def core(result: dict, keys) -> dict:
        out = {"n_repeats": result.get("n_repeats"), "dg_bind": rounded(result.get("dg_bind")),
               "repeats": [tuple(rounded(p.get(k)) for k in ("repeat", "dg_bind", "dg_complex", "dg_solvent",
                                                             "restraint_release"))
                           for p in result.get("repeats") or []]}
        if "failed_leg_ids" in keys:
            out["failed_leg_ids"] = sorted([int(r), str(leg)] for r, leg in result.get("failed_leg_ids") or [])
        if "inconsistent_legs" in keys:
            out["inconsistent_legs"] = list(result.get("inconsistent_legs") or [])
        return out

    keys = set(stored)
    return core(stored, keys) == core(again, keys)


def pressure_of_samples(conditions: dict) -> float | None:
    """The pressure of the samples of a leg, from their recorded conditions: the pressure of its
    barostat, or, for a leg without one because it makes grand canonical moves of water, the
    pressure at which the cell it is fixed in was equilibrated. None if they do not say."""
    pressure = conditions.get("pressure_bar")
    settings = conditions.get("settings") if isinstance(conditions.get("settings"), dict) else {}
    if pressure is None and isinstance(settings.get("gcmc"), dict):
        pressure = settings["gcmc"].get("cell_pressure_bar")
    return pressure


def gcmc_numbers(samples: dict, start: int) -> dict | None:
    """The grand canonical moves of water of the iterations from ``start`` on, by lambda window, as
    numbers without a verdict: the trials of an insertion and of a deletion and how many were made,
    and the number of waters in the sphere after the moves (its mean, lowest and highest value over
    the iterations that had a move). None for samples of a leg without the moves."""
    if "gcmc" not in samples:
        return None
    from cufep.sampling.gcmc import COUNTS

    data = np.asarray(samples["gcmc"][int(start):], dtype=np.int64)
    out = {name: [int(n) for n in data[:, :, column].sum(axis=0)] for column, name in enumerate(COUNTS[:4])}
    waters = np.where(data[:, :, 4] >= 0, data[:, :, 4], -1).astype(float)
    waters[waters < 0] = np.nan
    moved = np.isfinite(waters).any(axis=0)

    def per_window(function) -> list:
        return [float(function(waters[:, k][np.isfinite(waters[:, k])])) if moved[k] else None
                for k in range(waters.shape[1])]

    out[COUNTS[4]] = {"mean": per_window(np.mean), "lowest": per_window(np.min), "highest": per_window(np.max)}
    return out


def relocation_numbers(samples: dict, start: int) -> dict | None:
    """The water relocation moves of the iterations from ``start`` on, by lambda window, as numbers
    without a verdict: attempted and accepted, into the sphere and out of it. None for samples of
    a leg without the moves."""
    if "relocations" not in samples:
        return None
    counts = np.asarray(samples["relocations"][int(start):], dtype=np.int64).sum(axis=0)
    names = ("attempted_into_the_sphere", "accepted_into_the_sphere", "attempted_out_of_the_sphere",
             "accepted_out_of_the_sphere")
    return {name: [int(n) for n in counts[:, column]] for column, name in enumerate(names)}
