"""Preparation of an edge: the end states A and B of its two legs, in one environment, and the mapping.

The complex leg: the protein with ligand A in its pose, waters of the input that overlap a heavy atom of A or B
removed (the ABFE removes those of its one ligand), then water and ions, or the POPC bilayer
(``membrane.enabled``), as the ABFE builds them (``cufep.workflow.prepare_ligand``). The solvent leg: ligand A
in water. System B of a leg is the same atoms with ligand A replaced by ligand B, in the pose of the SDF (the
ligands of a network are given in one frame, as the public benchmark gives them); the hybrid System is made
from the two (:mod:`cufep.rbfe.hybrid`) when the leg runs. OpenFE builds its state B the same way, from the
topology of state A with the ligand replaced.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from cufep.files import publish_text


def edge_name(a: str, b: str) -> str:
    return f"{a}~{b}"


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t with ``source @ R.T + t`` closest to ``target`` (Kabsch)."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, ct - cs @ rotation.T


def _modeller_with_ligand_replaced(modeller, ligand_a, ligand_b):
    """A copy of ``modeller`` in which the ligand residue (LIG, ligand A) is replaced by ``ligand_b``, appended
    last. The building of the system may have moved ligand A (a bilayer turns the complex, water centres the
    solute): ligand B is moved with the same rigid motion, which carries the pose of A in the SDF onto A as it is
    in ``modeller``."""
    import openmm.app as app
    import openmm.unit as unit

    from cufep.prep.ligand import LIGAND_RESNAME
    from cufep.prep.solvate import ligand_atom_indices

    now = np.asarray(modeller.positions.value_in_unit(unit.nanometer))[ligand_atom_indices(modeller.topology)]
    rotation, shift = rigid_transform(ligand_a.positions_nm(), now)
    out = app.Modeller(modeller.topology, modeller.positions)
    out.delete([res for res in out.topology.residues() if res.name == LIGAND_RESNAME])
    top, _pos = ligand_b.to_openmm()
    out.add(top, (ligand_b.positions_nm() @ rotation.T + shift) * unit.nanometer)
    return out


def _write_end_state(prep_dir: Path, name: str, modeller, system, ligand_atoms: list[int]) -> None:
    from cufep.workflow import _write_system

    _write_system(prep_dir, name, modeller, system, ligand_atoms)


def system_mapping(mapping: dict[int, int], ligand_a_atoms: list[int], ligand_b_atoms: list[int]) -> dict[int, int]:
    """The mapping of molecule indices as indices of the Systems (the ligand atoms of a System are the atoms of
    its molecule, in their order)."""
    return {ligand_a_atoms[i]: ligand_b_atoms[j] for i, j in mapping.items()}


def prepare_edge(cfg, ligand_a, ligand_b, protein, edge_dir: Path, box_shape: str, log=print) -> dict:
    """Build and write the end states of both legs of the edge A -> B and its mapping. ``cfg`` is the
    configuration of the run (:class:`cufep.rbfe.config.RbfeConfig`)."""
    from cufep.prep.forcefield import ForceFieldSpec, SystemSpec, create_system, make_forcefield
    from cufep.prep.ligand import assign_partial_charges
    from cufep.prep.solvate import SolvationSpec, combine, ligand_atom_indices, min_box_height_nm, solvate
    from cufep.rbfe.charge import (
        alchemical_water,
        charge_change,
        charge_change_problems,
        ion_parameters,
        oxygen_first,
        water_into_ion,
    )
    from cufep.rbfe.mapping import constrained_bond_rule, kartograf_mapping, lomap_score, mapping_settings
    from cufep.workflow import WorkflowError, _check_box, _remove_clashing_waters

    prep_dir = Path(edge_dir) / "prep"
    done = prep_dir / "prep.json"
    made_from = edge_made_from(cfg, ligand_a, ligand_b, box_shape)
    info = prepared_edge(cfg, ligand_a, ligand_b, edge_dir, box_shape)
    if info is not None:
        return info
    problems = charge_change_problems([ligand_a, ligand_b], [(ligand_a.name, ligand_b.name)])
    if problems:
        raise WorkflowError("; ".join(problems))
    # charged ligands are taken (an edge between two ligands of one charge, or a change of one charge with an
    # alchemical water); the ABFE stops on them (cufep.workflow.check_ligand)
    dq = charge_change(ligand_a, ligand_b)
    prep_dir.mkdir(parents=True, exist_ok=True)
    for lig in (ligand_a, ligand_b):
        assign_partial_charges(lig, cfg.forcefield.ligand_charges, cache_dir=Path(cfg.output.dir) / "cache")
    m = cfg.mapping
    settings = mapping_settings(m)
    mapping = kartograf_mapping(ligand_a, ligand_b, settings)
    score = lomap_score(ligand_a, ligand_b, mapping)
    ffspec = ForceFieldSpec(protein=cfg.forcefield.protein, water=cfg.forcefield.water, lipid=cfg.forcefield.lipid,
                            ligand=cfg.forcefield.ligand)
    membrane = bool(cfg.membrane.enabled)
    # each end state with its own ligand alone: with both in one force field OpenMM matches a ligand of the same
    # bonded graph (E/Z isomers, another pose of one molecule) to the template of the other, and its System gets
    # the charges of the other (review of PR #17). The system is built (solvent, bilayer) with ligand A
    forcefield = make_forcefield(ffspec, [ligand_a], include_lipid=membrane)
    forcefield_b = make_forcefield(ffspec, [ligand_b], include_lipid=membrane)
    sysspec = SystemSpec(cutoff_nm=cfg.dynamics.cutoff_nm, hydrogen_mass_amu=cfg.dynamics.hmr_mass_amu,
                         constraints="hbonds" if cfg.dynamics.constraints in ("auto", "hbonds") else "none")
    plugin = cfg.backend.name == "cupotential"
    min_edge = min_box_height_nm(cfg.dynamics.cutoff_nm, cfg.backend.name)
    solvent_shape = "cube" if plugin else box_shape
    ionic = cfg.thermodynamics.ionic_strength_M
    heavy = np.vstack([ligand_a.positions_nm()[ligand_a.heavy_atom_indices()],
                       ligand_b.positions_nm()[ligand_b.heavy_atom_indices()]])
    notes: list[str] = []

    def build(with_protein: bool) -> tuple:
        modeller = combine((protein.topology, protein.positions) if with_protein else None, ligand_a)
        membrane_info: dict = {}
        if with_protein:
            n_removed = _remove_clashing_waters(modeller, heavy)
            if n_removed:
                notes.append(f"{n_removed} water(s) of the input overlapping ligand {ligand_a.name} or "
                             f"{ligand_b.name} were removed")
            if membrane:
                from cufep.prep.membrane import add_membrane

                # a water of more than three sites: the bilayer is built with the 3-site water of the lipid patch,
                # in a force field that add_membrane asks for (review of PR #17: none was given, TypeError)
                membrane_info = add_membrane(
                    modeller, forcefield, cfg, log, water=cfg.forcefield.water,
                    forcefield_with_water=lambda file: make_forcefield(replace(ffspec, water=file), [ligand_a],
                                                                       include_lipid=True))
                notes.extend(f"membrane: {w}" for w in membrane_info.pop("warnings", []))
            else:
                solvate(modeller, forcefield, SolvationSpec(padding_nm=cfg.solvation.padding_complex_nm,
                                                            box_shape=box_shape, water=cfg.forcefield.water,
                                                            ionic_strength_M=ionic, min_edge_nm=min_edge))
        else:
            solvate(modeller, forcefield, SolvationSpec(padding_nm=cfg.solvation.padding_solvent_nm,
                                                        box_shape=solvent_shape, water=cfg.forcefield.water,
                                                        ionic_strength_M=ionic, min_edge_nm=min_edge))
        what = f"{ligand_a.name} -> {ligand_b.name}: the {'complex' if with_protein else 'solvent'} leg"
        _check_box(modeller, cfg.dynamics.cutoff_nm, what)
        modeller_b = _modeller_with_ligand_replaced(modeller, ligand_a, ligand_b)
        system_a = create_system(forcefield, modeller.topology, sysspec)
        system_b = create_system(forcefield_b, modeller_b.topology, sysspec)
        water = None
        if dq != 0:
            # the water that is an ion in B (cufep.rbfe.charge), chosen in A, the same atoms in B
            import openmm.unit as unit

            positions = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
            cutoff = cfg.alchemy.alchemical_water_cutoff_nm
            atoms_a = oxygen_first(modeller.topology, alchemical_water(modeller.topology, positions, cutoff))
            in_a, in_b = set(ligand_atom_indices(modeller.topology)), set(ligand_atom_indices(modeller_b.topology))
            env_a = [i for i in range(system_a.getNumParticles()) if i not in in_a]
            env_b = [i for i in range(system_b.getNumParticles()) if i not in in_b]
            to_b = dict(zip(env_a, env_b))
            ion = ion_parameters(system_a, modeller.topology, int(np.sign(dq)), forcefield)
            atoms_b = [to_b[i] for i in atoms_a]
            water_into_ion(system_b, atoms_b, ion)
            water = {"atoms_a": atoms_a, "atoms_b": atoms_b, "ion": ion, "charge_change": dq,
                     "cutoff_nm": float(cfg.alchemy.alchemical_water_cutoff_nm)}
        return modeller, system_a, modeller_b, system_b, membrane_info, water

    info = {"ligand_a": ligand_a.name, "ligand_b": ligand_b.name, "lomap_score": score, "made_from": made_from,
            "membrane": membrane, "box_shape": box_shape, "solvent_box_shape": solvent_shape,
            "box_built_for_backend": cfg.backend.name, "input_chains": dict(protein.input_chains)}
    legs = {}
    for leg, with_protein in (("complex", True), ("solvent", False)):
        modeller_a, system_a, modeller_b, system_b, membrane_info, water = build(with_protein)
        atoms_a, atoms_b = ligand_atom_indices(modeller_a.topology), ligand_atom_indices(modeller_b.topology)
        hydrogens_a = {i for i, a in enumerate(ligand_a.offmol.atoms) if a.atomic_number == 1}
        hydrogens_b = {i for i, a in enumerate(ligand_b.offmol.atoms) if a.atomic_number == 1}
        kept = constrained_bond_rule(mapping, system_a, system_b, dict(enumerate(atoms_a)), dict(enumerate(atoms_b)),
                                     hydrogens_a, hydrogens_b)
        _write_end_state(prep_dir, f"{leg}_a", modeller_a, system_a, atoms_a)
        _write_end_state(prep_dir, f"{leg}_b", modeller_b, system_b, atoms_b)
        legs[leg] = {"n_atoms_a": system_a.getNumParticles(), "n_atoms_b": system_b.getNumParticles(),
                     "core_atoms": len(kept), "mapping": {str(i): j for i, j in sorted(kept.items())},
                     "barostat": "membrane" if (with_protein and membrane) else "isotropic",
                     "membrane_info": membrane_info}
        if water is not None:
            legs[leg]["alchemical_water"] = water
            notes.append(f"{leg} leg: the net charge changes by {-dq:+d}; water {water['atoms_a'][0]} is "
                         f"{water['ion']['name']} in {ligand_b.name}'s end state (an alchemical water)")
        removed = sorted(set(mapping) - set(kept))
        if removed:
            notes.append(f"{leg} leg: {len(removed)} hydrogen(s) left the mapping because a constrained bond changes "
                         "(OpenFE's rule)")
    info["legs"] = legs
    info["mapping_kartograf"] = {str(i): j for i, j in sorted(mapping.items())}
    info["notes"] = notes
    publish_text(prep_dir / "ligand_a.json", ligand_a.offmol.to_json())
    publish_text(prep_dir / "ligand_b.json", ligand_b.offmol.to_json())
    publish_text(done, json.dumps(info, indent=1))       # last: the mark that the preparation is complete
    log(f"{ligand_a.name} -> {ligand_b.name}: {len(mapping)} atoms mapped (LOMAP score {score:.2f}); complex "
        f"{legs['complex']['n_atoms_a']} atoms, solvent {legs['solvent']['n_atoms_a']}"
        + (f"; {'; '.join(notes)}" if notes else ""))
    return info


#: rules of the preparation that changed after relative calculations had been prepared with them: the version of each
#: rule is recorded for the edges it acts on (only there, so that the records of the other edges are those of earlier
#: versions), and an edge prepared by an earlier rule is not taken up (second review of PR #19)
PREPARATION_RULES = {
    # 2: the alchemical water of an edge that changes the net charge is chosen by the distances across the periodic
    #    cell (version 1 took the distances in the cell alone: a water 0.2 nm from the solutes across its boundary
    #    could be taken)
    "alchemical_water": 2,
    # 2: a custom core maps its atoms with their chirality (version 1 could map an atom of the other configuration)
    "custom_core": 2,
}


def earlier_rules_of(info: dict) -> list[str]:
    """The rules of :data:`PREPARATION_RULES` by which an edge with the record ``info`` (its prep.json) was prepared
    in an earlier version: the rule acts on the edge and its record does not hold the present version of it. The
    analysis leaves such an edge out as the run refuses it (independent review of PR #19: `cufep rbfe analyze` and
    `cufep benchmark report` took edges that `cufep rbfe run` refuses)."""
    made_from = info.get("made_from") or {}
    out = []
    charge_changes = ("alchemical_water_cutoff_nm" in made_from
                      or any((leg or {}).get("alchemical_water") for leg in (info.get("legs") or {}).values()))
    if charge_changes and made_from.get("alchemical_water_rule") != PREPARATION_RULES["alchemical_water"]:
        out.append("its alchemical water was chosen by the distances in the cell alone, which could take a water near "
                   "the solutes across the boundary of the cell")
    if ((made_from.get("mapping") or {}).get("core_smarts") is not None
            and made_from.get("custom_core_rule") != PREPARATION_RULES["custom_core"]):
        out.append("its custom core was mapped without the chirality of its atoms")
    return out


def prepared_edge(cfg, ligand_a, ligand_b, edge_dir: Path, box_shape: str) -> dict | None:
    """The record of the prepared systems of an edge if they exist and this run can take them, None if there are
    none yet; WorkflowError if they were made from other inputs or settings. Writes nothing: a run checks every edge
    it runs with it before it writes anything (review of PR #17)."""
    from cufep.workflow import WorkflowError

    prep_dir = Path(edge_dir) / "prep"
    done = prep_dir / "prep.json"
    if not done.exists():
        return None
    try:
        info = json.loads(done.read_text())
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"the record of the prepared systems {done} cannot be read ({exc}); remove the directory "
                            "of the edge to prepare it again") from exc
    if not isinstance(info, dict):
        raise WorkflowError(f"the record of the prepared systems {done} is not a record; remove the directory of the "
                            "edge to prepare it again")
    stored, current = info.get("made_from"), edge_made_from(cfg, ligand_a, ligand_b, box_shape)
    if stored != current:
        differing = sorted(k for k in set(stored or {}) | set(current) if (stored or {}).get(k) != current.get(k))
        rules = [k for k in differing if k.endswith("_rule")]
        raise WorkflowError(f"the prepared systems in {prep_dir} were made from other inputs or settings ("
                            + ", ".join(differing) + ")"
                            + (f"; {', '.join(rules)}: they were prepared by an earlier rule of cuFEP" if rules else "")
                            + "; remove the directory of the edge to prepare it again")
    return info


def edge_made_from(cfg, ligand_a, ligand_b, box_shape: str) -> dict:
    """What the prepared systems of an edge are made from: the inputs (digests) and the settings that enter."""
    import hashlib

    digest = hashlib.sha256(Path(cfg.system.protein).read_bytes()).hexdigest()
    return {"protein_sha256": digest, "ligand_a": ligand_a.digest(), "ligand_b": ligand_b.digest(),
            "forcefield": cfg.forcefield.model_dump(mode="json"), "membrane": cfg.membrane.model_dump(mode="json"),
            "solvation": cfg.solvation.model_dump(mode="json"), "box_shape": box_shape,
            # (a custom core only where one is given, so that the records of the other edges are those of earlier
            # versions)
            "mapping": {k: v for k, v in cfg.mapping.model_dump(mode="json").items()
                        if k != "core_smarts" or v is not None}, "backend": cfg.backend.name,
            # the pH sets the protonation states of a protein given without hydrogens (review of PR #17)
            "thermodynamics": {"ionic_strength_M": cfg.thermodynamics.ionic_strength_M, "ph": cfg.thermodynamics.ph},
            "dynamics": {k: getattr(cfg.dynamics, k) for k in ("cutoff_nm", "hmr_mass_amu", "constraints")},
            # an edge that changes the net charge: the rule of its alchemical water, and a custom core: the version
            # of its rule (only there, so that the records of the other edges are those of earlier versions)
            **({"alchemical_water_cutoff_nm": float(cfg.alchemy.alchemical_water_cutoff_nm),
                "alchemical_water_rule": PREPARATION_RULES["alchemical_water"]}
               if ligand_a.formal_charge != ligand_b.formal_charge else {}),
            **({"custom_core_rule": PREPARATION_RULES["custom_core"]} if cfg.mapping.core_smarts is not None else {})}


def load_end_states(edge_dir: Path, leg: str) -> dict:
    """The prepared end states of a leg of an edge, and its mapping in indices of the Systems."""
    from cufep.workflow import load_prepared

    prep_dir = Path(edge_dir) / "prep"
    info = json.loads((prep_dir / "prep.json").read_text())
    a, b = load_prepared(prep_dir, f"{leg}_a"), load_prepared(prep_dir, f"{leg}_b")
    mol_map = {int(i): int(j) for i, j in info["legs"][leg]["mapping"].items()}
    return {"a": a, "b": b, "info": info, "core_map": system_mapping(mol_map, a["ligand"], b["ligand"])}
