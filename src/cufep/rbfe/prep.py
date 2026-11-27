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
    from cufep.rbfe.mapping import (
        bridging_dummies,
        constrained_bond_rule,
        kartograf_mapping,
        lomap_score,
        mapping_settings,
    )
    from cufep.workflow import WorkflowError, _check_box, _remove_clashing_waters

    prep_dir = Path(edge_dir) / "prep"
    done = prep_dir / "prep.json"
    made_from = edge_made_from(cfg, ligand_a, ligand_b, box_shape)
    info = prepared_edge(cfg, ligand_a, ligand_b, edge_dir, box_shape)
    if info is not None:
        restore_ligand_records(prep_dir, info, ligand_a, ligand_b, cfg)
        return info
    problems = charge_change_problems([ligand_a, ligand_b], [(ligand_a.name, ligand_b.name)])
    if problems:
        raise WorkflowError("; ".join(problems))
    # charged ligands are taken (an edge between two ligands of one charge, or a change of one charge with an
    # alchemical water); the ABFE stops on them (cufep.workflow.check_ligand)
    dq = charge_change(ligand_a, ligand_b)
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

    def build(with_protein: bool, edge_ion: dict | None = None) -> tuple:
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
            # one ion for the edge, chosen in the complex leg and taken by the solvent leg: the two legs transform
            # the same particle, so that what it costs in water cancels in the ddG (eleventh review of PR #19: a K+
            # of the complex against the Na+ of the force field in the solvent left their difference in it)
            ion = edge_ion if edge_ion is not None else ion_parameters(system_a, modeller.topology, int(np.sign(dq)),
                                                                       forcefield)
            atoms_b = [to_b[i] for i in atoms_a]
            water_into_ion(system_b, atoms_b, ion)
            water = {"atoms_a": atoms_a, "atoms_b": atoms_b, "ion": ion, "charge_change": dq,
                     "cutoff_nm": float(cfg.alchemy.alchemical_water_cutoff_nm)}
        return modeller, system_a, modeller_b, system_b, membrane_info, water

    info = {"ligand_a": ligand_a.name, "ligand_b": ligand_b.name, "lomap_score": score, "made_from": made_from,
            "membrane": membrane, "box_shape": box_shape, "solvent_box_shape": solvent_shape,
            "box_built_for_backend": cfg.backend.name, "input_chains": dict(protein.input_chains)}
    legs, built = {}, {}
    hydrogens_a = {i for i, a in enumerate(ligand_a.offmol.atoms) if a.atomic_number == 1}
    hydrogens_b = {i for i, a in enumerate(ligand_b.offmol.atoms) if a.atomic_number == 1}
    for leg, with_protein in (("complex", True), ("solvent", False)):
        complex_water = built["complex"][5] if "complex" in built else None
        built[leg] = build(with_protein, (complex_water or {}).get("ion"))
        modeller_a, system_a, modeller_b, system_b = built[leg][:4]
        atoms_a, atoms_b = ligand_atom_indices(modeller_a.topology), ligand_atom_indices(modeller_b.topology)
        kept = constrained_bond_rule(mapping, system_a, system_b, dict(enumerate(atoms_a)), dict(enumerate(atoms_b)),
                                     hydrogens_a, hydrogens_b)
        # the mapping the hybrid is made of, after the rule of the constrained bonds: an atom that left it can join
        # mapped atoms (twelfth review of PR #19: the oxygen of oxirane, mapped onto a hydrogen of ethane, left it
        # and stayed bonded to both carbons). Refused before anything of the edge is written
        bridges = bridging_dummies(ligand_a.to_rdkit(), ligand_b.to_rdkit(), kept)
        if bridges:
            raise WorkflowError(f"{ligand_a.name} -> {ligand_b.name}: the mapping of the {leg} leg cannot be used "
                                f"after the atoms of constrained bonds that change left it: {'; '.join(bridges)} (the "
                                "hybrid would keep their bonds and angles at both ends); map hydrogens on hydrogens "
                                "only (mapping.map_hydrogens_on_hydrogens_only) or give a custom core")
        built[leg] = built[leg] + (atoms_a, atoms_b, kept)
    prep_dir.mkdir(parents=True, exist_ok=True)
    for leg, with_protein in (("complex", True), ("solvent", False)):
        modeller_a, system_a, modeller_b, system_b, membrane_info, water, atoms_a, atoms_b, kept = built[leg]
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
    # 3: its matches are sets of atoms, and a pattern with more than the matches searched is refused (version 2
    #    counted the orders of the atoms of a match one by one, and the search of RDKit could stop before a match of
    #    the ligand: twelfth review of PR #19). The custom core exists from this pull request on: an edge of version 1
    #    or 2 was made by a commit of it and is prepared again
    "custom_core": 3,
}


def _ligand_of_record(prep_dir: Path, end: str, name, digest):
    """The ligand of one end of a prepared edge from its record (``ligand_<end>.json``) where it is there, can be read
    and is the molecule the record of the edge was made from (``digest``); else None."""
    from cufep.prep.ligand import Ligand

    try:
        from openff.toolkit import Molecule

        found = Ligand(name=str(name), offmol=Molecule.from_json((Path(prep_dir) / f"ligand_{end}.json").read_text()))
        same = digest is not None and found.digest() == digest
    except Exception:  # noqa: BLE001 - a record that is missing, cannot be read or has no pose (fifteenth review)
        return None
    return found if same else None


def prepared_ligands(prep_dir: Path, info: dict, inputs=None):
    """The two ligands of a prepared edge (:class:`cufep.prep.ligand.Ligand`, A then B): those of the records of the
    preparation (``ligand_a.json``, ``ligand_b.json``), else the ligands of ``inputs`` (the ligands of the run, or a
    function that reads those of its SDF) of the names of the edge; each the molecule its record was made from (the
    digest of ``made_from``). None where they are not known (thirteenth review of PR #19: a record of the preparation
    without ligand_a.json was taken by the run, which checks its own ligands, and left out by the analysis right
    after; fourteenth review: every reader of the ligands of an edge takes them so)."""
    made_from = info.get("made_from") if isinstance(info, dict) and isinstance(info.get("made_from"), dict) else {}
    out = []
    for end in ("a", "b"):
        name, digest = (info or {}).get(f"ligand_{end}"), made_from.get(f"ligand_{end}")
        found = _ligand_of_record(prep_dir, end, name, digest)
        if found is None:
            candidates = (inputs() if callable(inputs) else inputs) or []
            found = next((lig for lig in candidates if lig.name == name and digest is not None
                          and lig.digest() == digest), None)
        if found is None:
            return None
        out.append(found)
    return tuple(out)


def ligands_of_the_edge(edge_dir: Path, inputs=None):
    """The two ligands of a prepared edge (:func:`prepared_ligands`) from its record; None where they are not known."""
    import json as _json

    try:
        info = _json.loads((Path(edge_dir) / "prep" / "prep.json").read_text())
    except (OSError, ValueError):
        return None
    return prepared_ligands(Path(edge_dir) / "prep", info, inputs) if isinstance(info, dict) else None


def restore_ligand_records(prep_dir: Path, info: dict, ligand_a, ligand_b, cfg) -> None:
    """Write the record of each ligand of a prepared edge that is missing, cannot be read or is not the molecule the
    edge was made from, from the ligand of the run that is (its digest), with its partial charges as the preparation
    gives them: so that the analysis checks what the run checked. A record that is sound is left as it is (fourteenth
    review of PR #19: the record of B, with its charges, was written again without them when that of A was
    missing)."""
    from cufep.prep.ligand import assign_partial_charges

    made_from = info.get("made_from") or {}
    for end, lig in (("a", ligand_a), ("b", ligand_b)):
        digest = made_from.get(f"ligand_{end}")
        if _ligand_of_record(prep_dir, end, info.get(f"ligand_{end}"), digest) is not None or lig.digest() != digest:
            continue
        assign_partial_charges(lig, cfg.forcefield.ligand_charges, cache_dir=Path(cfg.output.dir) / "cache")
        publish_text(Path(prep_dir) / f"ligand_{end}.json", lig.offmol.to_json())


def _is_index(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_atom_number(text) -> bool:
    """A key of a mapping as the preparation writes it: an atom number in ASCII decimal digits (fourteenth review of
    PR #19: "²" passed ``str.isdigit`` and stopped the analysis at ``int``)."""
    return isinstance(text, str) and text.isascii() and text.isdigit()


#: what every preparation records of what an edge was made from (all records since the relative calculations exist)
MADE_FROM_KEYS = ("protein_sha256", "ligand_a", "ligand_b", "forcefield", "membrane", "solvation", "box_shape",
                  "mapping", "backend", "thermodynamics", "dynamics")


def record_problems(info, ligands=None) -> list[str]:
    """What makes the record of a prepared edge (its prep.json) unusable, before anything is read of it: a record of
    what it was made from, the two legs, each a mapping of atoms that is not empty, one to one, of whole numbers within
    the atoms of the two ligands (where they are known), and an alchemical water whose ion has a charge, sigma and
    epsilon. Empty where it is usable. A record of another form is one of these, never an error (thirteenth review of
    PR #19: a record without legs passed, `mapping: {}` and 1.9 were taken, and a leg recorded as a list stopped the
    analysis of every edge)."""
    if not isinstance(info, dict):
        return ["its record is not a record"]
    made_from = info.get("made_from")
    if not isinstance(made_from, dict):
        return ["its record does not say what it was made from"]
    # what every preparation records of its inputs and settings (fourteenth review of PR #19: `made_from: {}` was
    # taken by the analysis, which the run refuses)
    missing = [k for k in MADE_FROM_KEYS if k not in made_from]
    if missing or not all(isinstance(made_from[k], str) and made_from[k] for k in ("ligand_a", "ligand_b")):
        return [f"its record does not say what it was made from ({', '.join(missing) or 'its ligands'})"]
    legs = info.get("legs")
    if not isinstance(legs, dict) or any(leg not in legs for leg in ("complex", "solvent")):
        return ["its record does not have both legs"]
    sizes = None if ligands is None else (ligands[0].n_atoms, ligands[1].n_atoms)
    # an edge that changes the net charge has an alchemical water in both legs, of one charge none (fourteenth review
    # of PR #19: a record with the water of one leg, or without its atoms, was taken)
    dq = None if ligands is None else int(ligands[0].formal_charge) - int(ligands[1].formal_charge)
    out = []
    for leg in ("complex", "solvent"):
        record = legs[leg]
        if not isinstance(record, dict):
            out.append(f"the record of its {leg} leg has no mapping")
            continue
        # the numbers of atoms of the two end states, which the jobs of a run are made with (fourteenth review: a
        # record without them stopped the run after it had written its record)
        if not all(_is_index(record.get(k)) and record.get(k) > 0 for k in ("n_atoms_a", "n_atoms_b")):
            out.append(f"the record of its {leg} leg does not give the numbers of atoms of its end states")
        mapping = record.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            out.append(f"the record of its {leg} leg has no mapping")
            continue
        pairs = list(mapping.items())
        if not all(_is_atom_number(k) and _is_index(v) for k, v in pairs):
            out.append(f"the mapping of its {leg} leg is not of atom numbers")
            continue
        a_atoms, b_atoms = [int(k) for k, _v in pairs], [v for _k, v in pairs]
        if len(set(a_atoms)) != len(a_atoms) or len(set(b_atoms)) != len(b_atoms):
            out.append(f"the mapping of its {leg} leg is not one to one")
        elif sizes is not None and (max(a_atoms) >= sizes[0] or max(b_atoms) >= sizes[1]):
            out.append(f"the mapping of its {leg} leg names atoms the ligands do not have")
        water = record.get("alchemical_water")
        if water is None:
            if dq:
                out.append(f"its {leg} leg records no alchemical water, and the net charge changes by {-dq:+d}")
            continue
        if dq == 0:
            out.append(f"its {leg} leg records an alchemical water, and the net charge does not change")
            continue
        ion = water.get("ion") if isinstance(water, dict) else None
        if not isinstance(ion, dict) or not isinstance(ion.get("name"), str) or not all(
                isinstance(ion.get(k), (int, float)) and not isinstance(ion.get(k), bool)
                for k in ("charge", "sigma_nm", "epsilon_kj_mol")):
            out.append(f"the alchemical water of its {leg} leg is not recorded with its ion")
            continue
        # its atoms, each once and of the end states (fifteenth review of PR #19: atoms repeated or past the end
        # states passed, and an ion without its name stopped the leg when it was run again)
        atoms_a, atoms_b = water.get("atoms_a"), water.get("atoms_b")
        if (not isinstance(atoms_a, list) or not atoms_a or not all(_is_index(i) for i in atoms_a)
                or not isinstance(atoms_b, list) or len(atoms_b) != len(atoms_a)
                or not all(_is_index(i) for i in atoms_b)
                or len(set(atoms_a)) != len(atoms_a) or len(set(atoms_b)) != len(atoms_b)
                or (_is_index(record.get("n_atoms_a")) and max(atoms_a) >= record["n_atoms_a"])
                or (_is_index(record.get("n_atoms_b")) and max(atoms_b) >= record["n_atoms_b"])):
            out.append(f"the alchemical water of its {leg} leg is not recorded with its atoms")
    return out


def _same_ion(one, other) -> bool:
    """Whether two ions of alchemical waters are the same particle: their charge, sigma and epsilon (where each was
    taken from is not compared: twelfth review of PR #19, a Na+ of the system and the Na+ of the force field)."""
    keys = ("charge", "sigma_nm", "epsilon_kj_mol")
    try:
        return all(abs(float(one[k]) - float(other[k])) <= 1e-9 * max(1.0, abs(float(one[k]))) for k in keys)
    except (TypeError, KeyError, ValueError):
        return False


def earlier_rules_of(info, ligands=None) -> list[str]:
    """Why the record ``info`` (its prep.json) of a prepared edge is not taken: a record that cannot be used
    (:func:`record_problems`), or one made by an earlier rule of :data:`PREPARATION_RULES` (the rule acts on the edge
    and its record does not hold the present version of it), or whose mapping, as the hybrid of each leg is made of
    it, has atoms unique to one ligand that join mapped atoms (:func:`cufep.rbfe.mapping.bridging_dummies`; twelfth
    review of PR #19: an edge prepared before that check was used again), or whose legs turn the alchemical water
    into other ions. ``ligands``: the two ligands of the edge (:func:`prepared_ligands`); None: they are not known,
    and the edge is not taken. The analysis leaves such an edge out as the run refuses it (independent review of PR
    #19: `cufep rbfe analyze` and `cufep benchmark report` took edges that `cufep rbfe run` refuses)."""
    from cufep.rbfe.mapping import bridging_dummies

    problems = record_problems(info, ligands)
    if problems:
        return problems
    if ligands is None:
        return ["its ligands are known neither from the records of the preparation nor from the input, and its "
                "mapping cannot be checked"]
    made_from, legs = info["made_from"], info["legs"]
    out = []
    molecules = (ligands[0].to_rdkit(), ligands[1].to_rdkit())
    for leg in ("complex", "solvent"):
        mapping = {int(i): j for i, j in legs[leg]["mapping"].items()}
        bridges = bridging_dummies(molecules[0], molecules[1], mapping)
        if bridges:
            out.append(f"the mapping of its {leg} leg joins mapped atoms by atoms unique to one ligand: "
                       + "; ".join(bridges))
    charge_changes = ("alchemical_water_cutoff_nm" in made_from
                      or any(legs[leg].get("alchemical_water") for leg in ("complex", "solvent")))
    if charge_changes and made_from.get("alchemical_water_rule") != PREPARATION_RULES["alchemical_water"]:
        out.append("its alchemical water was chosen by the distances in the cell alone, which could take a water near "
                   "the solutes across the boundary of the cell")
    mapping_made = made_from.get("mapping") if isinstance(made_from.get("mapping"), dict) else {}
    if (mapping_made.get("core_smarts") is not None
            and made_from.get("custom_core_rule") != PREPARATION_RULES["custom_core"]):
        out.append("its custom core was mapped by an earlier rule of cuFEP")
    ions = {leg: legs[leg]["alchemical_water"]["ion"] for leg in ("complex", "solvent")
            if legs[leg].get("alchemical_water")}
    found = list(ions.values())
    if any(not _same_ion(found[0], other) for other in found[1:]):
        # eleventh review of PR #19: each leg chose the ion of its own system
        out.append("its legs turn the alchemical water into other ions ("
                   + ", ".join(f"{leg} {(ion or {}).get('name')}" for leg, ion in sorted(ions.items())) + ")")
    return out


def prepared_edge(cfg, ligand_a, ligand_b, edge_dir: Path, box_shape: str) -> dict | None:
    """The record of the prepared systems of an edge if they exist and this run can take them, None if there are
    none yet; WorkflowError if they were made from other inputs or settings. Writes nothing: a run checks every edge
    it runs with it before it writes anything (review of PR #17)."""
    from cufep.workflow import WorkflowError

    prep_dir = Path(edge_dir) / "prep"
    done = prep_dir / "prep.json"
    if not done.exists():
        computed = sorted(Path(edge_dir).glob("repeat_*/*/*"))
        if computed:
            # legs computed from prepared systems whose record is lost: a new record would be taken as the one they
            # were made from (twelfth review of PR #19: a run of another repeat prepared the edge again, and the
            # analysis took the legs of the lost one with it). Nothing is written
            raise WorkflowError(f"the record of the prepared systems of this edge ({done}) is missing and its legs "
                                f"hold what was computed from them ({computed[0].parent}): what they were made from "
                                "cannot be told. Nothing was changed; move the directory of the edge elsewhere (or "
                                "remove it) to prepare and compute the edge anew")
        return None
    try:
        info = json.loads(done.read_text())
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"the record of the prepared systems {done} cannot be read ({exc}); remove the directory "
                            "of the edge to prepare it again") from exc
    if not isinstance(info, dict):
        raise WorkflowError(f"the record of the prepared systems {done} is not a record; remove the directory of the "
                            "edge to prepare it again")
    unusable = record_problems(info, (ligand_a, ligand_b))
    # the files of the end states of both legs, which every leg loads (fifteenth review of PR #19: a prep/ restored
    # without complex_b_system.xml stopped the leg after the run had set its result aside)
    missing = [f"{leg}_{end}{suffix}" for leg in ("complex", "solvent") for end in ("a", "b")
               for suffix in ("_system.xml", ".npz", "_topology.json")
               if not (prep_dir / f"{leg}_{end}{suffix}").is_file()]
    if missing:
        unusable.append(f"its prepared systems are incomplete ({', '.join(missing)} missing)")
    if unusable:
        raise WorkflowError(f"the record of the prepared systems {done} cannot be used: {'; '.join(unusable)}; remove "
                            "the directory of the edge to prepare it again")
    stored, current = info.get("made_from"), edge_made_from(cfg, ligand_a, ligand_b, box_shape)
    if stored != current:
        differing = sorted(k for k in set(stored or {}) | set(current) if (stored or {}).get(k) != current.get(k))
        rules = [k for k in differing if k.endswith("_rule")]
        raise WorkflowError(f"the prepared systems in {prep_dir} were made from other inputs or settings ("
                            + ", ".join(differing) + ")"
                            + (f"; {', '.join(rules)}: they were prepared by an earlier rule of cuFEP" if rules else "")
                            + "; remove the directory of the edge to prepare it again")
    # the mappings of the legs as the hybrid is made of them, and the ions of their alchemical waters (the inputs are
    # those of the record: their molecules are the run's)
    problems = [why for why in earlier_rules_of(info, (ligand_a, ligand_b))
                if why.startswith(("its legs turn", "the mapping of its"))]
    if problems:
        raise WorkflowError(f"the prepared systems in {prep_dir} cannot be used: {'; '.join(problems)}; remove the "
                            "directory of the edge to prepare it again")
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
