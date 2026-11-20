"""Lipid bilayer construction for membrane proteins.

Steps: orient the complex (membrane normal along z, membrane centre at z = 0), build a
bilayer plus solvent with OpenMM's ``Modeller.addMembrane``, hydrate cavities inside the
protein, and check the result.

Why the extra hydration step: ``addMembrane`` adds bulk water only above and below the
lipid patch. Inside the patch's z-range the only water is what the patch brought along,
and the patch has none where the protein is. Pockets and channels of a membrane protein
that lie at the depth of the head groups or deeper (a GPCR's orthosteric site, internal
water networks) therefore start dry unless the input structure carries crystal waters.
"""
from __future__ import annotations

import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm.app import element as elem

from cufep.prep.ligand import LIGAND_RESNAME
from cufep.prep.orient import (
    Orientation,
    OrientationError,
    consensus_orientation,
    memembed_orientation,
    orientation_matrix,
    slab_orientation,
    transfer_energy_orientation,
)
from cufep.prep.protein import AMINO_ACIDS, CAPS, WATER

WATER_SPACING_NM = 0.29
MIN_DISTANCE_TO_ATOMS_NM = 0.30
MIN_DISTANCE_TO_LIPID_NM = 0.45
BURIED_FRACTION = 0.75
MAX_INTERNAL_WATERS = 400


class MembraneError(RuntimeError):
    """The bilayer could not be built reliably."""


def _positions(modeller) -> np.ndarray:
    return np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=float)


def choose_orientation(modeller, method: str, log=print) -> Orientation | None:
    """Orientation of the protein in the Modeller, or None for ``input`` (already oriented)."""
    if method == "input":
        return None
    top, pos = modeller.topology, _positions(modeller)
    if method == "transfer":
        return transfer_energy_orientation(top, pos)
    if method == "slab":
        return slab_orientation(top, pos)
    if method == "memembed":
        o = memembed_orientation(top, pos)
        if o is None:
            raise OrientationError("memembed is not installed or failed")
        return o
    return consensus_orientation(top, pos, log=log)


def _unit_vectors(n: int = 40) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def hydrate_protein_interior(modeller, z_limit_nm: float, seed: int = 0, log=print) -> int:
    """Place water molecules in protein cavities within ``|z| < z_limit_nm``. Returns the number added.

    A grid point becomes a water if it is free (no heavy atom within 3.0 A), away from the
    lipids (4.5 A), and enclosed by protein in at least 75 % of directions.
    """
    from scipy.spatial import cKDTree

    top, pos = modeller.topology, _positions(modeller)
    heavy = np.array([a.index for a in top.atoms() if a.element is not None and a.element != elem.hydrogen])
    protein = np.array([a.index for a in top.atoms() if a.element is not None and a.element != elem.hydrogen
                        and (a.residue.name in AMINO_ACIDS or a.residue.name in CAPS)])
    lipid = np.array([a.index for a in top.atoms() if a.element is not None and a.element != elem.hydrogen
                      and a.residue.name not in AMINO_ACIDS and a.residue.name not in CAPS
                      and a.residue.name not in WATER and a.residue.name != LIGAND_RESNAME
                      and len(list(a.residue.atoms())) > 1], dtype=int)
    if len(protein) == 0:
        return 0
    p = pos[protein]
    lo = np.maximum(p.min(axis=0), [-np.inf, -np.inf, -z_limit_nm])
    hi = np.minimum(p.max(axis=0), [np.inf, np.inf, z_limit_nm])
    if np.any(hi <= lo):
        return 0
    axes = [np.arange(lo[k], hi[k], WATER_SPACING_NM) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    tree_all, tree_prot = cKDTree(pos[heavy]), cKDTree(p)
    free = tree_all.query(grid, k=1)[0] >= MIN_DISTANCE_TO_ATOMS_NM
    grid = grid[free]
    if len(lipid) and len(grid):
        grid = grid[cKDTree(pos[lipid]).query(grid, k=1)[0] >= MIN_DISTANCE_TO_LIPID_NM]
    if len(grid) == 0:
        return 0
    directions = _unit_vectors(40)
    buried = np.zeros(len(grid))
    for g, point in enumerate(grid):
        near = tree_prot.query_ball_point(point, 1.2)
        if len(near) < 30:
            continue
        v = p[near] - point
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        buried[g] = np.mean((v @ directions.T).max(axis=0) > 0.85)
    order = np.argsort(-buried)
    chosen: list[np.ndarray] = []
    for g in order:
        if buried[g] < BURIED_FRACTION or len(chosen) >= MAX_INTERNAL_WATERS:
            break
        if all(np.linalg.norm(grid[g] - c) >= 0.28 for c in chosen):
            chosen.append(grid[g])
    if not chosen:
        return 0
    rng = np.random.default_rng(seed)
    wat_top = app.Topology()
    chain = wat_top.addChain("W")
    wat_pos = []
    oh, half = 0.09572, np.radians(104.52) / 2.0
    for k, o in enumerate(chosen):
        res = wat_top.addResidue("HOH", chain, id=str(k + 1))
        ao = wat_top.addAtom("O", elem.oxygen, res)
        h1 = wat_top.addAtom("H1", elem.hydrogen, res)
        h2 = wat_top.addAtom("H2", elem.hydrogen, res)
        wat_top.addBond(ao, h1)
        wat_top.addBond(ao, h2)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))                 # random orientation
        wat_pos += [o, o + oh * (np.cos(half) * q[0] + np.sin(half) * q[1]),
                    o + oh * (np.cos(half) * q[0] - np.sin(half) * q[1])]
    box = top.getPeriodicBoxVectors()
    modeller.add(wat_top, np.asarray(wat_pos) * unit.nanometer)
    modeller.topology.setPeriodicBoxVectors(box)
    log(f"membrane: placed {len(chosen)} water molecules in protein cavities inside the bilayer region")
    return len(chosen)


def membrane_quality(modeller) -> dict:
    """Numbers used to judge a built bilayer."""
    top, pos = modeller.topology, _positions(modeller)
    box = top.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    phosphorus = np.array([a.index for a in top.atoms() if a.element == elem.phosphorus
                           and a.residue.name not in AMINO_ACIDS], dtype=int)
    protein = np.array([a.index for a in top.atoms() if a.residue.name in AMINO_ACIDS
                        and a.element is not None and a.element != elem.hydrogen], dtype=int)
    water_o = np.array(
        [a.index for a in top.atoms() if a.residue.name in WATER and a.element == elem.oxygen], dtype=int
    )
    zp = pos[phosphorus, 2] if len(phosphorus) else np.array([])
    upper, lower = int(np.sum(zp > 0)), int(np.sum(zp < 0))
    core_waters = 0
    if len(water_o) and len(protein):
        from scipy.spatial import cKDTree
        inside = water_o[np.abs(pos[water_o, 2]) < 0.8]
        if len(inside):
            core_waters = int(np.sum(cKDTree(pos[protein]).query(pos[inside], k=1)[0] > 0.6))
    return {"lipids_upper": upper, "lipids_lower": lower,
            "phosphate_plane_nm": float(np.abs(zp).mean()) if len(zp) else 0.0,
            "box_nm": [float(box[0][0]), float(box[1][1]), float(box[2][2])],
            "waters_in_hydrophobic_core_away_from_protein": core_waters,
            "n_atoms": top.getNumAtoms()}


def _molecules(topology) -> list[list[int]]:
    """Atom indices of every molecule (connected component of the bond graph)."""
    parent = list(range(topology.getNumAtoms()))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for b in topology.bonds():
        ra, rb = find(b[0].index), find(b[1].index)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(len(parent)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _kind(residue) -> str:
    if residue.name in AMINO_ACIDS or residue.name in CAPS:
        return "protein"
    if residue.name == LIGAND_RESNAME:
        return "ligand"
    if residue.name in WATER:
        return "water"
    if len(list(residue.atoms())) == 1:
        return "ion"
    return "lipid"


CLASH_NM = 0.24


def contact_is_new(pos, i: int, j: int, built_box_nm, box_nm) -> bool:
    """Whether the new periodic boundary brought heavy atoms ``i`` and ``j`` together: their nearest
    image in the trimmed box ``box_nm`` is nearer than in the box the bilayer was built in,
    ``built_box_nm`` (rectangular boxes). Only such contacts are resolved by trimming (review of
    7b49066: a water coordinating a Mg ion at 0.19 nm, a contact of the built system, was removed).
    A pair whose nearest image is the same in both boxes has the same distance in both, computed
    the same way, so a contact of the built system is never taken for a new one, also at exactly
    :data:`CLASH_NM` (review of 10589a6: a water 0.240 nm from an ion was removed)."""
    d = np.asarray(pos[i], dtype=float) - np.asarray(pos[j], dtype=float)

    def nearest(box) -> float:
        box = np.asarray(box, dtype=float)
        return float(np.linalg.norm(d - box * np.round(d / box)))

    return nearest(box_nm) < nearest(built_box_nm)


def trim_membrane(modeller, padding_nm: float, log=print) -> dict:
    """Cut the bilayer and solvent down to a square box just large enough for the protein.

    ``Modeller.addMembrane`` only tiles whole copies of its 6.4 x 6.7 nm lipid patch, so a
    protein that needs 7 nm gets a 12.7 x 13.4 nm membrane and almost three times the atoms.

    Lipids are kept or removed whole, by the position of their phosphorus atom. A kept
    lipid may reach across the new periodic boundary and overlap a lipid on the far side;
    such overlaps are resolved by removing the lipid with the most clashes until none is
    left, and waters overlapping a lipid across the boundary are removed too. The leaflets
    end with equal numbers of lipids. The small area deficit that remains closes under
    semi-isotropic NPT.
    """
    import openmm as mm

    top, pos = modeller.topology, _positions(modeller)
    atoms = list(top.atoms())
    kinds = np.array([_kind(a.residue) for a in atoms])
    heavy = np.array([a.element is not None and a.element != elem.hydrogen for a in atoms])
    solute = np.isin(kinds, ("protein", "ligand"))
    centre = 0.5 * (pos[solute, :2].min(axis=0) + pos[solute, :2].max(axis=0))
    radius = float(np.linalg.norm(pos[solute, :2] - centre, axis=1).max())
    target = 2.0 * radius + 2.0 * padding_nm
    box = top.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    old = (float(box[0][0]), float(box[1][1]), float(box[2][2]))
    if target > min(old[0], old[1]) - 0.4:
        return {"trimmed": False, "box_nm": list(old)}
    rel = pos[:, :2] - centre
    molecules = _molecules(top)
    area_per_lipid = 0.64                      # nm^2 after relaxation; only used to choose the cut size
    built_apl = 0.666                          # nm^2 in OpenMM's pre-equilibrated patches

    # Removing overlapping lipids leaves fewer lipids than the area needs, and the box then
    # shrinks under NPT. Cut a little larger so that the relaxed box still has the padding.
    edge, plan = target, None
    while True:
        plan = _select_for_box(edge, old, rel, pos, atoms, kinds, heavy, molecules)
        n_leaflet = len(plan["upper"])
        protein_area = max(0.0, edge * edge - plan["n_inside_per_leaflet"] * built_apl)
        relaxed = float(np.sqrt(protein_area + n_leaflet * area_per_lipid))
        if relaxed >= target - 0.1 or edge + 0.2 > min(old[0], old[1]) - 0.4:
            break
        edge += 0.2
    doomed, upper, lower = plan["doomed"], plan["upper"], plan["lower"]
    # The box will shrink laterally from `edge` to about `relaxed` and grow along z by the same
    # volume. Remove the water that would only make the relaxed box taller than the built one,
    # but keep at least 0.8 nm of water above and below the solute for the start.
    z_lo, z_hi = float(pos[:, 2].min()), float(pos[:, 2].max())
    z_mid = 0.5 * (z_lo + z_hi)
    solute_z = pos[solute, 2]
    new_z = max(old[2] * (relaxed / edge) ** 2, float(solute_z.max() - solute_z.min()) + 1.6,
                2.0 * float(np.abs(solute_z - z_mid).max()) + 1.6)
    new_z = min(new_z, old[2])
    removed_z = 0
    if new_z < old[2] - 0.05:
        for mol in molecules:
            if kinds[mol[0]] in ("water", "ion") and mol[0] not in doomed and abs(pos[mol[0], 2] - z_mid) > 0.5 * new_z:
                doomed.update(mol)
                removed_z += 1
        new_z += 0.15                           # a thin gap instead of overlaps at the z boundary
    # waters that overlap another molecule across the new periodic boundary, in the final box (review of
    # PR 1: waters were removed only where they overlapped a lipid; two waters on the two sides of the new
    # boundary ended 0.09 A apart, and the minimisation could not separate them)
    across = boundary_water_overlaps(pos, kinds, heavy, molecules, doomed, (edge, edge, new_z), old)
    doomed.update(across)
    removed_across = len({atoms[i].residue.index for i in across})
    modeller.delete([atoms[i] for i in sorted(doomed)])
    modeller.topology.setPeriodicBoxVectors(
        [mm.Vec3(edge, 0, 0), mm.Vec3(0, edge, 0), mm.Vec3(0, 0, new_z)] * unit.nanometer)
    old_z, old = old[2], (old[0], old[1], new_z)
    log(f"membrane: trimmed from {old[0]:.2f} x {old[1]:.2f} nm to {edge:.2f} x {edge:.2f} nm; "
        f"{len(upper)} + {len(lower)} lipids kept, {plan['removed_lipids']} lipid(s) and "
        f"{plan['removed_waters'] + removed_across} water(s) removed for overlapping across the new boundary; "
        f"box height {old_z:.2f} -> {new_z:.2f} nm; expected box after relaxation about "
        f"{relaxed:.2f} x {relaxed:.2f} x {new_z * (edge / relaxed) ** 2:.2f} nm")
    return {"trimmed": True, "box_nm": [edge, edge, new_z], "built_box_nm": [old[0], old[1], old_z],
            "lipids_removed_for_overlap": plan["removed_lipids"], "expected_relaxed_edge_nm": relaxed,
            "waters_removed_for_overlap": plan["removed_waters"] + removed_across}


def boundary_water_overlaps(pos, kinds, heavy, molecules, doomed, box_nm, built_box_nm) -> set[int]:
    """Atoms of the water molecules to remove so that no kept water comes within :data:`CLASH_NM`
    (heavy atoms) of another kept molecule through the new periodic boundary: in the trimmed box
    ``box_nm`` but not in the box the bilayer was built in, ``built_box_nm`` (lengths of rectangular
    boxes, minimum images). A contact that the built system already had is its own (a water
    coordinating an ion, 0.19 nm) and is left alone (review of 7b49066: a Mg coordination water was
    removed). Of two overlapping waters the one with more overlaps goes, then the later one; a water
    overlapping a lipid, the protein, the ligand or an ion goes."""
    from scipy.spatial import cKDTree

    box = np.asarray(box_nm, dtype=float)
    pos = np.asarray(pos, dtype=float)
    mol_of = np.full(len(kinds), -1, dtype=int)
    for m, mol in enumerate(molecules):
        mol_of[mol] = m
    kept = np.array([i for i in range(len(kinds)) if heavy[i] and i not in doomed], dtype=int)
    if len(kept) < 2:
        return set()
    tree = cKDTree(np.mod(pos[kept], box), boxsize=box)
    overlaps: dict[int, set[int]] = {}
    forced: set[int] = set()
    for a, b in tree.query_pairs(CLASH_NM):
        i, j = kept[a], kept[b]
        mi, mj = mol_of[i], mol_of[j]
        if mi == mj or not contact_is_new(pos, i, j, built_box_nm, box):
            continue
        wi, wj = kinds[i] == "water", kinds[j] == "water"
        if wi and wj:
            overlaps.setdefault(mi, set()).add(mj)
            overlaps.setdefault(mj, set()).add(mi)
        elif wi:
            forced.add(mi)
        elif wj:
            forced.add(mj)
    removed = set(forced)
    for m in forced:
        for other in overlaps.pop(m, set()):
            overlaps.get(other, set()).discard(m)
    while any(overlaps.values()):
        worst = max((m for m in overlaps if overlaps[m]), key=lambda m: (len(overlaps[m]), m))
        removed.add(worst)
        for other in overlaps.pop(worst):
            overlaps[other].discard(worst)
    return {i for m in removed for i in molecules[m]}


def _select_for_box(edge, old, rel, pos, atoms, kinds, heavy, molecules) -> dict:
    """Atoms to delete for a square box of the given edge (see :func:`trim_membrane`)."""
    from scipy.spatial import cKDTree

    half = 0.5 * edge
    inside = np.all(np.abs(rel) <= half, axis=1)
    doomed: set[int] = set()
    lipids: dict[int, dict] = {}               # molecule id -> atoms, leaflet
    mol_of = np.full(len(atoms), -1, dtype=int)
    for m, mol in enumerate(molecules):
        kind = kinds[mol[0]]
        if kind in ("protein", "ligand"):
            continue
        if kind == "lipid":
            anchor = [i for i in mol if atoms[i].element == elem.phosphorus] or mol
            if not inside[anchor].all():
                doomed.update(mol)
            else:
                lipids[m] = {"atoms": mol, "upper": float(pos[anchor, 2].mean()) > 0}
                mol_of[mol] = m
        elif not inside[mol[0]]:               # water or ion: decided by its first atom
            doomed.update(mol)

    n_inside_per_leaflet = len(lipids) / 2.0
    # ---- overlaps across the new periodic boundary ---------------------------------------
    wrapped = np.empty_like(pos)
    wrapped[:, :2] = (rel + half) % edge
    wrapped[:, 2] = (pos[:, 2] - pos[:, 2].min()) % old[2]
    lip_heavy = np.array([i for m in lipids for i in lipids[m]["atoms"] if heavy[i]], dtype=int)
    removed_lipids = 0
    if len(lip_heavy):
        tree = cKDTree(wrapped[lip_heavy], boxsize=[edge, edge, old[2]])
        clashes: dict[int, dict[int, int]] = {}
        for a, b in tree.query_pairs(CLASH_NM):
            ma, mb = mol_of[lip_heavy[a]], mol_of[lip_heavy[b]]
            if ma != mb and contact_is_new(pos, lip_heavy[a], lip_heavy[b], old, (edge, edge, old[2])):
                clashes.setdefault(ma, {}).setdefault(mb, 0)
                clashes.setdefault(mb, {}).setdefault(ma, 0)
                clashes[ma][mb] += 1
                clashes[mb][ma] += 1
        while clashes:
            worst = max(clashes, key=lambda m: sum(clashes[m].values()))
            for other in list(clashes[worst]):
                del clashes[other][worst]
                if not clashes[other]:
                    del clashes[other]
            del clashes[worst]
            doomed.update(lipids.pop(worst)["atoms"])
            removed_lipids += 1
    upper = [m for m in lipids if lipids[m]["upper"]]
    lower = [m for m in lipids if not lipids[m]["upper"]]

    def distance_to_edge(m: int) -> float:
        return float((half - np.abs(rel[lipids[m]["atoms"]])).min())

    for group, other in ((upper, lower), (lower, upper)):
        group.sort(key=distance_to_edge)
        while len(group) > len(other):
            doomed.update(lipids.pop(group.pop(0))["atoms"])
    # ---- waters overlapping a kept lipid across the boundary -------------------------------
    lip_heavy = np.array([i for m in lipids for i in lipids[m]["atoms"] if heavy[i]], dtype=int)
    water_o = np.array([a.index for a in atoms if kinds[a.index] == "water" and a.element == elem.oxygen
                        and a.index not in doomed], dtype=int)
    removed_waters = 0
    if len(lip_heavy) and len(water_o):
        tree = cKDTree(wrapped[lip_heavy], boxsize=[edge, edge, old[2]])
        for o, near in zip(water_o, tree.query_ball_point(wrapped[water_o], CLASH_NM)):
            if any(contact_is_new(pos, o, lip_heavy[k], old, (edge, edge, old[2])) for k in near):
                doomed.update(a.index for a in atoms[o].residue.atoms())
                removed_waters += 1
    return {"doomed": doomed, "upper": upper, "lower": lower, "removed_lipids": removed_lipids,
            "removed_waters": removed_waters, "n_inside_per_leaflet": n_inside_per_leaflet}


def add_ions(modeller, forcefield, ionic_strength_M: float, exclusion_z_nm: float, seed: int = 0, log=print) -> dict:
    """Neutralise the system and add salt by replacing bulk water molecules with Na+ / Cl-.

    Only waters outside the bilayer (``|z| > exclusion_z_nm``) and at least 0.5 nm from the
    protein and ligand are replaced.
    """
    import openmm as mm
    from scipy.spatial import cKDTree

    top = modeller.topology
    system = forcefield.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                 for i in range(system.getNumParticles()))
    total = int(round(charge))
    if abs(charge - total) > 0.01:
        raise MembraneError(f"the system has a non-integer net charge ({charge:.3f})")
    pos = _positions(modeller)
    atoms = list(top.atoms())
    solute = np.array([a.index for a in atoms if _kind(a.residue) in ("protein", "ligand")
                       and a.element is not None and a.element != elem.hydrogen], dtype=int)
    waters = [r for r in top.residues() if r.name in WATER]
    n_pairs = int(np.floor((len(waters) - abs(total)) * ionic_strength_M / 55.4 + 0.5))
    n_pos, n_neg = n_pairs + max(0, -total), n_pairs + max(0, total)
    if n_pos + n_neg == 0:
        # a neutral system without salt: no water is replaced
        log(f"membrane: net charge {total:+d}; no ion needed")
        return {"net_charge_before_ions": total, "n_sodium": 0, "n_chloride": 0}
    oxygen = {r.index: next(a.index for a in r.atoms() if a.element == elem.oxygen) for r in waters}
    tree = cKDTree(pos[solute])
    candidates = [r for r in waters if abs(pos[oxygen[r.index], 2]) > exclusion_z_nm
                  and tree.query(pos[oxygen[r.index]])[0] > 0.5]
    if len(candidates) < n_pos + n_neg:
        raise MembraneError("not enough bulk water to place the ions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    picked: list = []
    for k in order:
        if len(picked) == n_pos + n_neg:
            break
        r = candidates[k]
        here = pos[oxygen[r.index]]
        if all(np.linalg.norm(here - pos[oxygen[q.index]]) > 0.5 for q in picked):
            picked.append(r)
    if len(picked) < n_pos + n_neg:
        raise MembraneError("could not place the ions with 0.5 nm spacing")
    ion_top = app.Topology()
    chain = ion_top.addChain("I")
    ion_pos = []
    for k, r in enumerate(picked):
        positive = k < n_pos
        res = ion_top.addResidue("NA" if positive else "CL", chain, id=str(k + 1))
        ion_top.addAtom("NA" if positive else "CL", elem.sodium if positive else elem.chlorine, res)
        ion_pos.append(pos[oxygen[r.index]])
    box = top.getPeriodicBoxVectors()
    modeller.delete(picked)
    if ion_pos:
        modeller.add(ion_top, np.asarray(ion_pos) * unit.nanometer)
    modeller.topology.setPeriodicBoxVectors(box)
    log(f"membrane: net charge {total:+d}; added {n_pos} Na+ and {n_neg} Cl-")
    return {"net_charge_before_ions": total, "n_sodium": n_pos, "n_chloride": n_neg}


def add_membrane(modeller, forcefield, cfg, log=print) -> dict:
    """Orient the complex, build the bilayer and solvent in place, hydrate and check it."""
    import openmm as mm

    orientation = choose_orientation(modeller, cfg.membrane.orientation, log)
    info: dict = {"orientation_method": "input" if orientation is None else orientation.method}
    if orientation is not None:
        rot = orientation_matrix(orientation.normal)
        pos = (_positions(modeller) - orientation.centre_nm) @ rot.T
        modeller.positions = pos * unit.nanometer
        info["hydrophobic_thickness_nm"] = float(orientation.thickness_nm)
        log(f"membrane: oriented with the '{orientation.method}' method "
            f"(hydrophobic thickness {10 * orientation.thickness_nm:.0f} A)")
    pos = _positions(modeller)
    protein_z = pos[[a.index for a in modeller.topology.atoms() if a.residue.name in AMINO_ACIDS], 2]
    if protein_z.min() > -1.0 or protein_z.max() < 1.0:
        raise MembraneError("the protein does not span z = 0 +/- 1 nm; the membrane orientation is wrong. "
                            "With `membrane.orientation: input` the PDB must have the membrane normal along z "
                            "and the membrane centre at z = 0.")
    n_before = modeller.topology.getNumAtoms()
    # Ions are added afterwards: the bilayer is trimmed first, which would unbalance them.
    modeller.addMembrane(forcefield, lipidType=cfg.membrane.lipid, membraneCenterZ=0.0 * unit.nanometer,
                         minimumPadding=cfg.membrane.padding_nm * unit.nanometer,
                         ionicStrength=0.0 * unit.molar, neutralize=False)
    info.update(trim_membrane(modeller, cfg.membrane.padding_nm, log))
    quality = membrane_quality(modeller)
    z_limit = quality["phosphate_plane_nm"] + 0.3 if quality["phosphate_plane_nm"] > 0 else 2.0
    info["internal_waters_added"] = hydrate_protein_interior(modeller, z_limit, log=log)
    info.update(add_ions(modeller, forcefield, cfg.thermodynamics.ionic_strength_M, z_limit + 0.2, log=log))
    quality = membrane_quality(modeller)
    info.update(quality)
    up, low = quality["lipids_upper"], quality["lipids_lower"]
    log(f"membrane: {cfg.membrane.lipid} bilayer with {up} + {low} lipids, box "
        + " x ".join(f"{x:.2f}" for x in quality["box_nm"]) + f" nm, {quality['n_atoms'] - n_before} atoms added")
    # the numbers of the bilayer (lipids per leaflet, waters in the hydrophobic core away from the protein) are
    # in info, without a verdict: earlier versions stopped below 40 lipids and warned of leaflets that differ by
    # more than 10 % and of such waters, levels of cuFEP's own (removed by the owner's order of 2026-11-20)
    info["warnings"] = []
    _ = mm
    return info
