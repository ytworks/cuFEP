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
from cufep.prep.orient import (Orientation, OrientationError, consensus_orientation, detect_membrane,
                               memembed_orientation, orientation_matrix, transfer_energy_orientation)
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
        v = detect_membrane(top, pos)
        if v.normal is None:
            raise OrientationError(f"the hydrophobic-slab search found no slab ({v.reason})")
        return Orientation("slab", v.normal, v.centre_nm, v.thickness_nm, v.score)
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
    water_o = np.array([a.index for a in top.atoms() if a.residue.name in WATER and a.element == elem.oxygen], dtype=int)
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
    modeller.addMembrane(forcefield, lipidType=cfg.membrane.lipid, membraneCenterZ=0.0 * unit.nanometer,
                         minimumPadding=cfg.membrane.padding_nm * unit.nanometer,
                         ionicStrength=cfg.thermodynamics.ionic_strength_M * unit.molar, neutralize=True)
    quality = membrane_quality(modeller)
    z_limit = quality["phosphate_plane_nm"] + 0.3 if quality["phosphate_plane_nm"] > 0 else 2.0
    info["internal_waters_added"] = hydrate_protein_interior(modeller, z_limit, log=log)
    quality = membrane_quality(modeller)
    info.update(quality)
    up, low = quality["lipids_upper"], quality["lipids_lower"]
    log(f"membrane: {cfg.membrane.lipid} bilayer with {up} + {low} lipids, box "
        + " x ".join(f"{x:.2f}" for x in quality["box_nm"]) + f" nm, {quality['n_atoms'] - n_before} atoms added")
    warnings = []
    if up + low < 40:
        raise MembraneError(f"only {up + low} lipids were placed; the bilayer is too small to be meaningful")
    if abs(up - low) > max(2, 0.1 * (up + low)):
        warnings.append(f"the two leaflets differ by {abs(up - low)} lipids ({up} vs {low})")
    if quality["waters_in_hydrophobic_core_away_from_protein"]:
        warnings.append(f"{quality['waters_in_hydrophobic_core_away_from_protein']} water(s) in the hydrophobic "
                        "core away from the protein")
    info["warnings"] = warnings
    _ = mm
    return info
