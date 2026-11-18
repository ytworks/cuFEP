"""Solvation of soluble systems (complex and ligand-only legs)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm.app as app
import openmm.unit as unit

from cufep.prep.ligand import LIGAND_RESNAME, Ligand


@dataclass
class SolvationSpec:
    padding_nm: float = 1.0
    box_shape: str = "dodecahedron"        # cube | dodecahedron | octahedron
    ionic_strength_M: float = 0.15
    water_model: str = "tip3p"
    min_edge_nm: float = 0.0               # e.g. 3 x cutoff for the cuPotential plugin


def combine(protein: tuple[app.Topology, unit.Quantity] | None, ligand: Ligand | None) -> app.Modeller:
    """Merge protein (optional) and ligand (optional) into one Modeller."""
    modeller = app.Modeller(app.Topology(), [] * unit.nanometer)
    if protein is not None:
        modeller.add(protein[0], protein[1])
    if ligand is not None:
        top, pos = ligand.to_openmm()
        modeller.add(top, pos)
    return modeller


def solvate(modeller: app.Modeller, forcefield: app.ForceField, spec: SolvationSpec) -> None:
    """Add water and ions in place.

    OpenMM sizes the box as ``width = max(2 * radius + padding, 2 * padding)`` where
    ``radius`` is the radius of the sphere enclosing the solute. ``min_edge_nm`` raises the
    padding so that the shortest box height (the edge of a cube, 1/sqrt(2) of the edge of a
    rhombic dodecahedron) is at least that long.
    """
    pos = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    padding = spec.padding_nm
    if spec.min_edge_nm > 0 and len(pos):
        centre = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
        radius = float(np.linalg.norm(pos - centre, axis=1).max())
        min_width = spec.min_edge_nm * (2.0 ** 0.5 if spec.box_shape == "dodecahedron" else 1.0)
        # width >= min_width needs either 2r + padding >= min_width or 2 * padding >= min_width
        padding = max(padding, min(min_width - 2.0 * radius, 0.5 * min_width))
    modeller.addSolvent(
        forcefield, model=spec.water_model, padding=padding * unit.nanometer,
        boxShape=spec.box_shape, ionicStrength=spec.ionic_strength_M * unit.molar, neutralize=True)


def ligand_atom_indices(topology: app.Topology) -> list[int]:
    return [a.index for a in topology.atoms() if a.residue.name == LIGAND_RESNAME]
