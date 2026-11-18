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
    """Add water and ions in place."""
    pos = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    extent = float((pos.max(axis=0) - pos.min(axis=0)).max()) if len(pos) else 0.0
    padding = spec.padding_nm
    if spec.min_edge_nm > 0:
        # Guarantee the minimum box edge even for a tiny solute (ligand-only leg).
        padding = max(padding, 0.5 * (spec.min_edge_nm - extent) + 0.05)
    modeller.addSolvent(
        forcefield, model=spec.water_model, padding=padding * unit.nanometer,
        boxShape=spec.box_shape, ionicStrength=spec.ionic_strength_M * unit.molar, neutralize=True)


def ligand_atom_indices(topology: app.Topology) -> list[int]:
    return [a.index for a in topology.atoms() if a.residue.name == LIGAND_RESNAME]
