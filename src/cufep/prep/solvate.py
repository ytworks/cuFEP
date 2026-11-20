"""Solvation of soluble systems (complex and ligand-only legs)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm.app as app
import openmm.unit as unit

from cufep.prep.ligand import LIGAND_RESNAME, Ligand

#: OpenMM's rule for a periodic cutoff: the cutoff may be at most half of every height of the box
#: (NonbondedForce: "The cutoff distance cannot be greater than half the periodic box size"); the
#: cuPotential plugin needs three times the cutoff (:data:`cufep.backends.cupotential.MIN_BOX_CUTOFF_RATIO`)
STOCK_MIN_BOX_CUTOFF_RATIO = 2.0
#: a box is built with this margin over the rule of its backend, because it relaxes under NPT
BOX_MARGIN = 1.08


def min_box_height_nm(cutoff_nm: float, backend_name: str, margin: float = BOX_MARGIN) -> float:
    """The shortest height a box of ``backend_name`` (``cupotential`` or stock ``openmm``) is built
    with for the cutoff ``cutoff_nm``: the rule of the backend times ``margin``. Review of PR 1: stock
    OpenMM had no rule, and a dodecahedron with 1.5 nm padding and a 1.2 nm cutoff was 2.121 nm high,
    so no Context could be made."""
    from cufep.backends.cupotential import MIN_BOX_CUTOFF_RATIO

    ratio = MIN_BOX_CUTOFF_RATIO if backend_name == "cupotential" else STOCK_MIN_BOX_CUTOFF_RATIO
    return float(margin) * ratio * float(cutoff_nm)


def box_heights_nm(box_nm) -> np.ndarray:
    """The three heights of a periodic box given by its vectors (rows, nm): the distances between
    opposite faces (in the reduced form of OpenMM: a_x, b_y, c_z)."""
    a, b, c = np.asarray(box_nm, dtype=float)
    volume = abs(float(np.dot(a, np.cross(b, c))))
    return np.array([volume / np.linalg.norm(np.cross(b, c)), volume / np.linalg.norm(np.cross(c, a)),
                     volume / np.linalg.norm(np.cross(a, b))])


def box_height_problem(box_nm, cutoff_nm: float) -> str | None:
    """Why no backend can run a system in the box ``box_nm`` with the cutoff ``cutoff_nm`` (OpenMM's
    rule, without the margin of :func:`min_box_height_nm`), None if one can."""
    shortest = float(box_heights_nm(box_nm).min())
    need = min_box_height_nm(cutoff_nm, "openmm", margin=1.0)
    if shortest < need * (1.0 - 1e-9):
        return (f"the shortest height of its box is {shortest:.3f} nm, less than {STOCK_MIN_BOX_CUTOFF_RATIO:g} times "
                f"the cutoff of {float(cutoff_nm):g} nm, so OpenMM cannot make a Context of it")
    return None


@dataclass
class SolvationSpec:
    padding_nm: float = 1.0
    box_shape: str = "dodecahedron"        # cube | dodecahedron | octahedron
    ionic_strength_M: float = 0.15
    water: str = "amber14/tip3p.xml"      # forcefield.water; the box is built from cufep.prep.water
    min_edge_nm: float = 0.0               # shortest box height, :func:`min_box_height_nm`


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
    from cufep.prep.water import water_model_of

    water = water_model_of(spec.water)
    if water.sites > 3:
        # waters that came with the input (crystal waters) have no extra site yet; addSolvent
        # builds a System of what is there to find its charge, so they get it first
        modeller.addExtraParticles(forcefield)
    modeller.addSolvent(
        forcefield, model=water.box, padding=padding * unit.nanometer,
        boxShape=spec.box_shape, ionicStrength=spec.ionic_strength_M * unit.molar, neutralize=True)


def ligand_atom_indices(topology: app.Topology) -> list[int]:
    return [a.index for a in topology.atoms() if a.residue.name == LIGAND_RESNAME]
