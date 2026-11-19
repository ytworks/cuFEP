"""Ligand loading and partial charges.

The ligand SDF is taken as given: coordinates (the pose in the pocket), protonation
state and stereochemistry are not changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit

LIGAND_RESNAME = "LIG"
NAGL_DEFAULT_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"


class LigandError(ValueError):
    """The ligand input cannot be used as given."""


@dataclass
class Ligand:
    """One ligand: an OpenFF Molecule with a single conformer (the input pose)."""

    name: str
    offmol: object  # openff.toolkit.Molecule
    source: str = ""
    record: int = 0
    properties: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return self.offmol.n_atoms

    @property
    def formal_charge(self) -> int:
        return int(round(self.offmol.total_charge.m_as("elementary_charge")))

    def positions_nm(self) -> np.ndarray:
        return np.asarray(self.offmol.conformers[0].m_as("nanometer"), dtype=float)

    def heavy_atom_indices(self) -> list[int]:
        return [i for i, a in enumerate(self.offmol.atoms) if a.atomic_number > 1]

    def to_openmm(self) -> tuple[app.Topology, unit.Quantity]:
        """Topology (one chain, one residue named LIG) and positions."""
        top = self.offmol.to_topology().to_openmm()
        for res in top.residues():
            res.name = LIGAND_RESNAME
        return top, self.positions_nm() * unit.nanometer

    def to_rdkit(self):
        return self.offmol.to_rdkit()

    def digest(self) -> str:
        """Hash of chemistry and coordinates; keys the partial charge cache."""
        h = hashlib.sha256()
        h.update(self.offmol.to_smiles(mapped=True).encode())
        h.update(np.round(self.positions_nm(), 5).tobytes())
        return h.hexdigest()[:16]


def load_ligands(sdf_path: str | Path) -> list[Ligand]:
    """Read every record of an SDF file. Hydrogens must be explicit."""
    from openff.toolkit import Molecule
    from rdkit import Chem

    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        raise LigandError(f"ligand file not found: {sdf_path}")
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    ligands: list[Ligand] = []
    for i, rdmol in enumerate(supplier):
        if rdmol is None:
            raise LigandError(f"{sdf_path}: record {i} could not be parsed")
        name = rdmol.GetProp("_Name").strip() if rdmol.HasProp("_Name") else ""
        name = name or f"{sdf_path.stem}_{i}"
        if rdmol.GetNumConformers() == 0 or not rdmol.GetConformer().Is3D():
            raise LigandError(f"{sdf_path}: ligand '{name}' has no 3D coordinates")
        n_h = sum(1 for a in rdmol.GetAtoms() if a.GetAtomicNum() == 1)
        implicit = sum(a.GetNumImplicitHs() for a in rdmol.GetAtoms())
        if implicit > 0 or n_h == 0:
            raise LigandError(
                f"{sdf_path}: ligand '{name}' lacks explicit hydrogens. cuFEP does not guess the "
                "protonation state; add hydrogens (with 3D coordinates) for the intended state.")
        props = {k: rdmol.GetProp(k) for k in rdmol.GetPropNames()}
        offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        offmol.name = name
        ligands.append(Ligand(name=name, offmol=offmol, source=str(sdf_path), record=i, properties=props))
    if not ligands:
        raise LigandError(f"{sdf_path}: no molecules found")
    return ligands


def assign_partial_charges(ligand: Ligand, method: str = "am1bcc", cache_dir: str | Path | None = None) -> None:
    """Assign partial charges in place.

    ``am1bcc``       AM1-BCC with AmberTools sqm on the input conformer (OpenFE's default).
    ``am1bcc-nagl``  the OpenFF NAGL graph network trained to reproduce AM1-BCC (fast).
    """
    from openff.units import unit as off_unit

    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"charges_{method}_{ligand.digest()}.json"
        if cache_file.exists():
            q = np.asarray(json.loads(cache_file.read_text())["charges"], dtype=float)
            if len(q) == ligand.n_atoms:
                ligand.offmol.partial_charges = q * off_unit.elementary_charge
                return

    mol = ligand.offmol
    if method == "am1bcc":
        from openff.toolkit.utils import AmberToolsToolkitWrapper
        mol.assign_partial_charges("am1bcc", use_conformers=mol.conformers,
                                   toolkit_registry=AmberToolsToolkitWrapper())
    elif method == "am1bcc-nagl":
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
        mol.assign_partial_charges(NAGL_DEFAULT_MODEL, toolkit_registry=NAGLToolkitWrapper())
    else:
        raise LigandError(f"unknown ligand charge method '{method}' (am1bcc, am1bcc-nagl)")

    q = np.asarray(mol.partial_charges.m_as(off_unit.elementary_charge), dtype=float)
    # Remove the rounding residue so the molecule carries exactly its formal charge.
    q += (ligand.formal_charge - q.sum()) / len(q)
    mol.partial_charges = q * off_unit.elementary_charge
    if cache_file is not None:
        from cufep.files import publish_text

        # used again whenever it exists, so it is never seen half written
        publish_text(cache_file, json.dumps({"ligand": ligand.name, "method": method, "charges": q.tolist()}))
