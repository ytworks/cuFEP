"""Force field assembly and System creation."""
from __future__ import annotations

from dataclasses import dataclass

import openmm as mm
import openmm.app as app
import openmm.unit as unit


@dataclass
class ForceFieldSpec:
    protein: str = "amber14/protein.ff14SB.xml"
    water: str = "amber14/tip3p.xml"
    lipid: str | None = "amber19/lipid21.xml"
    ligand: str = "openff-2.2.1"
    extra: tuple[str, ...] = ()


@dataclass
class SystemSpec:
    """Nonbonded and constraint settings (defaults follow OpenFE's ABFE protocol)."""

    cutoff_nm: float = 0.9
    ewald_tolerance: float = 5e-4          # fixed by the cuPotential plugin
    constraints: str = "hbonds"
    rigid_water: bool = True
    hydrogen_mass_amu: float = 3.0
    remove_cm_motion: bool = True


def make_forcefield(spec: ForceFieldSpec, ligands=(), include_lipid: bool = False) -> app.ForceField:
    """Create a ForceField with SMIRNOFF templates registered for the given ligands.

    ``ligands`` are :class:`cufep.prep.ligand.Ligand` objects whose partial charges are
    already assigned; the template generator uses those charges as they are.
    """
    files = [spec.protein, spec.water]
    if include_lipid and spec.lipid:
        files.append(spec.lipid)
    files.extend(spec.extra)
    forcefield = app.ForceField(*files)
    molecules = [lig.offmol for lig in ligands]
    if molecules:
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        for mol in molecules:
            if mol.partial_charges is None:
                raise ValueError(f"ligand '{mol.name}' has no partial charges; assign them first")
        generator = SMIRNOFFTemplateGenerator(molecules=molecules, forcefield=spec.ligand)
        forcefield.registerTemplateGenerator(generator.generator)
    return forcefield


_CONSTRAINTS = {"none": None, "hbonds": app.HBonds, "allbonds": app.AllBonds}


def create_system(forcefield: app.ForceField, topology: app.Topology, spec: SystemSpec | None = None) -> mm.System:
    """Create a periodic PME System."""
    spec = spec or SystemSpec()
    kwargs = dict(
        nonbondedMethod=app.PME,
        nonbondedCutoff=spec.cutoff_nm * unit.nanometer,
        constraints=_CONSTRAINTS[spec.constraints.lower()],
        rigidWater=spec.rigid_water,
        ewaldErrorTolerance=spec.ewald_tolerance,
        removeCMMotion=spec.remove_cm_motion,
    )
    if spec.hydrogen_mass_amu and spec.hydrogen_mass_amu > 1.01:
        kwargs["hydrogenMass"] = spec.hydrogen_mass_amu * unit.amu
    system = forcefield.createSystem(topology, **kwargs)
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setUseDispersionCorrection(True)
            force.setUseSwitchingFunction(False)
    return system
