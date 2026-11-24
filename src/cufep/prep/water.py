"""The water models cuFEP can build: one table for the check of the configuration, the
solvation and the bilayer.

``Modeller.addSolvent`` builds its water box from one of a few templates (``model``), and
``ForceField.createSystem`` then needs every water to match the water of the force field
file, extra sites included: a 4-site file cannot use a box of 3-site waters (createSystem stops
with "no template found" for HOH). Measured with OpenMM 8.4 for every file below against the
boxes tip3p, spce and tip4pew: the 3-site files take a tip3p or spce box, the 4-site files only
a tip4pew box. Files outside this table are refused before anything is prepared.

A bilayer (``membrane.enabled``) is built with 3-site waters whatever the model:
``Modeller.addMembrane`` makes a System of its lipid patch, whose waters have 3 sites, and cuFEP
adds 3-site waters to the protein's cavities inside it. With a model of more sites the bilayer is
built with the force field of :data:`BILAYER_BUILD_WATER`, and every water gets its extra site
afterwards (:func:`cufep.prep.membrane.add_membrane`), as a crystal water does.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaterModel:
    box: str          # the ``model`` of Modeller.addSolvent that builds these waters
    sites: int        # sites per water molecule in the force field


#: OpenMM's Amber14 water files (the ones that go with ``amber14/protein.ff14SB.xml``)
WATER_MODELS: dict[str, WaterModel] = {
    "amber14/tip3p.xml": WaterModel("tip3p", 3),
    "amber14/tip3pfb.xml": WaterModel("tip3p", 3),
    "amber14/opc3.xml": WaterModel("tip3p", 3),
    "amber14/spce.xml": WaterModel("spce", 3),
    "amber14/tip4pew.xml": WaterModel("tip4pew", 4),
    "amber14/tip4pfb.xml": WaterModel("tip4pew", 4),
    "amber14/opc.xml": WaterModel("tip4pew", 4),
}

#: the 3-site water file with which a bilayer is built when the water model has more sites
BILAYER_BUILD_WATER = "amber14/tip3p.xml"


def water_problem(water_file: str, setting: str = "forcefield.water") -> str | None:
    """Why the water model of ``setting`` cannot be built, or None."""
    if str(water_file) not in WATER_MODELS:
        return (f"{setting} = {water_file!r} is not a water model cuFEP can build; use one of "
                + ", ".join(WATER_MODELS))
    return None


def water_model_of(water_file: str) -> WaterModel:
    """The table entry of ``forcefield.water``; ValueError if it has none."""
    problem = water_problem(water_file)
    if problem:
        raise ValueError(problem)
    return WATER_MODELS[str(water_file)]


#: What grand canonical moves of water (``sampling.gcmc``) need of a water model, with the settings a run
#: has (PME, the real-space cutoff of 0.9 nm, the dispersion correction, 298.15 K, 1 bar): the excess
#: chemical potential of a water in bulk water, kcal/mol (minus the free energy of decoupling one water
#: from the liquid), and the standard volume, nm^3 (the mean volume of the liquid per molecule). Both are
#: of the water model AND of how its interactions are cut: the values of the literature for TIP3P, -6.09
#: kcal/mol and 30.345 A^3 (Samways et al. 2020; the defaults of grand), are of grand's settings, without
#: a dispersion correction and with Lennard-Jones switched off between 1.0 and 1.2 nm, and none is
#: published for OPC. Measured with cuFEP itself on the cuPotential backend, two runs each (25 windows of 3
#: ns, 1,386 and 1,384 waters): TIP3P -6.125 ± 0.053 and -6.129 ± 0.049 kcal/mol, 30.319 ± 0.009 and
#: 30.343 ± 0.005 A^3; OPC -8.051 ± 0.068 and -8.046 ± 0.074 kcal/mol, 29.995 ± 0.009 and 30.003 ± 0.006
#: A^3. For TIP3P the values of the literature agree with the measured ones within their errors and are
#: used; for OPC the measured ones
#: (``spikes/m34_water_chemical_potential.py``; ``docs/configuration.md`` gives them with their errors).
#: A model without an entry needs both numbers in the configuration.
BULK_WATER: dict[str, tuple[float, float]] = {
    "amber14/tip3p.xml": (-6.09, 0.030345),
    "amber14/opc.xml": (-8.05, 0.030000),
}


def bulk_water_values(water_file: str) -> tuple[float, float] | None:
    """(excess chemical potential in kcal/mol, standard volume in nm^3) of the water model, or None."""
    return BULK_WATER.get(str(water_file))
