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
