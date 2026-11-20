"""The water models cuFEP can build: one table for the check of the configuration, the
solvation and the bilayer.

``Modeller.addSolvent`` builds its water box from one of a few templates (``model``), and
``ForceField.createSystem`` then needs every water to match the water of the force field
file, extra sites included: a 4-site file cannot use a box of 3-site waters (createSystem stops
with "no template found" for HOH). Measured with OpenMM 8.4 for every file below against the
boxes tip3p, spce and tip4pew: the 3-site files take a tip3p or spce box, the 4-site files only
a tip4pew box. Files outside this table are refused before anything is prepared.
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


def water_problem(water_file: str, membrane: bool) -> str | None:
    """Why ``forcefield.water`` cannot be built, or None.

    The bilayer of ``membrane.enabled`` comes with the 3-site waters of OpenMM's lipid patch, and
    cuFEP adds 3-site waters to the protein's cavities inside it, so a membrane run needs a
    3-site water model.
    """
    model = WATER_MODELS.get(str(water_file))
    if model is None:
        return (f"forcefield.water = {water_file!r} is not a water model cuFEP can build; use one of "
                + ", ".join(WATER_MODELS))
    if membrane and model.sites != 3:
        return (f"forcefield.water = {water_file!r} has {model.sites} sites per water, and the bilayer of "
                "membrane.enabled is built with 3-site waters; use one of "
                + ", ".join(name for name, m in WATER_MODELS.items() if m.sites == 3))
    return None


def water_model_of(water_file: str) -> WaterModel:
    """The table entry of ``forcefield.water``; ValueError if it has none."""
    problem = water_problem(water_file, membrane=False)
    if problem:
        raise ValueError(problem)
    return WATER_MODELS[str(water_file)]
