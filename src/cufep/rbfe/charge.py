"""Edges that change the net charge of the ligand: an alchemical water keeps the charge of the box.

The method (the co-alchemical water of Chen et al., J. Chem. Theory Comput. 2018, 14, 6346, whose full text could not
be obtained; the rule here is that of OpenFE's relative protocol, ``explicit_charge_correction``, from perses'
``transform_waters_into_ions``, read in its source, openfe main of 2026-11):
    the charge difference of an edge is dq = q(A) - q(B) (formal charges of the ligands); |dq| = 1 is treated, a larger
    change is refused (OpenFE: "Absolute charge changes greater than 1 are currently not supported");
    one water is chosen in the configuration of end state A: among the waters with no atom within
    ``cutoff_nm`` (OpenFE's 0.8 nm, ``explicit_charge_correction_cutoff``) of an atom of the solutes (every residue
    that is neither a water nor a single-atom ion: the protein, the ligand, lipids), periodic distances, the one
    of the lowest residue index (OpenFE takes the first of its list of such waters);
    in end state B that water is an ion of the sign of dq: its oxygen takes the charge, sigma and epsilon of the
    most abundant monovalent ion of that sign in the system (else Na+ or Cl- of the force field), its hydrogens
    charge 0 (their sigma and epsilon kept), so that A and B have the same total charge;
    in the hybrid (:mod:`cufep.rbfe.hybrid`) its parameters go from those of A to those of B as those of a core atom
    of the ligand: charges with ``lambda_electrostatics_core``, the Lennard-Jones of each of its pairs interpolated
    linearly with ``lambda_sterics_core``. The water keeps its geometry (rigid) and its masses.
In both legs of the edge. The water is not one of the waters of the grand canonical moves.
"""
from __future__ import annotations

import numpy as np

#: OpenFE's ``explicit_charge_correction_cutoff``
WATER_CUTOFF_NM = 0.8
#: the largest change of the net charge that is treated (OpenFE's)
MAX_CHARGE_CHANGE = 1
#: the name of the residue of the alchemical water in the topology of a hybrid (so that the grand canonical moves,
#: which take the residues named HOH, leave it out)
ALCHEMICAL_WATER_RESNAME = "ALW"
#: the ion of the force field that a system without one of the sign gets (OpenFE's fallback), by the sign of dq
FALLBACK_IONS = {1: ("NA", "Na", 1), -1: ("CL", "Cl", -1)}


class ChargeChangeError(ValueError):
    """An edge whose change of charge cannot be treated, or a system without a water to take it."""


def charge_change(ligand_a, ligand_b) -> int:
    """dq = q(A) - q(B), the formal charges of the two ligands."""
    return int(ligand_a.formal_charge) - int(ligand_b.formal_charge)


def charge_change_problems(ligands, edges) -> list[str]:
    """The edges whose change of the net charge is larger than one charge: refused before a run writes anything."""
    by_name = {lig.name: lig for lig in ligands}
    out = []
    for a, b in edges:
        dq = charge_change(by_name[a], by_name[b])
        if abs(dq) > MAX_CHARGE_CHANGE:
            out.append(f"the edge {a} -> {b} changes the net charge by {-dq:+d} (from {by_name[a].formal_charge:+d} "
                       f"to {by_name[b].formal_charge:+d}); an alchemical water takes a change of one charge only")
    return out


def _is_solvent(residue) -> bool:
    from cufep.prep.protein import WATER

    return residue.name in WATER or len(list(residue.atoms())) == 1


def alchemical_water(topology, positions_nm, cutoff_nm: float = WATER_CUTOFF_NM) -> list[int]:
    """The atoms (indices of the system) of the water that takes the charge (the module's rule)."""
    import mdtraj
    import openmm.unit as unit

    from cufep.prep.protein import WATER

    positions = np.asarray(positions_nm, dtype=float)
    traj = mdtraj.Trajectory(positions[None], mdtraj.Topology.from_openmm(topology))
    box = topology.getPeriodicBoxVectors()
    if box is None:
        raise ChargeChangeError("the system has no periodic cell to choose the alchemical water in")
    # the distances are of the nearest image only with the cell (review of PR #19: without it a water 0.2 nm from the
    # solutes across the boundary of the cell counted as 2.8 nm away)
    traj.unitcell_vectors = np.asarray(box.value_in_unit(unit.nanometer), dtype=float)[None]
    residues = list(topology.residues())
    solute = [a.index for r in residues if not _is_solvent(r) for a in r.atoms()]
    waters = [r for r in residues if r.name in WATER]
    if not waters:
        raise ChargeChangeError("the system has no water to take the change of the net charge")
    water_atoms = np.array([a.index for r in waters for a in r.atoms()], dtype=np.int64)
    near = set(int(i) for i in mdtraj.compute_neighbors(traj, float(cutoff_nm), np.asarray(solute, dtype=np.int64),
                                                        haystack_indices=water_atoms, periodic=True)[0])
    for r in sorted(waters, key=lambda r: r.index):
        atoms = [a.index for a in r.atoms()]
        if not near & set(atoms):
            return atoms
    raise ChargeChangeError(f"no water is farther than {cutoff_nm:g} nm from the solutes to take the change of the "
                            "net charge")


def ion_parameters(system, topology, sign: int, forcefield) -> dict:
    """Charge (e), sigma (nm) and epsilon (kJ/mol) of the most abundant monovalent ion of ``sign`` in the system,
    else of Na+ or Cl- of ``forcefield`` (OpenFE's ``_get_ion_parameters``)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    counts: dict[str, list[int]] = {}
    for r in topology.residues():
        atoms = list(r.atoms())
        if len(atoms) != 1:
            continue
        q = nb.getParticleParameters(atoms[0].index)[0].value_in_unit(u.elementary_charge)
        if abs(abs(q) - 1.0) < 0.01 and np.sign(q) == sign:
            counts.setdefault(r.name, []).append(atoms[0].index)
    if counts:
        name = max(sorted(counts), key=lambda k: len(counts[k]))
        q, s, e = nb.getParticleParameters(counts[name][0])
        source = "the system"
    else:
        name, element, charge = FALLBACK_IONS[int(sign)]
        top = app.Topology()
        residue = top.addResidue(name, top.addChain())
        top.addAtom(element.upper(), app.Element.getBySymbol(element), residue)
        ion_system = forcefield.createSystem(top, nonbondedMethod=app.NoCutoff)
        q, s, e = next(f for f in ion_system.getForces()
                       if isinstance(f, mm.NonbondedForce)).getParticleParameters(0)
        source = "the force field"
    return {"name": name, "charge": q.value_in_unit(u.elementary_charge), "sigma_nm": s.value_in_unit(u.nanometer),
            "epsilon_kj_mol": e.value_in_unit(u.kilojoule_per_mole), "from": source}


def water_into_ion(system, water_atoms: list[int], ion: dict) -> None:
    """End state B: the water ``water_atoms`` (oxygen first) of ``system`` (changed in place) is the ion: its oxygen
    with the parameters of ``ion``, its hydrogens with charge 0."""
    import openmm as mm
    import openmm.unit as u

    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    oxygen, *rest = (int(i) for i in water_atoms)
    nb.setParticleParameters(oxygen, ion["charge"] * u.elementary_charge, ion["sigma_nm"] * u.nanometer,
                             ion["epsilon_kj_mol"] * u.kilojoule_per_mole)
    for i in rest:
        _q, s, e = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0 * u.elementary_charge, s, e)


def oxygen_first(topology, atoms: list[int]) -> list[int]:
    """The atoms of a water with its oxygen first."""
    by_index = {a.index: a for a in topology.atoms()}
    oxygen = [i for i in atoms if by_index[i].element is not None and by_index[i].element.symbol == "O"]
    if len(oxygen) != 1:
        raise ChargeChangeError(f"the water of atoms {atoms} has {len(oxygen)} oxygen atoms")
    return oxygen + [i for i in atoms if i != oxygen[0]]
