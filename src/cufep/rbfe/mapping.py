"""Atom mapping of two ligands and the network of ligands.

Mapping
    Kartograf (Ries et al., J. Chem. Theory Comput. 2024; OpenFE's default mapper since 1.7) maps atoms of
    the two 3D poses as they are given: atoms at most ``atom_max_distance`` (0.95 A) apart, hydrogens only on
    hydrogens, whole rings only and no ring that changes, no bond broken (its defaults, OpenFE's). Then the
    rule of OpenFE's ``_remove_constraints`` (from perses): where a constrained bond between two mapped atoms
    changes its length, or is constrained at one end only, the hydrogen of the bond leaves the mapping.

Network
    The edges are given, or planned by LOMAP (Liu et al., J. Comput. Aided Mol. Des. 2013; lomap2's
    ``generate_lomap_network`` with ``default_lomap_score`` and Kartograf, as the OpenFE benchmark of Baumann et
    al. 2026 planned its networks): edges of score at least ``distance_cutoff`` (0.4), every ligand in a
    cycle (``require_cycle_covering``), paths of at most ``max_path_length`` (6) edges.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


class MappingError(ValueError):
    """Two ligands cannot be mapped as asked."""


@dataclass(frozen=True)
class MappingSettings:
    atom_max_distance: float = 0.95
    map_hydrogens_on_hydrogens_only: bool = True
    map_exact_ring_matches_only: bool = True
    allow_partial_fused_rings: bool = True
    allow_bond_breaks: bool = False


def _component(ligand):
    from gufe import SmallMoleculeComponent

    return SmallMoleculeComponent.from_rdkit(ligand.to_rdkit(), name=ligand.name)


def _mapper(settings: MappingSettings):
    from kartograf import KartografAtomMapper

    return KartografAtomMapper(atom_max_distance=settings.atom_max_distance, atom_map_hydrogens=True,
                               map_hydrogens_on_hydrogens_only=settings.map_hydrogens_on_hydrogens_only,
                               map_exact_ring_matches_only=settings.map_exact_ring_matches_only,
                               allow_partial_fused_rings=settings.allow_partial_fused_rings,
                               allow_bond_breaks=settings.allow_bond_breaks)


def kartograf_mapping(ligand_a, ligand_b, settings: MappingSettings | None = None) -> dict[int, int]:
    """Atoms of ligand A -> atoms of ligand B (indices of the molecules), from their poses."""
    settings = settings or MappingSettings()
    a, b = _component(ligand_a), _component(ligand_b)
    mappings = list(_mapper(settings).suggest_mappings(a, b))
    if not mappings or not mappings[0].componentA_to_componentB:
        raise MappingError(f"Kartograf maps no atom of {ligand_a.name} onto {ligand_b.name}: are the poses "
                           "placed on each other?")
    return {int(i): int(j) for i, j in mappings[0].componentA_to_componentB.items()}


def lomap_score(ligand_a, ligand_b, mapping: dict[int, int]) -> float:
    """The default LOMAP score of a mapping (lomap2, as OpenFE uses it)."""
    from gufe import LigandAtomMapping
    from lomap.gufe_bindings.scorers import default_lomap_score

    return float(default_lomap_score(LigandAtomMapping(_component(ligand_a), _component(ligand_b),
                                                       componentA_to_componentB=dict(mapping))))


def constrained_bond_rule(mapping: dict[int, int], system_a, system_b, to_system_a: dict[int, int],
                          to_system_b: dict[int, int], hydrogens_a: set[int], hydrogens_b: set[int]) -> dict[int, int]:
    """OpenFE's ``_remove_constraints``: a mapping (ligand indices) without the hydrogens of constrained bonds
    that change. ``to_system_*`` turn ligand indices into indices of the Systems; ``hydrogens_*`` are ligand
    indices."""
    out = dict(mapping)
    sys_to_a = {v: k for k, v in to_system_a.items()}
    sys_to_b = {v: k for k, v in to_system_b.items()}

    import openmm.unit as unit

    def constraints(system, back) -> dict[tuple[int, int], float]:
        found = {}
        for k in range(system.getNumConstraints()):
            i, j, d = system.getConstraintParameters(k)
            if i in back and j in back:
                found[(back[i], back[j])] = d.value_in_unit(unit.nanometer)
        return found

    old = {key: d for key, d in constraints(system_a, sys_to_a).items() if key[0] in mapping and key[1] in mapping}
    inverse = {v: k for k, v in mapping.items()}
    new = {key: d for key, d in constraints(system_b, sys_to_b).items() if key[0] in inverse and key[1] in inverse}

    def pick_h(i, j, x, y) -> int:
        if i in hydrogens_a or x in hydrogens_b:
            return i
        if j in hydrogens_a or y in hydrogens_b:
            return j
        raise MappingError(f"a constrained bond between heavy atoms changes: A {i}-{j}, B {x}-{y}")

    drop = []
    for (i, j), length in old.items():
        x, y = mapping[i], mapping[j]
        other = new.pop((x, y), None)
        if other is None:
            other = new.pop((y, x), None)
        if other is None or other != length:
            drop.append(pick_h(i, j, x, y))
    for x, y in new:
        drop.append(pick_h(inverse[x], inverse[y], x, y))
    for atom, count in Counter(drop).items():
        if count > 1:
            raise MappingError(f"atom {atom} of A is in more than one constrained bond that changes")
    for atom in drop:
        out.pop(atom, None)
    return out


def plan_network(ligands, settings: MappingSettings | None = None, distance_cutoff: float = 0.4,
                 max_path_length: int = 6, require_cycle_covering: bool = True) -> list[tuple[str, str]]:
    """Edges (name A, name B) of the LOMAP network of the ligands."""
    from lomap.gufe_bindings.network_generation import generate_lomap_network
    from lomap.gufe_bindings.scorers import default_lomap_score

    settings = settings or MappingSettings()
    components = [_component(lig) for lig in ligands]
    network = generate_lomap_network(components, mappers=[_mapper(settings)], scorer=default_lomap_score,
                                     distance_cutoff=distance_cutoff, max_path_length=max_path_length,
                                     require_cycle_covering=require_cycle_covering)
    return sorted((e.componentA.name, e.componentB.name) for e in network.edges)
