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
    al. 2026 planned its networks): pairs of a score below 1 - ``distance_cutoff`` (0.6) left out except those
    that join the parts the others leave (lomap2's ``GraphGen``; docs/rbfe.md, "The network"), every ligand in a
    cycle (``require_cycle_covering``), paths of at most ``max_path_length`` (6) edges.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


class MappingError(ValueError):
    """Two ligands cannot be mapped as asked."""


@dataclass(frozen=True)
class MappingSettings:
    atom_max_distance: float = 0.95
    map_hydrogens_on_hydrogens_only: bool = True
    map_exact_ring_matches_only: bool = True
    allow_partial_fused_rings: bool = True
    allow_bond_breaks: bool = False
    #: a custom core: only atoms in a match of this SMARTS in both ligands are mapped (None: Kartograf's mapping)
    core_smarts: str | None = None


def mapping_settings(m) -> MappingSettings:
    """The settings of the mapping from the ``mapping`` section of a relative configuration (one function for every
    place that maps)."""
    return MappingSettings(atom_max_distance=m.atom_max_distance_A,
                           map_hydrogens_on_hydrogens_only=m.map_hydrogens_on_hydrogens_only,
                           map_exact_ring_matches_only=m.map_exact_ring_matches_only,
                           allow_partial_fused_rings=m.allow_partial_fused_rings,
                           allow_bond_breaks=m.allow_bond_breaks, core_smarts=m.core_smarts)


#: the most matches of ``mapping.core_smarts`` (sets of atoms) taken in one ligand: a rule of computation, the
#: limit of the search of RDKit, which stops there. A pattern with more is refused, since a match beyond the limit
#: would be missing from the core (eleventh review of PR #19).
MAX_CORE_MATCHES = 1000


def core_atoms(molecule, smarts: str) -> set[int]:
    """The atoms of an RDKit molecule (with its hydrogens) in any match of ``smarts``, its stereochemistry respected (a
    pattern without it matches either configuration; review of PR #19: a pattern of one configuration matched the
    other); MappingError without one, or with more than ``MAX_CORE_MATCHES``. A match is a set of atoms: the orders of
    the same atoms (the permutations of a symmetric pattern) are one match. They were counted one by one, and the
    1296 orders of one tert-butyl group used up the search before the other group of the ligand was reached
    (eleventh review of PR #19)."""
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise MappingError(f"mapping.core_smarts {smarts!r} is not a SMARTS pattern")
    matches = molecule.GetSubstructMatches(pattern, uniquify=True, useChirality=True,
                                           maxMatches=MAX_CORE_MATCHES + 1)
    name = f" of {molecule.GetProp('_Name')}" if molecule.HasProp("_Name") else ""
    if len(matches) > MAX_CORE_MATCHES:
        raise MappingError(f"mapping.core_smarts {smarts!r} matches more than {MAX_CORE_MATCHES} sets of atoms{name}, "
                           f"the most that are taken: a match beyond them would be left out of the core; give a "
                           f"pattern with fewer matches")
    if not matches:
        raise MappingError(f"mapping.core_smarts {smarts!r} matches no atom{name}")
    return {int(i) for match in matches for i in match}


def bridging_dummies(molecule_a, molecule_b, mapping: dict[int, int]) -> list[str]:
    """The groups of atoms unique to one ligand (connected, of an RDKit molecule with its hydrogens) bonded to two or
    more mapped atoms of it. The hybrid keeps the bonds and angles of such a group with the mapped atoms at every
    lambda, so that at the end state without it it still holds the mapped atoms at its distances (eleventh review of
    PR #19: propane and 2,2-difluoropropane with the custom core ``[CH3]`` mapped the two methyl carbons, and the
    central carbon of each end tied them at both ends of the transformation). Empty where there is none."""
    out = []
    for name, molecule, mapped in (("A", molecule_a, set(mapping)), ("B", molecule_b, set(mapping.values()))):
        unique = {atom.GetIdx() for atom in molecule.GetAtoms()} - mapped
        seen: set[int] = set()
        for start in sorted(unique):
            if start in seen:
                continue
            group, todo = {start}, [start]
            while todo:
                for neighbour in molecule.GetAtomWithIdx(todo.pop()).GetNeighbors():
                    j = neighbour.GetIdx()
                    if j in unique and j not in group:
                        group.add(j)
                        todo.append(j)
            seen |= group
            anchors = sorted({nb.GetIdx() for i in group for nb in molecule.GetAtomWithIdx(i).GetNeighbors()
                              if nb.GetIdx() in mapped})
            if len(anchors) > 1:
                out.append(f"the atoms {sorted(group)} unique to {name} are bonded to the mapped atoms {anchors} of "
                           f"{name}")
    return out


def within_core(mapping: dict[int, int], core_a: set[int], core_b: set[int]) -> dict[int, int]:
    """The pairs of ``mapping`` whose two atoms are in the custom core of their ligand."""
    return {i: j for i, j in mapping.items() if i in core_a and j in core_b}


class CoreMapper:
    """Kartograf's mapper whose mappings keep only the atoms of the custom core (for the network of LOMAP, which
    calls ``suggest_mappings``)."""

    def __init__(self, mapper, smarts: str) -> None:
        self.mapper, self.smarts = mapper, smarts

    def suggest_mappings(self, a, b):
        from gufe import LigandAtomMapping

        core_a, core_b = core_atoms(a.to_rdkit(), self.smarts), core_atoms(b.to_rdkit(), self.smarts)
        for m in self.mapper.suggest_mappings(a, b):
            kept = within_core(dict(m.componentA_to_componentB), core_a, core_b)
            if kept:
                yield LigandAtomMapping(a, b, componentA_to_componentB=kept)


def without_mixed_hydrogens(molecule_a, molecule_b, mapping: dict[int, int]) -> dict[int, int]:
    """The mapping without the pairs that the rule of the constrained bonds (:func:`constrained_bond_rule`) takes out
    of it where the bonds of hydrogens are constrained, as far as they can join mapped atoms: a hydrogen mapped onto
    a heavy atom (or the other way round) whose bonded atom is mapped. Its bond is constrained in one end and not in
    the other. (Pairs of two hydrogens that it takes out leave a hydrogen, which joins nothing.)"""
    inverse = {j: i for i, j in mapping.items()}
    out = {}
    for i, j in mapping.items():
        h_a = molecule_a.GetAtomWithIdx(i).GetAtomicNum() == 1
        h_b = molecule_b.GetAtomWithIdx(j).GetAtomicNum() == 1
        if h_a != h_b:
            hydrogen = molecule_a.GetAtomWithIdx(i) if h_a else molecule_b.GetAtomWithIdx(j)
            mapped = mapping if h_a else inverse
            if any(nb.GetIdx() in mapped for nb in hydrogen.GetNeighbors()):
                continue
        out[i] = j
    return out


def as_prepared(molecule_a, molecule_b, mapping: dict[int, int], hydrogen_constraints: bool) -> dict[int, int]:
    """The mapping as the preparation leaves it as far as atoms that join mapped atoms go: without the pairs that the
    rule of the constrained bonds takes out where the bonds of hydrogens are constrained
    (:func:`without_mixed_hydrogens`), as it is without constraints. For what is planned and shown before an edge
    is prepared (the network, `cufep rbfe plan`), so that they check what the preparation checks (twelfth and
    thirteenth reviews of PR #19)."""
    return without_mixed_hydrogens(molecule_a, molecule_b, mapping) if hydrogen_constraints else dict(mapping)


def hydrogen_constraints_of(rcfg) -> bool:
    """Whether the bonds of hydrogens are constrained in the systems of a relative run (``dynamics.constraints``)."""
    return rcfg.dynamics.constraints in ("auto", "hbonds")


class WithoutBridges:
    """A mapper whose mappings in which atoms unique to one ligand join two mapped atoms (:func:`bridging_dummies`)
    are left out, for the network of LOMAP: an edge is planned only with a mapping the preparation takes. With
    ``hydrogen_constraints`` the mapping is checked as the rule of the constrained bonds leaves it
    (:func:`without_mixed_hydrogens`; twelfth review of PR #19: the oxygen of oxirane mapped onto a hydrogen of ethane
    left the mapping there and joined the two carbons)."""

    def __init__(self, mapper, hydrogen_constraints: bool = True) -> None:
        self.mapper, self.hydrogen_constraints = mapper, hydrogen_constraints

    def suggest_mappings(self, a, b):
        for m in self.mapper.suggest_mappings(a, b):
            mol_a, mol_b = a.to_rdkit(), b.to_rdkit()
            mapping = as_prepared(mol_a, mol_b, dict(m.componentA_to_componentB), self.hydrogen_constraints)
            if not bridging_dummies(mol_a, mol_b, mapping):
                yield m


def _component(ligand):
    from gufe import SmallMoleculeComponent

    return SmallMoleculeComponent.from_rdkit(ligand.to_rdkit(), name=ligand.name)


def _mapper(settings: MappingSettings, exclude=frozenset()):
    from kartograf import KartografAtomMapper

    def without_excluded(mol_a, mol_b, mapping: dict[int, int]) -> dict[int, int]:
        return {i: j for i, j in mapping.items() if (i, j) not in exclude}

    return KartografAtomMapper(atom_max_distance=settings.atom_max_distance, atom_map_hydrogens=True,
                               map_hydrogens_on_hydrogens_only=settings.map_hydrogens_on_hydrogens_only,
                               map_exact_ring_matches_only=settings.map_exact_ring_matches_only,
                               allow_partial_fused_rings=settings.allow_partial_fused_rings,
                               allow_bond_breaks=settings.allow_bond_breaks,
                               additional_mapping_filter_functions=[without_excluded] if exclude else None)


def kartograf_mapping(ligand_a, ligand_b, settings: MappingSettings | None = None,
                      exclude: frozenset = frozenset()) -> dict[int, int]:
    """Atoms of ligand A -> atoms of ligand B (indices of the molecules), from their poses. ``exclude``: pairs of
    atoms (A, B) that Kartograf does not map, passed to it as a filter of its own kind
    (``additional_mapping_filter_functions``); it then keeps, as with its own filters, the largest connected part."""
    settings = settings or MappingSettings()
    a, b = _component(ligand_a), _component(ligand_b)
    mappings = list(_mapper(settings, frozenset(exclude)).suggest_mappings(a, b))
    if not mappings or not mappings[0].componentA_to_componentB:
        raise MappingError(f"Kartograf maps no atom of {ligand_a.name} onto {ligand_b.name}: are the poses "
                           "placed on each other?")
    mapping = {int(i): int(j) for i, j in mappings[0].componentA_to_componentB.items()}
    if settings.core_smarts:
        mapping = within_core(mapping, core_atoms(ligand_a.to_rdkit(), settings.core_smarts),
                              core_atoms(ligand_b.to_rdkit(), settings.core_smarts))
        if not mapping:
            raise MappingError(f"no atom of {ligand_a.name} and {ligand_b.name} that Kartograf maps is in the custom "
                               f"core {settings.core_smarts!r}")
    bridges = bridging_dummies(ligand_a.to_rdkit(), ligand_b.to_rdkit(), mapping)
    if bridges:
        raise MappingError(f"the mapping of {ligand_a.name} onto {ligand_b.name} cannot be used: {'; '.join(bridges)} "
                           "(the hybrid would keep their bonds and angles at both ends)")
    return mapping


def _hydrogens(atom) -> list[int]:
    return [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1]


def _hydrogen_lengths(system, to_system: dict[int, int], molecule) -> dict[int, dict[int, float]]:
    """Heavy atom of the ligand -> {its hydrogen: the length of their constraint in nm} (ligand indices)."""
    import openmm.unit as unit

    back = {v: k for k, v in to_system.items()}
    out: dict[int, dict[int, float]] = {}
    for k in range(system.getNumConstraints()):
        x, y, d = system.getConstraintParameters(k)
        if x not in back or y not in back:
            continue
        p, q = back[x], back[y]
        h_p = molecule.GetAtomWithIdx(p).GetAtomicNum() == 1
        if h_p == (molecule.GetAtomWithIdx(q).GetAtomicNum() == 1):
            continue
        heavy, hydrogen = (q, p) if h_p else (p, q)
        out.setdefault(heavy, {})[hydrogen] = float(d.value_in_unit(unit.nanometer))
    return out


def _one_length(lengths) -> bool:
    """Constraint lengths that OpenMM puts into one SHAKE cluster (``ShakeCluster::addAtom``: within 1e-8)."""
    lengths = list(lengths)
    return all(abs(d - lengths[0]) <= 1e-8 * lengths[0] for d in lengths)


def mapping_for_the_plugin(ligand_a, ligand_b, mapping: dict[int, int], settings, system_a, system_b,
                           to_system_a: dict[int, int], to_system_b: dict[int, int]
                           ) -> tuple[dict[int, int], list[str]]:
    """Kartograf's mapping changed where the hybrid would put on one atom hydrogens that the integrator of the
    cuPotential plugin cannot constrain: it constrains three hydrogens of one length on an atom at most (SHAKE; the
    plugin: "only SETTLE and SHAKE constraints are implemented"), and the hydrogens of both ligands that are not mapped
    stay on a mapped atom of the hybrid. The lengths are those of the constraints of the end states (``system_*``,
    whose ligand atoms ``to_system_*`` give). Returns the mapping and what was changed; a mapping that needs neither
    change is returned as it is.

    1. A mapped pair of heavy atoms that both carry hydrogens constrained to other lengths (the O of an OH onto the C
       of a CH3, a CH2 onto an NH, a carbon of four bonds onto one of three): such hydrogens cannot be mapped onto each
       other (their constrained bond would change), so both stay on the atom. Kartograf maps again without that pair,
       passed to it as a filter of its own kind, as its ``filter_element_changes`` ("Forces a mapping to exclude any
       alchemical element changes in the core") leaves out every pair of other elements, and it then keeps whole
       rings only and the largest connected part, as after its own filters.
    2. A mapped heavy atom on which the hybrid would carry more than three hydrogens (a methyl group turned between
       the two poses, whose hydrogens are farther apart than Kartograf's ``atom_max_distance``): its unmapped
       hydrogens of A and of B are mapped onto each other, the closest pair first, as LOMAP maps the hydrogens of its
       mapped heavy atoms (lomap2, ``MCS.all_atom_match_list``: "In 3D mode, ensure that maps happen to the closest
       atom in 3D coordinates - this gets mappings to prochiral hydrogens correct")."""
    mol_a, mol_b = ligand_a.to_rdkit(), ligand_b.to_rdkit()
    len_a = _hydrogen_lengths(system_a, to_system_a, mol_a)
    len_b = _hydrogen_lengths(system_b, to_system_b, mol_b)
    notes = []
    exclude = frozenset((i, j) for i, j in mapping.items()
                        if len_a.get(i) and len_b.get(j)
                        and not _one_length(list(len_a[i].values()) + list(len_b[j].values())))
    if exclude:
        before = len(mapping)
        mapping = kartograf_mapping(ligand_a, ligand_b, settings, exclude)
        for i, j in sorted(exclude):
            notes.append(f"atom {i} ({mol_a.GetAtomWithIdx(i).GetSymbol()}) of {ligand_a.name} is not mapped onto atom "
                         f"{j} ({mol_b.GetAtomWithIdx(j).GetSymbol()}) of {ligand_b.name}: their hydrogens are "
                         f"constrained to {10 * next(iter(len_a[i].values())):.4f} and "
                         f"{10 * next(iter(len_b[j].values())):.4f} A, which the plugin cannot constrain on one atom")
        notes.append(f"Kartograf mapped again without them: {before} -> {len(mapping)} mapped atoms")
    pos_a, pos_b = np.asarray(ligand_a.positions_nm()), np.asarray(ligand_b.positions_nm())
    mapping = dict(mapping)
    for i, j in sorted(mapping.items()):
        if mol_a.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        inverse = {v: k for k, v in mapping.items()}
        on_a, on_b = _hydrogens(mol_a.GetAtomWithIdx(i)), _hydrogens(mol_b.GetAtomWithIdx(j))
        free_a = [h for h in on_a if h not in mapping]
        free_b = [h for h in on_b if h not in inverse]
        if len([h for h in on_a if mapping.get(h) in on_b]) + len(free_a) + len(free_b) <= 3:
            continue
        added = 0
        while free_a and free_b:
            x, y = min(((x, y) for x in free_a for y in free_b),
                       key=lambda pair: float(np.linalg.norm(pos_a[pair[0]] - pos_b[pair[1]])))
            mapping[x] = y
            free_a.remove(x)
            free_b.remove(y)
            added += 1
        notes.append(f"{added} hydrogen(s) of atom {i} of {ligand_a.name} mapped onto those of atom {j} of "
                     f"{ligand_b.name}, the closest first (the hybrid would carry more than three on the atom)")
    return mapping, notes


def hydrogens_the_plugin_cannot_constrain(mapping: dict[int, int], system_a, system_b, to_system_a: dict[int, int],
                                          to_system_b: dict[int, int], molecule_a, molecule_b) -> list[str]:
    """The mapped heavy atoms on which the hybrid of ``mapping`` (ligand indices, as the preparation leaves it) would
    carry hydrogens that the integrator of the cuPotential plugin cannot constrain: more than three, or constrained
    to lengths that differ (OpenMM's SHAKE clusters, ``ShakeCluster::addAtom`` in
    ``platforms/common/src/IntegrationUtilities.cpp`` 8.4: three at most, of one length within 1e-8)."""
    len_a = _hydrogen_lengths(system_a, to_system_a, molecule_a)
    len_b = _hydrogen_lengths(system_b, to_system_b, molecule_b)
    inverse = {v: k for k, v in mapping.items()}
    problems = []
    for i, j in sorted(mapping.items()):
        if molecule_a.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        on_a, on_b = len_a.get(i, {}), len_b.get(j, {})
        # each hydrogen once, with its length in A and, mapped, in B (the hybrid keeps one constraint of a mapped
        # hydrogen only where the two are one length: constrained_bond_rule)
        count = len(on_a) + len([h for h in on_b if inverse.get(h) not in on_a])
        lengths = list(on_a.values()) + list(on_b.values())
        if count > 3 or (lengths and not _one_length(lengths)):
            problems.append(f"atom {i} of A ({j} of B) would carry {count} constrained hydrogen(s) of lengths "
                            + ", ".join(f"{10 * d:.4f}" for d in sorted(set(lengths))) + " A")
    return problems


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
                 max_path_length: int = 6, require_cycle_covering: bool = True,
                 hydrogen_constraints: bool = True) -> list[tuple[str, str]]:
    """Edges (name A, name B) of the LOMAP network of the ligands (``hydrogen_constraints``: the bonds of hydrogens
    are constrained, as ``dynamics.constraints`` sets them)."""
    from lomap.gufe_bindings.network_generation import generate_lomap_network
    from lomap.gufe_bindings.scorers import default_lomap_score

    settings = settings or MappingSettings()
    components = [_component(lig) for lig in ligands]
    mapper = WithoutBridges(_mapper(settings) if not settings.core_smarts
                            else CoreMapper(_mapper(settings), settings.core_smarts), hydrogen_constraints)
    network = generate_lomap_network(components, mappers=[mapper], scorer=default_lomap_score,
                                     distance_cutoff=distance_cutoff, max_path_length=max_path_length,
                                     require_cycle_covering=require_cycle_covering)
    return sorted((e.componentA.name, e.componentB.name) for e in network.edges)
