"""Constraints of a hybrid System that the cuPotential plugin can run.

The plugin's integrator has SETTLE and SHAKE only ("only SETTLE and SHAKE constraints are implemented",
libOpenMMNvidiaCUDA.so); OpenMM's own integrators give the constraints that SHAKE cannot take to CCMA. A hybrid
System has the constraints of A and those of B that A does not have (OpenFE), so an atom of the core can carry
hydrogens of both ligands that are not mapped: more than three, or of other lengths (a methyl carbon that carries the
hydrogens of A and of B, the carbon of a CH2 mapped onto an NH). Such an atom is no SHAKE cluster, and the leg could
not run on the plugin.

The rule here, which keeps Kartograf's mapping as it is: OpenMM's clusters are found as
``IntegrationUtilities`` finds them (``platforms/common/src/IntegrationUtilities.cpp`` of OpenMM 8.4: the central atom
of a constraint is the atom with more than one constraint, else the atom of the lower index; a peripheral atom has one
constraint only; ``ShakeCluster::addAtom`` takes at most three peripheral atoms of one length and one mass, within a
relative 1e-8). At a central atom whose constraints make no such cluster, the largest group of constraints of one
length and one peripheral mass is kept, at most three (of equal groups the one with more mapped hydrogens, then more
of A, then of lower indices), and every other constraint of that atom is taken out; the hydrogen of a constraint taken
out moves on the bond term the force field gives that bond (the force field of the preparation with no constraints,
:func:`ligand_bond_terms`), as a bond of a unique atom of the hybrid: on at full strength at every lambda (OpenFE), of
both ends for a mapped pair. A constraint is the force field's bond held at its length (the bond terms of constrained
bonds are left out of the prepared Systems); the bond term is the force field itself.

Masses: the mapped atom that carries such hydrogens has the larger of its masses in A and in B, where OpenFE takes
their mean (:func:`cufep.rbfe.hybrid.make_hybrid_system`). With the hydrogen masses repartitioned (3 amu) the mean can
be light (a CH3 onto an NH2: 8.0 amu): MD at 4 fs on the plugin then sampled the N-H bonds of the NH2 with a standard
deviation of 0.040 A, against 0.025 A of the same System at 0.5 fs and 0.0248 A of the harmonic term; with the larger
mass (10.0 amu) 0.025 A (``spikes/out/m46``, ``tests/gpu``). Masses enter no configurational integral, so no free
energy.
"""
from __future__ import annotations

from collections import defaultdict

#: ``ShakeCluster::addAtom``: lengths and inverse masses within this relative difference make one cluster
SHAKE_TOLERANCE = 1e-8


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= SHAKE_TOLERANCE * abs(b)


def settle_constraints(constraints) -> set:
    """The constraints that make SETTLE clusters (rigid waters): three atoms with three constraints among them, each
    atom in exactly two. ``constraints``: list of ((i, j), length)."""
    of_atom = defaultdict(list)
    for n, ((i, j), _d) in enumerate(constraints):
        of_atom[i].append(n)
        of_atom[j].append(n)
    out = set()
    for n, ((i, j), _d) in enumerate(constraints):
        if len(of_atom[i]) != 2 or len(of_atom[j]) != 2:
            continue
        others_i = {k for k in of_atom[i] if k != n}
        others_j = {k for k in of_atom[j] if k != n}
        (ki,), (kj,) = others_i, others_j
        ai = set(constraints[ki][0]) - {i}
        aj = set(constraints[kj][0]) - {j}
        if ai and ai == aj and len(of_atom[next(iter(ai))]) == 2:
            out |= {n, ki, kj}
    return out


def shake_clusters(constraints, inverse_mass) -> tuple[dict, set]:
    """OpenMM's SHAKE clusters of ``constraints`` (list of ((i, j), length)) that are not SETTLE: central atom ->
    [index of constraint, ...], and the constraints that make no valid cluster (their central atom's cluster is
    invalid, or their peripheral atom has more than one constraint). ``inverse_mass``: atom -> 1 / mass."""
    settle = settle_constraints(constraints)
    count = defaultdict(int)
    for (i, j), _d in constraints:
        count[i] += 1
        count[j] += 1
    clusters, invalid = defaultdict(list), set()
    for n, ((a1, a2), _d) in enumerate(constraints):
        if n in settle:
            continue
        if count[a2] > 1:
            central, peripheral = a2, a1
        elif count[a1] > 1:
            central, peripheral = a1, a2
        elif a2 > a1:
            central, peripheral = a1, a2
        else:
            central, peripheral = a2, a1
        if count[peripheral] > 1:
            invalid.add(n)
            continue
        clusters[central].append(n)
    for central, members in clusters.items():
        first = members[0]
        d0, m0 = constraints[first][1], inverse_mass[_peripheral(constraints[first][0], central)]
        if len(members) > 3 or any(not _same(constraints[n][1], d0)
                                   or not _same(inverse_mass[_peripheral(constraints[n][0], central)], m0)
                                   for n in members):
            invalid.update(members)
    return dict(clusters), invalid


def _peripheral(pair, central):
    i, j = pair
    return j if i == central else i


def constraints_for_shake(constraints, inverse_mass, mapped, of_a) -> set:
    """The constraints to take out so that every other one is SETTLE or a valid SHAKE cluster (module docstring):
    their indices. ``mapped``: indices of constraints between two mapped atoms; ``of_a``: indices of the constraints
    of A (the hybrid takes those of A first)."""
    clusters, invalid = shake_clusters(constraints, inverse_mass)
    out = set()
    for central, members in clusters.items():
        if not set(members) & invalid:
            continue
        groups = []
        for n in members:
            d, m = constraints[n][1], inverse_mass[_peripheral(constraints[n][0], central)]
            for g in groups:
                if _same(d, g[0]) and _same(m, g[1]):
                    g[2].append(n)
                    break
            else:
                groups.append((d, m, [n]))

        def rank(n):
            return (n not in mapped, n not in of_a, n)

        def score(g):
            kept = sorted(g[2], key=rank)[:3]
            return (len(kept), sum(n in mapped for n in kept), sum(n in of_a for n in kept), -min(kept))

        best = max(groups, key=score)
        kept = set(sorted(best[2], key=rank)[:3])
        out |= set(members) - kept
    return out


def ligand_bond_terms(ligand, forcefield_spec, system_indices) -> dict:
    """The bond terms the force field of the preparation gives a ligand without constraints: (i, j) with i < j in
    ``system_indices`` (atom k of the molecule -> its index in the System) -> (length nm, force constant kJ/(mol
    nm^2)). ``ligand``: a :class:`cufep.prep.ligand.Ligand` with its partial charges; the System is made by the
    preparation's own force field (:func:`cufep.prep.forcefield.make_forcefield`) with no constraints."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from cufep.prep.forcefield import make_forcefield

    forcefield = make_forcefield(forcefield_spec, [ligand])
    topology = ligand.offmol.to_topology().to_openmm()
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    out = {}
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for k in range(force.getNumBonds()):
                i, j, r0, kk = force.getBondParameters(k)
                a, b = system_indices[i], system_indices[j]
                out[(min(a, b), max(a, b))] = (r0.value_in_unit(unit.nanometer),
                                               kk.value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2))
    return out
