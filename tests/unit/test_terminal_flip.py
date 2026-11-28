"""Terminal flip Monte Carlo (cufep.sampling.terminal_flip): which groups are turned, and that the move samples
the distribution of the state it is made in."""
from __future__ import annotations

import numpy as np
import pytest

from cufep.sampling.terminal_flip import (
    FlipGroup,
    FlipMover,
    FlipSpec,
    half_turn,
    hybrid_flip_groups,
    ligand_graph,
    terminal_groups,
)

pytestmark = pytest.mark.needs("rdkit")

KT = 8.31446261815324e-3 * 300.0


def _mol(smiles: str):
    from rdkit import Chem

    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _groups(smiles: str) -> list[tuple[str, int, bool]]:
    """(element of the anchor, heavy atoms beyond the pivot with the pivot, whether the orientations differ)."""
    mol = _mol(smiles)
    return [(mol.GetAtomWithIdx(g["anchor"]).GetSymbol(),
             1 + sum(mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in g["mobile"]), g["distinct"])
            for g in terminal_groups(mol)]


@pytest.mark.parametrize("smiles, expected", [
    # a ring on the smaller side of a bond that is in no ring; the two orientations of a phenyl ring are one
    ("O=C(NCC(C)(C)C)c1ccccc1", [("C", 6, False)]),
    ("O=C(NCC(C)(C)C)c1cccc(Cl)c1", [("C", 7, True)]),
    ("O=C(NCCC(C)(C)C)c1c(Cl)cccc1Cl", [("C", 8, False)]),          # 2,6-dichlorophenyl: equivalent ortho atoms
    ("O=C(NCCC(C)(C)C)c1c(Cl)cccc1F", [("C", 8, True)]),
    ("O=C(NCC(C)(C)C)c1cccnc1", [("C", 6, True)]),                  # pyridin-3-yl
    ("O=C(NCCCCC(C)(C)C)c1ccc2ccccc2c1", [("C", 10, True)]),           # a fused ring system is one group
    # the ring is the larger side, or as large as the other: which side is the terminal one is not determined
    ("Cc1ccccc1", []),
    ("c1ccccc1-c1ccccc1", []),
    # the anchor may be an atom of another ring (a biaryl); what lies beyond holds one ring system only
    ("CCCCCCCCC(=O)Nc1ccc(cc1)-c1cccc(F)c1", [("C", 7, True)]),
    # a ring that is not aromatic is not turned
    ("O=C(NCC(C)(C)C)C1CCCCC1", []),
])
def test_terminal_groups_are_aromatic_rings_on_the_smaller_side_of_a_bond_outside_rings(smiles, expected):
    assert _groups(smiles) == expected


def _replaced(mol, atom: int, number: int):
    """``mol`` with one atom of another element (the same atom numbers)."""
    from rdkit import Chem

    out = Chem.RWMol(mol)
    out.GetAtomWithIdx(atom).SetAtomicNum(number)
    out = out.GetMol()
    Chem.SanitizeMol(out)
    return out


def _meta_hydrogen(mol) -> int:
    """A hydrogen in a meta position of the benzoyl ring of N-neopentylbenzamide."""
    from rdkit import Chem

    ipso, ortho, meta = mol.GetSubstructMatch(Chem.MolFromSmarts("[c;$(cC=O)]:c:c"))[:3]
    return next(n.GetIdx() for n in mol.GetAtomWithIdx(meta).GetNeighbors() if n.GetAtomicNum() == 1)


def test_a_hybrid_turns_the_groups_both_ligands_have_and_the_dummies_with_them():
    """The benzoyl ring of N-neopentylbenzamide (A) gets a chlorine in a meta position (B); the hydrogen and the
    chlorine are unique atoms. The ring is a terminal group of both ligands and its two orientations differ in B:
    it is turned, with the unique atoms of both ligands. Whether the unique atoms are numbered as the atoms of a
    System or after the atoms of A changes nothing but the numbers."""
    a = _mol("O=C(NCC(C)(C)C)c1ccccc1")
    h = _meta_hydrogen(a)
    b = _replaced(a, h, 17)
    mapping = {i: i for i in range(a.GetNumAtoms()) if i != h}
    to_a, to_b, bonds, with_hydrogen = ligand_graph(a, b, mapping)
    assert to_b[h] == a.GetNumAtoms() and all(to_b[i] == i for i in mapping)
    groups, record = hybrid_flip_groups(a, b, mapping, to_a, to_b, bonds, with_hydrogen)
    assert len(groups) == 1 and [r["used"] for r in record] == [True]
    group = groups[0]
    from rdkit import Chem

    carbonyl, ipso = a.GetSubstructMatch(Chem.MolFromSmarts("[C;$(C=O)]c"))
    assert (group.anchor, group.pivot) == (carbonyl, ipso)
    assert h in group.mobile and a.GetNumAtoms() in group.mobile              # the dummies of both ligands
    assert len(group.mobile) == 5 + 5 + 1                                       # ring atoms, hydrogens of A, chlorine
    assert record[0]["distinct"] is False and record[0]["found_in"] == "A"      # A alone would not be turned
    # other numbers of the hybrid (those of a System): the same group in them
    shift = {i: i + 100 for i in range(a.GetNumAtoms() + 1)}
    moved, _ = hybrid_flip_groups(a, b, mapping, {i: shift[j] for i, j in to_a.items()},
                                  {i: shift[j] for i, j in to_b.items()},
                                  {(shift[i], shift[j]) for i, j in bonds},
                                  {(shift[i], shift[j]) for i, j in with_hydrogen})
    assert [(g.anchor, g.pivot, g.mobile) for g in moved] == [
        (shift[group.anchor], shift[group.pivot], tuple(shift[i] for i in group.mobile))]


def test_groups_that_are_not_those_of_both_ligands_or_cannot_turn_are_left_with_the_reason():
    from rdkit import Chem

    a = _mol("O=C(NCC(C)(C)C)c1ccccc1")
    h = _meta_hydrogen(a)
    # the same ring in both ligands, phenyl in both: its orientations do not differ
    same = {i: i for i in range(a.GetNumAtoms())}
    groups, record = hybrid_flip_groups(a, a, same, *ligand_graph(a, a, same))
    assert not groups and [r["why_not"] for r in record] == ["the two orientations do not differ in either ligand"]
    # the ring is in B only (it grows from a hydrogen of the amide carbon's neighbour): not turned
    small = _mol("O=CNCC(C)(C)C")
    b = _replaced(a, h, 9)
    match = b.GetSubstructMatch(Chem.MolFromSmarts("O=CNCC(C)(C)C"))
    core = dict(zip(small.GetSubstructMatch(Chem.MolFromSmarts("O=CNCC(C)(C)C")), match))
    groups, record = hybrid_flip_groups(small, b, core, *ligand_graph(small, b, core))
    assert not groups and [r["why_not"] for r in record] == ["the ring or the anchor is in one ligand only"]
    # a constraint of the System that holds an atom of the group to one that does not turn
    b = _replaced(a, h, 17)
    mapping = {i: i for i in range(a.GetNumAtoms()) if i != h}
    to_a, to_b, bonds, with_hydrogen = ligand_graph(a, b, mapping)
    carbonyl, ipso, ortho = a.GetSubstructMatch(Chem.MolFromSmarts("[C;$(C=O)]cc"))
    groups, record = hybrid_flip_groups(a, b, mapping, to_a, to_b, bonds, {*with_hydrogen, (ipso, ortho)})
    assert not groups
    assert [r["why_not"] for r in record] == ["a constraint joins an atom that turns to one that does not"]
    # the atoms beyond the pivot reach the anchor in the hybrid (a ring closed by atoms of B alone)
    groups, record = hybrid_flip_groups(a, b, mapping, to_a, to_b, {*bonds, (a.GetNumAtoms(), carbonyl)}, with_hydrogen)
    assert not groups
    assert [r["why_not"] for r in record] == ["the atoms beyond the pivot reach the anchor in the hybrid"]


# ----------------------------------------------------------------------------------------------- the move
GROUP = FlipGroup(anchor=0, pivot=1, mobile=(2, 3))


def _positions() -> np.ndarray:
    # anchor, pivot on the x axis; two atoms beyond the pivot, off the axis
    return np.array([[1.0, 1.0, 1.0], [1.15, 1.0, 1.0], [1.22, 1.12, 1.0], [1.22, 1.12, 1.1]])


def test_half_a_turn_twice_is_the_configuration_and_distances_within_the_group_stay():
    box = 3.0 * np.eye(3)
    x = _positions()
    y, rotation = half_turn(x, box, GROUP)
    assert np.allclose(rotation @ rotation, np.eye(3)) and np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(y[[0, 1]], x[[0, 1]])                                      # the axis does not move
    assert np.allclose(y[2], [1.22, 0.88, 1.0]) and np.allclose(y[3], [1.22, 0.88, 0.9])
    for i, j in ((1, 2), (2, 3), (0, 3)):                                         # to the axis and within the group
        assert np.isclose(np.linalg.norm(y[i] - y[j]), np.linalg.norm(x[i] - x[j]))
    assert np.allclose(half_turn(y, box, GROUP)[0], x)
    # a ligand that lies across the edge of the cell is turned as one molecule: the atoms beyond the pivot are
    # taken by their nearest image
    across = x.copy()
    across[[2, 3]] -= box[0]
    z, _ = half_turn(across, box, GROUP)
    assert np.allclose(z[[2, 3]], y[[2, 3]])


def _context(k_kj_per_nm: float, constraint: bool = False):
    """Four particles on the Reference platform; the potential energy is ``k`` times the y of atom 2, so the two
    orientations of the group differ in energy by 2 k times its distance from the axis."""
    import openmm as mm

    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    force = mm.CustomExternalForce("k*y")
    force.addGlobalParameter("k", k_kj_per_nm)
    force.addParticle(2, [])
    system.addForce(force)
    if constraint:
        system.addConstraint(2, 3, 0.1)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(_positions())
    return context


def test_the_flips_alone_keep_the_distribution_of_two_orientations_of_unequal_energy():
    """Moves alone, in a potential in which the two orientations of the group differ by 1 kT: the higher one is
    populated as exp(-1) / (1 + exp(-1)) = 0.269. 6000 moves with a fixed seed on the Reference platform; the
    tolerance is four standard errors of that many moves of this chain (its samples are correlated: 0.012)."""
    import openmm.unit as u

    delta = KT                                                   # U(y = 1.12) - U(y = 0.88) = 0.24 k
    context = _context(delta / 0.24)
    mover = FlipMover(FlipSpec(groups=[GROUP], temperature_K=300.0, interval_ps=1.0), context, seed=7)
    high, accepted, n = 0, 0, 6000
    for _ in range(n):
        accepted += mover.attempt()["accepted"]
        y = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)[2, 1]
        high += y > 1.0
    expected = np.exp(-1.0) / (1.0 + np.exp(-1.0))
    assert abs(high / n - expected) < 0.048
    # every move down is accepted and the moves up as exp(-1): the accepted fraction is 2 p(high)
    assert abs(accepted / n - 2.0 * expected) < 0.06


def test_an_accepted_flip_turns_the_velocities_with_the_group_and_a_rejected_one_changes_nothing():
    import openmm.unit as u

    def state(context):
        st = context.getState(getPositions=True, getVelocities=True)
        v = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
        # the kinetic energy of the velocities as they are stored (the one a leapfrog integrator reports is that
        # of velocities shifted by half a step of the forces)
        return st.getPositions(asNumpy=True).value_in_unit(u.nanometer), v, 0.5 * 12.0 * float((v ** 2).sum())

    velocities = np.array([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2], [0.5, -0.4, 0.3], [0.5, -0.4, 0.3]])
    # downhill: accepted whatever the random number
    context = _context(1000.0, constraint=True)
    context.setVelocities(velocities)
    x, v, kinetic = state(context)
    move = FlipMover(FlipSpec(groups=[GROUP], temperature_K=300.0, interval_ps=1.0), context, seed=1).attempt()
    assert move["accepted"] and move["group"] == 0 and move["log_r"] == pytest.approx(240.0 / KT)
    y, w, kinetic_after = state(context)
    assert np.allclose(y[2], [1.22, 0.88, 1.0]) and np.allclose(y[[0, 1]], x[[0, 1]])
    assert np.allclose(w[[0, 1]], v[[0, 1]]) and np.allclose(w[2], [0.5, 0.4, -0.3])     # a half turn about x
    assert kinetic_after == pytest.approx(kinetic)
    # the constrained pair inside the group: its length, and no velocity along it, as before
    assert np.linalg.norm(y[3] - y[2]) == pytest.approx(0.1) and abs((w[3] - w[2]) @ (y[3] - y[2])) < 1e-12
    # uphill by 96 kT: rejected, and positions and velocities are what they were
    context = _context(-1000.0)
    context.setVelocities(velocities)
    x, v, _ = state(context)
    move = FlipMover(FlipSpec(groups=[GROUP], temperature_K=300.0, interval_ps=1.0), context, seed=1).attempt()
    assert not move["accepted"]
    y, w, _ = state(context)
    assert np.array_equal(y, x) and np.array_equal(w, v)
