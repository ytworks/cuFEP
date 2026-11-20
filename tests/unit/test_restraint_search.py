"""Automatic Boresch restraint selection on a synthetic trajectory of the TYK2 complex."""
import numpy as np
import openmm.unit as u
import pytest

from cufep.alchemy.restraints import find_boresch_restraint, heavy_atom_symmetries
from cufep.prep.ligand import load_ligands
from cufep.prep.protein import prepare_protein
from cufep.prep.solvate import combine, ligand_atom_indices

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")


@pytest.fixture(scope="module")
def tyk2_complex(repo):
    protein = prepare_protein(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb")
    ligand = next(l for l in load_ligands(repo / "benchmarks/tyk2/inputs/tyk2_ligands.sdf") if l.name == "ejm_31")
    modeller = combine((protein.topology, protein.positions), ligand)
    pos = np.asarray(modeller.positions.value_in_unit(u.nanometer))
    return modeller.topology, pos, ligand_atom_indices(modeller.topology), ligand.to_rdkit()


def _trajectory(pos, lig, rng, n=40, noise=0.01, ligand_motion=None):
    frames = np.repeat(pos[None], n, axis=0) + rng.normal(scale=noise, size=(n, *pos.shape))
    if ligand_motion is not None:
        frames = ligand_motion(frames)
    # tumble and translate every frame as a whole: internal geometry is what matters
    for f in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        frames[f] = frames[f] @ q.T + rng.normal(size=3)
    return frames


def test_search_picks_rigid_ring_atoms_and_nearby_backbone(tyk2_complex):
    top, pos, lig, rdmol = tyk2_complex
    frames = _trajectory(pos, lig, np.random.default_rng(0))
    found = find_boresch_restraint(top, frames, lig, rdmol, 298.15)
    r = found.restraint
    atoms = list(top.atoms())
    assert all(atoms[i].name in ("N", "CA", "C") for i in r.host_atoms)
    local = [lig.index(i) for i in r.guest_atoms]
    assert all(rdmol.GetAtomWithIdx(i).GetIsAromatic() for i in local)
    bonded = {tuple(sorted((b[0].index, b[1].index))) for b in top.bonds()}
    assert tuple(sorted(r.host_atoms[:2])) in bonded and tuple(sorted(r.host_atoms[1:])) in bonded
    assert tuple(sorted(r.guest_atoms[:2])) in bonded and tuple(sorted(r.guest_atoms[1:])) in bonded
    assert 0.5 <= r.r0 <= 1.5
    assert np.radians(30) <= r.theta_a0 <= np.radians(150) and np.radians(30) <= r.theta_b0 <= np.radians(150)
    # reference values are those of the final frame, as in OpenFE
    v = r.values(frames)
    assert (r.r0, r.theta_a0, r.phi_b0) == pytest.approx((v[-1, 0], v[-1, 1], v[-1, 4]), abs=1e-9)
    assert found.ligand_rmsd_nm < 0.05
    # a rigid-body rattle of the ligand in its pocket does not disqualify its ring atoms
    def rattle(fr):
        rng = np.random.default_rng(1)
        fr[:, lig, :] += rng.normal(scale=0.06, size=(fr.shape[0], 1, 3))
        return fr
    again = find_boresch_restraint(top, _trajectory(pos, lig, np.random.default_rng(2), ligand_motion=rattle),
                                   lig, rdmol, 298.15)
    assert all(rdmol.GetAtomWithIdx(lig.index(i)).GetIsAromatic() for i in again.restraint.guest_atoms)


def test_the_search_skips_anchors_without_a_dihedral_plane_where_the_run_starts(tyk2_complex):
    """Review of 10589a6 (P2): the plane of the dihedral angles was checked in the frames superposed on the binding
    site, where rounding turned three anchors exactly on a line into a bent row, and the force of the run was NaN. The
    rule applies to the configuration the run starts from as well: with the chosen guest anchors put exactly on a line
    in the start positions, the search takes other anchors."""
    top, pos, lig, rdmol = tyk2_complex
    frames = _trajectory(pos, lig, np.random.default_rng(0))
    first = find_boresch_restraint(top, frames, lig, rdmol, 298.15, start_nm=frames[-1])
    start = frames[-1].copy()
    g0, g1, g2 = first.restraint.guest_atoms
    start[g1] = start[g0] + [0.14, 0.0, 0.0]                     # exactly on a line along x
    start[g2] = start[g0] + [0.28, 0.0, 0.0]
    again = find_boresch_restraint(top, frames, lig, rdmol, 298.15, start_nm=start)
    assert again.restraint.guest_atoms != first.restraint.guest_atoms


def test_a_ligand_that_moves_is_measured_and_the_search_goes_on(tyk2_complex):
    """OpenFE has no stop for a pose that changes and reports the ligand RMSD without a limit; so does cuFEP (the
    owner, 2026-11-20: the warning above restraints.pose_rmsd_warning_nm, a level of cuFEP's own, is gone)."""
    top, pos, lig, rdmol = tyk2_complex

    def shift(fr):
        fr[fr.shape[0] // 4:, lig, :] += np.array([0.25, 0.25, 0.0])        # 3.5 A, and it stays there
        return fr

    found = find_boresch_restraint(top, _trajectory(pos, lig, np.random.default_rng(3), ligand_motion=shift),
                                   lig, rdmol, 298.15)
    assert found.ligand_rmsd_nm == pytest.approx(0.354, abs=0.02)
    # the restraint holds the ligand where it is at the end, not at a place between the two positions
    frames = _trajectory(pos, lig, np.random.default_rng(3), ligand_motion=shift)
    v = found.restraint.values(frames)
    assert found.restraint.r0 == pytest.approx(v[-1, 0], abs=1e-9)
    assert abs(found.restraint.r0 - v[:, 0].mean()) > 0.01
    assert not any("moved away" in n for n in found.notes)


def test_the_pose_is_compared_with_the_input_pose(tyk2_complex):
    """Review of PR 1: a ligand that had moved before the frames kept for the search began was not reported."""
    top, pos, lig, rdmol = tyk2_complex
    shift = np.array([0.5, 0.0, 0.0])

    def moved_before_the_trajectory(fr):
        fr[:, lig, :] += shift                                      # in every frame: the ligand sits 5 A away
        return fr

    frames = _trajectory(pos, lig, np.random.default_rng(11), ligand_motion=moved_before_the_trajectory)
    unaware = find_boresch_restraint(top, frames, lig, rdmol, 298.15)
    assert unaware.ligand_rmsd_nm < 0.05
    aware = find_boresch_restraint(top, frames, lig, rdmol, 298.15, input_pose_nm=pos)
    assert aware.ligand_rmsd_nm == pytest.approx(0.5, abs=0.05)                # a number, without a note
    assert not any("moved away" in n for n in aware.notes)
    assert aware.restraint.host_atoms == unaware.restraint.host_atoms       # the restraint itself is as before
    assert aware.restraint.r0 == pytest.approx(unaware.restraint.r0)

    # a ligand written in another periodic image has not moved
    box = np.diag([8.0, 9.0, 10.0])
    rng = np.random.default_rng(12)
    still = np.repeat(pos[None], 40, axis=0) + rng.normal(scale=0.01, size=(40, *pos.shape))
    still[::2, lig, :] += box[0]                                    # every other frame: one box vector away
    still[5::10, lig, :] -= box[2]
    boxes = np.repeat(box[None], 40, axis=0)
    given = pos.copy()
    given[lig] += box[1]                                            # and the input in yet another image
    imaged = find_boresch_restraint(top, still, lig, rdmol, 298.15, input_pose_nm=given, frame_boxes_nm=boxes,
                                    input_box_nm=box)
    assert imaged.ligand_rmsd_nm < 0.05 and not any("moved away" in n for n in imaged.notes)
    assert 0.5 <= imaged.restraint.r0 <= 1.5


def test_pose_rmsd_ignores_swaps_of_symmetry_equivalent_atoms(tyk2_complex):
    top, pos, lig, rdmol = tyk2_complex
    perms = heavy_atom_symmetries(rdmol)
    assert len(perms) == 2 and perms[0].tolist() == list(range(len(perms[0])))
    heavy_local = [i for i, a in enumerate(rdmol.GetAtoms()) if a.GetAtomicNum() > 1]
    lig_heavy = np.array(lig)[heavy_local]

    def flip(fr):                                    # the symmetric ring turns over half way
        half = fr.shape[0] // 2
        fr[half:, lig_heavy, :] = fr[half:, lig_heavy[perms[1]], :]
        return fr

    frames = _trajectory(pos, lig, np.random.default_rng(4), ligand_motion=flip)
    found = find_boresch_restraint(top, frames, lig, rdmol, 298.15)
    assert found.ligand_rmsd_nm < 0.05 and not any("moved away" in n for n in found.notes)
    moved = np.linalg.norm(pos[lig_heavy] - pos[lig_heavy[perms[1]]], axis=1)
    assert np.sqrt((moved ** 2).mean()) > 0.1         # without the correction this is a large RMSD
    # the atoms that changed places are not used as anchors
    swapped = {int(lig_heavy[i]) for i in np.nonzero(moved > 0.05)[0]}
    assert not swapped & set(found.restraint.guest_atoms)


def test_force_constants_from_fluctuations():
    from cufep.alchemy.restraints import (
        K_ANGLE_DEFAULT,
        K_R_DEFAULT,
        BoreschRestraint,
        with_force_constants_from_fluctuations,
    )

    kt = 8.31446261815324e-3 * 298.15
    base = BoreschRestraint((0, 1, 2), (3, 4, 5), 0.8, 1.6, 1.6, 0.0, 2.0, 1.0)
    fluct = {"r": 0.025, "theta_A": 0.08, "theta_B": 0.10, "phi_A": 0.08, "phi_B": 0.5, "phi_C": 0.001}
    r = with_force_constants_from_fluctuations(base, fluct, 298.15)
    assert r.k_r == pytest.approx(kt / 0.025 ** 2) and r.k_theta_a == pytest.approx(kt / 0.08 ** 2)
    assert r.k_phi_b == pytest.approx(0.25 * K_ANGLE_DEFAULT)          # very floppy: lower limit
    assert r.k_phi_c == pytest.approx(4.0 * K_ANGLE_DEFAULT)           # very stiff: upper limit
    assert 0.25 * K_R_DEFAULT <= r.k_r <= 4.0 * K_R_DEFAULT
    assert (r.r0, r.host_atoms, r.guest_atoms) == (base.r0, base.host_atoms, base.guest_atoms)
    assert np.isfinite(r.release_free_energy(298.15)) and base.k_r == K_R_DEFAULT


def test_a_ligand_several_cells_away_is_brought_next_to_its_anchor():
    """Review of PR 1: only the neighbouring cells were tried, so a ligand two cells away from the
    protein (coordinates of a trajectory are not wrapped) stayed more than a box length away."""
    from cufep.alchemy.restraints import nearest_image_of_ligand

    rng = np.random.default_rng(6)
    anchor = np.array([1.0, 2.0, 3.0])
    pos = np.zeros((12, 3))
    ligand = np.arange(4, 8)
    pos[:4] = anchor + rng.normal(scale=0.3, size=(4, 3))              # atoms that are not the ligand
    home = anchor + np.array([0.2, 0.0, 0.0]) + rng.normal(scale=0.05, size=(4, 3))
    for box in (np.diag([4.0, 4.0, 4.0]),
                np.array([[4.0, 0.0, 0.0], [1.3, 3.8, 0.0], [-1.1, 1.6, 3.5]])):     # a cube and a skewed cell
        for cells in ([2, 0, 0], [-3, 2, 1], [0, 0, 0], [1, -1, 0], [5, 4, -6]):
            moved = pos.copy()
            moved[ligand] = home + np.array(cells, dtype=float) @ box
            back = nearest_image_of_ligand(moved, ligand, anchor, box)
            assert np.allclose(back[ligand], home, atol=1e-9)                        # the same pose again
            assert np.linalg.norm(back[ligand].mean(axis=0) - anchor) < 0.3
            assert np.array_equal(back[:4], moved[:4])                               # nothing else is moved
    # frames with a box each
    frames = np.repeat(pos[None], 3, axis=0)
    boxes = np.array([np.diag([4.0, 4.0, 4.0]), np.diag([4.1, 4.0, 3.9]), np.diag([4.2, 4.1, 4.0])])
    for f in range(3):
        frames[f, ligand] = home + np.array([2.0, -2.0, 3.0]) @ boxes[f]
    back = nearest_image_of_ligand(frames, ligand, np.repeat(anchor[None], 3, axis=0), boxes)
    assert np.allclose(back[:, ligand], home, atol=1e-9)

