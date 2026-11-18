"""Automatic Boresch restraint selection on a synthetic trajectory of the TYK2 complex."""
import numpy as np
import openmm.app as app
import openmm.unit as u
import pytest

from cufep.alchemy.restraints import RestraintError, find_boresch_restraint
from cufep.prep.ligand import load_ligands
from cufep.prep.protein import prepare_protein
from cufep.prep.solvate import combine, ligand_atom_indices


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
    # reference values are the trajectory means of the chosen coordinates
    v = r.values(frames)
    assert r.r0 == pytest.approx(v[:, 0].mean(), abs=1e-9)
    assert found.ligand_rmsd_nm < 0.05
    # a rigid-body rattle of the ligand in its pocket does not disqualify its ring atoms
    def rattle(fr):
        rng = np.random.default_rng(1)
        fr[:, lig, :] += rng.normal(scale=0.06, size=(fr.shape[0], 1, 3))
        return fr
    again = find_boresch_restraint(top, _trajectory(pos, lig, np.random.default_rng(2), ligand_motion=rattle),
                                   lig, rdmol, 298.15)
    assert all(rdmol.GetAtomWithIdx(lig.index(i)).GetIsAromatic() for i in again.restraint.guest_atoms)


def test_search_stops_when_the_ligand_leaves(tyk2_complex):
    top, pos, lig, rdmol = tyk2_complex

    def leave(fr):
        n = fr.shape[0]
        fr[:, lig, :] += np.linspace(0, 1.5, n)[:, None, None] * np.array([1.0, 0.0, 0.0])
        return fr

    frames = _trajectory(pos, lig, np.random.default_rng(3), ligand_motion=leave)
    with pytest.raises(RestraintError, match="not stable"):
        find_boresch_restraint(top, frames, lig, rdmol, 298.15)
