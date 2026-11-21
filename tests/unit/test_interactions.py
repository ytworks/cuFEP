"""The interactions of the frames of an MD as ProLIF finds them, on the TYK2 complex with ejm_31."""
import numpy as np
import openmm.unit as u
import pytest

pytestmark = pytest.mark.needs("prolif", "MDAnalysis", "openff.toolkit", "rdkit", "pdbfixer")


@pytest.fixture(scope="module")
def tyk2(repo):
    """(topology, positions in nm, ligand atoms, the ligand's molecule) of TYK2 with ejm_31 as prepared,
    and the topology of the protein alone."""
    from cufep.prep.ligand import load_ligands
    from cufep.prep.protein import prepare_protein
    from cufep.prep.solvate import combine, ligand_atom_indices

    protein = prepare_protein(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb")
    ligand = next(l for l in load_ligands(repo / "benchmarks/tyk2/inputs/tyk2_ligands.sdf") if l.name == "ejm_31")
    modeller = combine((protein.topology, protein.positions), ligand)
    return (modeller.topology, np.asarray(modeller.positions.value_in_unit(u.nanometer)),
            list(ligand_atom_indices(modeller.topology)), ligand.to_rdkit(), protein.topology)


def hinge(topology):
    """The index of the hinge residue, VAL 981 of the input."""
    return next(r.index for r in topology.residues() if r.name == "VAL" and r.id == "981")


def test_the_hinge_hydrogen_bonds_of_the_input_pose_are_found(tyk2):
    """ejm_31 in TYK2 (PDB 4GIH): the pyridine nitrogen accepts a hydrogen bond from the backbone NH
    of VAL 981 and the amide NH next to it donates one to its carbonyl O, the hinge contacts of the
    crystal structure. Each is given to the ligand's heavy atom, by its number in the ligand, and
    to the residue by its index in the topology; both are in the one frame."""
    from cufep.analysis.interactions import interactions_of_frames, occupancies

    topology, pos, ligand, rdmol, _protein = tyk2
    frame, = interactions_of_frames(topology, pos[None], np.arange(len(pos)), ligand, rdmol)
    symbols = [a.GetSymbol() for a in rdmol.GetAtoms()]
    found = {(symbols[atom], residue, kind) for atom, residue, kind in frame}
    assert found == {("N", hinge(topology), "HBAcceptor"), ("N", hinge(topology), "HBDonor")}
    acceptor = next(atom for atom, _r, kind in frame if kind == "HBAcceptor")
    donor = next(atom for atom, _r, kind in frame if kind == "HBDonor")
    assert rdmol.GetAtomWithIdx(acceptor).GetIsAromatic() and not rdmol.GetAtomWithIdx(donor).GetIsAromatic()
    assert any(n.GetSymbol() == "H" for n in rdmol.GetAtomWithIdx(donor).GetNeighbors())
    assert occupancies([frame]) == [
        {"residue": hinge(topology), "interaction": "HBAcceptor", "frames": 1, "fraction": 1.0},
        {"residue": hinge(topology), "interaction": "HBDonor", "frames": 1, "fraction": 1.0}]


def test_an_interaction_is_counted_in_the_frames_in_which_it_is_there(tyk2):
    """Ten frames; in the first seven the ligand is where the input has it, in the last three it is
    moved 6 nm away from the protein: the hinge hydrogen bonds are in seven frames, 0.7 of them. The
    frames may hold their atoms in any order, be turned and moved as a whole, and hold atoms that
    are no protein: the interactions are the same."""
    from cufep.analysis.interactions import interactions_of_frames, occupancies

    topology, pos, ligand, rdmol, _protein = tyk2
    frames = np.repeat(pos[None], 10, axis=0)
    centre = pos[ligand].mean(axis=0) - pos.mean(axis=0)
    frames[7:, ligand, :] += 6.0 * centre / np.linalg.norm(centre)
    records = interactions_of_frames(topology, frames, np.arange(len(pos)), ligand, rdmol)
    assert [len(frame) for frame in records] == [2] * 7 + [0] * 3
    assert [(row["interaction"], row["frames"], row["fraction"]) for row in occupancies(records)] == \
        [("HBAcceptor", 7, 0.7), ("HBDonor", 7, 0.7)]

    rng = np.random.default_rng(5)
    order = rng.permutation(len(pos))
    tumbled = frames[:, order, :].copy()
    for f in range(10):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        tumbled[f] = tumbled[f] @ q.T + rng.normal(size=3)
    assert interactions_of_frames(topology, tumbled, order, ligand, rdmol) == records


def test_the_molecules_made_once_give_what_prolif_gives_when_it_makes_them_for_every_frame(tyk2):
    """ProLIF's own run over a trajectory (``Fingerprint.run``) converts protein and ligand again for
    every frame. Here the molecules are made once and take the coordinates of each frame; over 30
    frames in which thermal noise makes and breaks interactions, the two give the same interactions
    with the same ligand atoms in every frame."""
    import MDAnalysis as mda
    import prolif

    from cufep.analysis.interactions import INTERACTIONS, VICINITY_CUTOFF_A, interactions_of_frames, protein_atoms_of

    topology, pos, ligand, rdmol, _protein = tyk2
    frames = np.repeat(pos[None], 30, axis=0) + np.random.default_rng(0).normal(scale=0.03, size=(30, *pos.shape))
    ours = [sorted(frame) for frame in interactions_of_frames(topology, frames, np.arange(len(pos)), ligand, rdmol)]
    universe = mda.Universe(topology, frames * 10.0, format="MEMORY")
    universe.residues.resids = np.arange(universe.residues.n_residues)
    fingerprint = prolif.Fingerprint(list(INTERACTIONS), count=True, vicinity_cutoff=VICINITY_CUTOFF_A)
    fingerprint.run(universe.trajectory, universe.atoms[np.array(ligand)], universe.atoms[protein_atoms_of(topology)],
                    n_jobs=1, progress=False)
    theirs = [sorted((occurrence["parent_indices"]["ligand"][0], residue.number, name)
                     for (_l, residue), by_type in fingerprint.ifp[f].items()
                     for name, occurrences in by_type.items() for occurrence in occurrences) for f in range(30)]
    assert ours == theirs
    assert len({tuple(frame) for frame in ours}) > 5 and max(len(frame) for frame in ours) >= 3


def test_residues_are_told_apart_by_their_index_whatever_the_input_calls_them(tyk2):
    """Two residues of one name and number (an insertion code, a second chain) are two residues:
    with every residue of the protein called number 1 of chain A, the hinge hydrogen bonds are those
    of the same residue index."""
    import copy

    from cufep.analysis.interactions import interactions_of_frames

    topology, pos, ligand, rdmol, _protein = tyk2
    renamed = copy.deepcopy(topology)
    for residue in renamed.residues():
        residue.id = "1"
    plain, = interactions_of_frames(topology, pos[None], np.arange(len(pos)), ligand, rdmol)
    same, = interactions_of_frames(renamed, pos[None], np.arange(len(pos)), ligand, rdmol)
    assert sorted(same) == sorted(plain) and {residue for _a, residue, _k in same} == {hinge(topology)}


def test_the_protein_has_the_charges_of_its_force_field_residue_by_residue(tyk2):
    """ProLIF finds ions and hydrogen-bond partners by the formal charges and the bonds that
    MDAnalysis infers from the atoms. Residue by residue the formal charge is the charge of the
    force field (ff14SB): aspartate and glutamate -1, lysine and arginine +1, a histidine with both
    hydrogens +1, everything else 0."""
    import openmm.app as app

    from cufep.analysis.interactions import InteractionFinder

    topology, _pos, ligand, rdmol, protein = tyk2
    finder = InteractionFinder(topology, ligand, rdmol)
    residue_of = {a.index: a.residue.index for a in topology.atoms()}
    formal: dict[int, int] = {}
    for atom, index in zip(finder.protein.GetAtoms(), finder.protein_atoms):
        formal[residue_of[int(index)]] = formal.get(residue_of[int(index)], 0) + atom.GetFormalCharge()
    # the protein alone, as prepared: the same residues in the same order
    assert [r.name for r in protein.residues()] == [r.name for r in topology.residues()][:protein.getNumResidues()]
    assert finder.protein_atoms.tolist() == list(range(protein.getNumAtoms()))
    system = app.ForceField("amber14/protein.ff14SB.xml").createSystem(protein)
    nonbonded = next(f for f in system.getForces() if f.__class__.__name__ == "NonbondedForce")
    of_force_field: dict[int, float] = {}
    for atom in protein.atoms():
        q = nonbonded.getParticleParameters(atom.index)[0].value_in_unit(u.elementary_charge)
        of_force_field[atom.residue.index] = of_force_field.get(atom.residue.index, 0.0) + q
    assert {r: int(round(q)) for r, q in of_force_field.items()} == formal
    assert sorted(set(formal.values())) == [-1, 0, 1] and sum(formal.values()) == round(sum(of_force_field.values()))


def test_the_ligand_is_read_with_rdkits_aromaticity():
    """The molecule of the preparation carries the MDL model of aromaticity, in which a
    five-membered ring is not aromatic; ProLIF's patterns for rings and for hydrogen-bond partners
    are written for RDKit's. The copy for ProLIF has RDKit's: thiazole, imidazole and furan rings
    are aromatic in it, with the atoms, their order and the coordinates unchanged."""
    from openff.toolkit import Molecule
    from rdkit import Chem

    from cufep.analysis.interactions import with_rdkit_aromaticity

    for smiles, ring_atoms in (("Cc1ncc(s1)c1ccccc1", 11), ("c1c[nH]cn1", 5), ("c1ccoc1", 5), ("c1ccccc1", 6)):
        molecule = Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)
        rdmol = molecule.to_rdkit()
        copy = with_rdkit_aromaticity(rdmol)
        assert sum(a.GetIsAromatic() for a in copy.GetAtoms()) == ring_atoms
        assert [a.GetAtomicNum() for a in copy.GetAtoms()] == [a.GetAtomicNum() for a in rdmol.GetAtoms()]
        assert np.array_equal(copy.GetConformer().GetPositions(), rdmol.GetConformer().GetPositions())
        assert Chem.MolToSmiles(Chem.RemoveHs(copy)) == Chem.CanonSmiles(smiles)
    five = Molecule.from_smiles("c1ccoc1").to_rdkit()
    assert sum(a.GetIsAromatic() for a in five.GetAtoms()) == 0          # what ProLIF would be given otherwise


def test_without_prolif_the_reason_is_given(monkeypatch):
    import builtins

    from cufep.analysis import interactions

    assert interactions.packages_problem() is None
    real = builtins.__import__

    def without(name, *args, **kwargs):
        if name == "prolif":
            raise ImportError("No module named 'prolif'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without)
    problem = interactions.packages_problem()
    assert "needs ProLIF and MDAnalysis" in problem and "prolif (ImportError" in problem
    assert "MDAnalysis (" not in problem
