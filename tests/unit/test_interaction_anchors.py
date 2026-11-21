"""The Boresch restraint of restraints.selection: interactions: its reference values, the rules
its anchors are held to, and what the rule reads of the protein."""
import json

import numpy as np
import pytest

from cufep.alchemy import restraints
from cufep.alchemy.restraints import RestraintError


def coordinates(n, r=0.6, theta_a=100.0, theta_b=80.0, phi=(50.0, -60.0, 120.0)):
    """``n`` frames of the six coordinates (nm, radians) at these values (nm, degrees)."""
    values = np.empty((n, 6))
    values[:, 0] = r
    values[:, 1:] = np.radians([theta_a, theta_b, *phi])
    return values


def test_the_reference_values_are_the_centres_of_the_fullest_bins():
    """Twenty bins: 9 degrees wide for the two angles, 18 for the three dihedral angles, a
    twentieth of the range of the data for the distance. Known by construction:

    * r: 700 frames between 0.600 and 0.610 nm and 300 between 0.900 and 0.910: the first bin of
      0.0155 nm, centre 0.60775, rounded to 0.608 nm (not the mean, 0.695);
    * theta_A at 100 degrees: bin 99 to 108, centre 103.5;
    * theta_B: 600 frames at 80 (bin 72 to 81, centre 76.5) and 400 at 130;
    * the dihedral angles at 50, -60 and 120: bins 36 to 54, -72 to -54, 108 to 126."""
    values = coordinates(1000)
    values[:700, 0] = np.linspace(0.600, 0.610, 700)
    values[700:, 0] = np.linspace(0.900, 0.910, 300)
    values[600:, 2] = np.radians(130.0)
    found = restraints.reference_values_from_histograms(values)
    assert found[0] == pytest.approx(0.608)
    assert np.degrees(found[1:]).tolist() == pytest.approx([103.5, 76.5, 45.0, -63.0, 117.0])


def test_frames_with_an_anchor_angle_out_of_range_give_no_reference_value():
    """Only the frames in which theta_A and theta_B both lie within 45 to 135 degrees are read, the
    ends included: 300 frames of another geometry with theta_A at 20 degrees, though they are the
    majority of no bin's neighbours, change nothing; without any frame in the range there is no
    restraint."""
    inside = coordinates(200)
    outside = coordinates(300, r=0.3, theta_a=20.0, theta_b=80.0, phi=(-150.0, 150.0, 0.0))
    both = np.concatenate([outside, inside])
    assert restraints.reference_values_from_histograms(both).tolist() == \
        pytest.approx(restraints.reference_values_from_histograms(inside).tolist())
    # frames exactly at the ends are read: 45 degrees is in the bin from 45 to 54, 135 in the one from 135 to 144
    edges = coordinates(10, theta_a=45.0, theta_b=135.0)
    assert np.degrees(restraints.reference_values_from_histograms(edges)[1:3]).tolist() == pytest.approx([49.5, 139.5])
    with pytest.raises(RestraintError, match="in no frame do both anchor angles"):
        restraints.reference_values_from_histograms(outside)
    with pytest.raises(RestraintError):
        restraints.reference_values_from_histograms(coordinates(10, theta_b=135.01))


def test_a_dihedral_peak_at_180_degrees_is_two_neighbouring_bins_like_a_peak_on_any_bin_edge():
    """The bins of a dihedral angle cover the circle. 300 frames at 175 and 300 at -175 degrees
    (one peak around 180) are the two bins on either side of 180, as 300 at -5 and 300 at 5 are the
    two on either side of 0; in both cases 400 frames in one bin elsewhere are the fullest bin.
    A second histogram from 0 to 360 degrees has the same bins and says the same."""
    around_180 = coordinates(1000)
    around_180[:300, 4] = np.radians(175.0)
    around_180[300:600, 4] = np.radians(-175.0)
    around_180[600:, 4] = np.radians(60.0)
    around_0 = coordinates(1000)
    around_0[:300, 4] = np.radians(5.0)
    around_0[300:600, 4] = np.radians(-5.0)
    around_0[600:, 4] = np.radians(60.0)
    for values in (around_180, around_0):
        assert np.degrees(restraints.reference_values_from_histograms(values)[4]) == pytest.approx(63.0)
    degrees = np.degrees(around_180[:, 4])
    shifted = np.where(degrees >= 0, degrees, degrees + 360)
    assert np.histogram(shifted, bins=20, range=(0.0, 360.0))[0].max() == 400
    # a peak that is inside one bin wins against the same 400 frames
    inside_a_bin = coordinates(1000)
    inside_a_bin[:600, 4] = np.radians(-170.0)
    inside_a_bin[600:, 4] = np.radians(60.0)
    assert np.degrees(restraints.reference_values_from_histograms(inside_a_bin)[4]) == pytest.approx(-171.0)


# --------------------------------------------------------------------------- measuring anchors
def chain(theta_a_deg=90.0, theta_b_deg=90.0):
    """Six anchors H2, H1, H0, G0, G1, G2 (atoms 0 to 5) and a seventh ligand atom, with the angles
    theta_A (H1-H0-G0) and theta_B (H0-G0-G1) given, the dihedral angles defined, in nm."""
    a, b = np.radians(theta_a_deg), np.radians(theta_b_deg)
    h0 = np.zeros(3)
    g0 = np.array([0.6, 0.0, 0.0])
    h1 = h0 + 0.15 * np.array([np.cos(a), np.sin(a), 0.0])
    h2 = h1 + np.array([0.0, 0.05, 0.14])
    g1 = g0 + 0.15 * np.array([-np.cos(b), np.sin(b) * np.cos(1.0), np.sin(b) * np.sin(1.0)])
    g2 = g1 + np.array([0.05, 0.10, -0.10])
    return np.array([h2, h1, h0, g0, g1, g2, g2 + [0.1, 0.0, 0.0]])


HOST, GUEST, LIGAND = (2, 1, 0), (3, 4, 5), [3, 4, 5, 6]
BOX = 5.0 * np.eye(3)


def test_anchors_are_measured_with_the_ligand_next_to_its_host_whatever_image_it_is_written_in():
    """The six coordinates of the anchors are those of neighbours: with the ligand written two cells
    away in half of the frames, and with other atoms in the frames, they are what they are without."""
    pose = chain()
    frames = np.repeat(pose[None], 6, axis=0) + np.random.default_rng(0).normal(scale=0.002, size=(6, 7, 3))
    boxes = np.repeat(BOX[None], 6, axis=0)
    plain = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, 0.3)
    away = frames.copy()
    away[::2, 3:] += np.array([10.0, -5.0, 0.0])
    moved = restraints.measure_anchors(HOST, GUEST, LIGAND, away, np.arange(7), boxes, 0.3)
    assert plain.problem is None and moved.problem is None
    assert np.allclose(plain.values, moved.values) and plain.values[0, 0] == pytest.approx(0.6, abs=0.01)
    # frames that hold more atoms, under other numbers of the System
    numbers = np.array([40, 41, 42, 50, 51, 52, 53, 7, 8])
    wide = np.concatenate([frames, np.zeros((6, 2, 3))], axis=1)
    renumbered = restraints.measure_anchors((42, 41, 40), (50, 51, 52), [50, 51, 52, 53], wide, numbers, boxes, 0.3)
    assert np.array_equal(renumbered.values, plain.values) and plain.start_values is None


def test_the_start_of_the_alchemical_run_is_held_to_the_rules_when_no_frame_is_the_start():
    """Reference values that are not those of the last frame leave the configuration the run starts
    from outside the frames. Its dihedral angles must be defined in any case; as one of the frames
    (``start_among_frames``) its anchor angles are held to OpenFE's rule too: frames at 90 degrees
    and a start with theta_A at 10 degrees are refused, where the frames alone are accepted."""
    frames = np.repeat(chain()[None], 5, axis=0)
    boxes = np.repeat(BOX[None], 5, axis=0)
    margin = np.radians(22.0)
    good = np.zeros((20, 3))
    good[:7] = chain(80.0, 100.0)
    fine = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, margin, start=good,
                                      start_box=BOX, start_among_frames=True)
    assert fine.problem is None
    assert np.degrees(fine.start_values[1:3]).tolist() == pytest.approx([80.0, 100.0])
    assert fine.values.shape == (5, 6)                                   # the start is no frame of the values
    flat = np.zeros((20, 3))
    flat[:7] = chain(10.0, 90.0)
    as_reference_only = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, margin,
                                                   start=flat, start_box=BOX)
    among = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, margin, start=flat,
                                       start_box=BOX, start_among_frames=True)
    assert as_reference_only.problem is None
    assert "comes within 22 degrees of a straight line" in among.problem
    # anchors exactly on a line where the run starts have no dihedral angle there: refused either way
    line = np.zeros((20, 3))
    line[:7] = chain()
    line[4] = line[3] + [0.15, 0.0, 0.0]
    line[5] = line[3] + [0.30, 0.0, 0.0]
    for among_frames in (False, True):
        refused = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, margin, start=line,
                                             start_box=BOX, start_among_frames=among_frames)
        assert refused.problem is not None


def test_anchors_on_a_line_in_every_frame_are_refused_though_the_start_is_bent():
    """Review of eb6f89f (P1). With the start among the frames, G0-G1-G2 exactly on a line in every frame the
    reference values come from and bent where the run starts was no longer "on a line in every frame": the
    anchors passed, and phi_C, which no frame defines, got the reference of the bin of 0 degrees. The rule
    of a trajectory is one of the frames; the start is held to the rules of a single configuration."""
    straight = chain()
    straight[4] = straight[3] + [0.0, 0.15, 0.0]                         # theta_B stays 90 degrees
    straight[5] = straight[3] + [0.0, 0.30, 0.0]
    frames = np.repeat(straight[None], 6, axis=0)
    boxes = np.repeat(BOX[None], 6, axis=0)
    bent = np.zeros((20, 3))
    bent[:7] = chain()
    for among in (False, True):
        measured = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, np.radians(22.0),
                                              start=bent, start_box=BOX, start_among_frames=among)
        assert "on a straight line in every frame" in measured.problem, among
        assert np.all(measured.values[:, 5] == 0.0)                      # what the frames say of phi_C: nothing


def test_force_constants_from_fluctuations_are_held_to_the_rule_of_the_flat_angle_themselves():
    """Review of eb6f89f (P1). OpenFE's check_angle_not_flat takes the force constant of the angle. At 298.15 K
    theta_A at 60 degrees in half of the frames and at 120 in the other half, and at 30 degrees where the run
    starts: with 80 kcal/mol/rad^2 the margin is 22.05 degrees and the anchors pass. Their fluctuation of 30
    degrees gives 20 kcal/mol/rad^2 (the lower limit), whose margin is 44.10 degrees: at 30 degrees the
    restraint is 4.6 kT from a straight line. The anchors are asked again with the constants they get."""
    frames = np.array([chain(60.0 if f % 2 == 0 else 120.0) for f in range(20)])
    boxes = np.repeat(BOX[None], 20, axis=0)
    start = np.zeros((20, 3))
    start[:7] = chain(30.0)
    k_80 = 80.0 * 4.184
    margin = restraints.flat_margin_of(k_80, 298.15)
    assert np.degrees(margin) == pytest.approx(22.05, abs=0.01)
    measured = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, margin, start=start,
                                          start_box=BOX, start_among_frames=True)
    assert measured.problem is None and measured.held_theta.shape == (21, 2)
    trial = restraints.BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    with_them, problem = restraints.constants_from_fluctuations(trial, measured.values, measured.held_theta, 298.15)
    assert with_them.k_theta_a == pytest.approx(20.0 * 4.184) and with_them.k_theta_b == pytest.approx(4 * k_80)
    assert np.degrees(restraints.flat_margin_of(with_them.k_theta_a, 298.15)) == pytest.approx(44.10, abs=0.01)
    assert "H1-H0-G0 (theta_A) comes within 44 degrees of a straight line (30.0 to 120.0" in problem
    # the energy of the restraint at 30 degrees with that constant, in kT: below the 10 of the rule
    in_kt = 0.5 * with_them.k_theta_a * np.radians(30.0) ** 2 / (restraints.R_GAS * 298.15)
    assert in_kt == pytest.approx(4.63, abs=0.01)
    # anchors that keep their distance from a straight line with the weaker constant too
    calm = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, margin, start=frames[0],
                                      start_box=BOX, start_among_frames=True)
    assert restraints.constants_from_fluctuations(trial, calm.values, calm.held_theta, 298.15)[1] is None

    # anchors that are given (restraints.selection: manual) with restraints.force_constants: auto
    from cufep.workflow import WorkflowError, manual_boresch_restraint

    given = np.concatenate([frames, start[None, :7]])
    eq = {"frames": given, "frame_atoms": np.arange(7), "frame_boxes": np.repeat(BOX[None], 21, axis=0),
          "positions": start[:7], "box": BOX}
    fixed, _fluctuations = manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 7, 10.0 * 4.184 * 100.0, k_80)
    assert fixed.k_theta_a == pytest.approx(k_80)
    with pytest.raises(WorkflowError, match="with the force constants of their fluctuations .*within 44 degrees"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 7, 10.0 * 4.184 * 100.0, k_80, force_constants="auto")


# --------------------------------------------------------------------------- the protein
@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_the_hosts_are_the_residues_with_a_backbone_and_dssp_is_counted_frame_by_frame(repo):
    """TYK2 as prepared: every amino-acid residue is a host with its N, CA and C, the two caps are
    none. The fraction of frames in a stable structure is that of DSSP frame by frame: with the
    helices of half of the frames pulled apart, a residue of a helix is at one half."""
    import mdtraj as md
    import openmm.unit as u

    from cufep.prep.protein import AMINO_ACIDS, prepare_protein

    protein = prepare_protein(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb")
    topology = protein.topology
    atoms = list(topology.atoms())
    pos = np.asarray(protein.positions.value_in_unit(u.nanometer))
    hosts = restraints.backbone_hosts(topology)
    amino = [r for r in topology.residues() if r.name in AMINO_ACIDS]
    assert len(hosts) == len(amino) == topology.getNumResidues() - 2
    assert all([atoms[i].name for i in trio] == ["N", "CA", "C"] and {atoms[i].residue.index for i in trio} == {index}
               for index, trio in hosts.items())
    frame_atoms = np.arange(topology.getNumAtoms())[::-1].copy()         # the frames need not follow the System
    frames = np.repeat(pos[frame_atoms][None], 4, axis=0)
    whole = restraints.stable_structure_fractions(topology, frames, frame_atoms)
    reference = md.compute_dssp(md.Trajectory(pos[None], md.Topology.from_openmm(topology)), simplified=False)[0]
    in_order = [r.index for r in topology.residues() if r.name in AMINO_ACIDS]
    expected = {index: float(code in "HGIEB") for index, code in zip([r.index for r in topology.residues()], reference)
                if index in in_order}
    assert whole == expected and 0.3 < np.mean(list(whole.values())) < 0.9
    helix = next(i for i in in_order if reference[i] == "H" and reference[i + 1] == "H")
    stretched = frames.copy()
    stretched[2:] *= 3.0                                                  # no hydrogen bond is left
    half = restraints.stable_structure_fractions(topology, stretched, frame_atoms)
    assert half[helix] == 0.5 and whole[helix] == 1.0


# --------------------------------------------------------------------------- the rule
def unit_vector(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def backbone_at(l1, l2, alpha_deg, theta_deg, distance=0.4):
    """N, CA and C of a residue (nm) such that the angle N-L1-L2 is ``alpha_deg``, the angle L1-N-CA
    is ``theta_deg`` and N, the backbone atom nearest to L1, is ``distance`` from it."""
    u = unit_vector(l2 - l1)
    v = unit_vector(np.cross(u, [0.31, 0.57, 0.76]))
    alpha, theta = np.radians(alpha_deg), np.radians(theta_deg)
    n = l1 + distance * (np.cos(alpha) * u + np.sin(alpha) * v)
    w = unit_vector(l1 - n)
    s = unit_vector(np.cross(w, np.cross(u, v)))
    ca = n + 0.146 * (np.cos(theta) * w + np.sin(theta) * s)
    c = ca + 0.152 * unit_vector(np.cross(w, s) + 0.5 * s - 0.5 * w)
    return np.array([n, ca, c])


class World:
    """A ligand that does not move and residues whose backbone is put where the test wants it, frame
    by frame. The ligand is N-methylaniline (C0 N1 c2 c3 ... c7): N1 is a suitable L1 with L2 = c2
    and L3 = c3 or c7, and the angle N1-c2-c3 is that of the ring in every frame. Atom numbers of
    the System: N, CA, C of residue r are 10 r, 10 r + 1, 10 r + 2; ligand atom i is 1000 + i."""

    L1, L2 = 1, 2

    def __init__(self, n_frames, smiles="CNc1ccccc1"):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from cufep.alchemy import interaction_anchors as ia

        self.ia = ia
        self.mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(self.mol, randomSeed=7)
        self.ligand = self.mol.GetConformer().GetPositions() * 0.1
        self.graph = ia.LigandGraph.from_rdkit(self.mol)
        self.n = n_frames
        self.backbones = {}

    def residue(self, index, alpha=90.0, theta=90.0, distance=0.4):
        """Residue ``index`` with these angles and this distance in every frame (numbers or one per frame)."""
        alpha, theta, distance = (np.broadcast_to(np.asarray(x, dtype=float), (self.n,))
                                  for x in (alpha, theta, distance))
        self.backbones[index] = np.array([backbone_at(self.ligand[self.L1], self.ligand[self.L2], a, t, d)
                                          for a, t, d in zip(alpha, theta, distance)])
        return self

    def host(self, index, order=(0, 1, 2)):
        return tuple(10 * index + k for k in order)

    def guest(self, *atoms):
        return tuple(1000 + a for a in atoms)

    def frames(self, stable=None, boxes=None, ligand_shift=None):
        residues = sorted(self.backbones)
        xyz = np.concatenate([self.backbones[r] for r in residues]
                             + [np.repeat(self.ligand[None], self.n, axis=0)], axis=1)
        if ligand_shift is not None:
            xyz[:, 3 * len(residues):, :] += np.asarray(ligand_shift, dtype=float)[:, None, :]
        atoms = [a for r in residues for a in self.host(r)] + [1000 + i for i in range(len(self.ligand))]
        return self.ia.Frames(xyz=xyz, atoms=np.array(atoms), ligand=[1000 + i for i in range(len(self.ligand))],
                              hosts={r: self.host(r) for r in residues}, stable_fraction=dict(stable or {}),
                              boxes=boxes, trajectory=np.arange(self.n) % 2)

    def records(self, *interactions):
        """One list of (ligand atom, residue, type) per frame from ``(residue, type, frames)``."""
        out = [[] for _ in range(self.n)]
        for residue, kind, frames in interactions:
            for f in frames:
                out[f].append((self.L1, residue, kind))
        return out

    def find(self, ranked, residue, order=(0, 1, 2), guest=(1, 2, 3)):
        return [c for c in ranked if c.host == self.host(residue, order) and c.guest == self.guest(*guest)]


pytest_rdkit = pytest.mark.needs("rdkit")


@pytest_rdkit
@pytest.mark.parametrize("n_frames,present,kept", [(800, 320, True), (800, 319, False), (799, 319, True),
                                                   (799, 318, False)])
def test_an_interaction_is_kept_in_two_fifths_of_the_frames_rounded_down(n_frames, present, kept):
    """"Retained if they occur in at least 40% of the analyzed frames", the number of frames rounded
    down: 320 of 800, and 319 of 799 (319.6)."""
    world = World(n_frames).residue(3)
    table = world.ia.interaction_table(world.records((3, "HBAcceptor", range(present))), world.graph, n_frames,
                                       world.ia.InteractionSettings())
    assert table.needed == {800: 320, 799: 319}[n_frames]
    assert (table.kept == [(1, 3, "HBAcceptor")]) is kept
    row, = table.rows(world.frames())
    assert (row["ligand_atom"], row["residue"], row["interaction"], row["frames"], row["kept"]) == \
        (1001, 3, "HBAcceptor", present, kept)
    assert row["fraction"] == pytest.approx(present / n_frames)
    assert sum(row["frames_of_trajectory"].values()) == present and set(row["frames_of_trajectory"]) == {0, 1}
    ranked = world.ia.rank_by_interactions(world.frames(), world.graph, table, world.ia.InteractionSettings())
    assert bool(ranked) is kept


@pytest_rdkit
def test_an_interaction_counts_once_in_a_frame_and_only_for_the_types_that_carry_anchors():
    world = World(10).residue(3)
    records = [[(1, 3, "HBAcceptor"), (1, 3, "HBAcceptor"), (1, 3, "VdWContact"), (1, 3, "Hydrophobic")]] * 10
    table = world.ia.interaction_table(records, world.graph, 10, world.ia.InteractionSettings())
    assert {key: len(frames) for key, frames in table.frames_of.items()} == {(1, 3, "HBAcceptor"): 10}


@pytest_rdkit
def test_the_types_of_one_ligand_atom_and_residue_are_merged():
    """The frames of the kept types together, and the type of the largest weight: a hydrogen bond in
    frames 0 to 299 and an ionic interaction in 200 to 399 of 500 are 400 frames of an ionic one. A
    type below the threshold (199 frames, of 200 needed) is not kept and adds no frame."""
    world = World(500).residue(3)
    records = world.records((3, "HBAcceptor", range(300)), (3, "Anionic", range(200, 400)),
                            (3, "XBDonor", range(301, 500)))
    settings = world.ia.InteractionSettings()
    table = world.ia.interaction_table(records, world.graph, 500, settings)
    assert table.kept == [(1, 3, "Anionic"), (1, 3, "HBAcceptor")]
    ranked = world.ia.rank_by_interactions(world.frames(), world.graph, table, settings)
    assert {(c.interaction, c.n_frames) for c in ranked} == {("Anionic", 400)}


@pytest_rdkit
@pytest.mark.parametrize("smiles,atom,l1", [
    ("CC(=O)Nc1ccccc1", 2, 1),             # the O of an amide: its carbon
    ("CC(=O)Nc1ccccc1", 3, 3),             # the N of an amide is suitable itself
    ("c1ccccc1S(=O)(=O)C(C)C", 7, 5),      # an O of a sulfonyl group: past S and O, the ring carbon (the lower number)
    ("CC(C)S(=O)(=O)c1ccccc1", 4, 1),      # the same group written the other way round: the carbon of lower number
    ("OC(=O)c1ccccc1", 0, 1),              # an O of an acid: the carbon; neither O is ever L1
    ("COC(=O)c1ccccc1", 1, 2),             # the O of an ester outside a ring is excluded by the second pattern
    ("Clc1ccccc1", 0, 1),                  # a halogen has one bonded atom
])
def test_an_interaction_is_given_to_the_first_suitable_atom_within_three_bonds(smiles, atom, l1):
    from rdkit import Chem

    from cufep.alchemy.interaction_anchors import LigandGraph

    graph = LigandGraph.from_rdkit(Chem.AddHs(Chem.MolFromSmiles(smiles)))
    assert graph.anchor_atom(atom) == l1
    assert all(not graph.suitable(a) for a in graph.excluded)


@pytest_rdkit
def test_atoms_without_a_suitable_atom_within_three_bonds_and_ligands_without_a_chain():
    from rdkit import Chem

    from cufep.alchemy.interaction_anchors import LigandGraph

    def graph(smiles):
        return LigandGraph.from_rdkit(Chem.AddHs(Chem.MolFromSmiles(smiles)))

    assert graph("FC(F)(F)S(=O)(=O)C(F)(F)F").anchor_atom(1) == 1            # the carbon of CF3: F and S are heavy
    assert graph("O=S(=O)(F)F").anchor_atom(0) is None                       # nothing but excluded atoms and ends
    # a chain L1-L2-L3 of atoms that are no chain ends: butane has one, propane, acetonitrile and water none
    assert graph("CCCC").has_guest_anchors() and graph("CNc1ccccc1").has_guest_anchors()
    assert not graph("CCC").has_guest_anchors() and not graph("CC#N").has_guest_anchors()
    assert not graph("O").has_guest_anchors()
    assert graph("CNc1ccccc1").guest_pairs(1) == [(2, 3), (2, 7)]            # the methyl carbon leads nowhere


def test_the_angle_range_includes_its_ends_and_the_deviation_is_from_90_degrees_over_all_frames():
    from cufep.alchemy.interaction_anchors import frames_in_range

    inside, away = frames_in_range(np.array([45.0, 135.0, 44.999, 135.001, 90.0]), (45.0, 135.0))
    assert inside == 3 and away == pytest.approx((45.0 + 45.0 + 45.001 + 45.001 + 0.0) / 5)
    assert frames_in_range(np.array([np.nan, 60.0]), (45.0, 135.0)) == (1, pytest.approx(60.0))


@pytest_rdkit
def test_anchors_are_candidates_if_two_angles_are_in_range_often_enough_among_the_frames_of_the_interaction():
    """100 frames, the interaction in frames 0 to 59, 40 frames needed. With P1-L1-L2 at 90 degrees
    in frames 0 to 39 the anchors are a candidate; at 90 degrees in frames 0 to 38 and 60 to 99 they
    are not: the frames outside the interaction do not count. L1-P1-P2 decides nothing: it may be
    out of range in every frame."""
    settings = World(1).ia.InteractionSettings()
    for in_range, candidate in ((range(40), True), ([*range(39), *range(60, 100)], False)):
        alpha = np.full(100, 20.0)
        alpha[list(in_range)] = 90.0
        world = World(100).residue(3, alpha=alpha, theta=20.0)
        table = world.ia.interaction_table(world.records((3, "HBAcceptor", range(60))), world.graph, 100, settings)
        ranked = world.ia.rank_by_interactions(world.frames(), world.graph, table, settings)
        found = world.find(ranked, 3)
        assert bool(found) is candidate
        if candidate:
            c, = found
            assert (c.n_frames, c.n_theta_b, c.n_theta_a, c.n_guest) == (60, 40, 0, 60)
            assert c.away_theta_a == pytest.approx(70.0) and c.away_theta_b == pytest.approx((40 * 0 + 20 * 70) / 60)
            # both guests of this L1 and L2, and the hosts in their six orders at most
            assert world.find(ranked, 3, guest=(1, 2, 7))
            assert {c.host for c in ranked} <= {world.host(3, order) for order in
                                               ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))}
    # with every angle at 90 degrees the backbone gives candidates in all six orders
    world = World(100).residue(3)
    table = world.ia.interaction_table(world.records((3, "HBAcceptor", range(100))), world.graph, 100, settings)
    assert len({c.host for c in world.ia.rank_by_interactions(world.frames(), world.graph, table, settings)}) == 6
    # a residue without a backbone (a cap) gives none
    frames = world.frames()
    frames.hosts = {}
    assert world.ia.rank_by_interactions(frames, world.graph, table, settings) == []


@pytest_rdkit
@pytest.mark.parametrize("deviation,stable,taper,counted,level", [
    (19.0, 0.0, 1.0, 799, 50),            # 38 degrees together: no taper; the largest count is level 50
    (23.5, 0.0, 0.895, 715, 45),          # 47: 0.7 + 13/20 x 0.3; 799 - 715 = 84 frames = 5 levels of 15
    (31.0, 0.0, 0.7, 559, 34),            # 62: the floor; 240 frames = 16 levels
    (23.5, 0.51, 0.895, 715, 48),         # a stable residue is raised by 3 levels of 15: 760, 39 frames below
    (23.5, 0.50, 0.895, 715, 45),         # stable is more than half of the frames
])
def test_the_score_of_a_candidate_by_hand(deviation, stable, taper, counted, level):
    """799 frames, a hydrogen bond in all of them, P1-L1-L2 and L1-P1-P2 in range in all of them and
    ``deviation`` degrees from 90 each. A level is max(10, 799 // 50) = 15 frames. The score is the
    two levels and the weight of a hydrogen bond, 0.6."""
    world = World(799).residue(3, alpha=90.0 + deviation, theta=90.0 - deviation)
    settings = world.ia.InteractionSettings()
    table = world.ia.interaction_table(world.records((3, "HBAcceptor", range(799))), world.graph, 799, settings)
    ranked = world.ia.rank_by_interactions(world.frames(stable={3: stable}), world.graph, table, settings)
    c, = world.find(ranked, 3)
    assert (c.n_theta_a, c.n_theta_b) == (799, 799)
    assert c.away_theta_a + c.away_theta_b == pytest.approx(2 * deviation)
    assert c.parts["taper"] == pytest.approx(taper) and int(c.parts["taper"] * 799) == counted
    assert (c.parts["level_theta_a"], c.parts["level_theta_b"], c.parts["frames_per_level"]) == (level, level, 15)
    assert c.score == pytest.approx(2 * level + 0.6) and c.stable is (stable > 0.5)
    assert ranked == sorted(ranked, key=lambda x: (-int(x.score * 100), x.host, x.guest))


@pytest_rdkit
def test_the_weight_of_the_interaction_the_bonus_and_the_atoms_decide_between_equal_levels():
    """Three residues with the same geometry and the same frames. A residue with two kept hydrogen
    bonds whose second is in more than 0.6 of the frames of the first gets 7 levels (480 of 799
    frames, not 479); a stacking interaction is no second one. Then an ionic interaction (0.9) is
    before a hydrogen bond (0.6); with all equal, the atoms of lower number come first."""
    def ranked_with(second_frames, second="HBDonor"):
        world = World(799).residue(3).residue(4).residue(5)
        records = world.records((3, "HBAcceptor", range(799)), (4, "Cationic", range(799)),
                                (5, "HBAcceptor", range(799)), (5, second, range(second_frames)))
        settings = world.ia.InteractionSettings()
        table = world.ia.interaction_table(records, world.graph, 799, settings)
        ranked = world.ia.rank_by_interactions(world.frames(), world.graph, table, settings)
        return world, {r: world.find(ranked, r)[0] for r in (3, 4, 5)}, ranked

    world, by_residue, ranked = ranked_with(480)
    assert [by_residue[r].score for r in (3, 4, 5)] == pytest.approx([100.6, 100.9, 107.6])
    assert [by_residue[r].parts["bonus"] for r in (3, 4, 5)] == [0, 0, 7]
    assert ranked[0].residue == 5 and ranked.index(by_residue[4]) < ranked.index(by_residue[3])
    assert ranked_with(479)[1][5].score == pytest.approx(100.6)                 # 479 / 799 is not more than 0.6
    assert ranked_with(799, second="EdgeToFace")[1][5].score == pytest.approx(100.6)
    # equal scores: by the numbers of the host atoms, then of the guest atoms
    world = World(799).residue(4).residue(3)
    settings = world.ia.InteractionSettings()
    table = world.ia.interaction_table(world.records((4, "HBAcceptor", range(799)), (3, "HBAcceptor", range(799))),
                                       world.graph, 799, settings)
    ranked = world.ia.rank_by_interactions(world.frames(), world.graph, table, settings)
    top = [c for c in ranked if int(c.score * 100) == int(ranked[0].score * 100)]
    assert [(c.host, c.guest) for c in top] == sorted((c.host, c.guest) for c in top) and top[0].residue == 3


@pytest_rdkit
def test_the_distance_method_takes_the_residues_whose_backbone_is_nearest_to_the_ligand():
    """No interaction: the nearest pair of a ligand N or O and a backbone atom within 1.0 nm, frame
    by frame; a residue with 15 of 100 frames is a candidate, with 14 it is not; 0.99 nm counts and
    1.01 nm does not; a ligand written one cell away is the same ligand."""
    from rdkit import Chem

    from cufep.alchemy.interaction_anchors import LigandGraph, distance_atoms

    settings = World(1).ia.InteractionSettings()
    assert distance_atoms(World(1).graph) == [1]                                   # N before any carbon
    assert distance_atoms(LigandGraph.from_rdkit(Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1")))) == [1, 2, 3, 4, 5, 6]

    def near(frames_near, distance=0.4):
        d = np.full(100, 1.5)
        d[:frames_near] = distance
        return World(100).residue(3, distance=d)

    for frames_near, candidate in ((15, True), (14, False)):
        world = near(frames_near)
        found = world.ia.contacts(world.frames(), world.graph, settings)
        assert {r: (l1, len(where)) for r, (l1, where) in found.items()} == {3: (1, frames_near)}
        ranked = world.ia.rank_by_distance(world.frames(), world.graph, settings)
        assert bool(ranked) is candidate
        if candidate:
            c, = world.find(ranked, 3)
            assert (c.interaction, c.n_frames, c.n_theta_a, c.n_theta_b, c.score) == ("contact", 15, 15, 15, None)
            assert ranked == sorted(ranked, key=lambda x: (-((x.n_theta_a + x.n_theta_b) // 2),
                                                           x.away_theta_a + x.away_theta_b,
                                                           x.away_host + x.away_guest, x.host, x.guest))
    assert len(near(100, 0.99).ia.contacts(near(100, 0.99).frames(), World(1).graph, settings)[3][1]) == 100
    assert near(100, 1.01).ia.contacts(near(100, 1.01).frames(), World(1).graph, settings) == {}
    # the ligand one cell away in every second frame, with the boxes of the frames
    world = near(100)
    shift = np.zeros((100, 3))
    shift[::2, 0] = 5.0
    boxes = np.repeat(5.0 * np.eye(3)[None], 100, axis=0)
    assert len(world.ia.contacts(world.frames(boxes=boxes, ligand_shift=shift), world.graph, settings)[3][1]) == 100
    assert len(world.ia.contacts(world.frames(ligand_shift=shift), world.graph, settings)[3][1]) == 50
    # two residues: each frame belongs to the residue of the nearest backbone atom
    d3, d4 = np.full(100, 0.5), np.full(100, 0.6)
    d4[:30] = 0.4
    world = World(100).residue(3, distance=d3).residue(4, alpha=60.0, distance=d4)
    found = world.ia.contacts(world.frames(), world.graph, settings)
    assert {r: len(where) for r, (_l1, where) in found.items()} == {3: 70, 4: 30}


@pytest_rdkit
def test_the_best_candidate_that_can_be_used_is_chosen_and_without_one_there_is_no_restraint():
    """The interactions first; a candidate that the rules of every restraint refuse is passed over
    and keeps the reason; if the interactions give none that can be used, the distance method; if
    that gives none either, the calculation stops."""
    nearer = np.full(100, 0.45)
    nearer[:50] = 0.35                      # the backbone of residue 4 is the nearest one in frames 0 to 49
    world = World(100).residue(3).residue(4, alpha=60.0, distance=nearer)
    records = world.records((3, "HBAcceptor", range(100)))
    frames = world.frames()
    choice = world.ia.choose_anchors(frames, world.graph, records)
    best = choice.ranked["interactions"][0]
    assert choice.method == "interactions" and (choice.host, choice.guest) == (best.host, best.guest)
    assert choice.chosen.residue == 3 and "distance" not in choice.ranked and best.refused is None

    second = world.ia.choose_anchors(frames, world.graph, records, problem_of=lambda host, guest: (
        "on a straight line" if (host, guest) == (best.host, best.guest) else None))
    assert second.chosen is second.ranked["interactions"][1]
    assert second.ranked["interactions"][0].refused == "on a straight line"

    by_distance = world.ia.choose_anchors(frames, world.graph, records, problem_of=lambda host, guest: (
        "refused" if host[0] // 10 == 3 else None))
    assert by_distance.method == "distance" and by_distance.chosen.residue == 4
    assert all(c.refused == "refused" for c in by_distance.ranked["interactions"])
    assert world.ia.choose_anchors(frames, world.graph, [[] for _ in range(100)]).method == "distance"

    with pytest.raises(RestraintError, match="no Boresch anchors can be chosen from the interactions"):
        world.ia.choose_anchors(frames, world.graph, records, problem_of=lambda host, guest: "refused")
    far = World(100).residue(3, distance=1.5)
    with pytest.raises(RestraintError):
        far.ia.choose_anchors(far.frames(), far.graph, [[] for _ in range(100)])


# ---- review of eb6f89f (Codex): through the choice of a leg -------------------------------------------------

@pytest_rdkit
def test_the_distance_method_measures_to_the_nearest_image_in_a_skewed_cell():
    """Review of eb6f89f (P1). Cell (4, 0, 0), (0, 4, 0), (2, 2, 2.83) nm and
    ``fallback_cutoff_nm: 2.0``: the backbone of residue 0 is 1.5 nm from the ligand's N along z, that of
    residue 1 is 1.6 nm along x. OpenMM's formula for the minimum image made 3.1 nm of the first, and every
    frame was counted for residue 1."""
    world = World(10)
    l1 = world.ligand[world.L1]
    world.backbones[0] = np.repeat((l1 + [[0.0, 0.0, 1.5], [0.0, 0.1, 1.6], [0.1, 0.1, 1.7]])[None], 10, axis=0)
    world.backbones[1] = np.repeat((l1 + [[1.6, 0.0, 0.0], [1.7, 0.1, 0.0], [1.8, 0.1, 0.1]])[None], 10, axis=0)
    box = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 2.0 * np.sqrt(2.0)]])
    settings = world.ia.InteractionSettings(fallback_cutoff_nm=2.0)
    found = world.ia.contacts(world.frames(boxes=np.repeat(box[None], 10, axis=0)), world.graph, settings)
    assert {residue: len(where) for residue, (_l1, where) in found.items()} == {0: 10}


def leg_of(world, records, monkeypatch, **restraints_settings):
    """What ``workflow.interaction_restraint`` gets for the residues 0, 1, ... of ``world``, whose last frame is
    the configuration the alchemical run starts from: ``(cfg, prep, molecule, eq, trajectories)``. Residue r of
    the topology has N, CA and C as the atoms 3 r, 3 r + 1 and 3 r + 2, and the ligand follows. In the place of
    ProLIF and DSSP: ``records`` (one list per frame) and no secondary structure."""
    import openmm.app as app

    from cufep.analysis import interactions
    from cufep.config.schema import Config
    from cufep.sampling.restraint_md import Trajectories

    residues = sorted(world.backbones)
    assert residues == list(range(len(residues)))
    top = app.Topology()
    chain = top.addChain("A")
    for r in residues:
        residue = top.addResidue("GLY", chain, id=str(100 + r))
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
            top.addAtom(name, app.element.get_by_symbol(element), residue)
    ligand_residue = top.addResidue("LIG", top.addChain("L"), id="1")
    for atom in world.mol.GetAtoms():
        top.addAtom(f"{atom.GetSymbol()}{atom.GetIdx()}", app.element.get_by_symbol(atom.GetSymbol()), ligand_residue)
    xyz = world.frames().xyz
    n, n_atoms = xyz.shape[0] - 1, xyz.shape[1]
    boxes = np.repeat(BOX[None], n, axis=0)
    monkeypatch.setattr(interactions, "interactions_of_frames", lambda *args, **kwargs: list(records))
    monkeypatch.setattr(restraints, "stable_structure_fractions", lambda *args, **kwargs: {})
    settings = {"interactions": {"discard_ns": 0.0}}
    for key, value in restraints_settings.items():
        if key in ("angle_range_deg", "min_frame_fraction"):
            settings["interactions"][key] = value
        else:
            settings[key] = value
    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"}, "restraints": settings})
    trajectories = Trajectories(frames=xyz[:-1], boxes=boxes, frame_atoms=np.arange(n_atoms),
                                trajectory=np.arange(n) % 2, time_ps=10.0 * (1 + np.arange(n) // 2),
                                backends=["openmm", "openmm"], timesteps_fs=[4.0, 4.0], notes=[])
    prep = {"topology": top, "ligand": list(range(3 * len(residues), n_atoms))}
    return cfg, prep, world.mol, {"positions": xyz[-1], "box": BOX}, trajectories


def test_anchors_whose_frames_give_no_reference_values_are_a_problem_of_the_candidate():
    """Review of eb6f89f (P2). Anchors can be candidates, be ranked first and pass OpenFE's rules without one
    frame in which theta_A and theta_B are both in range (P1-L1-L2 in range in the first half of the frames,
    L1-P1-P2 in the second): the histograms give them no reference values. The calculation stopped there
    ("in no frame do both anchor angles") although the next candidate could be the restraint. It is a reason
    to pass a candidate over, as a broken rule of OpenFE is (the choice of a leg asks both of every candidate:
    the test of the force constants below goes through it)."""
    never_both = coordinates(800)
    never_both[:400, 1] = np.radians(30.0)
    never_both[400:, 2] = np.radians(30.0)
    assert "in no frame do both anchor angles (theta_A, theta_B) lie within 45 to 135 degrees" in \
        restraints.reference_values_problem(never_both)
    with pytest.raises(RestraintError, match="in no frame do both anchor angles"):
        restraints.reference_values_from_histograms(never_both)
    assert restraints.reference_values_problem(coordinates(800)) is None
    assert restraints.reference_values_problem(coordinates(800, theta_a=88.0, theta_b=90.0), (89.0, 91.0)) is not None


@pytest.mark.needs("rdkit", "prolif", "MDAnalysis")
def test_a_candidate_is_held_to_the_flat_angle_rule_with_the_force_constants_it_gets(monkeypatch):
    """Review of eb6f89f (P1), through the choice of a leg with ``restraints.force_constants: auto``. Residue 0
    is ranked first; its theta_A is 60 and 120 degrees in the frames and 30 degrees where the run starts. With
    the constants its fluctuations give (20 kcal/mol/rad^2, margin 44 degrees) it breaks OpenFE's rule and
    was chosen all the same. With the configured constants it is the restraint, as before."""
    from cufep import workflow

    n = 800
    theta = np.where(np.arange(n + 1) % 2 == 0, 60.0, 120.0)
    theta[-1] = 30.0
    world = World(n + 1).residue(0, alpha=90.0, theta=theta).residue(1, alpha=90.0, theta=90.0, distance=0.45)
    records = world.records((0, "HBAcceptor", range(n)), (1, "HBAcceptor", range(320)))[:n]
    fixed, fixed_info, _report = workflow.interaction_restraint(*leg_of(world, records, monkeypatch), 298.15)
    assert fixed_info["start_values"]["theta_A"] == pytest.approx(30.0) and fixed_info["candidates_refused"] == 0
    restraint, info, report = workflow.interaction_restraint(
        *leg_of(world, records, monkeypatch, force_constants="auto"), 298.15)
    margins = [np.degrees(restraints.flat_margin_of(k, 298.15)) for k in (restraint.k_theta_a, restraint.k_theta_b)]
    start = [info["start_values"]["theta_A"], info["start_values"]["theta_B"]]
    assert all(m <= angle <= 180.0 - m for m, angle in zip(margins, start))
    # every candidate that was ranked before it and passed over is in interactions.json, with the reason
    passed_over = report["refused_candidates"]
    assert info["candidates_refused"] == len(passed_over) >= 1
    assert any("comes within 44 degrees of a straight line" in c["refused"] for c in passed_over)


@pytest.mark.needs("rdkit", "prolif", "MDAnalysis")
def test_the_result_names_the_frames_of_the_interaction_it_names(monkeypatch):
    """Review of eb6f89f (P1). One ligand atom and residue with a hydrogen bond in the frames 0 to 299 and an
    ionic interaction in 200 to 399 of 500: the candidate rests on the 400 frames with either and is named
    after the ionic one, which is in 200. The result said "Anionic ... in 400 of 500 frames"."""
    from cufep import workflow

    n = 500
    world = World(n + 1).residue(0)
    records = world.records((0, "HBAcceptor", range(300)), (0, "Anionic", range(200, 400)))[:n]
    _restraint, info, report = workflow.interaction_restraint(*leg_of(world, records, monkeypatch), 298.15)
    assert info["interaction"] == {"residue": "GLY100", "type": "Anionic", "frames": 200, "of_frames": 500,
                                   "frames_with_any_kept_type": 400,
                                   "kept_types": {"Anionic": 200, "HBAcceptor": 300}}
    assert report["chosen"]["n_frames"] == 400
    assert {(row["interaction"], row["frames"]) for row in report["interactions"]} == {("HBAcceptor", 300),
                                                                                       ("Anionic", 200)}


# ---- review of 877ab88 (Codex) ----------------------------------------------------------------------------

def test_the_rule_of_the_flat_angle_is_asked_once_with_the_constants_and_the_reference_of_the_restraint():
    """Review of 877ab88 (P1), three cases at 298.15 K.

    * Constants from fluctuations that are stronger: theta_A at 90 degrees in every frame and at 18 where the
      run starts gets 320 kcal/mol/rad^2 (the upper limit), whose margin is 11.03 degrees: 18 degrees is 26.7 kT
      from a straight line. The anchors were refused all the same, by the margin of the configured 80 (22.05).
    * The reference angle: with 10 kcal/mol/rad^2 (margin 62.37 degrees) theta_A at 62.8 degrees in every frame
      passes, and its reference is the centre of the bin from 54 to 63, 58.5 degrees: 8.8 kT. The reference
      angles are held to the rule too.
    * The last frame is not the reference of a restraint whose values come from histograms: anchors on a line
      in that one frame were refused, and not with the same frame anywhere else."""
    k_80 = 80.0 * 4.184
    boxes = np.repeat(BOX[None], 8, axis=0)
    frames = np.repeat(chain(90.0)[None], 8, axis=0)
    start = np.zeros((20, 3))
    start[:7] = chain(18.0)
    m = restraints.measure_anchors(HOST, GUEST, LIGAND, frames, np.arange(7), boxes, None, start=start,
                                   start_box=BOX, start_among_frames=True)
    assert m.problem is None and np.degrees(m.held_theta[-1, 0]) == pytest.approx(18.0)
    trial = restraints.BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    stronger, problem = restraints.constants_from_fluctuations(trial, m.values, m.held_theta, 298.15)
    assert stronger.k_theta_a == pytest.approx(4 * k_80) and problem is None
    assert np.degrees(restraints.flat_margin_of(4 * k_80, 298.15)) == pytest.approx(11.03, abs=0.01)
    assert "within 22 degrees" in restraints.flat_rule_problem(k_80, k_80, m.held_theta, 298.15)

    k_10 = 10.0 * 4.184
    assert np.degrees(restraints.flat_margin_of(k_10, 298.15)) == pytest.approx(62.37, abs=0.01)
    observed = coordinates(100, theta_a=62.8, theta_b=90.0)
    reference = restraints.reference_values_from_histograms(observed)
    assert np.degrees(reference[1]) == pytest.approx(58.5)
    assert restraints.flat_rule_problem(k_10, k_10, observed[:, 1:3], 298.15) is None
    with_reference = np.concatenate([observed[:, 1:3], [reference[1:3]]])
    assert "58.5 to 62.8 degrees" in restraints.flat_rule_problem(k_10, k_10, with_reference, 298.15)
    assert 0.5 * k_10 * np.radians(58.5) ** 2 / (restraints.R_GAS * 298.15) == pytest.approx(8.80, abs=0.01)

    straight = chain()
    straight[4] = straight[3] + [0.0, 0.15, 0.0]
    straight[5] = straight[3] + [0.0, 0.30, 0.0]
    for where in (0, 7):
        some = frames.copy()
        some[where] = straight
        from_histograms = restraints.measure_anchors(HOST, GUEST, LIGAND, some, np.arange(7), boxes, None,
                                                     start=frames[1], start_box=BOX, start_among_frames=True)
        assert from_histograms.problem is None, where
    last = frames.copy()
    last[7] = straight
    of_the_last_frame = restraints.measure_anchors(HOST, GUEST, LIGAND, last, np.arange(7), boxes, None)
    assert "exactly on a line in the last frame" in of_the_last_frame.problem


@pytest_rdkit
def test_a_frame_whose_nearest_backbone_atom_is_of_a_cap_is_a_frame_of_no_host():
    """Review of 877ab88 (P1). The source of the method looks for the nearest pair among every atom named N, CA
    or C of the protein, the C of an acetyl cap among them, and leaves out the residues without three such
    atoms afterwards. With the hosts alone, the six frames in which the cap is nearest (0.30 nm) went to the
    host that comes next (0.45 nm), which then had ten of ten frames."""
    world = World(10).residue(0, distance=0.45).residue(1, distance=np.where(np.arange(10) < 6, 0.30, 1.5))
    settings = world.ia.InteractionSettings()
    frames = world.frames()
    assert {r: len(where) for r, (_l1, where) in world.ia.contacts(frames, world.graph, settings).items()} == \
        {0: 4, 1: 6}                                                # two hosts: each has its frames
    capped = world.frames()
    capped.hosts.pop(1)
    capped.other_backbone = {1: world.host(1)[:1]}                  # residue 1 is a cap with one such atom
    found = world.ia.contacts(capped, world.graph, settings)
    assert {r: len(where) for r, (_l1, where) in found.items()} == {0: 4}


@pytest.mark.needs("rdkit", "prolif", "MDAnalysis")
def test_what_the_choice_of_a_leg_records_of_the_distance_method_and_of_what_it_tested(monkeypatch):
    """Review of 877ab88. A contact of the distance method is no interaction of ProLIF: the record said
    ``kept_types: {}`` and 800 frames "with any kept type". ``candidates_tested`` held the number of candidates
    that were ranked (32 in the TYK2 example, of which one was held to the rules): it is the number that were,
    and ``candidates_ranked`` the other. A candidate without reference values is asked for them (the choice
    stops there otherwise, review of eb6f89f). And the thresholds are of the frames there are: an MD that was
    repeated at 2 fs can keep fewer than were planned, and a threshold of no frame is refused then too."""
    from cufep import workflow
    from cufep.workflow import WorkflowError

    n = 100
    world = World(n + 1).residue(0).residue(1, alpha=60.0, distance=0.6)
    nothing = [[] for _ in range(n)]
    _restraint, info, report = workflow.interaction_restraint(*leg_of(world, nothing, monkeypatch), 298.15)
    assert info["method"] == "distance"
    assert info["interaction"] == {"residue": "GLY100", "type": "contact", "frames": n, "of_frames": n,
                                   "frames_with_any_kept_type": None, "kept_types": None}
    assert info["candidates_tested"] == info["candidates_refused"] + 1 == 1
    assert info["candidates_ranked"] == report["candidates_in_all"]["distance"] > 1
    assert report["start"] == info["start"] if "start" in info else "start" not in report

    asked = []
    passes = restraints.reference_values_problem

    def no_reference_for_the_first(values, angle_range_deg=(45.0, 135.0)):
        asked.append(len(values))
        return "in no frame do both anchor angles lie in range" if len(asked) == 1 else passes(values, angle_range_deg)

    monkeypatch.setattr(restraints, "reference_values_problem", no_reference_for_the_first)
    _restraint, info, report = workflow.interaction_restraint(*leg_of(world, nothing, monkeypatch), 298.15)
    assert info["candidates_refused"] == 1 and info["candidates_tested"] == 2
    assert report["refused_candidates"][0]["refused"] == "in no frame do both anchor angles lie in range"
    monkeypatch.setattr(restraints, "reference_values_problem", passes)

    few = World(6 + 1).residue(0)
    with pytest.raises(WorkflowError, match="6 frame.s. are analysed and fallback_frame_fraction = 0.15 of them is "
                                            "less than one frame"):
        workflow.interaction_restraint(*leg_of(few, [[] for _ in range(6)], monkeypatch), 298.15)


@pytest.mark.needs("rdkit", "prolif", "MDAnalysis")
def test_the_choice_of_a_leg_asks_the_flat_angle_rule_once_with_what_the_restraint_gets(monkeypatch):
    """The third review by Codex: the tests of the rule of the flat angle gave the functions their arguments
    themselves, so that a choice which went back to asking it with the configured constants, or of the frames
    alone, passed them. Through the choice of a leg with ``restraints.force_constants: auto``: no candidate is
    measured with a margin of its own (the configured constants), and the rule is asked of every candidate
    with the constants its fluctuations give it, of the frames, the start and its own reference angles."""
    from cufep import workflow

    n = 200
    theta = np.where(np.arange(n + 1) % 2 == 0, 60.0, 120.0)
    theta[-1] = 30.0
    world = World(n + 1).residue(0, alpha=90.0, theta=theta).residue(1, alpha=90.0, theta=90.0, distance=0.45)
    records = world.records((0, "HBAcceptor", range(n)), (1, "HBAcceptor", range(100)))[:n]
    margins, asked = [], []
    measure, rule = restraints.measure_anchors, restraints.flat_rule_problem

    def measured(*args, **kwargs):
        margins.append(args[6])
        return measure(*args, **kwargs)

    def flat_rule(k_theta_a, k_theta_b, held, temperature_K, *more):
        asked.append((k_theta_a, k_theta_b, np.array(held)))
        return rule(k_theta_a, k_theta_b, held, temperature_K, *more)

    monkeypatch.setattr(restraints, "measure_anchors", measured)
    monkeypatch.setattr(restraints, "flat_rule_problem", flat_rule)
    restraint, info, _report = workflow.interaction_restraint(
        *leg_of(world, records, monkeypatch, force_constants="auto"), 298.15)
    assert margins and set(margins) == {None}
    k_a, k_b, held = asked[-1]                                     # the judgement of the candidate that was taken
    assert (k_a, k_b) == (restraint.k_theta_a, restraint.k_theta_b) and held.shape == (n + 2, 2)
    assert held[-1].tolist() == [restraint.theta_a0, restraint.theta_b0]         # its reference angles
    assert np.degrees(held[-2]).tolist() == pytest.approx([info["start_values"]["theta_A"],
                                                           info["start_values"]["theta_B"]])
    assert info["flat_angle_rule"] == workflow.FLAT_ANGLE_RULE and info["candidates_refused"] >= 1
    # the restraint that was written is one this version uses again (the third review: no restraint said by which
    # rule its anchors were held, and a leg with constants from fluctuations refused its own restraint)
    from cufep.workflow import check_restraint_file, restraint_made_with

    cfg = leg_of(world, records, monkeypatch, force_constants="auto")[0]
    stored = {"restraint": restraint.to_dict(), "info": info, "made_with": restraint_made_with(cfg)}
    assert check_restraint_file(json.loads(json.dumps(stored)), "restraint.json", cfg, lambda line: None) is True


def test_given_anchors_are_held_to_the_rule_in_the_pose_the_pre_equilibration_started_from():
    """The third review by Codex: the test of given anchors with constants from fluctuations had the angle of
    30 degrees among the frames. Here it is in the prepared pose alone: frames at 60 and 120 degrees give
    20 kcal/mol/rad^2 (margin 44 degrees) and pass by themselves; with the pose the leg gives, they are refused."""
    from cufep.workflow import WorkflowError, manual_boresch_restraint

    frames = np.array([chain(60.0 if f % 2 == 0 else 120.0) for f in range(20)])
    eq = {"frames": frames, "frame_atoms": np.arange(7), "frame_boxes": np.repeat(BOX[None], 20, axis=0),
          "positions": frames[-1], "box": BOX}
    k_r, k_80 = 10.0 * 4.184 * 100.0, 80.0 * 4.184
    alone, _fluctuations = manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 7, k_r, k_80, force_constants="auto")
    assert alone.k_theta_a == pytest.approx(20.0 * 4.184)
    with pytest.raises(WorkflowError, match="with the force constants of their fluctuations .*30.0 to 120.0"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 7, k_r, k_80, force_constants="auto",
                                 input_pose=chain(30.0), input_box=BOX)

