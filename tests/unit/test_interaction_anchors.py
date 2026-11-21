"""The Boresch restraint of restraints.selection: interactions: its reference values, the rules
its anchors are held to, and what the rule reads of the protein."""
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
