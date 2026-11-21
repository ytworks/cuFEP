"""Every geometric quantity between the ligand and the protein is the same for a ligand that is one
or more cells away (review of PR 1 on a3462a5: the manual Boresch restraint and the pose check).

Coordinates are not wrapped into the cell, and the frames of a leg come from whichever replica is
in the coupled state, so a ligand may be any number of lattice vectors away from the protein. The
forces measure across the periodic boundary; everything cuFEP computes from coordinates must too.
"""
import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest

from cufep.alchemy.restraints import BoreschRestraint, nearest_image_of_ligand
from cufep.analysis.structure import structural_analysis
from cufep.workflow import WorkflowError, manual_boresch_restraint

K_R = 10.0 * 4.184 * 100.0          # restraints.k_r_kcal_per_A2 = 10 in kJ/mol/nm^2
K_A = 80.0 * 4.184

CUBE = np.eye(3) * 5.0
D = 5.0                              # rhombic dodecahedron of image distance 5 nm, OpenMM's reduced form
DODECAHEDRON = np.array([[D, 0.0, 0.0], [0.0, D, 0.0], [D / 2, D / 2, D * np.sqrt(2) / 2]])

# three host atoms and a ligand of three atoms whose first atom is 0.4 nm from the first host atom
POSITIONS = np.array([[1.0, 1.0, 1.0], [1.0, 1.15, 1.0], [1.15, 1.15, 1.0],
                      [0.6, 1.0, 1.0], [0.6, 0.85, 1.0], [0.6, 0.85, 1.15]])
HOST, GUEST, LIGAND = (0, 1, 2), (3, 4, 5), np.array([3, 4, 5])


def lattice_shifts(box):
    a, b, c = box
    return {"none": np.zeros(3), "-a": -a, "a+b": a + b, "2c": 2 * c, "-a-2b+c": -a - 2 * b + c}


def restraint_energy(restraint: BoreschRestraint, positions, box) -> float:
    """The energy of the restraint's own OpenMM force, which measures across the periodic boundary."""
    system = mm.System()
    for _ in range(len(positions)):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    system.addForce(restraint.create_force())
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    context.setPositions(positions * unit.nanometer)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def equilibration_with(positions, box):
    return {"frame_atoms": np.arange(len(positions)), "frames": np.array([positions, positions]),
            "frame_boxes": np.array([box, box])}


def test_the_reviewers_manual_restraint_one_cell_away():
    """5 nm cell, ligand 0.4 nm from the host, moved by one cell: from the raw coordinates r0 was 5.4 nm and
    the restraint held its own reference configuration with 52,300 kJ/mol."""
    moved = POSITIONS.copy()
    moved[LIGAND] -= CUBE[0]
    raw_values = BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0).values(moved)
    old = BoreschRestraint(HOST, GUEST, *[float(x) for x in raw_values], K_R, K_A, K_A, K_A, K_A, K_A)
    assert old.r0 == pytest.approx(5.4) and restraint_energy(old, moved, CUBE) == pytest.approx(52300.0, rel=1e-6)
    new, _ = manual_boresch_restraint(HOST, GUEST, LIGAND, equilibration_with(moved, CUBE), len(POSITIONS), K_R, K_A)
    assert new.r0 == pytest.approx(0.4) and restraint_energy(new, moved, CUBE) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("box", [CUBE, DODECAHEDRON], ids=["cube", "dodecahedron"])
def test_the_manual_restraint_is_the_same_in_every_image(box):
    reference, _ = manual_boresch_restraint(HOST, GUEST, LIGAND, equilibration_with(POSITIONS, box), len(POSITIONS),
                                            K_R, K_A)
    for name, shift in lattice_shifts(box).items():
        moved = POSITIONS.copy()
        moved[LIGAND] += shift
        r, _ = manual_boresch_restraint(HOST, GUEST, LIGAND, equilibration_with(moved, box), len(POSITIONS), K_R, K_A)
        fields = ("r0", "theta_a0", "theta_b0", "phi_a0", "phi_b0", "phi_c0")
        assert [getattr(r, f) for f in fields] == pytest.approx([getattr(reference, f) for f in fields]), name
        assert restraint_energy(r, moved, box) == pytest.approx(0.0, abs=1e-6), name


def test_a_manual_restraint_needs_the_boxes_of_the_frames():
    eq = equilibration_with(POSITIONS, CUBE)
    del eq["frame_boxes"]
    with pytest.raises(WorkflowError, match="holds no boxes of its frames"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, eq, len(POSITIONS), K_R, K_A)


def _coupled_frames(box, shifts):
    """Frames of a protein of eight C-alpha atoms and a ligand of three atoms that never moves, with the
    ligand put some lattice vectors away in some frames."""
    rng = np.random.default_rng(3)
    ca = rng.normal(scale=0.6, size=(8, 3)) + 2.5
    ligand = np.array([[2.5, 2.5, 3.1], [2.6, 2.5, 3.2], [2.5, 2.6, 3.3]])
    frames = []
    for shift in shifts:
        frames.append(np.vstack([ca, ligand + shift]) + rng.normal(scale=0.002, size=(11, 3)))
    n = len(frames)
    return {"frames": np.array(frames, dtype=np.float32), "frame_iter": np.arange(n) * 4,
            "frame_box": np.repeat(np.asarray(box, dtype=np.float32)[None], n, axis=0),
            "attrs": {"frame_ca": list(range(8)), "frame_ligand": [8, 9, 10]}}


def test_the_reviewers_pose_check_one_cell_away():
    """5 nm cell, the same pose in every frame, the ligand one cell away in every other frame: the pose check
    reported an RMSD and a drift of the centre of 50 A and said the ligand may have left the pocket."""
    shifts = [np.zeros(3) if f % 2 == 0 else CUBE[0] for f in range(12)]
    samples = _coupled_frames(CUBE, shifts)
    without_boxes = structural_analysis({k: v for k, v in samples.items() if k != "frame_box"})
    assert without_boxes["ligand_rmsd_max_A"] == pytest.approx(50.0, abs=0.2) and "warnings" not in without_boxes
    assert without_boxes["periodic_images_checked"] is False
    result = structural_analysis(samples)
    assert result["ligand_rmsd_max_A"] < 0.1 and result["ligand_com_drift_max_A"] < 0.1
    assert result["periodic_images_checked"] is True


@pytest.mark.parametrize("box", [CUBE, DODECAHEDRON], ids=["cube", "dodecahedron"])
def test_the_pose_check_is_the_same_in_every_image(box):
    shifts = list(lattice_shifts(box).values())
    result = structural_analysis(_coupled_frames(box, shifts * 2))
    still = structural_analysis(_coupled_frames(box, [np.zeros(3)] * (2 * len(shifts))))
    assert result["ligand_rmsd_max_A"] == pytest.approx(still["ligand_rmsd_max_A"], abs=0.05)
    assert result["ligand_com_drift_max_A"] < 0.1


@pytest.mark.parametrize("box", [CUBE, DODECAHEDRON], ids=["cube", "dodecahedron"])
def test_the_nearest_image_is_found_however_many_cells_away(box):
    for name, shift in lattice_shifts(box).items():
        moved = POSITIONS.copy()
        moved[LIGAND] += shift + 3 * box[0] - 2 * box[2]
        back = nearest_image_of_ligand(moved, LIGAND, POSITIONS[list(HOST)].mean(axis=0), box)
        assert np.allclose(back, POSITIONS, atol=1e-9), name


# ---- review of PR 1 on 64751ca: anchors that cannot define the restraint ----------------------------------

def _nitrile_frames(n=20, seed=0):
    """Host atoms as above and a guest N-C-C on a straight line (acetonitrile), with small thermal noise."""
    rng = np.random.default_rng(seed)
    base = POSITIONS.copy()
    base[4] = base[3] + [0.0, -0.116, 0.0]
    base[5] = base[4] + [0.0, -0.146, 0.0]
    return base[None] + rng.normal(scale=0.002, size=(n, 6, 3))


def test_a_guest_on_a_straight_line_is_refused_and_would_give_nan():
    """P1: an acetonitrile N-C-C was chosen as guest anchors; theta_A and theta_B were in order and the force of the
    restraint at its own reference was NaN. OpenFE's is_collinear on H2-H1-H0-G0-G1-G2 refuses it."""
    from cufep.alchemy.restraints import RestraintSearchSettings, anchor_geometry_problem, anchor_margins

    frames = _nitrile_frames()
    probe = BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    exact = frames[-1].copy()
    exact[4] = exact[3] + [0.0, -0.116, 0.0]
    exact[5] = exact[4] + [0.0, -0.146, 0.0]
    values = probe.values(exact)
    straight = BoreschRestraint(HOST, GUEST, *[float(x) for x in values], K_R, K_A, K_A, K_A, K_A, K_A)
    system = mm.System()
    for _ in range(6):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in CUBE])
    system.addForce(straight.create_force())
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(exact * unit.nanometer)
    forces = (
        context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    )
    assert not np.all(np.isfinite(forces))                                      # what the reviewer saw
    problem = anchor_geometry_problem(probe, frames, anchor_margins(RestraintSearchSettings(), 298.15))
    assert problem is not None and "on a straight line in every frame" in problem and "is_collinear" in problem
    assert anchor_geometry_problem(probe, POSITIONS[None], anchor_margins(RestraintSearchSettings(), 298.15)) is None
    eq = {"frame_atoms": np.arange(6), "frames": frames, "frame_boxes": np.repeat(CUBE[None], len(frames), axis=0)}
    with pytest.raises(WorkflowError, match="on a straight line in every frame"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 6, K_R, K_A)


def test_manual_anchors_take_their_force_constants_from_their_fluctuations():
    """P2: with selection: manual, force_constants: auto was ignored and the configured constants used."""
    from cufep.alchemy.restraints import coordinate_fluctuations, with_force_constants_from_fluctuations

    rng = np.random.default_rng(4)
    frames = POSITIONS[None] + rng.normal(scale=0.01, size=(200, 6, 3))
    eq = {"frame_atoms": np.arange(6), "frames": frames, "frame_boxes": np.repeat(CUBE[None], len(frames), axis=0)}
    fixed, fluct = manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 6, K_R, K_A)
    auto, fluct_auto = manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 6, K_R, K_A, force_constants="auto")
    assert fixed.k_r == K_R and fluct == fluct_auto
    expected = with_force_constants_from_fluctuations(fixed, fluct, 298.15)
    assert auto.k_r == pytest.approx(expected.k_r) and auto.k_phi_c == pytest.approx(expected.k_phi_c)
    assert auto.k_r != K_R
    assert fluct == pytest.approx(
        coordinate_fluctuations(BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0).values(frames))
    )


def _site_and_ligand(n_frames=10, seed=5):
    """A topology of eight alanines given by their backbone and a ligand of three carbons, with frames of a still
    protein; returns (topology, input positions, frames of the input pose, ligand atoms)."""
    import openmm.app as app

    top = app.Topology()
    chain = top.addChain()
    rng = np.random.default_rng(seed)
    positions = []
    for k in range(8):
        res = top.addResidue("ALA", chain, id=str(k + 1))
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
            top.addAtom(name, app.element.get_by_symbol(element), res)
            positions.append(rng.normal(scale=0.5, size=3) + 2.5)
    lig_res = top.addResidue("LIG", top.addChain())
    for name in ("C1", "C2", "C3"):
        top.addAtom(name, app.element.carbon, lig_res)
    positions += [[2.5, 2.5, 3.1], [2.65, 2.5, 3.15], [2.7, 2.62, 3.3]]
    positions = np.asarray(positions)
    ligand = np.arange(24, 27)
    frames = np.array([positions + rng.normal(scale=0.002, size=positions.shape) for _ in range(n_frames)])
    return top, positions, frames, ligand


@pytest.mark.needs("rdkit")
@pytest.mark.parametrize("box", [CUBE, DODECAHEDRON], ids=["cube", "dodecahedron"])
def test_manual_anchors_measure_how_far_the_ligand_moved_from_its_input_pose(box):
    """Review of 037bdf1 (P2): with manual anchors only the fluctuations of the restraint were stored, so how far a
    ligand had moved from its input pose was not known. The measure of the restraint search (periodic image,
    superposition on the site, symmetry) is used for them: one cell (5 nm) away is RMSD 0; 0.6 nm away from the site
    is 0.6 nm. It is a number of the result, without a verdict (the owner, 2026-11-20)."""
    from rdkit import Chem

    from cufep.alchemy import restraints
    from cufep.alchemy.restraints import ligand_pose_rmsd

    top, given, frames, ligand = _site_and_ligand()
    rdmol = Chem.MolFromSmiles("CCC")
    boxes = np.repeat(box[None], len(frames), axis=0)

    def rmsd(moved_frames):
        return ligand_pose_rmsd(top, moved_frames, ligand, rdmol, given, frame_boxes_nm=boxes, input_box_nm=box)

    for name, shift in lattice_shifts(box).items():
        moved = frames.copy()
        moved[:, ligand] += shift
        assert rmsd(moved) == pytest.approx(0.0, abs=0.01), name
    away = frames.copy()
    away[:, ligand] += np.array([0.6, 0.0, 0.0]) + box[0]          # 0.6 nm from the site, one cell away
    assert rmsd(away) == pytest.approx(0.6, abs=0.01)
    assert not hasattr(restraints, "pose_note")                     # no level and no note


def _pose_with_theta_a(degrees):
    """POSITIONS with the guest moved so that the anchor angle H1-H0-G0 (theta_A) is ``degrees``; the other angles
    stay far from a straight line."""
    pose = POSITIONS.copy()
    a = np.radians(degrees)
    pose[3] = pose[0] + 0.4 * np.array([-np.sin(a), np.cos(a), 0.0])
    pose[4] = pose[3] + [-0.15, 0.0, 0.0]
    pose[5] = pose[4] + [0.0, 0.0, 0.15]
    return pose


def test_the_prepared_pose_is_held_to_openfes_rule_of_every_frame(tmp_path, monkeypatch):
    """Independent review of c691e37 (P2) and the owner, 2026-11-20: the anchors are held to OpenFE's rules only. The
    rule of cuFEP's own that the frame the reference values come from be 30 degrees from a straight line is gone:
    theta_A
    of 27 degrees passes in the prepared pose, and a last frame of 25 degrees among frames of up to 45 passes too (25
    degrees is on a straight line by OpenFE's is_collinear, |cos| 0.906, which refuses anchors only on the line in every
    frame); 10 degrees is within 22 degrees (10 kT with the default constants, OpenFE's check_angle_not_flat) and is
    refused."""
    from cufep import workflow
    from cufep.alchemy.restraints import RestraintSearchSettings, anchor_geometry_problem, anchor_margins
    from cufep.config.schema import Config

    margin = anchor_margins(RestraintSearchSettings(k_r=K_R, k_angle=K_A), 298.15)
    assert np.degrees(margin) == pytest.approx(22.05, abs=0.01)
    probe = BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    assert anchor_geometry_problem(probe, _pose_with_theta_a(27.0)[None], margin) is None       # 30 degrees before
    assert "every frame" in anchor_geometry_problem(probe, _pose_with_theta_a(25.0)[None], margin)
    frames = np.array([_pose_with_theta_a(t) for t in np.linspace(45.0, 25.0, 8)])            # the last at 25
    assert anchor_geometry_problem(probe, frames, margin) is None
    frames[3] = _pose_with_theta_a(21.0)                                                        # one frame within 22
    assert "H1-H0-G0 (theta_A) comes within 22 degrees" in anchor_geometry_problem(probe, frames, margin)
    cfg = Config.model_validate(
        {
            "system": {"protein": "p.pdb", "ligand": "l.sdf"},
            "restraints": {"selection": "manual", "host_atoms": list(HOST), "guest_atoms": list(GUEST)},
        }
    )
    monkeypatch.setattr(workflow, "check_manual_anchors", lambda *args: None)  # atoms of protein and ligand: elsewhere
    pose = {"positions": _pose_with_theta_a(27.0), "ligand": list(LIGAND), "box": CUBE}
    monkeypatch.setattr(workflow, "load_prepared", lambda prep_dir, name: pose)
    workflow.check_before_jobs(cfg, tmp_path, {(0, "complex")})                  # accepted
    pose["positions"] = _pose_with_theta_a(10.0)  # within 22 degrees of a straight line
    with pytest.raises(
        WorkflowError, match="in the prepared pose: the anchor angle H1-H0-G0 .* comes within 22 degrees"
    ):
        workflow.check_before_jobs(cfg, tmp_path, {(0, "complex")})

    # review of 7b49066 (P2): theta_A of 25 degrees in the prepared pose (on a line by is_collinear, a rule of a
    # trajectory) and 25, 31, 36 and 40 degrees in the stored pre-equilibration: refused before any job although its
    # frames give the restraint. The prepared pose is held to the rule of every frame alone
    pose["positions"] = _pose_with_theta_a(25.0)
    workflow.check_before_jobs(cfg, tmp_path, {(0, "complex")})                  # accepted now
    stored = np.array([_pose_with_theta_a(t) for t in (25.0, 31.0, 36.0, 40.0)])
    restraint, _ = manual_boresch_restraint(HOST, GUEST, LIGAND, {"frame_atoms": np.arange(6), "frames": stored,
                                                                  "frame_boxes": np.repeat(CUBE[None], 4, axis=0)},
                                            6, K_R, K_A)
    assert np.isfinite(restraint_energy(restraint, stored[-1], CUBE))
    on_line = np.array([_pose_with_theta_a(25.0)] * 4)                         # 25 degrees in every frame: on a line
    with pytest.raises(WorkflowError, match="on a straight line in every frame"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, {"frame_atoms": np.arange(6), "frames": on_line,
                                                      "frame_boxes": np.repeat(CUBE[None], 4, axis=0)}, 6, K_R, K_A)


def test_anchors_whose_dihedrals_are_not_defined_in_the_last_frame_are_refused():
    """Review of 7b49066 (P2): the automatic choice took guest anchors bent in the earlier frames and exactly on a line
    in the last, from which the reference values come; the force of the restraint was NaN on OpenMM's Reference
    platform. Such anchors are refused, by whether the plane of the dihedral exists (the cross product of the two bond
    vectors is zero), not by a level; the search goes on with other anchors."""
    from cufep.alchemy.restraints import (
        RestraintSearchSettings,
        anchor_geometry_problem,
        anchor_margins,
        undefined_dihedral_planes,
    )

    frames = []
    for bend in (90.0, 70.0, 50.0, 30.0, 0.0):                  # the angle of G1-G2 away from the line G0-G1
        x = POSITIONS.copy()
        x[4] = x[3] + [0.0, -0.15, 0.0]
        a = np.radians(bend)
        x[5] = x[4] + 0.15 * np.array([0.0, -np.cos(a), np.sin(a)])
        frames.append(x)
    frames = np.array(frames)
    probe = BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    assert (
        undefined_dihedral_planes(probe, frames[-1]) == ["G0-G1-G2"]
        and undefined_dihedral_planes(probe, frames[0]) == []
    )
    margin = anchor_margins(RestraintSearchSettings(k_r=K_R, k_angle=K_A), 298.15)
    problem = anchor_geometry_problem(probe, frames, margin)
    assert problem.startswith("the anchors G0-G1-G2 are exactly on a line in the last frame")
    assert anchor_geometry_problem(probe, frames[:-1], margin) is None              # bent in the last frame: kept
    # the restraint that the last frame would give has a force of NaN
    values = probe.values(frames[-1])
    straight = BoreschRestraint(HOST, GUEST, *[float(v) for v in values], K_R, K_A, K_A, K_A, K_A, K_A)
    system = mm.System()
    for _ in range(6):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in CUBE])
    system.addForce(straight.create_force())
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(frames[-1] * unit.nanometer)
    forces = (
        context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    )
    assert not np.all(np.isfinite(forces))
    eq = {"frame_atoms": np.arange(6), "frames": frames, "frame_boxes": np.repeat(CUBE[None], len(frames), axis=0)}
    with pytest.raises(WorkflowError, match="exactly on a line in the last frame"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 6, K_R, K_A)

    # review of 10589a6 (P2): the earlier frames turned as a rigid body and all of them superposed on the host, as the
    # search does: rounding leaves the straight row bent in the superposed last frame. The copies of that configuration
    # (the frame before the superposition, the positions the run starts from) are held to the rule too
    from cufep.alchemy.restraints import kabsch_align

    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    turn = np.eye(3) + np.sin(0.7) * k + (1.0 - np.cos(0.7)) * k @ k
    turned = frames.copy()
    turned[:-1] = (turned[:-1] - 1.0) @ turn.T + 1.0
    aligned = kabsch_align(turned, turned[0], np.array(HOST))
    aligned = kabsch_align(aligned, aligned.mean(axis=0), np.array(HOST))
    assert undefined_dihedral_planes(probe, aligned[-1]) == []                   # what the check saw
    assert anchor_geometry_problem(probe, aligned, margin) is None
    assert "exactly on a line" in anchor_geometry_problem(probe, aligned, margin, reference_copies=[turned[-1]])
    # manual anchors: the frames are copies in single precision of the positions the run starts from
    bent = frames[:-1].tolist() + [frames[-1] + [0.0, 0.0, 1e-6] * (np.arange(6) == 5)[:, None]]
    eq = {
        "frame_atoms": np.arange(6),
        "frames": np.array(bent),
        "frame_boxes": np.repeat(CUBE[None], len(bent), axis=0),
        "positions": frames[-1],
        "box": CUBE,
    }
    with pytest.raises(WorkflowError, match="exactly on a line in the last frame"):
        manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 6, K_R, K_A)
    del eq["positions"]
    manual_boresch_restraint(HOST, GUEST, LIGAND, eq, 6, K_R, K_A)               # without them: the frames alone


def test_anchors_on_a_straight_line_are_refused_only_if_they_are_in_every_frame():
    """OpenFE's EvaluateBoreschAtoms: ``not all(collinear)``. Anchors with three in a row whose bond vectors have |cos|
    above 0.9 (OpenFE's is_collinear, from Yank) in every frame are refused; one frame off the line is enough to keep
    them. The angles H2-H1-H0 and G0-G1-G2 are not held to the flat margin any more (a rule of cuFEP's own)."""
    from cufep.alchemy.restraints import (
        COLLINEAR_THRESHOLD,
        RestraintSearchSettings,
        anchor_geometry_problem,
        anchor_margins,
        collinear_frames,
    )

    assert COLLINEAR_THRESHOLD == 0.9
    margin = anchor_margins(RestraintSearchSettings(), 298.15)
    probe = BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    line = _nitrile_frames()
    assert collinear_frames(probe, line).all() and not collinear_frames(probe, POSITIONS).any()
    assert "every frame" in anchor_geometry_problem(probe, line, margin)
    one_off = line.copy()
    one_off[7] = POSITIONS
    assert anchor_geometry_problem(probe, one_off, margin) is None
    # G0-G1-G2 of 160 degrees (cos 0.94 of the bond vectors): on the line by OpenFE's rule; 150 degrees (0.87) is not
    for angle, on_line in ((160.0, True), (150.0, False)):
        bent = POSITIONS.copy()
        a = np.radians(180.0 - angle)
        bent[4] = bent[3] + [0.0, -0.15, 0.0]
        bent[5] = bent[4] + 0.15 * np.array([np.sin(a), -np.cos(a), 0.0])
        assert bool(collinear_frames(probe, bent)[0]) is on_line, angle


# ---- review of f5a77ce: the image next to the protein is not the image nearest to its centre --------------

def _lopsided_complex(n_frames=40, seed=3):
    """A domain of 52 residues on one of 477, every residue its N, CA and C at the density of a protein (one
    residue in 0.137 nm^3), and pentan-1-ol 0.45 nm outside the small domain; in the cell that cuFEP's solvation
    gives this solute (a rhombic dodecahedron of width 2 x the radius of the solute + 1.0 nm). Returns
    (topology, frames, boxes, ligand atoms, C-alpha atoms, the molecule)."""
    import openmm.app as app
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rng = np.random.default_rng(seed)

    def ball(radius, centre):
        n = int(4.19 * radius ** 3 / 0.137)
        p = rng.normal(size=(4 * n, 3))
        p = p / np.linalg.norm(p, axis=1)[:, None] * radius * rng.random(4 * n)[:, None] ** (1 / 3)
        return p[:n] + np.asarray(centre, dtype=float)

    ca = np.concatenate([ball(1.2, (0.0, 0.0, 0.0)), ball(2.5, (-3.7, 0.0, 0.0))])
    backbone = np.stack([ca + [-0.10, 0.08, 0.0], ca, ca + [0.11, 0.07, 0.05]], axis=1).reshape(-1, 3)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCO"))
    AllChem.EmbedMolecule(mol, randomSeed=4)
    conformer = mol.GetConformer().GetPositions() / 10.0
    positions = np.concatenate([backbone, conformer - conformer.mean(axis=0) + [1.65, 0.0, 0.0]])
    top = app.Topology()
    chain = top.addChain("A")
    for k in range(len(ca)):
        residue = top.addResidue("GLY", chain, id=str(k + 1))
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
            top.addAtom(name, app.element.get_by_symbol(element), residue)
    ligand_residue = top.addResidue("LIG", top.addChain("L"))
    for atom in mol.GetAtoms():
        top.addAtom(f"{atom.GetSymbol()}{atom.GetIdx() + 1}", app.element.get_by_symbol(atom.GetSymbol()),
                    ligand_residue)
    centre = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    width = 2.0 * float(np.linalg.norm(positions - centre, axis=1).max()) + 1.0
    box = np.array([[width, 0.0, 0.0], [0.0, width, 0.0], [width / 2, width / 2, width * np.sqrt(2) / 2]])
    frames = positions[None] + rng.normal(scale=0.015, size=(n_frames, len(positions), 3))
    ligand = np.arange(3 * len(ca), len(positions))
    return top, frames, np.repeat(box[None], n_frames, axis=0), ligand, np.arange(1, 3 * len(ca), 3), mol


def _nearest_backbone_atom(frames, ligand):
    protein = np.setdiff1d(np.arange(frames.shape[1]), ligand)
    return np.linalg.norm(frames[:, ligand, None, :] - frames[:, None, protein, :], axis=-1).min(axis=(1, 2))


@pytest.mark.needs("rdkit")
def test_a_ligand_far_from_the_centre_of_a_lopsided_complex_stays_in_its_pocket():
    """Review of f5a77ce (P2): the ligand was brought to the image nearest to the centre of the C-alpha atoms. In
    this complex it is 4.93 nm from that centre in a cell 9.05 nm wide, and that image is another one: 1.19 nm
    or more from every backbone atom, where the MD wrote it 0.30 nm or less from one. The image next to the
    protein is the one of the nearest protein atom, wherever the ligand was written."""
    from cufep.alchemy.restraints import binding_site_frames

    top, frames, boxes, ligand, ca, _mol = _lopsided_complex()
    width = boxes[0, 0, 0]
    from_centre = np.linalg.norm(frames[0, ligand].mean(axis=0) - frames[0, ca].mean(axis=0))
    assert width == pytest.approx(9.05, abs=0.01) and from_centre == pytest.approx(4.93, abs=0.01)
    assert from_centre > width / 2
    written = _nearest_backbone_atom(frames, ligand)
    assert 0.2 < written.min() and written.max() < 0.31
    # the geometry of the finding: the image nearest to the centre of the protein is not the one in the pocket
    by_centre = nearest_image_of_ligand(frames, ligand, frames[:, ca, :].mean(axis=1), boxes)
    assert _nearest_backbone_atom(by_centre, ligand).min() > 1.19
    assert np.allclose(by_centre[0, ligand].mean(axis=0) - frames[0, ligand].mean(axis=0), [-width, 0.0, 0.0])
    # what the search of restraints.selection: auto and the RMSD of the pose look at
    looked_at, _aligned, _site = binding_site_frames(top, frames, ligand, ligand[:6], 1.5, boxes)
    assert np.allclose(looked_at, frames, atol=1e-9)
    # the rule they share
    from cufep.alchemy.restraints import ligand_next_to_protein

    assert np.allclose(ligand_next_to_protein(frames, ligand, ca, boxes), frames, atol=1e-9)
    for name, shift in lattice_shifts(boxes[0]).items():
        moved = frames.copy()
        moved[:, ligand] += shift + 2 * boxes[0, 1]
        assert np.allclose(ligand_next_to_protein(moved, ligand, ca, boxes), frames, atol=1e-9), name
    assert np.allclose(ligand_next_to_protein(frames[0], ligand, ca, boxes[0]), frames[0], atol=1e-9)


@pytest.mark.needs("rdkit")
def test_the_interactions_of_a_lopsided_complex_are_looked_for_in_the_pocket(monkeypatch):
    """The same complex through the choice of ``restraints.selection: interactions``: ProLIF and the rule got the
    ligand 1.21 to 1.28 nm from the nearest backbone atom, an image in which nothing interacts and no angle is
    that of the restraint (the distance method ranked six candidates there). They get it where the MD wrote it.
    Here no anchors can be chosen, and the calculation stops: pentan-1-ol outside a ball of backbone atoms."""
    from cufep import workflow
    from cufep.alchemy import restraints
    from cufep.analysis import interactions
    from cufep.config.schema import Config
    from cufep.sampling.restraint_md import Trajectories

    top, frames, boxes, ligand, _ca, mol = _lopsided_complex()
    n, n_atoms = frames.shape[0], frames.shape[1]
    seen = {}

    def no_interactions(topology, xyz, frame_atoms, ligand_atoms, rdmol):
        seen["nearest"] = _nearest_backbone_atom(np.asarray(xyz), ligand)
        return [[] for _ in range(len(xyz))]

    monkeypatch.setattr(interactions, "interactions_of_frames", no_interactions)
    monkeypatch.setattr(restraints, "stable_structure_fractions", lambda topology, xyz, frame_atoms: {})
    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                 "restraints": {"interactions": {"discard_ns": 0.0}}})
    trajectories = Trajectories(frames=frames, boxes=boxes, frame_atoms=np.arange(n_atoms),
                                trajectory=np.zeros(n, dtype=int), time_ps=10.0 * (1 + np.arange(n)),
                                backends=["openmm"], timesteps_fs=[4.0], notes=[])
    prep = {"topology": top, "ligand": [int(i) for i in ligand]}
    eq = {"positions": frames[-1], "box": boxes[-1]}
    with pytest.raises(WorkflowError, match="Nothing was sampled"):
        workflow.interaction_restraint(cfg, prep, mol, eq, trajectories, 298.15)
    assert seen["nearest"].max() < 0.31


# ---- review of eb6f89f (Codex): the image the force measures in, and the nearest image of a skewed cell ----

def test_the_nearest_image_of_a_skewed_cell_is_found_whatever_the_length_of_the_displacement():
    """OpenMM's formula for the minimum image is the nearest image up to half the smallest height of the cell
    (1.41 nm in the rhombic dodecahedron of 4 nm): of (0, 0, 1.5) nm it makes an image 3.125 nm long. The
    distance method measured with it and gave such a contact to a residue 1.6 nm away."""
    from cufep.alchemy.lrc import minimum_image, nearest_image

    box = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 2.0 * np.sqrt(2.0)]])
    delta = np.array([[0.0, 0.0, 1.5], [1.6, 0.0, 0.0], [0.0, 0.0, 1.3]])
    assert np.linalg.norm(minimum_image(delta, box), axis=1).tolist() == pytest.approx([3.125, 1.6, 1.3], abs=1e-3)
    assert np.linalg.norm(nearest_image(delta, box), axis=1).tolist() == pytest.approx([1.5, 1.6, 1.3])
    # against every lattice vector within two cells, for displacements many cells long, in both cells
    rng = np.random.default_rng(11)
    cells = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)], dtype=float)
    for cell in (box, CUBE, DODECAHEDRON):
        far = rng.uniform(-12.0, 12.0, size=(200, 3))
        reduced = minimum_image(far, cell)
        shortest = np.linalg.norm(reduced[:, None, :] + (cells @ cell)[None], axis=2).min(axis=1)
        assert np.allclose(np.linalg.norm(nearest_image(far, cell), axis=1), shortest)


def test_the_reference_distance_of_a_restraint_is_the_one_its_force_measures():
    """Review of eb6f89f (P1). H0 at the centre of the three host anchors, G0 2.1 nm from it and the centre of
    the ligand 1.8 nm, in a cell of 4 nm: the ligand was taken in the image nearest to the centre of the host
    anchors and r was 2.103 nm, where the force of the restraint measures 1.903 nm across the boundary; the
    restraint had an energy of 83 kJ/mol at its own reference (k_r 10 kcal/mol/A^2). The ligand is taken in the
    image in which G0 is nearest to H0."""
    from cufep.alchemy.restraints import measure_anchors

    box = np.eye(3) * 4.0
    host = np.array([[0.0, -0.1, -0.05], [0.0, 0.05, -0.05], [0.0, 0.05, 0.10]])            # their centre is the origin
    guest = np.array([[2.1, 0.0, 0.0], [2.1, 0.15, 0.0], [1.2, 0.15, 0.15]])                # their centre: x = 1.8
    positions = np.concatenate([host, guest])
    assert np.allclose(positions[:3].mean(axis=0), 0.0) and positions[3:].mean(axis=0)[0] == pytest.approx(1.8)
    measured = measure_anchors((0, 1, 2), (3, 4, 5), [3, 4, 5], positions[None], np.arange(6), box[None], 0.3)
    assert measured.values[0, 0] == pytest.approx(np.linalg.norm([1.9, 0.1, 0.05]), abs=1e-9)   # 1.903 nm
    restraint = BoreschRestraint((0, 1, 2), (3, 4, 5), *[float(x) for x in measured.values[0]], K_R, K_A, K_A,
                                 K_A, K_A, K_A)
    assert restraint_energy(restraint, positions, box) == pytest.approx(0.0, abs=1e-6)
    # in every image the ligand is written in
    for name, shift in lattice_shifts(box).items():
        moved = positions.copy()
        moved[3:] += shift
        again = measure_anchors((0, 1, 2), (3, 4, 5), [3, 4, 5], moved[None], np.arange(6), box[None], 0.3)
        assert np.allclose(again.values, measured.values, atol=1e-9), name


# ---- review of 877ab88 (Codex): the six coordinates as the force of the restraint measures them ------------

def test_the_six_coordinates_of_a_restraint_are_those_its_force_measures_in_any_cell():
    """Review of 877ab88 (P1). The force of the restraint is periodic: OpenMM takes every displacement between
    two of its particles by its formula for the minimum image. (1) In the rhombic dodecahedron of 4 nm that is
    2.935 nm for H0 to G0 of (0.1, 0.2, 1.5) nm, where the nearest image, which the reference was taken from,
    is 1.517 nm: thousands of kJ/mol at its own reference. (2) Given host anchors need not be bonded: H0 to H1
    of 2.5 nm in a cube of 4 nm is -1.5 nm for the force, and theta_A is 120 degrees where the written
    coordinates give 60: 184 kJ/mol. The anchors are laid out as the force sees them, each from the one before."""
    from cufep.alchemy.restraints import measure_anchors

    skewed = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 2.0 * np.sqrt(2.0)]])
    h0 = np.array([1.0, 1.0, 0.2])
    g0 = h0 + [0.1, 0.2, 1.5]             # no component on the edge of a cell, where the formula has a tie
    far = np.array([h0, h0 + [0.15, 0.0, 0.0], h0 + [0.15, 0.15, 0.0], g0, g0 + [0.0, 0.15, 0.0],
                    g0 + [0.15, 0.15, 0.15]])
    s3 = 0.3 * np.sqrt(3.0)
    apart = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [2.5, 0.0, 0.15], [0.3, s3, 0.0], [0.3, s3, 0.15],
                      [0.45, s3, 0.15]])
    as_the_force = float(np.linalg.norm([0.1 - 2.0, 0.2 - 2.0, 1.5 - 2.0 * np.sqrt(2.0)]))
    assert as_the_force == pytest.approx(2.935, abs=1e-3) and np.linalg.norm([0.1, 0.2, 1.5]) < 1.52
    for positions, box, r, theta_a in ((far, skewed, as_the_force, None), (apart, CUBE * 0.8, 0.6, 120.0)):
        measured = measure_anchors((0, 1, 2), (3, 4, 5), [3, 4, 5], positions[None], np.arange(6), box[None], None)
        assert measured.problem is None
        assert measured.values[0, 0] == pytest.approx(r, abs=1e-6)
        if theta_a is not None:
            assert np.degrees(measured.values[0, 1]) == pytest.approx(theta_a, abs=1e-6)
        restraint = BoreschRestraint((0, 1, 2), (3, 4, 5), *[float(x) for x in measured.values[0]], K_R, K_A, K_A,
                                     K_A, K_A, K_A)
        assert restraint_energy(restraint, positions, box) == pytest.approx(0.0, abs=1e-5)


def test_the_ligand_is_next_to_its_nearest_protein_atom_whatever_atom_that_is():
    """Review of 877ab88 (P1). A cell of 8 nm, one protein that is not wrapped: an acceptor of a side chain at
    x = 7.82 nm whose C-alpha is at 7.35, a C-alpha of the far surface at 0.65, and the ligand at 8.10, in a
    hydrogen bond of 0.28 nm with the acceptor. Among the C-alpha atoms alone the nearest is the one of the far
    surface across the boundary (0.55 nm): the ligand was moved to 0.10 nm and ProLIF saw the acceptor 7.72 nm
    away. The choice from interactions gives every atom of the protein."""
    from cufep.alchemy.restraints import ligand_next_to_protein

    box = np.eye(3) * 8.0
    positions = np.array([[7.35, 0.0, 0.0], [7.82, 0.0, 0.0], [0.65, 0.0, 0.0], [8.10, 0.0, 0.0]])
    ligand = np.array([3])
    by_c_alpha = ligand_next_to_protein(positions, ligand, np.array([0, 2]), box)
    assert by_c_alpha[3, 0] == pytest.approx(0.10) and abs(by_c_alpha[3, 0] - positions[1, 0]) == pytest.approx(7.72)
    by_every_atom = ligand_next_to_protein(positions, ligand, np.array([0, 1, 2]), box)
    assert np.allclose(by_every_atom, positions) and by_every_atom[3, 0] - positions[1, 0] == pytest.approx(0.28)
