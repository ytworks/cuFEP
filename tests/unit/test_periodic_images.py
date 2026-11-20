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
    assert without_boxes["ligand_rmsd_max_A"] == pytest.approx(50.0, abs=0.2) and without_boxes["warnings"]
    assert without_boxes["periodic_images_checked"] is False
    result = structural_analysis(samples)
    assert result["ligand_rmsd_max_A"] < 0.1 and result["ligand_com_drift_max_A"] < 0.1
    assert result["warnings"] == [] and result["periodic_images_checked"] is True


@pytest.mark.parametrize("box", [CUBE, DODECAHEDRON], ids=["cube", "dodecahedron"])
def test_the_pose_check_is_the_same_in_every_image(box):
    shifts = list(lattice_shifts(box).values())
    result = structural_analysis(_coupled_frames(box, shifts * 2))
    still = structural_analysis(_coupled_frames(box, [np.zeros(3)] * (2 * len(shifts))))
    assert result["ligand_rmsd_max_A"] == pytest.approx(still["ligand_rmsd_max_A"], abs=0.05)
    assert result["ligand_com_drift_max_A"] < 0.1 and result["warnings"] == []


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
    restraint at its own reference was NaN. Every angle of the anchors is checked in every frame."""
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
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    assert not np.all(np.isfinite(forces))                                      # what the reviewer saw
    problem = anchor_geometry_problem(probe, frames, *anchor_margins(RestraintSearchSettings(), 298.15))
    assert problem is not None and problem.startswith("the anchor angle G0-G1-G2")
    assert anchor_geometry_problem(probe, POSITIONS[None], *anchor_margins(RestraintSearchSettings(), 298.15)) is None
    eq = {"frame_atoms": np.arange(6), "frames": frames, "frame_boxes": np.repeat(CUBE[None], len(frames), axis=0)}
    with pytest.raises(WorkflowError, match="G0-G1-G2 comes within"):
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
    assert fluct == pytest.approx(coordinate_fluctuations(BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0).values(frames)))


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
    """Review of 037bdf1 (P2): with manual anchors only the fluctuations of the restraint were stored, so a ligand that
    left the binding site gave no pose_rmsd_warning_nm warning. The measure of the restraint search (periodic image,
    superposition on the site, symmetry) is used for them: one cell (5 nm) away is RMSD 0 and no warning; 0.6 nm away
    from the site is 0.6 nm, a warning at the level of 0.3 nm."""
    from rdkit import Chem

    from cufep.alchemy.restraints import ligand_pose_rmsd, pose_note

    top, given, frames, ligand = _site_and_ligand()
    rdmol = Chem.MolFromSmiles("CCC")
    boxes = np.repeat(box[None], len(frames), axis=0)

    def rmsd(moved_frames):
        return ligand_pose_rmsd(top, moved_frames, ligand, rdmol, given, frame_boxes_nm=boxes, input_box_nm=box)

    for name, shift in lattice_shifts(box).items():
        moved = frames.copy()
        moved[:, ligand] += shift
        assert rmsd(moved) == pytest.approx(0.0, abs=0.01), name
        assert pose_note(rmsd(moved), 0.3) is None
    away = frames.copy()
    away[:, ligand] += np.array([0.6, 0.0, 0.0]) + box[0]          # 0.6 nm from the site, one cell away
    assert rmsd(away) == pytest.approx(0.6, abs=0.01)
    assert pose_note(rmsd(away), 0.3).startswith("the ligand moved away from its input pose")


def test_the_pose_note_of_manual_anchors_is_made_with_the_level_of_the_run():
    """A stored restraint of manual anchors holds the distance from the input pose; its note is made with
    restraints.pose_rmsd_warning_nm of the run, as for anchors the search chose. One of an earlier version holds no
    distance and says so."""
    from cufep.config.schema import Config
    from cufep.workflow import restraint_notes

    def cfg(level):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                      "restraints": {"pose_rmsd_warning_nm": level}})

    stored = {"info": {"selection": "manual", "ligand_rmsd_A": 6.0, "ligand_rmsd_reference": "input pose",
                       "notes": []}}
    assert restraint_notes(stored, cfg(1.0)) == []
    assert restraint_notes(stored, cfg(0.3))[0].startswith("the ligand moved away from its input pose")
    earlier = {"info": {"selection": "manual", "fluctuations": {}}}
    assert "could not be compared with restraints.pose_rmsd_warning_nm" in restraint_notes(earlier, cfg(0.3))[0]


def _pose_with_theta_a(degrees):
    """POSITIONS with the guest moved so that the anchor angle H1-H0-G0 (theta_A) is ``degrees``; the other angles
    stay far from a straight line."""
    pose = POSITIONS.copy()
    a = np.radians(degrees)
    pose[3] = pose[0] + 0.4 * np.array([-np.sin(a), np.cos(a), 0.0])
    pose[4] = pose[3] + [-0.15, 0.0, 0.0]
    pose[5] = pose[4] + [0.0, 0.0, 0.15]
    return pose


def test_the_prepared_pose_is_held_to_the_rule_of_every_frame_only(tmp_path, monkeypatch):
    """Independent review of c691e37 (P2): before any job the prepared pose was held to both margins of the anchor
    geometry, also the one of the frame the reference values come from (30 degrees), which the run applies to the last
    frame of the pre-equilibration only. theta_A of 25 degrees in the prepared pose was refused although frames of
    38 to 45 degrees pass the rule of the run; and a run whose complex legs all have their restraint was refused too.
    The prepared pose is held to the rule of every frame (22 degrees with the default constants)."""
    from cufep import workflow
    from cufep.alchemy.restraints import RestraintSearchSettings, anchor_geometry_problem, anchor_margins
    from cufep.config.schema import Config

    margins = anchor_margins(RestraintSearchSettings(k_r=K_R, k_angle=K_A), 298.15)
    assert np.degrees(margins) == pytest.approx([22.05, 30.0], abs=0.01)
    probe = BoreschRestraint(HOST, GUEST, 0, 0, 0, 0, 0, 0)
    assert "in the frame the reference values come from" in anchor_geometry_problem(probe, _pose_with_theta_a(25.0)[None],
                                                                                   *margins)   # the rule before
    frames = np.array([_pose_with_theta_a(t) for t in np.linspace(38.0, 45.0, 8)])
    assert anchor_geometry_problem(probe, frames, *margins) is None               # what the run accepts
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                 "restraints": {"selection": "manual", "host_atoms": list(HOST), "guest_atoms": list(GUEST)}})
    monkeypatch.setattr(workflow, "check_manual_anchors", lambda *args: None)     # atoms of protein and ligand: elsewhere
    pose = {"positions": _pose_with_theta_a(25.0), "ligand": list(LIGAND), "box": CUBE}
    monkeypatch.setattr(workflow, "load_prepared", lambda prep_dir, name: pose)
    workflow.check_before_jobs(cfg, tmp_path, {(0, "complex")})                  # accepted now
    pose["positions"] = _pose_with_theta_a(10.0)                                  # within 22 degrees of a straight line
    with pytest.raises(WorkflowError, match="in the prepared pose: the anchor angle H1-H0-G0 .* comes within 22 degrees"):
        workflow.check_before_jobs(cfg, tmp_path, {(0, "complex")})
    leg = tmp_path / "repeat_0" / "complex"
    leg.mkdir(parents=True)
    (leg / "restraint.json").write_text("{}")                                     # the leg has its restraint
    workflow.check_before_jobs(cfg, tmp_path, {(0, "complex")})
    with pytest.raises(WorkflowError, match="in the prepared pose"):              # another repeat has none yet
        workflow.check_before_jobs(cfg, tmp_path, {(0, "complex"), (1, "complex")})
