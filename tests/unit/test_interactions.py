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


def test_a_membrane_protein_with_charged_histidines_and_an_ion_is_read_as_its_force_field_has_it(repo):
    """The A2A example: three peptide segments, two histidines with both hydrogens (HIS 264 and 278,
    +1 each) and a sodium ion. Residue by residue the formal charge of the protein that ProLIF gets
    is the charge of ff14SB; the ion is no residue of the protein and holds no anchors. The input
    pose of compound 4e has the contacts of the crystal structure (PDB 5OLZ): hydrogen bonds with
    ASN 253 in both directions and stacking with PHE 168."""
    import openmm.app as app

    from cufep.alchemy.restraints import backbone_hosts
    from cufep.analysis.interactions import InteractionFinder, interactions_of_frames
    from cufep.prep.ligand import load_ligands
    from cufep.prep.protein import prepare_protein
    from cufep.prep.solvate import combine, ligand_atom_indices

    protein = prepare_protein(repo / "examples/membrane_a2a/protein.pdb")
    ligand = load_ligands(repo / "examples/membrane_a2a/ligand.sdf")[0]
    modeller = combine((protein.topology, protein.positions), ligand)
    topology = modeller.topology
    pos = np.asarray(modeller.positions.value_in_unit(u.nanometer))
    ligand_atoms, rdmol = list(ligand_atom_indices(topology)), ligand.to_rdkit()
    names = {r.index: f"{r.name}{r.id}" for r in topology.residues()}
    finder = InteractionFinder(topology, ligand_atoms, rdmol)
    residue_of = {a.index: a.residue.index for a in topology.atoms()}
    formal: dict[int, int] = {}
    for atom, index in zip(finder.protein.GetAtoms(), finder.protein_atoms):
        formal[residue_of[int(index)]] = formal.get(residue_of[int(index)], 0) + atom.GetFormalCharge()
    system = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml").createSystem(protein.topology)
    nonbonded = next(f for f in system.getForces() if f.__class__.__name__ == "NonbondedForce")
    of_force_field: dict[int, float] = {}
    for atom in protein.topology.atoms():
        q = nonbonded.getParticleParameters(atom.index)[0].value_in_unit(u.elementary_charge)
        of_force_field[atom.residue.index] = of_force_field.get(atom.residue.index, 0.0) + q
    ion, = [r for r, name in names.items() if name.startswith("NA")]
    assert {r: int(round(q)) for r, q in of_force_field.items() if r != ion} == formal
    assert {names[r] for r, q in formal.items() if q == 1 and names[r].startswith("HIS")} == {"HIS264", "HIS278"}
    assert ion not in formal and ion not in backbone_hosts(topology)
    frame, = interactions_of_frames(topology, pos[None], np.arange(len(pos)), ligand_atoms, rdmol)
    found = {(names[residue], kind) for _atom, residue, kind in frame}
    assert {("ASN253", "HBDonor"), ("ASN253", "HBAcceptor"), ("PHE168", "FaceToFace")} <= found


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


# --------------------------------------------------------------------------- the restraint stage of a leg
BOX = 12.0 * np.eye(3)


def leg_inputs(tyk2, tmp_path, **interactions):
    """What the restraint stage of a complex leg gets, for TYK2 with ejm_31 and short trajectories:
    (configuration, job, leg directory, prepared complex, end of the pre-equilibration)."""
    from cufep.config.schema import Config
    from cufep.workflow import LegJob

    topology, pos, ligand, _rdmol, _protein = tyk2
    settings = dict(trajectories=2, length_ns=0.2, discard_ns=0.1, frame_interval_ps=10.0)
    settings.update(interactions)
    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                 "restraints": {"selection": "interactions", "interactions": settings}})
    leg_dir = tmp_path / "ejm_31" / "repeat_0" / "complex"
    leg_dir.mkdir(parents=True)
    job = LegJob(ligand="ejm_31", leg="complex", repeat=0, lig_dir=str(tmp_path / "ejm_31"), devices=[], config={},
                 production_ns=5.0, inline=True)
    n = len(pos)
    prep = {"topology": topology, "positions": pos, "box": BOX, "ligand": ligand, "solute": list(range(n)),
            "system_xml": "<not read: the trajectories of these tests are made by hand/>"}
    rng = np.random.default_rng(1)
    eq = {"positions": pos, "velocities": np.zeros_like(pos), "box": BOX, "timestep_fs": 4.0,
          "frames": (pos[None] + rng.normal(scale=0.01, size=(6, n, 3))).astype(np.float32),
          "frame_boxes": np.repeat(BOX[None], 6, axis=0), "frame_atoms": np.arange(n)}
    return cfg, job, leg_dir, prep, eq


def trajectories_by_hand(monkeypatch, pos, stop_at=None):
    """In the place of the MD: every trajectory is the input pose with thermal noise, written as the MD
    writes it. Returns the list of the trajectories that were made; ``stop_at`` fails at that one."""
    import json
    from pathlib import Path

    from cufep.files import publish_npz
    from cufep.sampling import restraint_md

    made = []

    def run(tasks, devices, log=print, inline=False):
        summaries = []
        for task in tasks:
            if task.index == stop_at:
                raise RuntimeError("the machine stopped")
            n = 20
            rng = np.random.default_rng(100 + task.index)
            frames = (pos[None, task.frame_atoms] + rng.normal(scale=0.01, size=(n, len(task.frame_atoms), 3)))
            Path(task.path).parent.mkdir(parents=True, exist_ok=True)
            publish_npz(task.path, frames=frames.astype(np.float32), frame_boxes=np.repeat(BOX[None], n, axis=0),
                        frame_atoms=np.array(task.frame_atoms), timestep_fs=task.timestep_fs,
                        time_ps=restraint_md.frame_times_ps(n, task.settings.production_ns,
                                                            task.settings.frame_interval_ps, task.timestep_fs),
                        simulated_ns=task.settings.production_ns, backend=task.settings.md_backend,
                        notes=json.dumps(list(task.notes)), made_with=json.dumps(task.made_with))
            made.append(task)
            summaries.append({"index": task.index, "backend": task.settings.md_backend,
                              "timestep_fs": task.timestep_fs, "frames": n, "notes": []})
        return summaries

    monkeypatch.setattr(restraint_md, "run_trajectories", run)
    return made


def test_the_restraint_of_a_leg_is_chosen_from_the_interactions_of_its_trajectories(tyk2, tmp_path, monkeypatch):
    """Two trajectories of 20 frames, of which the 10 later than 0.1 ns are analysed: the hinge
    hydrogen bonds are kept (8 of 20 frames needed), the host anchors are the backbone of VAL 981,
    the guest anchors three bonded heavy atoms of the ligand, and the reference values are the
    centres of the fullest bins over the analysed frames. The files of the leg say what the
    restraint rests on."""
    import json

    from cufep.alchemy.restraints import measure_anchors, reference_values_from_histograms
    from cufep.workflow import restraint_choice, restraint_from_interactions

    topology, pos, ligand, rdmol, _protein = tyk2
    cfg, job, leg_dir, prep, eq = leg_inputs(tyk2, tmp_path)
    made = trajectories_by_hand(monkeypatch, pos)
    lines = []
    restraint, info, _seconds = restraint_from_interactions(cfg, job, leg_dir, prep, rdmol, eq, False, 7, len(pos),
                                                            lines.append)
    atoms = list(topology.atoms())
    assert [task.index for task in made] == [0, 1]
    assert {atoms[i].residue.index for i in restraint.host_atoms} == {hinge(topology)}
    assert sorted(atoms[i].name for i in restraint.host_atoms) == ["C", "CA", "N"]
    local = [ligand.index(i) for i in restraint.guest_atoms]
    assert all(rdmol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in local)
    assert rdmol.GetBondBetweenAtoms(local[0], local[1]) and rdmol.GetBondBetweenAtoms(local[1], local[2])
    assert info["selection"] == "interactions" and info["method"] == "interactions"
    assert info["interaction"]["residue"] == "VAL981" and info["interaction"]["type"] in ("HBAcceptor", "HBDonor")
    assert info["interaction"]["of_frames"] == 20 and info["interaction"]["frames"] >= 8
    assert info["anchors"]["host"] == [f"VAL981:{atoms[i].name}" for i in restraint.host_atoms]
    assert info["md"] == {"trajectories": 2, "frames_analysed": 20, "backends": ["cupotential"] * 2,
                          "timesteps_fs": [4.0, 4.0]}
    assert info["ligand_rmsd_reference"] == "input pose" and 0.0 < info["ligand_rmsd_A"] < 1.0
    assert set(info["versions"]) == {"prolif", "MDAnalysis", "rdkit"}
    assert restraint_choice(info)["anchors"] == info["anchors"] and restraint_choice({"selection": "auto"}) is None
    assert json.loads(json.dumps(info)) == info                 # as restraint.json holds it
    # the record ties the restraint to the pre-equilibration its trajectories started from, as each trajectory
    # is tied to it: the restraint of another repeat, or of a pre-equilibration that was made again, has the
    # same settings and is not this leg's (review of eb6f89f)
    from cufep.sampling.restraint_md import start_digest
    from cufep.workflow import WorkflowError, check_restraint_start

    assert info["start"] == start_digest(eq["positions"], eq["velocities"], eq["box"])
    check_restraint_start({"info": info}, leg_dir / "restraint.json", eq)
    check_restraint_start({"info": {"selection": "auto"}}, leg_dir / "restraint.json", eq)
    with pytest.raises(WorkflowError, match="chosen from MD that started from another configuration"):
        check_restraint_start({"info": info}, leg_dir / "restraint.json", dict(eq, positions=eq["positions"] + 0.001))
    # where the alchemical run starts, in the units the record names (the restraint beside it is in radians)
    assert info["start_values_units"] == {"r": "nm", "angles": "degrees"}
    assert info["start_values"]["r"] == pytest.approx(restraint.r0, abs=0.1) and info["start_values"]["r"] < 1.0
    assert np.radians(info["start_values"]["theta_A"]) == pytest.approx(restraint.theta_a0, abs=0.5)
    # the reference values: the histograms of the analysed frames (those later than 0.1 ns of both trajectories)
    from cufep.sampling.restraint_md import read_trajectories

    analysed = read_trajectories(leg_dir, 2).after(0.1)
    assert analysed.frames.shape[0] == 20 and analysed.time_ps.min() > 100.0
    measured = measure_anchors(restraint.host_atoms, restraint.guest_atoms, ligand, analysed.frames,
                               analysed.frame_atoms, analysed.boxes, 0.3)
    expected = reference_values_from_histograms(measured.values)
    assert [restraint.r0, restraint.theta_a0, restraint.theta_b0, restraint.phi_a0, restraint.phi_b0,
            restraint.phi_c0] == pytest.approx(expected.tolist())
    assert np.radians(45) <= restraint.theta_a0 <= np.radians(135) and (restraint.k_r, restraint.k_theta_a) == \
        (10.0 * 4.184 * 100.0, 80.0 * 4.184)
    # the files of the leg
    report = json.loads((leg_dir / "interactions.json").read_text())
    assert report["frames_analysed"] == 20 and report["frames_needed"] == 8 and report["frames_needed_distance"] == 3
    assert report["method"] == "interactions" and report["chosen"]["host"] == info["anchors"]["host"]
    kept = [row for row in report["interactions"] if row["kept"]]
    assert all(row["frames"] >= 8 for row in kept) and "VAL981" in {row["residue_name"] for row in kept}
    assert all(row["frames"] < 8 for row in report["interactions"] if not row["kept"])
    assert all(sum(row["frames_of_trajectory"].values()) == row["frames"] for row in report["interactions"])
    assert len(report["coordinates"]["r"]) == 20 and report["made_with"]["selection"] == "interactions"
    assert {"residue": hinge(topology), "residue_name": "VAL981"} in [
        {key: row[key] for key in ("residue", "residue_name")} for row in report["stable_structure_fraction"]]
    # ... of every residue the report names, not of the candidates alone
    assert {row["residue"] for row in report["residues"]} | {row["residue"] for row in report["interactions"]} \
        <= {row["residue"] for row in report["stable_structure_fraction"]}
    table = (leg_dir / "interactions.csv").read_text().splitlines()
    assert table[0] == "ligand_atom,residue,interaction,frames,fraction,kept,frames_of_trajectory_0," \
                       "frames_of_trajectory_1"
    assert len(table) == 1 + len(report["interactions"]) and any(",VAL981,HB" in line for line in table[1:])
    assert any(line.startswith("restraint (interactions: HB") and "host VAL981:" in line for line in lines)
    assert all(row["in_frames"] == sorted(row["in_frames"]) and len(row["in_frames"]) == row["frames"]
               for row in report["interactions"])

    # the figure of the leg is drawn from that file; without the file there is none, and none of an earlier state
    from cufep.analysis.plots import plot_interactions

    # ... and is of the restraint the leg has. Stopped between interactions.json and restraint.json, a leg can go
    # on with given anchors: the analysis that is left over is of a restraint that was never used (review of
    # eb6f89f: its figure was drawn as the one of the result)
    assert plot_interactions(leg_dir) is None                    # no restraint.json yet
    stored = {"restraint": restraint.to_dict(), "info": info}
    (leg_dir / "restraint.json").write_text(json.dumps(stored))
    figure = plot_interactions(leg_dir)
    assert figure == leg_dir / "interactions.png" and figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    given = dict(stored["restraint"], guest_atoms=[ligand[0], ligand[1], ligand[2]])
    (leg_dir / "restraint.json").write_text(json.dumps({"restraint": given, "info": {"selection": "manual"}}))
    assert plot_interactions(leg_dir) is None and not figure.exists()
    # the same anchors chosen from another start are another analysis (review of 877ab88: two runs of the TYK2
    # example chose the same six atoms, with 672 and 717 frames of the hydrogen bond and other reference values)
    (leg_dir / "restraint.json").write_text(json.dumps({"restraint": stored["restraint"],
                                                        "info": dict(info, start="of another start")}))
    assert plot_interactions(leg_dir) is None and report["start"] == info["start"]
    (leg_dir / "restraint.json").write_text(json.dumps(stored))
    assert plot_interactions(leg_dir) == figure
    # an analysis written before it recorded its start: the anchors and the reference values tell that the pair
    # belongs together, and its figure stays (the third review by Codex: it was deleted)
    without_start = {key: value for key, value in report.items() if key != "start"}
    (leg_dir / "interactions.json").write_text(json.dumps(without_start))
    assert plot_interactions(leg_dir) == figure and figure.exists()
    (leg_dir / "interactions.json").write_text(json.dumps(report))
    # ... and every reading of the samples of a leg asks whether its restraint is of its pre-equilibration
    # (review of 877ab88: cufep analyze gave the release of a restraint of another repeat)
    from cufep.workflow import restraint_release_of

    np.savez(leg_dir / "equil.npz", positions=eq["positions"], velocities=eq["velocities"], box=eq["box"])
    states = {"states": [{"lambda_restraints": 0.0}, {"lambda_restraints": 1.0}]}
    assert restraint_release_of(leg_dir, states, 298.15)[0] != 0.0
    np.savez(leg_dir / "equil.npz", positions=eq["positions"] + 0.001, velocities=eq["velocities"], box=eq["box"])
    with pytest.raises(WorkflowError, match="chosen from MD that started from another configuration"):
        restraint_release_of(leg_dir, states, 298.15)
    (leg_dir / "equil.npz").unlink()
    (leg_dir / "interactions.json").rename(leg_dir / "interactions.kept")
    assert plot_interactions(leg_dir) is None and not figure.exists()


def test_residues_that_would_share_a_name_are_named_with_the_chain_of_the_input():
    """Review of f5a77ce (P3): the results named a residue by its name and number ("VAL981"), which the two
    chains of a dimer share, and the secondary structure of the residues was recorded under that name, so that
    one of two residues was lost. Review of eb6f89f (P2): the chain that was then written was the one of the
    prepared topology, which has a chain for every piece between chain breaks, named anew in their order: the
    two pieces of the one chain of the A2A example read "ALA59.A" and "HIS278.B", and with chains B, A in the
    input the first "VAL981" read "VAL981.A". A residue is named as the input names it, with its chain only
    where two residues would share the name, and with its insertion code."""
    import openmm.app as app

    from cufep.workflow import atom_label, residue_names

    def topology(*chains):
        """One prepared chain per entry: its id and the numbers of its valines."""
        top = app.Topology()
        for chain_id, numbers in chains:
            chain = top.addChain(chain_id)
            for number in numbers:
                top.addAtom("N", app.element.nitrogen, top.addResidue("VAL", chain, id=str(number)))
            top.addAtom("N", app.element.nitrogen, top.addResidue("ASP", chain, id="898", insertionCode=chain_id))
        top.addAtom("C1", app.element.carbon, top.addResidue("LIG", top.addChain("L"), id="1"))
        return top

    dimer = topology(("A", [981]), ("B", [981]))
    names = residue_names(dimer, {"A": "A", "B": "B"})
    assert [names[i] for i in range(5)] == ["VAL981.A", "ASP898A", "VAL981.B", "ASP898B", "LIG1"]
    assert atom_label(list(dimer.atoms()), 2, names) == "VAL981.B:N"
    # the chains of the input in the order B, A: the first piece is the input's chain B
    assert list(residue_names(dimer, {"A": "B", "B": "A"}).values())[::2][:2] == ["VAL981.B", "VAL981.A"]
    # two pieces of one chain of the input have no residue in common: no chain is written
    pieces = topology(("A", [59]), ("B", [278]))
    assert [residue_names(pieces, {"A": "A", "B": "A"})[i] for i in (0, 2)] == ["VAL59", "VAL278"]
    # chains that the input does not tell apart: also the number of the piece, behind a sign no chain id has
    # (review of 877ab88: behind a dot, the pieces X, X and 1 read VAL981.1, VAL981.2 and VAL981.1)
    assert list(residue_names(dimer, {"A": "", "B": ""}).values())[::2][:2] == ["VAL981#1", "VAL981#2"]
    three = topology(("A", [981]), ("B", [981]), ("C", [981]))
    assert list(residue_names(three, {"A": "X", "B": "X", "C": "1"}).values())[::2][:3] == [
        "VAL981.X#1", "VAL981.X#2", "VAL981.1"]
    assert list(residue_names(dimer).values())[::2][:2] == ["VAL981.A", "VAL981.B"]


def test_the_trajectories_take_the_settings_of_the_yaml_and_the_end_of_the_pre_equilibration(tyk2, tmp_path,
                                                                                             monkeypatch):
    """Each trajectory is a plain MD of the leg: the backend with its fallback and precision, the
    temperature, pressure, friction and barostat frequency are those of the configuration, the
    time step is the one the pre-equilibration ended with, and it starts from where that ended.
    Its record holds all of that, with the seed of the leg."""
    from cufep.config.schema import Config
    from cufep.sampling.restraint_md import start_digest
    from cufep.workflow import restraint_md_made_with, restraint_trajectories

    _topology, pos, _ligand, _rdmol, _protein = tyk2
    cfg, job, leg_dir, prep, eq = leg_inputs(tyk2, tmp_path)
    cfg = Config.model_validate({**cfg.model_dump(mode="json"),
                                 "thermodynamics": {"temperature_K": 310.0, "pressure_bar": 2.0},
                                 "dynamics": {"friction_per_ps": 0.5, "barostat_frequency": 50, "timestep_fs": 4.0},
                                 "backend": {"name": "cupotential", "fallback": "none", "precision": "single"}})
    eq = dict(eq, timestep_fs=2.0)                                # the pre-equilibration had lowered it
    made = trajectories_by_hand(monkeypatch, pos)
    job.devices = [None]                                          # a run without GPUs
    lines = []
    restraint_trajectories(cfg, job, leg_dir, prep, eq, True, 7, lines.append)
    # the log names the GPUs there are (review of f5a77ce: "(GPU None)")
    assert lines[0] == "MD for the restraint: 2 of 2 trajectories of 0.2 ns to run on 1 worker(s)"
    for k, task in enumerate(made):
        s = task.settings
        assert (s.md_backend, s.fallback, s.precision, s.platform) == ("cupotential", "none", "single", "CUDA")
        assert (s.temperature_K, s.pressure_bar, s.friction_per_ps, s.barostat_frequency) == (310.0, 2.0, 0.5, 50)
        assert (s.barostat, s.bilayer_report, s.production_ns, s.frame_interval_ps, s.seed) == \
            ("membrane", False, 0.2, 10.0, 7)
        assert task.timestep_fs == 2.0 and task.index == k and task.frame_atoms == prep["solute"]
        assert np.array_equal(task.positions, eq["positions"]) and np.array_equal(task.box, eq["box"])
        assert task.made_with == dict(restraint_md_made_with(cfg, 7, True, eq), trajectory=k)
    record = made[0].made_with
    assert record["start"] == start_digest(eq["positions"], eq["velocities"], eq["box"])
    assert (record["backend"], record["fallback"], record["precision"], record["barostat"], record["seed"],
            record["timestep_fs"], record["length_ns"]) == ("cupotential", "none", "single", "membrane", 7, 2.0, 0.2)
    assert "discard_ns" not in record and "trajectories" not in record


def test_an_interrupted_restraint_stage_goes_on_with_what_it_has_and_gives_the_same_restraint(tyk2, tmp_path,
                                                                                              monkeypatch):
    """Stopped while the second trajectory ran, and stopped when the numbers of the analysis were
    written: the leg has no restraint, keeps the trajectories that were complete without making
    them again, and when it is run again it gets the restraint of a leg that was never stopped."""
    from cufep import workflow
    from cufep.sampling.restraint_md import trajectory_file

    _topology, pos, _ligand, rdmol, _protein = tyk2
    cfg, job, whole_dir, prep, eq = leg_inputs(tyk2, tmp_path / "whole")
    trajectories_by_hand(monkeypatch, pos)
    whole, whole_info, _s = workflow.restraint_from_interactions(cfg, job, whole_dir, prep, rdmol, eq, False, 7,
                                                                 len(pos), lambda line: None)

    cfg, job, leg_dir, prep, eq = leg_inputs(tyk2, tmp_path / "stopped")
    trajectories_by_hand(monkeypatch, pos, stop_at=1)
    with pytest.raises(RuntimeError, match="the machine stopped"):
        workflow.restraint_from_interactions(cfg, job, leg_dir, prep, rdmol, eq, False, 7, len(pos), lambda line: None)
    first = trajectory_file(leg_dir, 0).read_bytes()
    assert not trajectory_file(leg_dir, 1).exists() and not (leg_dir / "interactions.json").exists()

    made = trajectories_by_hand(monkeypatch, pos)
    publish = workflow.publish_text

    def stopped(path, text):
        if path.name == "interactions.json":
            raise OSError(28, "No space left on device")
        return publish(path, text)

    monkeypatch.setattr(workflow, "publish_text", stopped)
    with pytest.raises(OSError):
        workflow.restraint_from_interactions(cfg, job, leg_dir, prep, rdmol, eq, False, 7, len(pos), lambda line: None)
    assert [task.index for task in made] == [1] and trajectory_file(leg_dir, 0).read_bytes() == first
    assert sorted(p.name for p in leg_dir.iterdir()) == ["restraint_md"]                # nothing half written

    monkeypatch.setattr(workflow, "publish_text", publish)
    again = trajectories_by_hand(monkeypatch, pos)
    restraint, info, _s = workflow.restraint_from_interactions(cfg, job, leg_dir, prep, rdmol, eq, False, 7,
                                                               len(pos), lambda line: None)
    assert again == [] and restraint.to_dict() == whole.to_dict() and info == whole_info
    assert (leg_dir / "interactions.json").read_text() == (whole_dir / "interactions.json").read_text()


def test_trajectories_of_other_settings_or_of_another_start_are_not_used(tyk2, tmp_path, monkeypatch):
    """A leg keeps its trajectories for the run that made them: with another length, another
    temperature, another backend or another end of the pre-equilibration they are refused, by the
    job and by the check before any job, and stay as they are. Leaving out more or less of their
    first part, or asking for more trajectories, uses them."""
    import json

    from cufep import workflow
    from cufep.config.schema import Config
    from cufep.sampling.restraint_md import trajectory_file

    _topology, pos, _ligand, rdmol, _protein = tyk2
    cfg, job, leg_dir, prep, eq = leg_inputs(tyk2, tmp_path)
    trajectories_by_hand(monkeypatch, pos)
    workflow.restraint_trajectories(cfg, job, leg_dir, prep, eq, False, 7, lambda line: None)
    stored = {k: trajectory_file(leg_dir, k).read_bytes() for k in (0, 1)}

    def other(**sections):
        data = cfg.model_dump(mode="json")
        for key, value in sections.items():
            data[key] = {**data[key], **value}
        return Config.model_validate(data)

    def with_interactions(**settings):
        return other(restraints={"interactions": {**cfg.restraints.interactions.model_dump(mode="json"), **settings}})

    for changed, why in ((with_interactions(length_ns=0.3), "length_ns: 0.2 -> 0.3"),
                         (other(thermodynamics={"temperature_K": 310.0}), "temperature_K: 298.15 -> 310.0"),
                         (other(backend={"name": "openmm"}), "backend: cupotential -> openmm")):
        with pytest.raises(workflow.WorkflowError, match=f"MD 0 for the restraint of this leg .*{why}"):
            workflow.restraint_trajectories(changed, job, leg_dir, prep, eq, False, 7, lambda line: None)
    moved = dict(eq, positions=eq["positions"] + 0.001)
    with pytest.raises(workflow.WorkflowError, match="start: "):
        workflow.restraint_trajectories(cfg, job, leg_dir, prep, moved, False, 7, lambda line: None)
    with pytest.raises(workflow.WorkflowError, match="seed: 7 -> 8"):
        workflow.restraint_trajectories(cfg, job, leg_dir, prep, eq, False, 8, lambda line: None)
    assert {k: trajectory_file(leg_dir, k).read_bytes() for k in (0, 1)} == stored

    made = trajectories_by_hand(monkeypatch, pos)
    read = workflow.restraint_trajectories(with_interactions(discard_ns=0.05), job, leg_dir, prep, eq, False, 7,
                                           lambda line: None)
    assert made == [] and read.frames.shape[0] == 40
    more = workflow.restraint_trajectories(with_interactions(trajectories=3), job, leg_dir, prep, eq, False, 7,
                                           lambda line: None)
    assert [task.index for task in made] == [2] and more.frames.shape[0] == 60

    # a trajectory that holds less MD than its record says is not used either
    with np.load(trajectory_file(leg_dir, 0)) as data:
        short = {key: data[key] for key in data.files}
    short["simulated_ns"] = 0.1
    np.savez(trajectory_file(leg_dir, 0), **short)
    with pytest.raises(workflow.WorkflowError, match="holds 100 ps where 200 ps are asked for"):
        workflow.restraint_trajectories(cfg, job, leg_dir, prep, eq, False, 7, lambda line: None)
    assert json.loads(str(short["made_with"]))["trajectory"] == 0


def test_the_job_asks_for_gpus_only_for_trajectories_it_must_run_and_goes_on_with_the_first_it_got(
        tyk2, tmp_path, monkeypatch):
    """The scheduler grants the trajectories their GPUs and leaves the job the first of them when it
    is serial again: the job goes on with that one, whatever GPU it had before. A leg that has its
    trajectories, or runs inline, asks for nothing."""
    from cufep.workflow import restraint_trajectories

    class Control:
        def __init__(self):
            self.asked, self.serial_calls = [], 0

        def request_devices(self, replicas, work, timeout=300.0, pinned=False, what=None):
            self.asked.append((replicas, pinned))
            self.work, self.what = work, what
            return [3, 3, 5]

        def serial(self):
            self.serial_calls += 1

    _topology, pos, _ligand, _rdmol, _protein = tyk2
    cfg, job, leg_dir, prep, eq = leg_inputs(tyk2, tmp_path, trajectories=3)
    job.devices, job.inline, job.control, job.oversubscription = [6], False, Control(), 2
    used = []
    made = trajectories_by_hand(monkeypatch, pos)
    from cufep.sampling import restraint_md

    run = restraint_md.run_trajectories


    def on_granted(tasks, devices, log=print, inline=False):
        used.append(list(devices))
        return run(tasks, devices)

    monkeypatch.setattr(restraint_md, "run_trajectories", on_granted)
    restraint_trajectories(cfg, job, leg_dir, prep, eq, False, 7, lambda line: None)
    assert job.control.asked == [(3, True)] and used == [[3, 3, 5]] and len(made) == 3
    assert job.devices == [3] and job.control.serial_calls == 1
    assert {task.mps_thread_percentage for task in made} == {100}          # two workers of this job on a GPU
    # the work of the three trajectories of 0.2 ns, two workers on a GPU, in the unit in which replica exchange
    # tells its iterations (review of f5a77ce: it was told in picoseconds, replica exchange tells seconds)
    from cufep.workflow import md_work

    iteration_ps = cfg.sampling.time_per_iteration_ps
    one_iteration = md_work(iteration_ps, len(pos), iteration_ps)
    assert job.control.work == pytest.approx(3 * (200.0 / iteration_ps) * one_iteration / 2)
    assert job.control.what == "3 trajectories for the restraint"
    # all trajectories are there: nothing is asked for, and the job stays where it is
    job.devices = [6]
    restraint_trajectories(cfg, job, leg_dir, prep, eq, False, 7, lambda line: None)
    assert job.control.asked == [(3, True)] and job.devices == [6] and job.control.serial_calls == 1


def test_stored_trajectories_are_checked_before_any_job_of_a_run(tyk2, tmp_path, monkeypatch):
    """Before a run writes or samples anything, the trajectories a complex leg has for a restraint
    that is still to be chosen must be ones this run would make: of its settings, and started from
    the pre-equilibration that is there. A leg that has its restraint is asked about that alone."""
    import json

    from cufep import workflow
    from cufep.config.schema import Config

    _topology, pos, _ligand, _rdmol, _protein = tyk2
    cfg, job, leg_dir, prep, eq = leg_inputs(tyk2, tmp_path)
    seed = workflow.leg_seed(cfg, 0, "complex")
    trajectories_by_hand(monkeypatch, pos)
    workflow.restraint_trajectories(cfg, job, leg_dir, prep, eq, False, seed, lambda line: None)
    np.savez(leg_dir / "equil.npz", positions=eq["positions"], velocities=eq["velocities"], box=eq["box"],
             timestep_fs=4.0, simulated_ns=5.0, simulated_ns_is_integrated=True, notes="[]",
             made_with=json.dumps(workflow.equil_made_with(cfg, "complex", False, seed)))
    lig_dir, prep_info = tmp_path / "ejm_31", {"complex_barostat": "isotropic"}
    workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, prep_info, 5.0)                 # usable

    data = cfg.model_dump(mode="json")
    data["restraints"]["interactions"]["frame_interval_ps"] = 5.0
    with pytest.raises(workflow.WorkflowError, match="MD 0 for the restraint of this leg .*frame_interval_ps"):
        workflow.check_leg_reuse(Config.model_validate(data), lig_dir, "complex", 0, prep_info, 5.0)
    # a restraint that is there is what the leg uses: its trajectories are not asked about any more
    (leg_dir / "restraint.json").write_text(json.dumps({"restraint": {}, "info": {"selection": "interactions"},
                                                        "made_with": workflow.restraint_made_with(cfg)}))
    workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, prep_info, 5.0)
    # a restraint that names the start of its MD, without the pre-equilibration it started from: the start cannot
    # be checked, and a pre-equilibration that is made again is another one. Refused before any job (the third
    # review by Codex: this was only tested with the restraint gone too, where the trajectories refuse it)
    of_a_start = {"restraint": {}, "info": {"selection": "interactions", "start": "the digest of a start"},
                  "made_with": workflow.restraint_made_with(cfg)}
    (leg_dir / "restraint.json").write_text(json.dumps(of_a_start))
    kept = (leg_dir / "equil.npz").read_bytes()
    (leg_dir / "equil.npz").unlink()
    with pytest.raises(workflow.WorkflowError, match="was chosen from MD that started from the pre-equilibration"):
        workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, prep_info, 5.0)
    with pytest.raises(workflow.WorkflowError, match="cannot be checked"):
        workflow.restraint_release_of(leg_dir, {"states": [{"lambda_restraints": 0.0}]}, 298.15)
    (leg_dir / "equil.npz").write_bytes(kept)
    (leg_dir / "restraint.json").unlink()
    (leg_dir / "equil.npz").unlink()
    with pytest.raises(workflow.WorkflowError, match="which it started from, is not there"):
        workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, prep_info, 5.0)
