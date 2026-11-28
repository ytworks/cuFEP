"""An edge from its preparation to its ddG, on the CPU with stock OpenMM (cufep.rbfe.prep, cufep.rbfe.leg,
cufep.rbfe.run): an alanine dipeptide is the protein, toluene becomes phenol next to it. The settings are made
small for the test; what is checked is the path, the windows the legs share, and that a run called again in the
same directory finds the same results."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields", "kartograf", "lomap", "cinnabar")


def _protein():
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    from conftest import _DIPEPTIDE_BONDS, _DIPEPTIDE_HEAVY

    from cufep.prep.protein import PreparedProtein

    top, pos = app.Topology(), []
    chain, previous_c = top.addChain("A"), None
    for number, res_name in enumerate(("ACE", "ALA", "NME"), start=1):
        res, local = top.addResidue(res_name, chain, str(number)), {}
        for rn, name, element, xyz in _DIPEPTIDE_HEAVY:
            if rn == res_name:
                local[name] = top.addAtom(name, app.Element.getBySymbol(element), res)
                pos.append(mm.Vec3(*(0.1 * np.array(xyz) + 1.0)))
        for a, b in _DIPEPTIDE_BONDS[res_name]:
            top.addBond(local[a], local[b])
        if previous_c is not None:
            top.addBond(previous_c, local["N"])
        previous_c = local["C"] if res_name != "NME" else None
    import random

    random.seed(20261125)                     # Modeller places the new hydrogens at random
    mod = app.Modeller(top, pos * u.nanometer)
    mod.addHydrogens(app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml"),
                     platform=mm.Platform.getPlatformByName("Reference"))
    return PreparedProtein(mod.topology, mod.positions, None, {"A": "A"})


def _ligands(tmp_path, out_dir):
    """Toluene and phenol on one ring, 0.6 nm from the peptide, with Gasteiger charges in the cache."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from rdkit.Geometry import Point3D

    from cufep.prep.ligand import load_ligands

    a = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    AllChem.EmbedMolecule(a, randomSeed=11)
    xyz = a.GetConformer().GetPositions()
    shift = np.array([16.0, 14.0, 16.0]) - xyz.mean(axis=0)
    for i, row in enumerate(xyz + shift):
        a.GetConformer().SetAtomPosition(i, row.tolist())
    ring = {**{i: i for i in range(1, 7)}, **{10 + k: 8 + k for k in range(5)}}
    b = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))
    xyz_a = a.GetConformer().GetPositions()
    AllChem.EmbedMolecule(b, randomSeed=11, coordMap={j: Point3D(*xyz_a[i]) for i, j in ring.items() if j < 7})
    rdMolAlign.AlignMol(b, a, atomMap=[(j, i) for i, j in ring.items()])
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, mol, dg in (("toluene", a, -5.0), ("phenol", b, -5.6)):
        mol.SetProp("_Name", name)
        mol.SetProp("r_exp_dg", str(dg))
        writer.write(mol)
    writer.close()
    ligands = load_ligands(tmp_path / "ligands.sdf")
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    for lig in ligands:
        mol = lig.offmol
        mol.assign_partial_charges("gasteiger")
        charges = [float(q) for q in mol.partial_charges.m_as("elementary_charge")]
        shift = sum(charges) / len(charges)
        (out_dir / "cache" / f"charges_am1bcc_{lig.digest()}.json").write_text(
            json.dumps({"charges": [q - shift for q in charges]}))
    return ligands


def _config(tmp_path, out_dir, mode: str, legs: str = "each"):
    from cufep.rbfe.config import RbfeConfig

    (tmp_path / "protein.pdb").write_text("REMARK built in the test\n")
    short = {"nvt_ns": 0.002, "npt_ns": 0.002, "md_ns": 0.004}
    return RbfeConfig.model_validate({
        "system": {"protein": str(tmp_path / "protein.pdb"), "ligands": str(tmp_path / "ligands.sdf")},
        "backend": {"name": "openmm", "platform": "CPU"},
        # small boxes with room under NPT: a cutoff of 0.6 nm and a padding of 0.8 nm (with 0.9 and 0.6 nm, boxes of
        # the precalculation fell below twice the cutoff now and then)
        "dynamics": {"cutoff_nm": 0.6},
        "solvation": {"box_shape": "cube", "padding_complex_nm": 0.8, "padding_solvent_nm": 0.8},
        "thermodynamics": {"ionic_strength_M": 0.0},
        "sampling": {"production_ns": 0.005, "equilibration_ns": 0.0025, "time_per_iteration_ps": 0.25,
                     "checkpoint_interval_ps": 1.0, "rest2": {"scale": 1.5},
                     "pre_equilibration": {"complex": short, "solvent": short},
                     # the grand canonical moves of the complex leg: the numbers of the water are given (auto is for
                     # the cutoff they were determined at), a move every 0.1 ps of few trials
                     "gcmc": {"interval_ps": 0.1, "attempts": 20, "batch_size": 10,
                              "excess_chemical_potential_kcal_per_mol": -6.09, "standard_volume_nm3": 0.030345}},
        "analysis": {"error_blocks": [2], "part_error_blocks": 2, "convergence_points": 2, "convergence_blocks": 2},
        "lambda_schedule": {"mode": mode, "windows": 3,
                            "adaptive": {"initial_windows": 3, "window_ps": 0.5, "sample_interval_ps": 0.1,
                                         "candidates": 11, "equilibration_ps": 0.0,
                                         "walk_ps": 0.2, "overlap_threshold": 0.001, "max_windows": 4,
                                         "legs": legs}},
        "resources": {"gpus": [], "cpu_threads_per_worker": 1},
        "output": {"dir": str(out_dir)}})


@pytest.fixture
def deterministic_cpu():
    """Contexts of the CPU platform with deterministic forces in this process (the legs of the test run in it): the
    same numbers in every run. Without them a first leg of three windows was connected in one run and not in
    another, and gave no free energy; the Reference platform repeats too but takes half an hour here."""
    import openmm as mm

    cpu = mm.Platform.getPlatformByName("CPU")
    before = cpu.getPropertyDefaultValue("DeterministicForces")
    cpu.setPropertyDefaultValue("DeterministicForces", "true")
    yield
    cpu.setPropertyDefaultValue("DeterministicForces", before)


@pytest.mark.parametrize("mode, legs", [("adaptive", "each"), ("adaptive", "solvent"), ("fixed", "each")])
def test_an_edge_runs_from_its_preparation_to_its_ddg_and_again_to_the_same(tmp_path, monkeypatch, mode, legs,
                                                                             deterministic_cpu):
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe

    # the start of the windows a leg does not choose, made short for the test (5000 steps of minimisation of every
    # window take minutes on one processor)
    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    from cufep.engine import leg as engine_leg
    from cufep.rbfe.prep import prepare_edge
    from cufep.rbfe.run import analyse_directory, combine_network, edge_result, write_results, write_run_files
    from cufep.sampling.gcmc import GcmcMover

    # the grand canonical moves that are made while the windows a leg does not choose are started (review of PR #21)
    moves = {"all": 0, "in_the_start_of_windows": 0}
    attempt, starts_from = GcmcMover.attempt, engine_leg.starts_from

    def counted(self, state):
        moves["all"] += 1
        return attempt(self, state)

    def start_with_count(*args, **kwargs):
        before = moves["all"]
        try:
            return starts_from(*args, **kwargs)
        finally:
            moves["in_the_start_of_windows"] += moves["all"] - before

    monkeypatch.setattr(GcmcMover, "attempt", counted)
    monkeypatch.setattr(engine_leg, "starts_from", start_with_count)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, mode, legs)
    edge_dir = out_dir / "edges" / "toluene~phenol"
    info = prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    assert info["legs"]["complex"]["core_atoms"] == info["legs"]["solvent"]["core_atoms"] >= 11
    jobs = [EdgeLegJob(edge="toluene~phenol", leg=leg, repeat=0, edge_dir=str(edge_dir), devices=[None],
                         config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True)
            for leg in ("solvent", "complex")]
    results = [run_hybrid_leg_safe(job) for job in jobs]
    assert all("error" not in r for r in results), results
    solvent, complex_ = results
    # the windows of the complex leg: its own precalculation (each), those that the precalculation of the solvent
    # leg chose (solvent), or the given ones (fixed)
    if mode == "fixed" or legs == "solvent":
        assert complex_["lambda_windows"] == solvent["lambda_windows"]
    with np.load(edge_dir / "repeat_0" / "complex" / "start_configs.npz") as start:
        made = json.loads(str(start["made_with"]))
    expected = {("adaptive", "each"): "chosen by this precalculation",
                ("adaptive", "solvent"): "those of the solvent leg", ("fixed", "each"): "given"}
    assert made["windows"] == expected[(mode, legs)]
    if mode == "fixed":
        assert len(solvent["lambda_windows"]["morph"]) == 3
    # the complex leg makes the grand canonical moves of water, the solvent leg none; its result says what its
    # samples were made with (review of PR #21: the edges of a network are compared by it)
    # (third review: with the rule of the mover for the velocities of an inserted water)
    assert complex_["water_moves"] == {"enabled": True, "excess_chemical_potential_kcal_per_mol": -6.09,
                                       "standard_volume_nm3": 0.030345,
                                       "velocities_of_an_inserted_water": "turned with it"}
    assert solvent["water_moves"] == {"enabled": False} and solvent["gcmc"] is None
    assert sum(complex_["gcmc"]["insertions_attempted"]) + sum(complex_["gcmc"]["deletions_attempted"]) > 0
    with np.load(edge_dir / "repeat_0" / "complex" / "start_configs.npz") as start:
        n_windows = len(complex_["lambda_windows"]["morph"])
        assert start["water_on"].shape[0] == n_windows                      # the waters of every start
    if mode == "fixed" or legs == "solvent":
        # review of PR #21 (P2): the windows a leg does not choose start with 0.2 ps of MD each (here), in which a
        # move is due every 0.1 ps: two per window, also where the process of the leg runs them itself (a leg
        # with workers made them, this one made none)
        assert moves["in_the_start_of_windows"] == 2 * n_windows
    else:
        assert moves["in_the_start_of_windows"] == 0
    assert all(np.isfinite(r["delta_g"]) and r["delta_g_error"] > 0 for r in results)
    h = solvent["hybrid"]
    assert h["core_atoms"] + h["unique_old_atoms"] == 15 and h["core_atoms"] + h["unique_new_atoms"] == 13
    edge = edge_result(edge_dir, "toluene", "phenol", 1)
    assert edge["ddg"] == pytest.approx((complex_["delta_g"] - solvent["delta_g"]) / 4.184)
    assert edge["conditions"]["water_moves"] == complex_["water_moves"]
    assert edge["lomap_score"] == pytest.approx(info["lomap_score"])
    summary = combine_network(out_dir, ["toluene", "phenol"], [("toluene", "phenol")], 1,
                              {"toluene": -5.0, "phenol": -5.6})
    dg = summary["dg"]["values"]
    assert summary["dg"]["reference"] == "mean of the experimental values"
    assert dg["phenol"]["dg"] - dg["toluene"]["dg"] == pytest.approx(edge["ddg"])
    assert dg["phenol"]["dg"] + dg["toluene"]["dg"] == pytest.approx(-10.6)
    write_results(out_dir, summary)
    first = (out_dir / "edges.csv").read_text()
    # `cufep rbfe analyze` makes the same results from the files of the run
    write_run_files(out_dir, rcfg, ["toluene", "phenol"], [("toluene", "phenol")], "given",
                    {"toluene": -5.0, "phenol": -5.6})
    assert analyse_directory(out_dir) == summary
    # called again: the legs are complete, their results are made again from the same samples
    again = [run_hybrid_leg_safe(job) for job in jobs]
    assert [r["delta_g"] for r in again] == [r["delta_g"] for r in results]
    write_results(out_dir, combine_network(out_dir, ["toluene", "phenol"], [("toluene", "phenol")], 1,
                                           {"toluene": -5.0, "phenol": -5.6}))
    assert (out_dir / "edges.csv").read_text() == first


def test_a_leg_with_samples_keeps_its_windows_when_their_start_configurations_are_made_again(tmp_path, monkeypatch,
                                                                                            deterministic_cpu):
    """Review of PR #18 (P1): a leg with samples and its schedule, whose start configurations are one window short,
    run again with a configuration that asks for other windows. The windows the leg has are kept and only their
    start configurations are made again, so that its samples are continued (here: analysed as they are); a schedule
    that is not of this leg stops the run before any file of the leg is written."""
    from cufep.alchemy.schedule import Schedule
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    # windows that adaptive lambda scheduling chose, started by its walk (the start the review names)
    rcfg = _config(tmp_path, out_dir, "adaptive")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)

    def job(cfg) -> EdgeLegJob:
        return EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                          config=cfg.model_dump(mode="json"), production_ns=cfg.sampling.production_ns, inline=True)

    first = run_hybrid_leg_safe(job(rcfg))
    assert "error" not in first, first
    leg_dir = edge_dir / "repeat_0" / "solvent"
    windows = Schedule.load(leg_dir / "schedule.yaml").states
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    other = rcfg.model_copy(update={"lambda_schedule": rcfg.lambda_schedule.model_copy(
        update={"mode": "fixed", "windows": 4})})
    again = run_hybrid_leg_safe(job(other))
    assert "error" not in again, again
    assert Schedule.load(leg_dir / "schedule.yaml").states == windows
    with np.load(leg_dir / "start_configs.npz") as start:
        assert len(start["positions"]) == len(windows)
    assert again["delta_g"] == first["delta_g"] and again["lambda_windows"] == first["lambda_windows"]
    # the samples go on from their checkpoint: their start is the one they were made from (second review of PR #18)
    assert again["start_configurations"] == first["start_configurations"]

    # the schedule of another repeat, of other windows than those of the samples: refused before anything is written
    # (second review of PR #18)
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    own = (leg_dir / "schedule.yaml").read_text()
    other_windows = Schedule.load(leg_dir / "schedule.yaml")
    other_windows.states = [type(s)(0.4 if 0.0 < s.lambda_hybrid < 1.0 else s.lambda_hybrid, s.rest2_scale)
                             for s in other_windows.states]
    other_windows.save(leg_dir / "schedule.yaml", leg="solvent")
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    before = {name: (leg_dir / name).read_bytes() for name in ("schedule.yaml", "start_configs.npz")}
    refused = run_hybrid_leg_safe(job(other))
    assert "not those of its samples" in refused["error"], refused["error"]
    assert all((leg_dir / name).read_bytes() == data for name, data in before.items())
    (leg_dir / "schedule.yaml").write_text(own)

    # third review of PR #18: another way of exchange (hrex -> none) is another condition of the samples than their
    # windows and settings; refused before anything is written, as the replica exchange would refuse it
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    before = {name: (leg_dir / name).read_bytes() for name in ("schedule.yaml", "start_configs.npz")}
    independent = other.model_copy(update={"sampling": other.sampling.model_copy(update={"exchange": "none"})})
    refused = run_hybrid_leg_safe(job(independent))
    assert "other conditions" in refused["error"], refused["error"]
    assert all((leg_dir / name).read_bytes() == data for name, data in before.items())

    # a schedule of another leg: refused before anything of the leg is written
    text = (leg_dir / "schedule.yaml").read_text().replace("leg: solvent", "leg: complex")
    (leg_dir / "schedule.yaml").write_text(text)
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    starts = (leg_dir / "start_configs.npz").read_bytes()
    refused = run_hybrid_leg_safe(job(other))
    assert "cannot be used for the solvent leg" in refused["error"]
    assert (leg_dir / "schedule.yaml").read_text() == text and (leg_dir / "start_configs.npz").read_bytes() == starts


def test_windows_without_samples_are_kept_only_for_the_settings_they_were_chosen_with(tmp_path, monkeypatch,
                                                                                    deterministic_cpu):
    """Second review of PR #18: windows and start configurations one window short, without samples, run again with
    another soft-core alpha. The record of the start configurations names the windows and the settings they were
    made with; the run is refused before anything is written, as for a leg whose start configurations are whole."""
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)

    def job(cfg) -> EdgeLegJob:
        return EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                          config=cfg.model_dump(mode="json"), production_ns=cfg.sampling.production_ns, inline=True)

    assert "error" not in run_hybrid_leg_safe(job(rcfg))
    leg_dir = edge_dir / "repeat_0" / "solvent"
    import shutil

    shutil.rmtree(leg_dir / "hrex")                     # the windows and their start configurations, no samples
    (leg_dir / "result.json").unlink()
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    before = {name: (leg_dir / name).read_bytes() for name in ("schedule.yaml", "start_configs.npz")}
    softer = rcfg.model_copy(update={"alchemy": rcfg.alchemy.model_copy(update={"softcore_alpha": 0.7})})
    refused = run_hybrid_leg_safe(job(softer))
    assert "were made with other settings" in refused["error"], refused["error"]
    assert all((leg_dir / name).read_bytes() == data for name, data in before.items())


def _same_graph_ligands(tmp_path, out_dir):
    """Toluene and the same toluene 0.05 A away (one bonded graph, two records), with other charges each."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep.prep.ligand import load_ligands

    a = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    AllChem.EmbedMolecule(a, randomSeed=11)
    xyz = a.GetConformer().GetPositions()
    shift = np.array([16.0, 14.0, 16.0]) - xyz.mean(axis=0)
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, extra in (("toluene", 0.0), ("toluene2", 0.05)):
        mol = Chem.Mol(a)
        for i, row in enumerate(xyz + shift + extra):
            mol.GetConformer().SetAtomPosition(i, row.tolist())
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    ligands = load_ligands(tmp_path / "ligands.sdf")
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    charges = {}
    for lig, scale in zip(ligands, (1.0, 1.5)):
        mol = lig.offmol
        mol.assign_partial_charges("gasteiger")
        q = np.array([float(x) for x in mol.partial_charges.m_as("elementary_charge")])
        charges[lig.name] = scale * (q - q.mean())
        (out_dir / "cache" / f"charges_am1bcc_{lig.digest()}.json").write_text(
            json.dumps({"charges": charges[lig.name].tolist()}))
    return ligands, charges


def test_each_end_state_has_the_charges_of_its_own_ligand(tmp_path):
    """Review of PR #17: both ligands were registered in one force field, and OpenMM matched B (the same bonded
    graph, as E/Z isomers have) to the template of A: System B had the charges of A, and the ddG was of other end
    states. Each end state is built with its own ligand alone."""
    import openmm as mm

    from cufep.rbfe.prep import load_end_states, prepare_edge

    out_dir = tmp_path / "run"
    ligands, charges = _same_graph_ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~toluene2"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    for leg in ("complex", "solvent"):
        end = load_end_states(edge_dir, leg)
        for which, name in (("a", "toluene"), ("b", "toluene2")):
            system = mm.XmlSerializer.deserialize(end[which]["system_xml"])
            nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
            got = [nb.getParticleParameters(i)[0].value_in_unit(mm.unit.elementary_charge)
                   for i in end[which]["ligand"]]
            assert np.allclose(got, charges[name], atol=1e-6), (leg, which)


def test_a_prepared_edge_of_another_ph_is_not_taken(tmp_path):
    """Review of PR #17: the protonation states of a PDB without hydrogens follow thermodynamics.ph, and a prepared
    edge was taken by a run with another pH (the record of the edge did not have it)."""
    from cufep.rbfe.prep import edge_made_from, prepare_edge
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    (edge_dir / "prep").mkdir(parents=True)
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(
        {"made_from": edge_made_from(rcfg, ligands[0], ligands[1], "cube"), "legs": {}}))
    other = rcfg.model_copy(update={"thermodynamics": rcfg.thermodynamics.model_copy(update={"ph": 6.0})})
    with pytest.raises(WorkflowError, match="other inputs or settings"):
        prepare_edge(other, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)


def test_a_bilayer_with_a_water_of_four_sites_gets_its_force_field_for_the_lipid_patch(tmp_path, monkeypatch):
    """Review of PR #17: with membrane.enabled and forcefield.water amber14/tip4pew.xml the bilayer is built with
    the 3-site water of the lipid patch (add_membrane asks forcefield_with_water for it), and the relative
    preparation gave it none: TypeError. The building of the bilayer is replaced here by solvation; what is checked
    is that the force field of the patch can be made and holds ligand A."""
    import openmm.unit as u

    from cufep.prep import membrane
    from cufep.prep.solvate import SolvationSpec, solvate
    from cufep.rbfe.prep import prepare_edge

    asked = []

    def bilayer(modeller, forcefield, cfg, log=print, forcefield_with_water=None, water=None):
        from cufep.prep.water import BILAYER_BUILD_WATER

        patch = forcefield_with_water(BILAYER_BUILD_WATER)
        lig = ligands[0].offmol.to_topology().to_openmm()
        lig.setPeriodicBoxVectors(np.eye(3) * 3.0 * u.nanometer)
        patch.createSystem(lig)                        # ligand A has its template in the force field of the patch
        asked.append(water)
        solvate(modeller, forcefield, SolvationSpec(padding_nm=0.8, box_shape="cube", water=water,
                                                    ionic_strength_M=0.0, min_edge_nm=1.3))
        return {"warnings": []}

    monkeypatch.setattr(membrane, "add_membrane", bilayer)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    base = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
    from cufep.rbfe.config import RbfeConfig

    # (cuFEP has no bulk values of this water for the grand canonical moves, which are on by default in a relative
    # calculation: they are switched off here, where the subject is the force field of the lipid patch)
    rcfg = RbfeConfig.model_validate({**base, "membrane": {"enabled": True},
                                      "forcefield": {**base["forcefield"], "water": "amber14/tip4pew.xml"},
                                      "sampling": {**base["sampling"], "gcmc": {"enabled": False}}})
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), out_dir / "edges" / "toluene~phenol", "cube", print)
    assert asked == ["amber14/tip4pew.xml"]


def test_a_run_refused_for_its_prepared_edges_keeps_the_record_of_the_run_before(tmp_path, monkeypatch):
    """Review of PR #17: a run in a finished directory with another cutoff wrote resolved_config.yaml and
    network.json, then prepare_edge refused the prepared edge: the record of the results the directory holds was
    lost, and `cufep rbfe analyze` read the network of the refused run. The prepared edges are checked first."""
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe.prep import edge_made_from
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    data = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
    from cufep.rbfe.config import RbfeConfig

    rcfg = RbfeConfig.model_validate({**data, "network": {"edges": [["toluene", "phenol"]]}})
    write_run_files(out_dir, rcfg, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})
    edge_dir = out_dir / "edges" / "toluene~phenol"
    (edge_dir / "prep").mkdir(parents=True)
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(
        {"made_from": edge_made_from(rcfg, ligands[0], ligands[1], "cube"), "legs": {}}))
    before = {name: (out_dir / name).read_bytes() for name in ("resolved_config.yaml", "network.json")}
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    other = RbfeConfig.model_validate({**rcfg.model_dump(mode="json"), "dynamics": {"cutoff_nm": 0.65}})
    with pytest.raises(WorkflowError, match="other inputs or settings"):
        run_rbfe(other, log=lambda line: None)
    assert all((out_dir / name).read_bytes() == data for name, data in before.items())


def test_edges_that_the_network_combines_are_checked_before_the_run_writes(tmp_path, monkeypatch):
    """Second review of PR #17: A~B finished, then the protein changed and only B~C was run (--edge): B~C had no
    prepared systems yet and passed, and the network combined the A~B of the other protein. Every prepared edge of
    the network is checked before the run writes anything."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.prep.ligand import load_ligands
    from cufep.rbfe import prep as prep_module
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.prep import edge_made_from
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, smiles in (("toluene", "Cc1ccccc1"), ("phenol", "Oc1ccccc1"), ("aniline", "Nc1ccccc1")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    ligands = {lig.name: lig for lig in load_ligands(tmp_path / "ligands.sdf")}
    data = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
    rcfg = RbfeConfig.model_validate({**data, "network": {"edges": [["toluene", "phenol"], ["phenol", "aniline"]]}})
    write_run_files(out_dir, rcfg, list(ligands), [("toluene", "phenol"), ("phenol", "aniline")], "given", {})
    done = out_dir / "edges" / "toluene~phenol" / "prep"
    done.mkdir(parents=True)
    (done / "prep.json").write_text(json.dumps(
        {"made_from": edge_made_from(rcfg, ligands["toluene"], ligands["phenol"], "cube"), "legs": {}}))
    (tmp_path / "protein.pdb").write_text("REMARK another protein\n")          # the receptor changes
    before = {name: (out_dir / name).read_bytes() for name in ("resolved_config.yaml", "network.json")}
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def not_to_be_reached(*args, **kwargs):
        raise AssertionError("an edge was prepared")

    monkeypatch.setattr(prep_module, "prepare_edge", not_to_be_reached)
    with pytest.raises(WorkflowError, match="toluene~phenol"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    assert all((out_dir / name).read_bytes() == data for name, data in before.items())


def test_a_leg_stopped_at_the_checkpoint_of_its_start_keeps_the_start_it_was_made_from(tmp_path, monkeypatch,
                                                                                    deterministic_cpu):
    """Third review of PR #18: the replica exchange writes a checkpoint of its start before its first sample. A leg
    stopped there has no samples and goes on from that checkpoint, not from start configurations made again; its
    result says the start it was made from (the walk), and the start configurations made again are said apart."""
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.sampling import hrex

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "adaptive")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    job = EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                     config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True)
    run = hrex.ReplicaExchangeSampler.run

    def stopped(self, *args, **kwargs):
        raise RuntimeError("stopped after the checkpoint of the start")

    monkeypatch.setattr(hrex.ReplicaExchangeSampler, "run", stopped)
    assert "stopped" in run_hybrid_leg_safe(job)["error"]
    leg_dir = edge_dir / "repeat_0" / "solvent"
    assert (leg_dir / "hrex" / "checkpoint.npz").exists()
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    monkeypatch.setattr(hrex.ReplicaExchangeSampler, "run", run)
    result = run_hybrid_leg_safe(job)
    assert "error" not in result, result
    assert result["start_configurations"].startswith("walk")


def test_samples_started_anew_without_their_checkpoint_have_the_start_made_again(tmp_path, monkeypatch,
                                                                              deterministic_cpu):
    """Fourth review of PR #18: a leg with samples that are not complete and no checkpoint (it was lost), whose start
    configurations are one window short, is started anew by the replica exchange from the start configurations made
    again (the samples it had are not continued); its result says that start, not the walk of the samples it had."""
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import PRE_EQUILIBRATION_START, EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "adaptive")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)

    def job(production_ns: float) -> EdgeLegJob:
        return EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                          config=rcfg.model_dump(mode="json"), production_ns=production_ns, inline=True)

    first = run_hybrid_leg_safe(job(rcfg.sampling.production_ns))
    assert first["start_configurations"].startswith("walk")
    leg_dir = edge_dir / "repeat_0" / "solvent"
    (leg_dir / "hrex" / "checkpoint.npz").unlink()
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    longer = run_hybrid_leg_safe(job(2 * rcfg.sampling.production_ns))        # not complete: started anew
    assert "error" not in longer, longer
    assert longer["start_configurations"] == PRE_EQUILIBRATION_START


def test_start_configurations_made_again_are_the_start_of_samples_started_anew_later(tmp_path, monkeypatch,
                                                                                   deterministic_cpu):
    """Fifth review of PR #18: start configurations made again while the samples were taken up from their checkpoint
    keep the start of those samples (the walk) and the start made again apart; when the checkpoint is lost later
    the replica exchange starts anew from them, and the result says that start. And a leg with samples and no
    schedule is refused by the run before its result is set aside."""
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import PRE_EQUILIBRATION_START, EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "adaptive")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)

    def job(production_ns: float) -> EdgeLegJob:
        return EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                          config=rcfg.model_dump(mode="json"), production_ns=production_ns, inline=True)

    first = run_hybrid_leg_safe(job(rcfg.sampling.production_ns))
    leg_dir = edge_dir / "repeat_0" / "solvent"
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    repaired = run_hybrid_leg_safe(job(rcfg.sampling.production_ns))           # taken up from the checkpoint
    assert repaired["start_configurations"] == first["start_configurations"]
    (leg_dir / "hrex" / "checkpoint.npz").unlink()
    # sixth review: refused by the replica exchange (another way of exchange), the leg is not started anew, and the
    # record keeps the start of its samples
    starts = (leg_dir / "start_configs.npz").read_bytes()
    independent = rcfg.model_copy(update={"sampling": rcfg.sampling.model_copy(update={"exchange": "none"})})
    refused = run_hybrid_leg_safe(EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir),
                                             devices=[None], config=independent.model_dump(mode="json"),
                                             production_ns=2 * rcfg.sampling.production_ns, inline=True))
    assert "error" in refused and (leg_dir / "start_configs.npz").read_bytes() == starts
    # eighth review: the samples it had could not be emptied (the manifest was not replaced): they stay, and so does
    # the record of their start
    from cufep.sampling import storage as storage_module

    rows = storage_module.stored_conditions(leg_dir / "hrex")[0]
    truncate = storage_module.LegStorage.truncate

    def not_emptied(self, n):
        raise OSError("the manifest could not be replaced")

    monkeypatch.setattr(storage_module.LegStorage, "truncate", not_emptied)
    assert "manifest" in run_hybrid_leg_safe(job(2 * rcfg.sampling.production_ns))["error"]
    monkeypatch.setattr(storage_module.LegStorage, "truncate", truncate)
    assert storage_module.stored_conditions(leg_dir / "hrex")[0] == rows
    assert (leg_dir / "start_configs.npz").read_bytes() == starts

    # seventh review: stopped right after the replica exchange wrote the checkpoint of its new start; the retry restores
    # that checkpoint, and the result says the start made again
    from cufep.sampling import hrex

    initialize = hrex.ReplicaExchangeSampler.initialize

    def stopped(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        raise RuntimeError("stopped after the checkpoint of the start")

    monkeypatch.setattr(hrex.ReplicaExchangeSampler, "initialize", stopped)
    assert "stopped" in run_hybrid_leg_safe(job(2 * rcfg.sampling.production_ns))["error"]
    monkeypatch.setattr(hrex.ReplicaExchangeSampler, "initialize", initialize)
    anew = run_hybrid_leg_safe(job(2 * rcfg.sampling.production_ns))           # not complete: started anew
    assert "error" not in anew, anew
    assert anew["start_configurations"] == PRE_EQUILIBRATION_START

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.workflow import WorkflowError

    data = rcfg.model_dump(mode="json")
    full = RbfeConfig.model_validate({**data, "network": {"edges": [["toluene", "phenol"]]}})
    write_run_files(out_dir, full, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})
    (leg_dir / "schedule.yaml").unlink()
    result = (leg_dir / "result.json").read_bytes()
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    with pytest.raises(WorkflowError, match="no schedule.yaml"):
        run_rbfe(full, log=lambda line: None, only_legs=["solvent"])
    assert (leg_dir / "result.json").read_bytes() == result


def test_a_directory_made_without_the_water_moves_is_refused_before_a_run_with_them_writes(tmp_path, monkeypatch):
    """Review of PR #21 (P2): a relative run of a version before the grand canonical moves of water, run again with
    the same YAML (the moves are on by default now): every leg refused its pre-equilibration, after the run had
    replaced resolved_config.yaml and the leg had set its result aside, and the edge that had a ddG had none. The
    pre-equilibrations of the legs are checked before the run writes or starts anything, and the refusal says how
    the directory is continued as it was made; with the moves off it is not refused."""
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe import run as rbfe_run
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.prep import edge_made_from
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.workflow import WorkflowError, equil_made_with, leg_seed

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    data = {**_config(tmp_path, out_dir, "fixed").model_dump(mode="json"),
            "network": {"edges": [["toluene", "phenol"]]}}
    without = RbfeConfig.model_validate({**data, "sampling": {**data["sampling"], "gcmc": {"enabled": False}}})
    with_moves = RbfeConfig.model_validate(data)
    assert with_moves.sampling.gcmc.enabled is True
    # the directory as the run without the moves left it
    write_run_files(out_dir, without, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})
    edge_dir = out_dir / "edges" / "toluene~phenol"
    (edge_dir / "prep").mkdir(parents=True)
    legs = {leg: {"barostat": "isotropic", "n_atoms_a": 100} for leg in ("complex", "solvent")}
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(
        {"made_from": edge_made_from(without, ligands[0], ligands[1], "cube"), "legs": legs}))
    for leg in ("complex", "solvent"):
        leg_dir = edge_dir / "repeat_0" / leg
        leg_dir.mkdir(parents=True)
        np.savez(leg_dir / "equil.npz", timestep_fs=4.0,
                 made_with=json.dumps(equil_made_with(without, leg, False, leg_seed(without, 0, leg))))
        (leg_dir / "result.json").write_text(json.dumps({"delta_g": 1.0, "delta_g_error": 0.1}))
    before = {path: path.read_bytes() for path in sorted(out_dir.rglob("*")) if path.is_file()}
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def not_to_be_reached(*args, **kwargs):
        raise AssertionError("the run went on to its legs")

    monkeypatch.setattr(run_module, "execute_jobs", not_to_be_reached)
    monkeypatch.setattr(rbfe_run, "write_run_files", not_to_be_reached)
    with pytest.raises(WorkflowError) as refused:
        run_rbfe(with_moves, log=lambda line: None)
    message = str(refused.value)
    assert "the pre-equilibration of this leg" in message and "gcmc" in message
    assert message.endswith("with sampling.gcmc.enabled: false the calculation is continued as it was made.")
    after = {path: path.read_bytes() for path in sorted(out_dir.rglob("*"))
             if path.is_file() and path.name != ".cufep.lock"}                 # (the lock of the directory)
    assert after == before
    assert not list(out_dir.rglob("result.previous.json"))
    # with the moves off, as the directory was made, the check passes (the run goes on to write and to its legs)
    monkeypatch.undo()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def stop_at_the_first_write(*args, **kwargs):
        raise RuntimeError("the run began to write")

    monkeypatch.setattr(rbfe_run, "write_run_files", stop_at_the_first_write)
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(without, log=lambda line: None)



def test_windows_and_samples_a_run_cannot_go_on_from_are_refused_before_it_writes(tmp_path, monkeypatch,
                                                                                   deterministic_cpu):
    """Second review of PR #21 (P2, and its P1): a finished edge run again in its directory with another chemical
    potential of water passed the check of its pre-equilibrations (which do not record that value); the run
    replaced resolved_config.yaml, the legs set their results aside, and then the windows of the complex leg were
    refused for their settings: the edge that had a ddG had none. The same for samples whose System this version
    would build otherwise (a run of an earlier version of the hybrid), and for samples of the mover before
    2026-11-28, which left the velocities of an inserted water as they were (extended by this version, both would
    go into one estimate). Every such leg is refused before the run writes or starts anything, and the directory
    stays as it was; the run as it was made passes the checks."""
    from conftest import edit_attributes

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe import run as rbfe_run
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.sampling.storage import read_manifest
    from cufep.workflow import WorkflowError

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    data = {**_config(tmp_path, out_dir, "fixed").model_dump(mode="json"),
            "network": {"edges": [["toluene", "phenol"]]}}
    rcfg = RbfeConfig.model_validate(data)
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    for leg in ("solvent", "complex"):
        result = run_hybrid_leg_safe(EdgeLegJob(
            edge="toluene~phenol", leg=leg, repeat=0, edge_dir=str(edge_dir), devices=[None],
            config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True))
        assert "error" not in result, result
    write_run_files(out_dir, rcfg, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})

    def files() -> dict:
        return {path: path.read_bytes() for path in sorted(out_dir.rglob("*"))
                if path.is_file() and path.name != ".cufep.lock"}

    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def not_to_be_reached(*args, **kwargs):
        raise AssertionError("the run went on to write")

    monkeypatch.setattr(run_module, "execute_jobs", not_to_be_reached)
    monkeypatch.setattr(rbfe_run, "write_run_files", not_to_be_reached)
    before = files()

    def refused(config: dict, match: str) -> None:
        with pytest.raises(WorkflowError, match=match):
            run_rbfe(RbfeConfig.model_validate(config), log=lambda line: None)
        assert files() == before and not list(out_dir.rglob("result.previous.json"))

    # 1. another chemical potential of water: the windows of the complex leg were chosen with the other one
    other = {**data, "sampling": {**data["sampling"], "gcmc": {**data["sampling"]["gcmc"],
                                                              "excess_chemical_potential_kcal_per_mol": -5.0}}}
    refused(other, "gcmc.excess_chemical_potential_kcal_per_mol: -6.09 -> -5.0")
    # 2. samples of the mover of the earlier rule (their record has none)
    hrex = edge_dir / "repeat_0" / "complex" / "hrex"
    recorded = read_manifest(hrex)["attrs"]["conditions"]
    earlier = {**recorded, "settings": {**recorded["settings"], "gcmc": {
        key: value for key, value in recorded["settings"]["gcmc"].items()
        if key != "velocities_of_an_inserted_water"}}}
    edit_attributes(hrex, set={"conditions": earlier})
    before = files()
    refused(data, "gcmc.velocities_of_an_inserted_water: not recorded -> turned with it")
    # 3. samples of another System (as a version that builds the hybrid otherwise made them)
    edit_attributes(hrex, set={"conditions": {**recorded, "system_sha256": "0" * 64}})
    before = files()
    refused(data, "the System")
    # 4. the directory as the run made it passes the checks (in processes of their own, as a run makes them)
    edit_attributes(hrex, set={"conditions": recorded})

    def stop_at_the_first_write(*args, **kwargs):
        raise RuntimeError("the run began to write")

    monkeypatch.setattr(rbfe_run, "write_run_files", stop_at_the_first_write)
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None)


def test_a_leg_stopped_at_the_checkpoint_of_its_start_is_checked_before_a_run_writes(tmp_path, monkeypatch,
                                                                                    deterministic_cpu):
    """Third review of PR #21 (P2): a leg stopped after its replica exchange wrote the checkpoint of its start, before
    its first sample, whose start configurations have to be made again, is held by the leg to the conditions
    recorded with that checkpoint (check_kept_windows). The check before the run ended at "no samples": a leg of a
    System this version builds otherwise was refused only after the run had replaced resolved_config.yaml and
    network.json. It is refused before the run writes anything; as the run made it, and with the start
    configurations of its windows (the replica exchange then takes the conditions of the run), it passes."""
    from conftest import edit_attributes

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe import run as rbfe_run
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.sampling import hrex
    from cufep.sampling.storage import read_manifest, stored_conditions
    from cufep.workflow import WorkflowError

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    data = {**_config(tmp_path, out_dir, "fixed").model_dump(mode="json"),
            "network": {"edges": [["toluene", "phenol"]]}}
    rcfg = RbfeConfig.model_validate(data)
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)

    def stopped(self, *args, **kwargs):
        raise RuntimeError("stopped after the checkpoint of the start")

    with monkeypatch.context() as patch:
        patch.setattr(hrex.ReplicaExchangeSampler, "run", stopped)
        assert "stopped" in run_hybrid_leg_safe(EdgeLegJob(
            edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
            config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True))["error"]
    leg_dir = edge_dir / "repeat_0" / "solvent"
    assert (leg_dir / "hrex" / "checkpoint.npz").exists() and stored_conditions(leg_dir / "hrex")[0] == 0
    write_run_files(out_dir, rcfg, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})
    starts = (leg_dir / "start_configs.npz").read_bytes()
    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})        # one window short
    recorded = read_manifest(leg_dir / "hrex")["attrs"]["conditions"]
    edit_attributes(leg_dir / "hrex", set={"conditions": {**recorded, "system_sha256": "0" * 64}})
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def not_to_be_reached(*args, **kwargs):
        raise AssertionError("the run went on to write")

    def stop_at_the_first_write(*args, **kwargs):
        raise RuntimeError("the run began to write")

    def files() -> dict:
        return {path: path.read_bytes() for path in sorted(out_dir.rglob("*"))
                if path.is_file() and path.name != ".cufep.lock"}

    monkeypatch.setattr(run_module, "execute_jobs", not_to_be_reached)
    monkeypatch.setattr(rbfe_run, "write_run_files", not_to_be_reached)
    before = files()
    with pytest.raises(WorkflowError, match="wrote the checkpoint of its start \\(it has no samples yet\\) under "
                                           "other conditions than this run asks for \\(the System"):
        run_rbfe(rcfg, log=lambda line: None)
    assert files() == before
    monkeypatch.setattr(rbfe_run, "write_run_files", stop_at_the_first_write)
    # with the start configurations of its windows the leg does not hold that checkpoint against the run
    (leg_dir / "start_configs.npz").write_bytes(starts)
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None)
    # and as the run made it, its start configurations to be made again
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})
    edit_attributes(leg_dir / "hrex", set={"conditions": recorded})
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None)
    # ... and its pre-equilibration too (the leg makes it again, with the time step of the configuration): the
    # checkpoint is held to the run all the same, and the leg is not said to hold samples
    (leg_dir / "equil.npz").rename(leg_dir / "equil.aside.npz")
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None)
    edit_attributes(leg_dir / "hrex", set={"conditions": {**recorded, "system_sha256": "0" * 64}})
    monkeypatch.setattr(rbfe_run, "write_run_files", not_to_be_reached)
    with pytest.raises(WorkflowError, match="wrote the checkpoint of its start.*the System"):
        run_rbfe(rcfg, log=lambda line: None)


def test_a_leg_lowered_to_2_fs_goes_on_at_2_fs_when_its_pre_equilibration_is_made_again(tmp_path, monkeypatch,
                                                                                       deterministic_cpu):
    """Fourth review of PR #21 (P2): the pre-equilibration of a leg was unstable at the 4 fs of the YAML and was
    made at 2 fs; the leg stopped at the checkpoint of the start of its replica exchange, which records 2 fs. With
    its start configurations and equil.npz to be made again, the check before the run built the conditions from the
    4 fs of the YAML and refused a leg that the same YAML had made. A leg that goes on from samples (or the
    checkpoint of their start) made at the lowered time step has its pre-equilibration made again at that time
    step, also where 4 fs would be stable this time; the check before the run compares with it; the leg goes on,
    and its pre-equilibration and its result say so."""
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe import run as rbfe_run
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.rbfe.run import run_rbfe, write_run_files
    from cufep.sampling import equilibrate, hrex
    from cufep.sampling.storage import read_manifest, stored_conditions

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    data = {**_config(tmp_path, out_dir, "fixed").model_dump(mode="json"),
            "network": {"edges": [["toluene", "phenol"]]}}
    rcfg = RbfeConfig.model_validate(data)
    assert rcfg.dynamics.timestep_fs == 4.0
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    job = EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                     config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True)
    production = equilibrate._production
    tried = []

    def unstable_at_4_fs(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs):
        tried.append(dt_fs)
        if dt_fs == 4.0:
            raise equilibrate.EquilibrationError("energy became non-finite during the pre-equilibration MD")
        return production(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs)

    def stopped(self, *args, **kwargs):
        raise RuntimeError("stopped after the checkpoint of the start")

    with monkeypatch.context() as patch:
        patch.setattr(equilibrate, "_production", unstable_at_4_fs)
        patch.setattr(hrex.ReplicaExchangeSampler, "run", stopped)
        assert "stopped" in run_hybrid_leg_safe(job)["error"]
    leg_dir = edge_dir / "repeat_0" / "solvent"
    assert tried == [4.0, 2.0]
    assert (leg_dir / "hrex" / "checkpoint.npz").exists() and stored_conditions(leg_dir / "hrex")[0] == 0
    assert read_manifest(leg_dir / "hrex")["attrs"]["conditions"]["timestep_fs"] == 2.0
    # the checkpoint stays; the start configurations and the pre-equilibration are to be made again
    (leg_dir / "start_configs.npz").unlink()
    (leg_dir / "equil.npz").unlink()
    write_run_files(out_dir, rcfg, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})
    protein = _protein()

    def stop_at_the_first_write(*args, **kwargs):
        raise RuntimeError("the run began to write")

    with monkeypatch.context() as patch:
        patch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
        patch.setattr(rbfe_run, "write_run_files", stop_at_the_first_write)
        with pytest.raises(RuntimeError, match="the run began to write"):           # the check before the run passes
            run_rbfe(rcfg, log=lambda line: None)
    # the leg: 4 fs would be stable now, and is not tried
    tried.clear()

    def stable(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs):
        tried.append(dt_fs)
        return production(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs)

    monkeypatch.setattr(equilibrate, "_production", stable)
    result = run_hybrid_leg_safe(job)
    assert "error" not in result, result
    assert tried == [2.0]
    with np.load(leg_dir / "equil.npz") as eq:
        assert float(eq["timestep_fs"]) == 2.0
    note = ("pre-equilibration made again at 2 fs, the time step that the samples of this leg (or the checkpoint of "
            "their start) record; dynamics.timestep_fs is 4 fs, and a pre-equilibration that is unstable at it is "
            "made at 2 fs")
    assert note in result["notes"]
    assert read_manifest(leg_dir / "hrex")["attrs"]["conditions"]["timestep_fs"] == 2.0


def test_a_run_does_not_combine_results_whose_record_the_summary_cites_and_the_directory_lost(tmp_path, monkeypatch):
    """Fifth review of PR #21 (P1), through the run: the summary of the directory refused toluene~phenol a ddG, for
    the other values of the water of its two repeats; repeat 0 has a result of a version before a leg recorded its
    moves, and its value is in the record of its samples. With that ``hrex/`` moved away, a run of the other edge
    alone (--edge) writes a summary of all edges when it ends: the value of repeat 0 is not recorded any more, the
    repeats are averaged, and the refused edge gives the dG of the network, as `cufep rbfe analyze --write` did.
    The results a run combines without sampling them are held to what the summary has of them before the run
    writes anything; with the record there the run goes on. The same for the result of a leg that the summary read
    and that was moved away (the repeat would be left out, and the refused edge get the ddG of the other), though
    not for the leg that a run began again and did not finish, and for the record of the prepared edge (the edge
    would be left out of the comparison of the prepared systems). A summary that cannot be read holds nothing.
    Each leg is held to the summary by itself, and only the repeats this run combines (sixth review)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.prep.ligand import load_ligands
    from cufep.rbfe import run as rbfe_run
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.prep import edge_made_from
    from cufep.rbfe.run import rewrite_results, run_rbfe, write_run_files
    from cufep.sampling.storage import MANIFEST
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, smiles in (("toluene", "Cc1ccccc1"), ("phenol", "Oc1ccccc1"), ("aniline", "Nc1ccccc1")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    ligands = {lig.name: lig for lig in load_ligands(tmp_path / "ligands.sdf")}
    data = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
    rcfg = RbfeConfig.model_validate({**data, "sampling": {**data["sampling"], "repeats": 2},
                                      "network": {"edges": [["toluene", "phenol"], ["phenol", "aniline"]]}})
    write_run_files(out_dir, rcfg, list(ligands), [("toluene", "phenol"), ("phenol", "aniline")], "given", {})
    edge_dir = out_dir / "edges" / "toluene~phenol"
    (edge_dir / "prep").mkdir(parents=True)
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(
        {"made_from": edge_made_from(rcfg, ligands["toluene"], ligands["phenol"], "cube"),
         "legs": {leg: {"barostat": "isotropic", "n_atoms_a": 100} for leg in ("complex", "solvent")}}))

    def result(dg_kcal: float, **moves) -> dict:
        return {"delta_g": dg_kcal * 4.184, "delta_g_error": 0.3 * 4.184, "temperature_K": 298.15,
                "pressure_bar": 1.0, "lrc_method": "none", "prepared_systems_sha256": "abc", "n_states": 9,
                "backend": "openmm", "notes": [], **moves,
                "quality": {"overlap": {"lowest": 0.1, "ok": True}, "samples": {"counted": 100.0}}}

    water = {"enabled": True, "excess_chemical_potential_kcal_per_mol": -6.09, "standard_volume_nm3": 0.030345}
    legs = {(0, "complex"): result(-12.0, gcmc={"insertions_accepted": [1]}),          # before the record of a leg
            (1, "complex"): result(-12.4, gcmc={"insertions_accepted": [1]}, water_moves={
                **water, "velocities_of_an_inserted_water": "turned with it"}),
            (0, "solvent"): result(-10.5, gcmc=None, water_moves={"enabled": False}),
            (1, "solvent"): result(-10.5, gcmc=None, water_moves={"enabled": False})}
    for (repeat, leg), content in legs.items():
        leg_dir = edge_dir / f"repeat_{repeat}" / leg
        leg_dir.mkdir(parents=True)
        (leg_dir / "result.json").write_text(json.dumps(content))
    hrex = edge_dir / "repeat_0" / "complex" / "hrex"
    hrex.mkdir()
    (hrex / MANIFEST).write_text(json.dumps({"parts": [], "attrs": {"conditions": {"settings": {"gcmc": {
        **water, "excess_chemical_potential_kcal_per_mol": -5.0}}}}}))
    summary = rewrite_results(out_dir)
    assert "ddg" not in summary["edges"][0] and summary["edges"][0]["inconsistent_legs"] and summary["dg"] is None

    def files() -> dict:
        return {path: path.read_bytes() for path in sorted(out_dir.rglob("*"))
                if path.is_file() and path.name != ".cufep.lock"}

    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def stop_at_the_first_write(*args, **kwargs):
        raise RuntimeError("the run began to write")

    monkeypatch.setattr(rbfe_run, "write_run_files", stop_at_the_first_write)
    hrex.rename(tmp_path / "aside")
    before = files()
    with pytest.raises(WorkflowError, match="holds the results of 1 edge\\(s\\) which this run would combine without "
                                           "sampling them.*toluene~phenol: repeat_0/complex/hrex.*Nothing was changed"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    assert files() == before
    (tmp_path / "aside").rename(hrex)
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])

    def refused_without(path, said: str) -> None:
        path.rename(tmp_path / "aside")
        before = files()
        with pytest.raises(WorkflowError, match=f"would combine without sampling them.*toluene~phenol: {said}"):
            run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
        assert files() == before
        (tmp_path / "aside").rename(path)

    result_of_a_leg = edge_dir / "repeat_0" / "complex" / "result.json"
    refused_without(result_of_a_leg, "repeat_0/complex/result.json")
    refused_without(edge_dir / "prep" / "prep.json", "prep/prep.json")
    # a leg that a run began again and did not finish, stopped (its marker of a started leg) or failed (its error),
    # has no result: an ordinary state, not refused. The backup of the result it had does not say that: it stays
    # when the leg has finished again (sixth review, P2: the current result of such a leg, moved away, passed for
    # it, and the mean of two repeats became the value of the other)
    backup = result_of_a_leg.with_name("result.previous.json")
    backup.write_text(result_of_a_leg.read_text())
    refused_without(result_of_a_leg, "repeat_0/complex/result.json")
    for marker in ("started.json", "error.json"):
        result_of_a_leg.rename(tmp_path / "aside")
        result_of_a_leg.with_name(marker).write_text("{}")
        with pytest.raises(RuntimeError, match="the run began to write"):
            run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
        result_of_a_leg.with_name(marker).unlink()
        (tmp_path / "aside").rename(result_of_a_leg)
    backup.unlink()
    # the edge itself run again: its legs are sampled, and nothing of them is held to the summary
    hrex.rename(tmp_path / "aside")
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None)
    # ... but with --leg solvent its complex legs are used as they are. Sixth review (P1): the solvent leg of repeat
    # 0 was stopped when it was run again and has no result, so the repeat was not complete when the run began; its
    # complex leg was not looked at, and both repeats were combined when the solvent leg had finished
    solvent = edge_dir / "repeat_0" / "solvent"
    (solvent / "result.json").rename(solvent / "result.previous.json")
    (solvent / "started.json").write_text("{}")
    before = files()
    with pytest.raises(WorkflowError, match="would combine without sampling them.*toluene~phenol: "
                                           "repeat_0/complex/hrex"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["toluene~phenol"], only_legs=["solvent"])
    assert files() == before
    (tmp_path / "aside").rename(hrex)
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["toluene~phenol"], only_legs=["solvent"])
    (solvent / "started.json").unlink()
    (solvent / "result.previous.json").rename(solvent / "result.json")
    # one repeat asked for where the summary read two (sixth review, P2): repeat 1 is not combined by this run, and
    # its files are not asked for; with two repeats they are
    settings = rcfg.model_dump(mode="json")
    one = RbfeConfig.model_validate({**settings, "sampling": {**settings["sampling"], "repeats": 1}})
    (edge_dir / "repeat_1").rename(tmp_path / "repeat_1")
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(one, log=lambda line: None, only_edges=["phenol~aniline"])
    with pytest.raises(WorkflowError, match="toluene~phenol: repeat_1/complex/result.json, "
                                           "repeat_1/solvent/result.json"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    (tmp_path / "repeat_1").rename(edge_dir / "repeat_1")
    # a result that cannot be read (seventh review, P2: a truncated result of a solvent leg passed, the run
    # sampled the other edge and failed when it combined the results, beside the summary of the run before)
    damaged = edge_dir / "repeat_0" / "solvent" / "result.json"
    whole = damaged.read_text()
    damaged.write_text("{")
    before = files()
    with pytest.raises(WorkflowError, match="would combine without sampling them.*toluene~phenol: "
                                           "repeat_0/solvent/result.json \\(it cannot be read\\)"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    assert files() == before
    damaged.write_text(whole)
    # a summary that cannot be read: the tables are the record of the edges this run does not run. With the legs
    # they were made from the run goes on; with the record of repeat 0 moved away the legs give toluene~phenol a ddG
    # where edges.csv has none (the review of the fixes of the sixth review: nothing was held then)
    (out_dir / "summary.json").write_text("{")
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    hrex.rename(tmp_path / "aside")
    before = files()
    with pytest.raises(WorkflowError, match="cannot be read or is not there, and edges.csv holds 1 value\\(s\\) of "
                                           "edges with a leg this run does not sample.*the ddG of toluene~phenol: "
                                           "none in edges.csv, -1.700 ± .* from the legs.*Nothing was changed"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    assert files() == before
    # ... and only of those. Eighth review (P2): an edge that the run is to run again was held to the table too.
    # With a ddG of toluene~phenol in edges.csv (its two repeats made alike) and a summary that cannot be read,
    # a run of every edge, and one of that edge alone, were refused for the value the table holds of it
    (tmp_path / "aside").rename(hrex)
    (hrex / MANIFEST).write_text(json.dumps({"parts": [], "attrs": {"conditions": {"settings": {"gcmc": water}}}}))
    for name in ("summary.json", "edges.csv", "ligands.csv"):          # (the results as these legs give them)
        (out_dir / name).unlink()
    assert rewrite_results(out_dir)["edges"][0]["ddg"] == pytest.approx(-1.7)
    assert "toluene~phenol,toluene,phenol,-1.700,0.200," in (out_dir / "edges.csv").read_text()
    (out_dir / "summary.json").write_text("{")
    for chosen in (None, ["toluene~phenol"], ["phenol~aniline"]):
        with pytest.raises(RuntimeError, match="the run began to write"):
            run_rbfe(rcfg, log=lambda line: None, only_edges=chosen)
    # ... but an edge of which the run samples one leg alone (--leg) uses the other as it is, and is held to the
    # table as a whole: without a summary no repeat has a record of its own (the review of that fix: with --leg
    # solvent, the complex result of repeat 1 moved away passed, and the mean of two repeats became -1.500)
    with pytest.raises(RuntimeError, match="the run began to write"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["toluene~phenol"], only_legs=["solvent"])
    moved = edge_dir / "repeat_1" / "complex" / "result.json"
    moved.rename(tmp_path / "aside.json")
    before = files()
    with pytest.raises(WorkflowError, match="edges with a leg this run does not sample.*the ddG of toluene~phenol: "
                                           "-1.700 ± 0.200 in edges.csv, -1.500 ± 0.424 from the legs"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["toluene~phenol"], only_legs=["solvent"])
    assert files() == before
    (tmp_path / "aside.json").rename(moved)
    # a file that is a record and not the result of a leg (the same review): named before the run writes, where
    # it failed with an error of its own when the results were combined
    whole = moved.read_text()
    moved.write_text(json.dumps({"delta_g": -43.9, "delta_g_error": 1.2}))
    with pytest.raises(WorkflowError, match="repeat_1/complex/result.json is not the result of a leg \\(it has no "
                                           "n_states\\)"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    moved.write_text(whole)
    # the record of a prepared edge that cannot be read: named
    prepared = edge_dir / "prep" / "prep.json"
    whole = prepared.read_text()
    prepared.write_text("{")
    with pytest.raises(WorkflowError, match="prep/prep.json cannot be read"):
        run_rbfe(rcfg, log=lambda line: None, only_edges=["phenol~aniline"])
    prepared.write_text(whole)


def test_given_windows_start_as_a_precalculation_starts_them_and_the_md_of_a_leg_is_counted(tmp_path, monkeypatch,
                                                                                         deterministic_cpu):
    """Item 1 of the plan of 2026-11-28 (issue #27). ``lambda_schedule.start_configurations: precalculation``: the
    windows a leg is given (here its own, ``windows: {complex: ..., solvent: ...}``) are started as a precalculation
    of adaptive lambda scheduling starts the windows of its first iteration, and the record of the start
    configurations and the result say so; by default they start from the end of the pre-equilibration. The MD of a
    leg after its pre-equilibration is in its result (``md_ps``: what made its windows, and its replica exchange),
    for a leg with given windows and for one whose precalculation chose them."""
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import PRE_EQUILIBRATION_START, PRECALCULATION_START, EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.workflow import start_record_of

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    given = {"mode": "fixed", "windows": {"complex": 4, "solvent": [0.0, 0.3, 1.0]}}
    results, records = {}, {}
    for name, more in (("precalculation", {**given, "start_configurations": "precalculation"}), ("default", given),
                       ("adaptive", {})):
        out_dir = tmp_path / name
        ligands = _ligands(tmp_path, out_dir)
        data = _config(tmp_path, out_dir, "adaptive").model_dump(mode="json")
        rcfg = RbfeConfig.model_validate({**data, "lambda_schedule": {**data["lambda_schedule"], **more}})
        edge_dir = out_dir / "edges" / "toluene~phenol"
        prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
        results[name] = run_hybrid_leg_safe(EdgeLegJob(
            edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
            config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production("solvent"), inline=True))
        assert "error" not in results[name], results[name]
        records[name] = start_record_of(edge_dir / "repeat_0" / "solvent" / "start_configs.npz")

    # the windows of the leg, and how their replicas were started
    for name in ("precalculation", "default"):
        assert results[name]["lambda_windows"] == {"morph": [0.0, 0.3, 1.0]}
        assert records[name]["windows"] == "given"
    made = records["precalculation"]
    assert made["start_configurations"] == PRECALCULATION_START
    assert (made["walk_ps"], made["window_md_ps"]) == pytest.approx((0.2, 0.5))
    assert "start_md_ps" not in made and "minimize_steps" not in made
    assert results["precalculation"]["start_configurations"] == (
        f"{PRECALCULATION_START}, 0.2 ps in each window, then 0.5 ps in each window")
    assert records["default"]["start_configurations"] == PRE_EQUILIBRATION_START
    assert results["default"]["start_configurations"] == PRE_EQUILIBRATION_START
    assert records["adaptive"]["windows"] == "chosen by this precalculation"

    # the MD after the pre-equilibration: every window 10 iterations of equilibration and 20 of production, of
    # 0.248 ps each as integrated (62 steps of 4 fs for the 0.25 ps asked)
    per_window = 30 * 62 * 0.004
    for name, windows_ps in (("precalculation", 3 * (0.2 + 0.5)), ("default", 3 * 0.2)):
        assert results[name]["md_ps"] == pytest.approx({
            "lambda_windows": windows_ps, "replica_exchange": 3 * per_window, "total": windows_ps + 3 * per_window})
    adaptive = results["adaptive"]
    n = adaptive["n_states"]
    from cufep.alchemy.schedule import Schedule

    simulated = Schedule.load(tmp_path / "adaptive" / "edges" / "toluene~phenol" / "repeat_0" / "solvent"
                              / "schedule.yaml").provenance["windows_simulated"]
    assert simulated >= 3
    # the walk through the three windows of the first iteration, and every window the precalculation ran
    assert adaptive["md_ps"]["lambda_windows"] == pytest.approx(3 * 0.2 + simulated * 0.5)
    assert adaptive["md_ps"]["replica_exchange"] == pytest.approx(n * per_window)
    assert adaptive["md_ps"]["total"] == pytest.approx(3 * 0.2 + simulated * 0.5 + n * per_window)


def test_each_leg_of_a_relative_run_gets_its_production(tmp_path, monkeypatch):
    """``sampling.production_ns: {complex: ..., solvent: ...}`` of a relative run: the job of each leg is given the
    production of its leg (item 1 of the plan of 2026-11-28: a leg with given windows is given the MD of one whose
    precalculation chose them, leg by leg); one number is that of both legs."""
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import run_rbfe

    class Captured(Exception):
        pass

    def capture(jobs, *args, **kwargs):
        raise Captured({job.leg: job.production_ns for job in jobs})

    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    monkeypatch.setattr(run_module, "execute_jobs", capture)
    for production, expected in (({"complex": 0.005, "solvent": 0.01}, {"complex": 0.005, "solvent": 0.01}),
                                 (0.005, {"complex": 0.005, "solvent": 0.005})):
        out_dir = tmp_path / f"run_{len(expected)}_{production if isinstance(production, float) else 'legs'}"
        _ligands(tmp_path, out_dir)
        data = {**_config(tmp_path, out_dir, "fixed").model_dump(mode="json"),
                "network": {"edges": [["toluene", "phenol"]]}}
        rcfg = RbfeConfig.model_validate({**data, "sampling": {**data["sampling"], "production_ns": production}})
        with pytest.raises(Captured) as jobs:
            run_rbfe(rcfg, log=lambda line: None)
        assert jobs.value.args[0] == pytest.approx(expected)
