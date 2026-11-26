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


def _ligands(tmp_path, out_dir, smiles_b: str = "Oc1ccccc1", name_b: str = "phenol", first_h_b: int = 8):
    """Toluene and phenol (or ``smiles_b``, whose first ring hydrogen is ``first_h_b``) on one ring, 0.6 nm from the
    peptide, with Gasteiger charges in the cache (shifted to the formal charge of each)."""
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
    ring = {**{i: i for i in range(1, 7)}, **{10 + k: first_h_b + k for k in range(5)}}
    b = Chem.AddHs(Chem.MolFromSmiles(smiles_b))
    xyz_a = a.GetConformer().GetPositions()
    AllChem.EmbedMolecule(b, randomSeed=11, coordMap={j: Point3D(*xyz_a[i]) for i, j in ring.items() if j < 7})
    rdMolAlign.AlignMol(b, a, atomMap=[(j, i) for i, j in ring.items()])
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, mol, dg in (("toluene", a, -5.0), (name_b, b, -5.6)):
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
        shift = (sum(charges) - lig.formal_charge) / len(charges)
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
                     "pre_equilibration": {"complex": short, "solvent": short}},
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
    from cufep.rbfe.prep import prepare_edge
    from cufep.rbfe.run import analyse_directory, combine_network, edge_result, write_results, write_run_files

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
    assert all(np.isfinite(r["delta_g"]) and r["delta_g_error"] > 0 for r in results)
    h = solvent["hybrid"]
    assert h["core_atoms"] + h["unique_old_atoms"] == 15 and h["core_atoms"] + h["unique_new_atoms"] == 13
    edge = edge_result(edge_dir, "toluene", "phenol", 1)
    assert edge["ddg"] == pytest.approx((complex_["delta_g"] - solvent["delta_g"]) / 4.184)
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


def test_an_edge_prepared_by_an_earlier_rule_is_not_taken(tmp_path):
    """Second review of PR #19 (P2): an edge that changes the net charge, prepared by the version that chose its
    alchemical water by the distances in the cell alone (a water 0.2 nm from the solutes across the boundary), and an
    edge of a custom core mapped without chirality, were taken up again when the cutoff or the SMARTS was the same.
    Their records hold the version of the rule; one without it (or of another) is refused before anything is written.
    The records of the other edges are those of earlier versions."""
    from cufep.rbfe.prep import PREPARATION_RULES, edge_made_from, prepared_edge
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    toluene, phenolate = _ligands(tmp_path, out_dir, "[O-]c1ccccc1", "phenolate", 7)
    rcfg = _config(tmp_path, out_dir, "fixed")
    custom = rcfg.model_copy(update={"mapping": rcfg.mapping.model_copy(update={"core_smarts": "c1ccccc1"})})
    (tmp_path / "n").mkdir()
    neutral = _ligands(tmp_path / "n", out_dir)
    plain = edge_made_from(rcfg, *neutral, "cube")
    assert not {"alchemical_water_rule", "custom_core_rule"} & set(plain)
    for cfg, ligands, key in ((rcfg, (toluene, phenolate), "alchemical_water_rule"),
                              (custom, neutral, "custom_core_rule")):
        record = edge_made_from(cfg, *ligands, "cube")
        assert record[key] == PREPARATION_RULES[key.removesuffix("_rule")]
        earlier = {k: v for k, v in record.items() if k != key}
        edge_dir = tmp_path / key
        (edge_dir / "prep").mkdir(parents=True)
        (edge_dir / "prep" / "prep.json").write_text(json.dumps({"made_from": earlier, "legs": {}}))
        before = (edge_dir / "prep" / "prep.json").read_text()
        with pytest.raises(WorkflowError, match=f"{key}.*an earlier rule"):
            prepared_edge(cfg, *ligands, edge_dir, "cube")
        assert (edge_dir / "prep" / "prep.json").read_text() == before
        (edge_dir / "prep" / "prep.json").write_text(json.dumps({"made_from": record, "legs": {}}))
        assert prepared_edge(cfg, *ligands, edge_dir, "cube") is not None


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

    # (cuFEP has no excess chemical potential of TIP4P-Ew for the grand canonical moves of the complex leg)
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


def test_an_edge_that_changes_the_charge_has_an_alchemical_water_in_both_legs(tmp_path):
    """Toluene into phenolate (charge -1): in each leg one water farther than the cutoff from the solutes is Na+ in
    end state B (the box of the test has no ion: the ion of the force field), so that both ends have one total
    charge; the hybrid of the leg has it as its alchemical water, in a residue that the grand canonical moves of the
    complex leg leave out; the record of the prepared edge names the rule."""
    import openmm as mm

    from cufep.rbfe.leg import build_leg
    from cufep.rbfe.prep import edge_made_from, load_end_states, prepare_edge

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir, "[O-]c1ccccc1", "phenolate", 7)
    assert [lig.formal_charge for lig in ligands] == [0, -1]
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenolate"
    info = prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    assert edge_made_from(rcfg, *ligands, "cube")["alchemical_water_cutoff_nm"] == 0.8
    for leg in ("complex", "solvent"):
        water = info["legs"][leg]["alchemical_water"]
        assert water["charge_change"] == 1 and water["ion"]["name"] == "NA" and water["ion"]["charge"] == 1.0
        end = load_end_states(edge_dir, leg)
        totals = []
        for which in ("a", "b"):
            system = mm.XmlSerializer.deserialize(end[which]["system_xml"])
            nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
            totals.append(sum(nb.getParticleParameters(i)[0].value_in_unit(mm.unit.elementary_charge)
                              for i in range(system.getNumParticles())))
        assert totals[0] == pytest.approx(totals[1], abs=1e-6)
        built = build_leg(rcfg, edge_dir, leg)
        assert built.hybrid.alchemical_water == tuple(sorted(water["atoms_a"]))
        names = {a.residue.name for a in built.prep["topology"].atoms() if a.index in set(water["atoms_a"])}
        assert names == {"ALW"}
        if built.gcmc is not None:
            assert not set(water["atoms_a"]) & set(built.gcmc.waters.reshape(-1).tolist())



def test_a_custom_core_restricts_the_mapping_of_an_edge_and_of_the_network(tmp_path):
    """With mapping.core_smarts of the ring carbons, toluene -> phenol maps the six ring carbons only (Kartograf's
    mapping without the core has the ring hydrogens too), and LOMAP plans its network from mappings so restricted."""
    from cufep.rbfe.mapping import MappingSettings, kartograf_mapping, plan_network

    ligands = _ligands(tmp_path, tmp_path / "run")
    full = kartograf_mapping(ligands[0], ligands[1])
    core = kartograf_mapping(ligands[0], ligands[1], MappingSettings(core_smarts="c1ccccc1"))
    assert sorted(core) == [1, 2, 3, 4, 5, 6] and len(full) > len(core)
    assert all(full[i] == j for i, j in core.items())
    edges = plan_network(ligands, MappingSettings(core_smarts="c1ccccc1"), require_cycle_covering=False)
    assert [tuple(sorted(e)) for e in edges] == [("phenol", "toluene")]


def test_an_edge_whose_preparation_stops_leaves_the_record_of_the_run_before(tmp_path, monkeypatch):
    """Review of PR #19: an edge that changes the charge in a box without a water far enough from the solutes stopped
    in its preparation after the run had written resolved_config.yaml and network.json, which then described a run
    that never began. The record of the run is written after its edges are prepared."""
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe import prep as prep_module
    from cufep.rbfe.charge import ChargeChangeError
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import run_rbfe

    out_dir = tmp_path / "run"
    _ligands(tmp_path, out_dir)
    data = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
    rcfg = RbfeConfig.model_validate({**data, "network": {"edges": [["toluene", "phenol"]]}})
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def no_water(*args, **kwargs):
        raise ChargeChangeError("no water is farther than 0.8 nm from the solutes")

    monkeypatch.setattr(prep_module, "prepare_edge", no_water)
    with pytest.raises(ChargeChangeError):
        run_rbfe(rcfg, log=lambda line: None)
    assert not (out_dir / "network.json").exists() and not (out_dir / "resolved_config.yaml").exists()


def test_the_perturbed_region_of_a_prepared_edge_is_chosen_by_the_rule(tmp_path):
    """sampling.rest2.region: perturbed in a leg: the hot atoms come from the two prepared ligands; toluene -> phenol,
    whose unique atoms sit on the ring by bonds that are not rotatable, has the ring and both substituents hot, the
    whole hybrid ligand; no protein residue is hot."""
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import build_leg
    from cufep.rbfe.prep import prepare_edge

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    data = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
    data["sampling"]["rest2"] = {**data["sampling"]["rest2"], "region": "perturbed"}
    rcfg = RbfeConfig.model_validate(data)
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    built = build_leg(rcfg, edge_dir, "complex")
    assert built.hybrid.rest2_ligand_atoms == built.hybrid.ligand_atoms and built.hot_atoms == []


def test_the_plan_refuses_an_edge_that_the_run_refuses(tmp_path):
    """Review of PR #19 (P3): `cufep rbfe plan` showed an edge from a charge of -1 to +1 as "+2 (an alchemical water)"
    and ended with 0, where `cufep rbfe run` refuses it. The plan refuses it with the same reason."""
    from click.testing import CliRunner
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import cli

    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, smiles in (("acetate", "CC(=O)[O-]"), ("ammonium", "CC[NH3+]")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=3)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    (tmp_path / "protein.pdb").write_text("REMARK built in the test\n")
    (tmp_path / "config.yaml").write_text("system: {protein: protein.pdb, ligands: ligands.sdf}\n"
                                          "network: {edges: [[acetate, ammonium]]}\n")
    result = CliRunner().invoke(cli.main, ["rbfe", "plan", str(tmp_path / "config.yaml")])
    assert result.exit_code != 0 and "changes the net charge by +2" in result.output



def test_legs_of_the_hybrid_before_its_torsion_fix_are_left_out_and_named_to_run_again(tmp_path):
    """Second review of PR #19 (P2): the REST2 factor of the proper torsions of the core multiplied their term of A
    alone (version 1 of the hybrid, as the A2A edge 4e~4b of the example was computed), and a result of such a leg was
    combined with a leg of this version after `--leg solvent --reschedule`. A leg of version 1 (its record has neither
    hybrid_version nor grand_canonical, which came with the fix) with REST2, on an edge with terms of B in those
    torsions, is left out of the ddG, and its RESULT line names the legs to run again; a leg of this version, one of
    the commit of the fix (grand_canonical recorded), one without REST2 and an edge without such terms keep it."""
    import shutil

    import openmm as mm
    from test_rbfe_network import _leg_result, _write_legs
    from test_rbfe_rest2_region import _change_a_core_torsion

    from cufep.rbfe.leg import terms_left_unscaled_by_version_1
    from cufep.rbfe.prep import load_end_states, prepare_edge
    from cufep.rbfe.run import combine_network, result_lines

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    prep = out_dir / "edges" / "toluene~phenol" / "prep"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), prep.parent, "cube", print)
    plain = terms_left_unscaled_by_version_1(prep.parent, "complex", "pocket")
    changed = tmp_path / "changed"
    shutil.copytree(prep, changed)
    for leg in ("complex", "solvent"):
        end = load_end_states(prep.parent, leg)
        xml = changed / f"{leg}_b_system.xml"
        system = mm.XmlSerializer.deserialize(xml.read_text())
        assert _change_a_core_torsion(system, end["b"]["ligand"], end["core_map"]) == 1
        xml.write_text(mm.XmlSerializer.serialize(system))

    def version_1(dg: float, scale: float = 2.0) -> dict:
        return {**_leg_result(dg), "run": {"rest2": {"scale": scale}}}

    def edge_of(name: str, complex_, solvent, prepared=changed) -> tuple[dict, list[str]]:
        root = tmp_path / name
        shutil.copytree(prepared, root / "edges" / "toluene~phenol" / "prep")
        _write_legs(root, "toluene~phenol", 0, complex_, solvent)
        summary = combine_network(root, ["toluene", "phenol"], [("toluene", "phenol")], 1, {})
        return summary["edges"][0], result_lines(summary)

    shutil.copytree(changed, tmp_path / "count" / "prep")
    n = terms_left_unscaled_by_version_1(tmp_path / "count", "complex", "pocket")
    assert n >= 1
    edge, lines = edge_of("both", version_1(-12.0), version_1(-10.5))
    assert "ddg" not in edge and len(edge["earlier_version"]) == 2
    assert f"REST2 left {n} terms of ligand B" in edge["earlier_version"][0]
    assert any("--edge toluene~phenol --leg complex --leg solvent --reschedule" in x for x in lines)
    current = {**_leg_result(-12.0)}                                       # this version (hybrid_version 2)
    edge, lines = edge_of("solvent", current, version_1(-10.5))
    assert "ddg" not in edge and any("--leg solvent --reschedule" in x and "--leg complex" not in x for x in lines)
    fixed = {**_leg_result(-12.0), "run": {"grand_canonical": {"enabled": False}, "rest2": {"scale": 2.0}}}
    assert "ddg" in edge_of("fix", fixed, {**fixed, "delta_g": -10.5 * 4.184})[0]
    assert "ddg" in edge_of("cold", version_1(-12.0, 1.0), version_1(-10.5, 1.0))[0]
    assert plain == 0 and "ddg" in edge_of("plain", version_1(-12.0), version_1(-10.5), prep)[0]
