"""An edge from its preparation to its ddG, on the CPU with stock OpenMM (cufep.rbfe.prep, cufep.rbfe.leg,
cufep.rbfe.run): an alanine dipeptide is the protein, toluene becomes phenol next to it. The settings are made
small for the test; what is checked is the path, the windows the legs share, and that a run called again in the
same directory finds the same results."""
from __future__ import annotations

import json
from pathlib import Path

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
    assert "ddg" not in edge and [item["leg"] for item in edge["earlier_version"]] == ["complex", "solvent"]
    assert f"REST2 left {n} terms of ligand B" in edge["earlier_version"][0]["why"]
    assert any("--edge 'toluene~phenol' --repeat 0 --leg complex --leg solvent --reschedule" in x for x in lines)
    current = {**_leg_result(-12.0)}                                       # this version (hybrid_version 2)
    edge, lines = edge_of("solvent", current, version_1(-10.5))
    assert "ddg" not in edge and any("--repeat 0 --leg solvent --reschedule" in x and "--leg complex" not in x
                                     for x in lines)
    fixed = {**_leg_result(-12.0), "run": {"grand_canonical": {"enabled": False}, "rest2": {"scale": 2.0}}}
    assert "ddg" in edge_of("fix", fixed, {**fixed, "delta_g": -10.5 * 4.184})[0]
    assert "ddg" in edge_of("cold", version_1(-12.0, 1.0), version_1(-10.5, 1.0))[0]
    assert plain == 0 and "ddg" in edge_of("plain", version_1(-12.0), version_1(-10.5), prep)[0]


@pytest.fixture(scope="module")
def edges_of_version_1(tmp_path_factory):
    """A prepared edge toluene -> phenol (``prep``: its prep/ directory; ligand B has no term in the proper torsions
    of the core that differs from A) and the same with one such term changed (``changed``)."""
    import shutil

    import openmm as mm
    from test_rbfe_rest2_region import _change_a_core_torsion

    from cufep.rbfe.prep import load_end_states, prepare_edge

    tmp_path = tmp_path_factory.mktemp("version_1")
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    edge0 = tmp_path / "prepared"
    prepare_edge(_config(tmp_path, out_dir, "fixed"), ligands[0], ligands[1], _protein(), edge0, "cube", print)
    changed = tmp_path / "changed"
    shutil.copytree(edge0 / "prep", changed)
    for leg in ("complex", "solvent"):
        end = load_end_states(edge0, leg)
        xml = changed / f"{leg}_b_system.xml"
        system = mm.XmlSerializer.deserialize(xml.read_text())
        assert _change_a_core_torsion(system, end["b"]["ligand"], end["core_map"]) == 1
        xml.write_text(mm.XmlSerializer.serialize(system))
    return {"prep": edge0 / "prep", "changed": changed}


def _version_1(dg: float, windows=(1.0, 2.0, 1.0)) -> dict:
    """A leg result of version 1 of the hybrid (its record has neither hybrid_version nor grand_canonical)."""
    from test_rbfe_network import _leg_result

    return {**_leg_result(dg), "run": {"rest2": {"scale": 2.0 if max(windows) > 1.0 else 1.5,
                                                   "scale_of_window": list(windows)}}}


def _run_dir(root: Path, legs: dict) -> Path:
    """A run directory of the edges ``legs`` (edge -> (prepared prep/ directory, complex result, solvent result)),
    with a summary.json of an earlier state that gives every edge a ddG of +0.91."""
    import shutil

    import yaml
    from test_rbfe_network import _write_legs

    for edge, (prepared, c, s) in legs.items():
        shutil.copytree(prepared, root / "edges" / edge / "prep")
        _write_legs(root, edge, 0, c, s)
    pairs = [edge.split("~") for edge in legs]
    names = sorted({x for pair in pairs for x in pair})
    (root / "resolved_config.yaml").write_text(yaml.safe_dump({"sampling": {"repeats": 1}}))
    (root / "network.json").write_text(json.dumps({"ligands": names, "edges": pairs}))
    (root / "summary.json").write_text(json.dumps({
        "ligands": names, "experiment_kcal_mol": {}, "cycle_closures": [], "notes": [], "dg": None,
        "edges": [{"edge": edge, "ligand_a": a, "ligand_b": b, "ddg": 0.91, "se": 0.2, "not_reliable": []}
                  for edge, (a, b) in zip(legs, pairs)]}))
    return root


def test_the_report_takes_the_edges_that_the_analysis_takes(tmp_path, edges_of_version_1):
    """Third review of PR #19 (P1): `cufep benchmark report` gave the ddG of an earlier summary.json (+0.91) for an
    edge whose legs of version 1 `cufep rbfe analyze` leaves out (the A2A example: +0.91 and +1.38 kcal/mol); and
    (fourth review, P2) for an edge none of whose legs has finished again yet. The report takes the legs of the
    directory as the analysis does, and says that legs are left out."""
    import csv
    import shutil

    from cufep.benchmarks.rbfe import report
    from cufep.rbfe.run import analyse_directory

    root = _run_dir(tmp_path / "run", {"toluene~phenol": (edges_of_version_1["changed"], _version_1(-12.0),
                                                          _version_1(-10.5))})
    assert "ddg" not in analyse_directory(root)["edges"][0]
    text = report("tyk2", root)
    row = next(csv.DictReader((root / "rbfe_report_edges.csv").open()))
    assert row["ddg"] == "" and "+0.91" not in text and "legs of an earlier version are left out" in text
    # (fourth review, P2) every leg being run again and none finished yet: the earlier summary.json is not reported
    for path in (root / "edges" / "toluene~phenol").glob("repeat_*"):
        shutil.rmtree(path)
    report("tyk2", root)
    assert next(csv.DictReader((root / "rbfe_report_edges.csv").open()))["ddg"] == ""


def test_the_line_of_a_left_out_leg_runs_that_leg_alone_again(tmp_path, edges_of_version_1):
    """Third and fourth reviews of PR #19 (P2, P3): the RESULT line of a complex leg of version 1 said to remove its
    pre-equilibration and run `--edge E --leg complex --reschedule`, which runs every repeat of that leg (a valid
    repeat 1 lost its result, and was refused for its own pre-equilibration) and threw away a pre-equilibration that a
    run without the moves uses again. The line runs the one repeat (`--repeat`), and removes nothing: --reschedule sets
    aside a pre-equilibration that cannot be used, and only that."""
    import shutil

    import yaml
    from test_rbfe_network import _leg_result, _write_legs

    from cufep.rbfe.run import analyse_directory, result_lines

    root = tmp_path / "run"
    shutil.copytree(edges_of_version_1["changed"], root / "edges" / "toluene~phenol" / "prep")
    _write_legs(root, "toluene~phenol", 0, _version_1(-12.0), _leg_result(-10.5))
    _write_legs(root, "toluene~phenol", 1, _leg_result(-11.9), _leg_result(-10.4))
    (root / "edges" / "toluene~phenol" / "repeat_0" / "complex" / "equil.npz").write_bytes(b"of version 1")
    (root / "resolved_config.yaml").write_text(yaml.safe_dump({"sampling": {"repeats": 2}}))
    (root / "network.json").write_text(json.dumps({"ligands": ["phenol", "toluene"], "edges": [["toluene", "phenol"]]}))
    line = next(x for x in result_lines(analyse_directory(root)) if x.startswith("RESULT toluene~phenol: left out"))
    assert line.endswith(f"To compute it again: cufep rbfe run CONFIG -o {root} --edge 'toluene~phenol' --repeat 0 "
                         "--leg complex --reschedule") and "remove" not in line
    # (sixth review, P2) the run directory is in the command: a directory made with -o is not output.dir of CONFIG
    import shlex

    words = shlex.split(line.split("To compute it again: ")[1])
    assert words[words.index("-o") + 1] == str(root)


def test_a_leg_of_version_1_without_a_heated_window_keeps_its_ddg(tmp_path, edges_of_version_1):
    """Third review of PR #19 (P2): `sampling.rest2.scale: 1.5` with the windows [0, 1] heats no window (both at scale
    1): version 1 left nothing unscaled there, and the leg is the leg of this version (``scale_of_window``)."""
    from cufep.rbfe.run import analyse_directory

    root = _run_dir(tmp_path / "run", {"toluene~phenol": (edges_of_version_1["changed"], _version_1(-12.0, (1.0, 1.0)),
                                                          _version_1(-10.5, (1.0, 1.0)))})
    assert analyse_directory(root)["edges"][0]["ddg"] == pytest.approx(-1.5)


def test_a_leg_of_version_1_with_a_perturbed_hot_region_is_left_out(tmp_path, monkeypatch, edges_of_version_1):
    """Third review of PR #19 (P2): the hot region `perturbed` of version 1 was chosen by an earlier rule (a ring of a
    long chain was heated that the present rule leaves cold), so the terms of B it left unscaled cannot be counted by
    the present rule (0 for terms in that ring): such a leg is left out, on an edge without such terms too."""
    from cufep.rbfe.run import analyse_directory
    from cufep.sampling import storage

    monkeypatch.setattr(storage, "stored_conditions", lambda directory: (
        10, {"settings": {"sampling.rest2": {"scale": 2.0, "region": "perturbed"}}}))
    root = _run_dir(tmp_path / "run", {"toluene~phenol": (edges_of_version_1["prep"], _version_1(-12.0),
                                                          _version_1(-10.5))})
    edge = analyse_directory(root)["edges"][0]
    assert "ddg" not in edge and "sampling.rest2.region: perturbed" in edge["earlier_version"][0]["why"]


def test_a_leg_whose_prepared_systems_cannot_be_read_is_left_out_alone(tmp_path, edges_of_version_1):
    """Third and fourth reviews of PR #19 (P2): a leg of version 1 whose edge has prep.json but not the XML of an end
    state, or an NPZ of it that is empty or cut short, stopped the analysis of every edge (FileNotFoundError,
    EOFError, BadZipFile). That leg is left out, and the other edges are analysed."""
    import shutil

    from test_rbfe_network import _leg_result

    from cufep.rbfe.run import analyse_directory

    # an XML missing (third review), an NPZ empty and one cut short (fourth review: EOFError, BadZipFile)
    for name, damage in (("xml", lambda prep: (prep / "solvent_a_system.xml").unlink()),
                         ("empty", lambda prep: (prep / "solvent_a.npz").write_bytes(b"")),
                         ("cut", lambda prep: (prep / "solvent_a.npz").write_bytes(
                             (prep / "solvent_a.npz").read_bytes()[:200]))):
        broken = tmp_path / f"broken_{name}"
        shutil.copytree(edges_of_version_1["prep"], broken)
        damage(broken)
        root = _run_dir(tmp_path / f"run_{name}", {
            "toluene~phenol": (broken, _version_1(-12.0), _version_1(-10.5)),
            "phenol~toluene": (edges_of_version_1["prep"], _leg_result(-10.5), _leg_result(-12.0))})
        edges = {e["edge"]: e for e in analyse_directory(root)["edges"]}
        assert "ddg" not in edges["toluene~phenol"], name
        assert "cannot be read" in edges["toluene~phenol"]["earlier_version"][0]["why"], name
        assert edges["phenol~toluene"]["ddg"] == pytest.approx(1.5), name


def test_a_leg_with_the_moves_is_computed_again_only_from_its_own_pre_equilibration(tmp_path, monkeypatch,
                                                                                    deterministic_cpu):
    """Third and fourth reviews of PR #19 and the independent review, on a complex leg with the grand canonical moves
    of water:
    a finished leg whose pre-equilibration has no record of its ghost waters (an earlier version) was refused on a
    resume after its result had been set aside: a leg that goes on from its samples does not start from its
    pre-equilibration, and keeps its result;
    a leg that would start from such a pre-equilibration is refused before the run writes anything, with the command
    that computes that leg alone, and `--reschedule` sets the pre-equilibration aside with the windows and samples
    (in previous_<n>/) and computes the leg anew;
    a leg whose start configurations were made again in the cell of a new pre-equilibration went on from its
    checkpoint in the old one: a checkpoint in another cell is refused before the run writes anything, and the System
    of the samples is that of the prepared hybrid, as earlier versions recorded it (fifth review: with the cell in it,
    finished legs of earlier versions could not be analysed again);
    start configurations in another cell than the pre-equilibration fixes are refused before the run writes
    anything."""
    import shutil

    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import EdgeLegJob, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.rbfe.run import check_pre_equilibrations_of_the_run
    from cufep.workflow import WorkflowError

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), out_dir / "edges" / "toluene~phenol", "cube", print)

    def copy(name: str, of: Path = out_dir) -> Path:
        shutil.copytree(of / "edges", tmp_path / name / "edges")
        return tmp_path / name

    def job(root: Path, reschedule: bool = False) -> EdgeLegJob:
        return EdgeLegJob(edge="toluene~phenol", leg="complex", repeat=0,
                          edge_dir=str(root / "edges" / "toluene~phenol"), devices=[None],
                          config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns,
                          inline=True, reschedule=reschedule)

    def checked(root: Path, reschedule: bool = False) -> None:
        check_pre_equilibrations_of_the_run(rcfg, root, [("toluene", "phenol")], [0], ["complex"], reschedule)

    def leg_dir(root: Path) -> Path:
        return root / "edges" / "toluene~phenol" / "repeat_0" / "complex"

    first = run_hybrid_leg_safe(job(out_dir))
    assert "error" not in first and first["gcmc"] is not None and (leg_dir(out_dir) / "equil_ghosts.json").exists()
    # a finished leg of an earlier version (no record of its ghosts): a resume keeps its result
    old = copy("old")
    (leg_dir(old) / "equil_ghosts.json").unlink()
    checked(old)
    again = run_hybrid_leg_safe(job(old))
    assert "error" not in again and again["delta_g"] == first["delta_g"] and (leg_dir(old) / "result.json").exists()
    # one that would start from that pre-equilibration: refused before anything is written, with its command
    shutil.rmtree(leg_dir(old) / "hrex")
    (leg_dir(old) / "result.json").unlink()
    with pytest.raises(WorkflowError, match="records no ghost waters") as refused:
        checked(old)
    assert (f"cufep rbfe run CONFIG -o {old} --edge 'toluene~phenol' --repeat 0 --leg complex --reschedule"
            in str(refused.value))
    checked(old, reschedule=True)
    anew = run_hybrid_leg_safe(job(old, reschedule=True))
    assert "error" not in anew and (leg_dir(old) / "equil_ghosts.json").exists()
    assert [p.parent.name for p in leg_dir(old).glob("previous_*/equil.npz")] == ["previous_1"]
    assert not (leg_dir(old) / "previous_1" / "equil_ghosts.json").exists()
    # (fifth review, P1) the samples record the System of the prepared hybrid, as every earlier version did: a
    # finished leg of an earlier version is analysed again with its result (the cell in the System changed it)
    import hashlib

    from cufep.rbfe.leg import build_leg
    from cufep.sampling.storage import stored_conditions

    _n, conditions = stored_conditions(leg_dir(out_dir) / "hrex")
    built = build_leg(rcfg, out_dir / "edges" / "toluene~phenol", "complex")
    assert conditions["system_sha256"] == hashlib.sha256(built.xml.encode()).hexdigest()
    # (sixth review, P1) samples of eb36e30 and 08d58d5 (the full TYK2 benchmark), which recorded the same System with
    # the fixed cell as its default cell, are continued and analysed again; a System of neither form is not
    import openmm as mm
    from conftest import edit_attributes

    from cufep.workflow import alchemical_box

    with np.load(leg_dir(out_dir) / "equil.npz") as eq:
        fixed = alchemical_box(eq, True)
    built.hybrid.system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in fixed])   # as 08d58d5 did
    of_08d58d5 = hashlib.sha256(mm.XmlSerializer.serialize(built.hybrid.system).encode()).hexdigest()
    assert of_08d58d5 != conditions["system_sha256"]
    for name, digest, kept in (("of_08d58d5", of_08d58d5, True), ("other", "0" * 64, False)):
        root = copy(name)
        edit_attributes(leg_dir(root) / "hrex", set={"conditions": {**conditions, "system_sha256": digest}})
        result = run_hybrid_leg_safe(job(root))
        if kept:
            assert "error" not in result and result["delta_g"] == first["delta_g"]
        else:
            assert "made under other conditions" in result["error"]
    # a checkpoint whose replicas are in another cell than the pre-equilibration fixes (a pre-equilibration made
    # again; fifth review: also one without samples, whose conditions are taken as new): not continued
    stale = copy("stale")
    (leg_dir(stale) / "result.json").unlink()
    with np.load(leg_dir(stale) / "hrex" / "checkpoint.npz") as ckpt:
        kept = {key: ckpt[key] for key in ckpt.files}
    kept["boxes"] = kept["boxes"] * 0.99
    np.savez(leg_dir(stale) / "hrex" / "checkpoint.npz", **kept)
    with pytest.raises(WorkflowError, match="the replicas of the checkpoint"):
        checked(stale)
    continued = run_hybrid_leg_safe(job(stale))
    assert "error" in continued and "the replicas of the checkpoint" in continued["error"]
    with np.load(leg_dir(out_dir) / "equil.npz") as eq:
        arrays = {key: eq[key] for key in eq.files}
    arrays["mean_box"] = arrays["mean_box"] * 1.01
    # start configurations in another cell than the pre-equilibration fixes: refused before anything is written
    cell = copy("cell")
    shutil.rmtree(leg_dir(cell) / "hrex")
    (leg_dir(cell) / "result.json").unlink()
    np.savez(leg_dir(cell) / "equil.npz", **arrays)
    with pytest.raises(WorkflowError, match="are in another cell than its pre-equilibration fixes"):
        checked(cell)
    assert "are in another cell than its pre-equilibration fixes" in run_hybrid_leg_safe(job(cell))["error"]


def test_a_run_of_one_repeat_runs_that_repeat_alone(tmp_path, monkeypatch):
    """Fourth review of PR #19 (P2): the command to compute a leg of repeat 0 again ran every repeat (`--edge` and
    `--leg` only). `--repeat` restricts the run to the repeats named; another number is refused before anything is
    written."""
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe.run import run_rbfe
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    rcfg.sampling.repeats = 2
    rcfg = rcfg.model_validate({**rcfg.model_dump(mode="json"), "network": {"edges": [["toluene", "phenol"]]}})
    seen = []

    def no_jobs(jobs, gpus, factor, log):
        seen.extend((job.edge, job.repeat, job.leg) for job in jobs)
        return [], None, False

    protein = _protein()
    monkeypatch.setattr(run_module, "execute_jobs", no_jobs)
    monkeypatch.setattr(run_module, "write_timing", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    with pytest.raises(WorkflowError, match="--repeat: 2 is not a repeat of the run"):
        run_rbfe(rcfg, log=lambda line: None, only_repeats=[2])
    assert not (out_dir / "network.json").exists()
    run_rbfe(rcfg, log=lambda line: None, only_repeats=[1], only_legs=["complex"])
    assert seen == [("toluene~phenol", 1, "complex")]


def test_an_edge_prepared_by_an_earlier_rule_is_left_out_by_every_entry_point(tmp_path, edges_of_version_1):
    """Independent review of PR #19: an edge prepared before the rule versions (a charge-changing edge whose
    alchemical water version 1 chose by distances in the cell alone, an edge of a custom core mapped without
    chirality) is refused by `cufep rbfe run`, and `cufep rbfe analyze` and `cufep benchmark report` gave its ddG and
    the network dG. It is left out of both, with a note and the way to compute it again."""
    import csv
    import shutil

    from test_rbfe_network import _leg_result

    from cufep.benchmarks.rbfe import report
    from cufep.rbfe.run import analyse_directory, result_lines

    def charge(made: dict, info: dict) -> None:
        made["alchemical_water_cutoff_nm"] = 0.8
        info["legs"]["complex"]["alchemical_water"] = {"atoms_a": [9]}

    def core(made: dict, info: dict) -> None:
        made["mapping"]["core_smarts"] = "c1ccccc1"

    for name, change in (("charge", charge), ("core", core)):
        prep = tmp_path / f"prep_{name}"
        shutil.copytree(edges_of_version_1["prep"], prep)
        info = json.loads((prep / "prep.json").read_text())
        change(info["made_from"], info)                                     # as an earlier version recorded it
        (prep / "prep.json").write_text(json.dumps(info))
        root = _run_dir(tmp_path / f"run_{name}", {"toluene~phenol": (prep, _leg_result(-12.0), _leg_result(-10.5))})
        summary = analyse_directory(root)
        assert "ddg" not in summary["edges"][0] and summary["edges"][0]["earlier_preparation"], name
        assert any("whose prepared systems this version does not take" in note for note in summary["notes"]), name
        line = next(x for x in result_lines(summary) if x.startswith("RESULT toluene~phenol"))
        assert f"remove 'edges/toluene~phenol' of {root}, then cufep rbfe run CONFIG -o {root} --edge" in line, name
        report("tyk2", root)
        assert next(csv.DictReader((root / "rbfe_report_edges.csv").open()))["ddg"] == "", name


def test_an_archive_of_a_pre_equilibration_that_a_run_stopped_in_is_completed_and_refused_without_reschedule(tmp_path):
    """Fifth review of PR #19 (P2): --reschedule moved a pre-equilibration that cannot be used into
    .previous_1.partial/ and stopped before the record of its ghost waters; the call that completed the archive no
    longer knew, and left the record behind; a run without --reschedule started a relative leg in that state (the ABFE
    refuses it). The archive records what it sets aside, and a relative run refuses an unfinished archive."""
    from cufep.engine.leg import EQUIL_GHOSTS
    from cufep.rbfe.run import check_pre_equilibrations_of_the_run
    from cufep.workflow import ARCHIVE_RECORD, WorkflowError, archive_leg

    out_dir = tmp_path / "run"
    rcfg = _config(tmp_path, out_dir, "fixed")
    leg_dir = out_dir / "edges" / "toluene~phenol" / "repeat_0" / "complex"
    leg_dir.mkdir(parents=True)
    for name, text in (("schedule.yaml", "windows"), ("result.json", "{}"), (EQUIL_GHOSTS, "{}")):
        (leg_dir / name).write_text(text)
    stage = leg_dir / ".previous_1.partial"                         # as the stopped call left it
    stage.mkdir()
    (stage / ARCHIVE_RECORD).write_text(json.dumps({"pre_equilibration": True}))
    (stage / "equil.npz").write_bytes(b"the pre-equilibration set aside")
    with pytest.raises(WorkflowError, match="was being started anew"):
        check_pre_equilibrations_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], None, False)
    check_pre_equilibrations_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], None, True)
    done = archive_leg(leg_dir)                                       # the call of --reschedule that completes it
    assert done == leg_dir / "previous_1"
    assert sorted(p.name for p in done.iterdir()) == sorted(["schedule.yaml", "result.json", EQUIL_GHOSTS,
                                                              "equil.npz"])
    assert not (leg_dir / EQUIL_GHOSTS).exists() and not list(leg_dir.glob(".previous_*"))
    # (sixth review, P2) begun by a version that wrote no record (eb36e30, 08d58d5): the pre-equilibration in the
    # archive and not in the leg was moved, and the record of its ghosts goes with it
    for name, text in (("schedule.yaml", "windows"), (EQUIL_GHOSTS, "{}")):
        (leg_dir / name).write_text(text)
    stage = leg_dir / ".previous_2.partial"
    stage.mkdir()
    (stage / "equil.npz").write_bytes(b"the pre-equilibration set aside")
    done = archive_leg(leg_dir)
    assert done == leg_dir / "previous_2" and (done / EQUIL_GHOSTS).exists() and not (leg_dir / EQUIL_GHOSTS).exists()
    assert not (done / ARCHIVE_RECORD).exists()


def test_start_configurations_that_are_made_again_are_not_refused_for_their_cell(tmp_path):
    """Fifth review of PR #19 (P2): start configurations of another cell were refused although the leg would make them
    again (no schedule: stopped between the two files; or windows they are not of). Those the leg uses are checked."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.engine.leg import fixed_cell_problems

    leg_dir = tmp_path / "leg"
    leg_dir.mkdir()
    box = np.eye(3) * 4.0
    np.savez(leg_dir / "equil.npz", mean_box=box, box=box)
    schedule = fixed_schedule(["morph"], {"morph": 3})
    states = [state.to_dict() for state in schedule.states]
    np.savez(leg_dir / "start_configs.npz", boxes=np.stack([box * 1.01] * len(states)),
             made_with=json.dumps({"states": states}))
    assert fixed_cell_problems(leg_dir) == []                        # no schedule: they are made again
    schedule.save(leg_dir / "schedule.yaml")
    assert "are in another cell" in fixed_cell_problems(leg_dir)[0]  # the windows have them: they are used
    np.savez(leg_dir / "start_configs.npz", boxes=np.stack([box * 1.01] * (len(states) - 1)),
             made_with=json.dumps({"states": states[:-1]}))
    assert fixed_cell_problems(leg_dir) == []                        # of other windows: made again


def test_edges_to_prepare_again_are_named_together_and_their_names_quoted(tmp_path, edges_of_version_1):
    """Fifth review of PR #19: (P2) with two edges prepared by an earlier rule, the line of each said to remove that
    edge and run it alone, which the run refused for the other one (a run checks every prepared edge first): the
    lines and the refusal of the run name all of them; (P2) a prep.json cut short was read again by the comparison of
    the network and stopped the analysis of every edge: such an edge is left out; (P3) the commands quote the names of
    edges, whose ligands may have spaces (`4e U`)."""
    import shlex
    import shutil

    from test_rbfe_network import _leg_result

    from cufep.rbfe.run import analyse_directory, prepare_again_command, rerun_command, result_lines

    old = tmp_path / "old"
    shutil.copytree(edges_of_version_1["prep"], old)
    info = json.loads((old / "prep.json").read_text())
    info["made_from"]["mapping"]["core_smarts"] = "c1ccccc1"           # a custom core of the earlier rule
    (old / "prep.json").write_text(json.dumps(info))
    cut = tmp_path / "cut"
    shutil.copytree(edges_of_version_1["prep"], cut)
    (cut / "prep.json").write_text((cut / "prep.json").read_text()[:40])
    root = _run_dir(tmp_path / "run", {"a~b": (old, _leg_result(-12.0), _leg_result(-10.5)),
                                       "b~c": (old, _leg_result(-11.0), _leg_result(-10.0)),
                                       "c~d": (cut, _leg_result(-9.0), _leg_result(-8.0)),
                                       "d~e": (edges_of_version_1["prep"], _leg_result(-12.0), _leg_result(-10.5))})
    summary = analyse_directory(root)
    edges = {e["edge"]: e for e in summary["edges"]}
    assert "ddg" not in edges["c~d"] and "cannot be read" in edges["c~d"]["earlier_preparation"][0]
    assert edges["d~e"]["ddg"] == pytest.approx(-1.5)
    lines = [x for x in result_lines(summary) if x.startswith("RESULT a~b") or x.startswith("RESULT b~c")]
    assert len(lines) == 2 and all(prepare_again_command(["a~b", "b~c", "c~d"], root) in x for x in lines)
    words = shlex.split(rerun_command("4e U~4b D", 0, ["complex"]))
    assert words[words.index("--edge") + 1] == "4e U~4b D"
    words = shlex.split(prepare_again_command(["4e U~4b D"]).split(" then ")[1])
    assert words[words.index("--edge") + 1] == "4e U~4b D"


def test_a_run_names_every_edge_it_does_not_take_and_how_to_prepare_them_again(tmp_path, monkeypatch):
    """Fifth review of PR #19 (P2): a run refused for the first prepared edge it did not take; removed and run alone as
    it said, the run was refused for the next. The run names every such edge, and one way that prepares them all."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.prep.ligand import load_ligands
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.prep import edge_made_from
    from cufep.rbfe.run import prepare_again_command, run_rbfe
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
    for a, b in (("toluene", "phenol"), ("phenol", "aniline")):
        done = out_dir / "edges" / f"{a}~{b}" / "prep"
        done.mkdir(parents=True)
        (done / "prep.json").write_text(json.dumps({"made_from": edge_made_from(rcfg, ligands[a], ligands[b], "cube"),
                                                    "legs": {}}))
    (tmp_path / "protein.pdb").write_text("REMARK another protein\n")          # the receptor changes
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    with pytest.raises(WorkflowError) as refused:
        run_rbfe(rcfg, log=lambda line: None)
    assert "toluene~phenol" in str(refused.value) and "phenol~aniline" in str(refused.value)
    assert prepare_again_command(["toluene~phenol", "phenol~aniline"], out_dir) in str(refused.value)


def test_a_checkpoint_of_08d58d5_without_samples_goes_on_with_start_configurations_made_again(
        tmp_path, monkeypatch, deterministic_cpu):
    """Seventh review of PR #19 (P2): a leg of 08d58d5 stopped after the checkpoint of its start and before its first
    sample, without its start configurations: the form of its System was read from the samples (none), the hybrid
    of the preparation chosen, and the check of the windows it keeps, which reads the conditions recorded with the
    checkpoint, refused it as another System. The form is read where that check reads it: the leg goes on."""
    import hashlib

    import openmm as mm
    from conftest import edit_attributes

    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import EdgeLegJob, build_leg, run_hybrid_leg_safe
    from cufep.rbfe.prep import prepare_edge
    from cufep.sampling.hrex import ReplicaExchangeSampler
    from cufep.sampling.storage import stored_layout
    from cufep.workflow import alchemical_box

    monkeypatch.setattr(leg_module, "MINIMIZE_STEPS", 100)
    monkeypatch.setattr(leg_module, "START_MD_PS", 0.2)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    job = EdgeLegJob(edge="toluene~phenol", leg="complex", repeat=0, edge_dir=str(edge_dir), devices=[None],
                     config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True)

    def stop(self, *args, **kwargs):
        raise RuntimeError("stopped after the checkpoint of the start")

    with monkeypatch.context() as patch:
        patch.setattr(ReplicaExchangeSampler, "run", stop)
        assert "stopped after the checkpoint" in run_hybrid_leg_safe(job)["error"]
    leg_dir = edge_dir / "repeat_0" / "complex"
    parts, attrs = stored_layout(leg_dir / "hrex")
    assert (leg_dir / "hrex" / "checkpoint.npz").exists() and sum(p[1] for p in parts) == 0
    # as 08d58d5 recorded it: the System with the fixed cell as its default cell; the start configurations missing
    built = build_leg(rcfg, edge_dir, "complex")
    with np.load(leg_dir / "equil.npz") as eq:
        built.hybrid.system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in alchemical_box(eq, True)])
    digest = hashlib.sha256(mm.XmlSerializer.serialize(built.hybrid.system).encode()).hexdigest()
    # (ninth review, P2) the checkpoint is continued on the backend it was made with (a leg that fell back to stock
    # OpenMM and stopped there chose the plugin again and was refused): its backend is that of the samples
    import shutil

    from cufep.rbfe.run import check_samples_of_the_run, write_run_files
    from cufep.sampling.storage import stored_backend

    assert stored_backend(leg_dir / "hrex") == "openmm"
    # (ninth review, P2) checked before a run writes anything, as the ABFE checks it; a checkpoint without the record
    # of its conditions is refused, before the run and in the leg (P3)
    from cufep.workflow import WorkflowError

    other_exchange = rcfg.model_validate({**rcfg.model_dump(mode="json"),
                                          "sampling": {**rcfg.model_dump(mode="json")["sampling"], "exchange": "none"}})
    check_samples_of_the_run(rcfg, edge_dir.parent.parent, [("toluene", "phenol")], [0], None, False)
    with pytest.raises(WorkflowError, match="exchange: hrex -> none"):
        check_samples_of_the_run(other_exchange, edge_dir.parent.parent, [("toluene", "phenol")], [0], None, False)
    check_samples_of_the_run(other_exchange, edge_dir.parent.parent, [("toluene", "phenol")], [0], None, True)
    # (tenth review, P2) another length of an iteration: other MD steps per iteration of the samples (625 -> 1250 at
    # 4 fs in the review), refused before the run writes
    longer = rcfg.model_validate({**rcfg.model_dump(mode="json"), "sampling": {
        **rcfg.model_dump(mode="json")["sampling"], "time_per_iteration_ps": 2 * rcfg.sampling.time_per_iteration_ps}})
    with pytest.raises(WorkflowError, match="steps_per_iteration"):
        check_samples_of_the_run(longer, edge_dir.parent.parent, [("toluene", "phenol")], [0], None, False)
    # (tenth review, P2) another temperature: the pre-equilibration cannot be used again, so --reschedule, which keeps
    # it, is not the way; refused before anything is set aside, with or without --reschedule, and no command of this
    # directory is offered
    from cufep.rbfe.run import check_pre_equilibrations_of_the_run

    warmer = rcfg.model_validate({**rcfg.model_dump(mode="json"), "thermodynamics": {
        **rcfg.model_dump(mode="json")["thermodynamics"], "temperature_K": 310.0}})
    for reschedule in (False, True):
        with pytest.raises(WorkflowError, match="cannot be used again") as refused:
            check_pre_equilibrations_of_the_run(warmer, edge_dir.parent.parent, [("toluene", "phenol")], [0],
                                                ["complex"], reschedule)
        assert "temperature_K" in str(refused.value) and "--reschedule" not in str(refused.value)
    check_pre_equilibrations_of_the_run(rcfg, edge_dir.parent.parent, [("toluene", "phenol")], [0], ["complex"], True)
    bare = tmp_path / "bare"
    shutil.copytree(edge_dir.parent, bare / "edges")
    (bare / "edges" / "toluene~phenol" / "repeat_0" / "complex" / "hrex" / "manifest.json").unlink()
    with pytest.raises(WorkflowError, match="no record of the conditions"):
        check_samples_of_the_run(rcfg, bare, [("toluene", "phenol")], [0], None, False)
    in_leg = run_hybrid_leg_safe(EdgeLegJob(
        edge="toluene~phenol", leg="complex", repeat=0, edge_dir=str(bare / "edges" / "toluene~phenol"),
        devices=[None], config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True))
    assert "no record of the conditions" in in_leg["error"]
    # the whole run: refused before it writes its record
    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe.run import run_rbfe

    write_run_files(edge_dir.parent.parent, rcfg, ["toluene", "phenol"], [("toluene", "phenol")], "given", {})
    record = (edge_dir.parent.parent / "resolved_config.yaml").read_bytes()
    protein = _protein()
    with monkeypatch.context() as patch:
        patch.setattr(run_module, "prepare_protein_and_settings",
                      lambda cfg, pdb, log: (cfg, AutoReport(), protein))
        patch.setattr(run_module, "execute_jobs", lambda *args, **kwargs: pytest.fail("a job was started"))
        given = other_exchange.model_validate({**other_exchange.model_dump(mode="json"),
                                               "network": {"edges": [["toluene", "phenol"]]}})
        with pytest.raises(WorkflowError, match="exchange: hrex -> none"):
            run_rbfe(given, log=lambda line: None, only_legs=["complex"])
    assert (edge_dir.parent.parent / "resolved_config.yaml").read_bytes() == record
    # (eighth review, P2) the checkpoint is held to the conditions it was made under, start configurations or not: under
    # other ones (exchange: none), or with a System of neither form, it is refused and its record kept
    from cufep.rbfe.leg import recorded_form_of_the_system
    from cufep.sampling.storage import stored_conditions

    other = rcfg.model_validate({**rcfg.model_dump(mode="json"),
                                 "sampling": {**rcfg.model_dump(mode="json")["sampling"], "exchange": "none"}})
    for name, cfg, conditions in (("exchange", other, attrs["conditions"]),
                                  ("neither", rcfg, {**attrs["conditions"], "system_sha256": "0" * 64})):
        root = tmp_path / name
        shutil.copytree(edge_dir.parent, root / "edges")
        dir_of_leg = root / "edges" / "toluene~phenol" / "repeat_0" / "complex"
        edit_attributes(dir_of_leg / "hrex", set={"conditions": conditions})
        refused = run_hybrid_leg_safe(EdgeLegJob(
            edge="toluene~phenol", leg="complex", repeat=0, edge_dir=str(root / "edges" / "toluene~phenol"),
            devices=[None], config=cfg.model_dump(mode="json"), production_ns=cfg.sampling.production_ns,
            inline=True))
        assert "made under other conditions" in refused.get("error", ""), name
        assert stored_layout(dir_of_leg / "hrex")[1]["conditions"] == conditions, name
    edit_attributes(leg_dir / "hrex", set={"conditions": {**attrs["conditions"], "system_sha256": digest}})
    # (eighth review, P3) the System is left in the form it was found in, for the Contexts the leg makes of it on one
    # device (start configurations made again)
    with np.load(leg_dir / "equil.npz") as eq:
        fixed = alchemical_box(eq, True)
    fresh = build_leg(rcfg, edge_dir, "complex")
    chosen = recorded_form_of_the_system(leg_dir, fresh.xml, fresh.hybrid.system, fixed)
    assert hashlib.sha256(chosen.encode()).hexdigest() == digest
    assert hashlib.sha256(mm.XmlSerializer.serialize(fresh.hybrid.system).encode()).hexdigest() == digest
    # (seventh review, the reply) with its start configurations too: the samples would take the conditions of the
    # run as new and the hybrid of the preparation, another PME grid where the default cell sets it. Both go on with
    # the System of the checkpoint, and the samples record its checksum
    kept = tmp_path / "kept"
    shutil.copytree(edge_dir.parent, kept / "edges")
    (leg_dir / "start_configs.npz").unlink()
    for root in (edge_dir.parent.parent, kept):
        dir_of_leg = root / "edges" / "toluene~phenol" / "repeat_0" / "complex"
        resumed = EdgeLegJob(edge="toluene~phenol", leg="complex", repeat=0,
                             edge_dir=str(root / "edges" / "toluene~phenol"), devices=[None],
                             config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns,
                             inline=True)
        result = run_hybrid_leg_safe(resumed)
        assert "error" not in result and np.isfinite(result["delta_g"]), root
        n, conditions = stored_conditions(dir_of_leg / "hrex")
        assert n > 0 and conditions["system_sha256"] == digest, root


def test_an_archive_that_adds_the_pre_equilibration_records_it_before_it_moves_anything(tmp_path, monkeypatch):
    """Seventh review of PR #19 (P3): an archive begun without the pre-equilibration (set_aside.json false) and taken
    up by a call that sets it aside too kept that in memory only: stopped after it moved equil.npz, the next call
    no longer knew, and left the record of the ghost waters behind. The record says it before anything is moved."""
    from pathlib import Path as PathClass

    from cufep.engine.leg import EQUIL_GHOSTS
    from cufep.workflow import ARCHIVE_RECORD, archive_leg

    leg_dir = tmp_path / "leg"
    leg_dir.mkdir()
    for name, text in (("schedule.yaml", "windows"), (EQUIL_GHOSTS, "{}"), ("equil.npz", "the pre-equilibration")):
        (leg_dir / name).write_text(text)
    stage = leg_dir / ".previous_1.partial"                          # an archive begun without it
    stage.mkdir()
    (stage / ARCHIVE_RECORD).write_text(json.dumps({"pre_equilibration": False}))
    real = PathClass.rename

    def stopping(self, target):
        if PathClass(self).name == EQUIL_GHOSTS:
            raise KeyboardInterrupt("stopped after equil.npz was moved")
        return real(self, target)

    with monkeypatch.context() as patch:
        patch.setattr(PathClass, "rename", stopping)
        with pytest.raises(KeyboardInterrupt):
            archive_leg(leg_dir, pre_equilibration=True)              # the pre-equilibration cannot be used now
    assert (stage / "equil.npz").exists() and (leg_dir / EQUIL_GHOSTS).exists()
    done = archive_leg(leg_dir)                                       # the next call: equil.npz is no longer there
    assert (done / EQUIL_GHOSTS).exists() and (done / "equil.npz").exists() and not (leg_dir / EQUIL_GHOSTS).exists()
    # (eighth review, P3) the state 9d48578 could leave: recorded without the pre-equilibration, which that version
    # moved all the same; equil.npz in the archive alone, the record of its ghosts in the leg
    for name in ("schedule.yaml", EQUIL_GHOSTS):
        (leg_dir / name).write_text("{}")
    stage = leg_dir / ".previous_2.partial"
    stage.mkdir()
    (stage / ARCHIVE_RECORD).write_text(json.dumps({"pre_equilibration": False}))
    (stage / "equil.npz").write_text("the pre-equilibration moved")
    done = archive_leg(leg_dir)
    assert done.name == "previous_2" and (done / EQUIL_GHOSTS).exists() and not (leg_dir / EQUIL_GHOSTS).exists()


def test_both_legs_of_an_edge_that_changes_the_charge_turn_their_water_into_one_ion(tmp_path, monkeypatch):
    """Eleventh review of PR #19 (P1): each leg chose the ion of its own system for the alchemical water, a K+ that the
    complex holds and the Na+ of the force field in the solvent, whose ddG then held the difference of their
    hydration. The ion of the edge is chosen in the complex leg and taken by the solvent leg; a prepared edge whose
    legs have other ions is refused by the run and left out by the analysis."""
    from cufep.rbfe import charge
    from cufep.rbfe.prep import earlier_rules_of, prepare_edge, prepared_edge
    from cufep.workflow import WorkflowError

    real = charge.ion_parameters

    def of_the_system(system, topology, sign, forcefield):
        """K+ where the system holds the protein (as a crystal K+ would be), Na+ of the force field elsewhere."""
        ion = real(system, topology, sign, forcefield)
        if any(r.name == "ALA" for r in topology.residues()):
            return {**ion, "name": "K", "sigma_nm": 0.4, "epsilon_kj_mol": 0.5}
        return ion

    monkeypatch.setattr(charge, "ion_parameters", of_the_system)
    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir, "[O-]c1ccccc1", "phenolate", 7)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenolate"
    info = prepare_edge(rcfg, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)
    ions = {leg: info["legs"][leg]["alchemical_water"]["ion"] for leg in ("complex", "solvent")}
    assert ions["complex"]["name"] == "K" and ions["solvent"] == ions["complex"]
    assert earlier_rules_of(info) == []
    # a record of an earlier version, each leg with its own ion
    info["legs"]["solvent"]["alchemical_water"]["ion"] = {**ions["solvent"], "name": "NA", "sigma_nm": 0.2439}
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(info))
    assert any("other ions" in why for why in earlier_rules_of(info))
    with pytest.raises(WorkflowError, match="other ions"):
        prepared_edge(rcfg, ligands[0], ligands[1], edge_dir, "cube")
