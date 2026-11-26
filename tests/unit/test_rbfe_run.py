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


def _usable(made_from: dict, water: bool = False, sizes=(15, 13)) -> dict:
    """A record of a prepared edge of toluene or phenol(ate) that the checks of a record pass: both legs, each mapping
    the first ring carbon (thirteenth review of PR #19: a record without legs was taken), with the alchemical water of
    an edge that changes the net charge (``water``; fourteenth review), its atoms within the end states of ``sizes``
    (the numbers of atoms of A and B)."""
    ion = {"name": "NA", "charge": 1.0, "sigma_nm": 0.2439, "epsilon_kj_mol": 0.3658}
    n_a, n_b = sizes
    leg = {"mapping": {"1": 1}, "n_atoms_a": n_a, "n_atoms_b": n_b,
           **({"alchemical_water": {"atoms_a": [n_a - 3, n_a - 2, n_a - 1], "atoms_b": [n_b - 3, n_b - 2, n_b - 1],
                                    "ion": ion}} if water else {})}
    return {"made_from": made_from, "legs": {name: dict(leg) for name in ("complex", "solvent")}}


def _prepared_with_files(prep_dir, made_from: dict, files_from, water: bool = False) -> None:
    """A preparation of an edge that differs from a valid one (``files_from``: the prep/ of a prepared edge, its end
    states) only in its record: the prepared systems of that one, and a record (:func:`_usable`) made from
    ``made_from`` with the numbers of atoms of those systems (sixteenth review of PR #19: records without the files
    were refused for the missing files before the rule a test checks was reached)."""
    import shutil

    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    for leg in ("complex", "solvent"):
        for end in ("a", "b"):
            for suffix in ("_system.xml", ".npz", "_topology.json"):
                shutil.copy(Path(files_from) / f"{leg}_{end}{suffix}", prep_dir / f"{leg}_{end}{suffix}")
    real = json.loads((Path(files_from) / "prep.json").read_text())["legs"]["complex"]
    (prep_dir / "prep.json").write_text(json.dumps(_usable(made_from, water,
                                                           (real["n_atoms_a"], real["n_atoms_b"]))))


def _of_the_record(result: dict, prep_dir) -> dict:
    """A leg result of the prepared systems of the record in ``prep_dir`` (the digest a leg records of what it was
    made from; fourteenth review of PR #19)."""
    from cufep.workflow import record_digest

    made = json.loads((Path(prep_dir) / "prep.json").read_text()).get("made_from")
    return {**result, "prepared_systems_sha256": record_digest(made)}


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
    # (seventeenth review) the windows of the solvent leg are those of this run where the complex leg takes them
    token = "this run" if (mode, legs) == ("adaptive", "solvent") else None
    jobs = [EdgeLegJob(edge="toluene~phenol", leg=leg, repeat=0, edge_dir=str(edge_dir), devices=[None],
                         config=rcfg.model_dump(mode="json"), production_ns=rcfg.sampling.production_ns, inline=True,
                         windows_token=token)
            for leg in ("solvent", "complex")]
    results = [run_hybrid_leg_safe(job) for job in jobs]
    assert all("error" not in r for r in results), results
    solvent, complex_ = results
    if token is not None:
        from cufep.rbfe.leg import WINDOWS_OF_THE_RUN

        assert json.loads((edge_dir / "repeat_0" / "solvent" / WINDOWS_OF_THE_RUN).read_text()) == {"run": token}
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


def test_every_hydrogen_of_the_hybrid_is_one_the_plugin_constrains(tmp_path):
    """The owner, 2026-11-26: six edges of TYK2 and A2A ran on stock OpenMM, as the integrator of the cuPotential
    plugin constrains three hydrogens of one length on an atom at most (SHAKE) and their hybrids put more on one atom:
    a methyl group turned between the poses (Kartograf maps atoms within 0.95 A only), and atoms of other elements
    mapped onto each other with their hydrogens (CH3 and NH2, CH2 and NH). Their hydrogens are mapped onto each other
    as LOMAP does; such a pair is not mapped, as Kartograf's filter_element_changes leaves it, and Kartograf keeps
    the largest connected part (the methyl beyond the CH2 too). Kartograf's mapping of them has hydrogens the plugin
    cannot constrain, the mapping the preparation keeps has none; an edge that needs neither keeps Kartograf's."""
    import openmm as mm
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from scipy.spatial.transform import Rotation

    from cufep.prep.ligand import load_ligands
    from cufep.rbfe.mapping import hydrogens_the_plugin_cannot_constrain
    from cufep.rbfe.prep import load_end_states, prepare_edge

    out_dir = tmp_path / "run"
    (out_dir / "cache").mkdir(parents=True)

    def write(directory, molecules) -> list:
        directory.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(directory / "ligands.sdf"))
        for name, mol in molecules:
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()
        ligands = load_ligands(directory / "ligands.sdf")
        for lig in ligands:
            lig.offmol.assign_partial_charges("gasteiger")
            q = np.array([float(x) for x in lig.offmol.partial_charges.m_as("elementary_charge")])
            (out_dir / "cache" / f"charges_am1bcc_{lig.digest()}.json").write_text(
                json.dumps({"charges": (q - q.mean()).tolist()}))
        return ligands

    def placed(smiles: str, seed: int):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        conf = mol.GetConformer()
        for k in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(k, (np.array(conf.GetAtomPosition(k)) + [16.0, 14.0, 16.0]).tolist())
        return mol

    toluene = placed("Cc1ccccc1", 11)            # 0 the methyl carbon, 1 the ring carbon, 7-9 its hydrogens
    turned, shifted = Chem.Mol(toluene), Chem.Mol(toluene)
    xyz = toluene.GetConformer().GetPositions()
    axis = (xyz[0] - xyz[1]) / np.linalg.norm(xyz[0] - xyz[1])
    for k, row in enumerate(Rotation.from_rotvec(np.radians(60.0) * axis).apply(xyz[7:10] - xyz[0]) + xyz[0]):
        turned.GetConformer().SetAtomPosition(7 + k, row.tolist())
    for k, row in enumerate(xyz + 0.05):
        shifted.GetConformer().SetAtomPosition(k, row.tolist())
    ethyl = placed("CCc1ccccc1", 7)              # 0 the methyl, 1 the CH2, 2-7 the ring
    amine = Chem.RWMol(ethyl)                    # the CH2 turned into an NH, in the same place
    amine.RemoveAtom(max(nb.GetIdx() for nb in amine.GetAtomWithIdx(1).GetNeighbors() if nb.GetAtomicNum() == 1))
    amine.GetAtomWithIdx(1).SetAtomicNum(7)
    amine = amine.GetMol()
    Chem.SanitizeMol(amine)
    tol, tol_turned, tol_shifted = write(tmp_path / "toluene", [("toluene", toluene), ("toluene_turned", turned),
                                                                ("toluene_shifted", shifted)])
    ethylbenzene, methylaniline = write(tmp_path / "chain", [("ethylbenzene", ethyl), ("methylaniline", amine)])
    (tmp_path / "aniline").mkdir()
    _tol, aniline = _ligands(tmp_path / "aniline", out_dir, "Nc1ccccc1", "aniline", 9)
    rcfg = _config(tmp_path, out_dir, "fixed")
    protein = _protein()
    for lig_a, lig_b, left_out, changed in ((tol, tol_turned, set(), True), (tol, aniline, {0}, True),
                                            (ethylbenzene, methylaniline, {0, 1}, True),
                                            (tol, tol_shifted, set(), False)):
        edge_dir = out_dir / "edges" / f"{lig_a.name}~{lig_b.name}"
        info = prepare_edge(rcfg, lig_a, lig_b, protein, edge_dir, "cube", print)
        kartograf = {int(i): j for i, j in info["mapping_kartograf"].items()}
        for leg in ("complex", "solvent"):
            end = load_end_states(edge_dir, leg)
            systems = [mm.XmlSerializer.deserialize(end[x]["system_xml"]) for x in ("a", "b")]
            check = [systems[0], systems[1], dict(enumerate(end["a"]["ligand"])), dict(enumerate(end["b"]["ligand"])),
                     lig_a.to_rdkit(), lig_b.to_rdkit()]
            kept = {int(i): j for i, j in info["legs"][leg]["mapping"].items()}
            assert hydrogens_the_plugin_cannot_constrain(kept, *check) == [], (lig_b.name, leg)
            assert bool(hydrogens_the_plugin_cannot_constrain(kartograf, *check)) == changed, (lig_b.name, leg)
            assert left_out <= set(kartograf) and left_out.isdisjoint(kept)
        if lig_b is tol_turned:
            assert all(info["legs"][leg]["core_atoms"] == 15 > len(kartograf) for leg in ("complex", "solvent"))
        if not changed:
            assert info["legs"]["solvent"]["mapping"] == info["mapping_kartograf"]
            assert not any("Kartograf mapped again" in n or "the closest first" in n for n in info["notes"])


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


def test_a_prepared_edge_of_another_ph_is_not_taken(tmp_path, edges_of_version_1):
    """Review of PR #17: the protonation states of a PDB without hydrogens follow thermodynamics.ph, and a prepared
    edge was taken by a run with another pH (the record of the edge did not have it)."""
    from cufep.rbfe.prep import edge_made_from, prepare_edge
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    _prepared_with_files(edge_dir / "prep", edge_made_from(rcfg, ligands[0], ligands[1], "cube"),
                         edges_of_version_1["prep"])
    other = rcfg.model_copy(update={"thermodynamics": rcfg.thermodynamics.model_copy(update={"ph": 6.0})})
    with pytest.raises(WorkflowError, match="other inputs or settings"):
        prepare_edge(other, ligands[0], ligands[1], _protein(), edge_dir, "cube", print)


def test_an_edge_prepared_by_an_earlier_rule_is_not_taken(tmp_path, edges_of_version_1):
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
        _prepared_with_files(edge_dir / "prep", earlier, edges_of_version_1["prep"], key == "alchemical_water_rule")
        before = (edge_dir / "prep" / "prep.json").read_text()
        with pytest.raises(WorkflowError, match=f"{key}.*an earlier rule"):
            prepared_edge(cfg, *ligands, edge_dir, "cube")
        assert (edge_dir / "prep" / "prep.json").read_text() == before
        _prepared_with_files(edge_dir / "prep", record, edges_of_version_1["prep"], key == "alchemical_water_rule")
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


def test_a_run_refused_for_its_prepared_edges_keeps_the_record_of_the_run_before(tmp_path, monkeypatch,
                                                                                 edges_of_version_1):
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
    _prepared_with_files(edge_dir / "prep", edge_made_from(rcfg, ligands[0], ligands[1], "cube"),
                         edges_of_version_1["prep"])
    before = {name: (out_dir / name).read_bytes() for name in ("resolved_config.yaml", "network.json")}
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    other = RbfeConfig.model_validate({**rcfg.model_dump(mode="json"), "dynamics": {"cutoff_nm": 0.65}})
    with pytest.raises(WorkflowError, match="other inputs or settings"):
        run_rbfe(other, log=lambda line: None)
    assert all((out_dir / name).read_bytes() == data for name, data in before.items())


def test_the_legs_of_an_edge_start_when_it_is_prepared_while_the_next_edges_are_prepared(tmp_path, monkeypatch):
    """The owner, 2026-11-26: the run prepared all 37 edges of A2A (two minutes each in a bilayer) before it started
    the first job. On GPUs the legs of an edge start as soon as it is prepared, the record of the run written before;
    an edge whose preparation stops after that is a failure of the run and the others go on; one that stops before
    any job stops the run, the record of the directory as it was."""
    import threading
    import time

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import run as run_module
    from cufep.config.auto import AutoReport
    from cufep.rbfe import prep as prep_module
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.run import run_rbfe
    from cufep.workflow import WorkflowError

    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, smiles in (("toluene", "Cc1ccccc1"), ("phenol", "Oc1ccccc1"), ("aniline", "Nc1ccccc1")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    monkeypatch.setattr(run_module, "write_timing", lambda *args, **kwargs: None)

    def run(out_dir, stops=()):
        data = _config(tmp_path, out_dir, "fixed").model_dump(mode="json")
        rcfg = RbfeConfig.model_validate({**data, "resources": {**data["resources"], "gpus": [0]},
                                          "network": {"edges": [["toluene", "phenol"], ["phenol", "aniline"]]}})
        events, first_jobs = [], threading.Event()

        def prepare(cfg, a, b, protein, edge_dir, box_shape, log):
            name = f"{a.name}~{b.name}"
            if name == "phenol~aniline":
                events.append(("waited for the jobs of the first edge", first_jobs.wait(30)))
            if name in stops:
                raise WorkflowError(f"no alchemical water for {name}")
            events.append(("prepared", name, (out_dir / "network.json").exists()))
            return {"made_from": {"ligand_a": a.name}, "legs": {"solvent": {"n_atoms_a": 15},
                                                                 "complex": {"n_atoms_a": 15}}}

        def execute(jobs, gpus, factor, log, more=None):
            events.append(("jobs", [job.edge for job in jobs], (out_dir / "network.json").exists()))
            first_jobs.set()
            coming = True
            while coming:
                new, coming = more()
                if new:
                    events.append(("jobs", [job.edge for job in new]))
                    jobs.extend(new)
                time.sleep(0.01)
            return [], None, False

        monkeypatch.setattr(prep_module, "prepare_edge", prepare)
        monkeypatch.setattr(run_module, "execute_jobs", execute)
        return run_rbfe(rcfg, log=lambda line: None), events

    summary, events = run(tmp_path / "run")
    assert events == [("prepared", "toluene~phenol", False), ("jobs", ["toluene~phenol"] * 2, True),
                      ("waited for the jobs of the first edge", True), ("prepared", "phenol~aniline", True),
                      ("jobs", ["phenol~aniline"] * 2)]
    assert summary["failed_in_this_run"] == []
    summary, events = run(tmp_path / "later", stops=("phenol~aniline",))
    assert summary["failed_in_this_run"] == ["phenol~aniline: its preparation stopped: no alchemical water for "
                                             "phenol~aniline"]
    assert events[-1] == ("waited for the jobs of the first edge", True)
    with pytest.raises(WorkflowError, match="no alchemical water for toluene~phenol"):
        run(tmp_path / "first", stops=("toluene~phenol",))
    assert not (tmp_path / "first" / "network.json").exists()


def test_edges_that_the_network_combines_are_checked_before_the_run_writes(tmp_path, monkeypatch, edges_of_version_1):
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
    _prepared_with_files(done, edge_made_from(rcfg, ligands["toluene"], ligands["phenol"], "cube"),
                         edges_of_version_1["prep"])
    (tmp_path / "protein.pdb").write_text("REMARK another protein\n")          # the receptor changes
    before = {name: (out_dir / name).read_bytes() for name in ("resolved_config.yaml", "network.json")}
    protein = _protein()
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))

    def not_to_be_reached(*args, **kwargs):
        raise AssertionError("an edge was prepared")

    monkeypatch.setattr(prep_module, "prepare_edge", not_to_be_reached)
    # refused for its receptor, not for anything else (sixteenth review of PR #19: the edge name alone also matched a
    # refusal for missing files)
    with pytest.raises(WorkflowError, match=r"toluene~phenol/prep were made from other inputs or settings \("
                                            r"protein_sha256\)"):
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
        _write_legs(root, "toluene~phenol", 0, _of_the_record(complex_, prepared), _of_the_record(solvent, prepared))
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

    from cufep.workflow import record_digest

    for edge, (prepared, c, s) in legs.items():
        shutil.copytree(prepared, root / "edges" / edge / "prep")
        # legs of this preparation: the digest of what its record was made from (fourteenth review of PR #19); a
        # record that cannot be read leaves the legs as they are
        try:
            made = record_digest(json.loads((Path(prepared) / "prep.json").read_text()).get("made_from"))
            c, s = {**c, "prepared_systems_sha256": made}, {**s, "prepared_systems_sha256": made}
        except ValueError:
            pass
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
    assert row["ddg"] == "" and "+0.91" not in text and "legs are left out of" in text
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
    prep = edges_of_version_1["changed"]
    _write_legs(root, "toluene~phenol", 0, _of_the_record(_version_1(-12.0), prep),
                _of_the_record(_leg_result(-10.5), prep))
    _write_legs(root, "toluene~phenol", 1, _of_the_record(_leg_result(-11.9), prep),
                _of_the_record(_leg_result(-10.4), prep))
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
    # (eleventh review) a record of the ghosts that cannot be read stopped every run of the leg with an error of JSON:
    # a leg that goes on from its samples does not read it; one that would start from its pre-equilibration is
    # refused, and --reschedule sets both aside and computes the leg anew
    for text in ("", "{}"):
        broken = copy("broken" + str(len(text)))
        (leg_dir(broken) / "equil_ghosts.json").write_text(text)
        checked(broken)
        again = run_hybrid_leg_safe(job(broken))
        assert "error" not in again and again["delta_g"] == first["delta_g"]
    shutil.rmtree(leg_dir(broken) / "hrex")
    (leg_dir(broken) / "result.json").unlink()
    with pytest.raises(WorkflowError, match="record of ghost waters .* cannot be read"):
        checked(broken)
    checked(broken, reschedule=True)
    anew = run_hybrid_leg_safe(job(broken, reschedule=True))
    assert "error" not in anew and json.loads((leg_dir(broken) / "equil_ghosts.json").read_text())["ghosts"]
    assert (leg_dir(broken) / "previous_1" / "equil_ghosts.json").read_text() == "{}"
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
        identities.extend(job.identity() for job in jobs)
        tokens.extend((job.leg, job.windows_token) for job in jobs)
        return [], None, False

    identities, tokens = [], []

    protein = _protein()
    monkeypatch.setattr(run_module, "execute_jobs", no_jobs)
    monkeypatch.setattr(run_module, "write_timing", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "prepare_protein_and_settings", lambda cfg, pdb, log: (cfg, AutoReport(), protein))
    with pytest.raises(WorkflowError, match="--repeat: 2 is not a repeat of the run"):
        run_rbfe(rcfg, log=lambda line: None, only_repeats=[2])
    assert not (out_dir / "network.json").exists()
    run_rbfe(rcfg, log=lambda line: None, only_repeats=[1], only_legs=["complex"])
    assert seen == [("toluene~phenol", 1, "complex")]
    # the job names the prepared systems it runs on, for the marker of its start and its error (sixteenth review)
    from cufep.workflow import record_digest

    made_from = json.loads((out_dir / "edges" / "toluene~phenol" / "prep" / "prep.json").read_text())["made_from"]
    assert identities == [{"edge": "toluene~phenol", "leg": "complex", "repeat": 1,
                           "prepared_systems_sha256": record_digest(made_from)}]
    # (seventeenth review) the windows of the solvent legs chosen anew beside the complex legs that take them: both
    # jobs of a run carry its token; a run that keeps the windows of the solvent legs gives none
    data = rcfg.model_dump(mode="json")
    taking = rcfg.model_validate({**data, "lambda_schedule": {**data["lambda_schedule"], "mode": "adaptive",
                                                              "adaptive": {**data["lambda_schedule"]["adaptive"],
                                                                           "legs": "solvent"}}})
    tokens.clear()
    run_rbfe(taking, reschedule=True, log=lambda line: None, only_repeats=[1])
    assert [leg for leg, _token in tokens] == ["solvent", "complex"]
    assert tokens[0][1] is not None and tokens[0][1] == tokens[1][1]
    tokens.clear()
    run_rbfe(taking, log=lambda line: None, only_repeats=[1])
    assert tokens == [("solvent", None), ("complex", None)]


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
    # (eleventh review, P3) an archive that keeps the pre-equilibration in the leg has a copy of it and of the record
    # of its ghosts: it can be used as a leg again (the record was left out)
    for name, text in (("schedule.yaml", "windows"), (EQUIL_GHOSTS, '{"ghosts": [3]}'), ("equil.npz", "kept")):
        (leg_dir / name).write_text(text)
    done = archive_leg(leg_dir)
    assert done == leg_dir / "previous_3"
    assert sorted(p.name for p in done.iterdir()) == sorted(["schedule.yaml", "equil.npz", EQUIL_GHOSTS])
    assert all((d / EQUIL_GHOSTS).read_text() == '{"ghosts": [3]}' for d in (leg_dir, done))
    assert (leg_dir / "equil.npz").exists()


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


def test_a_run_names_every_edge_it_does_not_take_and_how_to_prepare_them_again(tmp_path, monkeypatch,
                                                                               edges_of_version_1):
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
        _prepared_with_files(done, edge_made_from(rcfg, ligands[a], ligands[b], "cube"), edges_of_version_1["prep"])
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
    # (twelfth review, P2) the pre-equilibration lost (a backup without equil.npz): the steps of an iteration are
    # compared at the time step the samples record, before the run writes anything (it was not compared)
    lost = tmp_path / "lost"
    shutil.copytree(edge_dir.parent, lost / "edges")
    (lost / "edges" / "toluene~phenol" / "repeat_0" / "complex" / "equil.npz").unlink()
    with pytest.raises(WorkflowError, match="steps_per_iteration"):
        check_samples_of_the_run(longer, lost, [("toluene", "phenol")], [0], None, False)
    check_samples_of_the_run(rcfg, lost, [("toluene", "phenol")], [0], None, False)
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
    molecules = (ligands[0], ligands[1])
    assert earlier_rules_of(info, molecules) == []
    # a record of an earlier version, each leg with its own ion
    info["legs"]["solvent"]["alchemical_water"]["ion"] = {**ions["solvent"], "name": "NA", "sigma_nm": 0.2439}
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(info))
    assert any("other ions" in why for why in earlier_rules_of(info, molecules))
    with pytest.raises(WorkflowError, match="other ions"):
        prepared_edge(rcfg, ligands[0], ligands[1], edge_dir, "cube")


def test_the_plan_reads_the_states_as_the_run_does(tmp_path):
    """Eleventh review of PR #19 (P3): `cufep rbfe plan` gave the mean of the experimental values of the states as the
    reference, where the run takes the compounds, and did not read system.states, which the run refuses when it does
    not name a ligand. The plan reads and checks it as the run does."""
    from click.testing import CliRunner

    from cufep import cli

    out_dir = tmp_path / "run"
    _ligands(tmp_path, out_dir)
    (tmp_path / "protein.pdb").write_text("REMARK built in the test\n")
    (tmp_path / "states.csv").write_text("ligand,compound\ntoluene,T\nphenol,P\n")
    (tmp_path / "config.yaml").write_text("system: {protein: protein.pdb, ligands: ligands.sdf, states: states.csv}\n"
                                          "network: {edges: [[toluene, phenol]]}\n")
    result = CliRunner().invoke(cli.main, ["rbfe", "plan", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0, result.output
    said = "dG of every compound from the edges, relative to the mean of the experimental values of the compounds"
    assert said in result.output
    (tmp_path / "states.csv").write_text("ligand,compound\ntoluene,T\n")
    result = CliRunner().invoke(cli.main, ["rbfe", "plan", str(tmp_path / "config.yaml")])
    assert result.exit_code != 0 and "does not name the ligands phenol" in result.output
    # (thirteenth review, P3) the list of the ligands shows the value of the compound of a state, also one the file
    # gives for a state the run leaves out, and a value of 0.0
    (tmp_path / "states.csv").write_text("ligand,compound\ntoluene,T\nphenol,P\nother,T\n")
    (tmp_path / "exp.csv").write_text("ligand,dg\nother,-10\nphenol,0.0\n")
    (tmp_path / "config.yaml").write_text("system: {protein: protein.pdb, ligands: ligands.sdf, states: states.csv, "
                                          "experiment: exp.csv}\nnetwork: {edges: [[toluene, phenol]]}\n")
    result = CliRunner().invoke(cli.main, ["rbfe", "plan", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0, result.output
    assert "ligand toluene: 15 atoms, experiment -10.00 kcal/mol (compound T)" in result.output
    assert "ligand phenol: 13 atoms, experiment 0.00 kcal/mol (compound P)" in result.output


def test_a_prepared_edge_whose_mapping_joins_mapped_atoms_or_whose_record_is_lost_is_not_taken(tmp_path,
                                                                                               edges_of_version_1):
    """Twelfth review of PR #19:
    (P1) an edge prepared before the check of the mapping, whose atoms unique to one ligand join two mapped atoms,
    was taken again by `cufep rbfe run` and `cufep rbfe analyze`: the mappings of its legs, as the hybrid is made of
    them, are checked when it is taken again and in the analysis;
    (P2) a run of another repeat prepared again an edge whose record was lost, and the analysis took the legs of the
    lost preparation with the new record: an edge whose legs hold results and whose record is missing is refused
    before anything is written;
    a custom core of version 2 of its rule is taken where both versions give the same core."""
    import shutil

    from test_rbfe_network import _leg_result, _write_legs

    from cufep.rbfe.prep import PREPARATION_RULES, edge_made_from, prepared_edge
    from cufep.rbfe.run import analyse_directory, result_lines
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = tmp_path / "bridged"
    shutil.copytree(edges_of_version_1["prep"], edge_dir / "prep")
    assert prepared_edge(rcfg, *ligands, edge_dir, "cube") is not None
    info = json.loads((edge_dir / "prep" / "prep.json").read_text())
    for leg in info["legs"].values():                    # ring carbon 3 of toluene out of the mapping, 2 and 4 in it
        assert {"2", "3", "4"} <= set(leg["mapping"])
        leg["mapping"].pop("3")
    (edge_dir / "prep" / "prep.json").write_text(json.dumps(info))
    with pytest.raises(WorkflowError, match="joins mapped atoms by atoms unique to one ligand"):
        prepared_edge(rcfg, *ligands, edge_dir, "cube")
    root = _run_dir(tmp_path / "run_bridged", {"toluene~phenol": (edge_dir / "prep", _leg_result(-12.0),
                                                                   _leg_result(-10.5))})
    summary = analyse_directory(root)
    assert "ddg" not in summary["edges"][0]
    assert any("joins mapped atoms" in why for why in summary["edges"][0]["earlier_preparation"])
    assert any(x.startswith("RESULT toluene~phenol") and "remove" in x for x in result_lines(summary))
    # the record lost, legs with results: refused, nothing written
    lost = tmp_path / "lost"
    _write_legs(tmp_path / "lost_run", "toluene~phenol", 0, _leg_result(-12.0), _leg_result(-10.5))
    shutil.copytree(tmp_path / "lost_run" / "edges" / "toluene~phenol" / "repeat_0", lost / "repeat_0")
    with pytest.raises(WorkflowError, match="is missing and its legs hold what was computed from them"):
        prepared_edge(rcfg, *ligands, lost, "cube")
    assert not (lost / "prep").exists()
    assert prepared_edge(rcfg, *ligands, tmp_path / "new", "cube") is None    # nothing computed: prepared anew
    # a custom core of an earlier version of its rule (which exists from this pull request on) is prepared again
    custom = rcfg.model_copy(update={"mapping": rcfg.mapping.model_copy(update={"core_smarts": "c1ccccc1"})})
    record = edge_made_from(custom, *ligands, "cube")
    assert record["custom_core_rule"] == PREPARATION_RULES["custom_core"] == 3
    of_rule_2 = tmp_path / "rule_2"
    _prepared_with_files(of_rule_2 / "prep", {**record, "custom_core_rule": 2}, edges_of_version_1["prep"])
    with pytest.raises(WorkflowError, match="custom_core_rule"):
        prepared_edge(custom, *ligands, of_rule_2, "cube")
    _prepared_with_files(of_rule_2 / "prep", record, edges_of_version_1["prep"])
    assert prepared_edge(custom, *ligands, of_rule_2, "cube") is not None
    # (thirteenth review, P2) the records of the ligands lost (a backup without ligand_a.json): the run takes the
    # ligands of its input, which the record was made from, and writes their records again; the analysis takes the
    # ligands of the SDF of the run where its records are missing: both judge the edge alike
    from cufep.rbfe.prep import prepare_edge

    kept = tmp_path / "kept"
    shutil.copytree(edges_of_version_1["prep"], kept / "prep")
    (kept / "prep" / "ligand_a.json").unlink()
    root = _run_dir(tmp_path / "run_kept", {"toluene~phenol": (kept / "prep", _leg_result(-12.0),
                                                                _leg_result(-10.5))})
    import yaml

    (root / "resolved_config.yaml").write_text(yaml.safe_dump({"sampling": {"repeats": 1},
                                                                "system": {"ligands": str(tmp_path / "ligands.sdf")}}))
    assert "ddg" in analyse_directory(root)["edges"][0]
    (root / "resolved_config.yaml").write_text(yaml.safe_dump({"sampling": {"repeats": 1}}))
    assert "are known neither" in analyse_directory(root)["edges"][0]["earlier_preparation"][0]
    # (fourteenth review, P2) the record of A alone is written again, with the partial charges of the preparation; the
    # record of B, sound, is left as it is (it was written again without its charges)
    from openff.toolkit import Molecule

    of_b = (kept / "prep" / "ligand_b.json").read_text()
    assert Molecule.from_json(of_b).partial_charges is not None
    from cufep.prep.ligand import load_ligands

    read_again = load_ligands(tmp_path / "ligands.sdf")              # as a run reads its SDF: without charges
    assert read_again[1].offmol.partial_charges is None
    prepare_edge(rcfg, *read_again, _protein(), kept, "cube", print)
    assert (kept / "prep" / "ligand_b.json").read_text() == of_b
    assert Molecule.from_json((kept / "prep" / "ligand_a.json").read_text()).partial_charges is not None
    # (fourteenth review, P2) REST2 on the perturbed region reads the ligands of the edge where its record of A is
    # missing: those of the input of the run, before anything is written (the check before a run stopped with
    # FileNotFoundError)
    from cufep.rbfe.leg import perturbed_region
    from cufep.rbfe.prep import load_end_states

    end = load_end_states(kept, "solvent")
    region = (end["a"], end["b"], end["core_map"],
              sorted(j for j in end["b"]["ligand"] if j not in set(end["core_map"].values())))
    hot = perturbed_region(kept, *region)
    (kept / "prep" / "ligand_a.json").unlink()
    with pytest.raises(WorkflowError, match="known neither from the records of its preparation"):
        perturbed_region(kept, *region)
    assert perturbed_region(kept, *region, inputs=ligands) == hot


def test_windows_a_leg_would_start_from_are_compared_with_the_settings_of_the_run_before_it_writes(
        tmp_path, monkeypatch, edges_of_version_1):
    """Fourteenth review of PR #19 (P2): a leg stopped before its replica exchange (its windows and their start
    configurations, no samples, no checkpoint) whose windows were made with other settings was refused in the leg,
    after the run had written its record. The windows are compared with the settings of the run, as the leg compares
    them, before anything is written, with the command that starts the leg anew; --reschedule sets them aside."""
    import shutil

    from cufep.alchemy.schedule import fixed_schedule
    from cufep.rbfe.leg import STAGES, hybrid_settings
    from cufep.rbfe.run import check_windows_of_the_run
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    edge_dir = out_dir / "edges" / "toluene~phenol"
    shutil.copytree(edges_of_version_1["prep"], edge_dir / "prep")
    rcfg = _config(tmp_path, out_dir, "fixed")
    other = rcfg.model_validate({**rcfg.model_dump(mode="json"), "sampling": {
        **rcfg.model_dump(mode="json")["sampling"], "rest2": {"scale": 2.0}}})
    leg_dir = edge_dir / "repeat_0" / "solvent"
    leg_dir.mkdir(parents=True)
    schedule = fixed_schedule(list(STAGES), {stage: 2 for stage in STAGES}, 2.0)
    schedule.save(leg_dir / "schedule.yaml")
    np.savez(leg_dir / "start_configs.npz", positions=np.zeros((len(schedule.states), 1, 3)),
             boxes=np.zeros((len(schedule.states), 3, 3)), velocities=np.zeros((len(schedule.states), 1, 3)),
             made_with=json.dumps({"states": [state.to_dict() for state in schedule.states],
                                   "settings": hybrid_settings(other, "solvent")}))
    with pytest.raises(WorkflowError, match="were made with other settings") as refused:
        check_windows_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)
    assert "--leg solvent --reschedule" in str(refused.value)
    check_windows_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], ["solvent"], True)
    check_windows_of_the_run(other, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)
    # (sixteenth review, P2) the REST2 scale of the schedule is compared whatever start configurations it has: start
    # configurations of these windows and of the settings of the run, a schedule of scale 1 (the leg refuses it)
    fixed_schedule(list(STAGES), {stage: 2 for stage in STAGES}).save(leg_dir / "schedule.yaml")
    with pytest.raises(WorkflowError, match="made with a REST2 scale of 1 and sampling.rest2.scale is 2"):
        check_windows_of_the_run(other, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)
    # (sixteenth review, P2) the windows a complex leg borrows from the solvent leg (adaptive.legs: solvent), of
    # another scale: refused before the complex leg writes them and its start configurations
    borrowing = other.model_validate({**other.model_dump(mode="json"), "lambda_schedule": {
        **other.model_dump(mode="json")["lambda_schedule"], "mode": "adaptive",
        "adaptive": {**other.model_dump(mode="json")["lambda_schedule"]["adaptive"], "legs": "solvent"}}})
    # (sixteenth review, P2) also with --reschedule of the complex leg alone, which takes the same windows again: the
    # refusal names what makes them usable (the scale they were made with, or the windows of the solvent leg anew);
    # a --reschedule of both legs chooses the windows of the solvent leg anew and passes
    for reschedule in (False, True):
        with pytest.raises(WorkflowError, match="the complex leg of toluene~phenol, repeat 0, takes the lambda "
                                                "windows of its solvent leg") as refused:
            check_windows_of_the_run(borrowing, out_dir, [("toluene", "phenol")], [0], ["complex"], reschedule)
        assert "Set sampling.rest2.scale to 1 if so they were meant" in str(refused.value)
        assert "--repeat 0 --leg solvent --reschedule); the complex leg then takes those" in str(refused.value)
        assert "--leg complex" not in str(refused.value)
    check_windows_of_the_run(borrowing, out_dir, [("toluene", "phenol")], [0], None, True)
    # (seventeenth review, P3) the scale the refusal tells is the one the windows were made with: 4/3 told as 1.33333
    # was refused again; the value of the refusal, set as it is, passes
    import re

    fixed_schedule(list(STAGES), {stage: 2 for stage in STAGES}, 4 / 3).save(leg_dir / "schedule.yaml")
    with pytest.raises(WorkflowError, match="takes the lambda windows of its solvent leg") as refused:
        check_windows_of_the_run(borrowing, out_dir, [("toluene", "phenol")], [0], ["complex"], False)
    told = re.search(r"Set sampling\.rest2\.scale to (\S+) if so they were meant", str(refused.value)).group(1)
    assert told == "1.3333333333333333"
    as_told = borrowing.model_validate({**borrowing.model_dump(mode="json"), "sampling": {
        **borrowing.model_dump(mode="json")["sampling"], "rest2": {"scale": float(told)}}})
    check_windows_of_the_run(as_told, out_dir, [("toluene", "phenol")], [0], ["complex"], False)
    schedule.save(leg_dir / "schedule.yaml")
    # (fifteenth review, P2) with samples too: the leg reads its windows whatever it goes on from
    (leg_dir / "hrex").mkdir()
    (leg_dir / "hrex" / "checkpoint.npz").write_bytes(b"")
    with pytest.raises(WorkflowError, match="were made with other settings"):
        check_windows_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)
    # start configurations that cannot be read: a record cut short, or no velocities
    for broken in ({"made_with": "{\"states\": "}, {"velocities": None}):
        arrays = {"positions": np.zeros((len(schedule.states), 1, 3)), "boxes": np.zeros((len(schedule.states), 3, 3)),
                  "velocities": np.zeros((len(schedule.states), 1, 3)),
                  "made_with": json.dumps({"states": [state.to_dict() for state in schedule.states],
                                           "settings": hybrid_settings(other, "solvent")}), **broken}
        np.savez(leg_dir / "start_configs.npz", **{k: v for k, v in arrays.items() if v is not None})
        with pytest.raises(WorkflowError, match="cannot be read"):
            check_windows_of_the_run(other, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)
    # a schedule without start configurations of another REST2 scale, and a schedule of the other leg
    (leg_dir / "start_configs.npz").unlink()
    fixed_schedule(list(STAGES), {stage: 2 for stage in STAGES}).save(leg_dir / "schedule.yaml")
    with pytest.raises(WorkflowError, match="made with a REST2 scale of 1 and sampling.rest2.scale is 2"):
        check_windows_of_the_run(other, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)
    fixed_schedule(list(STAGES), {stage: 2 for stage in STAGES}, 2.0).save(leg_dir / "schedule.yaml", leg="complex")
    with pytest.raises(WorkflowError, match="written for the complex leg"):
        check_windows_of_the_run(other, out_dir, [("toluene", "phenol")], [0], ["solvent"], False)


def test_a_complex_leg_run_beside_its_solvent_leg_chosen_anew_takes_the_windows_of_this_run(tmp_path, monkeypatch):
    """Seventeenth review of PR #19 (P2): with lambda_schedule.adaptive.legs: solvent and --reschedule of both legs
    run side by side, the complex leg read the windows the solvent leg had (REST2 scale 1) before the solvent leg set
    them aside, and stopped after the run had written its record (with the same scale it would have taken the windows
    of before). The complex leg takes only the windows the solvent leg publishes in this run."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import STAGES, WINDOWS_OF_THE_RUN, _schedule_from_solvent
    from cufep.workflow import archive_leg

    edge_dir = tmp_path / "edge"
    solvent = edge_dir / "repeat_0" / "solvent"
    solvent.mkdir(parents=True)

    def windows(scale: float, run: str | None = None) -> None:
        fixed_schedule(list(STAGES), {stage: 2 for stage in STAGES}, scale).save(solvent / "schedule.yaml")
        np.savez(solvent / "start_configs.npz", positions=np.zeros((1, 1, 3)))
        if run is not None:
            (solvent / WINDOWS_OF_THE_RUN).write_text(json.dumps({"run": run}))

    windows(1.0, "an earlier run")
    waited = []

    def the_solvent_leg_of_this_run(seconds):        # sets its windows aside and publishes new ones meanwhile
        waited.append(seconds)
        archive_leg(solvent)
        windows(2.0, "this run")

    monkeypatch.setattr(leg_module.time, "sleep", the_solvent_leg_of_this_run)
    assert _schedule_from_solvent(None, edge_dir, 0, lambda line: None, None, "this run").rest2_scale == 2.0
    assert len(waited) == 1 and (solvent / "previous_1" / "schedule.yaml").exists()
    # a run that keeps the windows of the solvent leg (the complex leg alone) takes those it has
    windows(1.0)
    assert _schedule_from_solvent(None, edge_dir, 0, lambda line: None).rest2_scale == 1.0
    assert len(waited) == 1


def test_a_mapping_that_the_rule_of_constrained_bonds_makes_join_mapped_atoms_is_refused_before_writing(tmp_path):
    """Twelfth review of PR #19 (P1): oxirane -> ethane with hydrogens mapped onto heavy atoms (Kartograf's settings
    of the review, 1.3 A): the oxygen, mapped onto a hydrogen of ethane, leaves the mapping by the rule of constrained
    bonds (the C-H of ethane is constrained, the C-O of oxirane not) and stays bonded to both carbons, which the hybrid
    then holds at both end states. The final mapping of each leg is checked: refused before anything of the edge is
    written; the network leaves such a mapping out."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    from cufep.prep.ligand import load_ligands
    from cufep.rbfe.mapping import (
        MappingSettings,
        WithoutBridges,
        _component,
        _mapper,
        bridging_dummies,
        kartograf_mapping,
        without_mixed_hydrogens,
    )
    from cufep.rbfe.prep import prepare_edge
    from cufep.workflow import WorkflowError

    out_dir = tmp_path / "run"
    centre = np.array([16.0, 14.0, 16.0])
    coordinates = {"C1CO1": [(0.75, 0, 0), (-0.75, 0, 0), (0, 1.22, 0), (1.22, -0.47, 0.88), (1.22, -0.47, -0.88),
                             (-1.22, -0.47, 0.88), (-1.22, -0.47, -0.88)],
                   "CC": [(0.77, 0, 0), (-0.77, 0, 0), (1.13, -0.515, 0.892), (1.13, -0.515, -0.892), (1.13, 1.03, 0),
                          (-1.13, -0.515, 0.892), (-1.13, -0.515, -0.892), (-1.13, 1.03, 0)]}
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, smiles in (("oxirane", "C1CO1"), ("ethane", "CC")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))         # heavy atoms first, then the hydrogens of each in order
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(coordinates[smiles]):
            conf.SetAtomPosition(i, Point3D(*(np.array(xyz) + centre)))
        mol.AddConformer(conf, assignId=True)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    oxirane, ethane = load_ligands(tmp_path / "ligands.sdf")
    (out_dir / "cache").mkdir(parents=True)
    for lig in (oxirane, ethane):
        lig.offmol.assign_partial_charges("gasteiger")
        charges = [float(q) for q in lig.offmol.partial_charges.m_as("elementary_charge")]
        (out_dir / "cache" / f"charges_am1bcc_{lig.digest()}.json").write_text(
            json.dumps({"charges": [q - sum(charges) / len(charges) for q in charges]}))
    loose = {"map_hydrogens_on_hydrogens_only": False, "map_exact_ring_matches_only": False, "allow_bond_breaks": True,
             "atom_max_distance_A": 1.3}
    base = _config(tmp_path, out_dir, "fixed")
    rcfg = base.model_validate({**base.model_dump(mode="json"), "mapping": loose})
    settings = MappingSettings(**{k.removesuffix("_A"): v for k, v in loose.items()})
    mapping = kartograf_mapping(oxirane, ethane, settings)
    assert ethane.offmol.atoms[mapping[2]].atomic_number == 1                 # the oxygen onto a hydrogen
    final = without_mixed_hydrogens(oxirane.to_rdkit(), ethane.to_rdkit(), mapping)
    assert 2 not in final and bridging_dummies(oxirane.to_rdkit(), ethane.to_rdkit(), final)
    edge_dir = out_dir / "edges" / "oxirane~ethane"
    with pytest.raises(WorkflowError, match=r"the atoms \[2\] unique to A are bonded to the mapped atoms \[0, 1\]"):
        prepare_edge(rcfg, oxirane, ethane, _protein(), edge_dir, "cube", print)
    assert not edge_dir.exists()
    a, b = _component(oxirane), _component(ethane)
    assert list(WithoutBridges(_mapper(settings), True).suggest_mappings(a, b)) == []
    assert list(WithoutBridges(_mapper(settings), False).suggest_mappings(a, b))
    # (thirteenth review, P2) `cufep rbfe plan` of the edge given in network.edges: refused as the run refuses it with
    # the bonds of hydrogens constrained, shown without constraints, where the mapping keeps the oxygen
    import yaml
    from click.testing import CliRunner

    from cufep import cli

    for constraints, refused in (("hbonds", True), ("none", False)):
        config = {**rcfg.model_dump(mode="json"), "network": {"edges": [["oxirane", "ethane"]]},
                  "dynamics": {**rcfg.model_dump(mode="json")["dynamics"], "constraints": constraints}}
        (tmp_path / "plan.yaml").write_text(yaml.safe_dump(config))
        result = CliRunner().invoke(cli.main, ["rbfe", "plan", str(tmp_path / "plan.yaml")])
        assert (result.exit_code != 0) == refused, result.output
        assert ("unique to A are bonded to the mapped atoms [0, 1]" in result.output) == refused, result.output
    # (thirteenth review, P2) without constraints the oxygen stays mapped onto the hydrogen of ethane: the heavy atoms
    # of the hybrid ligand were those of the elements of A (C, C, O one way, C, C the other), and one configuration had
    # two spheres. They are the atoms heavy in either end: one centre, one sphere both ways
    import openmm as mm

    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import (
        centre_positions_of,
        end_states_heavy,
        hybrid_heavy_atoms,
        hybrid_topology,
        start_moves,
    )
    from cufep.rbfe.prep import load_end_states
    from cufep.sampling.gcmc import GcmcMover

    free = rcfg.model_validate({**rcfg.model_dump(mode="json"),
                                "dynamics": {**rcfg.model_dump(mode="json")["dynamics"], "constraints": "none"}})
    prepare_edge(free, oxirane, ethane, _protein(), edge_dir, "cube", print)
    end = load_end_states(edge_dir, "complex")
    a, b, core = end["a"], end["b"], end["core_map"]
    assert ethane.offmol.atoms[{int(i): j for i, j in end["info"]["legs"]["complex"]["mapping"].items()}[2]
                               ].atomic_number == 1                     # O of A onto H of B, mapped
    ways = []
    for first, second, mapping in ((a, b, core), (b, a, {j: i for i, j in core.items()})):
        hybrid = make_hybrid_system(mm.XmlSerializer.deserialize(first["system_xml"]), first["positions"],
                                    first["ligand"], mm.XmlSerializer.deserialize(second["system_xml"]),
                                    second["positions"], second["ligand"], mapping)
        unique = sorted(j for j in second["ligand"] if j not in set(mapping.values()))
        top = hybrid_topology(first["topology"], second["topology"], unique)
        box = np.asarray(first["box"], dtype=float)
        moves = start_moves(free, hybrid, top, box, end_states_heavy(first, second),
                            centre_positions_of(hybrid, first, second),
                            heavy_atoms=hybrid_heavy_atoms(hybrid, first, second))
        ways.append((hybrid, moves, box))
    (forward, f_moves, box), (backward, b_moves, _box) = ways
    x_f = np.asarray(forward.positions, dtype=float)
    x_b = np.array(backward.positions, dtype=float)
    for i, h in forward.a_to_hybrid.items():
        x_b[backward.b_to_hybrid[i]] = x_f[h]
    for j, h in forward.b_to_hybrid.items():
        x_b[backward.a_to_hybrid[j]] = x_f[h]
    f_mover, b_mover = GcmcMover(f_moves, None, 1), GcmcMover(b_moves, None, 1)
    assert np.allclose(f_mover.centre(x_f, box), b_mover.centre(x_b, box), rtol=0.0, atol=1e-9)
    on = np.ones(len(f_moves.waters), dtype=bool)
    assert list(f_mover.in_sphere(x_f, box, on)) == list(b_mover.in_sphere(x_b, box, on))


def test_the_plan_gives_the_mapping_of_kartograf_and_the_one_the_preparation_leaves(tmp_path):
    """Fourteenth review of PR #19 (P3): methanol -> methane, the carbon and three hydrogens on each other and the
    oxygen on the fourth hydrogen of methane: Kartograf maps 5 atoms, and the rule of the constrained bonds takes the
    oxygen out (its dummy joins the carbon alone: no bridge). The plan gave 5 mapped and 1/0 unique; it gives the
    mapping of Kartograf and, apart, the most the preparation leaves."""
    import yaml
    from click.testing import CliRunner
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    from cufep import cli

    centre = np.array([16.0, 14.0, 16.0])
    tetra = [(1.03, 0.0, -0.36), (-0.51, 0.89, -0.36), (-0.51, -0.89, -0.36)]
    writer = Chem.SDWriter(str(tmp_path / "ligands.sdf"))
    for name, smiles, rest in (("methanol", "CO", [(0, 0, 1.43), (0.0, 0.9, 1.75)]),
                               ("methane", "C", [(0, 0, 1.09)])):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        hydrogens_of_c = [n.GetIdx() for n in mol.GetAtomWithIdx(0).GetNeighbors() if n.GetAtomicNum() == 1]
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(0, Point3D(*centre))
        for i, xyz in zip(hydrogens_of_c, tetra + ([rest[0]] if name == "methane" else [])):
            conf.SetAtomPosition(i, Point3D(*(np.array(xyz) + centre)))
        if name == "methanol":
            conf.SetAtomPosition(heavy[1], Point3D(*(np.array(rest[0]) + centre)))
            h_of_o = next(n.GetIdx() for n in mol.GetAtomWithIdx(heavy[1]).GetNeighbors() if n.GetAtomicNum() == 1)
            conf.SetAtomPosition(h_of_o, Point3D(*(np.array(rest[1]) + centre)))
        mol.AddConformer(conf, assignId=True)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    (tmp_path / "protein.pdb").write_text("REMARK built in the test\n")
    config = {"system": {"protein": "protein.pdb", "ligands": "ligands.sdf"},
              "network": {"edges": [["methanol", "methane"]]},
              "mapping": {"map_hydrogens_on_hydrogens_only": False}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(cli.main, ["rbfe", "plan", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0, result.output
    # (fifteenth review, P3) no count of the preparation, which its systems decide: the pairs the rule can take out
    assert ("edge methanol -> methane: 5 atoms mapped by Kartograf, 1 unique to methanol, 0 unique to methane; "
            "the rule of the constrained bonds can take pairs of it out when the systems are built (1 pair(s) of a "
            "hydrogen and a heavy atom)" in result.output), result.output
    assert "at most" not in result.output


def test_a_leg_this_version_cannot_continue_is_refused_before_the_run_writes(tmp_path, monkeypatch,
                                                                            edges_of_version_1):
    """Fifteenth and sixteenth reviews of PR #19 (P2): a leg whose result is of other prepared systems than the record
    of its edge (a prep/ of another receptor restored), or of version 1 of the hybrid (a REST2 leg of main), was run
    again: the run set its result aside, and the leg stopped on the System of its samples. A result of other prepared
    systems is refused with or without --reschedule (which keeps the pre-equilibration made from them), with the way
    on that recovers it; a leg that goes on from samples of another version, or without a result yet (stopped after
    the checkpoint of its start), is continued where its samples are of the System this version makes of it (a leg of
    main without REST2) and refused, with --reschedule, where they are not; an edge whose prepared systems are
    incomplete is refused too."""
    import hashlib
    import shutil

    from test_rbfe_network import _leg_result, _write_legs

    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.leg import build_leg
    from cufep.rbfe.prep import prepared_edge
    from cufep.rbfe.run import check_results_of_the_run
    from cufep.workflow import WorkflowError, record_digest

    out_dir = tmp_path / "run"
    ligands = _ligands(tmp_path, out_dir)
    rcfg = _config(tmp_path, out_dir, "fixed")
    edge_dir = out_dir / "edges" / "toluene~phenol"
    shutil.copytree(edges_of_version_1["prep"], edge_dir / "prep")
    made = record_digest(json.loads((edge_dir / "prep" / "prep.json").read_text())["made_from"])
    current = {**_leg_result(-12.0), "prepared_systems_sha256": made}
    _write_legs(out_dir, "toluene~phenol", 0, current, current)

    def check(reschedule=False):
        check_results_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], ["solvent"], reschedule)

    check()
    leg_dir = edge_dir / "repeat_0" / "solvent"
    result = leg_dir / "result.json"
    before = result.read_text()
    result.write_text(json.dumps({**json.loads(before), "prepared_systems_sha256": "another receptor"}))
    for reschedule in (False, True):
        with pytest.raises(WorkflowError, match="other prepared systems") as refused:
            check(reschedule)
        assert "Restore the preparation they were computed from" in str(refused.value)
    # a result of version 1 (main), its samples of the System this version makes (no REST2: the same) or of another
    (leg_dir / "hrex").mkdir()
    (leg_dir / "hrex" / "checkpoint.npz").write_bytes(b"")
    same = hashlib.sha256(build_leg(rcfg, edge_dir, "solvent").xml.encode()).hexdigest()
    result.write_text(json.dumps({**json.loads(before), "run": {}}))
    for digest, refused_too in ((same, False), ("0" * 64, True)):
        monkeypatch.setattr(leg_module, "_samples_system_digest", lambda leg_dir, digest=digest: digest)
        if refused_too:
            with pytest.raises(WorkflowError, match="another System than this version makes") as refused:
                check()
            assert "--repeat 0 --leg solvent --reschedule" in str(refused.value)
            check(True)
        else:
            check()
    # no result yet, the checkpoint of the start of another System: refused before anything is written
    result.unlink()
    with pytest.raises(WorkflowError, match="another System than this version makes"):
        check()
    # (sixteenth review, P2) stopped before its result, of other prepared systems by the marker of its start or by its
    # error, which the job of this version writes: refused with or without --reschedule (which would keep the
    # pre-equilibration made from them), with the way on; of the same prepared systems, the System of its samples
    # decides as above
    from cufep.rbfe.leg import EdgeLegJob
    from cufep.workflow import STARTED_FILE, begin_leg, write_leg_error

    job = EdgeLegJob(edge="toluene~phenol", leg="solvent", repeat=0, edge_dir=str(edge_dir), devices=[None],
                     config={}, production_ns=0.005, prepared_systems_sha256="another receptor")
    begin_leg(leg_dir, job)
    assert json.loads((leg_dir / STARTED_FILE).read_text())["prepared_systems_sha256"] == "another receptor"
    for _record in ("marker", "error"):
        for reschedule in (False, True):
            with pytest.raises(WorkflowError, match="other prepared systems") as refused:
                check(reschedule)
            assert "Restore the preparation they were computed from" in str(refused.value)
        write_leg_error(leg_dir, "stopped", job)
        assert not (leg_dir / STARTED_FILE).exists()
    (leg_dir / "error.json").unlink()
    begin_leg(leg_dir, EdgeLegJob(**{**job.__dict__, "prepared_systems_sha256": made}))
    with pytest.raises(WorkflowError, match="another System than this version makes"):
        check()
    # (seventeenth review, P2) a leg of main stopped before its result (its marker names no prepared systems), its
    # samples of another System and its pre-equilibration of another number of atoms (another receptor): not told
    # --reschedule, which keeps the pre-equilibration, and refused by --reschedule before anything is set aside; one
    # whose pre-equilibration fits the System of the leg is told --reschedule as above
    from cufep.rbfe.run import check_pre_equilibrations_of_the_run

    (leg_dir / STARTED_FILE).write_text(json.dumps({"edge": "toluene~phenol", "leg": "solvent", "repeat": 0}))
    n_particles = build_leg(rcfg, edge_dir, "solvent").hybrid.system.getNumParticles()
    np.savez(leg_dir / "equil.npz", positions=np.zeros((n_particles - 3, 3)))
    with pytest.raises(WorkflowError, match="other prepared systems") as refused:
        check()
    assert "--reschedule" not in str(refused.value).split("Restore the preparation")[0]
    with pytest.raises(WorkflowError, match=f"holds {n_particles - 3} atoms and the System this run makes of the "
                                            f"leg {n_particles}") as refused:
        check_pre_equilibrations_of_the_run(rcfg, out_dir, [("toluene", "phenol")], [0], ["solvent"], True)
    assert "Restore the preparation they were made from, or compute the edge in a new output.dir" in str(refused.value)
    np.savez(leg_dir / "equil.npz", positions=np.zeros((n_particles, 3)))
    with pytest.raises(WorkflowError, match="another System than this version makes") as refused:
        check()
    assert "--repeat 0 --leg solvent --reschedule" in str(refused.value)
    (leg_dir / "equil.npz").unlink()
    assert prepared_edge(rcfg, *ligands, edge_dir, "cube") is not None
    (edge_dir / "prep" / "complex_b_system.xml").unlink()
    with pytest.raises(WorkflowError, match="incomplete .complex_b_system.xml missing"):
        prepared_edge(rcfg, *ligands, edge_dir, "cube")
