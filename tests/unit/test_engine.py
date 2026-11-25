"""The engine of a leg (cufep.engine): the protocols of a leg and what their files record."""
from __future__ import annotations

import pytest
import yaml


def test_files_of_the_abfe_name_no_protocol_and_those_of_a_hybrid_leg_name_theirs(tmp_path):
    """The conditions of the samples and the schedule of an ABFE leg are what they were before the engine (samples of
    earlier runs are continued under them); those of a hybrid leg name their protocol, and a schedule file is read
    as states of the protocol of its stages."""
    from cufep.alchemy.schedule import Schedule, schedule_from_windows
    from cufep.engine.protocol import validate_states
    from cufep.rbfe.states import HybridState
    from cufep.sampling.states import AlchemicalState
    from cufep.sampling.worker import LegSpec

    abfe = schedule_from_windows(["discharge", "vanish"], {"discharge": 3, "vanish": 3}, {"discharge": 3, "vanish": 3})
    hybrid = schedule_from_windows(["morph"], {"morph": 3}, {"morph": 3}, 1.5)
    assert all(isinstance(s, AlchemicalState) for s in abfe.states)
    assert all(isinstance(s, HybridState) for s in hybrid.states) and hybrid.states[1].rest2_scale == 1.5

    def spec(schedule, **more) -> LegSpec:
        return LegSpec(name="x", system_xml="<System/>", states=[s.to_dict() for s in schedule.states], n_atoms=1,
                       **more)

    assert "protocol" not in spec(abfe).conditions()
    assert spec(hybrid, protocol="hybrid").conditions()["protocol"] == "hybrid"
    assert spec(hybrid, protocol="hybrid").lambda_states() == hybrid.states

    abfe.save(tmp_path / "abfe.yaml", leg="solvent")
    hybrid.save(tmp_path / "hybrid.yaml", leg="solvent")
    assert "protocol" not in yaml.safe_load((tmp_path / "abfe.yaml").read_text())
    assert yaml.safe_load((tmp_path / "hybrid.yaml").read_text())["protocol"] == "hybrid"
    assert Schedule.load(tmp_path / "abfe.yaml").states == abfe.states
    assert Schedule.load(tmp_path / "hybrid.yaml").states == hybrid.states
    # a schedule of a relative leg written before the engine names no protocol: its stages say which it is
    written = yaml.safe_load((tmp_path / "hybrid.yaml").read_text())
    (tmp_path / "old.yaml").write_text(yaml.safe_dump({k: v for k, v in written.items() if k != "protocol"}))
    assert Schedule.load(tmp_path / "old.yaml").states == hybrid.states
    (tmp_path / "wrong.yaml").write_text(yaml.safe_dump({**written, "protocol": "abfe"}))
    with pytest.raises(ValueError, match="not those of the protocol abfe"):
        Schedule.load(tmp_path / "wrong.yaml")
    with pytest.raises(ValueError, match="not those of one protocol"):
        validate_states([abfe.states[0], hybrid.states[-1]])


def test_the_timing_of_a_run_names_each_leg_by_its_job(tmp_path):
    """Regression: the timing of a run read the ligand of every leg from its timing file, and a relative run, whose
    legs are of edges, ended with KeyError 'ligand' after all its legs had finished (TYK2 example, image 5cc23eb)."""
    import json

    from cufep.rbfe.leg import EdgeLegJob
    from cufep.run import write_timing
    from cufep.workflow import LegJob

    jobs = [EdgeLegJob(edge="a~b", leg="complex", repeat=0, edge_dir=str(tmp_path / "a~b"), devices=[], config={},
                       production_ns=1.0),
            LegJob(ligand="c", leg="solvent", repeat=0, lig_dir=str(tmp_path / "c"), devices=[], config={},
                   production_ns=1.0)]
    for job, unit in zip(jobs, ({"edge": "a~b"}, {"ligand": "c"})):
        job.leg_dir.mkdir(parents=True)
        (job.leg_dir / "timing.json").write_text(json.dumps({
            "phases": {"sampling": 2.0}, "finished": 10.0, **unit, "leg": job.leg, "repeat": 0,
            "sampling": {"iterations": 3, "lambda_windows": 9, "seconds_per_iteration": 0.2,
                         "worker_busy_fraction": 0.5}}))

    class Monitor:
        gpus = []

        def mean(self):
            return {}

        def mean_over(self, intervals):
            return {}

    lines = []
    write_timing(tmp_path, jobs, 0.0, 1.0, Monitor(), False, lines.append)
    assert any(line.startswith("timing: a~b/r0/complex: 9 lambda windows") for line in lines)
    assert any(line.startswith("timing: c/r0/solvent: 9 lambda windows") for line in lines)


def _abfe_solvent_leg(tmp_path):
    """A solvent leg of the ABFE end to end on the CPU platform (benzene next to an alanine dipeptide, Gasteiger
    charges), adaptive lambda scheduling started by its walk."""
    import json as _json
    import random

    import numpy as np
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    from conftest import _DIPEPTIDE_BONDS, _DIPEPTIDE_HEAVY
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep.config.schema import Config
    from cufep.prep.ligand import load_ligands
    from cufep.prep.protein import PreparedProtein
    from cufep.workflow import LegJob, prepare_ligand

    random.seed(20261126)
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
    mod = app.Modeller(top, pos * u.nanometer)
    mod.addHydrogens(app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml"),
                     platform=mm.Platform.getPlatformByName("Reference"))
    protein = PreparedProtein(mod.topology, mod.positions, None, {"A": "A"})
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    xyz = mol.GetConformer().GetPositions()
    for i, row in enumerate(xyz - xyz.mean(axis=0) + np.array([14.0, 16.0, 14.0])):
        mol.GetConformer().SetAtomPosition(i, row.tolist())
    mol.SetProp("_Name", "benzene")
    w = Chem.SDWriter(str(tmp_path / "ligand.sdf"))
    w.write(mol)
    w.close()
    lig = load_ligands(tmp_path / "ligand.sdf")[0]
    out = tmp_path / "run"
    (out / "cache").mkdir(parents=True)
    (out / "cache" / f"charges_am1bcc_{lig.digest()}.json").write_text(
        _json.dumps({"charges": [-0.13] * 6 + [0.13] * 6}))
    (tmp_path / "protein.pdb").write_text("REMARK built in the test\n")
    short = {"nvt_ns": 0.002, "npt_ns": 0.002, "md_ns": 0.004}
    data = {"system": {"protein": str(tmp_path / "protein.pdb"), "ligand": str(tmp_path / "ligand.sdf")},
            "backend": {"name": "openmm", "platform": "CPU"}, "dynamics": {"cutoff_nm": 0.6},
            "solvation": {"box_shape": "cube", "padding_complex_nm": 0.8, "padding_solvent_nm": 0.8},
            "thermodynamics": {"ionic_strength_M": 0.0},
            "sampling": {"production_ns": 0.005, "equilibration_ns": 0.0025, "time_per_iteration_ps": 0.25,
                         "checkpoint_interval_ps": 1.0, "pre_equilibration": {"complex": short, "solvent": short}},
            "analysis": {"error_blocks": [2], "part_error_blocks": 2, "convergence_points": 2, "convergence_blocks": 2},
            "lambda_schedule": {"mode": {"complex": "fixed", "solvent": "adaptive"}, "adaptive": {
                "initial_windows": {"solvent": {"discharge": 2, "vanish": 2}}, "candidates": 11, "window_ps": 0.5,
                "equilibration_ps": 0.0, "sample_interval_ps": 0.1, "walk_ps": 0.2, "overlap_threshold": 0.001}},
            "restraints": {"selection": "auto"},
            "resources": {"gpus": [], "cpu_threads_per_worker": 1}, "output": {"dir": str(out)}}
    cfg = Config.model_validate(data)
    lig_dir = out / "benzene"
    prepare_ligand(cfg, lig, protein, lig_dir, False, "cube", print)

    def job(config) -> LegJob:
        return LegJob(ligand="benzene", leg="solvent", repeat=0, lig_dir=str(lig_dir), devices=[None],
                      config=config.model_dump(mode="json"), production_ns=config.sampling.production("solvent"),
                      inline=True)

    return cfg, job, lig_dir / "repeat_0" / "solvent"


def _one_start_short(leg_dir) -> None:
    import numpy as np

    with np.load(leg_dir / "start_configs.npz") as start:
        kept = {key: start[key] for key in start.files}
    np.savez(leg_dir / "start_configs.npz", **{key: (value[:-1] if key != "made_with" else value)
                                               for key, value in kept.items()})


def test_an_abfe_leg_keeps_its_windows_by_the_checks_of_the_engine(tmp_path, monkeypatch):
    """The class of the reviews of PR #18 in the ABFE: a leg whose start configurations are one window short and whose
    replica exchange goes on from its samples (or the checkpoint of their start) is checked before anything is
    written (another way of exchange: refused, files unchanged), keeps the start its samples were made from (the
    walk), and samples without a schedule are refused before new windows are written."""
    import openmm as mm

    from cufep.workflow import run_leg_safe

    cpu = mm.Platform.getPlatformByName("CPU")
    before_default = cpu.getPropertyDefaultValue("DeterministicForces")
    cpu.setPropertyDefaultValue("DeterministicForces", "true")
    try:
        cfg, job, leg_dir = _abfe_solvent_leg(tmp_path)
        first = run_leg_safe(job(cfg))
        assert "error" not in first, first
        assert first["start_configurations"].startswith("walk,")

        _one_start_short(leg_dir)
        unchanged = {name: (leg_dir / name).read_bytes() for name in ("schedule.yaml", "start_configs.npz")}
        independent = cfg.model_copy(update={"sampling": cfg.sampling.model_copy(update={"exchange": "none"})})
        refused = run_leg_safe(job(independent))
        assert "other conditions" in refused["error"], refused["error"]
        assert all((leg_dir / name).read_bytes() == data for name, data in unchanged.items())

        again = run_leg_safe(job(cfg))
        assert "error" not in again, again
        assert again["start_configurations"] == first["start_configurations"]
        # the same samples, analysed again (a free energy, or none for windows that the samples do not connect)
        assert str(again["delta_g_decouple"]) == str(first["delta_g_decouple"])

        # fifth review: a schedule given (lambda_schedule.file) that is the one of the samples, taken from their
        # checkpoint: the start of the samples is kept as for the leg's own schedule
        own = (leg_dir / "schedule.yaml").read_text()
        (tmp_path / "given.yaml").write_text(own)
        (leg_dir / "schedule.yaml").unlink()
        _one_start_short(leg_dir)
        given = cfg.model_copy(update={"lambda_schedule": cfg.lambda_schedule.model_copy(
            update={"file": tmp_path / "given.yaml"})})
        taken = run_leg_safe(job(given))
        assert "error" not in taken, taken
        assert taken["start_configurations"] == first["start_configurations"]

        # fifth review: no record of the start configurations at all (the file is lost): the samples are taken up
        # from their checkpoint, their start is not known, and the run goes on (it stopped with a KeyError)
        (leg_dir / "start_configs.npz").unlink()
        unknown = run_leg_safe(job(cfg))
        assert "error" not in unknown, unknown
        assert unknown["start_configurations"] is None

        # fifth review: a finished leg whose schedule is lost is refused by the run before its result is set aside
        from cufep import run as run_module
        from cufep.config.auto import AutoReport
        from cufep.prep.ligand import load_ligands

        (leg_dir / "schedule.yaml").unlink()
        result = (leg_dir / "result.json").read_bytes()
        ligand = load_ligands(tmp_path / "ligand.sdf")[0]
        monkeypatch.setattr(run_module, "plan", lambda config, log, ligs=None: (config, AutoReport(), None,
                                                                                 [ligand]))
        with pytest.raises(Exception, match="no schedule.yaml"):
            run_module.run(cfg, only_legs=["solvent"], log=lambda line: None)
        assert (leg_dir / "result.json").read_bytes() == result and not (leg_dir / "schedule.yaml").exists()
    finally:
        cpu.setPropertyDefaultValue("DeterministicForces", before_default)


def test_samples_that_do_not_record_their_conditions_are_taken_up_as_they_are(tmp_path):
    """Fifth review of PR #18: samples of an earlier version, which do not record their conditions, are analysed as
    they are (cufep.workflow.sample_leg); without a checkpoint they were taken for samples started anew."""
    import numpy as np

    from cufep.engine.leg import restores_its_samples
    from cufep.sampling.storage import LegStorage

    hrex = tmp_path / "leg" / "hrex"
    storage = LegStorage(hrex, 2, 2)
    for _ in range(3):
        storage.append(np.zeros((2, 2)), np.arange(2), np.ones(2))
    storage.close()
    assert not (hrex / "checkpoint.npz").exists()
    assert restores_its_samples(tmp_path / "leg", total=10)

