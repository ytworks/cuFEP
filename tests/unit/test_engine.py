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
