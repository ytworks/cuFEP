"""The configuration of a relative run (cufep.rbfe.config): its documented defaults and its refusals."""
from __future__ import annotations

import re

import pytest
import yaml

from cufep.rbfe.config import RbfeConfig


def _minimal(**more) -> dict:
    return {"system": {"protein": "p.pdb", "ligands": "l.sdf"}, **more}


def test_documented_defaults_are_the_defaults_of_the_code(repo):
    """docs/rbfe.md gives the defaults of the sections of a relative run; they are those of the code."""
    text = (repo / "docs" / "rbfe.md").read_text()
    block = yaml.safe_load(re.search(r"```yaml\n(.*?)```", text, re.S).group(1))
    defaults = RbfeConfig.model_validate(_minimal()).model_dump(mode="json")
    for section, values in block.items():
        for key, value in values.items():
            assert defaults[section][key] == value, f"{section}.{key}"
    for section in ("network", "mapping", "alchemy", "lambda_schedule"):
        assert block[section] == defaults[section], section


@pytest.mark.parametrize("more, message", [
    ({"sampling": {"relocation_mc": {"enabled": True}}}, "a move of the ABFE"),
    # the checks of the grand canonical moves are those of the ABFE (cufep.config.schema.gcmc_problems)
    ({"sampling": {"gcmc": {"batch_size": 3000}}}, "batch_size"),
    ({"sampling": {"gcmc": {"enabled": True}}, "forcefield": {"water": "amber14/spce.xml"}},
     "no value for forcefield.water"),
    ({"membrane": {"enabled": True, "complex_leg": "water"}}, "complex_leg"),
    ({"network": {"edges": [["a", "a"]]}}, "to itself"),
    ({"network": {"edges": [["a", "b"], ["b", "a"]]}}, "twice"),
    ({"analysis": {"lrc": "reweight"}}, "lrc"),
    # review of PR #17: nine windows to start with and at most two
    ({"lambda_schedule": {"adaptive": {"max_windows": 2}}}, "max_windows"),
    ({"lambda_schedule": {"adaptive": {"initial_windows": 12, "candidates": 11}}}, "initial_windows"),
])
def test_settings_a_relative_run_cannot_take_are_refused_when_read(more, message):
    with pytest.raises(ValueError, match=message):
        RbfeConfig.model_validate(_minimal(**more))


def test_the_complex_leg_is_pre_equilibrated_with_the_solute_held_and_no_free_md():
    """The pre-equilibration of a complex leg of a relative run: the pose that was given, held through NVT and NPT
    and through the equilibration of a new bilayer, without free MD (docs/rbfe.md, "Sampling")."""
    from cufep.workflow import pre_equilibration_parts

    rcfg = RbfeConfig.model_validate(_minimal())
    assert pre_equilibration_parts(rcfg, "complex", False) == (0.0, True)
    in_bilayer = RbfeConfig.model_validate(_minimal(membrane={"enabled": True}))
    assert pre_equilibration_parts(in_bilayer, "complex", True) == (in_bilayer.membrane.equilibration_ns, True)
    assert pre_equilibration_parts(rcfg, "solvent", False) == (rcfg.sampling.pre_equilibration.solvent.md_ns, False)


def test_complex_legs_alone_that_take_the_windows_of_a_solvent_leg_without_them_are_refused(tmp_path):
    """With `lambda_schedule.adaptive.legs: solvent` a complex leg takes the windows of the solvent leg of its edge;
    `cufep rbfe run --leg complex` before that leg has them is refused before the edges are prepared (the complex
    leg would otherwise wait for ever for a solvent leg that is not run)."""
    from cufep.rbfe.run import check_windows_of_the_solvent_leg
    from cufep.workflow import WorkflowError

    paper = RbfeConfig.model_validate(_minimal(lambda_schedule={"adaptive": {"legs": "solvent"}}))
    edges = [("a", "b")]
    with pytest.raises(WorkflowError, match="a~b/repeat_0"):
        check_windows_of_the_solvent_leg(paper, tmp_path, edges, ["complex"])
    for allowed in (None, ["solvent"], ["complex", "solvent"]):
        check_windows_of_the_solvent_leg(paper, tmp_path, edges, allowed)
    check_windows_of_the_solvent_leg(RbfeConfig.model_validate(_minimal()), tmp_path, edges, ["complex"])
    solvent = tmp_path / "edges" / "a~b" / "repeat_0" / "solvent"
    solvent.mkdir(parents=True)
    for name in ("schedule.yaml", "start_configs.npz"):
        (solvent / name).write_text("")
    check_windows_of_the_solvent_leg(paper, tmp_path, edges, ["complex"])


@pytest.mark.parametrize("failed, missing, code", [([], ["a~b"], 0), (["a~b/r1/complex"], [], 1)])
def test_the_exit_code_is_that_of_the_jobs_of_this_run(monkeypatch, tmp_path, failed, missing, code):
    """Review of PR #17: the exit code said whether every edge had a ddG. A run of --leg solvent alone that
    succeeded ended with 1 (the complex legs were not run), and a repeat that failed next to one that gave the ddG
    ended with 0. It is 1 exactly when a job of this run failed."""
    from click.testing import CliRunner

    from cufep import cli
    from cufep.rbfe import run as run_module

    summary = {"edges": [{"edge": e} for e in missing] + [{"edge": "x~y", "ddg": 1.0}], "failed_in_this_run": failed}
    monkeypatch.setattr(run_module, "run_rbfe", lambda *args, **kwargs: summary)
    monkeypatch.setattr(cli, "_load_rbfe", lambda path: RbfeConfig.model_validate(_minimal()))
    (tmp_path / "c.yaml").write_text("x: 1\n")
    result = CliRunner().invoke(cli.main, ["rbfe", "run", str(tmp_path / "c.yaml")])
    assert result.exit_code == code, result.output


def test_a_relative_run_makes_grand_canonical_moves_of_water_in_the_complex_leg_by_default():
    """The owner's decision of 2026-11-28: on by default in the complex leg of a relative calculation (off in the
    ABFE); the solvent leg never has them."""
    from cufep.config.schema import Config
    from cufep.workflow import gcmc_in

    rcfg = RbfeConfig.model_validate(_minimal())
    assert rcfg.sampling.gcmc.enabled is True
    assert gcmc_in(rcfg, "complex") and not gcmc_in(rcfg, "solvent")
    assert Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}}).sampling.gcmc.enabled is False
    off = RbfeConfig.model_validate(_minimal(sampling={"gcmc": {"enabled": False}}))
    assert not gcmc_in(off, "complex")
    # a YAML that gives only some of the settings of the moves keeps them on (they were switched off so before)
    some = RbfeConfig.model_validate(_minimal(sampling={"gcmc": {"attempts": 5000}}))
    assert gcmc_in(some, "complex") and some.sampling.gcmc.attempts == 5000

