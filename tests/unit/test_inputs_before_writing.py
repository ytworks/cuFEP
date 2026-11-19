"""Inputs that contradict the calculation are refused by every command before it writes anything
(review of PR 1 on a3462a5: manual restraint anchors, ligand names, the ligand of `cufep validate`).
"""
import re
from pathlib import Path

import numpy as np
import pytest

from cufep.alchemy.restraints import BoreschRestraint
from cufep.config.schema import Config
from cufep.workflow import (OUTPUT_ROOT_NAMES, WorkflowError, ligand_dir_name, ligand_directories,
                            restraint_anchor_problems, select_names)


def write_sdf(path: Path, repo: Path, titles: list[str]) -> Path:
    """Records of the TYK2 ligand ejm_31 under other titles."""
    from rdkit import Chem

    source = next(m for m in Chem.SDMolSupplier(str(repo / "benchmarks/tyk2/inputs/tyk2_ligands.sdf"), removeHs=False)
                  if m.GetProp("_Name") == "ejm_31")
    writer = Chem.SDWriter(str(path))
    for title in titles:
        mol = Chem.Mol(source)
        mol.SetProp("_Name", title)
        writer.write(mol)
    writer.close()
    return path


def config_for(tmp_path: Path, repo: Path, sdf: Path) -> Config:
    return Config.model_validate({"system": {"protein": str(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb"),
                                             "ligand": str(sdf)},
                                  "output": {"dir": str(tmp_path / "out")}})


def written(out: Path) -> list[str]:
    return sorted(str(p.relative_to(out)) for p in out.rglob("*")) if out.exists() else []


@pytest.mark.parametrize("titles,why", [
    (["."], "cannot have a directory of its own"),
    ([".."], "cannot have a directory of its own"),
    ([".hidden"], "cannot have a directory of its own"),
    (["cache"], "cannot have a directory of its own"),
    (["summary.json"], "cannot have a directory of its own"),
    (["a b", "a_b"], "would share the directory a_b/"),
])
def test_a_ligand_without_a_directory_of_its_own_is_refused_by_every_command(tmp_path, repo, monkeypatch, titles, why):
    """Review of PR 1 (P1): a ligand named '.' wrote prep/ and result.json into the output root, '..' into its
    parent."""
    from cufep import run as driver
    from cufep import validate as validation
    from cufep.backends.cupotential import CuPotentialBackend

    cfg = config_for(tmp_path, repo, write_sdf(tmp_path / "ligands.sdf", repo, titles))
    monkeypatch.setattr(CuPotentialBackend, "available", staticmethod(lambda: (True, "")))
    entries = {"plan": lambda: driver.plan(cfg, lambda m: None),
               "run": lambda: driver.run(cfg, log=lambda m: None),
               "validate": lambda: validation.validate(cfg, None, 0, lambda m: None)}
    for name, entry in entries.items():
        with pytest.raises(WorkflowError, match=why):
            entry()
        assert set(written(tmp_path / "out")) <= {".cufep.lock"}, name
    assert written(tmp_path)[:1] != ["prep"] and not (tmp_path / "prep").exists()


def test_ligand_directory_names():
    assert ligand_dir_name("ejm_31") == "ejm_31" and ligand_dir_name("4e (A2A)") == "4e__A2A_"
    for name in (".", "..", " .. ", ".x", "cache", "frozen_schedules"):
        with pytest.raises(WorkflowError, match="cannot have a directory of its own"):
            ligand_dir_name(name)
    assert ligand_directories("/o", ["x", "y"]) == {"x": Path("/o/x"), "y": Path("/o/y")}


def test_every_name_cufep_writes_at_the_output_root_is_reserved(repo):
    """A ligand directory must never take the name of a file or directory cuFEP writes at the root of the
    output; the reserved names are found in the source, so a new one cannot be missed."""
    names = set()
    for path in (repo / "src" / "cufep").rglob("*.py"):
        text = path.read_text()
        names |= set(re.findall(r'(?:out_dir|run_dir|output\.dir|out)\s*/\s*"([^"/]+)"', text))
        names |= set(re.findall(r'file_name="([^"]+)"', text))
        names |= set(re.findall(r'open\(Path\(out_dir\)\s*/\s*"([^"]+)"', text))
    names.discard("")
    assert names, "the scan found nothing"
    assert names <= OUTPUT_ROOT_NAMES, sorted(names - OUTPUT_ROOT_NAMES)


def test_the_ligand_named_on_the_command_line_must_exist(tmp_path, repo, monkeypatch):
    """Review of PR 1 (P2): a mistyped `cufep validate --ligand` took the first record of the SDF and could end
    with success for another ligand. The first ligand is the default only when no name is given."""
    from cufep import run as driver
    from cufep import validate as validation
    from cufep import workflow
    from cufep.backends.cupotential import CuPotentialBackend
    from cufep.prep.ligand import load_ligands

    sdf = write_sdf(tmp_path / "ligands.sdf", repo, ["ejm_31", "ejm_42"])
    cfg = config_for(tmp_path, repo, sdf)
    ligands = load_ligands(sdf)
    monkeypatch.setattr(CuPotentialBackend, "available", staticmethod(lambda: (True, "")))
    monkeypatch.setattr(driver, "plan", lambda cfg, echo=print: (cfg, None, None, ligands))

    def must_not_prepare(*args, **kwargs):
        raise AssertionError("a ligand was prepared")

    monkeypatch.setattr(workflow, "prepare_ligand", must_not_prepare)
    with pytest.raises(WorkflowError, match="--ligand: not among the ligands of this calculation: ejm_3l"):
        validation.validate(cfg, "ejm_3l", 0, lambda m: None)
    with pytest.raises(AssertionError, match="a ligand was prepared"):        # the named one is taken
        validation.validate(cfg, "ejm_42", 0, lambda m: None)
    assert select_names(["a", "b", "c"], ["c", "a"], "--ligand") == ["a", "c"]
    assert select_names(["a", "b"], [], "--ligand") == ["a", "b"]
    with pytest.raises(WorkflowError, match="--ligand: not among"):
        driver.select_jobs(["a", "b"], 1, only_ligands=["x"])


def test_manual_anchors_belong_to_the_protein_and_the_ligand():
    """Review of PR 1 (P1): three protein atoms given as guest atoms passed the check; with all six atoms in
    the protein the restraint does not change when the ligand moves, and its release was still added to the
    binding free energy."""
    prep = {"protein": list(range(0, 100)), "ligand": list(range(100, 110))}
    assert restraint_anchor_problems([10, 11, 12], [100, 101, 102], prep) == []
    problems = restraint_anchor_problems([10, 11, 12], [20, 21, 22], prep)
    assert problems == ["restraints.guest_atoms: [20, 21, 22] are not atoms of the ligand in the prepared complex "
                        "(atoms are numbered from 0 as in prep/complex.pdb)"]
    assert restraint_anchor_problems([100, 11, 12], [100, 101, 101], prep)[0].startswith("restraints.host_atoms")
    # what the reviewer saw: six protein atoms do not hold the ligand
    pos = np.random.default_rng(1).normal(size=(110, 3))
    r = BoreschRestraint((10, 11, 12), (20, 21, 22), 1.0, 1.0, 1.0, 0.5, 0.5, 0.5)
    moved = pos.copy()
    moved[100:110] += 2.0
    assert r.energy(moved) == pytest.approx(r.energy(pos))


@pytest.mark.parametrize("restraints,why", [
    ({"selection": "manual"}, "host_atoms must be three distinct"),
    ({"selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [4, 4, 5]}, "guest_atoms must be three distinct"),
    ({"selection": "manual", "host_atoms": [1, 2], "guest_atoms": [4, 5, 6]}, "host_atoms must be three distinct"),
    ({"selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [3, 4, 5]}, "share atoms"),
])
def test_manual_anchors_are_checked_when_the_configuration_is_read(restraints, why):
    with pytest.raises(Exception, match=why):
        Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "restraints": restraints})
