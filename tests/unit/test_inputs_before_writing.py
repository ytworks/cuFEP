"""Inputs that contradict the calculation are refused by every command before it writes anything
(review of PR 1 on a3462a5: manual restraint anchors, ligand names, the ligand of `cufep validate`).
"""
import re
from pathlib import Path

import numpy as np
import pytest

from cufep.alchemy.restraints import BoreschRestraint
from cufep.config.schema import Config
from cufep.workflow import (
    OUTPUT_ROOT_NAMES,
    WorkflowError,
    ligand_dir_name,
    ligand_directories,
    restraint_anchor_problems,
    select_names,
)


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


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
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
        assert not (tmp_path / "out").exists(), name     # not even the directory, its lock or its log
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


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
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
    monkeypatch.setattr(driver, "plan", lambda cfg, echo=print, chosen=None: (cfg, None, None, chosen or ligands))

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


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_cufep_run_with_an_unknown_ligand_writes_nothing(tmp_path, repo):
    """Review (independent check before the push): `cufep run -o new_dir --ligand typo` made new_dir with its lock and
    its log before it said that the ligand does not exist. Duplicate titles count among the selected ligands only,
    as before."""
    from cufep import run as driver

    cfg = config_for(tmp_path, repo, write_sdf(tmp_path / "ligands.sdf", repo, ["ejm_31", "ejm_42", "dup", "dup"]))
    cfg.system.ligand_names = ["ejm_31", "ejm_42"]              # the duplicate titles are not selected
    with pytest.raises(WorkflowError, match="--ligand: not among the ligands of this calculation: ejm_3l"):
        driver.run(cfg, log=lambda m: None, only_ligands=["ejm_3l"])
    assert not (tmp_path / "out").exists()
    assert [lig.name for lig in driver.ligands_of_run(cfg)] == ["ejm_31", "ejm_42"]
    cfg.system.ligand_names = ["dup"]
    with pytest.raises(WorkflowError, match="not unique"):
        driver.ligands_of_run(cfg)


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


def test_what_a_complex_job_would_refuse_is_refused_before_any_job(tmp_path, monkeypatch, dipeptides_in_water):
    """Review (independent check before the push): REST2 residues with prepared systems of an earlier version
    (no record of the input chains), and a manual restraint with an equil.npz without frame boxes, were refused only
    inside each complex job, after its result had been set aside and its pre-equilibration run."""
    from cufep import workflow
    from cufep.workflow import WorkflowError, check_before_jobs

    system, top, pos, box, ligand, other = dipeptides_in_water
    old_prep = {"topology": top, "positions": pos, "ligand": ligand, "protein": other, "input_chains": None,
                "box": np.asarray(box)}
    monkeypatch.setattr(workflow, "load_prepared", lambda prep_dir, name: old_prep)
    lig_dir = tmp_path / "lig"
    residues = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                      "sampling": {"rest2": {"scale": 1.5, "residues": ["2"]}}})
    with pytest.raises(WorkflowError, match="did not record the chains of the input"):
        check_before_jobs(residues, lig_dir, {(0, "complex")})
    check_before_jobs(residues, lig_dir, {(0, "solvent")})          # a run of the solvent leg alone needs no region
    # the anchors below are atoms of two peptides chosen for the files, not for their angles (theta_B 158 degrees,
    # refused by the check of the geometry, tests/unit/test_periodic_images.py); that check is set aside here
    from cufep.alchemy import restraints

    monkeypatch.setattr(restraints, "anchor_geometry_problem", lambda *args, **kwargs: None)
    manual = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                    "restraints": {"selection": "manual", "host_atoms": other[:3],
                                                   "guest_atoms": ligand[:3]}})
    leg = lig_dir / "repeat_0" / "complex"
    leg.mkdir(parents=True)
    np.savez(leg / "equil.npz", positions=pos, frames=pos[None], frame_atoms=np.arange(len(pos)))
    with pytest.raises(WorkflowError, match="holds no boxes of its frames"):
        check_before_jobs(manual, lig_dir, {(0, "complex")})
    np.savez(leg / "equil.npz", positions=pos, frames=pos[None], frame_atoms=np.arange(len(pos)),
             frame_boxes=np.asarray(box)[None])
    check_before_jobs(manual, lig_dir, {(0, "complex")})


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer", "prolif", "MDAnalysis")
def test_a_restraint_from_interactions_that_cannot_be_chosen_is_refused_before_anything_is_written(
        tmp_path, repo, monkeypatch):
    """``restraints.selection: interactions`` needs ProLIF, and a ligand with three bonded heavy atoms
    that can be guest anchors. Without either, `cufep run` and `cufep plan` stop before the protein is
    prepared and before a directory is made; acetonitrile has no such atoms, ejm_31 has."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import run as driver
    from cufep.analysis import interactions

    def config(sdf, selection="interactions"):
        data = config_for(tmp_path, repo, sdf).model_dump(mode="json")
        data["restraints"] = {"selection": selection}
        return Config.model_validate(data)

    small = Chem.AddHs(Chem.MolFromSmiles("CC#N"))
    AllChem.EmbedMolecule(small, randomSeed=1)
    small.SetProp("_Name", "acetonitrile")
    writer = Chem.SDWriter(str(tmp_path / "small.sdf"))
    writer.write(small)
    writer.close()
    out = tmp_path / "out"
    for command in (driver.run, lambda cfg: driver.plan(cfg, lambda line: None)):
        with pytest.raises(WorkflowError, match="cannot give acetonitrile a restraint: the ligand has no atom"):
            command(config(tmp_path / "small.sdf"))
        assert written(out) == []
    good = write_sdf(tmp_path / "good.sdf", repo, ["ejm_31"])
    monkeypatch.setattr(interactions, "packages_problem",
                        lambda: "restraints.selection: interactions needs ProLIF and MDAnalysis, which cannot be "
                                "imported here: prolif (ImportError: No module named 'prolif')")
    for command in (driver.run, lambda cfg: driver.plan(cfg, lambda line: None)):
        with pytest.raises(WorkflowError, match="needs ProLIF and MDAnalysis"):
            command(config(good))
        assert written(out) == []
    # another selection asks for neither
    from cufep.workflow import check_interaction_inputs, ligands_of_run

    cfg = config(tmp_path / "small.sdf", selection="auto")
    check_interaction_inputs(cfg, ligands_of_run(cfg))


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer", "prolif", "MDAnalysis")
def test_only_the_ligands_whose_complex_leg_is_run_need_guest_anchors(tmp_path, repo, monkeypatch):
    """Review of f5a77ce (P2): with ejm_31 and acetonitrile in the SDF, `cufep run --ligand ejm_31` and
    `cufep run --ligand ejm_31 --leg solvent` were refused because of acetonitrile ("cannot give acetonitrile a
    restraint"), whose complex leg they do not run. Asked of a run are the ligands whose complex leg it runs,
    and without a complex leg not ProLIF either."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep import run as driver
    from cufep.analysis import interactions

    source = next(iter(Chem.SDMolSupplier(str(write_sdf(tmp_path / "one.sdf", repo, ["ejm_31"])), removeHs=False)))
    small = Chem.AddHs(Chem.MolFromSmiles("CC#N"))
    AllChem.EmbedMolecule(small, randomSeed=1)
    small.SetProp("_Name", "acetonitrile")
    writer = Chem.SDWriter(str(tmp_path / "two.sdf"))
    writer.write(source)
    writer.write(small)
    writer.close()
    cfg = config_for(tmp_path, repo, tmp_path / "two.sdf")
    assert cfg.restraints.selection == "interactions"
    out = tmp_path / "out"

    class Reached(Exception):
        """The run went on to its plan: it was not refused before."""

    def no_preparation(*args, **kwargs):
        raise Reached

    monkeypatch.setattr("cufep.prep.protein.prepare_protein", no_preparation)
    for options in ({}, {"only_ligands": ["acetonitrile"]}, {"only_legs": ["complex"]}):
        with pytest.raises(WorkflowError, match="cannot give acetonitrile a restraint"):
            driver.run(cfg, log=lambda line: None, **options)
        assert written(out) == []
    for options in ({"only_ligands": ["ejm_31"]}, {"only_ligands": ["ejm_31"], "only_legs": ["solvent"]},
                    {"only_legs": ["solvent"]}, {"only_ligands": ["acetonitrile"], "only_legs": ["solvent"]}):
        with pytest.raises(Reached):
            driver.run(cfg, log=lambda line: None, **options)
    # a run of solvent legs alone chooses no restraint and needs no ProLIF
    monkeypatch.setattr(interactions, "packages_problem", lambda: "ProLIF cannot be imported here")
    with pytest.raises(Reached):
        driver.run(cfg, log=lambda line: None, only_ligands=["ejm_31"], only_legs=["solvent"])
    with pytest.raises(WorkflowError, match="ProLIF cannot be imported here"):
        driver.run(cfg, log=lambda line: None, only_ligands=["ejm_31"])
