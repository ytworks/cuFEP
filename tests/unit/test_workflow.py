"""Input checks, membrane detection, GPU allocation and result combination."""
import json

import numpy as np
import pytest

from cufep.prep.ligand import LigandError, load_ligands
from cufep.prep.orient import detect_membrane, orient_positions
from cufep.prep.protein import prepare_protein
from cufep.scheduler import assign_gpu_counts
from cufep.workflow import LegJob, check_ligand, combine_ligand


@pytest.fixture(scope="module")
def tyk2(repo):
    return prepare_protein(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb")


@pytest.fixture(scope="module")
def ligands(repo):
    return load_ligands(repo / "benchmarks/tyk2/inputs/tyk2_ligands.sdf")


def test_ligands_load_with_names_and_experimental_values(ligands):
    assert len(ligands) == 16
    lig = next(l for l in ligands if l.name == "ejm_31")
    assert lig.n_atoms == 32 and lig.formal_charge == 0
    assert float(lig.properties["r_exp_dg"]) == pytest.approx(-9.54)


def test_bound_pose_passes_and_displaced_pose_stops(tyk2, ligands):
    lig = ligands[0]
    assert check_ligand(lig, tyk2.topology, tyk2.positions_nm()) == []
    from openff.units import unit
    moved = load_ligands(lig.source)[0]
    moved.offmol._conformers = [moved.offmol.conformers[0] + np.array([40.0, 0.0, 0.0]) * unit.angstrom]
    with pytest.raises(LigandError, match="pocket"):
        check_ligand(moved, tyk2.topology, tyk2.positions_nm())


def test_charged_ligand_stops(repo, tyk2):
    ligs = load_ligands(repo / "benchmarks/a2a/inputs/a2a_hip278_ligands.sdf")
    charged = next(l for l in ligs if l.formal_charge != 0)
    with pytest.raises(LigandError, match="formal charge"):
        check_ligand(charged, tyk2.topology, tyk2.positions_nm())


def test_ligand_without_hydrogens_stops(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    w = Chem.SDWriter(str(tmp_path / "noh.sdf"))
    w.write(Chem.RemoveHs(mol))
    w.close()
    with pytest.raises(LigandError, match="hydrogens"):
        load_ligands(tmp_path / "noh.sdf")


def test_membrane_detection(repo, tyk2):
    v = detect_membrane(tyk2.topology, tyk2.positions_nm())
    assert v.confident and not v.is_membrane
    a2a = prepare_protein(repo / "benchmarks/a2a/inputs/a2a_hip278_protein.pdb")
    v = detect_membrane(a2a.topology, a2a.positions_nm())
    assert v.confident and v.is_membrane and v.n_crossing_helices >= 5
    assert 2.0 <= v.thickness_nm <= 4.0
    z = orient_positions(a2a.positions_nm(), v)[:, 2]
    assert z.min() < -1.5 and z.max() > 1.5                      # the protein spans the slab along z


def _job(leg):
    return LegJob("lig", leg, 0, "x", [], {}, 5.0)


def test_gpu_allocation():
    jobs = [_job("complex"), _job("solvent")]
    assert assign_gpu_counts(jobs, 8) == {0: 7, 1: 1}
    assert assign_gpu_counts(jobs, 2) == {0: 1, 1: 1}
    assert assign_gpu_counts(jobs, 1) == {0: 1, 1: 1}
    six = [_job("complex")] * 3 + [_job("solvent")] * 3
    assert assign_gpu_counts(six, 8) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    many = [_job("complex")] * 20
    assert set(assign_gpu_counts(many, 8).values()) == {1}


def test_combination_follows_the_thermodynamic_cycle(tmp_path):
    def leg(r, name, dg, release=0.0):
        d = tmp_path / f"repeat_{r}" / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "leg": name, "delta_g_decouple": dg, "delta_g_mbar_error": 0.4, "restraint_release": release,
            "lrc": -1.0, "backend": "openmm", "analysis": {"stage_delta_g": {"sterics": dg}},
            "sampling_sufficient": True, "sampling_reasons": [], "production_ns": 5.0, "notes": []}))
    for r, (c, s) in enumerate([(160.0, 80.0), (164.184, 80.0)]):
        leg(r, "complex", c, release=-40.0)
        leg(r, "solvent", s)
    out = combine_ligand(tmp_path, "lig", 3)
    # dG_bind = solvent - complex - release, converted to kcal/mol
    assert out["n_repeats"] == 2 and "repeat 2 is incomplete" in out["warnings"]
    assert out["repeats"][0]["dg_bind"] == pytest.approx((80.0 - 160.0 + 40.0) / 4.184)
    assert out["dg_bind"] == pytest.approx((-40.0 - 44.184) / 2 / 4.184)
    assert out["dg_bind_std"] == pytest.approx(np.std([-40.0, -44.184], ddof=1) / 4.184)
