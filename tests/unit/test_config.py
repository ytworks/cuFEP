import pytest
import yaml

from cufep.config.schema import Config, dump_config, load_config


def write(tmp_path, data):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_minimal_config_only_needs_protein_and_ligand(tmp_path):
    cfg = load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}))
    assert cfg.system.protein == (tmp_path / "p.pdb").resolve()
    assert cfg.backend.name == "cupotential" and cfg.backend.fallback == "openmm"
    assert cfg.lambda_schedule.mode == "adaptive"
    assert cfg.membrane.enabled == "auto"
    assert cfg.resources.gpus == "auto"
    assert cfg.sampling.repeats == 3 and cfg.sampling.production_ns == [5.0, 10.0]
    assert cfg.forcefield.ligand_charges == "am1bcc"
    assert cfg.restraints.k_r_kcal_per_A2 == 10.0 and cfg.restraints.k_angle_kcal_per_rad2 == 80.0


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "sampling": {"repeat": 2}}))
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "typo": 1}))


def test_missing_inputs_are_rejected(tmp_path):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb"}}))


@pytest.mark.parametrize("bad", [[], [5, 5], [10, 5], [0, 5], [-1]])
def test_production_lengths_must_increase(tmp_path, bad):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                     "sampling": {"production_ns": bad}}))


def test_round_trip(tmp_path):
    cfg = load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                       "resources": {"gpus": [0, 3]}, "membrane": {"enabled": True}}))
    text = dump_config(cfg, tmp_path / "out.yaml")
    again = Config.model_validate(yaml.safe_load(text))
    assert again.resources.gpus == [0, 3] and again.membrane.enabled is True
    assert again.model_dump(mode="json") == cfg.model_dump(mode="json")
