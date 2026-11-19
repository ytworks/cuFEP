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
    assert cfg.membrane.enabled is False and cfg.membrane.orientation == "auto"
    assert cfg.resources.gpus == "auto"
    assert cfg.sampling.repeats == 1                       # one run, as FEP+ and Flare
    assert cfg.sampling.equilibration("complex") == 1.0 and cfg.sampling.production("solvent") == 5.0
    assert cfg.sampling.time_per_iteration_ps == 2.5
    assert cfg.dynamics.timestep_fs == 4.0 and cfg.dynamics.constraints == "hbonds"
    assert cfg.resources.oversubscription_factor == "auto"
    assert cfg.forcefield.ligand_charges == "am1bcc"
    assert cfg.restraints.k_r_kcal_per_A2 == 10.0 and cfg.restraints.k_angle_kcal_per_rad2 == 80.0


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "sampling": {"repeat": 2}}))
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "typo": 1}))


def test_membrane_is_an_explicit_switch(tmp_path):
    base = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}
    assert load_config(write(tmp_path, {**base, "membrane": {"enabled": True}})).membrane.enabled is True
    with pytest.raises(Exception):
        load_config(write(tmp_path, {**base, "membrane": {"enabled": "auto"}}))


def test_missing_inputs_are_rejected(tmp_path):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb"}}))


BASE = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}


def test_every_setting_has_a_default():
    """Only the two input files are required; nothing else is left undefined."""
    dumped = Config.model_validate(BASE).model_dump(mode="json")
    optional = {("system", "ligand_names"), ("restraints", "host_atoms"), ("restraints", "guest_atoms"),
                ("lambda_schedule", "file")}

    def walk(node, path=()):
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, path + (key,))
            elif value is None:
                assert path + (key,) in optional, f"{'.'.join(path + (key,))} has no default"

    walk(dumped)
    with pytest.raises(Exception):
        Config.model_validate({})


def test_lambda_windows_have_recommended_defaults_and_can_be_set(tmp_path):
    ls = load_config(write(tmp_path, BASE)).lambda_schedule
    assert ls.windows.complex.restrain == 6 and ls.windows.complex.discharge == 11
    assert len(ls.windows.complex.vanish) == 15 and ls.windows.solvent.discharge == 5
    assert len(ls.windows.solvent.vanish) == 10
    a = ls.adaptive
    assert (a.overlap_threshold, a.candidates) == (0.04, 101)
    # the precalculation: the paper's 50 ps in the solvent leg, 500 ps after 500 ps in the complex leg
    assert (a.window("solvent"), a.equilibration("solvent")) == (50.0, 0.0)
    assert (a.window("complex"), a.equilibration("complex")) == (500.0, 500.0)
    assert ls.fixed_walk_ps == 50.0
    assert a.initial_windows.for_leg("complex") == {"restrain": 3, "discharge": 5, "vanish": 9}
    assert a.initial_windows.for_leg("solvent") == {"discharge": 5, "vanish": 9}
    assert (a.max_windows.complex, a.max_windows.solvent) == (48, 32)
    cfg = load_config(write(tmp_path, {**BASE, "lambda_schedule": {
        "mode": "fixed",
        "windows": {"complex": {"restrain": 4, "vanish": 20}, "solvent": {"vanish": [0.0, 0.5, 0.8, 1.0]}},
        "adaptive": {"overlap_threshold": 0.03, "initial_windows": {"solvent": {"vanish": 5}}}}}))
    w = cfg.lambda_schedule.windows
    assert cfg.lambda_schedule.mode == "fixed" and w.complex.restrain == 4 and w.complex.vanish == 20
    assert w.complex.discharge == 11 and w.solvent.vanish == [0.0, 0.5, 0.8, 1.0] and w.solvent.discharge == 5
    assert w.for_leg("solvent") == {"discharge": 5, "vanish": [0.0, 0.5, 0.8, 1.0]}
    assert list(w.for_leg("complex")) == ["restrain", "discharge", "vanish"]
    assert cfg.lambda_schedule.adaptive.overlap_threshold == 0.03
    assert cfg.lambda_schedule.adaptive.initial_windows.for_leg("solvent") == {"discharge": 5, "vanish": 5}
    assert cfg.lambda_schedule.adaptive.initial_windows.for_leg("complex") == {"restrain": 3, "discharge": 5, "vanish": 9}


def test_precalculation_lengths_are_one_number_or_one_per_leg(tmp_path):
    one = load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {"window_ps": 20, "equilibration_ps": 0}}}))
    a = one.lambda_schedule.adaptive
    assert (a.window("complex"), a.window("solvent"), a.equilibration("complex")) == (20.0, 20.0, 0.0)
    per_leg = load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {
        "window_ps": {"complex": 200, "solvent": 50}, "equilibration_ps": {"complex": 100, "solvent": 0}}}}))
    a = per_leg.lambda_schedule.adaptive
    assert (a.window("complex"), a.window("solvent")) == (200.0, 50.0)
    assert (a.equilibration("complex"), a.equilibration("solvent")) == (100.0, 0.0)
    for bad in ({"window_ps": 0}, {"window_ps": {"complex": 500, "solvent": 0}}, {"equilibration_ps": -1},
                {"window_ps": {"complex": 500}}):
        with pytest.raises(Exception):
            load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": bad}}))


def test_rest2_is_on_by_default_and_can_be_set(tmp_path):
    from cufep.cli import lambda_window_table

    cfg = load_config(write(tmp_path, BASE))
    r2 = cfg.sampling.rest2
    assert (r2.scale, r2.region, r2.pocket_cutoff_nm, r2.residues) == (1.5, "ligand", 0.4, [])
    assert "hot region: the ligand; 1.5 times the temperature" in lambda_window_table(cfg)
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"rest2": {"scale": 1.0}}}))
    assert "off (sampling.rest2.scale is 1)" in lambda_window_table(cfg)
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"rest2": {"scale": 2, "region": "pocket", "residues": ["A:960"]}}}))
    assert "protein residues within 0.4 nm of it (A:960)" in lambda_window_table(cfg)
    for bad in ({"scale": 0.9}, {"region": "protein"}, {"pocket_cutoff_nm": 0}):
        with pytest.raises(Exception):
            load_config(write(tmp_path, {**BASE, "sampling": {"rest2": bad}}))


def test_one_value_of_a_section_can_be_set_and_the_rest_keep_their_defaults(tmp_path):
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"pre_equilibration": {"complex": {"md_ns": 1.0},
                                                                                 "solvent": {"nvt_ns": 0.05}}}}))
    pe = cfg.sampling.pre_equilibration
    assert (pe.complex.nvt_ns, pe.complex.npt_ns, pe.complex.md_ns) == (0.25, 0.5, 1.0)
    assert (pe.solvent.nvt_ns, pe.solvent.npt_ns, pe.solvent.md_ns) == (0.05, 0.2, 0.5)


@pytest.mark.parametrize("bad", [1, 0, [0.0], [0.1, 1.0], [0.0, 0.9], [0.0, 0.6, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0]])
def test_bad_lambda_windows_are_rejected(tmp_path, bad):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": {"windows": {"solvent": {"vanish": bad}}}}))


@pytest.mark.parametrize("section", [{"adaptive": {"overlap_threshold": 0.0}}, {"adaptive": {"candidates": 2}},
                                     {"adaptive": {"initial_windows": {"complex": {"vanish": 1}}}},
                                     {"adaptive": {"initial_windows": {"vanish": 9}}}, {"mode": "automatic"},
                                     {"preset": "fast"}])
def test_bad_schedule_settings_are_rejected(tmp_path, section):
    with pytest.raises(Exception):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": section}))


def test_lengths_per_leg(tmp_path):
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"production_ns": {"complex": 10, "solvent": 2},
                                                           "equilibration_ns": 0.5, "repeats": 3}}))
    assert cfg.sampling.production("complex") == 10.0 and cfg.sampling.production("solvent") == 2.0
    assert cfg.sampling.equilibration("complex") == 0.5 == cfg.sampling.equilibration("solvent")
    assert cfg.sampling.repeats == 3
    for bad in ({"production_ns": 0}, {"production_ns": {"complex": 5}}, {"repeats": 0}, {"production_ns": [5, 10]}):
        with pytest.raises(Exception):
            load_config(write(tmp_path, {**BASE, "sampling": bad}))


def test_round_trip(tmp_path):
    cfg = load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                       "resources": {"gpus": [0, 3]}, "membrane": {"enabled": True}}))
    text = dump_config(cfg, tmp_path / "out.yaml")
    again = Config.model_validate(yaml.safe_load(text))
    assert again.resources.gpus == [0, 3] and again.membrane.enabled is True
    assert again.model_dump(mode="json") == cfg.model_dump(mode="json")


def test_documented_defaults_are_the_defaults_of_the_code(repo):
    """docs/configuration.md lists every setting with its default; it must not drift."""
    text = (repo / "docs" / "configuration.md").read_text()
    block = text.split("```yaml", 1)[1].split("```", 1)[0]
    documented = yaml.safe_load(block)
    actual = Config.model_validate({"system": documented["system"]}).model_dump(mode="json")
    for section in actual:
        if section in ("system", "output"):
            continue
        assert documented[section] == actual[section], f"docs/configuration.md and the code differ in '{section}'"
    assert set(documented) == set(actual)
