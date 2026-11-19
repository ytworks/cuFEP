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
    # 1 ns of equilibration and 10 ns of production per lambda window, as OpenFE's ABFE protocol
    assert cfg.sampling.equilibration("complex") == 1.0 and cfg.sampling.production("solvent") == 10.0
    assert cfg.sampling.production("complex") == 10.0
    assert cfg.sampling.time_per_iteration_ps == 2.5
    assert cfg.dynamics.timestep_fs == 4.0 and cfg.dynamics.constraints == "hbonds"
    assert cfg.resources.oversubscription_factor == "auto"
    assert cfg.forcefield.ligand_charges == "am1bcc"
    assert cfg.restraints.k_r_kcal_per_A2 == 10.0 and cfg.restraints.k_angle_kcal_per_rad2 == 80.0


def test_files_of_earlier_versions_are_read(tmp_path, capsys):
    """Review of PR 1: every full configuration and resolved_config.yaml of the previous version holds
    analysis.convergence_tolerance_sigma, which this version does not have."""
    import copy

    from cufep import cli
    from cufep.config.schema import load_config_with_notices, read_settings

    base = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}
    full = Config.model_validate(base).model_dump(mode="json")
    old = copy.deepcopy(full)
    old["analysis"]["convergence_tolerance_sigma"] = 2.0                    # the default of that version
    cfg, notices = load_config_with_notices(write(tmp_path, old))
    assert cfg.analysis.halves_tolerance_kT == 1.0 and "convergence_tolerance_sigma" not in cfg.analysis.model_dump()
    assert len(notices) == 1 and notices[0].startswith(
        "analysis.convergence_tolerance_sigma: 2.0 is a setting of an earlier version and is not used")
    new_settings = ("analysis.min_effective_samples", "analysis.halves_tolerance_kT",
                    "analysis.estimators_tolerance_kcal_mol")
    assert all(name in notices[0] for name in new_settings)
    assert load_config(write(tmp_path, old)).analysis.min_overlap == 0.03   # the plain loader reads it as well
    cli._load(str(write(tmp_path, old)))                                    # the commands print the notice
    assert "note: " in capsys.readouterr().err

    # a value the user had chosen cannot be carried over: stop and say what the settings are now
    old["analysis"]["convergence_tolerance_sigma"] = 3.0
    with pytest.raises(ValueError, match="cannot be turned into kT or kcal/mol") as stopped:
        load_config(write(tmp_path, old))
    assert all(name in str(stopped.value) for name in new_settings)

    # the other setting that an earlier version wrote and this one does not have
    older = copy.deepcopy(full)
    older["lambda_schedule"]["adaptive"]["equilibration_check_ps"] = 250.0
    cfg, notices = load_config_with_notices(write(tmp_path, older))
    assert len(notices) == 1 and "equilibration_check_ps" in notices[0]
    older["lambda_schedule"]["adaptive"]["equilibration_check_ps"] = 100.0
    with pytest.raises(ValueError, match="no longer adds windows during the equilibration"):
        load_config(write(tmp_path, older))

    # a key that no version had is still an error, and a file of this version gives no notice
    with pytest.raises(Exception):
        load_config(write(tmp_path, {**base, "analysis": {"convergence_tolerance": 2.0}}))
    assert load_config_with_notices(write(tmp_path, full))[1] == []
    # a record of an earlier run is read as it is written, whatever it holds
    assert read_settings(write(tmp_path, old))["analysis"]["convergence_tolerance_sigma"] == 3.0
    assert read_settings(tmp_path / "nothing.yaml") is None


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


def test_the_numbers_of_time_blocks_for_the_standard_error_are_settings(tmp_path):
    from cufep.analysis.mbar import BLOCK_COUNTS, PART_BLOCK_COUNT
    from cufep.config.schema import load_config_with_notices
    from cufep.workflow import analysis_record, analysis_settings_of

    cfg = load_config(write(tmp_path, BASE))
    assert tuple(cfg.analysis.error_blocks) == BLOCK_COUNTS == (40, 20, 10, 5)
    assert cfg.analysis.part_error_blocks == PART_BLOCK_COUNT == 5
    other = load_config(write(tmp_path, {**BASE, "analysis": {"error_blocks": [20, 8], "part_error_blocks": 4}}))
    record = analysis_record(other)
    assert record["error_blocks"] == [20, 8] and record["part_error_blocks"] == 4
    assert record != analysis_record(cfg)                       # a result made with others is made again
    assert analysis_settings_of(record).analysis.error_blocks == [20, 8]
    for bad in ({"error_blocks": []}, {"error_blocks": [10, 1]}, {"part_error_blocks": 1}):
        with pytest.raises(Exception):
            load_config(write(tmp_path, {**BASE, "analysis": bad}))
    # the name of an earlier version is the setting of this one, in files and in stored records alike
    # (review of PR 1: a record with 2 was read as 5 without a word, a file with 3 was refused)
    for value in (5, 3):
        cfg, notices = load_config_with_notices(write(tmp_path, {**BASE, "analysis": {"halves_error_blocks": value}}))
        assert cfg.analysis.part_error_blocks == value
        assert notices[0].startswith(f"analysis.halves_error_blocks: {value} is analysis.part_error_blocks in this version")
    assert analysis_settings_of({"version": 4, "halves_error_blocks": 2}).analysis.part_error_blocks == 2


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


# ---- review of PR 1 on a3462a5: settings that contradict each other are refused when read ----------------

def test_barostat_frequency_is_a_setting_and_every_shared_stretch_of_md_gets_moves(tmp_path):
    """Review of PR 1 (P1): the barostat counts the steps of a Context that several replicas share, so
    no replica may get fewer MD steps in one go than ``dynamics.barostat_frequency``."""
    assert load_config(write(tmp_path, BASE)).dynamics.barostat_frequency == 25
    # 0.02 ps at 4 fs are 5 steps per replica and iteration, the reviewer's case
    with pytest.raises(Exception, match=r"time_per_iteration_ps .* 5 MD step\(s\), fewer than "
                                        r"dynamics.barostat_frequency = 25"):
        load_config(write(tmp_path, {**BASE, "sampling": {"time_per_iteration_ps": 0.02}}))
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"time_per_iteration_ps": 0.02},
                                       "dynamics": {"barostat_frequency": 5}}))
    assert cfg.dynamics.barostat_frequency == 5
    # the stretches of the precalculation count only where the lambda schedule uses them
    with pytest.raises(Exception, match="walk_ps"):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {"walk_ps": 0.05}}}))
    load_config(write(tmp_path, {**BASE, "lambda_schedule": {"mode": "fixed", "adaptive": {"walk_ps": 0.05}}}))
    with pytest.raises(Exception, match="fixed_walk_ps"):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": {"mode": "fixed", "fixed_walk_ps": 0.05}}))
    with pytest.raises(Exception):
        load_config(write(tmp_path, {**BASE, "dynamics": {"barostat_frequency": 0}}))


def test_the_tail_correction_must_begin_beyond_the_real_space_cutoff(tmp_path):
    """Review of PR 1 (P2): ``lrc_cutoff_nm`` 0.45 nm with a cutoff of 0.9 nm counted the dispersion
    between 0.45 and 0.9 nm, which the MD has, a second time."""
    with pytest.raises(Exception, match=r"analysis.lrc_cutoff_nm = 0.45 nm must be larger than dynamics.cutoff_nm"):
        load_config(write(tmp_path, {**BASE, "analysis": {"lrc_cutoff_nm": 0.45}}))
    with pytest.raises(Exception, match="must be larger"):
        load_config(write(tmp_path, {**BASE, "analysis": {"lrc_cutoff_nm": 0.9}}))
    load_config(write(tmp_path, {**BASE, "analysis": {"lrc_cutoff_nm": 0.45, "lrc": "none"}}))


def test_the_water_model_is_one_cufep_can_build(tmp_path):
    """Review of PR 1 (P2): ``amber14/tip4pew.xml`` was solvated with 3-site waters and createSystem failed."""
    load_config(write(tmp_path, {**BASE, "forcefield": {"water": "amber14/tip4pew.xml"}}))
    with pytest.raises(Exception, match="3-site"):
        load_config(write(tmp_path, {**BASE, "forcefield": {"water": "amber14/tip4pew.xml"},
                                     "membrane": {"enabled": True}}))
    with pytest.raises(Exception, match="not a water model cuFEP can build"):
        load_config(write(tmp_path, {**BASE, "forcefield": {"water": "amber14/tip5p.xml"}}))


def test_settings_changed_after_reading_are_checked_before_anything_is_written(tmp_path):
    """Options of the command line change a configuration that was valid when it was read; the run checks
    the whole configuration again before it makes its output directory."""
    from cufep import run as driver
    from cufep.config.schema import ConfigurationError

    cfg = load_config(write(tmp_path, {**BASE, "output": {"dir": str(tmp_path / "out")}}))
    cfg.sampling.time_per_iteration_ps = 0.02           # a section checks its own fields only
    with pytest.raises(ConfigurationError, match="barostat_frequency"):
        driver.run(cfg)
    assert not (tmp_path / "out").exists()
    with pytest.raises(ConfigurationError, match="barostat_frequency"):
        driver.plan(cfg)
