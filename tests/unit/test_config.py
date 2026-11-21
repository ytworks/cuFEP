import pytest
import yaml
from pydantic import ValidationError

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
    assert cfg.sampling.repeats == 1                       # one run per compound
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
    assert cfg.analysis.min_effective_samples == 50.0 and "convergence_tolerance_sigma" not in cfg.analysis.model_dump()
    assert len(notices) == 1 and notices[0].startswith(
        "analysis.convergence_tolerance_sigma: 2.0 is a setting of an earlier version and is not used")
    # the only levels now are the literature's two
    new_settings = ("analysis.min_overlap", "analysis.min_effective_samples")
    assert all(name in notices[0] for name in new_settings)
    assert load_config(write(tmp_path, old)).analysis.min_overlap == 0.03   # the plain loader reads it as well
    cli._load(str(write(tmp_path, old)))                                    # the commands print the notice
    assert "note: " in capsys.readouterr().err

    # a value the user had chosen cannot be carried over: stop and say what the settings are now
    old["analysis"]["convergence_tolerance_sigma"] = 3.0
    with pytest.raises(ValueError, match="no level of the literature") as stopped:
        load_config(write(tmp_path, old))
    assert all(name in str(stopped.value) for name in new_settings)

    # the owner, 2026-11-20: the halves and BAR against MBAR are no longer checks. Their settings are read with a
    # notice at their old default and refused with any other value
    for key in ("halves_tolerance_kT", "estimators_tolerance_kcal_mol"):
        assert key not in full["analysis"]
        removed = copy.deepcopy(full)
        removed["analysis"][key] = 1.0
        cfg, notices = load_config_with_notices(write(tmp_path, removed))
        assert len(notices) == 1 and notices[0].startswith(f"analysis.{key}: 1.0 is a setting of an earlier version")
        assert "levels of the literature" in notices[0]
        removed["analysis"][key] = 2.0
        with pytest.raises(ValueError, match="Remove the line"):
            load_config(write(tmp_path, removed))

    # the owner, 2026-11-20: no verdict with a level of cuFEP's own on the bilayer or on the pose of the ligand; the
    # three levels are read with a notice at their old default and refused with any other value
    for path, default, other in ((("membrane", "area_tolerance"), 0.01, 0.02),
                                 (("membrane", "pocket_water_tolerance"), 1.5, 3.0),
                                 (("restraints", "pose_rmsd_warning_nm"), 0.3, 0.5)):
        assert path[1] not in full[path[0]]
        removed = copy.deepcopy(full)
        removed[path[0]][path[1]] = default
        cfg, notices = load_config_with_notices(write(tmp_path, removed))
        assert len(notices) == 1 and notices[0].startswith(f"{'.'.join(path)}: {default} is a setting of an earlier "
                                                           "version") and "no longer judged" in notices[0]
        removed[path[0]][path[1]] = other
        with pytest.raises(ValueError, match="Remove the line"):
            load_config(write(tmp_path, removed))

    # the other setting that an earlier version wrote and this one does not have
    older = copy.deepcopy(full)
    older["lambda_schedule"]["adaptive"]["equilibration_check_ps"] = 250.0
    cfg, notices = load_config_with_notices(write(tmp_path, older))
    assert len(notices) == 1 and "equilibration_check_ps" in notices[0]
    older["lambda_schedule"]["adaptive"]["equilibration_check_ps"] = 100.0
    with pytest.raises(ValueError, match="no longer adds windows during the equilibration"):
        load_config(write(tmp_path, older))

    # a key that no version had is still an error, and a file of this version gives no notice
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {**base, "analysis": {"convergence_tolerance": 2.0}}))
    assert load_config_with_notices(write(tmp_path, full))[1] == []
    # a record of an earlier run is read as it is written, whatever it holds
    assert read_settings(write(tmp_path, old))["analysis"]["convergence_tolerance_sigma"] == 3.0
    assert read_settings(tmp_path / "nothing.yaml") is None


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "sampling": {"repeat": 2}}))
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "typo": 1}))


def test_membrane_is_an_explicit_switch(tmp_path):
    base = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}
    assert load_config(write(tmp_path, {**base, "membrane": {"enabled": True}})).membrane.enabled is True
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {**base, "membrane": {"enabled": "auto"}}))


def test_missing_inputs_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        Config.model_validate({})


def test_lambda_windows_have_recommended_defaults_and_can_be_set(tmp_path):
    ls = load_config(write(tmp_path, BASE)).lambda_schedule
    assert ls.windows.complex.restrain == 6 and ls.windows.complex.discharge == 11
    assert len(ls.windows.complex.vanish) == 15 and ls.windows.solvent.discharge == 5
    assert len(ls.windows.solvent.vanish) == 10
    a = ls.adaptive
    # the solvent leg: 0.18 of the matrix of the stage (the owner, 2026-11-20; Midgley et al. 2025
    # use 0.04). The complex leg: 0.30 of the overlap of a pair alone (the owner, 2026-11-22)
    assert (a.threshold("solvent"), a.matrix("solvent")) == (0.18, "stage")
    assert (a.threshold("complex"), a.matrix("complex")) == (0.30, "pair")
    assert a.candidates == 101
    # the precalculation: the paper's 50 ps in the solvent leg, 500 ps after 500 ps in the complex leg
    assert (a.window("solvent"), a.equilibration("solvent")) == (50.0, 0.0)
    assert (a.window("complex"), a.equilibration("complex")) == (500.0, 500.0)
    assert ls.fixed_walk_ps == 50.0
    # the complex leg starts from the recommended windows of mode fixed (the owner, 2026-11-22)
    assert a.initial_windows.for_leg("complex") == ls.windows.for_leg("complex")
    assert a.initial_windows.for_leg("solvent") == {"discharge": 5, "vanish": 9}
    # the owner, 2026-11-21: 100 in both legs (48 and 32 until then)
    assert (a.max_windows.complex, a.max_windows.solvent) == (100, 100)
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
    # the start of the complex leg is a setting of its own: it does not follow `windows`
    assert cfg.lambda_schedule.adaptive.initial_windows.for_leg("complex") == ls.windows.for_leg("complex")


def test_initial_windows_are_a_number_or_lambda_values(tmp_path):
    """The start of adaptive lambda scheduling: evenly spaced windows by their number, or the
    lambda values themselves (the owner, 2026-11-21: the start need not be evenly spaced)."""
    from cufep.cli import lambda_window_table

    values = [0.0, 0.5, 0.75, 0.82, 0.88, 1.0]
    cfg = load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {
        "initial_windows": {"complex": {"vanish": values}, "solvent": {"discharge": 7}}}}}))
    a = cfg.lambda_schedule.adaptive
    assert a.initial_windows.for_leg("complex") == {"restrain": 6, "discharge": 11, "vanish": values}
    assert a.initial_windows.for_leg("solvent") == {"discharge": 7, "vanish": 9}
    assert ("start: complex leg restrain 6, discharge 11, vanish 6 (0 0.5 0.75 0.82 0.88 1); "
            "solvent leg discharge 7, vanish 9;") in lambda_window_table(cfg)


@pytest.mark.parametrize("adaptive, why", [
    ({"initial_windows": {"complex": {"vanish": [0.0, 0.125, 1.0]}}},
     "initial_windows.complex.vanish: 0.125 is not one of the 101 candidate lambda values"),
    ({"candidates": 11, "initial_windows": {"solvent": {"vanish": 12}}},
     "initial_windows.solvent.vanish: 12 windows do not fit on 11 candidates"),
    ({"initial_windows": {"solvent": {"discharge": 60, "vanish": 60}}},
     "gives the solvent leg 119 lambda windows to start with .*more than lambda_schedule.adaptive.max_windows."
     "solvent = 100"),
    ({"initial_windows": {"complex": {"vanish": [0.1, 1.0]}}}, "must increase from 0.0"),
    ({"initial_windows": {"complex": {"vanish": [0.0, 0.5, 0.5, 1.0]}}}, "must increase from 0.0"),
    ({"initial_windows": {"complex": {"vanish": 1}}}, "a stage needs at least two lambda windows"),
    # the default start of the complex leg is 30 windows given as values (review of the defaults of
    # 2026-11-22): a setting that it does not fit is refused, and the message says whose start it is
    ({"candidates": 51},
     r"initial_windows.complex.vanish: 0.65 is not one of the 51 candidate lambda values .*; this is the default "
     r"start of the complex leg: give lambda_schedule.adaptive.initial_windows.complex for these settings as well"),
    ({"max_windows": {"complex": 29}},
     r"gives the complex leg 30 lambda windows to start with .*max_windows.complex = 29; this is the default start "
     r"of the complex leg"),
])
def test_a_start_the_precalculation_cannot_use_is_refused_when_the_configuration_is_read(tmp_path, adaptive, why):
    """Before this, a number of windows that does not fit on the candidates stopped the leg after
    its pre-equilibration; now no command gets as far as writing anything."""
    with pytest.raises(ValidationError, match=why):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": adaptive}}))
    # a configuration that runs no precalculation does not use the start
    load_config(write(tmp_path, {**BASE, "lambda_schedule": {"mode": "fixed", "adaptive": {
        "candidates": 11, "initial_windows": {"solvent": {"vanish": 12}}}}}))
    # nor does a leg that runs none: the default start of the complex leg with settings it does not fit
    load_config(write(tmp_path, {**BASE, "lambda_schedule": {
        "mode": {"complex": "fixed", "solvent": "adaptive"},
        "adaptive": {"candidates": 51, "max_windows": {"complex": 29}}}}))


def test_window_settings_are_one_value_or_one_per_leg_and_reach_the_precalculation(tmp_path):
    """The owner, 2026-11-21: the complex leg and the solvent leg can run different window
    settings, set in the YAML. One value is for both legs; ``{complex: ..., solvent: ...}`` gives
    each leg its own, and the precalculation of a leg is run with the values of that leg."""
    from cufep.cli import lambda_window_table
    from cufep.workflow import als_settings

    default = load_config(write(tmp_path, BASE))
    c, s = als_settings(default, "complex"), als_settings(default, "solvent")
    # the defaults (the owner, 2026-11-22): the complex leg reads the overlap of a pair alone and
    # starts from the recommended windows of mode fixed; the solvent leg is as the paper starts
    assert (c.overlap_threshold, c.overlap_matrix, c.candidates, c.max_windows) == (0.3, "pair", 101, 100)
    assert (s.overlap_threshold, s.overlap_matrix, s.candidates, s.max_windows) == (0.18, "stage", 101, 100)
    assert c.initial_windows == default.lambda_schedule.windows.for_leg("complex")
    assert s.initial_windows == {"discharge": 5, "vanish": 9}
    assert default.lambda_schedule.precalculated_legs() == ["complex", "solvent"]
    assert ("  a window is added where the overlap of neighbouring windows is complex leg at or below 0.3 "
            "(the overlap of the two windows alone); solvent leg at or below 0.18;\n"
            "  101 candidate lambda values per stage; MD per new window: complex leg 500 ps after 500 ps of "
            "equilibration; solvent leg 50 ps;") in lambda_window_table(default)

    # one value is for both legs (21 candidates: the default start of the complex leg lies on them)
    one = load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {
        "overlap_threshold": 0.3, "overlap_matrix": "pair", "candidates": 21}}}))
    for leg in ("complex", "solvent"):
        s = als_settings(one, leg, 1.5)
        assert (s.overlap_threshold, s.overlap_matrix, s.candidates, s.rest2_scale) == (0.3, "pair", 21, 1.5)

    per_leg = load_config(write(tmp_path, {**BASE, "lambda_schedule": {
        "mode": {"complex": "adaptive", "solvent": "fixed"},
        "adaptive": {"overlap_threshold": {"complex": 0.3, "solvent": 0.18},
                     "overlap_matrix": {"complex": "pair", "solvent": "stage"},
                     "candidates": {"complex": 101, "solvent": 51},
                     "initial_windows": {"complex": {"vanish": [0.0, 0.5, 0.75, 1.0]}},
                     "max_windows": {"complex": 60, "solvent": 20}}}}))
    ls = per_leg.lambda_schedule
    assert (ls.mode_of("complex"), ls.mode_of("solvent")) == ("adaptive", "fixed")
    assert ls.precalculated_legs() == ["complex"]
    c, s = als_settings(per_leg, "complex"), als_settings(per_leg, "solvent")
    assert (c.overlap_threshold, c.overlap_matrix, c.candidates, c.max_windows) == (0.3, "pair", 101, 60)
    assert (s.overlap_threshold, s.overlap_matrix, s.candidates, s.max_windows) == (0.18, "stage", 51, 20)
    assert c.initial_windows == {"restrain": 6, "discharge": 11, "vanish": [0.0, 0.5, 0.75, 1.0]}
    assert (c.window_ps, c.equilibration_ps, s.window_ps, s.equilibration_ps) == (500.0, 500.0, 50.0, 0.0)
    table = lambda_window_table(per_leg)
    assert "mode adaptive in the complex leg:" in table and "mode fixed in the solvent leg:" in table
    assert "is at or below 0.3 (the overlap of the two windows alone);" in table
    assert "at most 60 (complex) windows." in table

    differ = lambda_window_table(load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {
        "overlap_threshold": {"complex": 0.3, "solvent": 0.18},
        "overlap_matrix": {"complex": "pair", "solvent": "stage"}}}})))
    assert ("is complex leg at or below 0.3 (the overlap of the two windows alone); "
            "solvent leg at or below 0.18;") in differ


def test_a_threshold_and_the_overlap_matrix_it_is_compared_with_belong_together(tmp_path):
    """Review of the defaults of 2026-11-22: the two settings have a default per leg, so a file that
    gives one of them gets the other from the defaults. A matrix alone brings the threshold that
    belongs to it (``overlap_matrix: stage`` alone kept 0.30 in the complex leg, which the matrix of
    a stage does not reach). A threshold alone is compared with the default matrix of each leg, and
    the file is told: files written before ``overlap_matrix`` existed meant the stage in both legs."""
    from cufep.cli import lambda_window_table
    from cufep.config.schema import load_config_with_notices
    from cufep.workflow import als_settings

    def read(adaptive):
        cfg, notices = load_config_with_notices(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": adaptive}}))
        return cfg, [(s.overlap_threshold, s.overlap_matrix) for s in (als_settings(cfg, "complex"),
                                                                       als_settings(cfg, "solvent"))], notices

    cfg, legs, notices = read({"overlap_matrix": "stage"})
    assert legs == [(0.18, "stage"), (0.18, "stage")] and notices == []
    assert "neighbouring windows is at or below 0.18;" in lambda_window_table(cfg)
    # the resolved configuration holds that threshold, so it reads the same again
    again = Config.model_validate(cfg.model_dump(mode="json")).lambda_schedule.adaptive
    assert (again.threshold("complex"), again.matrix("complex")) == (0.18, "stage")
    assert read({"overlap_matrix": "pair"})[1] == [(0.3, "pair"), (0.3, "pair")]
    assert read({"overlap_matrix": {"complex": "stage", "solvent": "pair"}})[1] == [(0.18, "stage"), (0.3, "pair")]

    _cfg, legs, notices = read({"overlap_threshold": 0.2})
    assert legs == [(0.2, "pair"), (0.2, "stage")]
    assert len(notices) == 1 and notices[0].startswith(
        "lambda_schedule.adaptive.overlap_threshold is given without overlap_matrix: in the complex leg the "
        "threshold (0.2) is compared with the overlap of the two windows of a pair alone")
    assert "overlap_matrix: stage beside the threshold does that" in notices[0]
    # with the matrix beside it, or where no leg reads a pair, nothing is to be said
    assert read({"overlap_threshold": 0.2, "overlap_matrix": "stage"})[1:] == ([(0.2, "stage"), (0.2, "stage")], [])
    _cfg, notices = load_config_with_notices(write(tmp_path, {**BASE, "lambda_schedule": {
        "mode": {"complex": "fixed", "solvent": "adaptive"}, "adaptive": {"overlap_threshold": 0.2}}}))
    assert notices == []


@pytest.mark.parametrize("section", [
    {"adaptive": {"overlap_threshold": {"complex": 0.3}}},                                # one leg only
    {"adaptive": {"overlap_matrix": {"complex": "pair", "solvent": "stage", "membrane": "pair"}}},
    {"adaptive": {"overlap_matrix": "neighbours"}},
    {"adaptive": {"overlap_threshold": {"complex": 0.6, "solvent": 0.18}}},               # no overlap is above 0.5
    {"adaptive": {"overlap_threshold": 0.5}},
    {"adaptive": {"candidates": {"complex": 2, "solvent": 101}}},
    {"mode": {"complex": "adaptive"}},
    {"mode": {"complex": "adaptive", "solvent": "automatic"}},
])
def test_window_settings_per_leg_name_both_legs_and_nothing_else(tmp_path, section):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": section}))


def test_precalculation_lengths_are_one_number_or_one_per_leg(tmp_path):
    one = load_config(
        write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {"window_ps": 20, "equilibration_ps": 0}}})
    )
    a = one.lambda_schedule.adaptive
    assert (a.window("complex"), a.window("solvent"), a.equilibration("complex")) == (20.0, 20.0, 0.0)
    per_leg = load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {
        "window_ps": {"complex": 200, "solvent": 50}, "equilibration_ps": {"complex": 100, "solvent": 0}}}}))
    a = per_leg.lambda_schedule.adaptive
    assert (a.window("complex"), a.window("solvent")) == (200.0, 50.0)
    assert (a.equilibration("complex"), a.equilibration("solvent")) == (100.0, 0.0)
    for bad in ({"window_ps": 0}, {"window_ps": {"complex": 500, "solvent": 0}}, {"equilibration_ps": -1},
                {"window_ps": {"complex": 500}}):
        with pytest.raises(ValidationError):
            load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": bad}}))


def test_rest2_is_on_by_default_and_can_be_set(tmp_path):
    from cufep.cli import lambda_window_table

    cfg = load_config(write(tmp_path, BASE))
    r2 = cfg.sampling.rest2
    # the pocket is the default, the owner's choice of 2026-11-20 (SOMD2 heats the ligand alone)
    assert (r2.scale, r2.region, r2.pocket_cutoff_nm, r2.residues) == (1.5, "pocket", 0.4, [])
    assert "hot region: the ligand and protein residues within 0.4 nm of it; 1.5 times the temperature" \
        in lambda_window_table(cfg)
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"rest2": {"region": "ligand"}}}))
    assert "hot region: the ligand; 1.5 times the temperature" in lambda_window_table(cfg)
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"rest2": {"scale": 1.0}}}))
    assert "off (sampling.rest2.scale is 1)" in lambda_window_table(cfg)
    cfg = load_config(
        write(tmp_path, {**BASE, "sampling": {"rest2": {"scale": 2, "region": "pocket", "residues": ["A:960"]}}})
    )
    assert "protein residues within 0.4 nm of it (A:960)" in lambda_window_table(cfg)
    for bad in ({"scale": 0.9}, {"region": "protein"}, {"pocket_cutoff_nm": 0}):
        with pytest.raises(ValidationError):
            load_config(write(tmp_path, {**BASE, "sampling": {"rest2": bad}}))


def test_one_value_of_a_section_can_be_set_and_the_rest_keep_their_defaults(tmp_path):
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"pre_equilibration": {"complex": {"md_ns": 1.0},
                                                                                 "solvent": {"nvt_ns": 0.05}}}}))
    pe = cfg.sampling.pre_equilibration
    assert (pe.complex.nvt_ns, pe.complex.npt_ns, pe.complex.md_ns) == (0.25, 0.5, 1.0)
    assert (pe.solvent.nvt_ns, pe.solvent.npt_ns, pe.solvent.md_ns) == (0.05, 0.2, 0.5)


@pytest.mark.parametrize("bad", [1, 0, [0.0], [0.1, 1.0], [0.0, 0.9], [0.0, 0.6, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0]])
def test_bad_lambda_windows_are_rejected(tmp_path, bad):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": {"windows": {"solvent": {"vanish": bad}}}}))


@pytest.mark.parametrize("section", [{"adaptive": {"overlap_threshold": 0.0}}, {"adaptive": {"candidates": 2}},
                                     {"adaptive": {"initial_windows": {"complex": {"vanish": 1}}}},
                                     {"adaptive": {"initial_windows": {"vanish": 9}}}, {"mode": "automatic"},
                                     {"preset": "fast"}])
def test_bad_schedule_settings_are_rejected(tmp_path, section):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, {**BASE, "lambda_schedule": section}))


def test_lengths_per_leg(tmp_path):
    cfg = load_config(write(tmp_path, {**BASE, "sampling": {"production_ns": {"complex": 10, "solvent": 2},
                                                           "equilibration_ns": 0.5, "repeats": 3}}))
    assert cfg.sampling.production("complex") == 10.0 and cfg.sampling.production("solvent") == 2.0
    assert cfg.sampling.equilibration("complex") == 0.5 == cfg.sampling.equilibration("solvent")
    assert cfg.sampling.repeats == 3
    for bad in ({"production_ns": 0}, {"production_ns": {"complex": 5}}, {"repeats": 0}, {"production_ns": [5, 10]}):
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
            load_config(write(tmp_path, {**BASE, "analysis": bad}))
    # the name of an earlier version is the setting of this one, in files and in stored records alike
    # (review of PR 1: a record with 2 was read as 5 without a word, a file with 3 was refused)
    for value in (5, 3):
        cfg, notices = load_config_with_notices(write(tmp_path, {**BASE, "analysis": {"halves_error_blocks": value}}))
        assert cfg.analysis.part_error_blocks == value
        assert notices[0].startswith(
            f"analysis.halves_error_blocks: {value} is analysis.part_error_blocks in this version"
        )
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
    # a window of the precalculation starts from a configuration of its own in a Context it shares, so it
    # counts; the walks through the windows are one trajectory, and no other shares its Context
    with pytest.raises(Exception, match="window_ps"):
        load_config(
            write(
                tmp_path,
                {
                    **BASE,
                    "lambda_schedule": {
                        "adaptive": {"window_ps": 0.05, "equilibration_ps": 0.0, "sample_interval_ps": 0.025}
                    },
                },
            )
        )
    load_config(write(tmp_path, {**BASE, "lambda_schedule": {"mode": "fixed", "adaptive": {
        "window_ps": 0.05, "equilibration_ps": 0.0, "sample_interval_ps": 0.025}}}))
    load_config(write(tmp_path, {**BASE, "lambda_schedule": {"adaptive": {"walk_ps": 0.05}}}))
    load_config(write(tmp_path, {**BASE, "lambda_schedule": {"mode": "fixed", "fixed_walk_ps": 0.05}}))
    with pytest.raises(ValidationError):
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


def test_the_restraint_from_interactions_is_a_selection_with_settings_of_its_own():
    """``restraints.selection`` is interactions (the default), auto or manual. The section
    ``restraints.interactions`` holds the MD and the thresholds of that selection; a configuration that
    would analyse no frame, or whose thresholds are less than one frame, is refused when it is read."""
    from cufep.config.schema import analysed_frames

    chosen = Config.model_validate(BASE)
    assert chosen.restraints.selection == "interactions"
    assert Config.model_validate({**BASE, "restraints": {"selection": "auto"}}).restraints.selection == "auto"
    ri = chosen.restraints.interactions
    assert (ri.trajectories, ri.length_ns, ri.discard_ns, ri.frame_interval_ps) == (4, 3.0, 1.0, 10.0)
    assert analysed_frames(chosen) == 800                      # 4 x 200 frames later than 1 ns, at 4 fs
    at_3_fs = Config.model_validate({**BASE, "restraints": {"selection": "interactions"},
                                     "dynamics": {"timestep_fs": 3.0}})
    assert analysed_frames(at_3_fs) == 804                     # a frame interval of 3333 steps is 9.999 ps

    def with_settings(**settings):
        return Config.model_validate({**BASE, "restraints": {"selection": "interactions", "interactions": settings}})

    longer = with_settings(trajectories=1, length_ns=12.0)
    assert analysed_frames(longer) == 1100
    for settings, why in (({"discard_ns": 3.0}, "leaves nothing of a trajectory"),
                          ({"angle_range_deg": [135.0, 45.0]}, "the lower first"),
                          ({"trajectories": 0}, "greater than or equal to 1"),
                          ({"length_ns": 0.02, "discard_ns": 0.0, "trajectories": 1}, "is less than one frame"),
                          ({"length_ns": 0.05, "discard_ns": 0.0, "trajectories": 1}, "fallback_frame_fraction")):
        with pytest.raises(ValidationError, match=why):
            with_settings(**settings)
    # the thresholds are not read for another selection
    Config.model_validate({**BASE, "restraints": {"selection": "auto", "interactions": {
        "length_ns": 0.02, "discard_ns": 0.0, "trajectories": 1}}})
    with pytest.raises(ValidationError):
        Config.model_validate({**BASE, "restraints": {"selection": "prolif"}})


def test_plan_says_how_the_restraint_is_chosen():
    from cufep.cli import restraint_selection_lines

    said = restraint_selection_lines(Config.model_validate(BASE))
    assert "4 trajectories of 3 ns from the pose that was given (held while the solvent settles), a frame every " \
           "10 ps, the first 1 ns of each left out: 800 frames" in said
    assert "on the backend of the leg (cupotential, fallback openmm)" in said
    assert "kept in 320 frames or more (0.4), anchor angles within 45 to 135 degrees" in said
    assert "within 1 nm of the ligand in 120 frames or more (0.15)" in said
    assert "selection auto" in restraint_selection_lines(
        Config.model_validate({**BASE, "restraints": {"selection": "auto"}}))


def test_a_fraction_of_the_frames_is_rounded_down_as_it_is_written():
    """Review of eb6f89f (P1). "Frame-count thresholds are rounded down to integers": 0.29 of 800 frames is
    232. The product of the two binary numbers is 231.99999999999997, and its integer part kept an interaction
    of 231 frames; of the fractions of two decimals, 14 were one frame too low for 20 to 1200 frames. The rule,
    the check of the configuration and `cufep plan` round with this one function."""
    from cufep.alchemy import interaction_anchors
    from cufep.config.schema import frames_needed

    assert interaction_anchors.frames_needed is frames_needed
    assert int(800 * 0.29) == 231 and frames_needed(800, 0.29) == 232
    assert (frames_needed(800, 0.40), frames_needed(800, 0.15), frames_needed(804, 0.40), frames_needed(799, 0.40)) \
        == (320, 120, 321, 319)
    for n in (20, 100, 300, 800, 804, 1200):
        for hundredths in range(1, 100):
            assert frames_needed(n, hundredths / 100) == n * hundredths // 100
    assert frames_needed(800, 0.3999999999) == 319 and frames_needed(5, 0.15) == 0 and frames_needed(800, 1e-05) == 0

