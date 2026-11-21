"""Lambda windows from the configuration, and the rule of adaptive lambda scheduling.

The adaptive rule is the one of Midgley et al., J. Chem. Inf. Model. 2025, 65, 512; the
module docstring of cufep.alchemy.schedule states it step by step.
"""
import numpy as np
import openmm as mm
import pytest
import yaml

from cufep.alchemy.factory import make_alchemical_system
from cufep.alchemy.schedule import (
    AlsSettings,
    Schedule,
    adaptive_schedule,
    als_step,
    bad_sections,
    choose_section,
    fixed_schedule,
    initial_indices,
    initial_window_indices,
    pair_overlap,
    redistribute,
    run_window,
    schedule_from_windows,
    stage_overlap,
    stage_values,
)
from cufep.config.schema import RECOMMENDED_VANISH_COMPLEX, RECOMMENDED_VANISH_SOLVENT, WindowsConfig
from cufep.sampling.states import AlchemicalState

T = 0.04


# ---------------------------------------------------------------- windows from the configuration
def test_recommended_windows_are_those_of_openfe():
    rec = WindowsConfig()
    cx = schedule_from_windows(["restrain", "discharge", "vanish"], rec.for_leg("complex"), rec.for_leg("complex"))
    assert len(cx.states) == 30
    assert cx.windows() == {"restrain": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            "discharge": [round(0.1 * i, 6) for i in range(11)],
                            "vanish": RECOMMENDED_VANISH_COMPLEX}
    assert cx.states[0] == AlchemicalState(0.0, 1.0, 1.0)             # no restraint, fully interacting
    assert cx.states[5] == AlchemicalState(1.0, 1.0, 1.0)             # end of restrain = start of discharge
    assert cx.states[15] == AlchemicalState(1.0, 0.0, 1.0)            # end of discharge = start of vanish
    assert cx.states[-1] == AlchemicalState(1.0, 0.0, 0.0)
    sv = schedule_from_windows(["discharge", "vanish"], rec.for_leg("solvent"), rec.for_leg("solvent"))
    assert len(sv.states) == 14
    assert sv.windows() == {"discharge": [0.0, 0.25, 0.5, 0.75, 1.0], "vanish": RECOMMENDED_VANISH_SOLVENT}


def test_a_number_of_windows_keeps_the_recommended_spacing():
    assert stage_values(5, 11) == [0.0, 0.25, 0.5, 0.75, 1.0]                      # evenly spaced stays even
    assert stage_values(15, RECOMMENDED_VANISH_COMPLEX) == RECOMMENDED_VANISH_COMPLEX
    more = stage_values(29, RECOMMENDED_VANISH_COMPLEX)
    assert len(more) == 29 and more[0] == 0.0 and more[-1] == 1.0
    assert more[::2] == pytest.approx(RECOMMENDED_VANISH_COMPLEX)                  # the old values, one new in between
    fewer = np.array(stage_values(8, RECOMMENDED_VANISH_COMPLEX))
    steps = np.diff(fewer)
    assert steps[0] > steps[-1]                                                    # still denser at the end
    assert stage_values([0.0, 0.3, 1.0], 6) == [0.0, 0.3, 1.0]                     # explicit values are used as given
    with pytest.raises(ValueError):
        stage_values(1, 6)


def test_windows_given_as_numbers_or_values():
    rec = WindowsConfig().for_leg("solvent")
    s = schedule_from_windows(["discharge", "vanish"], {"discharge": 3, "vanish": [0.0, 0.5, 0.9, 1.0]}, rec)
    assert s.windows() == {"discharge": [0.0, 0.5, 1.0], "vanish": [0.0, 0.5, 0.9, 1.0]}
    assert [st.lambda_sterics for st in s.states] == [1.0, 1.0, 1.0, 0.5, pytest.approx(0.1), 0.0]


def test_schedule_file_round_trip_and_old_stage_names(tmp_path):
    s = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 5})
    assert len(s.states) == 9
    assert s.states[0] == AlchemicalState(1.0, 1.0, 1.0) and s.states[-1] == AlchemicalState(1.0, 0.0, 0.0)
    s.save(tmp_path / "schedule.yaml", leg="solvent")
    again = Schedule.load(tmp_path / "schedule.yaml")
    assert again.states == s.states and again.stages == s.stages and again.windows() == s.windows()
    data = yaml.safe_load((tmp_path / "schedule.yaml").read_text())
    data["stages"] = ["electrostatics", "sterics"]                    # as written by earlier versions
    data["progress"] = data.pop("stage_lambda")
    (tmp_path / "old.yaml").write_text(yaml.safe_dump(data))
    old = Schedule.load(tmp_path / "old.yaml")
    assert old.stages == ["discharge", "vanish"] and old.states == s.states and old.windows() == s.windows()
    assert fixed_schedule(["electrostatics", "sterics"], {"electrostatics": 3, "sterics": 5}).states == s.states


# ---------------------------------------------------------------- the adaptive rule
def test_nine_initial_windows_on_101_candidates():
    assert initial_indices(9, 101) == [0, 12, 25, 38, 50, 62, 75, 88, 100]        # 0.12 is the value in the paper
    assert initial_indices(3, 101) == [0, 50, 100] and initial_indices(5, 101) == [0, 25, 50, 75, 100]
    with pytest.raises(ValueError):
        initial_indices(1)


def test_a_stage_starts_from_a_number_of_windows_or_from_given_candidates():
    assert initial_window_indices(5, 21) == initial_indices(5, 21) == [0, 5, 10, 15, 20]
    assert initial_window_indices([0.0, 0.5, 0.8, 0.9, 1.0], 21) == [0, 10, 16, 18, 20]      # not evenly spaced
    assert initial_window_indices([0.0, 0.29, 0.58, 1.0], 101) == [0, 29, 58, 100]            # 0.29 * 100 = 28.99...
    for bad, why in (([0.0, 0.125, 1.0], "0.125 is not one of the 101 candidate lambda values"),
                     ([0.0, 0.5], "increase from 0.0"), ([0.5, 1.0], "increase from 0.0"),
                     ([0.0, 0.5, 0.5, 1.0], "increase from 0.0"), ([0.0], "increase from 0.0")):
        with pytest.raises(ValueError, match=why):
            initial_window_indices(bad, 101)
    with pytest.raises(ValueError, match="12 windows do not fit on 11 candidates"):
        initial_window_indices(12, 11)
    # Reviews of PR 7: lambda values are written with six decimals, and what one run wrote as its
    # start (or its windows) was refused as the start of another unless a candidate step is a
    # decimal: "0.266667 is not one of the 31 candidate lambda values".
    assert initial_window_indices([0.0, 0.266667, 0.5, 0.733333, 1.0], 31) == [0, 8, 15, 22, 30]
    assert initial_window_indices([0.0, 0.333333, 0.666667, 1.0], 100) == initial_indices(4, 100) == [0, 33, 66, 99]
    assert initial_window_indices([0.0, 0.333333, 0.666667, 1.0], 7) == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="0.33334 is not one of the 100 candidate lambda values"):
        initial_window_indices([0.0, 0.33334, 1.0], 100)                           # not to six decimals


def test_a_pair_is_deficient_at_or_below_the_threshold():
    assert bad_sections([0.2, 0.05, 0.3], T) == []
    assert bad_sections([0.2, 0.04, 0.3], T) == [(1, 2)]                           # exactly at the threshold
    assert bad_sections([0.01, 0.02, 0.3, 0.03], T) == [(0, 2), (3, 4)]
    assert bad_sections([0.3, float("nan")], T) == [(1, 2)]


def test_the_example_of_the_paper():
    """A bad section with lambda 0.00, 0.06, 0.12 becomes 0.00, 0.04, 0.08, 0.12."""
    windows = [0, 6, 12, 25, 38, 50, 62, 75, 88, 100]
    sup = [0.01, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    new, rec = als_step(windows, sup, T)
    assert new == [0, 4, 8, 12, 25, 38, 50, 62, 75, 88, 100]
    assert rec["status"] == "inserted" and rec["section"] == [0, 12] and rec["new_windows"] == [4, 8]
    assert len(new) == len(windows) + 1                                            # one window per iteration


def test_three_iterations_as_in_the_plan():
    w = initial_indices(9, 101)
    w, rec = als_step(w, [0.20, 0.15, 0.09, 0.03, 0.02, 0.06, 0.03, 0.10], T)
    assert rec["bad_sections"] == [[38, 62], [75, 88]] and rec["section"] == [38, 62]        # the longest one
    assert w == [0, 12, 25, 38, 46, 54, 62, 75, 88, 100] and rec["new_windows"] == [46, 54]
    w, rec = als_step(w, [0.20, 0.15, 0.09, 0.08, 0.05, 0.07, 0.06, 0.03, 0.10], T)
    assert rec["section"] == [75, 88] and rec["new_windows"] == [82]
    assert w == [0, 12, 25, 38, 46, 54, 62, 75, 82, 88, 100]
    new, rec = als_step(w, [0.20, 0.15, 0.09, 0.08, 0.05, 0.07, 0.06, 0.07, 0.06, 0.10], T)
    assert new is None and rec["status"] == "converged"


def test_equally_long_sections_the_lowest_overlap_decides():
    sup = [0.2, 0.03, 0.2, 0.01, 0.2]
    sections = bad_sections(sup, T)
    assert sections == [(1, 2), (3, 4)] and choose_section(sections, sup) == (3, 4)
    assert choose_section(bad_sections([0.03, 0.2, 0.01, 0.02], T), [0.03, 0.2, 0.01, 0.02]) == (2, 4)   # longer wins


def test_a_section_without_free_candidates_is_skipped():
    windows = [0, 1, 2, 50, 100]
    assert redistribute(windows, (0, 2)) is None                                   # 0, 1, 2 cannot take a fourth value
    new, rec = als_step(windows, [0.01, 0.01, 0.2, 0.02], T)
    assert rec["status"] == "inserted" and rec["section"] == [50, 100] and new == [0, 1, 2, 50, 75, 100]
    new, rec = als_step([0, 1, 2], [0.01, 0.01], T)
    assert new is None and rec["status"] == "saturated"


def test_the_window_limit_stops_the_insertion():
    new, rec = als_step([0, 50, 100], [0.01, 0.2], T, room=0)
    assert new is None and rec["status"] == "max_windows"
    new, rec = als_step([0, 50, 100], [0.2, 0.2], T, room=0)
    assert new is None and rec["status"] == "converged"                            # nothing to add anyway


def test_the_rule_ends_with_every_pair_above_the_threshold():
    """A model in which the overlap of two windows is known in closed form.

    Windows are unit-width Gaussians along x(lambda); for equal sample numbers the MBAR overlap
    of two neighbours that only overlap with each other is 0.5 * exp(-d^2 / 8)-like and falls
    monotonically with their distance d. x(lambda) is steep near lambda = 1, as a soft-core
    stage is, so windows must be added there and nowhere else.
    """
    def x(lam):
        return 6.0 * lam + 30.0 * lam ** 6

    def overlap(a, b):
        d = x(b / 100.0) - x(a / 100.0)
        return 0.5 * np.exp(-d * d / 8.0)

    w = initial_indices(9, 101)
    history = [list(w)]
    for _ in range(100):
        sup = [overlap(a, b) for a, b in zip(w, w[1:])]
        new, rec = als_step(w, sup, T)
        if new is None:
            break
        assert len(new) == len(w) + 1 and set(new) >= {0, 100}
        w = new
        history.append(list(w))
    assert rec["status"] == "converged"
    assert min(overlap(a, b) for a, b in zip(w, w[1:])) > T
    assert 9 < len(w) < 30
    assert w[:4] == [0, 12, 25, 38]                                               # nothing added where it was fine
    assert sum(1 for i in w if i > 62) > sum(1 for i in history[0] if i > 62)       # added at the steep end


# ---------------------------------------------------------------- overlap from samples, and the precalculation
def test_stage_overlap_of_harmonic_windows():
    rng = np.random.default_rng(3)
    centres = np.array([0.0, 1.0, 2.0, 8.0])                                       # the last one is far away
    grid = np.linspace(0.0, 8.0, 9)                                                # candidates at 0, 1, ..., 8
    windows = [0, 1, 2, 8]
    energies = [0.5 * (rng.normal(c, 1.0, size=4000)[:, None] - grid[None, :]) ** 2 for c in centres]
    sup = stage_overlap(energies, windows)
    assert sup.shape == (3,)
    assert sup[0] == pytest.approx(sup[1], abs=0.03) and 0.15 < sup[0] < 0.5       # unit distance: good overlap
    assert sup[2] < 0.01                                                           # six sigma apart: none
    assert bad_sections(sup, T) == [(2, 3)]


def harmonic_windows(spacing, length=8.0, samples=4000, seed=5):
    """Unit-width windows along a line of ``length``, ``spacing`` apart: what stage_overlap takes."""
    rng = np.random.default_rng(seed)
    centres = np.arange(0.0, length + 1e-9, spacing)
    energies = [0.5 * (rng.normal(c, 1.0, size=samples)[:, None] - centres[None, :]) ** 2 for c in centres]
    return energies, list(range(len(centres)))


def test_the_overlap_of_a_pair_alone_rises_as_windows_get_closer_and_the_matrix_of_the_stage_falls():
    """Why a threshold above about 0.2 cannot be met in the matrix of a stage, and what
    ``overlap_matrix: pair`` reads instead (the owner's decision of 2026-11-21).

    For unit-width harmonic windows d apart the element of a pair alone is known and rises to
    0.5; in the matrix of the stage it is d * exp(-d^2 / 4) / (2 sqrt(pi)) once windows are close,
    shared among all windows that overlap, with its largest value 0.242 at d = sqrt(2). The
    numbers are those measured with three seeds, which differ by 0.002 at most.
    """
    alone, in_stage = [], []
    for spacing in (2.0, 1.0, 0.5, 0.25):
        energies, windows = harmonic_windows(spacing)
        middle = len(windows) // 2
        alone.append(pair_overlap(energies, windows)[middle])
        in_stage.append(stage_overlap(energies, windows)[middle])
    assert alone == pytest.approx([0.223, 0.400, 0.470, 0.492], abs=0.01)
    assert alone[0] < alone[1] < alone[2] < alone[3] < 0.5
    # in the matrix of the stage: nearly the same while only neighbours overlap, then falling
    assert in_stage == pytest.approx([0.207, 0.220, 0.1325, 0.0694], abs=0.01)
    assert max(in_stage) < 0.25 and in_stage[1] > in_stage[2] > in_stage[3]
    # windows that only overlap with each other: the two matrices give the same number
    energies, windows = harmonic_windows(1.0)
    assert pair_overlap(energies[:2], windows[:2])[0] == stage_overlap(energies[:2], windows[:2])[0]
    assert pair_overlap(energies, windows).shape == (len(windows) - 1,)


def precalculation_of_harmonic_windows(context, integrator, overlap_matrix, max_windows, per_lambda=30.0,
                                       threshold=0.33, candidates=101, start=21):
    """The precalculation of a vanish stage whose windows are unit-width harmonic windows,
    ``per_lambda`` widths per unit of lambda: the pool returns the energies of such windows at
    the candidates, the Context only gives the stage and the start configurations."""
    class Pool:
        def run(self, tasks):
            out = []
            for task in tasks:
                lam = 1.0 - task["state"]["lambda_sterics"]
                grid = np.linspace(0.0, 1.0, task["candidates"])
                x = np.random.default_rng(int(round(lam * 100)) + 17).normal(per_lambda * lam, 1.0, task["n_samples"])
                out.append({"energies": {"vanish": 0.5 * (x[:, None] - per_lambda * grid[None, :]) ** 2},
                            "config": task["config"]})
            return out

    settings = AlsSettings(overlap_threshold=threshold, overlap_matrix=overlap_matrix, candidates=candidates,
                           initial_windows={"vanish": start}, window_ps=4.0, sample_interval_ps=0.002, walk_ps=0.004,
                           max_windows=max_windows)
    return adaptive_schedule(context, integrator, ["vanish"], 300.0, 2.0, settings, lambda m: None, pool=Pool())[0]


def test_many_initial_windows_and_a_raised_threshold_converge_by_the_overlap_of_a_pair_alone(water_ligand_system):
    """The owner's choice of 2026-11-21: a start with many windows and a threshold above 0.18.

    21 windows 1.5 widths apart overlap by 0.30 as pairs and by 0.24 in the matrix of the stage.
    With a threshold of 0.33 the rule on the pair adds windows until neighbours are 1.2 widths
    apart or less (0.36) and ends; on the matrix of the stage every added window lowers the
    elements, so it adds until ``max_windows`` and says so.
    """
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    integ = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integ.setRandomNumberSeed(31)
    ctx = mm.Context(alch.system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300.0, 5)

    by_pair = precalculation_of_harmonic_windows(ctx, integ, "pair", max_windows=100)
    last = by_pair.provenance["iterations"][-1]
    values = by_pair.windows()["vanish"]
    assert by_pair.provenance["overlap_matrix"] == "pair" and last["status"] == "converged" and by_pair.notes == []
    assert min(last["pair_overlap"]) > 0.33 and len(last["pair_overlap"]) == len(last["superdiagonal"])
    assert max(np.diff(values)) < 0.045 and 26 <= len(values) <= 32            # 1.2 widths or less, and no more
    assert max(last["superdiagonal"][1:-1]) < 0.26                             # what the matrix of the stage reads

    by_stage = precalculation_of_harmonic_windows(ctx, integ, "stage", max_windows=28)
    last = by_stage.provenance["iterations"][-1]
    assert by_stage.provenance["overlap_matrix"] == "stage" and "pair_overlap" not in last
    assert last["status"] == "max_windows" and len(by_stage.states) == 28
    assert any("stopped in the vanish stage at the limit of 28 windows" in note for note in by_stage.notes)
    assert max(last["superdiagonal"][1:-1]) < 0.26                             # no window could have raised it
    with pytest.raises(ValueError, match="overlap_matrix is 'stage' or 'pair'"):
        precalculation_of_harmonic_windows(ctx, integ, "neighbours", max_windows=28)

    # the start and the windows that a schedule records are a start again, also where a candidate
    # step is no decimal (100 candidates: 1/99)
    thirds = precalculation_of_harmonic_windows(ctx, integ, "pair", max_windows=100, per_lambda=3.0,
                                                candidates=100, start=4)
    assert thirds.provenance["initial_lambda"]["vanish"] == [0.0, 0.333333, 0.666667, 1.0]
    assert initial_window_indices(thirds.provenance["initial_lambda"]["vanish"], 100) == [0, 33, 66, 99]
    again = initial_window_indices(thirds.windows()["vanish"], 100)
    assert again[0] == 0 and again[-1] == 99 and len(again) == len(thirds.states)


def test_a_leg_that_is_only_told_its_overlap_matrix_gets_the_threshold_of_that_matrix(water_ligand_system):
    """Review of the defaults of 2026-11-22. The complex leg reads the overlap of a pair alone
    with a threshold of 0.30; ``overlap_matrix: stage`` alone left it that threshold, which the
    elements of the matrix of a stage between close windows do not reach: from the default start
    of the complex leg, harmonic windows 30 widths per unit of lambda went to ``max_windows``
    (100; lowest element 0.083). A threshold that is not given is the one that belongs to the matrix its leg reads:
    0.18 for the stage, 0.30 for the pair."""
    from cufep.config.schema import AdaptiveConfig

    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    integ = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integ.setRandomNumberSeed(31)
    ctx = mm.Context(alch.system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300.0, 5)

    def precalculation(adaptive, max_windows, threshold=None):
        start = adaptive.initial_windows.for_leg("complex")["vanish"]
        return precalculation_of_harmonic_windows(
            ctx, integ, adaptive.matrix("complex"), max_windows=max_windows, start=start,
            threshold=adaptive.threshold("complex") if threshold is None else threshold)

    default, stage = AdaptiveConfig(), AdaptiveConfig(overlap_matrix="stage")
    assert (default.threshold("complex"), default.matrix("complex")) == (0.30, "pair")
    assert (stage.threshold("complex"), stage.matrix("complex")) == (0.18, "stage")
    assert (stage.threshold("solvent"), stage.matrix("solvent")) == (0.18, "stage")
    assert AdaptiveConfig(overlap_matrix="pair").threshold("solvent") == 0.30
    mixed = AdaptiveConfig(overlap_matrix={"complex": "stage", "solvent": "pair"})
    assert (mixed.threshold("complex"), mixed.threshold("solvent")) == (0.18, 0.30)
    # a threshold that is given is kept, whatever the matrix
    given = AdaptiveConfig(overlap_matrix="stage", overlap_threshold=0.1)
    assert (given.threshold("complex"), given.threshold("solvent")) == (0.1, 0.1)

    for adaptive, windows in ((default, 25), (stage, 22)):
        schedule = precalculation(adaptive, 100)
        last = schedule.provenance["iterations"][-1]
        assert last["status"] == "converged" and schedule.notes == [], adaptive.matrix("complex")
        assert abs(len(schedule.states) - windows) <= 3
    # what the review found: the threshold of the pair on the matrix of the stage does not end
    unreachable = precalculation(stage, 40, threshold=0.30)
    last = unreachable.provenance["iterations"][-1]
    assert last["status"] == "max_windows" and len(unreachable.states) == 40
    assert max(last["superdiagonal"][1:-1]) < 0.30


def test_precalculation_on_a_small_system(water_ligand_system):
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    integ = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integ.setRandomNumberSeed(11)
    ctx = mm.Context(alch.system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300.0, 2)
    messages = []
    settings = AlsSettings(overlap_threshold=0.04, candidates=21, initial_windows={"discharge": 2, "vanish": 3},
                           window_ps=0.4, sample_interval_ps=0.02, max_windows=9)
    schedule, configs = adaptive_schedule(ctx, integ, ["discharge", "vanish"], 300.0, 2.0, settings, messages.append)
    assert schedule.mode == "adaptive" and schedule.stages == ["discharge", "vanish"]
    assert len(configs) == len(schedule.states) <= 9
    assert schedule.states[0] == AlchemicalState(1.0, 1.0, 1.0) and schedule.states[-1] == AlchemicalState(
        1.0, 0.0, 0.0
    )
    w = schedule.windows()
    assert w["discharge"][0] == 0.0 and w["discharge"][-1] == 1.0 and w["vanish"][0] == 0.0 and w["vanish"][-1] == 1.0
    assert all(abs(v * 20 - round(v * 20)) < 1e-9 for vals in w.values() for v in vals)       # on the candidate grid
    prov = schedule.provenance
    assert prov["overlap_threshold"] == 0.04 and prov["candidates"] == 21
    last = {}
    for rec in prov["iterations"]:
        assert len(rec["superdiagonal"]) == len(rec["lambda"]) - 1
        if rec["status"] == "inserted":
            assert len(rec["new_lambda"]) >= 1 and rec["section"][0] < rec["section"][1]
        last[rec["stage"]] = rec
    for stage, rec in last.items():
        assert rec["status"] in ("converged", "saturated", "max_windows")
        assert rec["lambda"] == w[stage]
        if rec["status"] == "converged":
            assert min(rec["superdiagonal"]) > 0.04
    for p, b, v in configs:
        assert np.all(np.isfinite(p)) and np.all(np.isfinite(v)) and b.shape == (3, 3)
    assert any("lambda schedule" in m for m in messages)


def test_precalculation_from_given_lambda_values(water_ligand_system):
    """The start of a stage given as lambda values: those windows are run first. The schedule
    never has fewer windows than its start; where the rule adds a window it spaces the windows
    of that bad section evenly again, so only values of the start outside such a section stay."""
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    integ = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integ.setRandomNumberSeed(13)
    ctx = mm.Context(alch.system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300.0, 4)
    start = [0.0, 0.5, 0.8, 0.9, 1.0]
    settings = AlsSettings(overlap_threshold=0.04, candidates=21, initial_windows={"discharge": 2, "vanish": start},
                           window_ps=0.2, sample_interval_ps=0.02, max_windows=9)
    schedule, configs = adaptive_schedule(ctx, integ, ["discharge", "vanish"], 300.0, 2.0, settings, lambda m: None)
    prov = schedule.provenance
    assert prov["initial_windows"] == {"discharge": 2, "vanish": 5}
    assert prov["initial_lambda"] == {"discharge": [0.0, 1.0], "vanish": start}
    first = next(rec for rec in prov["iterations"] if rec["stage"] == "vanish")
    assert first["iteration"] == 1 and first["lambda"] == start
    final = schedule.windows()["vanish"]
    refined = [rec["section"] for rec in prov["iterations"] if rec["stage"] == "vanish" and rec["status"] == "inserted"]
    assert len(final) >= len(start)
    assert {v for v in start if not any(a < v < b for a, b in refined)} <= set(final)
    assert len(configs) == len(schedule.states) <= 9


def test_precalculation_with_the_windows_of_an_iteration_run_by_a_pool(water_ligand_system):
    """With a pool, all new windows of an iteration are handed over together."""
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)

    def make(seed):
        integ = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
        integ.setRandomNumberSeed(seed)
        ctx = mm.Context(alch.system, integ, mm.Platform.getPlatformByName("CPU"))
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(pos)
        ctx.setVelocitiesToTemperature(300.0, seed)
        return ctx, integ

    ctx, integ = make(21)
    helper = make(22)

    class Pool:
        calls: list = []

        def run(self, tasks):
            self.calls.append([t["state"] for t in tasks])
            assert all(t["equilibration_steps"] == 10 and t["n_samples"] == 10 for t in tasks)
            return [run_window(*helper, t) for t in tasks]

    pool = Pool()
    settings = AlsSettings(overlap_threshold=0.04, candidates=21, initial_windows={"discharge": 2, "vanish": 3},
                           window_ps=0.2, equilibration_ps=0.02, sample_interval_ps=0.02, walk_ps=0.02, max_windows=6)
    schedule, configs = adaptive_schedule(ctx, integ, ["discharge", "vanish"], 300.0, 2.0, settings, lambda m: None,
                                          pool=pool)
    assert len(pool.calls[0]) == 4                                      # 2 + 3 windows, one shared
    assert pool.calls[0][0] == AlchemicalState(1.0, 1.0, 1.0).to_dict()
    assert pool.calls[0][-1] == AlchemicalState(1.0, 0.0, 0.0).to_dict()
    assert sum(len(c) for c in pool.calls) == schedule.provenance["windows_simulated"]
    assert schedule.provenance["windows_run_concurrently"] is True and schedule.provenance["equilibration_ps"] == 0.02
    assert len(configs) == len(schedule.states) <= 6
    assert schedule.states[0] == AlchemicalState(1.0, 1.0, 1.0) and schedule.states[-1] == AlchemicalState(
        1.0, 0.0, 0.0
    )


def test_a_schedule_is_accepted_only_for_its_own_leg(tmp_path):
    """Review of PR 1: a complex leg took the frozen schedule of a solvent leg and lost its restrain stage."""
    from dataclasses import replace

    from cufep.workflow import WorkflowError, _schedule_of_this_leg

    complex_stages, solvent_stages = ["restrain", "discharge", "vanish"], ["discharge", "vanish"]
    of_complex = fixed_schedule(complex_stages, {"restrain": 2, "discharge": 2, "vanish": 3}, rest2_scale=1.5)
    of_solvent = fixed_schedule(solvent_stages, {"discharge": 2, "vanish": 3})
    of_complex.check_leg("complex", complex_stages)
    of_solvent.check_leg("solvent", solvent_stages)
    with pytest.raises(ValueError, match="its stages are discharge, vanish; the complex leg has restrain"):
        of_solvent.check_leg("complex", complex_stages)
    with pytest.raises(ValueError, match="its stages are restrain"):
        of_complex.check_leg("solvent", solvent_stages)

    # the right stage names do not help if the states are not the ends of the leg
    restrained_from_the_start = replace(
        of_complex, states=[replace(st, lambda_restraints=1.0) for st in of_complex.states]
    )
    with pytest.raises(ValueError, match="its first state .restraint 1,.* is not the coupled end of the complex leg"):
        restrained_from_the_start.check_leg("complex", complex_stages)
    cut_short = replace(of_complex, states=of_complex.states[:-1])
    with pytest.raises(ValueError, match="its last state .* is not the decoupled end"):
        cut_short.check_leg("complex", complex_stages)

    # files: the leg a schedule was saved for is kept, and the workflow names the file it refuses
    solvent_file, complex_file = tmp_path / "solvent.yaml", tmp_path / "complex.yaml"
    of_solvent.save(solvent_file, leg="solvent", ligand="lig", repeat=0)
    of_complex.save(complex_file, leg="complex", ligand="lig", repeat=0)
    assert Schedule.load(solvent_file).leg == "solvent"
    assert len(_schedule_of_this_leg(complex_file, "complex", complex_stages).states) == len(of_complex.states)
    with pytest.raises(
        WorkflowError, match="solvent.yaml cannot be used for the complex leg: it was written for the solvent leg"
    ):
        _schedule_of_this_leg(solvent_file, "complex", complex_stages)
    of_solvent.save(solvent_file)                                       # a file that does not name its leg
    with pytest.raises(WorkflowError, match="cannot be used for the complex leg: its stages are discharge, vanish"):
        _schedule_of_this_leg(solvent_file, "complex", complex_stages)

