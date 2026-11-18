import numpy as np
import pytest

from cufep.alchemy.schedule import (STERICS_GRID, Schedule, _next_step, fixed_schedule, refine, respace,
                                    split_wide_intervals)
from cufep.sampling.states import AlchemicalState, stage_state

BETA = 1.0 / (8.31446261815324e-3 * 298.15)


def test_restraint_stage_is_linear():
    e = np.array([10.0, 14.0, 12.0, 8.0, 16.0])                 # kJ/mol
    dt, std = _next_step("restraints", 0.0, e, BETA, 1.0, 1e-3)
    assert std == pytest.approx(1.0)
    assert dt == pytest.approx(1.0 / (BETA * e.std()))
    dt, std = _next_step("restraints", 0.0, e * 1e-3, BETA, 1.0, 1e-3)      # tiny energies: one step
    assert dt == 1.0 and std < 1.0


def test_electrostatics_stage_solves_the_quadratic():
    rng = np.random.default_rng(0)
    a1, a2 = rng.normal(40, 12, 200), rng.normal(5, 2, 200)
    data = np.stack([np.zeros(200), 0.5 * a1 + 0.25 * a2, a1 + a2], axis=1)   # G0 at lambda 0, 0.5, 1
    t = 0.2
    dt, std = _next_step("electrostatics", t, data, BETA, 1.0, 1e-3)
    lam, new = 1 - t, 1 - t - dt
    check = BETA * np.std(a1 * (new - lam) + a2 * (new ** 2 - lam ** 2))
    assert std == pytest.approx(1.0, abs=1e-6) and check == pytest.approx(1.0, abs=1e-6)
    dt_end, _ = _next_step("electrostatics", 0.999, data, BETA, 1.0, 1e-3)
    assert dt_end == pytest.approx(0.001)


def test_sterics_stage_interpolates_on_the_grid():
    lam = 0.6
    offsets = np.minimum(STERICS_GRID, lam)
    slope = np.array([20.0, 30.0, 40.0, 50.0])[:, None]         # per-frame energy slope, kJ/mol per lambda
    data = np.concatenate([np.zeros((4, 1)), -slope * offsets[None, :]], axis=1)
    dt, std = _next_step("sterics", 1 - lam, data, BETA, 1.0, 1e-3)
    assert std == pytest.approx(1.0)
    assert dt == pytest.approx(1.0 / (BETA * slope.std()), rel=1e-6)
    flat = np.zeros((4, 1 + len(STERICS_GRID)))
    assert _next_step("sterics", 1 - lam, flat, BETA, 1.0, 1e-3)[0] == pytest.approx(lam)


def _toy_schedule():
    stages = ["restraints", "electrostatics", "sterics"]
    prog = [(0, 0.0), (0, 0.5), (1, 0.0), (1, 0.3), (1, 0.6), (2, 0.0), (2, 0.2), (2, 0.5), (2, 0.8), (2, 1.0)]
    states = [stage_state(stages[k], t) for k, t in prog]
    std = [1.0] * (len(prog) - 1) + [0.0]
    return Schedule(stages, states, [k for k, _ in prog], [t for _, t in prog], std)


def test_respace_keeps_stage_ends_and_hits_the_requested_count():
    s = _toy_schedule()
    out, picks = respace(s, 7)
    assert len(out.states) == 7 == len(picks)
    assert out.states[0] == s.states[0] and out.states[-1] == AlchemicalState(1.0, 0.0, 0.0)
    assert AlchemicalState(1.0, 1.0, 1.0) in out.states and AlchemicalState(1.0, 0.0, 1.0) in out.states
    assert all(0 <= p < len(s.states) for p in picks)


def test_refine_inserts_only_where_overlap_is_low():
    s = _toy_schedule()
    overlap = [0.3] * 9
    assert refine(s, overlap, 0.03, 48) is None
    overlap[6] = 0.01
    out, picks = refine(s, overlap, 0.03, 48)
    assert len(out.states) == len(s.states) + 1
    new = out.states[7]
    assert s.states[6].lambda_sterics > new.lambda_sterics > s.states[7].lambda_sterics
    assert picks[7] == 6
    assert refine(s, overlap, 0.03, len(s.states)) is None      # no room left


def test_split_wide_intervals():
    s = _toy_schedule()
    s.step_std_kT[3] = 3.1
    out, picks = split_wide_intervals(s, [3], 1.0)
    assert len(out.states) == len(s.states) + 2
    assert sum(out.step_std_kT) == pytest.approx(sum(s.step_std_kT))


def test_fixed_schedule_and_round_trip(tmp_path):
    s = fixed_schedule(["electrostatics", "sterics"], {"electrostatics": 3, "sterics": 5})
    assert len(s.states) == 9
    assert s.states[0] == AlchemicalState(1.0, 1.0, 1.0) and s.states[-1] == AlchemicalState(1.0, 0.0, 0.0)
    s.save(tmp_path / "schedule.yaml", leg="solvent")
    again = Schedule.load(tmp_path / "schedule.yaml")
    assert again.states == s.states and again.stages == s.stages
