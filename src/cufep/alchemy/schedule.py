"""Adaptive lambda schedules (derivative-free "trailblazing").

The cuPotential plugin provides no dU/dlambda, so windows are placed from energy
differences only. Starting from the coupled state, a short simulation is run in the
current state and the next state is placed where the standard deviation of the reduced
energy difference to it, over the configurations just sampled, reaches a target
(1 kT by default). This spaces the states at roughly equal thermodynamic length, which
equalises the overlap between neighbours. The last configuration of every window seeds
the corresponding replica.

Thanks to the force-group decomposition the search itself is almost free:
  * restraints stage:      dU is linear in lambda            -> closed form
  * electrostatics stage:  dU is quadratic in lambda          -> three energies per frame
  * sterics stage:         soft-core energy on a fixed grid   -> interpolation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from cufep.alchemy.factory import (LAMBDA_ELECTROSTATICS, LAMBDA_RESTRAINTS, LAMBDA_STERICS, MAIN_GROUP,
                                   RESTRAINT_GROUP, SOFTCORE_GROUP)
from cufep.sampling.states import AlchemicalState, R_GAS, set_state, stage_state, validate_states

#: offsets in lambda_sterics probed below the current value
STERICS_GRID = np.array([0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09,
                         0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0])


@dataclass
class ScheduleSettings:
    target_std_kT: float = 1.0
    max_windows: int = 48
    pilot_ps: float = 40.0            # MD per window during the search
    n_frames: int = 20
    equilibration_fraction: float = 0.5
    min_step: float = 1.0e-3          # smallest step in stage progress
    seed: int = 0


@dataclass
class Schedule:
    stages: list[str]
    states: list[AlchemicalState]
    stage_of_state: list[int]         # stage index of the step that *starts* at each state
    progress: list[float]             # progress t within that stage
    step_std_kT: list[float]          # measured std of dU to the next state (last entry 0)
    target_std_kT: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"stages": self.stages, "target_std_kT": self.target_std_kT,
                "states": [s.to_dict() for s in self.states], "stage_of_state": self.stage_of_state,
                "progress": [float(p) for p in self.progress],
                "step_std_kT": [float(x) for x in self.step_std_kT], "notes": self.notes}

    def save(self, path: str | Path, **meta) -> None:
        data = self.to_dict()
        data.update(meta)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "Schedule":
        d = yaml.safe_load(Path(path).read_text())
        states = [AlchemicalState(**s) for s in d["states"]]
        validate_states(states)
        n = len(states)
        return cls(stages=list(d["stages"]), states=states,
                   stage_of_state=list(d.get("stage_of_state", [0] * n)),
                   progress=list(d.get("progress", [0.0] * n)),
                   step_std_kT=list(d.get("step_std_kT", [0.0] * n)),
                   target_std_kT=float(d.get("target_std_kT", 1.0)), notes=list(d.get("notes", [])))


def _group_energy(context, group: int) -> float:
    import openmm.unit as unit
    return context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _state_energy(context, state: AlchemicalState) -> float:
    """Total alchemical energy (kJ/mol) of the current configuration in ``state``."""
    set_state(context, state)
    params = context.getParameters()
    energy = _group_energy(context, MAIN_GROUP)
    if LAMBDA_STERICS in params:
        energy += _group_energy(context, SOFTCORE_GROUP)
    if LAMBDA_RESTRAINTS in params:
        energy += _group_energy(context, RESTRAINT_GROUP)
    return energy


def split_wide_intervals(schedule: "Schedule", wide: list[int], target: float) -> tuple["Schedule", list[int]]:
    """Subdivide intervals whose width is well above the target into equal parts in progress."""
    states, stage, prog, std, picks = [], [], [], [], []
    n = len(schedule.states)
    for i in range(n):
        states.append(schedule.states[i]); stage.append(schedule.stage_of_state[i])
        prog.append(schedule.progress[i]); std.append(schedule.step_std_kT[i]); picks.append(i)
        if i in wide and i + 1 < n:
            k = schedule.stage_of_state[i]
            t0 = schedule.progress[i]
            t1 = schedule.progress[i + 1] if (schedule.stage_of_state[i + 1] == k and schedule.progress[i + 1] > t0) else 1.0
            parts = int(min(4, max(2, round(schedule.step_std_kT[i] / target))))
            std[-1] = schedule.step_std_kT[i] / parts
            for j in range(1, parts):
                tm = t0 + (t1 - t0) * j / parts
                states.append(stage_state(schedule.stages[k], tm)); stage.append(k); prog.append(tm)
                std.append(schedule.step_std_kT[i] / parts)
                picks.append(i if j <= parts // 2 else i + 1)
    out = Schedule(stages=schedule.stages, states=states, stage_of_state=stage, progress=prog, step_std_kT=std,
                   target_std_kT=schedule.target_std_kT,
                   notes=schedule.notes + [f"split {len(wide)} wide interval(s)"])
    validate_states(out.states)
    return out, picks


def _next_step(stage: str, t: float, data: np.ndarray, beta: float, target: float, min_step: float) -> tuple[float, float]:
    """Step in stage progress whose dU standard deviation equals ``target`` (kT).

    Returns ``(dt, std at that step)``; ``dt`` may reach the end of the stage.
    """
    remaining = 1.0 - t
    if stage == "restraints":                                   # data: restraint energy per frame
        slope = beta * float(np.std(data))
        dt = remaining if slope * remaining <= target else max(min_step, target / slope)
        dt = min(dt, remaining)
        return dt, slope * dt
    if stage == "electrostatics":                               # data: (frames, 3) G0 at lambda 0, 0.5, 1
        g0, g05, g1 = data[:, 0], data[:, 1], data[:, 2]
        a2 = 2.0 * (g1 - 2.0 * g05 + g0)
        a1 = g1 - g0 - a2
        lam = 1.0 - t

        def std(dt: float) -> float:
            new = lam - dt
            return beta * float(np.std(a1 * (new - lam) + a2 * (new * new - lam * lam)))

        if std(remaining) <= target:
            return remaining, std(remaining)
        lo, hi = 0.0, remaining
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if std(mid) > target:
                hi = mid
            else:
                lo = mid
        dt = min(remaining, max(min_step, lo))
        return dt, std(dt)
    if stage == "sterics":                                      # data: (frames, 1 + n_grid) G1
        lam = 1.0 - t
        offsets = np.minimum(STERICS_GRID, lam)
        du = data[:, 1:] - data[:, :1]
        stds = beta * du.std(axis=0)
        offsets, idx = np.unique(offsets, return_index=True)
        stds = stds[idx]
        if stds[-1] <= target and offsets[-1] >= lam - 1e-12:
            return remaining, float(stds[-1])
        above = np.nonzero(stds > target)[0]
        if len(above) == 0:
            return float(offsets[-1]), float(stds[-1])
        j = int(above[0])
        if j == 0:
            dt = max(min_step, float(offsets[0]) * target / max(float(stds[0]), 1e-12))
            return min(dt, remaining), target
        x0, x1, y0, y1 = offsets[j - 1], offsets[j], stds[j - 1], stds[j]
        dt = float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-12))
        return min(max(min_step, dt), remaining), target
    raise ValueError(stage)


def trailblaze(context, integrator, stages: list[str], temperature_K: float, timestep_fs: float,
               settings: ScheduleSettings | None = None, log=print):
    """Find a schedule along ``stages`` starting from the Context's current configuration.

    Returns ``(schedule, configurations)`` where ``configurations`` is a list with one
    ``(positions_nm, box_nm, velocities)`` tuple per state.
    """
    import openmm.unit as unit

    s = settings or ScheduleSettings()
    beta = 1.0 / (R_GAS * temperature_K)
    params = context.getParameters()
    stages = [st for st in stages if {"restraints": LAMBDA_RESTRAINTS, "electrostatics": LAMBDA_ELECTROSTATICS,
                                      "sterics": LAMBDA_STERICS}[st] in params]
    if not stages:
        raise ValueError("the Context has no alchemical parameters")
    steps_total = max(s.n_frames, int(round(s.pilot_ps * 1000.0 / timestep_fs)))
    n_equil = int(steps_total * s.equilibration_fraction)
    interval = max(1, (steps_total - n_equil) // s.n_frames)

    states: list[AlchemicalState] = []
    stage_of_state: list[int] = []
    progress: list[float] = []
    stds: list[float] = []
    reverse: list[float] = []         # std of dU back to the previous state, on this state's samples
    configs = []
    stage_idx, t = 0, 0.0
    previous: AlchemicalState | None = None
    while True:
        if t >= 1.0 and stage_idx + 1 < len(stages):
            stage_idx, t = stage_idx + 1, 0.0                   # same physical state, next stage
        stage = stages[stage_idx]
        state = stage_state(stage, t)
        set_state(context, state)
        integrator.step(n_equil)
        rows = []
        back = []
        final = t >= 1.0
        for _ in range(s.n_frames):
            integrator.step(interval)
            if previous is not None:
                e_here = _state_energy(context, state)
                e_prev = _state_energy(context, previous)
                back.append(e_prev - e_here)
                set_state(context, state)
            if final:
                continue
            if stage == "restraints":
                context.setParameter(LAMBDA_RESTRAINTS, 1.0)
                rows.append(_group_energy(context, RESTRAINT_GROUP))
            elif stage == "electrostatics":
                row = []
                for le in (0.0, 0.5, 1.0):
                    context.setParameter(LAMBDA_ELECTROSTATICS, le)
                    row.append(_group_energy(context, MAIN_GROUP))
                rows.append(row)
            else:
                lam = state.lambda_sterics
                row = [_group_energy(context, SOFTCORE_GROUP)]
                for off in STERICS_GRID:
                    context.setParameter(LAMBDA_STERICS, max(0.0, lam - float(off)))
                    row.append(_group_energy(context, SOFTCORE_GROUP))
                rows.append(row)
            set_state(context, state)
        st = context.getState(getPositions=True, getVelocities=True)
        configs.append((st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                        st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                        st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)))
        states.append(state)
        stage_of_state.append(stage_idx)
        progress.append(t)
        reverse.append(beta * float(np.std(back)) if back else 0.0)
        previous = state
        if final:
            stds.append(0.0)
            break
        dt, std = _next_step(stage, t, np.asarray(rows, dtype=float), beta, s.target_std_kT, s.min_step)
        stds.append(std)
        t = t + dt
        if 1.0 - t < 0.5 * s.min_step:
            t = 1.0
        if len(states) > 20 * s.max_windows:
            raise RuntimeError("lambda search does not terminate; the system may be unstable")
    # The forward estimate can be too optimistic where the energy distribution changes shape
    # (the soft-core end). Use the larger of the forward and reverse widths per interval.
    width = [max(stds[i], reverse[i + 1]) for i in range(len(states) - 1)] + [0.0]
    schedule = Schedule(stages=stages, states=states, stage_of_state=stage_of_state, progress=progress,
                        step_std_kT=width, target_std_kT=s.target_std_kT)
    validate_states(schedule.states)
    log(f"lambda search: {len(states)} states "
        f"({', '.join(f'{st}: {stage_of_state.count(i)}' for i, st in enumerate(stages))})")
    wide = [i for i, w in enumerate(width[:-1]) if w > 1.5 * s.target_std_kT]
    if wide:
        schedule, picks = split_wide_intervals(schedule, wide, s.target_std_kT)
        configs = [configs[i] for i in picks]
        log(f"lambda search: split {len(wide)} interval(s) whose reverse width exceeded 1.5 x target; "
            f"{len(schedule.states)} states")
    if len(states) > s.max_windows:
        schedule, picks = respace(schedule, s.max_windows)
        configs = [configs[i] for i in picks]
        log(f"lambda search: re-spaced to the limit of {s.max_windows} states")
    return schedule, configs


def respace(schedule: Schedule, n_states: int) -> tuple[Schedule, list[int]]:
    """Reduce a schedule to ``n_states`` at equal thermodynamic length, keeping stage ends.

    Returns the new schedule and, for each new state, the index of the old state whose
    configuration should seed it.
    """
    n_stages = len(schedule.stages)
    seg: list[list[int]] = [[] for _ in range(n_stages)]
    for i, k in enumerate(schedule.stage_of_state):
        seg[k].append(i)
    lengths = [sum(schedule.step_std_kT[i] for i in idx if i + 1 < len(schedule.states)) for idx in seg]
    total = sum(lengths) or 1.0
    intervals = max(n_stages, n_states - 1)
    alloc = [max(1, int(round(intervals * ln / total))) for ln in lengths]
    while sum(alloc) > intervals:
        alloc[int(np.argmax(alloc))] -= 1
    while sum(alloc) < intervals:
        alloc[int(np.argmax([ln / a for ln, a in zip(lengths, alloc)]))] += 1

    new_states, new_stage, new_prog, new_std, picks = [], [], [], [], []
    for k, idx in enumerate(seg):
        t_old = np.array([schedule.progress[i] for i in idx] + [1.0])
        step = np.array([schedule.step_std_kT[i] for i in idx])
        cum = np.concatenate([[0.0], np.cumsum(step)])
        length = cum[-1]
        for j in range(alloc[k]):
            target = length * j / alloc[k]
            t_new = float(np.interp(target, cum, t_old))
            new_states.append(stage_state(schedule.stages[k], t_new))
            new_stage.append(k)
            new_prog.append(t_new)
            new_std.append(length / alloc[k])
            picks.append(idx[int(np.argmin(np.abs(t_old[:-1] - t_new)))])
    last = len(schedule.states) - 1
    new_states.append(schedule.states[last])
    new_stage.append(n_stages - 1)
    new_prog.append(1.0)
    new_std.append(0.0)
    picks.append(last)
    out = Schedule(stages=schedule.stages, states=new_states, stage_of_state=new_stage, progress=new_prog,
                   step_std_kT=new_std, target_std_kT=schedule.target_std_kT,
                   notes=schedule.notes + [f"re-spaced from {len(schedule.states)} to {len(new_states)} states"])
    validate_states(out.states)
    return out, picks


def refine(schedule: Schedule, neighbour_overlap: list[float], min_overlap: float,
           max_windows: int) -> tuple[Schedule, list[int]] | None:
    """Insert a state in the middle of every interval whose MBAR overlap is too small.

    Returns ``None`` if nothing needs to change, else the new schedule and the old state
    index that seeds each new state.
    """
    weak = [i for i, o in enumerate(neighbour_overlap) if o < min_overlap]
    if not weak:
        return None
    room = max_windows - len(schedule.states)
    if room <= 0:
        return None
    weak = sorted(weak, key=lambda i: neighbour_overlap[i])[:room]
    states, stage, prog, std, picks = [], [], [], [], []
    for i, st in enumerate(schedule.states):
        states.append(st); stage.append(schedule.stage_of_state[i]); prog.append(schedule.progress[i])
        std.append(schedule.step_std_kT[i]); picks.append(i)
        if i in weak:
            k = schedule.stage_of_state[i]
            t_next = schedule.progress[i + 1] if schedule.stage_of_state[i + 1] == k and schedule.progress[i + 1] > schedule.progress[i] else 1.0
            t_mid = 0.5 * (schedule.progress[i] + t_next)
            std[-1] = 0.5 * schedule.step_std_kT[i]
            states.append(stage_state(schedule.stages[k], t_mid)); stage.append(k); prog.append(t_mid)
            std.append(0.5 * schedule.step_std_kT[i]); picks.append(i)
    out = Schedule(stages=schedule.stages, states=states, stage_of_state=stage, progress=prog, step_std_kT=std,
                   target_std_kT=schedule.target_std_kT,
                   notes=schedule.notes + [f"inserted {len(weak)} state(s) where overlap < {min_overlap}"])
    validate_states(out.states)
    return out, picks


def fixed_schedule(stages: list[str], n_per_stage: dict[str, int]) -> Schedule:
    """Evenly spaced schedule (tests and explicit ``mode: fixed`` without a file)."""
    states, stage_of, prog = [], [], []
    for k, stage in enumerate(stages):
        n = max(1, int(n_per_stage.get(stage, 4)))
        for j in range(n):
            states.append(stage_state(stage, j / n)); stage_of.append(k); prog.append(j / n)
    states.append(stage_state(stages[-1], 1.0)); stage_of.append(len(stages) - 1); prog.append(1.0)
    validate_states(states)
    return Schedule(stages=list(stages), states=states, stage_of_state=stage_of, progress=prog,
                    step_std_kT=[0.0] * len(states))
