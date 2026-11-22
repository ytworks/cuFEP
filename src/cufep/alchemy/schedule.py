"""Lambda schedules: fixed windows from the configuration, and adaptive lambda scheduling.

A leg is a sequence of stages (complex: restrain, discharge, vanish; solvent: discharge,
vanish). Within a stage lambda runs from 0 at its start to 1 at its end; the window at the
end of a stage is the window at the start of the next one.

Fixed mode
    The configuration gives, per stage, either the lambda values or a number of windows.
    A number is turned into values by re-sampling the recommended values of that stage, so
    the relative spacing is kept.

Adaptive mode: adaptive lambda scheduling (ALS)
    Midgley, Bariami, Habgood and Mackey, J. Chem. Inf. Model. 2025, 65, 512. A short
    precalculation decides the windows; its data are discarded afterwards. The rule of the
    paper, applied here to every stage separately:

    1. Lambda can take ``candidates`` equally spaced values (101). Start with a few evenly
       spaced windows. Run a short MD (50 ps) in every window that has not been run yet and
       record the energy of every sample at all candidate values.
    2. Compute the overlap matrix of the current windows. A pair of neighbouring windows is
       *deficient* if its overlap (the superdiagonal element) is at or below the threshold
       (0.04). If no pair is deficient, the schedule is final.
    3. A *bad section* is a maximal run of consecutive deficient pairs. Take the longest
       one; among equally long ones the one that contains the lowest overlap.
    4. Keep the two end windows of that section, add one window between them and space all
       windows of the section evenly. Only lambda values that have not been simulated yet
       are run; data of the others are kept. Go to 2.

    So one window is added per iteration, only where neighbouring windows overlap too
    little, and windows are never removed.

    The paper does not say what to do in the cases below; cuFEP's rules are:
    evenly spaced values are rounded to the nearest candidate, halves to the even candidate
    (0.125 becomes 0.12, the value that appears in the paper); a section that cannot take
    another distinct candidate is skipped; the precalculation stops with a warning when no
    section can be refined or when the leg has ``max_windows`` windows; the overlap matrix
    uses the data of the windows currently in the schedule.

    cuFEP's own: the start of a stage can also be given as lambda values, which need not be
    evenly spaced (:mod:`cufep.alchemy.candidates`). They are candidates. As with every start
    the schedule never has fewer windows than it; step 4 spaces the windows of a bad section
    evenly, so given values inside a bad section move and those outside every bad section stay.

    cuFEP's own, by a setting (``overlap_matrix: pair``): in step 2 the overlap of a pair is
    read from the overlap matrix of its two windows alone (:func:`pair_overlap`), not from the
    matrix of all windows of the stage. The element of the matrix of the stage is shared among
    all windows that overlap, so it falls when windows get closer than they need to be, and a
    start with many windows, or a threshold above what that element can reach, makes the rule
    add windows where none are needed until ``max_windows``. The overlap of a pair alone rises
    to 0.5 as its windows meet. The rule itself (steps 3 and 4) is the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from cufep.alchemy.candidates import initial_indices, initial_window_indices
from cufep.alchemy.factory import LAMBDA_ELECTROSTATICS, LAMBDA_RESTRAINTS, LAMBDA_STERICS
from cufep.sampling.states import (
    R_GAS,
    STAGE_ALIASES,
    AlchemicalState,
    StateEnergyEvaluator,
    canonical_stage,
    set_state,
    stage_state,
    validate_states,
)

__all__ = ["Schedule", "AlsSettings", "stage_values", "schedule_from_windows", "fixed_schedule", "initial_indices",
           "initial_window_indices", "bad_sections", "choose_section", "redistribute", "als_step", "stage_overlap",
           "pair_overlap", "adaptive_schedule", "run_window", "STAGE_ALIASES"]


# ============================================================================ the schedule
@dataclass
class Schedule:
    stages: list[str]
    states: list[AlchemicalState]
    stage_of_state: list[int]         # stage in which the state is listed (a boundary belongs to the earlier one)
    stage_lambda: list[float]         # lambda of the state within that stage
    mode: str = "fixed"
    #: how the schedule was obtained: for adaptive schedules the threshold and one record per iteration
    provenance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: REST2: temperature scale of the hot region in the middle of the leg (1 = no REST2); the
    #: scale of each window is in its state
    rest2_scale: float = 1.0
    #: the leg the schedule was written for, if the file says so
    leg: str | None = None

    def check_leg(self, leg: str, stages: list[str]) -> None:
        """Raise ValueError unless this is a schedule of the given leg.

        A schedule of the solvent leg has no restrain stage and its states carry the restraint at
        full strength; used for a complex leg it would leave out the free energy of switching the
        restraint on, while the correction for releasing it is still applied. So the stages must
        be those of the leg, and the first and the last state must be its physical ends: the
        ligand fully interacting (in a complex leg without the restraint) and the ligand
        decoupled (in a complex leg with the restraint).
        """
        from cufep.sampling.states import stage_state

        stages = [canonical_stage(s) for s in stages]
        if self.leg is not None and self.leg != leg:
            raise ValueError(f"it was written for the {self.leg} leg")
        if list(self.stages) != stages:
            raise ValueError(f"its stages are {', '.join(self.stages)}; the {leg} leg has {', '.join(stages)}")
        for which, state, wanted in (("first", self.states[0], stage_state(stages[0], 0.0)),
                                     ("last", self.states[-1], stage_state(stages[-1], 1.0))):
            have = (state.lambda_restraints, state.lambda_electrostatics, state.lambda_sterics, state.rest2_scale)
            want = (wanted.lambda_restraints, wanted.lambda_electrostatics, wanted.lambda_sterics, 1.0)
            if any(abs(a - b) > 1e-9 for a, b in zip(have, want)):
                raise ValueError(
                    f"its {which} state (restraint {have[0]:g}, electrostatics {have[1]:g}, sterics {have[2]:g}, "
                    f"REST2 scale {have[3]:g}) is not the {'coupled' if which == 'first' else 'decoupled'} end of "
                    f"the {leg} leg (restraint {want[0]:g}, electrostatics {want[1]:g}, sterics {want[2]:g})")

    def windows(self) -> dict[str, list[float]]:
        """Lambda values per stage, both ends included."""
        out: dict[str, list[float]] = {}
        for k, name in enumerate(self.stages):
            vals = [lam for i, lam in enumerate(self.stage_lambda) if self.stage_of_state[i] == k]
            if k > 0:
                vals = [0.0] + vals
            out[name] = [round(float(v), 6) for v in vals]
        return out

    def to_dict(self) -> dict:
        return {"mode": self.mode, "stages": self.stages, "windows": self.windows(),
                "states": [s.to_dict() for s in self.states], "stage_of_state": self.stage_of_state,
                "stage_lambda": [float(v) for v in self.stage_lambda], "rest2_scale": float(self.rest2_scale),
                "provenance": self.provenance, "notes": self.notes}

    def save(self, path: str | Path, **meta) -> None:
        data = self.to_dict()
        data.update(meta)
        from cufep.files import publish_text

        publish_text(path, yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "Schedule":
        d = yaml.safe_load(Path(path).read_text())
        states = [AlchemicalState(**s) for s in d["states"]]
        validate_states(states)
        n = len(states)
        return cls(stages=[canonical_stage(s) for s in d["stages"]], states=states,
                   stage_of_state=list(d.get("stage_of_state", [0] * n)),
                   stage_lambda=list(d.get("stage_lambda", d.get("progress", [0.0] * n))),
                   mode=str(d.get("mode", "fixed")), provenance=dict(d.get("provenance", {})),
                   notes=list(d.get("notes", [])), rest2_scale=float(d.get("rest2_scale", 1.0)),
                   leg=d.get("leg"))


def stage_values(spec, recommended) -> list[float]:
    """Lambda values of one stage from a number of windows or a list of values.

    A number re-samples ``recommended`` (itself a number, meaning even spacing, or a list) so
    that the relative spacing of the recommended values is kept.
    """
    if not isinstance(spec, (int, np.integer)):
        return [float(v) for v in spec]
    n = int(spec)
    if n < 2:
        raise ValueError("a stage needs at least two lambda windows")
    base = np.linspace(0.0, 1.0, int(recommended)) if isinstance(recommended, (int, np.integer)) \
        else np.asarray(recommended, dtype=float)
    if n == len(base):
        return [float(v) for v in base]
    values = np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(base)), base)
    values = np.round(values, 4)
    values[0], values[-1] = 0.0, 1.0
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{n} windows cannot be placed on this stage")
    return [float(v) for v in values]


def _assemble(stages: list[str], values: dict[str, list[float]], mode: str, rest2_scale: float = 1.0,
              **extra) -> Schedule:
    states, stage_of, lam = [], [], []
    for k, name in enumerate(stages):
        for j, v in enumerate(values[name]):
            if k > 0 and j == 0:
                continue                                   # shared with the end of the previous stage
            states.append(stage_state(name, v, stages, rest2_scale))
            stage_of.append(k)
            lam.append(float(v))
    validate_states(states)
    return Schedule(stages=list(stages), states=states, stage_of_state=stage_of, stage_lambda=lam, mode=mode,
                    rest2_scale=float(rest2_scale), **extra)


def schedule_from_windows(stages: list[str], windows: dict, recommended: dict, rest2_scale: float = 1.0) -> Schedule:
    """Fixed schedule from the ``lambda_schedule.windows`` section of the configuration."""
    stages = [canonical_stage(s) for s in stages]
    values = {name: stage_values(windows[name], recommended[name]) for name in stages}
    return _assemble(stages, values, "fixed", rest2_scale)


def fixed_schedule(stages: list[str], n_per_stage: dict[str, int], rest2_scale: float = 1.0) -> Schedule:
    """Evenly spaced schedule with ``n`` intervals per stage (tests and diagnostics)."""
    stages = [canonical_stage(s) for s in stages]
    counts = {canonical_stage(k): int(v) for k, v in n_per_stage.items()}
    values = {name: [float(v) for v in np.linspace(0.0, 1.0, max(1, counts.get(name, 4)) + 1)] for name in stages}
    return _assemble(stages, values, "fixed", rest2_scale)


# ============================================================================ ALS: the rule
# The windows a stage starts with (initial_indices, initial_window_indices) are in
# cufep.alchemy.candidates, where the check of the configuration finds them as well.
def bad_sections(superdiagonal, threshold: float) -> list[tuple[int, int]]:
    """Maximal runs of deficient neighbouring pairs, as (first window, last window) positions.

    Pair ``k`` joins windows ``k`` and ``k + 1`` and is deficient if its overlap is at or
    below ``threshold``. The length of a section is its number of pairs, ``last - first``.
    """
    out, start = [], None
    for k, value in enumerate(superdiagonal):
        deficient = not (float(value) > threshold)        # NaN counts as deficient
        if deficient and start is None:
            start = k
        if not deficient and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, len(superdiagonal)))
    return out


def choose_section(sections: list[tuple[int, int]], superdiagonal) -> tuple[int, int]:
    """The longest section; among equally long ones the one containing the lowest overlap."""
    def key(section):
        a, b = section
        lowest = min(float(v) if np.isfinite(v) else -np.inf for v in superdiagonal[a:b])
        return (-(b - a), lowest, a)
    return min(sections, key=key)


def redistribute(windows: list[int], section: tuple[int, int]) -> list[int] | None:
    """Add one window to ``section`` and space its windows evenly; ``None`` if it cannot be done.

    ``windows`` are candidate indices in increasing order. The two end windows of the section
    stay where they are.
    """
    a, b = section
    lo, hi = windows[a], windows[b]
    n_intervals = (b - a) + 1                              # one more than before
    if hi - lo < n_intervals:
        return None                                       # not enough distinct candidates in between
    inner = [int(round(lo + (hi - lo) * j / n_intervals)) for j in range(1, n_intervals)]
    new = windows[:a + 1] + inner + windows[b:]
    if any(y <= x for x, y in zip(new, new[1:])):
        return None
    return new


def als_step(windows: list[int], superdiagonal, threshold: float, room: int = 1) -> tuple[list[int] | None, dict]:
    """One decision of adaptive lambda scheduling for one stage.

    Returns the new window list, or ``None`` if nothing is changed, and a record with
    ``status``: ``converged`` (no deficient pair), ``inserted``, ``saturated`` (deficient
    pairs remain but no section can take another candidate) or ``max_windows``.
    """
    sections = bad_sections(superdiagonal, threshold)
    record = {"windows": list(windows), "superdiagonal": [float(v) for v in superdiagonal],
              "bad_sections": [[windows[a], windows[b]] for a, b in sections]}
    if not sections:
        return None, {**record, "status": "converged"}
    if room < 1:
        return None, {**record, "status": "max_windows"}
    remaining = list(sections)
    while remaining:
        section = choose_section(remaining, superdiagonal)
        new = redistribute(windows, section)
        if new is not None:
            return new, {**record, "status": "inserted", "section": [windows[section[0]], windows[section[1]]],
                         "new_windows": sorted(set(new) - set(windows))}
        remaining.remove(section)
    return None, {**record, "status": "saturated"}


# ============================================================================ ALS: the precalculation
@dataclass
class AlsSettings:
    """The settings of the precalculation of one leg. A run takes them from the configuration,
    leg by leg (``cufep.workflow.als_settings``, whose defaults are those of
    ``lambda_schedule.adaptive``); the values here are those of a call that gives none."""

    overlap_threshold: float = 0.18             # the owner's choice of 2026-11-20 (the paper: 0.04)
    #: the overlap matrix whose element is compared with the threshold: "stage", the matrix of all
    #: windows of the stage (the paper), or "pair", that of the two neighbouring windows alone
    overlap_matrix: str = "stage"
    candidates: int = 101
    #: per stage the windows of the first iteration: a number of evenly spaced windows, or the
    #: lambda values (candidates, not necessarily evenly spaced)
    initial_windows: dict = field(default_factory=lambda: {"restrain": 3, "discharge": 5, "vanish": 9})
    window_ps: float = 50.0
    #: MD in a new window before its energies are recorded
    equilibration_ps: float = 0.0
    sample_interval_ps: float = 0.5
    #: when the windows are run at the same time by a pool of workers: MD per window of the walk
    #: through the initial windows that gives each of them a start configuration
    walk_ps: float = 5.0
    max_windows: int = 100
    #: REST2: temperature scale of the hot region in the middle of the leg (1 = no REST2). The
    #: windows of the precalculation carry the scale of their place along the leg, and so do
    #: the candidates whose energies are recorded.
    rest2_scale: float = 1.0


def candidate_energies(context, stage: str, grid: np.ndarray, beta: float, leg_stages=None,
                       rest2_scale: float = 1.0) -> np.ndarray:
    """Reduced energy of the current configuration at every candidate lambda of ``stage``.

    The Context is left at the last state that was evaluated; the caller sets its own state
    again. With REST2 the stages of the leg are needed (``leg_stages``).
    """
    states = [stage_state(stage, lam, leg_stages, rest2_scale) for lam in grid]
    return StateEnergyEvaluator(context, states, 1.0 / (R_GAS * beta), None).reduced_potentials()[0]


def stage_overlap(energies: list[np.ndarray], windows: list[int]) -> np.ndarray:
    """Superdiagonal of the overlap matrix of ``windows``.

    ``energies[i]`` holds the samples of window ``i`` at all candidates, shape (frames, candidates).
    """
    from cufep.analysis.mbar import overlap_matrix

    u_kn = np.concatenate([e[:, windows] for e in energies], axis=0).T        # (windows, all samples)
    n_k = np.array([len(e) for e in energies])
    try:
        o = overlap_matrix(u_kn, n_k)
    except Exception:  # noqa: BLE001 - no solution means no usable overlap
        return np.zeros(len(windows) - 1)
    return np.array([o[i, i + 1] for i in range(len(windows) - 1)])


def pair_overlap(energies: list[np.ndarray], windows: list[int]) -> np.ndarray:
    """Overlap of every pair of neighbouring ``windows``, from the samples of the two alone.

    The element of the overlap matrix of just the two windows of a pair, by the same estimator
    as :func:`stage_overlap` (for a stage of two windows the two are the same number). It lies
    between 0 and 0.5 and rises as the two windows get closer, whatever other windows the stage
    has. An element of the matrix of the whole stage is shared among all windows that overlap:
    it falls again when windows get close, and cannot say then whether a pair is close enough.
    """
    return np.array([stage_overlap(energies[i:i + 2], windows[i:i + 2])[0] for i in range(len(windows) - 1)])


def _snapshot(context):
    import openmm.unit as unit
    st = context.getState(getPositions=True, getVelocities=True)
    return (st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
            st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))


def _load(context, config) -> None:
    import openmm as mm
    pos, box, vel = config
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in np.asarray(box)])
    context.setPositions(pos)
    context.setVelocities(vel)


def run_window(context, integrator, task: dict) -> dict:
    """One window of the precalculation in an existing Context (this process or a pool worker).

    ``task``: ``state`` (lambda parameters), ``config`` (positions, box, velocities) to start
    from, ``stages`` whose candidate energies are recorded, ``candidates``, ``temperature_K``,
    ``equilibration_steps``, ``n_samples`` and ``steps_per_sample``.
    Returns the energies per stage, shape (samples, candidates), and the last configuration.
    """
    grid = np.linspace(0.0, 1.0, int(task["candidates"]))
    state = AlchemicalState(**task["state"])
    leg_stages, peak = task.get("leg_stages"), float(task.get("rest2_scale", 1.0))
    # one evaluator per stage over its candidates; terms common to all candidates cancel in the overlap
    evaluators = {name: StateEnergyEvaluator(context, [stage_state(name, lam, leg_stages, peak) for lam in grid],
                                             task["temperature_K"], None) for name in task["stages"]}
    _load(context, task["config"])
    set_state(context, state)
    if task["equilibration_steps"] > 0:
        integrator.step(int(task["equilibration_steps"]))
    energies = {name: np.empty((int(task["n_samples"]), len(grid))) for name in task["stages"]}
    for i in range(int(task["n_samples"])):
        integrator.step(int(task["steps_per_sample"]))
        for name in task["stages"]:
            energies[name][i] = evaluators[name].reduced_potentials(restore=state)[0]
    return {"energies": energies, "config": _snapshot(context)}


def adaptive_schedule(context, integrator, stages: list[str], temperature_K: float, timestep_fs: float,
                      settings: AlsSettings | None = None, log=print, pool=None):
    """Adaptive lambda scheduling for one leg, starting from the Context's current configuration.

    Without ``pool`` the windows are run one after another in ``context``, each starting from
    the end of the nearest window that has been run. With ``pool`` (an object whose
    ``run(tasks)`` executes :func:`run_window` tasks concurrently, e.g. on several GPUs) the
    new windows of an iteration run at the same time; for the first iteration a short walk
    through the windows in ``context`` provides their start configurations.

    Returns ``(schedule, configurations)`` with one ``(positions_nm, box_nm, velocities)``
    tuple per window: the last configuration of that window's precalculation run.
    """

    s = settings or AlsSettings()
    if s.overlap_matrix not in ("stage", "pair"):
        raise ValueError(f"overlap_matrix is 'stage' or 'pair', not {s.overlap_matrix!r}")
    by_pair = s.overlap_matrix == "pair"
    params = context.getParameters()
    present = {"restrain": LAMBDA_RESTRAINTS, "discharge": LAMBDA_ELECTROSTATICS, "vanish": LAMBDA_STERICS}
    stages = [st for st in (canonical_stage(x) for x in stages) if present[st] in params]
    if not stages:
        raise ValueError("the Context has no alchemical parameters")
    top = s.candidates - 1
    from cufep.steps import md_steps, precalculation_steps

    equilibration_steps, n_samples, steps_per_sample = precalculation_steps(
        s.window_ps, s.equilibration_ps, s.sample_interval_ps, timestep_fs)

    def node(k: int, idx: int) -> tuple[int, int]:
        """The window at the end of a stage is the window at the start of the next one."""
        return (k + 1, 0) if (idx == top and k + 1 < len(stages)) else (k, idx)

    def path(nd: tuple[int, int]) -> float:
        return nd[0] + nd[1] / top

    def state_of(nd: tuple[int, int]) -> AlchemicalState:
        return stage_state(stages[nd[0]], nd[1] / top, stages, s.rest2_scale)

    windows = {k: initial_window_indices(s.initial_windows.get(name, 3), s.candidates)
               for k, name in enumerate(stages)}
    first = {k: list(v) for k, v in windows.items()}        # the start: the rule only adds to it
    runs: dict[tuple[int, int], dict] = {}
    start = _snapshot(context)

    def task_for(nd: tuple[int, int], config) -> dict:
        member = [stages[k] for k in range(len(stages)) if nd in (node(k, 0), node(k, top)) or nd[0] == k]
        return {"state": state_of(nd).to_dict(), "config": config, "stages": member, "candidates": s.candidates,
                "temperature_K": temperature_K, "equilibration_steps": equilibration_steps, "n_samples": n_samples,
                "steps_per_sample": steps_per_sample, "leg_stages": list(stages), "rest2_scale": s.rest2_scale}

    def nearest(nd: tuple[int, int]):
        return runs[min(runs, key=lambda other: abs(path(other) - path(nd)))]["config"] if runs else start

    def run_all(todo: list[tuple[int, int]]) -> None:
        if pool is None:
            for nd in todo:                               # one after another, each from its neighbour
                runs[nd] = run_window(context, integrator, task_for(nd, nearest(nd)))
            return
        if runs:
            starts = [nearest(nd) for nd in todo]
        else:                                             # first iteration: a short walk gives the starts
            starts, walk_steps = [], md_steps(s.walk_ps, timestep_fs)
            _load(context, start)
            for nd in todo:
                set_state(context, state_of(nd))
                integrator.step(walk_steps)
                starts.append(_snapshot(context))
        for nd, result in zip(todo, pool.run([task_for(nd, cfg) for nd, cfg in zip(todo, starts)])):
            runs[nd] = result

    def total_windows() -> int:
        return len({node(k, i) for k in windows for i in windows[k]})

    records, status = [], {k: "running" for k in windows}
    iteration = 0
    while True:
        iteration += 1
        run_all(sorted({node(k, i) for k in windows for i in windows[k]} - set(runs), key=path))
        changed = False
        for k, name in enumerate(stages):
            if status[k] != "running":
                continue
            energies = [runs[node(k, i)]["energies"][name] for i in windows[k]]
            # the element of the matrix of the windows of the stage is always recorded (from the
            # samples of the precalculation; the analysis of the production checks the matrix of all
            # windows of the leg from its own samples); the rule reads it, or each pair alone
            sup = stage_overlap(energies, windows[k])
            read = pair_overlap(energies, windows[k]) if by_pair else sup
            new, rec = als_step(windows[k], read, s.overlap_threshold, room=s.max_windows - total_windows())
            rec = {"iteration": iteration, "stage": name,
                   "lambda": [round(i / top, 6) for i in rec.pop("windows")],
                   "superdiagonal": [round(float(v), 4) for v in sup],
                   **({"pair_overlap": [round(float(v), 4) for v in read]} if by_pair else {}),
                   "bad_sections": [[round(a / top, 6), round(b / top, 6)] for a, b in rec.pop("bad_sections")],
                   "status": rec["status"],
                   **({"section": [round(v / top, 6) for v in rec["section"]],
                       "new_lambda": [round(v / top, 6) for v in rec["new_windows"]]} if new is not None else {})}
            records.append(rec)
            if new is None:
                status[k] = rec["status"]
                log(f"lambda schedule, {name}: {len(windows[k])} windows, {rec['status']} "
                    f"(lowest overlap {min(rec['pair_overlap' if by_pair else 'superdiagonal']):.3f}"
                    + (f" of a pair alone, {min(rec['superdiagonal']):.3f} in the matrix of the stage)"
                       if by_pair else ")"))
            else:
                log(f"lambda schedule, {name}: iteration {iteration}, overlap at or below {s.overlap_threshold:g} "
                    f"between {rec['section'][0]:g} and {rec['section'][1]:g}; running {rec['new_lambda']}")
                windows[k] = new
                changed = True
        if not changed:
            break
    notes = []
    for k, name in enumerate(stages):
        if status[k] == "saturated":
            notes.append(f"adaptive lambda scheduling could not reach the overlap threshold in the {name} stage: "
                         f"the windows there are already one candidate apart")
        elif status[k] == "max_windows":
            notes.append(f"adaptive lambda scheduling stopped in the {name} stage at the limit of "
                         f"{s.max_windows} windows with neighbouring windows still below the overlap threshold")
    values = {name: [i / top for i in windows[k]] for k, name in enumerate(stages)}
    schedule = _assemble(stages, values, "adaptive", s.rest2_scale, notes=notes,
                         provenance={"method": "adaptive lambda scheduling", "overlap_threshold": s.overlap_threshold,
                                     # the matrix the threshold was compared with (a schedule written
                                     # before this key existed: that of the stage)
                                     "overlap_matrix": s.overlap_matrix,
                                     "candidates": s.candidates, "window_ps": s.window_ps,
                                     "equilibration_ps": s.equilibration_ps,
                                     "sample_interval_ps": s.sample_interval_ps,
                                     # the MD each window ran: whole sample intervals, as many as are
                                     # nearest to window_ps (precalculation_steps), and its equilibration
                                     "window_ps_integrated": n_samples * steps_per_sample * float(timestep_fs) / 1000.0,
                                     "equilibration_ps_integrated": equilibration_steps * float(timestep_fs) / 1000.0,
                                     "windows_run_concurrently": pool is not None,
                                     # the start, as a number per stage and as its lambda values
                                     "initial_windows": {n: len(first[k]) for k, n in enumerate(stages)},
                                     "initial_lambda": {n: [round(i / top, 6) for i in first[k]]
                                                        for k, n in enumerate(stages)},
                                     "windows_simulated": len(runs), "iterations": records})
    configs = []
    for k, _name in enumerate(stages):
        for j, i in enumerate(windows[k]):
            if k > 0 and j == 0:
                continue
            configs.append(runs[node(k, i)]["config"])
    log(f"lambda schedule: {len(schedule.states)} windows ("
        + ", ".join(f"{name} {len(windows[k])}" for k, name in enumerate(stages))
        + f"), {len(runs)} window runs of {s.window_ps:g} ps in {iteration} iteration(s)")
    return schedule, configs
