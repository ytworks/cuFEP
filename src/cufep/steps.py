"""Numbers of MD steps, by one rule for every path, and the condition under which every
configuration that shares an OpenMM Context gets the moves of its barostat.

Every length in the configuration is a time (ps or ns); the MD runs whole steps of the time
step. :func:`md_steps` is the one rounding of a time to steps, used by the run of a leg, the
precalculation of adaptive lambda scheduling, the walk through given windows and the checks of
the configuration, so that no two of them can round differently.

OpenMM's Monte Carlo barostats (isotropic and membrane, also the one of the cuPotential plugin)
count the steps of the Context they belong to and try a move of the box every ``frequency``
steps: the counter is a member of the barostat's implementation in the Context (OpenMM 8.4,
``MonteCarloBarostatImpl::step``) and cannot be set from outside. cuFEP advances several
configurations in turn in one Context: the replicas a worker holds, one segment of
``steps_per_iteration`` each per iteration, and the windows of the precalculation. The counter
runs on from one configuration to the next. A configuration that is advanced by fewer steps than
``frequency`` can therefore get no move at all, and with five replicas of five steps and a
frequency of 25 the same replica got every move while the other four kept their volume (review
of PR 1). With a segment of at least ``frequency`` steps every configuration gets
``steps // frequency`` moves per segment, or one more, wherever the counter stands.
"""
from __future__ import annotations


def md_steps(ps: float, timestep_fs: float, least: int = 1) -> int:
    """Whole MD steps of ``timestep_fs`` in ``ps`` picoseconds, at least ``least``."""
    return max(int(least), int(round(float(ps) * 1000.0 / float(timestep_fs))))


def frames_of(ps: float, frame_interval_ps: float, timestep_fs: float) -> int:
    """The frames an MD of ``ps`` picoseconds keeps when it takes one after every
    ``frame_interval_ps`` and one at its end (the pre-equilibration,
    :mod:`cufep.sampling.equilibrate`): its :func:`md_steps` in pieces of one interval."""
    steps, per_frame = md_steps(ps, timestep_fs), md_steps(frame_interval_ps, timestep_fs)
    return -(-steps // per_frame)


def integrated_before_revision_15_ps(ps: float, frame_interval_ps: float, timestep_fs: float) -> float:
    """The MD that versions before revision 15 integrated in the pre-equilibration when ``ps``
    picoseconds were asked (and recorded): whole frame intervals only, round(ps / interval) of them
    and at least one (review of PR 1: 40 ps of the 50 ps asked)."""
    return max(1, int(round(float(ps) / float(frame_interval_ps)))) * md_steps(frame_interval_ps, timestep_fs) \
        * float(timestep_fs) / 1000.0


def precalculation_steps(window_ps: float, equilibration_ps: float, sample_interval_ps: float,
                         timestep_fs: float) -> tuple[int, int, int]:
    """Steps of one window of the precalculation of adaptive lambda scheduling:
    ``(equilibration_steps, n_samples, steps_per_sample)``. A window runs its equilibration and
    then ``n_samples`` times ``steps_per_sample`` steps, one configuration without interruption."""
    steps_per_sample = md_steps(sample_interval_ps, timestep_fs)
    n_samples = max(2, int(round(float(window_ps) / float(sample_interval_ps))))
    equilibration_steps = md_steps(equilibration_ps, timestep_fs, least=0)
    return equilibration_steps, n_samples, steps_per_sample


def barostat_moves(steps: int, frequency: int) -> int:
    """The fewest barostat moves a configuration gets in a segment of ``steps`` MD steps in a
    Context that other configurations share, wherever the counter of the barostat stands."""
    return int(steps) // int(frequency)


def barostat_segment_problem(steps: int, frequency: int, what: str) -> str | None:
    """Why a segment of ``steps`` MD steps of one configuration in a shared Context would not get
    the moves of the barostat, or None. ``what`` names the segment in the words of the
    configuration. A segment of no steps (an evaluation of energies only) moves nothing."""
    steps, frequency = int(steps), int(frequency)
    if frequency < 1:
        return f"dynamics.barostat_frequency must be at least 1 step, not {frequency}"
    if steps <= 0 or barostat_moves(steps, frequency) >= 1:
        return None
    return (f"{what} is {steps} MD step(s), fewer than dynamics.barostat_frequency = {frequency}: the "
            "configurations that share one OpenMM Context are advanced one after the other and the barostat "
            "counts the steps of the Context, so some of them would never get a move of the box and would be "
            "sampled at constant volume. Lengthen the segment or lower dynamics.barostat_frequency")
