"""The candidate lambda values of adaptive lambda scheduling, and the windows a stage starts with.

A stage has ``n_candidates`` equally spaced candidate values from 0 to 1 (101 by default: 0.00,
0.01, ..., 1.00), and every window of the precalculation is one of them. The rules here say
which candidates a stage starts with. The precalculation and the check of the configuration use
the same functions, so that a start the precalculation would refuse is refused before anything
is written.

The paper (Midgley et al., J. Chem. Inf. Model. 2025, 65, 512) starts from a few evenly spaced
windows: :func:`initial_indices`. Lambda values given one by one, which need not be evenly
spaced, are cuFEP's own: :func:`candidate_indices`.
"""
from __future__ import annotations

from numbers import Integral

__all__ = ["initial_indices", "candidate_indices", "initial_window_indices"]

#: lambda values are written with six decimals; a given value that close to a candidate is that candidate
WRITTEN_PRECISION = 1e-6


def initial_indices(n_windows: int, n_candidates: int = 101) -> list[int]:
    """Candidate indices of ``n_windows`` evenly spaced windows (halves go to the even candidate)."""
    if n_windows < 2:
        raise ValueError("at least two windows are needed")
    top = n_candidates - 1
    idx = [int(round(i * top / (n_windows - 1))) for i in range(n_windows)]    # round() rounds halves to even
    idx[0], idx[-1] = 0, top
    if len(set(idx)) != len(idx):
        raise ValueError(f"{n_windows} windows do not fit on {n_candidates} candidates")
    return idx


def candidate_indices(values, n_candidates: int = 101) -> list[int]:
    """Candidate indices of the lambda ``values`` of a stage.

    Every value must be a candidate, and the values must increase from 0.0 (the start of the
    stage) to 1.0 (its end); a ValueError says which of the two a list breaks. A value is a
    candidate if it is one to the six decimals with which cuFEP writes lambda values
    (``schedule.yaml``, the results), so that what one run wrote can be the start of another
    whatever the number of candidates (a third of 99 steps is written 0.333333).
    """
    top = int(n_candidates) - 1
    idx = []
    for value in values:
        index = int(round(float(value) * top))
        if abs(float(value) - index / top) > WRITTEN_PRECISION:
            raise ValueError(f"{float(value):g} is not one of the {int(n_candidates)} candidate lambda values "
                             f"(multiples of {1.0 / top:g})")
        idx.append(index)
    if len(idx) < 2 or idx[0] != 0 or idx[-1] != top or any(b <= a for a, b in zip(idx, idx[1:])):
        raise ValueError("the lambda values of a stage must increase from 0.0 (its start) to 1.0 (its end)")
    return idx


def initial_window_indices(spec, n_candidates: int = 101) -> list[int]:
    """Candidate indices of the windows a stage starts with.

    ``spec`` is a number of evenly spaced windows or the lambda values themselves, as
    ``lambda_schedule.adaptive.initial_windows`` gives them.
    """
    if isinstance(spec, bool):
        raise ValueError("give a number of lambda windows or a list of lambda values")
    if isinstance(spec, Integral):
        return initial_indices(int(spec), n_candidates)
    return candidate_indices(spec, n_candidates)
