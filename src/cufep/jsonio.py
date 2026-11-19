"""JSON that every parser reads.

Python writes NaN and Infinity by default, which are not JSON; a result that holds one cannot be
read by a standard parser. Results are written through :func:`dumps`: a number that has no value
becomes ``null`` (the reason is given in a text field next to it), and NumPy types become plain
ones.
"""
from __future__ import annotations

import json
import math

import numpy as np


def clean(value):
    """``value`` with every non-finite number replaced by None and NumPy types by Python types."""
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def dumps(value, **kwargs) -> str:
    """``json.dumps`` of the cleaned value; raises rather than write NaN or Infinity."""
    return json.dumps(clean(value), allow_nan=False, **kwargs)
