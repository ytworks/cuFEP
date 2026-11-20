"""Compute backends: NVIDIA cuPotential (default) and stock OpenMM."""
from cufep.backends.base import Backend, BackendError, get_backend

__all__ = ["Backend", "BackendError", "get_backend"]
