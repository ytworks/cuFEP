"""The engine of a leg, shared by the absolute (``cufep run``) and the relative (``cufep rbfe``) calculations.

- :mod:`cufep.engine.protocol`: the protocols of a leg (the states it passes through and how their energies are
  evaluated) and the functions that serve any of them;
- :mod:`cufep.engine.leg`: the steps of a leg that do not depend on its protocol: the pre-equilibration, the backend,
  the lambda windows and their start configurations, the replica exchange, the fallback to stock OpenMM, the
  analysis of the samples;
- :mod:`cufep.engine.jobs`: a leg as the scheduler runs it.
"""
