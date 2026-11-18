"""Stock OpenMM backend (reference, CPU CI and fallback)."""
from __future__ import annotations

import openmm as mm

from cufep.backends.base import Backend, clone_system


class StockBackend(Backend):
    name = "openmm"

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, "stock OpenMM is always available"

    def prepare_system(self, system: mm.System) -> mm.System:
        return clone_system(system)

    def make_integrator(self, temperature, friction, timestep, seed: int = 0) -> mm.Integrator:
        integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
        integrator.setConstraintTolerance(1e-6)
        integrator.setRandomNumberSeed(int(seed))
        return integrator

    def make_isotropic_barostat(self, pressure, temperature, frequency: int = 25, seed: int = 0) -> mm.Force:
        b = mm.MonteCarloBarostat(pressure, temperature, frequency)
        b.setRandomNumberSeed(int(seed))
        return b
