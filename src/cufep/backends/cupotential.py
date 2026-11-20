"""NVIDIA cuPotential backend.

All calls into ``nv_openmm_plugin`` live in this file. The plugin is early access and
its API may change between releases; keep the surface used here small.

Layout ("split"): the stock ``NonbondedForce`` (PME, with particle parameter offsets
that carry the charge scaling) is replaced by ``nv.NonbondedForce``. Every other force,
including the soft-core ``CustomNonbondedForce`` with its interaction group and all
restraint forces, stays on the stock CUDA kernels in the same System.
"""
from __future__ import annotations

import openmm as mm
import openmm.unit as unit

from cufep.backends.base import Backend, BackendError, clone_system, find_forces, remove_forces

#: Ewald tolerance the plugin requires on the source force (plugin 0.3.5).
REQUIRED_EWALD_TOLERANCE = 5e-4
#: Each box edge must be at least this many cutoffs (plugin 0.3.5).
MIN_BOX_CUTOFF_RATIO = 3.0


def _load():
    try:
        import nv_openmm_plugin as nv
    except Exception as exc:  # noqa: BLE001 - any import failure means "not available"
        raise BackendError(f"cuPotential plugin (nv_openmm_plugin) is not usable: {exc}") from exc
    return nv


class CuPotentialBackend(Backend):
    name = "cupotential"

    def __init__(self, precision: str = "mixed", platform: str = "CUDA",
                 integrator: str = "nv", box_safety: float | None = None, cpu_threads: int | None = None) -> None:
        if platform != "CUDA":
            raise BackendError("the cuPotential backend only runs on the CUDA platform")
        if precision not in ("mixed", "single"):
            raise BackendError("the cuPotential backend supports only 'mixed' or 'single' precision")
        if integrator not in ("nv", "stock"):
            raise BackendError("integrator must be 'nv' or 'stock'")
        super().__init__(precision=precision, platform=platform, cpu_threads=cpu_threads)
        self.integrator_kind = integrator
        self.box_safety = box_safety

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            _load()
            mm.Platform.getPlatformByName("CUDA")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return True, "nv_openmm_plugin loaded and CUDA platform present"

    # -- checks ----------------------------------------------------------------
    @staticmethod
    def check_system(system: mm.System) -> None:
        """Raise ``BackendError`` if the plugin cannot take this System."""
        nb = find_forces(system, mm.NonbondedForce)
        if len(nb) != 1:
            raise BackendError(f"expected exactly one NonbondedForce, found {len(nb)}")
        force = nb[0][1]
        if force.getNonbondedMethod() not in (mm.NonbondedForce.PME, mm.NonbondedForce.LJPME):
            raise BackendError("cuPotential needs PME or LJPME on the NonbondedForce")
        tol = force.getEwaldErrorTolerance()
        if abs(tol - REQUIRED_EWALD_TOLERANCE) > 1e-12:
            raise BackendError(f"cuPotential needs ewaldErrorTolerance={REQUIRED_EWALD_TOLERANCE}, got {tol}")
        cutoff = force.getCutoffDistance().value_in_unit(unit.nanometer)
        a, b, c = system.getDefaultPeriodicBoxVectors()
        edges = [a[0].value_in_unit(unit.nanometer), b[1].value_in_unit(unit.nanometer),
                 c[2].value_in_unit(unit.nanometer)]
        if min(edges) < MIN_BOX_CUTOFF_RATIO * cutoff:
            raise BackendError(
                f"box edge {min(edges):.3f} nm is below {MIN_BOX_CUTOFF_RATIO:g} x cutoff ({cutoff:.3f} nm)")

    # -- system / integrator / barostat ---------------------------------------------
    def prepare_system(self, system: mm.System) -> mm.System:
        nv = _load()
        self.check_system(system)
        out = clone_system(system)
        (index, source), = find_forces(out, mm.NonbondedForce)
        if self.box_safety is None:
            converted = nv.NonbondedForce(source)
        else:
            converted = nv.NonbondedForce(source, float(self.box_safety))
        converted.setForceGroup(source.getForceGroup())
        out.removeForce(index)
        out.addForce(converted)
        if self.integrator_kind == "nv":
            # The plugin integrator removes centre-of-mass motion itself.
            remove_forces(out, mm.CMMotionRemover)
        return out

    def make_integrator(self, temperature, friction, timestep, seed: int = 0) -> mm.Integrator:
        if self.integrator_kind == "stock":
            integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
            integrator.setRandomNumberSeed(int(seed))
        else:
            nv = _load()
            integrator = nv.LangevinMiddleIntegrator(temperature, friction, timestep, seed=int(seed))
        integrator.setConstraintTolerance(1e-6)
        return integrator

    def make_isotropic_barostat(self, pressure, temperature, frequency: int = 25, seed: int = 0) -> mm.Force:
        nv = _load()
        return nv.MonteCarloBarostat(pressure, temperature, int(frequency), int(seed))
