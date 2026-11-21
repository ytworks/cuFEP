"""Backend interface.

A backend turns a *stock* OpenMM ``System`` (built only from classes that ship with
OpenMM) into the objects that actually run: a possibly converted System, an integrator,
a barostat and platform properties. The alchemical System is always built once from
stock classes, so both backends evaluate the same Hamiltonian and can be compared
state by state.
"""
from __future__ import annotations

import abc
from typing import Any

import openmm as mm
import openmm.unit as unit


class BackendError(RuntimeError):
    """Raised when a backend cannot run the requested configuration."""


def clone_system(system: mm.System) -> mm.System:
    """Deep copy of a System through XML serialization."""
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def find_forces(system: mm.System, cls: type) -> list[tuple[int, Any]]:
    """Return ``(index, force)`` for every force that is exactly of type ``cls``."""
    return [(i, f) for i, f in enumerate(system.getForces()) if type(f) is cls]


def remove_forces(system: mm.System, cls: type) -> int:
    """Remove every force of type ``cls``; return how many were removed."""
    idx = [i for i, f in enumerate(system.getForces()) if type(f) is cls]
    for i in reversed(idx):
        system.removeForce(i)
    return len(idx)


class Backend(abc.ABC):
    """Factory for the runtime objects of one compute backend."""

    name: str = "abstract"

    def __init__(self, precision: str = "mixed", platform: str = "CUDA", cpu_threads: int | None = None) -> None:
        self.precision = precision
        self.platform_name = platform
        #: threads of a Context of the CPU platform (resources.cpu_threads_per_worker)
        self.cpu_threads = cpu_threads

    # -- availability -------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def available(cls) -> tuple[bool, str]:
        """Return ``(ok, reason)``."""

    # -- system / integrator / barostat --------------------------------------
    @abc.abstractmethod
    def prepare_system(self, system: mm.System) -> mm.System:
        """Return a copy of ``system`` converted for this backend (no barostat added)."""

    @abc.abstractmethod
    def make_integrator(self, temperature, friction, timestep, seed: int = 0) -> mm.Integrator:
        """Langevin-middle integrator."""

    @abc.abstractmethod
    def make_isotropic_barostat(self, pressure, temperature, frequency: int = 25, seed: int = 0) -> mm.Force:
        """Isotropic Monte Carlo barostat."""

    def make_membrane_barostat(self, pressure, temperature, frequency: int = 25, seed: int = 0,
                               surface_tension=0.0 * unit.bar * unit.nanometer) -> mm.Force:
        """Semi-isotropic barostat for bilayers (always the stock OpenMM class)."""
        b = mm.MonteCarloMembraneBarostat(
            pressure, surface_tension, temperature,
            mm.MonteCarloMembraneBarostat.XYIsotropic, mm.MonteCarloMembraneBarostat.ZFree, frequency)
        b.setRandomNumberSeed(int(seed))
        return b

    # -- context --------------------------------------------------------------
    def platform(self) -> mm.Platform:
        return mm.Platform.getPlatformByName(self.platform_name)

    def platform_properties(self, device_index: int | None = None) -> dict[str, str]:
        """The properties of a Context: precision and device on CUDA and OpenCL (review of PR 1: OpenCL
        got none and ran in its default single precision while the result recorded the one asked
        for), threads on the CPU platform (the inline workers of CPU runs got the threads of the
        environment instead of resources.cpu_threads_per_worker)."""
        props: dict[str, str] = {}
        if self.platform_name in ("CUDA", "OpenCL"):
            props["Precision"] = self.precision
            if device_index is not None:
                props["DeviceIndex"] = str(device_index)
        elif self.platform_name == "CPU" and self.cpu_threads:
            props["Threads"] = str(int(self.cpu_threads))
        return props

    def make_context(self, system: mm.System, integrator: mm.Integrator,
                     device_index: int | None = None) -> mm.Context:
        """Create a Context for an already prepared System; the properties asked for are read back
        from it, so that what a result records is what the Context does."""
        props = self.platform_properties(device_index)
        context = mm.Context(system, integrator, self.platform(), props)
        platform = context.getPlatform()
        for name, wanted in props.items():
            if name in platform.getPropertyNames():
                got = platform.getPropertyValue(context, name)
                if str(got) != str(wanted):
                    raise BackendError(f"the {platform.getName()} Context has {name} = {got}, not {wanted} as asked")
        return context


def backend_precision(name: str, precision: str) -> str:
    """The precision in which the backend ``name`` runs when the configuration asks for
    ``precision``: the cuPotential plugin has no double precision and runs in mixed precision then.
    One rule for the alchemical run and for the plain MD of a leg."""
    if name.lower() in ("cupotential", "cupo", "nv") and precision == "double":
        return "mixed"
    return precision


def get_backend(name: str, **kwargs) -> Backend:
    """Instantiate a backend by name (``cupotential`` or ``openmm``)."""
    key = name.lower()
    if key in ("cupotential", "cupo", "nv"):
        from cufep.backends.cupotential import CuPotentialBackend
        return CuPotentialBackend(**kwargs)
    if key in ("openmm", "stock"):
        from cufep.backends.openmm_stock import StockBackend
        return StockBackend(**kwargs)
    raise BackendError(f"unknown backend '{name}' (expected 'cupotential' or 'openmm')")
