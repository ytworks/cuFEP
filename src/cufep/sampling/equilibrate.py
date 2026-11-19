"""Minimisation and plain MD of the fully interacting system before the alchemical run.

Follows OpenFE's ABFE protocol: minimise, NVT, NPT, then an NPT production run whose
frames are used for the restraint search. Minimisation, NVT and NPT run on stock OpenMM
(the cuPotential plugin is reported to fail on unminimised coordinates). The production
run, which is the long part and cannot be spread over GPUs, runs on the backend of the leg
(``md_backend``); if that backend cannot run the system or fails, it is repeated on stock
OpenMM and the fact is recorded.

Time step: 4 fs with hydrogen mass repartitioning by default. If the run blows up it is
repeated once from the minimised structure at 2 fs and the fact is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import openmm as mm
import openmm.unit as u

from cufep.backends import get_backend


class EquilibrationError(RuntimeError):
    """The plain MD of the fully interacting system failed."""


@dataclass
class EquilibrationSettings:
    temperature_K: float = 298.15
    pressure_bar: float = 1.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 4.0               # an unstable run is repeated once at 2 fs
    minimize_iterations: int = 5000
    nvt_ns: float = 0.25
    npt_ns: float = 0.5
    production_ns: float = 5.0
    frame_interval_ps: float = 20.0
    barostat: str = "isotropic"            # isotropic | membrane
    platform: str = "CUDA"
    precision: str = "mixed"
    seed: int = 0
    #: backend of the production run (the minimisation, NVT and NPT always use stock OpenMM)
    md_backend: str = "openmm"
    #: heavy atoms held by harmonic position restraints during NVT/NPT (released for production)
    restrained_atoms: list[int] = field(default_factory=list)
    restraint_k: float = 1000.0            # kJ/mol/nm^2
    #: if set, the production run is extended in 5 ns blocks up to this length while the box area
    #: or the number of waters around the ligand still drifts (off by default)
    max_production_ns: float | None = None
    area_tolerance: float = 0.01           # relative change of the XY area between the last two quarters
    water_tolerance: float = 1.5           # change of the mean pocket water count between them
    ligand_atoms: list[int] = field(default_factory=list)
    water_oxygens: list[int] = field(default_factory=list)
    #: bilayers, restrained phases only: keep these water oxygens out of the hydrophobic core
    #: (|z - wall_centre_nm| < wall_half_width_nm) while a freshly built membrane closes up
    wall_atoms: list[int] = field(default_factory=list)
    wall_centre_nm: float = 0.0
    wall_half_width_nm: float = 0.0
    wall_k: float = 500.0                  # kJ/mol/nm^2


@dataclass
class EquilibrationResult:
    positions: np.ndarray                  # nm, final frame
    velocities: np.ndarray                 # nm/ps
    box: np.ndarray                        # (3, 3) nm
    frames: np.ndarray                     # (n_frames, n_frame_atoms, 3) nm, whole molecules
    frame_boxes: np.ndarray
    frame_atoms: list[int]
    timestep_fs: float
    notes: list[str]
    mean_volume_nm3: float = 0.0
    mean_box: np.ndarray | None = None
    simulated_ns: float = 0.0
    area_nm2: np.ndarray | None = None     # XY area per frame
    pocket_waters: np.ndarray | None = None  # waters within 0.6 nm of the ligand per frame
    #: membrane systems: had the bilayer area and the pocket hydration settled? (membrane_settled)
    settled: dict | None = None


def _context(system: mm.System, backend, settings: EquilibrationSettings, dt_fs: float, device, barostat: bool,
             restraint_reference: np.ndarray | None, seed: int):
    prepared = backend.prepare_system(system)
    temperature = settings.temperature_K * u.kelvin
    if barostat:
        pressure = settings.pressure_bar * u.bar
        if settings.barostat == "membrane":
            prepared.addForce(backend.make_membrane_barostat(pressure, temperature, 25, seed=seed + 1))
        else:
            prepared.addForce(backend.make_isotropic_barostat(pressure, temperature, 25, seed=seed + 1))
    if restraint_reference is not None and settings.restrained_atoms:
        force = mm.CustomExternalForce("0.5*k_posres*periodicdistance(x,y,z,x0,y0,z0)^2")
        force.addGlobalParameter("k_posres", settings.restraint_k)
        for name in ("x0", "y0", "z0"):
            force.addPerParticleParameter(name)
        for i in settings.restrained_atoms:
            force.addParticle(int(i), [float(v) for v in restraint_reference[i]])
        prepared.addForce(force)
    if restraint_reference is not None and settings.wall_atoms and settings.wall_half_width_nm > 0:
        wall = mm.CustomExternalForce(
            "k_wall*step(wall_d-dz)*(wall_d-dz)^2; dz=periodicdistance(x,y,z,x,y,wall_z)")
        wall.addGlobalParameter("k_wall", settings.wall_k)
        wall.addGlobalParameter("wall_d", settings.wall_half_width_nm)
        wall.addGlobalParameter("wall_z", settings.wall_centre_nm)
        for i in settings.wall_atoms:
            wall.addParticle(int(i), [])
        prepared.addForce(wall)
    integrator = backend.make_integrator(temperature, settings.friction_per_ps / u.picosecond,
                                         dt_fs * u.femtoseconds, seed=seed)
    return backend.make_context(prepared, integrator, device), integrator


def _state_arrays(context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = context.getState(getPositions=True, getVelocities=True)
    return (np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer)),
            np.asarray(st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)),
            np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)))


def _load(context, positions, box, velocities=None) -> None:
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in np.asarray(box)])
    context.setPositions(positions)
    if velocities is not None:
        context.setVelocities(velocities)


def _run_once(system, positions, box, frame_atoms, settings: EquilibrationSettings, dt_fs: float, device, log):
    backend = get_backend("openmm", precision=settings.precision, platform=settings.platform)
    temperature = settings.temperature_K * u.kelvin
    steps = lambda ns: int(round(ns * 1.0e6 / dt_fs))                       # noqa: E731

    # ---- minimise + NVT -----------------------------------------------------------------
    ctx, integ = _context(system, backend, settings, dt_fs, device, False, np.asarray(positions), settings.seed)
    _load(ctx, positions, box)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, settings.minimize_iterations)
    e_min = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    if not np.isfinite(e_min):
        raise EquilibrationError("energy is not finite after minimisation; check the input structure for clashes")
    ctx.setVelocitiesToTemperature(temperature, settings.seed + 11)
    integ.setStepSize(min(dt_fs, 1.0) * u.femtoseconds)                      # gentle start
    integ.step(5000)
    integ.setStepSize(dt_fs * u.femtoseconds)
    integ.step(steps(settings.nvt_ns))
    pos, vel, bx = _state_arrays(ctx)
    del ctx, integ

    # ---- NPT (restraints still on) -------------------------------------------------------
    ctx, integ = _context(system, backend, settings, dt_fs, device, True, np.asarray(positions), settings.seed + 100)
    _load(ctx, pos, bx, vel)
    integ.step(steps(settings.npt_ns))
    pos, vel, bx = _state_arrays(ctx)
    del ctx, integ

    # ---- production (no position restraints), frames for the restraint search -----------
    start = (pos, vel, bx)
    notes: list[str] = []
    md_backend = backend
    if settings.md_backend != "openmm":
        try:
            md_backend = get_backend(settings.md_backend, precision=settings.precision, platform=settings.platform)
            if hasattr(md_backend, "check_system"):
                probe = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
                probe.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in bx])
                md_backend.check_system(probe)
        except Exception as exc:  # noqa: BLE001 - any reason the backend cannot run this system
            log(f"pre-equilibration MD on stock OpenMM ({settings.md_backend} cannot run it: {str(exc)[:150]})")
            md_backend = backend
    try:
        return _production(system, md_backend, start, frame_atoms, settings, dt_fs, device, log, notes)
    except (mm.OpenMMException, EquilibrationError) as exc:
        if md_backend is backend:
            raise
        log(f"pre-equilibration MD failed on {settings.md_backend} ({str(exc)[:150]}); repeating it on stock OpenMM")
        notes.append(f"the pre-equilibration MD failed on {settings.md_backend} and was repeated on stock OpenMM")
        return _production(system, backend, start, frame_atoms, settings, dt_fs, device, log, notes)


def membrane_settled(areas, waters, area_tolerance: float, water_tolerance: float) -> dict:
    """Whether a bilayer has settled by the end of its equilibration.

    Compares the third and the fourth quarter of the run: the mean area of the box in the
    membrane plane (relative change at most ``area_tolerance``) and, if counted, the mean
    number of water molecules near the ligand (change at most ``water_tolerance``).
    ``settled`` is None if the run has fewer than five frames per quarter.
    """
    q = len(areas) // 4
    out = {"settled": None, "area_change": None, "pocket_water_change": None,
           "area_tolerance": float(area_tolerance), "pocket_water_tolerance": float(water_tolerance)}
    if q < 5:
        return out
    a3, a4 = float(np.mean(areas[2 * q:3 * q])), float(np.mean(areas[3 * q:]))
    out["area_change"] = (a4 - a3) / a3
    ok = abs(out["area_change"]) <= area_tolerance
    if len(waters):
        out["pocket_water_change"] = float(np.mean(waters[3 * q:]) - np.mean(waters[2 * q:3 * q]))
        ok = ok and abs(out["pocket_water_change"]) <= water_tolerance
    out["settled"] = bool(ok)
    return out


def _production(system, backend, start, frame_atoms, settings: EquilibrationSettings, dt_fs: float, device, log,
                notes: list[str]) -> EquilibrationResult:
    """The unrestrained NPT run after NVT and NPT; its frames are used for the restraint search."""
    pos, vel, bx = start
    ctx, integ = _context(system, backend, settings, dt_fs, device, True, None, settings.seed + 200)
    _load(ctx, pos, bx, vel)
    per_frame = max(1, int(round(settings.frame_interval_ps * 1000.0 / dt_fs)))
    lig = np.asarray(settings.ligand_atoms, dtype=int)
    wat = np.asarray(settings.water_oxygens, dtype=int)
    frames, boxes, volumes, areas, waters = [], [], [], [], []

    def advance(ns: float) -> None:
        for _ in range(max(1, int(round(ns * 1000.0 / settings.frame_interval_ps)))):
            integ.step(per_frame)
            st = ctx.getState(getPositions=True, getEnergy=True)
            p = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer))
            if not np.isfinite(st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)):
                raise EquilibrationError("energy became non-finite during the pre-equilibration MD")
            bx_f = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
            frames.append(p[frame_atoms].astype(np.float32))
            boxes.append(bx_f.astype(np.float32))
            volumes.append(st.getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3))
            areas.append(float(bx_f[0, 0] * bx_f[1, 1]))
            if len(lig) and len(wat):
                d = p[wat][:, None, :] - p[lig][None, :, :]
                d -= np.round(d / np.diag(bx_f)) * np.diag(bx_f)       # rectangular membrane box
                waters.append(float(np.sum(np.min(np.einsum("ijk,ijk->ij", d, d), axis=1) < 0.36)))

    def settled() -> bool:
        return membrane_settled(areas, waters, settings.area_tolerance, settings.water_tolerance)["settled"] is not False

    advance(settings.production_ns)
    total = settings.production_ns
    if settings.max_production_ns is not None:
        while not settled() and total + 5.0 <= settings.max_production_ns + 1e-9:
            log(f"bilayer area or pocket hydration still drifting after {total:g} ns; extending by 5 ns")
            advance(5.0)
            total += 5.0
        if settled() and total > settings.production_ns:
            notes.append(f"membrane equilibration extended to {total:g} ns")
    # a membrane system: say whether the bilayer has settled; the run is not made longer for it
    verdict = None
    if settings.barostat == "membrane":
        verdict = membrane_settled(areas, waters, settings.area_tolerance, settings.water_tolerance)
        if verdict["settled"] is False:
            water = "" if verdict["pocket_water_change"] is None else (
                f" and the number of water molecules near the ligand by {verdict['pocket_water_change']:+.1f} "
                f"(tolerance {settings.water_tolerance:g})")
            notes.append(
                f"the bilayer had not settled after {total:g} ns of equilibration: between the last two quarters "
                f"the area in the membrane plane changed by {100 * verdict['area_change']:+.1f} % (tolerance "
                f"{100 * settings.area_tolerance:g} %){water}. A longer membrane.equilibration_ns, in a new "
                "output.dir, gives it more time")
            log(f"WARNING: {notes[-1]}")
    pos, vel, bx = _state_arrays(ctx)
    del ctx, integ
    half = len(frames) // 2
    boxes_arr = np.asarray(boxes)
    return EquilibrationResult(positions=pos, velocities=vel, box=bx, frames=np.asarray(frames), frame_boxes=boxes_arr,
                               frame_atoms=list(frame_atoms), timestep_fs=dt_fs, notes=notes,
                               mean_volume_nm3=float(np.mean(volumes[half:])),
                               mean_box=boxes_arr[half:].mean(axis=0).astype(float), simulated_ns=float(total),
                               area_nm2=np.asarray(areas), pocket_waters=np.asarray(waters), settled=verdict)


def pre_equilibrate(system: mm.System, positions, box, frame_atoms: list[int],
                    settings: EquilibrationSettings | None = None, device: int | None = None,
                    log=print) -> EquilibrationResult:
    """Minimise and equilibrate the fully interacting system (see the module docstring)."""
    settings = settings or EquilibrationSettings()
    first = float(settings.timestep_fs or 4.0)
    attempts = [first] + ([2.0] if first > 2.0 else [])
    last: Exception | None = None
    for i, dt in enumerate(attempts):
        try:
            result = _run_once(system, positions, box, frame_atoms, settings, float(dt), device, log)
            if i > 0:
                result.notes.append(f"time step lowered automatically to {dt:g} fs after an unstable run at "
                                    f"{attempts[0]:g} fs")
            return result
        except (mm.OpenMMException, EquilibrationError, FloatingPointError, ValueError) as exc:
            last = exc
            log(f"pre-equilibration at {dt:g} fs failed: {str(exc)[:200]}")
    raise EquilibrationError(f"pre-equilibration failed at every time step tried ({attempts}): {last}")
