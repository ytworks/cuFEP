"""Minimisation and plain MD of the fully interacting system before the alchemical run.

Follows OpenFE's ABFE protocol: minimise, NVT, NPT, then an NPT production run whose
frames are used for the restraint search. Minimisation, NVT and NPT run on stock OpenMM
(the cuPotential plugin is reported to fail on unminimised coordinates). The production
run, which is the long part and cannot be spread over GPUs, runs on the backend of the leg
(``md_backend``); if that backend cannot run the system or fails, it is repeated on stock
OpenMM and the fact is recorded, unless ``backend.fallback`` is ``none``: then nothing else runs
it and the leg stops with the reason, as the alchemical run of the leg does.

Time step: 4 fs with hydrogen mass repartitioning by default. If the run blows up it is
repeated once from the minimised structure at 2 fs and the fact is recorded.

Every unrestrained MD of the fully interacting system goes through :func:`unrestrained_md` (the
backend of the leg and the repeat on stock OpenMM) and :func:`with_time_step_recovery` (the
repeat at 2 fs), so that the pre-equilibration and the MD from which a restraint is chosen
(:mod:`cufep.sampling.restraint_md`) choose their backend and recover by one rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import openmm as mm
import openmm.unit as u

from cufep.backends import get_backend
from cufep.backends.base import BackendError, backend_precision


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
    barostat_frequency: int = 25           # MD steps between barostat moves (dynamics.barostat_frequency)
    platform: str = "CUDA"
    precision: str = "mixed"
    seed: int = 0
    #: backend of the production run (the minimisation, NVT and NPT always use stock OpenMM)
    md_backend: str = "openmm"
    #: what runs the production if ``md_backend`` cannot or fails: openmm | none (backend.fallback)
    fallback: str = "openmm"
    #: threads of a Context of the CPU platform (resources.cpu_threads_per_worker); None: OpenMM's own number
    cpu_threads: int | None = None
    #: the MD after NVT and NPT keeps the position restraints of ``restrained_atoms`` (not the wall): the solute
    #: stays in the pose that was given while its surroundings settle (restraints.selection: interactions, whose
    #: trajectories start from that pose). With ``production_ns`` 0 there is no such MD and the pre-equilibration
    #: ends with the NPT
    hold_solute: bool = False
    #: heavy atoms held by harmonic position restraints during NVT/NPT (released for production)
    restrained_atoms: list[int] = field(default_factory=list)
    restraint_k: float = 1000.0            # kJ/mol/nm^2
    ligand_atoms: list[int] = field(default_factory=list)
    water_oxygens: list[int] = field(default_factory=list)
    #: bilayers, restrained phases only: keep these water oxygens out of the hydrophobic core
    #: (|z - wall_centre_nm| < wall_half_width_nm) while a freshly built membrane closes up
    wall_atoms: list[int] = field(default_factory=list)
    wall_centre_nm: float = 0.0
    wall_half_width_nm: float = 0.0
    wall_k: float = 500.0                  # kJ/mol/nm^2
    #: a bilayer system: say how the bilayer changed at the end of the unrestrained MD (the
    #: equilibration of a bilayer does; an MD that starts from an equilibrated one does not)
    bilayer_report: bool = True


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
    #: membrane systems: how the bilayer area and the pocket hydration changed at the end (membrane_drift)
    drift: dict | None = None
    #: the backend that ran the unrestrained MD: the one of the leg, or stock OpenMM after a recovery
    backend: str = "openmm"


def _context(system: mm.System, backend, settings: EquilibrationSettings, dt_fs: float, device, barostat: bool,
             restraint_reference: np.ndarray | None, seed: int, barostat_seed: int | None = None,
             wall: bool = True):
    """A Context of ``backend`` for one part of the MD. ``seed`` is that of the integrator; the
    barostat takes ``barostat_seed``, or the next number."""
    prepared = backend.prepare_system(system)
    temperature = settings.temperature_K * u.kelvin
    if barostat:
        pressure = settings.pressure_bar * u.bar
        barostat_seed = seed + 1 if barostat_seed is None else int(barostat_seed)
        if settings.barostat == "membrane":
            prepared.addForce(backend.make_membrane_barostat(pressure, temperature, settings.barostat_frequency,
                                                              seed=barostat_seed))
        else:
            prepared.addForce(backend.make_isotropic_barostat(pressure, temperature, settings.barostat_frequency,
                                                              seed=barostat_seed))
    if restraint_reference is not None and settings.restrained_atoms:
        force = mm.CustomExternalForce("0.5*k_posres*periodicdistance(x,y,z,x0,y0,z0)^2")
        force.addGlobalParameter("k_posres", settings.restraint_k)
        for name in ("x0", "y0", "z0"):
            force.addPerParticleParameter(name)
        for i in settings.restrained_atoms:
            force.addParticle(int(i), [float(v) for v in restraint_reference[i]])
        prepared.addForce(force)
    if wall and restraint_reference is not None and settings.wall_atoms and settings.wall_half_width_nm > 0:
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
    backend = get_backend("openmm", precision=settings.precision, platform=settings.platform,
                          cpu_threads=settings.cpu_threads)
    from cufep.steps import md_steps

    temperature = settings.temperature_K * u.kelvin
    steps = lambda ns: md_steps(ns * 1000.0, dt_fs, least=0)                  # noqa: E731

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

    if settings.production_ns <= 0:
        # no MD after the NPT: its end is the end of the pre-equilibration, and its one frame
        atoms = [int(i) for i in frame_atoms]
        return EquilibrationResult(
            positions=pos, velocities=vel, box=bx, frames=pos[atoms][None].astype(np.float32),
            frame_boxes=np.asarray(bx, dtype=float)[None], frame_atoms=atoms, timestep_fs=dt_fs, notes=[],
            mean_volume_nm3=float(abs(np.linalg.det(bx))), mean_box=np.asarray(bx, dtype=float), simulated_ns=0.0,
            area_nm2=np.array([float(bx[0][0] * bx[1][1])]), pocket_waters=np.array([]), drift=None, backend="openmm")
    # ---- production: frames for the restraint search; the solute free, or held where it was given ----
    return unrestrained_md(system, (pos, vel, bx), frame_atoms, settings, dt_fs, device, log,
                           held_at=np.asarray(positions) if settings.hold_solute else None)


def unrestrained_md(system, start, frame_atoms, settings: EquilibrationSettings, dt_fs: float, device, log,
                    what: str = "pre-equilibration MD", seed: int | None = None,
                    barostat_seed: int | None = None, held_at: np.ndarray | None = None) -> EquilibrationResult:
    """An unrestrained NPT run of the fully interacting system from ``start`` (positions,
    velocities, box) on the backend of the leg, ``settings.md_backend``.

    If that backend cannot run the System, stock OpenMM runs it; if the run fails on that backend,
    it is repeated from ``start`` on stock OpenMM, and the log and the notes of the result say so.
    With ``settings.fallback`` "none" (``backend.fallback``) stock OpenMM runs neither: a backend
    that cannot run the System is a :class:`~cufep.backends.base.BackendError`, and the error of a
    run that fails is passed on (review of the plan of 2026-11-21: this recovery took no notice of
    the setting, which the alchemical run of the same leg obeys). ``what`` names the MD in those
    lines. The backend runs in the precision the alchemical run would give it
    (:func:`~cufep.backends.base.backend_precision`). The result records the backend that ran it.
    """
    stock = get_backend("openmm", precision=settings.precision, platform=settings.platform,
                        cpu_threads=settings.cpu_threads)
    notes: list[str] = []
    md_backend = stock
    # the System in the box the MD starts in: the backend is asked about that box, and makes its Context of
    # the same System. Asked about the start and given the System of the preparation, whose box can be the
    # smaller one, the plugin refused it after it had been found able, outside the fallback (review of eb6f89f)
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in np.asarray(start[2], dtype=float)])
    if settings.md_backend != "openmm":
        try:
            md_backend = get_backend(settings.md_backend, platform=settings.platform,
                                     precision=backend_precision(settings.md_backend, settings.precision),
                                     cpu_threads=settings.cpu_threads)
            if hasattr(md_backend, "check_system"):
                md_backend.check_system(system)
        except Exception as exc:  # noqa: BLE001 - any reason the backend cannot run this system
            if settings.fallback == "none":
                raise BackendError(f"the {settings.md_backend} backend cannot run the {what} ({str(exc)[:150]}) and "
                                   "backend.fallback is none") from exc
            log(f"{what} on stock OpenMM ({settings.md_backend} cannot run it: {str(exc)[:150]})")
            notes.append(f"the {what} ran on stock OpenMM: {settings.md_backend} cannot run it ({str(exc)[:150]})")
            md_backend = stock
    try:
        return _production(system, md_backend, start, frame_atoms, settings, dt_fs, device, log, notes, seed,
                           barostat_seed, held_at)
    except (mm.OpenMMException, EquilibrationError) as exc:
        if md_backend is stock:
            raise
        if settings.fallback == "none":
            log(f"{what} failed on {settings.md_backend} ({str(exc)[:150]}); backend.fallback is none, so it is not "
                "repeated on stock OpenMM")
            raise
        log(f"{what} failed on {settings.md_backend} ({str(exc)[:150]}); repeating it on stock OpenMM")
        notes.append(f"the {what} failed on {settings.md_backend} and was repeated on stock OpenMM")
        return _production(system, stock, start, frame_atoms, settings, dt_fs, device, log, notes, seed,
                           barostat_seed, held_at)


#: how the note of earlier versions about a bilayer that had not settled began (a verdict with levels of
#: cuFEP's own: no longer made, and removed from stored notes)
BILAYER_NOTE = "the bilayer had not settled after"


def membrane_drift(areas, waters) -> dict:
    """How the bilayer changed at the end of its equilibration, as numbers without a verdict (cuFEP
    judges nothing by levels of its own, the owner 2026-11-20): the relative change of the mean area
    of the box in the membrane plane and, if counted, the change of the mean number of water
    molecules near the ligand, from the third to the fourth quarter of the run. None where the run
    has fewer than five frames per quarter."""
    q = len(areas) // 4
    out = {"area_change": None, "pocket_water_change": None}
    if q < 5:
        return out
    a3, a4 = float(np.mean(areas[2 * q:3 * q])), float(np.mean(areas[3 * q:]))
    out["area_change"] = (a4 - a3) / a3
    if len(waters):
        out["pocket_water_change"] = float(np.mean(waters[3 * q:]) - np.mean(waters[2 * q:3 * q]))
    return out


def _production(system, backend, start, frame_atoms, settings: EquilibrationSettings, dt_fs: float, device, log,
                notes: list[str], seed: int | None = None, barostat_seed: int | None = None,
                held_at: np.ndarray | None = None) -> EquilibrationResult:
    """An unrestrained NPT run of ``settings.production_ns`` on ``backend`` with a frame after
    every ``settings.frame_interval_ps`` and one at its end: the one after NVT and NPT, and each
    trajectory from which a restraint is chosen. ``seed`` and ``barostat_seed`` are those of the
    integrator and of the barostat (by default the ones of the pre-equilibration)."""
    pos, vel, bx = start
    # held_at: the positions at which ``settings.restrained_atoms`` are held (the solute of a pre-equilibration
    # that keeps the pose that was given); None: nothing is held
    ctx, integ = _context(system, backend, settings, dt_fs, device, True, held_at,
                          settings.seed + 200 if seed is None else int(seed), barostat_seed, wall=False)
    _load(ctx, pos, bx, vel)
    from cufep.steps import md_steps

    per_frame = md_steps(settings.frame_interval_ps, dt_fs)
    lig = np.asarray(settings.ligand_atoms, dtype=int)
    wat = np.asarray(settings.water_oxygens, dtype=int)
    frames, boxes, volumes, areas, waters = [], [], [], [], []
    steps_done = 0                     # MD steps integrated in this Context (counted: a plugin's
                                       # integrator need not advance the time of the Context)

    def advance(ns: float) -> None:
        # all the MD that is asked for, in pieces of one frame interval and a last piece of what is
        # left, with a frame after each (review of PR 1: round(50 / 20) frames integrated 40 ps of
        # the 50 ps asked, and 50 ps were recorded)
        nonlocal steps_done
        remaining = md_steps(ns * 1000.0, dt_fs)
        while remaining > 0:
            piece = min(per_frame, remaining)
            integ.step(piece)
            steps_done += piece
            remaining -= piece
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

    advance(settings.production_ns)
    # a membrane system: how the bilayer changed at the end, as numbers (no verdict, and the run is never
    # made longer for it)
    drift = None
    if settings.barostat == "membrane" and settings.bilayer_report:
        drift = membrane_drift(areas, waters)
        if drift["area_change"] is not None:
            water = "" if drift["pocket_water_change"] is None else \
                f", water molecules near the ligand {drift['pocket_water_change']:+.1f}"
            log(f"bilayer between the last two quarters of its equilibration: "
                f"area {100 * drift['area_change']:+.2f} %{water}")
    pos, vel, bx = _state_arrays(ctx)
    # the time that was integrated, not the one asked for
    simulated_ns = steps_done * float(dt_fs) * 1.0e-6
    ctx = integ = None                  # release the Context (advance() is not called again)
    half = len(frames) // 2
    boxes_arr = np.asarray(boxes)
    return EquilibrationResult(positions=pos, velocities=vel, box=bx, frames=np.asarray(frames), frame_boxes=boxes_arr,
                               frame_atoms=list(frame_atoms), timestep_fs=dt_fs, notes=notes,
                               mean_volume_nm3=float(np.mean(volumes[half:])),
                               mean_box=boxes_arr[half:].mean(axis=0).astype(float), simulated_ns=simulated_ns,
                               area_nm2=np.asarray(areas), pocket_waters=np.asarray(waters), drift=drift,
                               backend=str(getattr(backend, "name", "openmm")))


def pre_equilibrate(system: mm.System, positions, box, frame_atoms: list[int],
                    settings: EquilibrationSettings | None = None, device: int | None = None,
                    log=print) -> EquilibrationResult:
    """Minimise and equilibrate the fully interacting system (see the module docstring)."""
    settings = settings or EquilibrationSettings()
    return with_time_step_recovery(
        lambda dt: _run_once(system, positions, box, frame_atoms, settings, dt, device, log),
        float(settings.timestep_fs or 4.0), log)


def with_time_step_recovery(run, first_fs: float, log, what: str = "pre-equilibration") -> EquilibrationResult:
    """``run(time step in fs)`` at ``first_fs`` and, if that fails and is above 2 fs, once more at
    2 fs from its start; the notes of the result say that the time step was lowered. ``what``
    names the MD in the log and in the error of a run that fails at every time step."""
    first = float(first_fs)
    attempts = [first] + ([2.0] if first > 2.0 else [])
    last: Exception | None = None
    for i, dt in enumerate(attempts):
        try:
            result = run(float(dt))
            if i > 0:
                result.notes.append(f"time step lowered automatically to {dt:g} fs after an unstable run at "
                                    f"{attempts[0]:g} fs")
            return result
        except (mm.OpenMMException, EquilibrationError, FloatingPointError, ValueError) as exc:
            last = exc
            log(f"{what} at {dt:g} fs failed: {str(exc)[:200]}")
    raise EquilibrationError(f"{what} failed at every time step tried ({attempts}): {last}")
