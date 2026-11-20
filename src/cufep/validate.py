"""`cufep validate`: do the two backends give the same alchemical energies for *this* system?

Prepares the first ligand of the configuration (or the one named with ``--ligand``; or reuses a
finished preparation), relaxes each leg briefly on stock OpenMM, and evaluates every state of a
test schedule on stock OpenMM (mixed and double precision) and on cuPotential for the same
configurations. What matters for the free energy are differences between states, so those are
compared.

The alchemical System is the one a run samples (:func:`cufep.workflow.alchemical_leg`: soft-core,
REST2 with its region, the tail record), without the restraint, which is the same force on both
backends; the test schedule carries the REST2 scale of the run, so the windows in the middle of a
leg compare the scaled terms (review of PR 1: the comparison used a System without REST2 while
runs use it by default). A leg whose comparison cannot be made makes the validation fail, with
the reason (review of PR 1: two skipped legs returned success).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validation_system(cfg, system, prep: dict, box_nm, leg: str):
    """The alchemical System and the test schedule of the validation of one leg: the System a run
    samples (without its restraint), and evenly spaced discharge and vanish windows with the REST2
    scale of the run along them."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.sampling.states import STAGES_SOLVENT
    from cufep.workflow import alchemical_leg

    built = alchemical_leg(cfg, system, prep, np.asarray(box_nm, dtype=float), None, leg == "complex")
    scale = cfg.sampling.rest2.scale if built.rest2_on else 1.0
    return built, fixed_schedule(list(STAGES_SOLVENT), {"discharge": 4, "vanish": 6}, scale)


#: a pair of state and configuration is not compared when the state lies this much (kT) or more above
#: the sampled one on both backends: such states have no weight in the free energy
COMPARED_BELOW_KT = 60.0


def between_state_differences(reference_rows, rows, own_states, level_kT: float = COMPARED_BELOW_KT) -> dict:
    """How far the reduced potentials ``rows`` (configuration x state) of a backend are from
    ``reference_rows`` in what matters for the free energy: the differences between the states, each
    taken from the state the configuration was sampled in (``own_states``).

    A pair of configuration and state is compared when the signed difference between the states is
    below ``level_kT`` on either side, or finite on one side only. Only a state that is unfavourable
    by ``level_kT`` or more on both sides has no weight in the free energy; a state that one backend
    finds favourable, however much, is not left out because the other one does not (review of PR 1:
    +80 kT in the reference and -5 kT in cuPotential were left out, and the validation said OK; then,
    with the absolute value of the difference, +80 kT against -80 kT). Returns ``worst`` (kT), the
    largest relative difference of the energies themselves (``relative``) and the number of pairs
    compared; a pair with a non-finite difference makes ``worst`` infinite."""
    ref, x = np.asarray(reference_rows, dtype=float), np.asarray(rows, dtype=float)
    n = np.arange(ref.shape[0])
    own = np.asarray(own_states, dtype=int)
    ref_d, d = ref - ref[n, own][:, None], x - x[n, own][:, None]
    with np.errstate(invalid="ignore"):
        compared = (ref_d < level_kT) | (d < level_kT) | (np.isfinite(ref_d) != np.isfinite(d))
    if not compared.any():
        return {"worst": 0.0, "relative": 0.0, "compared": 0}
    gap = np.abs(d - ref_d)[compared]
    worst = float(gap.max()) if np.all(np.isfinite(gap)) else float("inf")
    scale = np.abs(ref[compared])
    rel_gap = np.abs(x - ref)[compared]
    relative = float(rel_gap.max() / scale.max()) if np.all(np.isfinite(rel_gap)) and scale.max() > 0 else float("inf")
    return {"worst": worst, "relative": relative, "compared": int(compared.sum())}


def verdict(results: dict[str, bool | None]) -> bool:
    """Whether a validation passed: every leg it was asked for was compared (None: not compared) and
    agreed. A validation that compared nothing has not passed."""
    return bool(results) and all(value is True for value in results.values())


def validate(cfg, ligand_name: str | None = None, device: int = 0, echo=print) -> bool:
    from cufep import run as driver
    from cufep.backends.cupotential import CuPotentialBackend
    from cufep.workflow import ligand_directories, ligands_of_run, select_ligands

    ok_plugin, why = CuPotentialBackend.available()
    if not ok_plugin:
        echo(f"cuPotential is not available ({why}); nothing to compare.")
        return False
    # the first ligand only when none is named; a name that is not there is an error, before anything is prepared
    ligands = ligands_of_run(cfg, [ligand_name] if ligand_name else [])
    resolved, _report, protein, ligands = driver.plan(cfg, echo, ligands)
    ligand = select_ligands(ligands, [ligand_name], "--ligand")[0] if ligand_name else ligands[0]
    out_dir = Path(resolved.output.dir)
    lig_dir = ligand_directories(out_dir, [ligand.name])[ligand.name]
    # the prepared systems are those of `cufep run` in the same directory: the lock of the directory
    # keeps the two from making them at the same time (review of PR 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = driver.lock_output_directory(out_dir)
    try:
        return _compare(resolved, ligand, protein, lig_dir, device, echo)
    finally:
        lock.close()


def _compare(resolved, ligand, protein, lig_dir, device: int, echo) -> bool:
    import openmm as mm
    import openmm.unit as u

    from cufep.backends import get_backend
    from cufep.backends.cupotential import CuPotentialBackend
    from cufep.sampling.states import StateEnergyEvaluator
    from cufep.workflow import load_prepared, prepare_ligand

    prepare_ligand(resolved, ligand, protein, lig_dir, bool(resolved.membrane.enabled), resolved.solvation.box_shape, echo)
    info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    T = resolved.thermodynamics.temperature_K
    results: dict[str, bool | None] = {}
    not_compared: dict[str, str] = {}
    for leg in ("complex", "solvent"):
        results[leg] = None
        prep = load_prepared(lig_dir / "prep", leg)
        system = mm.XmlSerializer.deserialize(prep["system_xml"])
        stock = get_backend("openmm")
        barostat = info["complex_barostat"] if leg == "complex" else "isotropic"
        # brief relaxation on stock OpenMM, then a few frames at three states
        prepared = stock.prepare_system(system)
        # the time step of the run is the upper limit of every MD of the validation (review of PR 1:
        # with constraints none and 1 fs the fixed 4 fs gave NaN)
        dt_run = float(resolved.dynamics.timestep_fs)
        integ = stock.make_integrator(T * u.kelvin, 1 / u.picosecond, min(2.0, dt_run) * u.femtoseconds, seed=1)
        ctx = stock.make_context(prepared, integ, device)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in prep["box"]])
        ctx.setPositions(prep["positions"])
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
        ctx.setVelocitiesToTemperature(T * u.kelvin, 1)
        integ.step(5000)
        st = ctx.getState(getPositions=True)
        pos0, box0 = st.getPositions(asNumpy=True), st.getPeriodicBoxVectors()
        del ctx
        box_nm = np.array([[float(v) for v in vec.value_in_unit(u.nanometer)] for vec in box0])
        built, schedule = validation_system(resolved, system, prep, box_nm, leg)
        alch = built.alch
        try:
            CuPotentialBackend.check_system(alch.system)
        except Exception as exc:  # noqa: BLE001
            not_compared[leg] = f"cuPotential cannot run this system ({exc}); cuFEP would use stock OpenMM for this leg"
            echo(f"{leg}: NOT COMPARED: {not_compared[leg]}")
            continue
        contexts = {}
        for name, backend in (("stock mixed", stock), ("stock double", get_backend("openmm", precision="double")),
                              ("cuPotential", get_backend("cupotential"))):
            integ = backend.make_integrator(T * u.kelvin, 1 / u.picosecond, min(4.0, dt_run) * u.femtoseconds, seed=2)
            c = backend.make_context(backend.prepare_system(alch.system), integ, device)
            contexts[name] = (c, integ, StateEnergyEvaluator(c, schedule.states, T, None))
        frames = []
        c, integ, _ = contexts["stock mixed"]
        c.setPeriodicBoxVectors(*box0); c.setPositions(pos0); c.setVelocitiesToTemperature(T * u.kelvin, 3)
        from cufep.sampling.states import set_state
        for state in (schedule.states[0], schedule.states[len(schedule.states) // 2], schedule.states[-1]):
            set_state(c, state)
            integ.step(1500)
            for _ in range(3):
                integ.step(500)
                s = c.getState(getPositions=True)
                frames.append((state, s.getPositions(asNumpy=True), s.getPeriodicBoxVectors()))
        rows = {name: [] for name in contexts}
        for _state, p, b in frames:
            for name, (c, _i, ev) in contexts.items():
                c.setPeriodicBoxVectors(*b); c.setPositions(p)
                rows[name].append(ev.reduced_potentials()[0])
        own = [schedule.states.index(f[0]) for f in frames]
        compared = {name: between_state_differences(rows["stock double"], rows[name], own)
                    for name in ("stock mixed", "cuPotential")}
        worst = {name: c["worst"] for name, c in compared.items()}
        rel = compared["cuPotential"]["relative"]
        good = compared["cuPotential"]["compared"] > 0 and worst["cuPotential"] < 0.15 and rel < 1e-4
        results[leg] = bool(good)
        region = (f"REST2 scale {schedule.rest2_scale:g}, {len(built.hot_atoms)} protein atoms hot"
                  if built.rest2_on else "no REST2")
        echo(f"{leg} ({alch.system.getNumParticles()} atoms, barostat {barostat}, {region}): between-state energy differences vs "
             f"stock double precision: cuPotential max {worst['cuPotential']:.3f} kT, stock mixed max "
             f"{worst['stock mixed']:.3f} kT; relative total-energy difference {rel:.1e} -> {'OK' if good else 'MISMATCH'}")
        for c, _i, _e in contexts.values():
            del c
    echo("The two backends approximate Ewald sums differently, so differences of a few hundredths of kT are "
         "expected; cuFEP never mixes samples from both backends in one estimate.")
    if not_compared:
        echo("FAILED: the comparison asked for was not made for " + "; ".join(f"the {leg} leg ({why})"
                                                                              for leg, why in not_compared.items()))
    return verdict(results)
