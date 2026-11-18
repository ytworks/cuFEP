"""`cufep validate`: do the two backends give the same alchemical energies for *this* system?

Prepares the first ligand of the configuration (or reuses a finished preparation), relaxes
each leg briefly on stock OpenMM, and evaluates every state of a fixed test schedule on
stock OpenMM (mixed and double precision) and on cuPotential for the same configurations.
What matters for the free energy are differences between states, so those are compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate(cfg, ligand_name: str | None = None, device: int = 0, echo=print) -> bool:
    import openmm as mm
    import openmm.unit as u

    from cufep import run as driver
    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.backends import get_backend
    from cufep.backends.cupotential import CuPotentialBackend
    from cufep.sampling.states import STAGES_SOLVENT, StateEnergyEvaluator
    from cufep.workflow import load_prepared, prepare_ligand, safe_name

    ok_plugin, why = CuPotentialBackend.available()
    if not ok_plugin:
        echo(f"cuPotential is not available ({why}); nothing to compare.")
        return False
    resolved, _report, protein, ligands = driver.plan(cfg, echo)
    ligand = next((l for l in ligands if l.name == ligand_name), ligands[0])
    out_dir = Path(resolved.output.dir)
    lig_dir = out_dir / safe_name(ligand.name)
    prepare_ligand(resolved, ligand, protein, lig_dir, bool(resolved.membrane.enabled), resolved.solvation.box_shape, echo)
    info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    T = resolved.thermodynamics.temperature_K
    schedule = fixed_schedule(list(STAGES_SOLVENT), {"discharge": 4, "vanish": 6})
    all_ok = True
    for leg in ("complex", "solvent"):
        prep = load_prepared(lig_dir / "prep", leg)
        system = mm.XmlSerializer.deserialize(prep["system_xml"])
        stock = get_backend("openmm")
        barostat = info["complex_barostat"] if leg == "complex" else "isotropic"
        # brief relaxation on stock OpenMM, then a few frames at three states
        prepared = stock.prepare_system(system)
        integ = stock.make_integrator(T * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=1)
        ctx = stock.make_context(prepared, integ, device)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in prep["box"]])
        ctx.setPositions(prep["positions"])
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
        ctx.setVelocitiesToTemperature(T * u.kelvin, 1)
        integ.step(5000)
        st = ctx.getState(getPositions=True)
        pos0, box0 = st.getPositions(asNumpy=True), st.getPeriodicBoxVectors()
        del ctx
        system.setDefaultPeriodicBoxVectors(*box0)
        alch = make_alchemical_system(system, prep["ligand"], resolved.alchemy.softcore_alpha)
        try:
            CuPotentialBackend.check_system(alch.system)
        except Exception as exc:  # noqa: BLE001
            echo(f"{leg}: cuPotential cannot run this system ({exc}); cuFEP would use stock OpenMM for this leg.")
            continue
        contexts = {}
        for name, backend in (("stock mixed", stock), ("stock double", get_backend("openmm", precision="double")),
                              ("cuPotential", get_backend("cupotential"))):
            integ = backend.make_integrator(T * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=2)
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
        ref = np.array(rows["stock double"])
        ref_d = ref - ref[np.arange(len(frames)), own][:, None]
        mask = np.abs(ref_d) < 60.0
        worst = {}
        for name in ("stock mixed", "cuPotential"):
            x = np.array(rows[name])
            d = x - x[np.arange(len(frames)), own][:, None]
            worst[name] = float(np.abs(d - ref_d)[mask].max())
        rel = float(np.abs(np.array(rows["cuPotential"]) - ref)[mask].max() / np.abs(ref[mask]).max())
        good = worst["cuPotential"] < 0.15 and rel < 1e-4
        all_ok &= good
        echo(f"{leg} ({alch.system.getNumParticles()} atoms, barostat {barostat}): between-state energy differences vs "
             f"stock double precision: cuPotential max {worst['cuPotential']:.3f} kT, stock mixed max "
             f"{worst['stock mixed']:.3f} kT; relative total-energy difference {rel:.1e} -> {'OK' if good else 'MISMATCH'}")
        for c, _i, _e in contexts.values():
            del c
    echo("The two backends approximate Ewald sums differently, so differences of a few hundredths of kT are "
         "expected; cuFEP never mixes samples from both backends in one estimate.")
    return all_ok
