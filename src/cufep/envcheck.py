"""Environment report and a quick functional test of both backends."""
from __future__ import annotations


def check_environment(device: int = 0, echo=print) -> bool:
    import numpy as np
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    from cufep import __version__
    from cufep.backends import get_backend
    from cufep.backends.cupotential import CuPotentialBackend
    from cufep.scheduler import detect_gpus

    echo(f"cufep {__version__}, OpenMM {mm.__version__}")
    platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    echo(f"OpenMM platforms: {', '.join(platforms)}")
    echo(f"GPUs not busy: {detect_gpus() or 'none'}")
    ok_plugin, why = CuPotentialBackend.available()
    echo(f"cuPotential plugin: {'available' if ok_plugin else 'NOT available'} ({why})")
    if "CUDA" not in platforms:
        echo("CUDA platform missing: start the container with `--gpus all` and a recent NVIDIA driver.")
        return False

    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(3.2, 3.2, 3.2) * u.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    T = 298.15 * u.kelvin
    stock = get_backend("openmm")
    integ = stock.make_integrator(T, 1 / u.picosecond, 2 * u.femtoseconds, seed=1)
    ctx = stock.make_context(stock.prepare_system(system), integ, device)
    ctx.setPositions(mod.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 300)
    integ.step(1000)
    st = ctx.getState(getPositions=True, getEnergy=True, getForces=True)
    pos, box = st.getPositions(asNumpy=True), st.getPeriodicBoxVectors()
    e_stock = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f_stock = np.asarray(st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))
    echo(f"stock OpenMM on GPU {device}: OK ({system.getNumParticles()} atoms, E = {e_stock:.1f} kJ/mol)")
    del ctx
    if not ok_plugin:
        echo("cuPotential test skipped; cuFEP will fall back to stock OpenMM.")
        return True
    cupo = get_backend("cupotential")
    integ = cupo.make_integrator(T, 1 / u.picosecond, 2 * u.femtoseconds, seed=1)
    ctx = cupo.make_context(cupo.prepare_system(system), integ, device)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f = np.asarray(st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))
    rel_e = abs(e - e_stock) / abs(e_stock)
    rel_f = float(np.linalg.norm(f - f_stock) / np.linalg.norm(f_stock))
    integ.step(1000)
    finite = bool(np.isfinite(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)))
    good = rel_e < 1e-4 and rel_f < 5e-3 and finite
    echo(f"cuPotential on GPU {device}: {'OK' if good else 'MISMATCH'} "
         f"(relative energy difference {rel_e:.1e}, relative force difference {rel_f:.1e}, dynamics "
         f"{'stable' if finite else 'unstable'})")
    return good
