"""Is the soft-core (interaction group) neighbour list kept up to date during MD on each backend?

The soft-core force is a stock CustomNonbondedForce on both backends, so its energy and
forces at given positions must equal those of a fresh stock double-precision Context. A
Context runs MD at a state with Lennard-Jones on, and after every segment the soft-core
group is compared with the fresh reference.

    spikes/run.sh spikes/m2_softcore_list_during_md.py [device]
"""
import sys

import numpy as np
import openmm as mm
import openmm.unit as u

from cufep.alchemy.factory import SOFTCORE_GROUP
from cufep.alchemy.schedule import fixed_schedule
from cufep.backends import get_backend
from cufep.sampling.states import STAGES_SOLVENT, set_state

KJ = u.kilojoule_per_mole
KT = 8.31446261815324e-3 * 298.15
DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
xml = open("spikes/out/tyk2_ejm_31_cube_alch_system.xml").read()
data = np.load("spikes/out/tyk2_ejm_31_cube_equil.npz")
schedule = fixed_schedule(list(STAGES_SOLVENT), {"discharge": 4, "vanish": 6})
S = schedule.states[6]                               # charges off, lambda_sterics = 0.8
print("state:", S)


def make(name, precision):
    be = get_backend(name, precision=precision)
    integ = be.make_integrator(298.15 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=1)
    return be.make_context(be.prepare_system(mm.XmlSerializer.deserialize(xml)), integ, DEVICE), integ


def soft(c):
    st = c.getState(getEnergy=True, getForces=True, groups={SOFTCORE_GROUP})
    return (st.getPotentialEnergy().value_in_unit(KJ) / KT,
            st.getForces(asNumpy=True).value_in_unit(KJ / u.nanometer))


ref_c, _ = make("openmm", "double")
set_state(ref_c, S)
box0 = [mm.Vec3(*r) for r in data["box"]]
for name, prec in (("openmm", "mixed"), ("cupotential", "mixed")):
    c, integ = make(name, prec)
    c.setPeriodicBoxVectors(*box0)
    c.setPositions(data["positions"])
    c.setVelocities(data["velocities"])
    set_state(c, S)
    for seg in range(6):
        integ.step(1250)                              # 5 ps
        st = c.getState(getPositions=True)
        e, f = soft(c)                                # soft-core group alone, straight after MD
        ref_c.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
        ref_c.setPositions(st.getPositions(asNumpy=True))
        ref_c.getState(getEnergy=True)                # all groups once: lists rebuilt together
        e_ref, f_ref = soft(ref_c)
        print(
            f"{name:11s} {prec} after {5 * (seg + 1):3d} ps: soft-core energy {e:10.4f} kT, error {e - e_ref:+9.4f} "
            f"kT; "
            f"max |force error| {np.abs(f - f_ref).max():8.3f} kJ/mol/nm (max |force| {np.abs(f_ref).max():.0f})",
            flush=True,
        )
    del c
