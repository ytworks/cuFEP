"""Is the interaction-group neighbour list stale after setPositions? Which call refreshes it?

Same trajectory as an earlier check (segments at states 0, 5, 10, 5, 0). For every
candidate remedy the Context is first put back at the last configuration, then moved to the
first one (10 ps earlier), the remedy is applied and all state energies are compared with the
values the same Context computed right after the dynamics that produced that configuration.
"""
import sys

import numpy as np
import openmm as mm
import openmm.unit as u

from cufep.alchemy.factory import SOFTCORE_GROUP
from cufep.alchemy.schedule import fixed_schedule
from cufep.backends import get_backend
from cufep.sampling.states import STAGES_SOLVENT, StateEnergyEvaluator, set_state

KJ = u.kilojoule_per_mole
DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
prec = sys.argv[2] if len(sys.argv) > 2 else "double"
xml = open("spikes/out/tyk2_ejm_31_cube_alch_system.xml").read()
data = np.load("spikes/out/tyk2_ejm_31_cube_equil.npz")
schedule = fixed_schedule(list(STAGES_SOLVENT), {"discharge": 4, "vanish": 6})

system = mm.XmlSerializer.deserialize(xml)
be = get_backend("openmm", precision=prec)
integ = be.make_integrator(298.15 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=1)
c = be.make_context(be.prepare_system(system), integ, DEVICE)
ev = StateEnergyEvaluator(c, schedule.states, 298.15, None)
c.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in data["box"]])
c.setPositions(data["positions"])
c.setVelocities(data["velocities"])
saved = []
for k in (0, 5, 10, 5, 0):
    set_state(c, schedule.states[k])
    integ.step(625)
    st = c.getState(getPositions=True, getForces=True)
    saved.append((st.getPositions(asNumpy=True), st.getPeriodicBoxVectors(), ev.reduced_potentials()[0],
                  st.getForces(asNumpy=True).value_in_unit(KJ / u.nanometer)))
first, last = saved[0], saved[-1]


def go_to(cfg):
    c.setPeriodicBoxVectors(*cfg[1])
    c.setPositions(cfg[0])


def all_forces():
    return c.getState(getForces=True).getForces(asNumpy=True).value_in_unit(KJ / u.nanometer)


REMEDIES = {
    "nothing": lambda: None,
    "energy of all groups": lambda: c.getState(getEnergy=True),
    "forces of all groups": all_forces,
    "energy of the soft-core group": lambda: c.getState(getEnergy=True, groups={SOFTCORE_GROUP}),
    "forces of the soft-core group": lambda: c.getState(getForces=True, groups={SOFTCORE_GROUP}),
    "integrator.step(0)": lambda: integ.step(0),
    "reinitialize(preserveState=True)": lambda: c.reinitialize(preserveState=True),
}
np.set_printoptions(linewidth=200, precision=3, suppress=True)
for label, remedy in REMEDIES.items():
    go_to(last)
    all_forces()                         # consistent at the last configuration
    go_to(first)
    remedy()
    x = ev.reduced_potentials()[0]
    err = x - first[2]
    print(f"[{prec}] setPositions + {label:34s}: max |state energy error| {np.abs(err).max():8.4f} kT")
go_to(last)
all_forces()
go_to(first)
set_state(c, schedule.states[0])
f = all_forces()
print(f"[{prec}] first all-group force evaluation after setPositions: max |force error| "
      f"{np.abs(f - first[3]).max():.4f} kJ/mol/nm (mean |force| {np.abs(first[3]).mean():.0f})")
