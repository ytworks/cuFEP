"""How long must the windows of the adaptive precalculation run for their overlaps to be right?

The paper behind adaptive lambda scheduling (relative binding, ligand in solution) uses 50 ps
per window. This script runs the windows of a finished leg for much longer, in the same order
and with the same start rule as the precalculation, records the energies at all candidates
and prints the overlap of neighbouring windows from different parts of the data, next to the
overlaps of the production.

    spikes/run.sh spikes/m9_precalculation_length.py runs/tyk2_default_5gpu ejm_31 0 complex --device 0 --ns 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("ligand")
ap.add_argument("repeat", type=int)
ap.add_argument("leg")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--ns", type=float, default=1.0)
ap.add_argument("--stage", default="vanish")
ap.add_argument("--out", default="spikes/out/m9_precalc")
args = ap.parse_args()

import openmm as mm  # noqa: E402
import openmm.unit as u  # noqa: E402

from cufep.alchemy.factory import LAMBDA_STERICS, make_alchemical_system  # noqa: E402
from cufep.alchemy.restraints import BoreschRestraint  # noqa: E402
from cufep.alchemy.schedule import candidate_energies, stage_overlap  # noqa: E402
from cufep.backends import get_backend  # noqa: E402
from cufep.config.schema import Config  # noqa: E402
from cufep.sampling.states import R_GAS, set_state, stage_state  # noqa: E402
from cufep.workflow import load_prepared  # noqa: E402

run_dir = Path(args.run_dir)
raw = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
# runs made before initial_windows became a per-leg setting; this script does not use it
raw.get("lambda_schedule", {}).get("adaptive", {}).pop("initial_windows", None)
cfg = Config.model_validate(raw)
lig_dir = run_dir / args.ligand
leg_dir = lig_dir / f"repeat_{args.repeat}" / args.leg
prep_info = json.loads((lig_dir / "prep" / "prep.json").read_text())
prep = load_prepared(lig_dir / "prep", args.leg)
system = mm.XmlSerializer.deserialize(prep["system_xml"])
eq = np.load(leg_dir / "equil.npz")
system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in eq["box"]])
alch = make_alchemical_system(system, prep["ligand"], cfg.alchemy.softcore_alpha)
if args.leg == "complex":
    alch.system.addForce(
        BoreschRestraint.from_dict(json.loads((leg_dir / "restraint.json").read_text())["restraint"]).create_force()
    )
sched = yaml.safe_load((leg_dir / "schedule.yaml").read_text())
T = cfg.thermodynamics.temperature_K
dt = float(eq["timestep_fs"])
backend = get_backend(sched.get("backend", "cupotential"), precision="mixed")
prepared = backend.prepare_system(alch.system)
prepared.addForce(backend.make_isotropic_barostat(cfg.thermodynamics.pressure_bar * u.bar, T * u.kelvin, 25, seed=3))
integ = backend.make_integrator(T * u.kelvin, cfg.dynamics.friction_per_ps / u.picosecond, dt * u.femtoseconds, seed=4)
ctx = backend.make_context(prepared, integ, args.device)
start = np.load(leg_dir / "start_configs.npz")

stage = args.stage
lams = sched["windows"][stage]
n_cand = int(sched["provenance"].get("candidates", 101))
grid = np.linspace(0.0, 1.0, n_cand)
windows = [int(round(v * (n_cand - 1))) for v in lams]
beta = 1.0 / (R_GAS * T)
interval_ps = 0.5
steps = int(round(interval_ps * 1000.0 / dt))
n_samples = int(round(args.ns * 1000.0 / interval_ps))
has_soft = LAMBDA_STERICS in ctx.getParameters()

# the window at the start of the stage as the precalculation left it; every further window
# starts from the end of the previous one, as in the precalculation
first_state = stage_state(stage, lams[0]).to_dict()
index0 = next(i for i, s in enumerate(sched["states"]) if all(abs(s[k] - first_state[k]) < 1e-9 for k in first_state))
ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in start["boxes"][index0]])
ctx.setPositions(start["positions"][index0])
ctx.setVelocities(start["velocities"][index0])
energies = []
for lam in lams:
    state = stage_state(stage, lam)
    set_state(ctx, state)
    e = np.empty((n_samples, n_cand))
    for i in range(n_samples):
        integ.step(steps)
        if has_soft:
            ctx.getState(getEnergy=True)
        e[i] = candidate_energies(ctx, stage, grid, beta)
        set_state(ctx, state)
    energies.append(e)
    print(f"window lambda {lam:g} done", flush=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out / f"{args.ligand}_{args.leg}_{stage}.npz", energies=np.array(energies), windows=np.array(windows)
)

res = json.loads((leg_dir / "result.json").read_text())
stage_names = list(sched["windows"])
offset = sum(len(sched["windows"][s]) - 1 for s in stage_names[: stage_names.index(stage)])
prod = res["analysis"]["overlap_neighbour"][offset: offset + len(lams) - 1]
print(f"\n{args.leg} leg, {stage} stage, windows {lams}")
print(f"{'data used':34s}" + " ".join(f"{a:g}-{b:g}".rjust(10) for a, b in zip(lams, lams[1:])))


def row(label, sl):
    sup = stage_overlap([e[sl] for e in energies], windows)
    print(f"{label:34s}" + " ".join(f"{v:10.3f}" for v in sup))


per_ps = 1.0 / interval_ps
for a, b in ((0, 50), (0, 200), (0, 500), (0, 1000), (50, 100), (150, 200), (450, 500), (950, 1000), (500, 1000)):
    if b <= args.ns * 1000:
        row(f"precalculation, {a}-{b} ps", slice(int(a * per_ps), int(b * per_ps)))
print(f"{'production (5 ns, replica exchange)':34s}" + " ".join(f"{v:10.3f}" for v in prod))
