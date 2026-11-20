"""Time the plain MD before the alchemical run with the production part on either backend.

Results are for internal use only (they contain timings of the cuPotential plugin).

    spikes/run.sh spikes/m3_preequil.py runs/tyk2_ejm_31 ejm_31 complex cupotential 0
"""
import json
import sys
import time
from pathlib import Path

import openmm as mm

from cufep.sampling.equilibrate import EquilibrationSettings, pre_equilibrate
from cufep.workflow import load_prepared

run_dir, ligand, leg, md_backend, device = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
prep = load_prepared(run_dir / ligand / "prep", leg)
system = mm.XmlSerializer.deserialize(prep["system_xml"])
cx = leg == "complex"
settings = EquilibrationSettings(
    nvt_ns=0.25 if cx else 0.1,
    npt_ns=0.5 if cx else 0.2,
    production_ns=5.0 if cx else 0.5,
    seed=7,
    md_backend=md_backend,
)
marks = [time.perf_counter()]
t0 = time.perf_counter()
eq = pre_equilibrate(system, prep["positions"], prep["box"], prep["solute"], settings, device, print)
wall = time.perf_counter() - t0
out = {
    "what": "preequil",
    "leg": leg,
    "md_backend": md_backend,
    "atoms": system.getNumParticles(),
    "seconds": wall,
    "timestep_fs": eq.timestep_fs,
    "mean_volume_nm3": eq.mean_volume_nm3,
    "frames": int(len(eq.frames)),
    "notes": eq.notes,
}
Path("spikes/out/m3_speed").mkdir(parents=True, exist_ok=True)
with open("spikes/out/m3_speed/preequil.jsonl", "a") as fh:
    fh.write(json.dumps(out) + "\n")
print(json.dumps(out))
