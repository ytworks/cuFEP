"""Trace side-chain rotamers near the ligand through the plain MD, the coupled window and all replicas.

Used to explain why repeats of the TYK2 ejm_31 complex leg differ (ILE960 chi1).

    spikes/run.sh spikes/m2_repeat_rotamers.py [run_dir [ligand [input_protein.pdb]]]
"""
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
from stored_frames import stored_frames

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/tyk2_ejm_31")
lig_dir = run_dir / (sys.argv[2] if len(sys.argv) > 2 else "ejm_31")
input_pdb = sys.argv[3] if len(sys.argv) > 3 else "examples/kinase_tyk2/protein.pdb"
top_all = md.load(str(lig_dir / "prep/complex.pdb")).topology
WATCH = [("ILE", 960), ("GLU", 979), ("VAL", 981), ("LEU", 1030), ("ARG", 1027)]

def chi_table(traj):
    out = {}
    for fn, lab in ((md.compute_chi1, "chi1"), (md.compute_chi2, "chi2")):
        idx, val = fn(traj)
        for col, ix in enumerate(idx):
            res = traj.topology.atom(int(ix[1])).residue
            if (res.name, res.resSeq) in WATCH:
                out[f"{res.name}{res.resSeq} {lab}"] = np.degrees(val[:, col])
    return out

def code(deg):
    return np.where((deg >= -120) & (deg < 0), "-", np.where((deg >= 0) & (deg < 120), "+", "t"))

# input structure
inp = md.load(input_pdb)
print("input PDB:", {k: f"{v[0]:.0f}" for k, v in chi_table(inp).items()})
prep = md.load(str(lig_dir / "prep/complex.pdb"))
print("prepared :", {k: f"{v[0]:.0f}" for k, v in chi_table(prep).items()})
for rep_dir in sorted(lig_dir.glob("repeat_*")):
    r = int(rep_dir.name.split("_")[1])
    leg = rep_dir / "complex"
    if not (leg / "hrex" / "checkpoint.npz").exists():
        continue
    eq = np.load(leg / "equil.npz")
    top = top_all.subset(eq["frame_atoms"])
    t = md.Trajectory(eq["frames"], top)
    tab = chi_table(t)
    print(f"\nrepeat {r} plain MD (250 frames over 5 ns), rotamer per 10 frames (- = g-, t = trans, + = g+):")
    for k in ("ILE960 chi1", "GLU979 chi1", "VAL981 chi1", "LEU1030 chi2", "ILE960 chi2"):
        print(f"  {k:13s}", "".join(code(tab[k])[::10]))
    fr = stored_frames(leg / "hrex")
    tab = chi_table(md.Trajectory(fr, top))
    print(f"repeat {r} window 0 during replica exchange ({len(fr)} frames, every 10 ps), per 25 frames:")
    for k in ("ILE960 chi1", "GLU979 chi1", "VAL981 chi1", "LEU1030 chi2"):
        print(f"  {k:13s}", "".join(code(tab[k])[::25]))
    ck = np.load(leg / "hrex/checkpoint.npz")
    order = np.argsort(ck["state_of_replica"])
    tt = md.Trajectory(ck["positions"][order].astype(np.float32), top_all)
    tab = chi_table(tt)
    print(f"repeat {r} all replicas at the last checkpoint, ordered by lambda window (coupled -> decoupled):")
    for k in ("ILE960 chi1", "GLU979 chi1", "VAL981 chi1", "LEU1030 chi2"):
        print(f"  {k:13s}", "".join(code(tab[k])))
    sc = np.load(leg / "start_configs.npz")
    tab = chi_table(md.Trajectory(sc["positions"].astype(np.float32), top_all))
    print(f"repeat {r} start configurations of the windows:")
    print(f"  {'ILE960 chi1':13s}", "".join(code(tab['ILE960 chi1'])))
