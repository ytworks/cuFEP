"""Why do repeats of the same complex leg differ? Structural comparison of the repeats.

For every repeat of a finished run this compares
  * the plain MD before the alchemical run (250 frames of protein + ligand in equil.npz),
  * the fully interacting window during replica exchange (frames in hrex/samples.h5),
  * every replica at the last checkpoint (all atoms, one per lambda window),
and reports ligand-protein polar contacts, ligand RMSD, ligand torsions, side-chain
rotamers near the ligand and the number of water molecules in the pocket per window.

    spikes/run.sh spikes/m2_repeat_spread.py runs/tyk2_ejm_31 ejm_31
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
from stored_frames import stored_frames

run_dir, ligand = Path(sys.argv[1]), sys.argv[2]
lig_dir = run_dir / ligand
top_all = md.load(str(lig_dir / "prep" / "complex.pdb")).topology
repeats = sorted(
    int(p.name.split("_")[1]) for p in lig_dir.glob("repeat_*") if (p / "complex" / "result.json").exists()
)
KJ = 4.184


def leg(r):
    return lig_dir / f"repeat_{r}" / "complex"


first = np.load(leg(repeats[0]) / "equil.npz")
frame_atoms = first["frame_atoms"]
top = top_all.subset(frame_atoms)
lig_res = [
    res
    for res in top.residues
    if res.name in ("LIG", "UNL", "MOL") or (not res.is_protein and not res.is_water and res.n_atoms > 6)
][-1]
lig_idx = np.array([a.index for a in lig_res.atoms])
lig_heavy = np.array([a.index for a in lig_res.atoms if a.element.symbol != "H"])
lig_polar = np.array([a.index for a in lig_res.atoms if a.element.symbol in ("N", "O")])
prot_polar = np.array([a.index for a in top.atoms if a.residue.is_protein and a.element.symbol in ("N", "O")])
ca = np.array([a.index for a in top.atoms if a.name == "CA" and a.residue.is_protein])
print(f"ligand residue {lig_res.name}: {len(lig_idx)} atoms ({len(lig_heavy)} heavy, {len(lig_polar)} N/O); "
      f"{len(ca)} CA atoms; repeats {repeats}")


def traj(frames):
    return md.Trajectory(np.asarray(frames, dtype=np.float32), top)


def name(i):
    a = top.atom(int(i))
    return (
        f"{a.residue.name}{a.residue.resSeq}:{a.name}"
        if a.residue.is_protein
        else f"LIG:{a.name or a.element.symbol + str(a.index - lig_idx[0])}"
    )


# ---- reference: the input pose (first pre-equilibration frame of the first repeat) -----------
ref = traj(first["frames"][:1])
near = md.compute_neighbors(
    ref, 0.5, lig_heavy, haystack_indices=np.array([a.index for a in top.atoms if a.residue.is_protein])
)[0]
pocket_res = sorted({top.atom(int(i)).residue.index for i in near})
print(f"pocket: {len(pocket_res)} residues within 5 A of the ligand: "
      + " ".join(f"{top.residue(i).name}{top.residue(i).resSeq}" for i in pocket_res))

pairs = np.array([(i, j) for i in lig_polar for j in prot_polar])
chi1 = md.compute_chi1(ref)[0]
chi1_res = [top.atom(int(ix[1])).residue.index for ix in chi1]
chi2 = md.compute_chi2(ref)[0]
chi2_res = [top.atom(int(ix[1])).residue.index for ix in chi2]

# ligand torsions: every bond between two heavy atoms that each have another heavy neighbour, not in a ring
bonds = [(a.index, b.index) for a, b in top.bonds if a.residue is lig_res and b.residue is lig_res]
nbrs: dict[int, list[int]] = {}
for a, b in bonds:
    nbrs.setdefault(a, []).append(b)
    nbrs.setdefault(b, []).append(a)
heavy = set(lig_heavy.tolist())
graph = top.to_bondgraph()


def in_ring(a, b):
    g = graph.copy()
    g.remove_edge(top.atom(a), top.atom(b))
    import networkx as nx
    return nx.has_path(g, top.atom(a), top.atom(b))


torsions = []
for a, b in bonds:
    if a in heavy and b in heavy and not in_ring(a, b):
        na = [x for x in nbrs[a] if x != b and x in heavy]
        nb = [x for x in nbrs[b] if x != a and x in heavy]
        if na and nb:
            torsions.append((na[0], a, b, nb[0]))
torsions = np.array(torsions)
print(f"{len(torsions)} rotatable ligand torsions")


def describe(frames, label):
    t = traj(frames)
    t.superpose(ref, atom_indices=ca)
    rmsd = 10 * np.sqrt(((t.xyz[:, lig_heavy] - ref.xyz[0, lig_heavy]) ** 2).sum(-1).mean(-1))
    d = md.compute_distances(t, pairs, periodic=False)
    occ = (d < 0.33).mean(0)
    tor = md.compute_dihedrals(t, torsions, periodic=False) if len(torsions) else np.zeros((t.n_frames, 0))
    c1 = md.compute_dihedrals(t, chi1, periodic=False)
    c2 = md.compute_dihedrals(t, chi2, periodic=False)
    return {"label": label, "n": t.n_frames, "rmsd": rmsd, "occ": occ, "tor": tor, "chi1": c1, "chi2": c2}


def rotamer(x):
    """0: g- (-120..0), 1: trans, 2: g+ (0..120) as fractions."""
    deg = np.degrees(x)
    return np.array([np.mean((deg >= -120) & (deg < 0), axis=0), np.mean((deg < -120) | (deg >= 120), axis=0),
                     np.mean((deg >= 0) & (deg < 120), axis=0)])


data = {}
for r in repeats:
    eq = np.load(leg(r) / "equil.npz")
    fr = stored_frames(leg(r) / "hrex")                # the run may have been stopped while writing
    res = json.loads((leg(r) / "result.json").read_text())
    data[r] = {
        "pre": describe(eq["frames"], "plain MD"),
        "hrex": describe(fr, "window 0"),
        "dg": res["delta_g_decouple"] / KJ,
        "stage": {k: v / KJ for k, v in res["analysis"]["stage_delta_g"].items()},
    }

print("\n== free energies (kcal/mol)")
for r in repeats:
    print(
        f"repeat {r}: complex leg {data[r]['dg']:.2f}; stages "
        + ", ".join(f"{k} {v:.2f}" for k, v in data[r]["stage"].items())
    )

print("\n== ligand RMSD to the input pose after CA superposition (A): mean, max")
for r in repeats:
    print(
        f"repeat {r}: "
        + "; ".join(
            f"{data[r][p]['label']} {data[r][p]['rmsd'].mean():.2f}, {data[r][p]['rmsd'].max():.2f}"
            for p in ("pre", "hrex")
        )
    )

print("\n== ligand-protein polar contacts (N/O within 3.3 A), occupancy; rows where any repeat exceeds 0.2")
keep = sorted({int(i) for r in repeats for p in ("pre", "hrex") for i in np.nonzero(data[r][p]["occ"] > 0.2)[0]})
print(f"{'contact':38s}" + "".join(f" r{r}:MD  r{r}:w0" for r in repeats))
for i in keep:
    a, b = pairs[i]
    print(
        f"{name(a) + ' - ' + name(b):38s}"
        + "".join(f"  {data[r]['pre']['occ'][i]:5.2f}  {data[r]['hrex']['occ'][i]:5.2f}" for r in repeats)
    )

print("\n== ligand torsions: circular mean (deg) in window 0, and fraction of frames within 60 deg of repeat-0 mean")
for k, tq in enumerate(torsions):
    means = [np.degrees(np.angle(np.exp(1j * data[r]["hrex"]["tor"][:, k]).mean())) for r in repeats]
    frac = [
        np.mean(np.abs(np.angle(np.exp(1j * (data[r]["hrex"]["tor"][:, k] - np.radians(means[0]))))) < np.pi / 3)
        for r in repeats
    ]
    flag = "  <-- differs" if max(frac) - min(frac) > 0.3 else ""
    print(f"{'-'.join(name(a).split(':')[1] for a in tq):28s} " + " ".join(f"{m:7.1f}" for m in means)
          + "   " + " ".join(f"{f:4.2f}" for f in frac) + flag)

print(
    "\n== side-chain rotamers of pocket residues in window 0 (g-/t/g+ fractions); rows that differ between repeats by "
    "> 0.3"
)
for label, idx_list, key in (("chi1", chi1_res, "chi1"), ("chi2", chi2_res, "chi2")):
    for col, ri in enumerate(idx_list):
        if ri not in pocket_res:
            continue
        rot = np.array([rotamer(data[r]["hrex"][key][:, col]) for r in repeats])
        if np.max(rot.max(0) - rot.min(0)) > 0.3:
            res_ = top.residue(ri)
            print(
                f"{res_.name}{res_.resSeq} {label}: "
                + " | ".join(f"r{r} " + "/".join(f"{x:.2f}" for x in rot[n]) for n, r in enumerate(repeats))
            )

# ---- water in the pocket, all atoms, per lambda window -----------------------------------------
print("\n== water oxygens within 3.5 A of any ligand heavy atom, per lambda window at the last checkpoint")
water_o = np.array([a.index for a in top_all.atoms if a.residue.is_water and a.element.symbol == "O"])
lig_all = np.array([a.index for a in top_all.atoms if a.index in set(frame_atoms[lig_heavy].tolist())])
import yaml  # noqa: E402

for r in repeats:
    ck = np.load(leg(r) / "hrex" / "checkpoint.npz")
    states = yaml.safe_load((leg(r) / "schedule.yaml").read_text())["states"]
    order = np.argsort(ck["state_of_replica"])
    counts = []
    for rep in order:
        box = ck["boxes"][rep]
        t = md.Trajectory(ck["positions"][rep][None].astype(np.float32), top_all)
        t.unitcell_vectors = box[None].astype(np.float32)
        counts.append(len(md.compute_neighbors(t, 0.35, lig_all, haystack_indices=water_o, periodic=True)[0]))
    lab = [f"{s['lambda_restraints']:.0f}/{s['lambda_electrostatics']:.2f}/{s['lambda_sterics']:.2f}" for s in states]
    print(f"repeat {r} (iteration {int(ck['iteration'])}): " + " ".join(f"{c:2d}" for c in counts))
    if r == repeats[0]:
        print("   windows (restraint/elec/sterics): " + " ".join(lab))
    eq = np.load(leg(r) / "equil.npz")
    t = md.Trajectory(eq["positions"][None].astype(np.float32), top_all)
    t.unitcell_vectors = eq["box"][None].astype(np.float32)
    print(
        f"   end of plain MD: "
        f"{len(md.compute_neighbors(t, 0.35, lig_all, haystack_indices=water_o, periodic=True)[0])} waters"
    )
