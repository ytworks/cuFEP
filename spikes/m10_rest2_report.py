"""Does REST2 make the slow degrees of freedom of TYK2 ejm_31 move? One line per run.

Solvent leg: the torsion C5-C3-N12-C4 of the ligand (acetamide group against the pyridine
ring), which without REST2 turns over once in a few nanoseconds. Complex leg: chi1 of ILE960
and VAL981. Both are read from the frames that the coupled window (the physical state) wrote
during replica exchange: the fraction of frames in the state of the bound pose, and how often
the state changed. Next to them the lambda windows, the lowest overlap and exchange between
neighbouring windows, and the free energies.

    spikes/run.sh spikes/m10_rest2_report.py runs/val_tyk2_adaptive runs/rest2_L2 ... [--timing]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
from stored_frames import stored_frames

KJ = 4.184
LIGAND_TORSION = (5, 3, 12, 4)          # atom numbers within ejm_31


def dihedral(x, i, j, k, l):
    b0, b1, b2 = x[..., i, :] - x[..., j, :], x[..., k, :] - x[..., j, :], x[..., l, :] - x[..., k, :]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def two_state(angle_deg, centre_a: float, centre_b: float, half_width: float = 50.0):
    """Fraction of frames in state a and number of changes between a and b (frames in between keep the last state)."""
    def near(c):
        return np.abs((angle_deg - c + 180.0) % 360.0 - 180.0) < half_width
    a, b = near(centre_a), near(centre_b)
    state, changes, in_a, counted = None, 0, 0, 0
    for fa, fb in zip(a, b):
        new = "a" if fa else "b" if fb else state
        if new is not None and state is not None and new != state:
            changes += 1
        state = new
        if state is not None:
            counted += 1
            in_a += state == "a"
    return (in_a / counted if counted else float("nan")), changes


def frames_of(leg_dir: Path):
    return stored_frames(leg_dir / "hrex", step=50)


def main() -> None:
    from cufep.workflow import load_prepared

    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    timing = "--timing" in sys.argv
    for run in map(Path, runs):
        lig_dir = run / "ejm_31"
        top_all = md.load(str(lig_dir / "prep/complex.pdb")).topology
        for rep in sorted(lig_dir.glob("repeat_*")):
            out = [f"{run.name} {rep.name}:"]
            res = {}
            for leg in ("complex", "solvent"):
                f = rep / leg / "result.json"
                if f.exists():
                    res[leg] = json.loads(f.read_text())
            if len(res) < 2:
                print(out[0], "incomplete")
                continue
            c, s = res["complex"], res["solvent"]
            bind = (s["delta_g_decouple"] - c["delta_g_decouple"] - c["restraint_release"]) / KJ
            out.append(
                f"dG_bind {bind:6.2f} (complex {c['delta_g_decouple'] / KJ:.2f}, solvent "
                f"{s['delta_g_decouple'] / KJ:.2f})"
            )
            for leg, r in res.items():
                a = r["analysis"]
                rest = r.get("rest2", {})
                out.append(
                    f"\n    {leg:7s} {r['n_states']:2d} windows ("
                    + "/".join(str(len(v)) for v in r["lambda_windows"].values())
                    + f"), REST2 scale {rest.get('scale', 1.0):g}"
                    + (f" with {len(rest.get('hot_residues', []))} residues" if rest.get("hot_residues") else "")
                    + f"; lowest overlap {a['min_overlap']:.3f}, lowest exchange "
                      f"{min(x for x in (a.get('swap_acceptance_neighbour')
                                         or a.get('neighbour_transition_fraction') or [float('nan')])
                             if x is not None):.3f}, "
                    f"round trips {a['round_trips']}, forward/reverse gap {a['forward_reverse_gap'] / KJ:.2f}"
                )
                eq = np.load(rep / leg / "equil.npz")
                frames = frames_of(rep / leg)
                prep = load_prepared(lig_dir / "prep", leg)
                atoms = list(eq["frame_atoms"])
                local = np.array([atoms.index(i) for i in prep["ligand"]])
                angle = dihedral(frames[:, local, :], *LIGAND_TORSION)
                frac, changes = two_state(angle, 0.0, 180.0)
                out.append(
                    f"\n            ligand torsion: as bound in {100 * frac:3.0f}% of {len(frames)} frames, {changes} "
                    f"changes"
                )
                if leg == "complex":
                    traj = md.Trajectory(frames, top_all.subset(eq["frame_atoms"]))
                    idx, val = md.compute_chi1(traj)
                    for resseq, name in ((960, "ILE960"), (981, "VAL981")):
                        col = next(
                            k for k, ix in enumerate(idx) if traj.topology.atom(int(ix[1])).residue.resSeq == resseq
                        )
                        frac, changes = two_state(np.degrees(val[:, col]), 180.0, -60.0)
                        out.append(f"; {name} chi1 trans in {100 * frac:3.0f}%, {changes} changes")
                if timing and (rep / leg / "timing.json").exists():
                    t = json.loads((rep / leg / "timing.json").read_text())
                    out.append("\n            timing: " + ", ".join(f"{k} {v:.0f} s" for k, v in t["phases"].items())
                               + f"; {t['sampling']['seconds_per_iteration']:.3f} s per iteration")
            print(" ".join(out[:2]) + "".join(out[2:]))


if __name__ == "__main__":
    main()
