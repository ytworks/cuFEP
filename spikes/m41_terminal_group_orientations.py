"""The orientation of the terminal groups that terminal flip moves would turn, in the frames a relative run stored
(the replica at lambda 0 of each leg, one frame per stored iteration): how often each of the two orientations is
seen. No experimental value is read, nothing is run, and the directory is only read.

For every group (:func:`cufep.rbfe.leg.hybrid_flips`) the dihedral X-anchor-pivot-N is taken, X the lowest heavy
neighbour of the anchor other than the pivot, N the lowest ring neighbour of the pivot; a half turn of the group
changes it by 180 degrees. Reported per leg: the dihedral of the first frame, the fraction of the frames within 90
degrees of it and beyond (the other orientation), and how often consecutive frames differ in it.

    scripts/in_docker.sh python spikes/m41_terminal_group_orientations.py runs/rbfe_bench_a2a_957a9cf [OUT.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def dihedral(p: np.ndarray) -> np.ndarray:
    """Dihedral (degrees) of four points per frame: ``p`` (frames, 4, 3)."""
    b0, b1, b2 = p[:, 0] - p[:, 1], p[:, 2] - p[:, 1], p[:, 3] - p[:, 2]
    b1 = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v, w = b0 - (b0 * b1).sum(1)[:, None] * b1, b2 - (b2 * b1).sum(1)[:, None] * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(1), (v * w).sum(1)))


def main(run_dir: str, out: str | None = None) -> None:
    from stored_frames import stored_frames

    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import build_leg, hybrid_flips

    run = Path(run_dir)
    rcfg = RbfeConfig.model_validate(yaml.safe_load((run / "resolved_config.yaml").read_text()))
    rows = []
    for edge_dir in sorted(path for path in (run / "edges").iterdir() if (path / "prep" / "prep.json").exists()):
        for leg in ("complex", "solvent"):
            hrex = edge_dir / "repeat_0" / leg / "hrex"
            if not hrex.exists():
                continue
            built = build_leg(rcfg, edge_dir, leg)
            groups, _record = hybrid_flips(edge_dir, built, leg)
            if not groups:
                continue
            hybrid = built.hybrid
            frame_atoms = built.prep["solute"] if leg == "complex" else list(hybrid.ligand_atoms)
            where = {a: i for i, a in enumerate(frame_atoms)}
            frames = stored_frames(hrex, step=200)
            atoms = list(built.prep["topology"].atoms())
            neighbours: dict[int, set[int]] = {}
            for i, j in hybrid.bonds:
                neighbours.setdefault(i, set()).add(j)
                neighbours.setdefault(j, set()).add(i)

            def heavy(i: int, atoms=atoms) -> bool:
                return atoms[i].element is not None and atoms[i].element.symbol != "H"

            for g in groups:
                # atoms of ligand A (the frames are of lambda 0)
                in_a = set(hybrid.atoms_a)
                x = min(i for i in neighbours[g.anchor] if i != g.pivot and i in in_a and heavy(i))
                n = min(i for i in neighbours[g.pivot] if i in g.mobile and i in in_a and heavy(i))
                phi = dihedral(frames[:, [where[x], where[g.anchor], where[g.pivot], where[n]]].astype(float))
                turned = np.abs((phi - phi[0] + 180.0) % 360.0 - 180.0) > 90.0
                row = {"edge": edge_dir.name, "leg": leg, "anchor": g.anchor, "pivot": g.pivot,
                       "mobile_atoms": len(g.mobile), "frames": int(len(phi)), "first_dihedral": float(phi[0]),
                       "fraction_other_orientation": float(turned.mean()),
                       "changes_between_frames": int((turned[1:] != turned[:-1]).sum())}
                rows.append(row)
                print(f"{row['edge']} {leg}: {row['frames']} frames, dihedral of the first "
                      f"{row['first_dihedral']:.0f}, "
                      f"other orientation in {100 * row['fraction_other_orientation']:.1f} %, "
                      f"{row['changes_between_frames']} changes between frames", flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
