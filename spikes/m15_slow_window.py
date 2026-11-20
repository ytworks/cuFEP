"""Which motion is slow in the coupled window (restrain stage, window 0) of a complex leg?

The window with the fewest independent samples of the A2A 4e complex leg at 5 ns was window 0 of the
restrain stage (the bound ligand without restraint): its energy difference to window 1 was correlated over
161 iterations. That difference is what the restraint at half strength costs in the configurations of
window 0 (plus what REST2 changes between the two windows, if anything). This script reads, from the
samples and the frames that window 0 wrote during the production of one leg:

* the energy difference u1 - u0 in window 0 (kT): its statistical inefficiency g (iterations), its mean
  over five time blocks;
* the six Boresch coordinates and the energy of each term at full strength (kT): g, the five block means,
  and the correlation with u1 - u0;
* the heavy-atom RMSD of the ligand from its input pose, in the frame of the binding site, with the
  symmetries of the molecule (the measure of the restraint search);
* chi1 of every residue with a heavy atom within 0.5 nm of the ligand in the input: the rotamer of each
  frame (g+, t, g-), the number of changes and the fraction of frames in the most populated one;
* every rotatable torsion of the ligand: the fraction of frames within 60 degrees of the input value and
  the number of changes between near and away.

Nothing is written. Read-only on the run directories.

    spikes/run.sh spikes/m15_slow_window.py runs/a2a_t010_p5:4e runs/a2a_r15b_example:4e ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from stored_frames import stored_frames

from cufep.alchemy.restraints import R_GAS


def g_of(x: np.ndarray) -> float:
    from pymbar import timeseries

    x = np.asarray(x, dtype=float)
    if len(x) < 10 or np.std(x) == 0:
        return float("nan")
    return float(timeseries.statistical_inefficiency(x))


def blocks(x: np.ndarray, n: int = 5) -> list[float]:
    return [round(float(np.mean(b)), 2) for b in np.array_split(np.asarray(x, dtype=float), n)]


def corr(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dihedral(x, i, j, k, l):
    b0, b1, b2 = x[..., i, :] - x[..., j, :], x[..., k, :] - x[..., j, :], x[..., l, :] - x[..., k, :]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def changes(labels) -> int:
    labels = [x for x in labels if x is not None]
    return int(sum(a != b for a, b in zip(labels, labels[1:])))


def rotamer(angle_deg: np.ndarray) -> list[str]:
    out = []
    for a in angle_deg:
        a = (a + 360.0) % 360.0
        out.append("g+" if a < 120.0 else "t" if a < 240.0 else "g-")
    return out


def main() -> None:
    from openff.toolkit import Molecule
    from rdkit import Chem

    from cufep.alchemy.restraints import (
        BoreschRestraint,
        heavy_atom_symmetries,
        kabsch_align,
        nearest_image_of_ligand,
    )
    from cufep.sampling.storage import equilibration_boundary, read_samples
    from cufep.workflow import load_prepared

    for spec in sys.argv[1:]:
        run, name = spec.split(":")
        lig_dir = Path(run) / name
        leg = lig_dir / "repeat_0" / "complex"
        samples = read_samples(leg / "hrex")
        attrs = samples["attrs"]
        n_eq = equilibration_boundary(attrs, leg)
        u, states = np.asarray(samples["u"]), np.asarray(samples["states"])
        # the length of an iteration and the temperature of this run, as its samples record them (review of
        # 7b49066: fixed at 2.5 ps and 298.15 K, a run of 5 ps at 330 K was read wrong)
        iteration_ps, kt = float(attrs["iteration_ps"]), R_GAS * float(attrs["temperature_K"])   # kJ/mol
        rows = np.arange(n_eq, u.shape[0])
        rep0 = np.array([int(np.nonzero(states[n] == 0)[0][0]) for n in rows])
        du = u[rows, rep0, 1] - u[rows, rep0, 0]
        prod_ns = len(rows) * iteration_ps / 1000.0
        print(f"\n=== {run} ({name}): production {len(rows)} iterations ({prod_ns:g} ns), equilibration {n_eq}")
        print(f"u1 - u0 in window 0 (kT): mean {du.mean():.2f}, sd {du.std():.2f}, g {g_of(du):.0f} iterations, "
              f"block means {blocks(du)}")

        prep = load_prepared(lig_dir / "prep", "complex")
        top = prep["topology"]
        atoms = list(top.atoms())
        solute = list(prep["solute"])
        where = {a: i for i, a in enumerate(solute)}
        frames_all = stored_frames(leg / "hrex", step=50).astype(float)
        frame_iter = np.asarray(samples["frame_iter"], dtype=int)[:len(frames_all)]
        boxes = np.asarray(samples["frame_box"], dtype=float)[:len(frames_all)]
        keep = frame_iter >= n_eq
        frames, frame_iter, boxes = frames_all[keep], frame_iter[keep], boxes[keep]
        du_at = u[frame_iter, [int(np.nonzero(states[n] == 0)[0][0]) for n in frame_iter], 1] - \
            u[frame_iter, [int(np.nonzero(states[n] == 0)[0][0]) for n in frame_iter], 0]
        lig_sys = [int(i) for i in prep["ligand"]]
        lig = np.array([where[i] for i in lig_sys])
        print(
            f"frames of window 0 in the production: {len(frames)} (every {int(np.median(np.diff(frame_iter)))} "
            f"iterations)"
        )

        # ---- Boresch coordinates ---------------------------------------------------------------
        stored = json.loads((leg / "restraint.json").read_text())["restraint"]
        host = tuple(where[int(i)] for i in stored["host_atoms"])
        guest = tuple(where[int(i)] for i in stored["guest_atoms"])
        frames = nearest_image_of_ligand(frames, lig, frames[:, list(host), :].mean(axis=1), boxes)
        r = BoreschRestraint(host, guest, *(stored[k] for k in ("r0", "theta_a0", "theta_b0", "phi_a0", "phi_b0",
                                                                    "phi_c0", "k_r", "k_theta_a", "k_theta_b",
                                                                    "k_phi_a", "k_phi_b", "k_phi_c")))
        values = r.values(frames)
        ref = np.array([r.r0, r.theta_a0, r.theta_b0, r.phi_a0, r.phi_b0, r.phi_c0])
        ks = np.array([r.k_r, r.k_theta_a, r.k_theta_b, r.k_phi_a, r.k_phi_b, r.k_phi_c])
        dev = values - ref
        dev[:, 3:] = (dev[:, 3:] + np.pi) % (2 * np.pi) - np.pi
        terms = 0.5 * ks * dev ** 2 / kt
        total = terms.sum(axis=1)
        print(f"restraint energy at full strength (kT): mean {total.mean():.2f}, g {g_of(total):.0f}; correlation with "
              f"u1 - u0: {corr(total, du_at):.2f}; mean of u1 - u0 over these frames {du_at.mean():.2f} "
              f"(half the restraint energy: {0.5 * total.mean():.2f})")
        names = ["r", "theta_A", "theta_B", "phi_A", "phi_B", "phi_C"]
        for k, n in enumerate(names):
            shown = values[:, k] * (10.0 if k == 0 else 180.0 / np.pi)
            unit = "A" if k == 0 else "deg"
            print(
                f"  {n:7s} ref {ref[k] * (10.0 if k == 0 else 180.0 / np.pi):8.2f} {unit}; mean {shown.mean():8.2f}, "
                f"sd "
                f"{shown.std():6.2f}; energy {terms[:, k].mean():5.2f} kT (share of the variance of the total "
                f"{np.cov(terms[:, k], total)[0, 1] / max(total.var(), 1e-12):5.2f}); g {g_of(terms[:, k]):5.0f}; "
                f"blocks {blocks(shown)}"
            )

        # ---- ligand pose -------------------------------------------------------------------------
        rdmol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
        heavy_local = [i for i, a in enumerate(rdmol.GetAtoms()) if a.GetAtomicNum() > 1]
        backbone = np.array([where[a.index] for a in atoms if a.index in where and a.name in ("N", "CA", "C")
                             and a.residue.name not in ("LIG",)])
        centroid = frames[0, lig[heavy_local]].mean(axis=0)
        site = backbone[np.linalg.norm(frames[0, backbone] - centroid, axis=1) < 2.0]
        aligned = kabsch_align(frames, frames[0], site)
        given = np.asarray(prep["positions"], dtype=float)[solute]
        given = nearest_image_of_ligand(given[None], lig, given[None, list(host), :].mean(axis=1),
                                        np.asarray(prep["box"], dtype=float)[None])[0]
        reference = kabsch_align(given[None], aligned.mean(axis=0), site)[0][lig[heavy_local]]
        rmsd = None
        for perm in heavy_atom_symmetries(rdmol):
            d = np.sqrt(((aligned[:, lig[heavy_local][perm], :] - reference) ** 2).sum(axis=2).mean(axis=1))
            rmsd = d if rmsd is None else np.minimum(rmsd, d)
        print(
            f"ligand heavy-atom RMSD from the input pose (A): mean {10 * rmsd.mean():.2f}, max {10 * rmsd.max():.2f}, "
            f"g {g_of(rmsd):.0f}, blocks {[round(10 * b, 2) for b in blocks(rmsd)]}, correlation with u1 - u0 "
            f"{corr(rmsd, du_at):.2f}"
        )

        # ---- ligand torsions -----------------------------------------------------------------------
        patt = Chem.MolFromSmarts("[!$([NH])&!$([NH2])&!$(C(F)(F)F)&!D1&!$(*#*)]-&!@[!$([NH])&!$([NH2])&!D1&!$(*#*)]")
        for b, c in rdmol.GetSubstructMatches(patt):
            ends = []
            for x, y in ((b, c), (c, b)):
                nbrs = sorted((n for n in rdmol.GetAtomWithIdx(x).GetNeighbors() if n.GetIdx() != y),
                              key=lambda n: -n.GetAtomicNum())      # a heavy neighbour if there is one
                ends.append(nbrs[0].GetIdx() if nbrs else None)
            a_, d_ = ends
            if a_ is None or d_ is None:
                continue
            ang = dihedral(aligned[:, lig, :], a_, b, c, d_)
            ang0 = dihedral(given[lig][None], a_, b, c, d_)[0]
            near = np.abs((ang - ang0 + 180.0) % 360.0 - 180.0) < 60.0
            print(
                f"  ligand torsion {a_}-{b}-{c}-{d_}: input {ang0:7.1f} deg; near it in {100 * near.mean():5.1f} % of "
                f"the frames, {changes(list(near))} changes; correlation of near/away with u1 - u0 "
                f"{corr(near.astype(float), du_at):.2f}"
            )

        # ---- pocket side chains --------------------------------------------------------------------
        lig_xyz = given[lig[heavy_local]]
        gamma = ("CG", "CG1", "OG", "OG1", "SG")
        report = []
        for res in top.residues():
            names_ = {a.name: a.index for a in res.atoms()}
            if not all(n in names_ for n in ("N", "CA", "CB")) or not any(g in names_ for g in gamma):
                continue
            heavy_res = [
                a.index for a in res.atoms() if a.element is not None and a.element.symbol != "H" and a.index in where
            ]
            if not heavy_res:
                continue
            dmin = np.min(np.linalg.norm(given[[where[i] for i in heavy_res]][:, None] - lig_xyz[None], axis=2))
            if dmin > 0.5:
                continue
            gname = next(g for g in gamma if g in names_)
            idx = [where[names_[n]] for n in ("N", "CA", "CB")] + [where[names_[gname]]]
            ang = dihedral(frames, *idx)
            rot = rotamer(ang)
            values_, counts = np.unique(rot, return_counts=True)
            top_rot = values_[np.argmax(counts)]
            report.append(
                (
                    changes(rot),
                    f"  {res.name}{res.id}/{res.chain.id} chi1: {dict(zip(values_, counts))}, "
                    f"{changes(rot)} changes, most populated {top_rot} "
                    f"{100 * counts.max() / len(rot):.0f} %, input {rotamer(dihedral(given[None], *idx))[0]}; "
                    f"correlation of 'in the most populated' with u1 - u0 "
                    f"{corr(np.array([x == top_rot for x in rot], dtype=float), du_at):.2f}",
                )
            )
        print(f"pocket residues with chi1 within 0.5 nm of the ligand: {len(report)}")
        for _n, line in sorted(report, key=lambda t: t[0]):
            print(line)


if __name__ == "__main__":
    main()
