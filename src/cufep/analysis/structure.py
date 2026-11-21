"""Structural numbers of the coupled-state configurations saved during replica exchange.

Ligand RMSD and centre-of-mass drift relative to the first saved frame, after superposing
on the protein's C-alpha atoms, as numbers without a verdict: earlier versions warned above
3 A and 5 A, levels of cuFEP's own (removed by the owner's order of 2026-11-20).
"""
from __future__ import annotations

import numpy as np

from cufep.alchemy.restraints import kabsch_align, ligand_next_to_protein


def structural_analysis(samples: dict, n_equilibration: int = 0) -> dict | None:
    """Return RMSD statistics, or None if the leg has no protein frames."""
    attrs = samples.get("attrs", {})
    if "frames" not in samples or "frame_ligand" not in attrs or "frame_ca" not in attrs:
        return None
    lig, ca = np.asarray(attrs["frame_ligand"], dtype=int), np.asarray(attrs["frame_ca"], dtype=int)
    keep = samples["frame_iter"] >= n_equilibration
    frames = np.asarray(samples["frames"][keep], dtype=float)
    if len(frames) < 2 or len(ca) < 4 or len(lig) < 1:
        return None
    # the frames come from whichever replica was in the coupled state, and coordinates are not
    # wrapped: a ligand one cell away from the protein is the same pose. Each frame's ligand is
    # put into the periodic image nearest to the protein first (the rule of the restraint search)
    boxes = samples.get("frame_box")
    images_checked = boxes is not None and len(boxes) == len(samples["frame_iter"])
    if images_checked:
        frames = ligand_next_to_protein(frames, lig, ca, np.asarray(boxes, dtype=float)[keep])
    aligned = kabsch_align(frames, frames[0], ca)
    d_lig = aligned[:, lig, :] - aligned[0, lig, :]
    lig_rmsd = 10.0 * np.sqrt((d_lig ** 2).sum(axis=2).mean(axis=1))
    com = 10.0 * np.linalg.norm(aligned[:, lig, :].mean(axis=1) - aligned[0, lig, :].mean(axis=0), axis=1)
    d_ca = aligned[:, ca, :] - aligned[0, ca, :]
    ca_rmsd = 10.0 * np.sqrt((d_ca ** 2).sum(axis=2).mean(axis=1))
    half = len(frames) // 2
    out = {"n_frames": int(len(frames)), "ligand_rmsd_mean_A": float(lig_rmsd[1:].mean()),
           "ligand_rmsd_last_half_A": float(lig_rmsd[half:].mean()), "ligand_rmsd_max_A": float(lig_rmsd.max()),
           "ligand_com_drift_last_half_A": float(com[half:].mean()), "ligand_com_drift_max_A": float(com.max()),
           "protein_ca_rmsd_last_half_A": float(ca_rmsd[half:].mean()),
           "periodic_images_checked": bool(images_checked)}
    return out
