"""Structural checks on the coupled-state configurations saved during replica exchange.

Ligand RMSD and centre-of-mass drift relative to the first saved frame, after superposing
on the protein's C-alpha atoms. A ligand that leaves the pocket or changes its pose makes
an absolute binding free energy meaningless without anyone noticing from the energies.
"""
from __future__ import annotations

import numpy as np

from cufep.alchemy.restraints import kabsch_align

LIGAND_RMSD_WARN_A = 3.0
COM_DRIFT_WARN_A = 5.0


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
           "protein_ca_rmsd_last_half_A": float(ca_rmsd[half:].mean()), "warnings": []}
    if out["ligand_rmsd_last_half_A"] > LIGAND_RMSD_WARN_A:
        out["warnings"].append(f"the ligand moved {out['ligand_rmsd_last_half_A']:.1f} A (heavy-atom RMSD) from its "
                               "starting pose in the coupled state; the pose may have changed")
    if out["ligand_com_drift_last_half_A"] > COM_DRIFT_WARN_A:
        out["warnings"].append(f"the ligand's centre of mass drifted {out['ligand_com_drift_last_half_A']:.1f} A in "
                               "the coupled state; it may have left the pocket")
    return out
