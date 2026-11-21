"""Long-range dispersion correction for the ligand-environment Lennard-Jones interaction.

During sampling the soft-core force is truncated at the nonbonded cutoff and carries no
tail correction. An isotropic tail would be wrong inside a protein or a bilayer, so the
missing energy is evaluated explicitly on coupled-state configurations:

    dU_tail(x) = sum over ligand-environment pairs with r_c < r < r_long of the LJ energy
               + a homogeneous estimate beyond r_long (small)

and the free energy of switching it on, ``-kT ln < exp(-dU_tail / kT) >``, is added at
the coupled end state of each leg (the decoupled end state has no ligand-environment
interaction, so nothing is added there).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image displacement for an OpenMM reduced triclinic box (rows a, b, c), as OpenMM
    computes it. It is the nearest image of a displacement up to half the smallest height of the
    cell, which is all that the MD and the tail correction need (a cutoff is never longer); a
    longer displacement in a skewed cell can come out as an image that is not the nearest one
    (:func:`nearest_image`)."""
    delta = delta - np.outer(np.round(delta[:, 2] / box[2, 2]), box[2])
    delta = delta - np.outer(np.round(delta[:, 1] / box[1, 1]), box[1])
    delta = delta - np.outer(np.round(delta[:, 0] / box[0, 0]), box[0])
    return delta


_NEIGHBOUR_CELLS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


def nearest_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """The shortest periodic image of every displacement of ``delta`` (n x 3), whatever its length:
    :func:`minimum_image` and the 26 cells around it are compared. In the rhombic dodecahedron of
    4 nm, (0, 0, 1.5) nm is its own nearest image, and ``minimum_image`` gives one of 3.1 nm
    (review of eb6f89f)."""
    box = np.asarray(box, dtype=float)
    images = minimum_image(np.asarray(delta, dtype=float), box)[:, None, :] + (_NEIGHBOUR_CELLS @ box)[None]
    return images[np.arange(images.shape[0]), np.argmin((images * images).sum(axis=2), axis=1)]


@dataclass
class TailEnergy:
    ligand: np.ndarray           # atom indices
    lig_sigma: np.ndarray        # nm
    lig_epsilon: np.ndarray      # kJ/mol
    env: np.ndarray              # environment atoms with epsilon > 0
    env_sigma: np.ndarray
    env_epsilon: np.ndarray
    cutoff_nm: float = 0.9
    long_cutoff_nm: float = 2.0

    def __post_init__(self) -> None:
        self.ligand = np.asarray(self.ligand, dtype=int)
        self.env = np.asarray(self.env, dtype=int)
        for name in ("lig_sigma", "lig_epsilon", "env_sigma", "env_epsilon"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        sig = 0.5 * (self.lig_sigma[:, None] + self.env_sigma[None, :])
        eps = np.sqrt(self.lig_epsilon[:, None] * self.env_epsilon[None, :])
        self._s6 = float(np.sum(eps * sig ** 6))
        self._s12 = float(np.sum(eps * sig ** 12))

    @classmethod
    def from_system(cls, system, ligand_atoms, ligand_parameters, cutoff_nm: float,
                    long_cutoff_nm: float = 2.0) -> "TailEnergy":
        """Build from an (alchemical or plain) System; ``ligand_parameters`` are the original
        (charge, sigma, epsilon) of the ligand atoms."""
        import openmm as mm
        import openmm.unit as unit

        nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
        # the tail begins where the MD stops: the real-space cutoff of the System itself
        md_cutoff = nb.getCutoffDistance().value_in_unit(unit.nanometer)
        if abs(md_cutoff - float(cutoff_nm)) > 1e-9:
            raise ValueError(f"the tail correction is set up from {float(cutoff_nm):g} nm but the System cuts the "
                             f"nonbonded interactions at {md_cutoff:g} nm")
        if not float(long_cutoff_nm) > md_cutoff:
            raise ValueError(f"the outer radius of the tail correction ({float(long_cutoff_nm):g} nm) must be larger "
                             f"than the real-space cutoff ({md_cutoff:g} nm), where the tail begins")
        ligand = list(ligand_atoms)
        lig_set = set(ligand)
        env, es, ee = [], [], []
        for i in range(system.getNumParticles()):
            if i in lig_set:
                continue
            _q, sig, eps = nb.getParticleParameters(i)
            e = eps.value_in_unit(unit.kilojoule_per_mole)
            if e > 0.0:
                env.append(i)
                es.append(sig.value_in_unit(unit.nanometer))
                ee.append(e)
        return cls(np.array(ligand), np.array([p[1] for p in ligand_parameters]),
                   np.array([p[2] for p in ligand_parameters]), np.array(env), np.array(es), np.array(ee),
                   cutoff_nm, long_cutoff_nm)

    def outer_radius(self, box: np.ndarray) -> float:
        """Where the explicit sum stops and the homogeneous tail begins, nm: ``long_cutoff_nm``,
        but within the cell (0.49 of its shortest height) so that no pair is counted twice, and
        never inside the real-space cutoff, whose interactions the MD already has. A cell too small
        for any explicit shell gives the homogeneous tail from the cutoff (OpenMM's correction)."""
        box = np.asarray(box, dtype=float)
        return max(float(self.cutoff_nm), min(float(self.long_cutoff_nm), 0.49 * min(box[0, 0], box[1, 1], box[2, 2])))

    def energy(self, positions: np.ndarray, box: np.ndarray) -> float:
        """dU_tail in kJ/mol for one configuration (positions in nm, box rows a, b, c)."""
        box = np.asarray(box, dtype=float)
        r_long = self.outer_radius(box)
        rc2, rl2 = self.cutoff_nm ** 2, r_long ** 2
        env_pos = positions[self.env]
        total = 0.0
        for k, i in enumerate(self.ligand):
            if self.lig_epsilon[k] <= 0.0:
                continue
            d = minimum_image(env_pos - positions[i], box)
            r2 = np.einsum("ij,ij->i", d, d)
            m = (r2 > rc2) & (r2 < rl2)
            if not m.any():
                continue
            sig = 0.5 * (self.lig_sigma[k] + self.env_sigma[m])
            eps = np.sqrt(self.lig_epsilon[k] * self.env_epsilon[m])
            x6 = (sig * sig / r2[m]) ** 3
            total += float(np.sum(4.0 * eps * (x6 * x6 - x6)))
        volume = float(abs(np.linalg.det(box)))
        total += (16.0 * np.pi / volume) * (self._s12 / (9.0 * r_long ** 9) - self._s6 / (3.0 * r_long ** 3))
        return total

    def summary(self) -> dict:
        """What identifies this tail correction, small enough to be stored with the samples: the two
        radii, the numbers of atoms, the two sums of the dispersion terms and a checksum of the
        arrays. (All parameters of all environment atoms are megabytes for a protein system.)"""
        import hashlib

        digest = hashlib.sha256()
        for array in (self.ligand, self.lig_sigma, self.lig_epsilon, self.env, self.env_sigma, self.env_epsilon):
            digest.update(np.ascontiguousarray(array).tobytes())
        return {"cutoff_nm": float(self.cutoff_nm), "long_cutoff_nm": float(self.long_cutoff_nm),
                "n_ligand": int(self.ligand.size), "n_environment": int(self.env.size),
                "s6": self._s6, "s12": self._s12, "sha256": digest.hexdigest()}

    def to_dict(self) -> dict:
        return {"ligand": self.ligand.tolist(), "lig_sigma": self.lig_sigma.tolist(),
                "lig_epsilon": self.lig_epsilon.tolist(), "env": self.env.tolist(),
                "env_sigma": self.env_sigma.tolist(), "env_epsilon": self.env_epsilon.tolist(),
                "cutoff_nm": self.cutoff_nm, "long_cutoff_nm": self.long_cutoff_nm}

    @classmethod
    def from_dict(cls, d: dict) -> "TailEnergy":
        return cls(**d)


def tail_free_energy(du_kj: np.ndarray, temperature_K: float) -> tuple[float, float]:
    """``-kT ln <exp(-dU/kT)>`` and its standard error for independent observations (kJ/mol).

    Raises ValueError without observations: no observation is no estimate, not a correction of 0.
    """
    kt = 8.31446261815324e-3 * temperature_K
    x = -np.asarray(du_kj, dtype=float) / kt
    if x.size == 0:
        raise ValueError("the tail correction needs at least one recorded tail energy")
    shift = x.max()
    w = np.exp(x - shift)
    dg = -kt * (shift + np.log(w.mean()))
    se = kt * w.std(ddof=1) / (w.mean() * np.sqrt(len(w))) if len(w) > 1 else 0.0
    return float(dg), float(se)
