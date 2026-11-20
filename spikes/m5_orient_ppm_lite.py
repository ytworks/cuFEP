"""M5 spike: orientation by minimising an implicit water->membrane transfer energy
(a simplified PPM-like model) and comparison with OPM for several structures."""
from __future__ import annotations

import glob
import io
import sys
import time

import mdtraj as md
import numpy as np
import openmm.app as app
import openmm.unit as u

from cufep.prep.orient import _sphere_points
from cufep.prep.protein import AMINO_ACIDS

# atomic solvation parameters, kcal/mol/A^2 (water -> hydrocarbon core; signs as in PPM-type models)
SIGMA = {"C": -0.022, "S": -0.010, "N": 0.053, "O": 0.057}
CHARGED = {("LYS", "NZ"): 0.25, ("ARG", "NH1"): 0.25, ("ARG", "NH2"): 0.25, ("ARG", "NE"): 0.15,
           ("ASP", "OD1"): 0.25, ("ASP", "OD2"): 0.25, ("GLU", "OE1"): 0.25, ("GLU", "OE2"): 0.25}
LAMBDA = 0.09          # nm, width of the interface


def atom_table(topology, pos):
    atoms = [
        a
        for a in topology.atoms()
        if a.residue.name in AMINO_ACIDS and a.element is not None and a.element.symbol != "H"
    ]
    idx = np.array([a.index for a in atoms])
    sub = md.Topology.from_openmm(topology).subset(idx)
    asa = md.shrake_rupley(md.Trajectory(pos[idx][None], sub), mode="atom")[0] * 100.0     # A^2
    sig = np.array([CHARGED.get((a.residue.name, a.name), SIGMA.get(a.element.symbol, 0.0)) for a in atoms])
    w = asa * sig
    keep = np.abs(w) > 1e-6
    return pos[idx][keep], w[keep]


def best_slab(xyz, w, normals, half_grid=(1.0, 1.2, 1.4, 1.6, 1.8), step=0.1):
    centre0 = xyz.mean(axis=0)
    x = xyz - centre0
    best = (1e9, None, 0.0, 0.0)
    for n in normals:
        z = x @ n
        cs = np.arange(z.min() + 0.5, z.max() - 0.5 + 1e-9, step)
        dz = np.abs(z[None, :] - cs[:, None])                      # (centres, atoms)
        for h in half_grid:
            e = (w[None, :] / (1.0 + np.exp((dz - h) / LAMBDA))).sum(axis=1)
            k = int(np.argmin(e))
            if e[k] < best[0]:
                best = (float(e[k]), n, float(cs[k]), h)
    e, n, c, h = best
    return e, n, centre0 + c * n, 2 * h


def refine(xyz, w, normal, n_points=400, cone_deg=12.0):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n_points, 3)) * np.tan(np.radians(cone_deg)) / 2.0
    cand = normal[None, :] + v - (v @ normal)[:, None] * normal[None, :]
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    return best_slab(xyz, w, np.vstack([normal[None, :], cand]), half_grid=np.arange(1.0, 1.85, 0.1), step=0.05)


def orient(topology, pos):
    xyz, w = atom_table(topology, pos)
    e, n, c, d = best_slab(xyz, w, _sphere_points(900))
    e, n, c, d = refine(xyz, w, n)
    return e, n, c, d


def main():
    for path in sorted(glob.glob("spikes/out/opm/*.pdb")):
        lines = [l for l in open(path) if l.startswith("ATOM")]
        half = [float(l.split(":")[1]) for l in open(path) if "bilayer thickness" in l]
        pdb = app.PDBFile(io.StringIO("".join(lines) + "END\n"))
        pos = np.array(pdb.positions.value_in_unit(u.nanometer))
        t0 = time.perf_counter()
        e, n, c, d = orient(pdb.topology, pos)
        tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
        print(
            f"{path.split('/')[-1]:10s} E={e:8.1f} kcal/mol  tilt vs OPM {tilt:5.1f} deg  centre z {10 * c[2]:+5.1f} A "
            f" "
            f"thickness {10 * d:.0f} A (OPM {2 * half[0]:.0f})  [{time.perf_counter() - t0:.0f} s]"
        )
    from cufep.prep.protein import prepare_protein
    for path in ("benchmarks/tyk2/inputs/tyk2_protein.pdb", "benchmarks/a2a/inputs/a2a_hip278_protein.pdb"):
        p = prepare_protein(path)
        e, n, c, d = orient(p.topology, p.positions_nm())
        print(
            f"{path.split('/')[-1]:28s} E={e:8.1f} kcal/mol normal {np.round(n, 3)} centre {np.round(c, 3)} thickness "
            f"{10 * d:.0f} A"
        )


if __name__ == "__main__":
    sys.exit(main())
