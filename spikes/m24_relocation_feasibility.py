"""Would water relocation moves be accepted, and with how many trial poses? (plan
docs/plans/2026-11-22-accuracy-causes-and-fixes.md, step 4d, before RelocationMC is implemented).
Reads stored configurations of a complex leg of a bilayer system (a rectangular cell); CPU only;
writes only under spikes/out/m24/. Numbers only.

    scripts/in_docker.sh python spikes/m24_relocation_feasibility.py RUN LIGAND [--window last|INDEX]
            [--from start|checkpoint] [--moves 200] [--trials 16,128,1024,4096] [--seed 1]
            [--choice uniform|biased]

The move is the one of Ding et al. (arXiv:2603.22274 v3, RelocationMC) in the form a program without
their kernel can run: a water molecule is taken out of one of two regions, the sphere around the
ligand (radius: half the largest distance between two ligand heavy atoms in the prepared pose plus
0.3 nm, as in the paper) or the rest of the cell, and K rigid trial poses are drawn in the other
region, uniform in position and orientation; one is chosen with its Boltzmann weight, and the
multiple-try rule of Liu, Liang and Wong (2000) with K - 1 reference poses in the region the water
left decides. With N_S and N_T water molecules in the two regions before the move and volumes
V_S, V_T the acceptance is min(1, r),

    r = N_S / (N_T + 1) * V_T sum_j exp(-u(y_j) / kT) / (V_S sum_j exp(-u(x_j) / kT)),

u the energy of the water with everything else. That is ``--choice uniform``: the water that is
moved is any of the region with the same probability. With ``--choice biased`` it is chosen as the
paper chooses it, with probability exp(s_D / kT) / Z_S, s the energy of a water with everything
else and Z_S the sum of exp(s / kT) over the water molecules of the region, which prefers a water
that is held weakly; r then has Z_S(x) / Z_T(y) * exp((s_C - s_D) / kT) in place of
N_S / (N_T + 1), C the pose that was chosen (Z_T(y) is taken with the energies the other water
molecules of the target region have before the move: their change by the moved water is left
out here). Here u is not the energy of the simulation but a
cheap stand-in: Lennard-Jones and reaction-field Coulomb (dielectric 78.5) within the cutoff of
the run, of the three sites of the water with all other atoms; the ligand counts with its full
parameters in a window with lambda 1 of both stages and not at all in the last window (other
windows are refused). So the numbers say how often such a move would pass ITS FIRST test with
this stand-in; a second test with the energy of the simulation would follow.

Printed per direction and number of trial poses: the mean acceptance over the moves, the share of
moves with r above 0.01 and above 1, and the numbers of water molecules in the two regions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes/out/m24"
KT = 8.31446261815324e-3 * 298.15
ONE_4PI_EPS0 = 138.93545764438198
DIELECTRIC = 78.5


def rotations(rng, n: int) -> np.ndarray:
    """n rotation matrices, uniform over orientations (unit quaternions from a normal distribution)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


class Environment:
    """Everything but one water molecule, in a rectangular cell: the stand-in energy of water poses."""

    def __init__(self, positions, box, charge, sigma, epsilon, cutoff: float) -> None:
        from scipy.spatial import cKDTree

        self.lengths = np.diag(box).astype(float)
        self.x = np.mod(positions, self.lengths)
        self.q, self.sigma, self.eps, self.cutoff = charge, sigma, epsilon, float(cutoff)
        self.tree = cKDTree(self.x, boxsize=self.lengths)
        self.k_rf = (DIELECTRIC - 1.0) / ((2.0 * DIELECTRIC + 1.0) * cutoff ** 3)
        self.c_rf = 3.0 * DIELECTRIC / ((2.0 * DIELECTRIC + 1.0) * cutoff)

    def energy(self, sites: np.ndarray, q_sites, sigma_o: float, eps_o: float, leave_out=()) -> float:
        """Energy (kJ/mol) of one water (sites: O, H, H) with all atoms but ``leave_out``."""
        total = 0.0
        skip = set(int(i) for i in leave_out)
        for s, (pos, q) in enumerate(zip(sites, q_sites)):
            near = [i for i in self.tree.query_ball_point(np.mod(pos, self.lengths), self.cutoff) if i not in skip]
            if not near:
                continue
            near = np.asarray(near)
            d = self.x[near] - np.mod(pos, self.lengths)
            d -= np.round(d / self.lengths) * self.lengths
            r = np.linalg.norm(d, axis=1)
            total += float(np.sum(ONE_4PI_EPS0 * q * self.q[near] * (1.0 / r + self.k_rf * r * r - self.c_rf)))
            if s == 0:                                                     # the oxygen carries the Lennard-Jones
                sig = 0.5 * (sigma_o + self.sigma[near])
                eps = np.sqrt(eps_o * self.eps[near])
                x6 = (sig / r) ** 6
                total += float(np.sum(4.0 * eps * (x6 * x6 - x6)))
        return total


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    run, ligand = Path(args[0]).resolve(), args[1]
    window, source, n_moves, trials, seed, choice = "last", "start", 200, [16, 128, 1024, 4096], 1, "uniform"
    rest = args[2:]
    while rest:
        a = rest.pop(0)
        if a == "--window":
            window = rest.pop(0)
        elif a == "--from":
            source = rest.pop(0)
        elif a == "--moves":
            n_moves = int(rest.pop(0))
        elif a == "--trials":
            trials = [int(v) for v in rest.pop(0).split(",")]
        elif a == "--seed":
            seed = int(rest.pop(0))
        elif a == "--choice":
            choice = rest.pop(0)
        else:
            raise SystemExit(f"unknown argument {a}")
    import openmm as mm
    import openmm.unit as u
    import yaml

    from cufep.config.schema import Config, read_settings
    from cufep.sampling.storage import read_samples
    from cufep.workflow import load_prepared

    cfg = Config.model_validate(read_settings(run / "resolved_config.yaml"))
    lig_dir, leg = run / ligand, run / ligand / "repeat_0" / "complex"
    prep = load_prepared(lig_dir / "prep", "complex")
    system = mm.XmlSerializer.deserialize(prep["system_xml"])
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    units = (u.elementary_charge, u.nanometer, u.kilojoule_per_mole)
    params = np.array([[p.value_in_unit(unit) for p, unit in zip(nb.getParticleParameters(i), units)]
                       for i in range(system.getNumParticles())])
    charge, sigma, epsilon = params[:, 0].copy(), params[:, 1].copy(), params[:, 2].copy()
    states = yaml.safe_load((leg / "schedule.yaml").read_text())["states"]
    k = len(states) - 1 if window == "last" else int(window)
    state = states[k]
    ligand_atoms = np.array(prep["ligand"])
    if state["lambda_electrostatics"] == 0.0 and state["lambda_sterics"] == 0.0:
        charge[ligand_atoms] = 0.0
        epsilon[ligand_atoms] = 0.0
    elif not (state["lambda_electrostatics"] == 1.0 and state["lambda_sterics"] == 1.0):
        raise SystemExit(f"window {k} is neither a fully interacting nor the decoupled window: {state}")
    rows = read_samples(leg / "hrex")["states"]
    if source == "start":
        data, replica = np.load(leg / "start_configs.npz"), int(np.flatnonzero(rows[0] == k)[0])
    else:
        data, replica = np.load(leg / "hrex" / "checkpoint.npz"), int(np.flatnonzero(rows[-1] == k)[0])
    positions = np.asarray(data["positions"][replica], dtype=float)
    box = np.asarray(data["boxes"][replica], dtype=float)
    if np.abs(box - np.diag(np.diag(box))).max() > 1e-9:
        raise SystemExit("this script needs a rectangular cell")
    lengths = np.diag(box)

    atoms = list(prep["topology"].atoms())
    heavy = [i for i in prep["ligand"] if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    ref = np.asarray(prep["positions"], dtype=float)[heavy]
    radius = 0.5 * float(np.max(np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2))) + 0.3
    centre = positions[heavy].mean(axis=0)                    # the solute is whole in the stored configurations
    waters = [[a.index for a in res.atoms()] for res in prep["topology"].residues() if res.name == "HOH"]
    waters = np.array([w for w in waters if len(w) == 3])
    oxygen = positions[waters[:, 0]]
    d = oxygen - centre
    d -= np.round(d / lengths) * lengths
    inside = np.linalg.norm(d, axis=1) < radius
    v_in = 4.0 / 3.0 * np.pi * radius ** 3
    v_out = float(np.prod(lengths)) - v_in
    q_w = charge[waters[0]]
    sig_o, eps_o = float(sigma[waters[0, 0]]), float(epsilon[waters[0, 0]])
    env = Environment(positions, box, charge, sigma, epsilon, float(cfg.dynamics.cutoff_nm))
    rng = np.random.default_rng(seed)
    print(f"{ligand}, window {k} {state}, {source} of the replica-exchange run: sphere of {radius:.3f} nm "
          f"({v_in:.2f} nm3 of {v_in + v_out:.1f}), {int(inside.sum())} water molecules inside, "
          f"{int((~inside).sum())} outside",
          flush=True)

    def poses(molecule: np.ndarray, into_sphere: bool, n: int) -> np.ndarray:
        """n rigid poses of the water whose oxygen is uniform in the region (the sphere, or the rest)."""
        local = positions[molecule] - positions[molecule[0]]
        local -= np.round(local / lengths) * lengths
        out = np.empty((n, 3, 3))
        filled = 0
        while filled < n:
            m = n - filled
            if into_sphere:
                v = rng.normal(size=(m, 3))
                v *= (radius * rng.random(m) ** (1.0 / 3.0) / np.linalg.norm(v, axis=1))[:, None]
                anchors = centre + v
            else:
                anchors = rng.random((m, 3)) * lengths
                e = anchors - centre
                e -= np.round(e / lengths) * lengths
                anchors = anchors[np.linalg.norm(e, axis=1) >= radius]
            rot = rotations(rng, len(anchors))
            out[filled:filled + len(anchors)] = anchors[:, None, :] + np.einsum("nij,sj->nsi", rot, local)
            filled += len(anchors)
        return out

    if choice not in ("uniform", "biased"):
        raise SystemExit(__doc__)
    scores = None
    if choice == "biased":                      # the energy of every water molecule with everything else
        scores = np.array([env.energy(positions[w], q_w, sig_o, eps_o, leave_out=w) for w in waters])
        print(f"  energy of a water molecule with everything else, kJ/mol: inside mean {scores[inside].mean():.1f} "
              f"(highest {scores[inside].max():.1f}), outside mean {scores[~inside].mean():.1f} "
              f"(highest {scores[~inside].max():.1f})", flush=True)
    result = {"ligand": ligand, "window": k, "state": state, "from": source, "radius_nm": radius, "choice": choice,
              "water_inside": int(inside.sum()), "water_outside": int((~inside).sum()), "directions": {}}
    k_max = max(trials)
    for label, from_inside in (("out of the sphere", True), ("into the sphere", False)):
        pool = np.flatnonzero(inside if from_inside else ~inside)
        n_s, n_t = len(pool), int((~inside).sum() if from_inside else inside.sum())
        v_s, v_t = (v_in, v_out) if from_inside else (v_out, v_in)
        if n_s == 0:
            print(f"  {label}: no water molecule to take")
            continue
        log_r = {n: [] for n in trials}
        others = np.flatnonzero(~inside if from_inside else inside)
        for _ in range(n_moves):
            if scores is None:
                picked = rng.choice(pool)
            else:
                weights = np.exp((scores[pool] - scores[pool].max()) / KT)
                picked = rng.choice(pool, p=weights / weights.sum())
            molecule = waters[picked]
            u_own = env.energy(positions[molecule], q_w, sig_o, eps_o, leave_out=molecule)
            forward = np.array([env.energy(p, q_w, sig_o, eps_o, leave_out=molecule)
                                for p in poses(molecule, not from_inside, k_max)])
            back = np.array([env.energy(p, q_w, sig_o, eps_o, leave_out=molecule)
                             for p in poses(molecule, from_inside, k_max - 1)])
            for n in trials:
                wf = np.logaddexp.reduce(-forward[:n] / KT)
                wr = np.logaddexp.reduce(np.append(-back[:n - 1] / KT, -u_own / KT))
                if scores is None:
                    selection = np.log(n_s / (n_t + 1.0))
                else:
                    w = np.exp(-(forward[:n] - forward[:n].min()) / KT)
                    u_new = forward[rng.choice(n, p=w / w.sum())]
                    z_s = np.logaddexp.reduce(scores[pool] / KT)
                    z_t = np.logaddexp.reduce(np.append(scores[others] / KT, u_new / KT))
                    selection = z_s - z_t + (u_new - u_own) / KT
                log_r[n].append(selection + np.log(v_t / v_s) + wf - wr)
        result["directions"][label] = {}
        for n in trials:
            x = np.array(log_r[n])
            acceptance = float(np.mean(np.minimum(1.0, np.exp(np.minimum(x, 0.0)))))
            row = {"mean_acceptance": acceptance, "share_r_above_0.01": float(np.mean(x > np.log(0.01))),
                   "share_r_above_1": float(np.mean(x > 0.0)), "median_log_r": float(np.median(x))}
            result["directions"][label][str(n)] = row
            print(f"  {label}, {n:5d} trial poses, {n_moves} moves: mean acceptance {acceptance:.4f}, r above 0.01 in "
                  f"{row['share_r_above_0.01']:.3f}, above 1 in {row['share_r_above_1']:.3f}, median ln r "
                  f"{row['median_log_r']:+.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ligand}_window{k}_{source}_{choice}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
