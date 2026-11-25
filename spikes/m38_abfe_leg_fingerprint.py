"""A fingerprint of the ABFE legs, to show that a refactoring of the shared code changes nothing.

An alanine dipeptide is the protein and benzene the ligand (Gasteiger charges, in the charge cache); both legs run
end to end on the Reference platform with fixed seeds (the CPU platform, without deterministic forces, gave another
System after the pre-equilibration in a second run), with short settings and a manual Boresch restraint.
What is written: each leg's result.json (without the parts that are not of the calculation: timing), the
conditions its samples record (with the checksum of the alchemical System) and its schedule. Run it before and
after the refactoring into two directories and compare the two fingerprint.json: they must be identical.

    spikes/run.sh spikes/m38_abfe_leg_fingerprint.py runs/m38_before
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def protein():
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.protein import PreparedProtein

    heavy = [("ACE", "CH3", "C", (0.00, 0.00, 0.0)), ("ACE", "C", "C", (1.25, 0.85, 0.0)),
             ("ACE", "O", "O", (1.25, 2.08, 0.0)), ("ALA", "N", "N", (2.45, 0.20, 0.0)),
             ("ALA", "CA", "C", (3.70, 0.95, 0.0)), ("ALA", "CB", "C", (3.70, 1.75, 1.30)),
             ("ALA", "C", "C", (4.95, 0.10, 0.0)), ("ALA", "O", "O", (4.95, -1.13, 0.0)),
             ("NME", "N", "N", (6.15, 0.75, 0.0)), ("NME", "C", "C", (7.40, 0.00, 0.0))]
    bonds = {"ACE": [("CH3", "C"), ("C", "O")], "ALA": [("N", "CA"), ("CA", "CB"), ("CA", "C"), ("C", "O")],
             "NME": [("N", "C")]}
    top, pos = app.Topology(), []
    chain, previous_c = top.addChain("A"), None
    for number, res_name in enumerate(("ACE", "ALA", "NME"), start=1):
        res, local = top.addResidue(res_name, chain, str(number)), {}
        for rn, name, element, xyz in heavy:
            if rn == res_name:
                local[name] = top.addAtom(name, app.Element.getBySymbol(element), res)
                pos.append(mm.Vec3(*(0.1 * np.array(xyz) + 1.0)))
        for a, b in bonds[res_name]:
            top.addBond(local[a], local[b])
        if previous_c is not None:
            top.addBond(previous_c, local["N"])
        previous_c = local["C"] if res_name != "NME" else None
    import random

    random.seed(20261125)
    mod = app.Modeller(top, pos * u.nanometer)
    mod.addHydrogens(app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml"),
                     platform=mm.Platform.getPlatformByName("Reference"))
    return PreparedProtein(mod.topology, mod.positions, None, {"A": "A"})


def ligand(out: Path, cache: Path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep.prep.ligand import load_ligands

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    xyz = mol.GetConformer().GetPositions()
    for i, row in enumerate(xyz - xyz.mean(axis=0) + np.array([14.0, 16.0, 14.0])):
        mol.GetConformer().SetAtomPosition(i, row.tolist())
    mol.SetProp("_Name", "benzene")
    w = Chem.SDWriter(str(out / "ligand.sdf"))
    w.write(mol)
    w.close()
    lig = load_ligands(out / "ligand.sdf")[0]
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"charges_am1bcc_{lig.digest()}.json").write_text(json.dumps({"charges": [-0.13] * 6 + [0.13] * 6}))
    return lig


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from cufep.config.schema import Config
    from cufep.workflow import LegJob, load_prepared, prepare_ligand, run_leg_safe

    prot = protein()
    lig = ligand(out, out / "run" / "cache")
    (out / "protein.pdb").write_text("REMARK built by the script\n")
    short = {"nvt_ns": 0.002, "npt_ns": 0.002, "md_ns": 0.006}
    data = {"system": {"protein": str(out / "protein.pdb"), "ligand": str(out / "ligand.sdf")},
            "backend": {"name": "openmm", "platform": "Reference"},
            "dynamics": {"cutoff_nm": 0.6},
            "solvation": {"box_shape": "cube", "padding_complex_nm": 0.8, "padding_solvent_nm": 0.8},
            "thermodynamics": {"ionic_strength_M": 0.0},
            "sampling": {"production_ns": 0.005, "equilibration_ns": 0.0025, "time_per_iteration_ps": 0.25,
                         "checkpoint_interval_ps": 1.0, "rest2": {"scale": 1.5},
                         "pre_equilibration": {"complex": short, "solvent": short}},
            "analysis": {"error_blocks": [2], "part_error_blocks": 2, "convergence_points": 2,
                         "convergence_blocks": 2},
            "lambda_schedule": {"mode": "fixed", "fixed_walk_ps": 0.5,
                                "windows": {"complex": {"restrain": 2, "discharge": 3, "vanish": 3},
                                            "solvent": {"discharge": 3, "vanish": 3}}},
            "restraints": {"selection": "manual", "host_atoms": [0, 0, 0], "guest_atoms": [0, 0, 0]},
            "resources": {"gpus": [], "cpu_threads_per_worker": 1},
            "output": {"dir": str(out / "run")}}
    lig_dir = out / "run" / "benzene"
    # the anchors: three heavy atoms of the peptide's backbone and three carbons of benzene, in the prepared complex
    cfg = Config.model_validate({**data, "restraints": {"selection": "auto"}})
    prepare_ligand(cfg, lig, prot, lig_dir, False, "cube", print)
    prep = load_prepared(lig_dir / "prep", "complex")
    atoms = list(prep["topology"].atoms())
    host = [next(a.index for a in atoms if a.residue.name == "ALA" and a.name == n) for n in ("N", "CA", "C")]
    guest = [i for i in prep["ligand"] if atoms[i].element.symbol == "C"][:5:2]
    data["restraints"] = {"selection": "manual", "host_atoms": host, "guest_atoms": guest}
    cfg = Config.model_validate(data)
    fingerprint = {}
    for leg in ("solvent", "complex"):
        job = LegJob(ligand="benzene", leg=leg, repeat=0, lig_dir=str(lig_dir), devices=[None],
                     config=cfg.model_dump(mode="json"), production_ns=cfg.sampling.production(leg), inline=True)
        result = run_leg_safe(job)
        if "error" in result:
            raise SystemExit(f"{leg}: {result['error']}")
        result.pop("timing", None)
        from cufep.sampling.storage import read_samples

        samples = read_samples(job.leg_dir / "hrex")
        fingerprint[leg] = {"result": result, "conditions": samples["attrs"].get("conditions"),
                            "schedule": (job.leg_dir / "schedule.yaml").read_text(),
                            "u_sum": float(np.sum(samples["u"])), "states": np.asarray(samples["states"]).tolist()}
    text = json.dumps(fingerprint, indent=1, sort_keys=True, default=str)
    (out / "fingerprint.json").write_text(text)
    print(f"wrote {out / 'fingerprint.json'}: dG solvent {fingerprint['solvent']['result']['delta_g_decouple']:.6f}, "
          f"complex {fingerprint['complex']['result']['delta_g_decouple']:.6f} kJ/mol")


if __name__ == "__main__":
    main()
