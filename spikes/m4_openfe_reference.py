"""M4: run OpenFE's AbsoluteBindingProtocol with its default settings (reference for cuFEP).

Runs inside the `cufep-openfe-ref` image (stock OpenMM). One process = one repeat, so
repeats can be spread over GPUs. The ligand SDF should carry partial charges (written from
a cuFEP run with `spikes/m4_export_inputs.py`), so both programs use identical charges and
the same prepared protein.

    python spikes/m4_openfe_reference.py --protein P.pdb --ligand L.sdf --out DIR [--test]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true", help="very short run to check the set-up")
    args = ap.parse_args()

    import openfe
    from gufe.protocols import execute_DAG
    from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
    from openff.units import unit
    from rdkit import Chem

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rdmol = Chem.SDMolSupplier(args.ligand, removeHs=False)[0]
    ligand = openfe.SmallMoleculeComponent(rdmol)
    protein = openfe.ProteinComponent.from_pdb_file(args.protein)
    solvent = openfe.SolventComponent()
    state_a = openfe.ChemicalSystem({"ligand": ligand, "protein": protein, "solvent": solvent})
    state_b = openfe.ChemicalSystem({"protein": protein, "solvent": solvent})

    settings = AbsoluteBindingProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.engine_settings.compute_platform = "CUDA"
    if args.test:
        for sim in (settings.complex_simulation_settings, settings.solvent_simulation_settings):
            sim.equilibration_length = 10 * unit.picosecond
            sim.production_length = 50 * unit.picosecond
        for eq in (settings.complex_equil_simulation_settings, settings.solvent_equil_simulation_settings):
            eq.equilibration_length_nvt = 10 * unit.picosecond
            eq.equilibration_length = 10 * unit.picosecond
            eq.production_length = 100 * unit.picosecond
    (out / "settings.json").write_text(json.dumps(json.loads(settings.model_dump_json()), indent=1))
    protocol = AbsoluteBindingProtocol(settings=settings)
    dag = protocol.create(stateA=state_a, stateB=state_b, mapping=None)
    t0 = time.time()
    shared, scratch = out / "shared", out / "scratch"
    shared.mkdir(exist_ok=True); scratch.mkdir(exist_ok=True)
    dag_result = execute_DAG(dag, shared_basedir=shared, scratch_basedir=scratch, keep_shared=True, n_retries=0)
    result = protocol.gather([dag_result])
    kcal = unit.kilocalorie_per_mole
    summary = {"dg_bind_kcal_mol": result.get_estimate().m_as(kcal),
               "uncertainty_kcal_mol": result.get_uncertainty().m_as(kcal), "openfe_version": openfe.__version__,
               "hours": (time.time() - t0) / 3600.0, "test": bool(args.test), "legs": {}}
    try:
        for leg, values in result.get_individual_estimates().items():
            summary["legs"][leg] = [[v[0].m_as(kcal), v[1].m_as(kcal)] for v in values]
    except Exception as exc:  # noqa: BLE001
        summary["legs_error"] = str(exc)
    (out / "result.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
