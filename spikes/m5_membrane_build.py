"""M5 spike: build the A2A complex in a bilayer and report what was built (no dynamics)."""
import time
from pathlib import Path

import openmm.app as app

from cufep.config.schema import Config
from cufep.prep.forcefield import ForceFieldSpec, SystemSpec, create_system, make_forcefield
from cufep.prep.ligand import assign_partial_charges, load_ligands
from cufep.prep.membrane import add_membrane
from cufep.prep.protein import prepare_protein
from cufep.prep.solvate import combine, ligand_atom_indices

out = Path("spikes/out/mem")
out.mkdir(parents=True, exist_ok=True)
cfg = Config.model_validate({"system": {"protein": "examples/membrane_a2a/protein.pdb",
                                        "ligand": "examples/membrane_a2a/ligand.sdf"}})
protein = prepare_protein(cfg.system.protein)
ligand = load_ligands(cfg.system.ligand)[0]
assign_partial_charges(ligand, "am1bcc", cache_dir="spikes/out/cache")
ff = make_forcefield(ForceFieldSpec(), [ligand], include_lipid=True)
modeller = combine((protein.topology, protein.positions), ligand)
t0 = time.perf_counter()
info = add_membrane(modeller, ff, cfg, print)
print(f"built in {time.perf_counter() - t0:.0f} s")
for k, v in info.items():
    print(f"   {k}: {v}")
system = create_system(ff, modeller.topology, SystemSpec())
print("system:", system.getNumParticles(), "atoms; ligand atoms", len(ligand_atom_indices(modeller.topology)))
with open(out / "a2a_membrane.pdb", "w") as fh:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
