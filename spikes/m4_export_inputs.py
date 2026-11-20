"""Write the inputs of an OpenFE reference run from a cuFEP run directory:
the prepared protein and the ligand with the partial charges cuFEP used.

    spikes/run.sh spikes/m4_export_inputs.py RUN_DIR LIGAND OUT_DIR
"""
import shutil
import sys
from pathlib import Path

from openff.toolkit import Molecule

run_dir, ligand, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
shutil.copy(run_dir / "protein_prepared.pdb", out / "protein.pdb")
mol = Molecule.from_json((run_dir / ligand / "prep" / "ligand.json").read_text())
mol.to_file(str(out / "ligand_charged.sdf"), "SDF")
print(
    "wrote", out / "protein.pdb", "and", out / "ligand_charged.sdf", "| charge sum", float(sum(mol.partial_charges.m))
)
