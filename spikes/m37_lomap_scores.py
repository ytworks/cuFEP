"""LOMAP scores (Kartograf mapping, as cufep rbfe maps) of every pair of the given ligands of an SDF.

    spikes/run.sh spikes/m37_lomap_scores.py benchmarks/tyk2/ligands.sdf ejm_31 ejm_46 ejm_49 ...
"""
import itertools
import sys

from cufep.prep.ligand import load_ligands
from cufep.rbfe.mapping import kartograf_mapping, lomap_score

sdf, *names = sys.argv[1:]
ligands = {lig.name: lig for lig in load_ligands(sdf)}
for a, b in itertools.combinations(names, 2):
    m = kartograf_mapping(ligands[a], ligands[b])
    unique = f"{ligands[a].n_atoms - len(m):2d} + {ligands[b].n_atoms - len(m):2d}"
    print(f"{a:8s} {b:8s} mapped {len(m):3d}  unique {unique}  LOMAP {lomap_score(ligands[a], ligands[b], m):.2f}",
          flush=True)
