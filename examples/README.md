# cuFEP examples

Each directory holds the only two inputs cuFEP needs, a protein PDB and a ligand SDF with
the ligand in the pocket, plus a configuration that names them and an output directory.
Both structures come from the Schrödinger public binding free energy benchmark (MIT
licence, see `benchmarks/THIRD_PARTY/`).

| directory | system | experiment |
|---|---|---|
| `kinase_tyk2/` | TYK2 kinase (PDB 4GIH) with ligand ejm_31 | -9.54 kcal/mol |
| `membrane_a2a/` | adenosine A2A receptor (PDB 5OLZ) with compound 4e, in a POPC bilayer | -12.07 kcal/mol |

```bash
cufep() { docker run --rm --gpus all --shm-size=16g -u "$(id -u):$(id -g)" -v "$PWD":/work cufep:latest "$@"; }

cufep plan examples/kinase_tyk2/config.yaml     # settings and lambda windows of the run
cufep run  examples/kinase_tyk2/config.yaml     # default settings
cufep run  examples/membrane_a2a/config.yaml
```

For a first look that finishes in minutes, shorten the sampling on the command line
(the result is then not converged):

```bash
cufep run examples/kinase_tyk2/config.yaml --production-ns 0.2 -o runs/tyk2_try
```

Everything a configuration does not name is a default. Among the defaults, the Boresch restraint
of the complex leg is chosen from the interactions between protein and ligand in four short
unrestrained trajectories of the complex (`restraints.selection: interactions`,
`docs/configuration.md`); they run before the lambda windows are chosen, also in the short first
look above.

The A2A configuration switches the membrane on with `membrane: {enabled: true}`. cuFEP then
orients the complex, builds a POPC bilayer around it and hydrates the pocket. Without that
line a system is always simulated in water.
