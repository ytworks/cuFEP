"""AM1-BCC charges of a ligand (cufep.prep.ligand.assign_partial_charges)."""
import os
import subprocess

import numpy as np
from conftest import REPO


def test_one_input_gets_the_same_am1bcc_charges_whatever_the_threads_of_the_caller(monkeypatch):
    """TYK2 (2026-11-28): the charges of jmc_23, ejm_44, ejm_49 and jmc_30 differed by 5e-4 to 1e-3 e per atom between
    two preparations of the same input, and three calculations of ejm_49 on 128 threads differed among themselves:
    AmberTools' sqm sums in no fixed order on more than one OpenMP thread, and inherits the threads of the process
    (jmc_23 on one thread gave the charges of one preparation, on two, four and 128 those of the other). antechamber,
    and with it sqm, runs on one thread whatever the caller's setting, which is left as it was, and the charges are
    the same."""
    from cufep.prep.ligand import assign_partial_charges, load_ligands

    real, seen = subprocess.check_output, []

    def check_output(command, *args, **kwargs):        # how OpenFF starts antechamber (and other programs)
        if isinstance(command, (list, tuple)) and str(command[0]).endswith("antechamber"):
            seen.append(os.environ.get("OMP_NUM_THREADS"))
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    charges = {}
    for threads in ("4", "1"):
        monkeypatch.setenv("OMP_NUM_THREADS", threads)
        (ligand,) = [x for x in load_ligands(REPO / "benchmarks" / "tyk2" / "ligands.sdf") if x.name == "jmc_23"]
        assign_partial_charges(ligand, "am1bcc", None)
        charges[threads] = np.asarray(ligand.offmol.partial_charges.m)
        assert os.environ["OMP_NUM_THREADS"] == threads
    assert seen and set(seen) == {"1"}
    assert np.array_equal(charges["4"], charges["1"])
