import numpy as np
import openmm.app as app
import openmm.unit as u
import pytest

from cufep.prep.protein import ProteinError, infer_variant, prepare_protein


def heavy_coordinates(path):
    pdb = app.PDBFile(str(path))
    pos = np.array(pdb.positions.value_in_unit(u.nanometer))
    return {(a.residue.id, a.name): pos[a.index] for a in pdb.topology.atoms()
            if a.element.symbol != "H" and a.residue.name != "UNK"}


@pytest.fixture(scope="module")
def tyk2(repo):
    return prepare_protein(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb")


@pytest.fixture(scope="module")
def a2a(repo):
    return prepare_protein(repo / "benchmarks/a2a/inputs/a2a_hip278_protein.pdb")


def test_caps_written_as_unknown_residues_are_recognised(tyk2):
    caps = [(r.name, [a.name for a in r.atoms()]) for r in tyk2.topology.residues() if r.name in ("ACE", "NME")]
    assert [c[0] for c in caps] == ["ACE", "NME"]
    assert set(caps[0][1]) == {"C", "O", "CH3", "H1", "H2", "H3"}
    assert set(caps[1][1]) == {"N", "H", "C", "H1", "H2", "H3"}
    assert tyk2.report.had_hydrogens and not tyk2.report.caps_added and not tyk2.report.chain_breaks
    assert tyk2.topology.getNumAtoms() == 4670


def test_given_protonation_states_are_kept(tyk2):
    states = {}
    for r in tyk2.topology.residues():
        if r.name == "HIS":
            states[r.id] = infer_variant(r)
    assert states["907"] == "HIP"
    assert sorted(set(states.values())) == ["HID", "HIE", "HIP"]
    assert list(states.values()).count("HIP") == 1


def test_no_heavy_atom_moves(repo, tyk2, a2a):
    for prep, path in ((tyk2, "benchmarks/tyk2/inputs/tyk2_protein.pdb"),
                       (a2a, "benchmarks/a2a/inputs/a2a_hip278_protein.pdb")):
        ref = heavy_coordinates(repo / path)
        pos = prep.positions_nm()
        d = [np.linalg.norm(pos[a.index] - ref[(a.residue.id, a.name)]) for a in prep.topology.atoms()
             if a.element.symbol != "H" and (a.residue.id, a.name) in ref]
        assert len(d) == len(ref)
        assert max(d) < 1e-4


def test_chain_break_is_split_and_capped(a2a):
    assert a2a.report.chain_breaks == ["LEU208/A | GLU219/A"]
    assert a2a.report.caps_added == ["NME after LEU208/A", "ACE before GLU219/A", "NME after SER305/A"]
    chains = [len(list(c.residues())) for c in a2a.topology.chains()]
    assert chains[:2] == [211, 89]
    ions = [r.name for r in a2a.topology.residues() if r.name == "NA"]
    assert ions == ["NA"]
    # the explicitly charged N-terminus (three hydrogens on N) is left alone
    first = next(a2a.topology.residues())
    n_atom = next(a for a in first.atoms() if a.name == "N")
    bonded_h = [b for b in a2a.topology.bonds()
                if n_atom in (b[0], b[1]) and "H" in (b[0].element.symbol, b[1].element.symbol)]
    assert first.name == "GLY" and len(bonded_h) == 3


def test_systems_can_be_parameterized(tyk2, a2a):
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    for prep, charge in ((tyk2, -3), (a2a, 11)):
        system = ff.createSystem(prep.topology, nonbondedMethod=app.NoCutoff)
        nb = [f for f in system.getForces() if f.__class__.__name__ == "NonbondedForce"][0]
        q = sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles()))
        assert round(q) == charge and abs(q - round(q)) < 1e-3


def test_unknown_residue_stops_with_a_message(tmp_path, repo):
    lines = (repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb").read_text().splitlines()
    het = "HETATM 9001  C1  XYZ B   1      10.000  10.000  10.000  1.00  0.00           C  "
    het2 = "HETATM 9002  C2  XYZ B   1      11.500  10.000  10.000  1.00  0.00           C  "
    out = [l for l in lines if not l.startswith(("END", "CONECT", "MASTER"))] + [het, het2, "END"]
    bad = tmp_path / "bad.pdb"
    bad.write_text("\n".join(out) + "\n")
    with pytest.raises(ProteinError, match="XYZ"):
        prepare_protein(bad)


def test_hydrogen_free_input_gets_hydrogens(tmp_path, repo):
    lines = (repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb").read_text().splitlines()
    keep = [l for l in lines if l.startswith("ATOM") and l[76:78].strip() != "H" and int(l[22:26]) <= 960]
    p = tmp_path / "noh.pdb"
    p.write_text("\n".join(keep) + "\nEND\n")
    prep = prepare_protein(p)
    assert not prep.report.had_hydrogens
    assert prep.report.protonation_assigned
    n_h = sum(1 for a in prep.topology.atoms() if a.element.symbol == "H")
    assert n_h > 400
    # chain ends of a hydrogen-free file become charged termini
    last = [r for r in prep.topology.residues()][-1]
    assert "OXT" in {a.name for a in last.atoms()}
