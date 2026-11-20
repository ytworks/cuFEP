import numpy as np
import openmm.app as app
import openmm.unit as u
import pytest

from cufep.prep.protein import ProteinError, infer_variant, prepare_protein

pytestmark = pytest.mark.needs("pdbfixer")


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
        q = sum(
            nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles())
        )
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


# ---- review of PR 1 on a3462a5: the identity of a residue is its chain, number and insertion code --------

def _modified_tyk2(repo, tmp_path, *, chain="A", insertion=False, drop=(), extra=(), name="tyk2_modified.pdb"):
    """The TYK2 input with its chain renamed, LYS 899 renumbered 898A, some atoms removed and some lines added."""
    lines = []
    for line in (repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb").read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name, resname, number = line[12:16].strip(), line[17:20], line[22:26].strip()
            if (resname, number, name) in drop:
                continue
            line = line[:21] + chain + line[22:]
            if insertion and number == "899":
                line = line[:22] + " 898" + "A" + line[27:]
        lines.append(line)
    for line in extra:
        if line.startswith("ATOM"):                     # an atom of a residue goes right after that residue
            key = (line[17:20], line[21], line[22:26])
            last = max(i for i, other in enumerate(lines)
                       if other.startswith("ATOM") and (other[17:20], other[21], other[22:26]) == key)
            lines.insert(last + 1, line)
        else:                                           # waters at the end, before END
            end = next((i for i, other in enumerate(lines) if other.startswith("END")), len(lines))
            lines.insert(end, line)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _atom_line(serial, name, resname, chain, number, xyz_nm, element):
    x, y, z = (10.0 * v for v in xyz_nm)
    return (f"HETATM{serial:5d} {name:<4s} {resname:>3s} {chain}{number:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            f"          {element:>2s}")


def _tyk2_atom(repo, resname, number, name):
    for line in (repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb").read_text().splitlines():
        if line.startswith("ATOM") and line[17:20] == resname and line[22:26].strip() == str(number) \
                and line[12:16].strip() == name:
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) / 10.0
    raise KeyError((resname, number, name))


def test_residues_with_an_insertion_code_stay_residues_of_their_own(repo, tmp_path):
    """Review of PR 1 (P1): LYS 898 and LYS 898A of one chain became one residue of 22 atoms instead of two of 44
    (the insertion code was lost when the residues were copied)."""
    from cufep.prep.protein import residue_key

    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, insertion=True))
    by_key = {residue_key(r): r for r in prepared.topology.residues()}
    pair = [by_key[("A", "898", "")], by_key[("A", "898", "A")]]
    assert [r.name for r in pair] == ["LYS", "LYS"] and [len(list(r.atoms())) for r in pair] == [22, 22]
    reference = {residue_key(r)[1]: len(list(r.atoms())) for r in prepare_protein(
        repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb").topology.residues()}
    assert prepared.topology.getNumAtoms() == sum(reference.values())
    # the insertion code survives the topology file of the prepared systems
    from cufep.prep.topology_io import topology_from_dict, topology_to_dict

    again = topology_from_dict(topology_to_dict(prepared.topology))
    assert [residue_key(r) for r in again.residues()] == [residue_key(r) for r in prepared.topology.residues()]


def test_rest2_residues_are_named_as_in_the_input_pdb(repo, tmp_path):
    """Review of PR 1 (P2): the preparation names its chains A, B, ... so with an input of chain B only, "B:960"
    found nothing, and with several chains a number could take a residue of another chain."""
    from cufep.config.schema import Rest2Config
    from cufep.workflow import WorkflowError, rest2_region

    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, chain="B", insertion=True))
    assert set(prepared.input_chains.values()) == {"B"}
    top = prepared.topology
    pos = prepared.positions.value_in_unit(u.nanometer)
    protein = [a.index for a in top.atoms()]
    chains = prepared.input_chains

    def region(*residues, input_chains=chains):
        return rest2_region(
            Rest2Config(scale=1.5, region="ligand", residues=list(residues)), top, pos, [], protein, input_chains
        )[1]

    assert region("B:960") == ["ILE960"] and region("960") == ["ILE960"]
    assert region("B:898A") == ["LYS898A"] and region("B:898") == ["LYS898"]
    with pytest.raises(WorkflowError, match="no protein residue 'A:960'"):
        region("A:960")
    with pytest.raises(WorkflowError, match="not a residue number"):
        region("B:ILE960")
    # a preparation of an earlier version recorded no chains of the input: refused, not guessed
    with pytest.raises(WorkflowError, match="did not record the chains of the input"):
        region("B:960", input_chains=None)


def test_hydrogens_that_do_not_decide_a_state_are_left_to_the_ph(repo, tmp_path):
    """Review of PR 1 (P2): HIS 893 without HD1 and HE2 (its ring protons removed) was made HIN, which Amber14
    has no template for, and the preparation failed. Such residues get their state from addHydrogens at the pH
    of the run and are listed as assigned; so do a lysine without HZ and a cysteine without HG."""
    drop = {("HIS", "893", "HE2"), ("LYS", "898", "HZ1"), ("LYS", "898", "HZ2"), ("LYS", "898", "HZ3")}
    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, drop=drop))
    assigned = prepared.report.protonation_assigned
    assert any(a.startswith("HIS893/A=") for a in assigned) and any(a.startswith("LYS898/A=") for a in assigned)
    from cufep.prep.protein import residue_key

    residues = {residue_key(r): r for r in prepared.topology.residues()}
    names = {a.name for a in residues[("A", "898", "")].atoms()}
    assert {"HZ1", "HZ2", "HZ3"} <= names                       # charged lysine at pH 7.4
    ring = {a.name for a in residues[("A", "893", "")].atoms()} & {"HD1", "HE2"}
    assert len(ring) >= 1


def test_variants_are_read_only_from_hydrogens_that_decide_them():
    top = app.Topology()
    chain = top.addChain()

    def residue(name, hydrogens):
        res = top.addResidue(name, chain)
        for atom in ["N", "CA", "C", "O"]:
            top.addAtom(atom, app.element.get_by_symbol(atom[0]), res)
        for h in hydrogens:
            top.addAtom(h, app.element.hydrogen, res)
        return res

    assert infer_variant(residue("HIS", ["H", "HA", "HB2", "HB3", "HD2", "HE1"])) is None          # no ring proton
    assert infer_variant(residue("HIS", ["H", "HA", "HB2", "HD1"])) == "HID"
    assert infer_variant(residue("HIS", ["H", "HA", "HB2", "HD1", "HE2"])) == "HIP"
    assert infer_variant(residue("ASP", ["H", "HA"])) is None                                    # backbone only
    assert infer_variant(residue("ASP", ["H", "HA", "HB2", "HB3"])) == "ASP"
    assert infer_variant(residue("LYS", ["H", "HA", "HB2", "HE2"])) is None
    assert infer_variant(residue("LYS", ["H", "HA", "HB2", "HZ1", "HZ2"])) == "LYN"
    assert infer_variant(residue("CYS", ["H", "HA", "HB2", "HB3"])) is None
    assert infer_variant(residue("CYS", ["H", "HA", "HB2", "HG"])) == "CYS"



# ---- review of PR 1 on 64751ca: the preparation keeps what the input says -------------------------------

def test_crystal_waters_of_two_chains_with_one_number_stay_two(repo, tmp_path):
    """P1: HOH A401 and HOH B401 went into one chain under their own numbers and were read back as one water."""
    centre = _tyk2_atom(repo, "LYS", 898, "NZ")
    waters = [_atom_line(9001, "O", "HOH", "A", 401, centre + [0.0, 0.0, 1.2], "O"),
              "TER", _atom_line(9002, "O", "HOH", "B", 401, centre + [0.0, 1.2, 0.0], "O")]
    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, extra=waters))
    hoh = [r for r in prepared.topology.residues() if r.name == "HOH"]
    assert len(hoh) == 2 and all(len(list(r.atoms())) == 3 for r in hoh)
    assert "kept 2 waters" in " ".join(prepared.report.notes)


@pytest.mark.parametrize("keep_water", [True, False])
def test_crystal_waters_and_ions_of_alternating_chains(repo, tmp_path, keep_water):
    """Review of 037bdf1 (P2): crystal waters and Na given in the order of the chains A, B, A, B stopped the
    preparation: a residue was added to chain A after chain B had been made (OpenMM's addResidue requires the
    residues of a chain to follow each other). Two waters of one number in different chains and both Na stay; without
    waters the two Na stay."""
    from cufep.prep.protein import residue_key

    centre = _tyk2_atom(repo, "LYS", 898, "NZ")
    extra = [_atom_line(9001, "O", "HOH", "A", 401, centre + [0.0, 0.0, 1.2], "O"), "TER",
             _atom_line(9002, "O", "HOH", "B", 401, centre + [0.0, 1.2, 0.0], "O"), "TER",
             _atom_line(9003, "NA", "NA", "A", 402, centre + [1.5, 0.0, 0.0], "NA"), "TER",
             _atom_line(9004, "NA", "NA", "B", 402, centre + [-1.5, 0.0, 0.0], "NA")]
    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, extra=extra), keep_water=keep_water)
    residues = list(prepared.topology.residues())
    hoh = [r for r in residues if r.name == "HOH"]
    na = [r for r in residues if r.name == "NA"]
    assert len(na) == 2 and len({r.chain.index for r in na}) == 2
    if keep_water:
        assert len(hoh) == 2 and all(len(list(r.atoms())) == 3 for r in hoh)
        assert len({residue_key(r) for r in hoh}) == 2 and {r.id for r in hoh} == {"401"}
        assert "kept 2 waters and 2 ions" in " ".join(prepared.report.notes)
    else:
        assert hoh == [] and "kept 0 waters and 2 ions" in " ".join(prepared.report.notes)
    # every chain holds residues that follow each other, so the written file reads back as it was
    for chain in prepared.topology.chains():
        indices = [r.index for r in chain.residues()]
        assert indices == list(range(indices[0], indices[0] + len(indices)))


@pytest.mark.parametrize("resname,number,carbon,oxygen,proton", [("ASP", 902, "CG", "OD1", "HD1"),
                                                                  ("GLU", 905, "CD", "OE1", "HE1")])
def test_a_carboxyl_proton_stays_on_its_oxygen(repo, tmp_path, resname, number, carbon, oxygen, proton):
    """P1: ASP 902 with HD1 on OD1 came out with the proton on OD2, 3.0 A from where it was; GLU with HE1 likewise.
    The oxygen the proton is bonded to takes the name the force field puts the proton on."""
    try:
        o = _tyk2_atom(repo, resname, number, oxygen)
    except KeyError:
        pytest.skip(f"{resname} {number} is not in the TYK2 input")
    c = _tyk2_atom(repo, resname, number, carbon)
    h = o + 0.095 * (o - c) / np.linalg.norm(o - c)
    line = (f"ATOM   9100 {proton:<4s} {resname} A{number:>4d}    "
            f"{10 * h[0]:8.3f}{10 * h[1]:8.3f}{10 * h[2]:8.3f}  1.00  0.00           H")
    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, extra=[line]))
    pos = prepared.positions_nm()
    residue = next(r for r in prepared.topology.residues() if r.id == str(number) and r.chain.id == "A")
    atoms = {a.name: a for a in residue.atoms()}
    hydrogen = next(a for a in residue.atoms() if a.name in ("HD2", "HE2"))
    bonded = min((a for a in residue.atoms() if a.element.symbol == "O" and a.name != "O"),
                 key=lambda a: np.linalg.norm(pos[a.index] - pos[hydrogen.index]))
    assert np.linalg.norm(pos[bonded.index] - o) < 1e-3                  # the oxygen the proton was on
    assert np.linalg.norm(pos[hydrogen.index] - pos[bonded.index]) < 0.11
    assert residue.name in ("ASH", "GLH", "ASP", "GLU") and len([a for a in atoms if a.startswith("H")]) > 2


def test_a_crystal_water_with_a_four_site_model(repo, tmp_path):
    """P2: with amber14/tip4pew.xml a crystal water given by its oxygen stopped the preparation (one site short)."""
    centre = _tyk2_atom(repo, "LYS", 898, "NZ")
    water = [_atom_line(9001, "O", "HOH", "A", 401, centre + [0.0, 0.0, 1.2], "O")]
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip4pew.xml")
    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, extra=water), forcefield=ff)
    hoh = [r for r in prepared.topology.residues() if r.name == "HOH"]
    assert len(hoh) == 1 and len(list(hoh[0].atoms())) == 4


def test_a_missing_backbone_atom_is_rebuilt_and_does_not_break_the_chain(repo, tmp_path):
    """P2: residues 890-895 with the N of PHE 892 missing stopped with "PHE892/A has no backbone N"."""
    prepared = prepare_protein(_modified_tyk2(repo, tmp_path, drop={("PHE", "892", "N"), ("PHE", "892", "H")}))
    assert prepared.report.chain_breaks == []
    residue = next(r for r in prepared.topology.residues() if r.id == "892" and r.chain.id == "A")
    assert "N" in {a.name for a in residue.atoms()}
    assert any("backbone atom missing" in n for n in prepared.report.notes)


def test_one_input_gives_one_prepared_protein(repo):
    """Review of 771a7ab (B5, the A2A example): two preparations of one input placed the caps up to 1.8 A apart
    (PDBFixer's integrator had the seed 0, which OpenMM reads as a new seed every time, on whatever platform), so
    the bilayer was trimmed to another box each time. The minimisations of the preparation run on the Reference
    platform with a fixed seed."""
    first = prepare_protein(repo / "examples/membrane_a2a/protein.pdb")
    second = prepare_protein(repo / "examples/membrane_a2a/protein.pdb")
    assert [(a.residue.name, a.name) for a in first.topology.atoms()] == [(a.residue.name, a.name)
                                                                         for a in second.topology.atoms()]
    assert np.array_equal(first.positions_nm(), second.positions_nm())
    caps = [a.index for a in first.topology.atoms() if a.residue.name in ("ACE", "NME")]
    assert len(caps) == 18
    with pytest.raises(ValueError, match="must not be 0"):
        prepare_protein(repo / "examples/membrane_a2a/protein.pdb", seed=0)
