"""Protein preparation from a bare PDB file.

Goal: accept "a protein PDB" as the only receptor input and return a Topology that the
force field can parameterize, changing as little as possible and recording every change.

Rules (see docs/protocol.md, "Inputs"):

* Hydrogens present in the file are kept and define protonation/tautomer states.
  Missing hydrogens are added at the requested pH.
* Missing heavy atoms of standard residues are rebuilt (PDBFixer).
* Chain breaks split the chain; the two new ends get neutral ACE/NME caps. Missing
  loops are never modelled.
* Chain ends that are explicitly charged in the file (N with three hydrogens, C with
  OXT) stay charged. Ends carrying a hydrogen cap or a lone backbone hydrogen are
  capped with ACE/NME. In files without hydrogens, chain ends become charged termini.
* Water and monatomic ions are kept. Any other unknown residue stops the run.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm.app import element as elem

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
}
_CANONICAL = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
              "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
CAPS = {"ACE", "NME", "NMA"}
WATER = {"HOH", "WAT", "H2O", "TIP", "TIP3", "SOL"}
TITRATABLE = {"HIS", "ASP", "GLU", "LYS", "CYS", "TYR", "ARG"}
#: residues whose state infer_variant reads from the hydrogens of the file
INFERRED = {"HIS", "ASP", "GLU", "LYS", "CYS"}
MAX_PEPTIDE_BOND_NM = 0.16
#: neighbouring residues whose C-alpha atoms are this close are consecutive in the chain (3.8 A in a
#: peptide); used where the C or the N of the peptide bond is missing and PDBFixer is to rebuild it
MAX_CA_CA_NM = 0.42
#: protonated carboxyl groups: the two oxygens, the oxygen and the hydrogen names of the protonated
#: form in the force field (Amber: ASH carries HD2 on OD2, GLH HE2 on OE2), and the hydrogen names a
#: file may use
CARBOXYLS = {
    "ASP": (("OD1", "OD2"), "OD2", "HD2", ("HD1", "HD2")),
    "ASH": (("OD1", "OD2"), "OD2", "HD2", ("HD1", "HD2")),
    "GLU": (("OE1", "OE2"), "OE2", "HE2", ("HE1", "HE2")),
    "GLH": (("OE1", "OE2"), "OE2", "HE2", ("HE1", "HE2")),
}
MAX_OH_BOND_NM = 0.13
_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class ProteinError(ValueError):
    """The protein input cannot be prepared automatically."""


@dataclass
class ProteinReport:
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    had_hydrogens: bool = False
    chain_breaks: list[str] = field(default_factory=list)
    caps_added: list[str] = field(default_factory=list)
    protonation_assigned: list[str] = field(default_factory=list)
    net_charge: float | None = None

    def lines(self) -> list[str]:
        out = [f"note: {n}" for n in self.notes] + [f"WARNING: {w}" for w in self.warnings]
        return out


@dataclass
class PreparedProtein:
    topology: app.Topology
    positions: unit.Quantity
    report: ProteinReport
    #: chain of the input PDB of every chain of the prepared protein: the preparation gives every
    #: peptide segment a chain of its own (a chain break makes two), named A, B, ... in order, so
    #: residues named in the words of the input (sampling.rest2.residues) are found through this
    input_chains: dict[str, str] = field(default_factory=dict)

    def positions_nm(self) -> np.ndarray:
        return np.asarray(self.positions.value_in_unit(unit.nanometer), dtype=float)

    def write_pdb(self, path: str | Path) -> None:
        from cufep.files import publish_with

        publish_with(path, lambda fh: app.PDBFile.writeFile(self.topology, self.positions, fh, keepIds=True))


# --------------------------------------------------------------------------- helpers
def residue_key(res) -> tuple[str, str, str]:
    """What identifies a residue in a PDB file: chain, residue number and insertion code.
    ``LYS 898`` and ``LYS 898A`` of one chain are two residues (review of PR 1)."""
    return (res.chain.id, str(res.id), (getattr(res, "insertionCode", "") or "").strip())


def _label(res) -> str:
    chain, number, code = residue_key(res)
    return f"{res.name}{number}{code}{'/' + chain if chain.strip() else ''}"


def _bonded(topology: app.Topology) -> dict[int, set[int]]:
    nbrs: dict[int, set[int]] = {a.index: set() for a in topology.atoms()}
    for b in topology.bonds():
        nbrs[b[0].index].add(b[1].index)
        nbrs[b[1].index].add(b[0].index)
    return nbrs


def _is_ion(res) -> bool:
    atoms = list(res.atoms())
    return len(atoms) == 1 and atoms[0].element is not None and atoms[0].element not in (elem.hydrogen, elem.carbon)


def _cap_kind(res, nbrs, atoms_by_index) -> tuple[str, int] | None:
    """Recognise an ACE or NME cap written as an unknown residue.

    Returns ``(kind, index of the protein atom it is bonded to)`` or ``None``.
    """
    atoms = list(res.atoms())
    heavy = sorted(a.element.symbol for a in atoms if a.element is not None and a.element != elem.hydrogen)
    own = {a.index for a in atoms}
    external = [(a, j) for a in atoms for j in nbrs[a.index] if j not in own]
    if len(external) != 1:
        return None
    cap_atom, partner_index = external[0]
    partner = atoms_by_index[partner_index]
    if heavy == ["C", "C", "O"] and cap_atom.element == elem.carbon and partner.name == "N":
        return "ACE", partner_index
    if heavy == ["C", "N"] and cap_atom.element == elem.nitrogen and partner.name == "C":
        return "NME", partner_index
    return None


def _rename_cap(kind: str, atoms, nbrs) -> dict[int, str]:
    """Standard atom names for a recognised cap."""
    own = {a.index for a in atoms}
    names: dict[int, str] = {}
    if kind == "ACE":
        carbonyl = next(a for a in atoms if a.element == elem.carbon and any(j not in own for j in nbrs[a.index]))
        names[carbonyl.index] = "C"
        for a in atoms:
            if a.element == elem.oxygen:
                names[a.index] = "O"
            elif a.element == elem.carbon and a.index != carbonyl.index:
                names[a.index] = "CH3"
        hs = [a for a in atoms if a.element == elem.hydrogen]
        for k, a in enumerate(hs):
            names[a.index] = f"H{k + 1}"
    else:  # NME
        n_atom = next(a for a in atoms if a.element == elem.nitrogen)
        names[n_atom.index] = "N"
        c_atom = next(a for a in atoms if a.element == elem.carbon)
        names[c_atom.index] = "C"
        k = 1
        for a in atoms:
            if a.element == elem.hydrogen:
                if a.index in nbrs[n_atom.index]:
                    names[a.index] = "H"
                else:
                    names[a.index] = f"H{k}"
                    k += 1
    return names


#: hydrogens of the backbone; every other hydrogen of an amino acid is on its side chain
BACKBONE_HYDROGENS = {"H", "H1", "H2", "H3", "HN", "HA", "HA2", "HA3"}


def carboxyl_renames(res, positions_nm) -> dict[int, str]:
    """New names for the atoms of a protonated carboxyl group, so that the proton keeps the oxygen it
    is bonded to in the file. The force field puts it on one named oxygen (OD2 of ASH, OE2 of GLH):
    a proton given as HD1 on OD1 was otherwise rebuilt on OD2, 2.9 A away from where it was, and the
    donor of a hydrogen bond in the pocket changed sides (review of PR 1). The two oxygens of a
    carboxyl group are equivalent, so the one with the proton takes the name of the force field and
    the other takes the other name. Empty if the residue has no single carboxyl proton bonded to one
    of its oxygens."""
    spec = CARBOXYLS.get(res.name)
    if spec is None:
        return {}
    oxygens, target_o, target_h, h_names = spec
    atoms = {a.name: a for a in res.atoms()}
    protons = [atoms[name] for name in h_names if name in atoms]
    if len(protons) != 1 or not all(name in atoms for name in oxygens):
        return {}
    h = protons[0]
    pos = np.asarray(positions_nm)
    distance = {name: float(np.linalg.norm(pos[atoms[name].index] - pos[h.index])) for name in oxygens}
    bonded = min(distance, key=distance.get)
    if distance[bonded] > MAX_OH_BOND_NM:
        return {}
    out = {h.index: target_h}
    if bonded != target_o:
        out[atoms[bonded].index] = target_o
        out[atoms[target_o].index] = bonded
    return out


def infer_variant(res) -> str | None:
    """Protonation variant implied by the hydrogens already on a residue, or None where they do
    not decide it; ``Modeller.addHydrogens`` then assigns the state at the pH of the run.

    The hydrogens decide nothing when the side chain carries none (a file with hydrogens on the
    backbone only), when a histidine has neither HD1 nor HE2 (a ring without its proton is no
    state of the force field: HIN has no template in Amber14, review of PR 1), when a lysine has
    neither two nor three HZ, and when a cysteine without HG is not in a disulfide bond (CYX
    belongs to a bonded pair; addHydrogens makes it from the bonds of the topology).
    """
    names = {a.name for a in res.atoms() if a.element == elem.hydrogen}
    if not names - BACKBONE_HYDROGENS:
        return None
    base = _CANONICAL.get(res.name, res.name)
    if base == "HIS":
        d, e = "HD1" in names, "HE2" in names
        return "HIP" if d and e else "HID" if d else "HIE" if e else None
    if base == "ASP":
        return "ASH" if names & {"HD2", "HD1"} else "ASP"
    if base == "GLU":
        return "GLH" if names & {"HE2", "HE1"} else "GLU"
    if base == "LYS":
        hz = len([n for n in names if n.startswith("HZ")])
        return "LYS" if hz >= 3 else "LYN" if hz == 2 else None
    if base == "CYS":
        return "CYS" if "HG" in names else None
    return None


# --------------------------------------------------------------------------- main
#: the seed of PDBFixer's integrator when it places missing atoms and caps. OpenMM reads 0 as "a new seed
#: for every Context", which made the caps of one input land up to 1.8 A apart from one preparation to the
#: next (review of PR 1)
PREPARATION_SEED = 1


class _SeededPythonRandom:
    """Python's own random numbers seeded with ``seed`` while the block runs, and as they were after it:
    OpenMM's Modeller.addHydrogens starts every new hydrogen from random.random(), so the hydrogens of
    the caps of one input landed up to 1.9 A apart from one preparation to the next."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __enter__(self):
        import random

        self.state = random.getstate()
        random.seed(self.seed)
        return self

    def __exit__(self, *exc) -> None:
        import random

        random.setstate(self.state)


def _preparation_platform():
    """The platform of the small minimisations of the preparation (missing atoms, caps, hydrogens): the
    Reference platform, whose results do not depend on the GPU or the order of summation, so that one
    input gives one prepared protein."""
    import openmm as mm

    return mm.Platform.getPlatformByName("Reference")


def prepare_protein(pdb_path: str | Path, forcefield: app.ForceField | None = None, ph: float = 7.4,
                    keep_water: bool = True, seed: int = PREPARATION_SEED) -> PreparedProtein:
    """Prepare a protein PDB for simulation. See the module docstring for the rules. The result is
    the same every time for one input: the minimisations that place missing atoms, caps and
    hydrogens run on the Reference platform with the integrator seed ``seed`` (never 0)."""
    from pdbfixer import PDBFixer

    if int(seed) == 0:
        raise ValueError("the seed of the preparation must not be 0: OpenMM reads 0 as a new seed every time")
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise ProteinError(f"protein file not found: {pdb_path}")
    report = ProteinReport()
    forcefield = forcefield or app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    # ---- 1. read; replace non-standard residues PDBFixer knows (e.g. MSE -> MET) -------------
    fixer0 = PDBFixer(filename=str(pdb_path))
    fixer0.findNonstandardResidues()
    if fixer0.nonstandardResidues:
        report.warnings.append("non-standard residues replaced by their standard parent: "
                               + ", ".join(f"{_label(r)}->{new}" for r, new in fixer0.nonstandardResidues))
        fixer0.replaceNonstandardResidues()
    top, pos = fixer0.topology, np.asarray(fixer0.positions.value_in_unit(unit.nanometer), dtype=float)
    nbrs = _bonded(top)
    atoms_by_index = {a.index: a for a in top.atoms()}

    # ---- 2. classify residues ---------------------------------------------------------------
    kinds: dict[int, str] = {}                 # residue.index -> aa | cap | water | ion
    cap_info: dict[int, tuple[str, int]] = {}  # residue.index -> (ACE|NME, bonded protein atom index)
    unknown = []
    for res in top.residues():
        name = res.name.upper()
        if name in AMINO_ACIDS:
            kinds[res.index] = "aa"
        elif name in CAPS:
            kinds[res.index] = "cap"
        elif name in WATER:
            kinds[res.index] = "water"
        elif _is_ion(res):
            kinds[res.index] = "ion"
        else:
            cap = _cap_kind(res, nbrs, atoms_by_index)
            if cap is not None:
                kinds[res.index] = "cap"
                cap_info[res.index] = cap
            else:
                unknown.append(res)
    if unknown:
        names = ", ".join(sorted({_label(r) for r in unknown})[:12])
        raise ProteinError(
            f"{pdb_path.name}: residues that cannot be parameterized automatically: {names}. "
            "Remove them, or pass cofactors as SDF files through `system.cofactors`. "
            "The ligand must be given as an SDF file, not inside the PDB.")
    aa_res = [r for r in top.residues() if kinds[r.index] == "aa"]
    if not aa_res:
        raise ProteinError(f"{pdb_path.name}: no amino-acid residues found")
    n_with_h = sum(1 for r in aa_res if any(a.element == elem.hydrogen for a in r.atoms()))
    report.had_hydrogens = n_with_h > 0.5 * len(aa_res)
    report.notes.append("hydrogens found in the file: protonation states are taken from it" if report.had_hydrogens
                        else f"no hydrogens in the file: adding them at pH {ph}")

    # ---- 3. peptide segments: split chains at breaks -----------------------------------------
    def atom(res, name):
        return next((a for a in res.atoms() if a.name == name), None)

    segments: list[list] = []
    for chain in top.chains():
        current: list = []
        for res in chain.residues():
            if kinds[res.index] != "aa":
                continue
            if current:
                c_prev, n_cur = atom(current[-1], "C"), atom(res, "N")
                if c_prev is not None and n_cur is not None:
                    connected = np.linalg.norm(pos[c_prev.index] - pos[n_cur.index]) <= MAX_PEPTIDE_BOND_NM
                else:
                    # the C or the N of the bond is missing: PDBFixer rebuilds it (step 6) if the
                    # residues are neighbours, which their C-alpha atoms tell (review of PR 1: a
                    # missing N stopped the preparation as if the chain were broken there)
                    ca_prev, ca_cur = atom(current[-1], "CA"), atom(res, "CA")
                    connected = (ca_prev is not None and ca_cur is not None
                                 and np.linalg.norm(pos[ca_prev.index] - pos[ca_cur.index]) <= MAX_CA_CA_NM)
                    if connected:
                        report.notes.append(f"backbone atom missing between {_label(current[-1])} and {_label(res)}: "
                                            "rebuilt, the chain is not broken there")
                if not connected:
                    report.chain_breaks.append(f"{_label(current[-1])} | {_label(res)}")
                    segments.append(current)
                    current = []
            current.append(res)
        if current:
            segments.append(current)
    if report.chain_breaks:
        report.warnings.append("chain breaks (ends are capped, missing loops are not modelled): "
                               + "; ".join(report.chain_breaks))

    # existing caps (standard-named or recognised from unknown residues), keyed by protein residue
    ace_for: dict[int, object] = {}
    nme_for: dict[int, object] = {}
    for res in top.residues():
        if kinds[res.index] != "cap":
            continue
        if res.index in cap_info:
            kind, partner_index = cap_info[res.index]
        else:
            kind = "ACE" if res.name.upper() == "ACE" else "NME"
            own = {a.index for a in res.atoms()}
            partners = [j for a in res.atoms() for j in nbrs[a.index] if j not in own]
            if len(partners) != 1:
                raise ProteinError(f"cap residue {_label(res)} is not bonded to exactly one protein atom")
            partner_index = partners[0]
        (ace_for if kind == "ACE" else nme_for)[atoms_by_index[partner_index].residue.index] = res

    # ---- 4. decide what happens at every segment end ------------------------------------------
    drop_atoms: set[int] = set()
    add_ace: set[int] = set()     # protein residue.index that needs a new ACE before it
    add_nme: set[int] = set()
    no_oxt: set[int] = set()      # residues that must not get an OXT from PDBFixer
    for s, seg in enumerate(segments):
        first, last = seg[0], seg[-1]
        internal_start = s > 0 and seg[0].chain.index == segments[s - 1][-1].chain.index
        internal_end = s + 1 < len(segments) and seg[-1].chain.index == segments[s + 1][0].chain.index
        # N-terminus
        if first.index not in ace_for:
            n_atom = atom(first, "N")
            if n_atom is None:
                raise ProteinError(f"residue {_label(first)} has no backbone N")
            n_h = [j for j in nbrs[n_atom.index] if atoms_by_index[j].element == elem.hydrogen]
            full = 2 if first.name == "PRO" else 3
            if len(n_h) >= full:
                pass                                        # explicit charged N-terminus
            elif n_h or internal_start:
                drop_atoms.update(n_h)                      # hydrogen cap / lone amide H / break
                add_ace.add(first.index)
        # C-terminus
        if last.index not in nme_for:
            c_atom, oxt = atom(last, "C"), atom(last, "OXT")
            if c_atom is None:
                raise ProteinError(f"residue {_label(last)} has no backbone C")
            stray = [a.index for a in last.atoms() if a.name == "HXT"]
            c_h = [j for j in nbrs[c_atom.index] if atoms_by_index[j].element == elem.hydrogen]
            if oxt is not None:
                if stray:
                    drop_atoms.update(stray)
                    report.warnings.append(f"{_label(last)}: neutral COOH terminus made charged (HXT removed)")
            elif stray or c_h or internal_end or report.had_hydrogens:
                # no OXT in a file that carries hydrogens: not a charged terminus -> neutral cap
                drop_atoms.update(stray)
                drop_atoms.update(c_h)
                add_nme.add(last.index)
                no_oxt.add(last.index)
            # else: no hydrogens in file, chain end without OXT -> PDBFixer adds OXT (charged)
        else:
            no_oxt.add(last.index)

    # ---- 5. rebuild a clean PDB: one chain per segment, caps in sequence order ----------------
    new_top = app.Topology()
    new_pos: list = []
    old_to_new: dict[int, object] = {}
    cap_plan: dict[tuple[int, int], list[str]] = {}

    def copy_res(res, chain, name=None, rename=None, resid=None):
        # the insertion code goes along: without it the PDB written below is read back with two
        # residues of one number merged into one
        code = residue_key(res)[2] if resid is None else ""
        new_res = new_top.addResidue(name or res.name, chain, id=str(resid if resid is not None else res.id),
                                     insertionCode=code)
        for a in res.atoms():
            if a.index in drop_atoms:
                continue
            na = new_top.addAtom((rename or {}).get(a.index, a.name), a.element, new_res)
            old_to_new[a.index] = na
            new_pos.append(pos[a.index])
        return new_res

    if len(segments) > len(_CHAIN_IDS) - 1:
        raise ProteinError("too many protein segments for PDB chain identifiers")
    input_chains: dict[str, str] = {}
    for s, seg in enumerate(segments):
        chain = new_top.addChain(_CHAIN_IDS[s])
        input_chains[_CHAIN_IDS[s]] = seg[0].chain.id
        first, last = seg[0], seg[-1]
        n_in_chain = 0
        if first.index in ace_for:
            cap = ace_for[first.index]
            copy_res(cap, chain, "ACE", _rename_cap("ACE", list(cap.atoms()), nbrs), resid=_int_id(first.id) - 1)
            n_in_chain += 1
        elif first.index in add_ace:
            cap_plan[(s, 0)] = ["ACE"]
            report.caps_added.append(f"ACE before {_label(first)}")
        for res in seg:
            copy_res(res, chain, rename=carboxyl_renames(res, pos))
            n_in_chain += 1
        if last.index in nme_for:
            cap = nme_for[last.index]
            copy_res(cap, chain, "NME", _rename_cap("NME", list(cap.atoms()), nbrs), resid=_int_id(last.id) + 1)
        elif last.index in add_nme:
            cap_plan[(s, n_in_chain)] = ["NME"]
            report.caps_added.append(f"NME after {_label(last)}")
    # waters and ions keep their numbers in a chain for each chain of the input they come from: in one
    # chain two waters numbered 401 of chains A and B were read back as one (review of PR 1). Each such
    # chain is built in one piece, in the order in which its chain first appears: the residues of an
    # OpenMM chain must follow each other, and waters of chains A, B, A stopped the preparation (review
    # of PR 1)
    hetero: dict[str, list] = {}
    for res in top.residues():
        kind = kinds[res.index]
        if kind == "ion" or (kind == "water" and keep_water):
            hetero.setdefault(res.chain.id, []).append(res)
    if len(segments) + len(hetero) > len(_CHAIN_IDS):
        raise ProteinError("too many chains of protein segments, waters and ions for PDB chain identifiers")
    n_water = n_ion = 0
    for k, residues in enumerate(hetero.values()):
        chain = new_top.addChain(_CHAIN_IDS[len(segments) + k])
        for res in residues:
            kind = kinds[res.index]
            copy_res(res, chain, "HOH" if kind == "water" else res.name)
            n_water += kind == "water"
            n_ion += kind == "ion"
    if report.caps_added:
        report.notes.append("caps added: " + ", ".join(report.caps_added))
    report.notes.append(f"{len(aa_res)} residues in {len(segments)} segment(s); kept {n_water} waters and {n_ion} ions")

    buf = io.StringIO()
    app.PDBFile.writeFile(new_top, np.asarray(new_pos) * unit.nanometer, buf, keepIds=True)
    buf.seek(0)

    # ---- 6. PDBFixer: missing heavy atoms, caps, charged C-termini ----------------------------
    fixer = PDBFixer(pdbfile=buf, platform=_preparation_platform())
    fixer.missingResidues = dict(cap_plan)
    fixer.findMissingAtoms()
    capped_last = {
        (_CHAIN_IDS[s], *residue_key(seg[-1])[1:]) for s, seg in enumerate(segments) if seg[-1].index in no_oxt
    }
    fixer.missingTerminals = {r: a for r, a in fixer.missingTerminals.items() if residue_key(r) not in capped_last}
    rebuilt = [f"{_label(r)}:{'+'.join(a.name for a in atoms)}" for r, atoms in fixer.missingAtoms.items()]
    if rebuilt:
        report.warnings.append("missing heavy atoms rebuilt: " + ", ".join(rebuilt[:20])
                               + (" ..." if len(rebuilt) > 20 else ""))
    fixer.addMissingAtoms(seed=seed)

    # ---- 7. hydrogens: keep what is there, add what is missing ----------------------------------
    variants = [infer_variant(res) if res.name in AMINO_ACIDS else None for res in fixer.topology.residues()]
    # residues whose hydrogens were given but do not decide their state (infer_variant)
    undecided = {residue_key(r) for r, v in zip(fixer.topology.residues(), variants)
                 if v is None and r.name in AMINO_ACIDS and _CANONICAL.get(r.name, r.name) in INFERRED
                 and any(a.element == elem.hydrogen for a in r.atoms())}
    before = {(*residue_key(r), r.name): sum(1 for a in r.atoms() if a.element == elem.hydrogen)
              for r in fixer.topology.residues()}
    # Crystal waters get their hydrogens apart from the protein, from OpenMM's TIP3P template (the
    # geometry of every water model of cufep.prep.water): addHydrogens builds a System of what it is
    # given, and a water of a 4-site model given by its oxygen has no template before its hydrogens
    # and its extra site are there (review of PR 1). Their extra sites follow below.
    modeller = app.Modeller(fixer.topology, fixer.positions)
    waters = [r for r in modeller.topology.residues() if r.name == "HOH"]
    water_part = None
    if waters:
        water_part = app.Modeller(fixer.topology, fixer.positions)
        water_part.delete([r for r in water_part.topology.residues() if r.name != "HOH"])
        with _SeededPythonRandom(seed):
            water_part.addHydrogens(app.ForceField("amber14/tip3p.xml"), platform=_preparation_platform())
        modeller.delete(waters)
        variants = [v for r, v in zip(fixer.topology.residues(), variants) if r.name != "HOH"]
    with _SeededPythonRandom(seed):
        actual = modeller.addHydrogens(forcefield, pH=ph, variants=variants, platform=_preparation_platform())
    for res, variant, given in zip(modeller.topology.residues(), actual, variants):
        base = _CANONICAL.get(res.name, res.name)
        if given is None and base in TITRATABLE and (before.get((*residue_key(res), res.name), 0) == 0
                                                       or residue_key(res) in undecided):
            report.protonation_assigned.append(f"{_label(res)}={variant or res.name}")
    if report.protonation_assigned:
        his = [p for p in report.protonation_assigned if p.startswith("HIS")]
        report.notes.append(f"protonation assigned at pH {ph} for {len(report.protonation_assigned)} titratable "
                            f"residues (histidines: {', '.join(his) if his else 'none'})")

    if water_part is not None:
        modeller.add(water_part.topology, water_part.positions)
        # waters of a 4-site model need their extra site before their template can match (review of
        # PR 1: a crystal water stopped the preparation with amber14/tip4pew.xml); nothing is added
        # for a 3-site model
        modeller.addExtraParticles(forcefield)

    # ---- 8. check that every residue has a force-field template ----------------------------------
    unmatched = forcefield.getUnmatchedResidues(modeller.topology)
    if unmatched:
        raise ProteinError("no force-field template for: " + ", ".join(sorted({_label(r) for r in unmatched})[:12])
                           + ". Check termini, protonation and missing atoms of these residues.")
    return PreparedProtein(modeller.topology, modeller.positions, report, input_chains)


def _int_id(res_id: str) -> int:
    try:
        return int(res_id)
    except ValueError:
        return 0
