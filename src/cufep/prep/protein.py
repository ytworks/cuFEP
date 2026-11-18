"""Protein preparation from a bare PDB file.

Goal: accept "a protein PDB" as the only receptor input and return a Topology that the
force field can parameterize, changing as little as possible and recording every change.

Rules (see docs/PLAN.md, "入力契約"):

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
MAX_PEPTIDE_BOND_NM = 0.16
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

    def positions_nm(self) -> np.ndarray:
        return np.asarray(self.positions.value_in_unit(unit.nanometer), dtype=float)

    def write_pdb(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            app.PDBFile.writeFile(self.topology, self.positions, fh, keepIds=True)


# --------------------------------------------------------------------------- helpers
def _label(res) -> str:
    return f"{res.name}{res.id}{'/' + res.chain.id if res.chain.id.strip() else ''}"


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
                    names[a.index] = f"H{k}"; k += 1
    return names


def infer_variant(res) -> str | None:
    """Protonation variant implied by the hydrogens already on a residue (None = no hydrogens)."""
    names = {a.name for a in res.atoms()}
    if not any(a.element == elem.hydrogen for a in res.atoms()):
        return None
    base = _CANONICAL.get(res.name, res.name)
    if base == "HIS":
        d, e = "HD1" in names, "HE2" in names
        return "HIP" if d and e else "HID" if d else "HIE" if e else "HIN"
    if base == "ASP":
        return "ASH" if names & {"HD2", "HD1"} else "ASP"
    if base == "GLU":
        return "GLH" if names & {"HE2", "HE1"} else "GLU"
    if base == "LYS":
        return "LYS" if len([n for n in names if n.startswith("HZ")]) >= 3 else "LYN"
    if base == "CYS":
        return "CYS" if "HG" in names else "CYX"
    return None


# --------------------------------------------------------------------------- main
def prepare_protein(pdb_path: str | Path, forcefield: app.ForceField | None = None, ph: float = 7.4,
                    keep_water: bool = True, seed: int = 0) -> PreparedProtein:
    """Prepare a protein PDB for simulation. See the module docstring for the rules."""
    from pdbfixer import PDBFixer

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
                connected = (c_prev is not None and n_cur is not None
                             and np.linalg.norm(pos[c_prev.index] - pos[n_cur.index]) <= MAX_PEPTIDE_BOND_NM)
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
                drop_atoms.update(stray); drop_atoms.update(c_h)
                add_nme.add(last.index); no_oxt.add(last.index)
            # else: no hydrogens in file, chain end without OXT -> PDBFixer adds OXT (charged)
        else:
            no_oxt.add(last.index)

    # ---- 5. rebuild a clean PDB: one chain per segment, caps in sequence order ----------------
    new_top = app.Topology()
    new_pos: list = []
    old_to_new: dict[int, object] = {}
    cap_plan: dict[tuple[int, int], list[str]] = {}

    def copy_res(res, chain, name=None, rename=None, resid=None):
        new_res = new_top.addResidue(name or res.name, chain, id=str(resid if resid is not None else res.id))
        for a in res.atoms():
            if a.index in drop_atoms:
                continue
            na = new_top.addAtom((rename or {}).get(a.index, a.name), a.element, new_res)
            old_to_new[a.index] = na
            new_pos.append(pos[a.index])
        return new_res

    if len(segments) > len(_CHAIN_IDS) - 1:
        raise ProteinError("too many protein segments for PDB chain identifiers")
    for s, seg in enumerate(segments):
        chain = new_top.addChain(_CHAIN_IDS[s])
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
            copy_res(res, chain)
            n_in_chain += 1
        if last.index in nme_for:
            cap = nme_for[last.index]
            copy_res(cap, chain, "NME", _rename_cap("NME", list(cap.atoms()), nbrs), resid=_int_id(last.id) + 1)
        elif last.index in add_nme:
            cap_plan[(s, n_in_chain)] = ["NME"]
            report.caps_added.append(f"NME after {_label(last)}")
    hetero_chain = None
    n_water = n_ion = 0
    for res in top.residues():
        kind = kinds[res.index]
        if kind == "ion" or (kind == "water" and keep_water):
            if hetero_chain is None:
                hetero_chain = new_top.addChain(_CHAIN_IDS[len(segments)])
            copy_res(res, hetero_chain, "HOH" if kind == "water" else res.name)
            n_water += kind == "water"; n_ion += kind == "ion"
    if report.caps_added:
        report.notes.append("caps added: " + ", ".join(report.caps_added))
    report.notes.append(f"{len(aa_res)} residues in {len(segments)} segment(s); kept {n_water} waters and {n_ion} ions")

    buf = io.StringIO()
    app.PDBFile.writeFile(new_top, np.asarray(new_pos) * unit.nanometer, buf, keepIds=True)
    buf.seek(0)

    # ---- 6. PDBFixer: missing heavy atoms, caps, charged C-termini ----------------------------
    fixer = PDBFixer(pdbfile=buf)
    fixer.missingResidues = dict(cap_plan)
    fixer.findMissingAtoms()
    capped_last = {(_CHAIN_IDS[s], str(seg[-1].id)) for s, seg in enumerate(segments) if seg[-1].index in no_oxt}
    fixer.missingTerminals = {r: a for r, a in fixer.missingTerminals.items() if (r.chain.id, r.id) not in capped_last}
    rebuilt = [f"{_label(r)}:{'+'.join(a.name for a in atoms)}" for r, atoms in fixer.missingAtoms.items()]
    if rebuilt:
        report.warnings.append("missing heavy atoms rebuilt: " + ", ".join(rebuilt[:20])
                               + (" ..." if len(rebuilt) > 20 else ""))
    fixer.addMissingAtoms(seed=seed)

    # ---- 7. hydrogens: keep what is there, add what is missing ----------------------------------
    variants = [infer_variant(res) if res.name in AMINO_ACIDS else None for res in fixer.topology.residues()]
    before = {(r.chain.id, r.id, r.name): sum(1 for a in r.atoms() if a.element == elem.hydrogen)
              for r in fixer.topology.residues()}
    modeller = app.Modeller(fixer.topology, fixer.positions)
    actual = modeller.addHydrogens(forcefield, pH=ph, variants=variants)
    for res, variant, given in zip(modeller.topology.residues(), actual, variants):
        base = _CANONICAL.get(res.name, res.name)
        if given is None and base in TITRATABLE and before.get((res.chain.id, res.id, res.name), 0) == 0:
            report.protonation_assigned.append(f"{res.name}{res.id}/{res.chain.id}={variant or res.name}")
    if report.protonation_assigned:
        his = [p for p in report.protonation_assigned if p.startswith("HIS")]
        report.notes.append(f"protonation assigned at pH {ph} for {len(report.protonation_assigned)} titratable "
                            f"residues (histidines: {', '.join(his) if his else 'none'})")

    # ---- 8. check that every residue has a force-field template ----------------------------------
    unmatched = forcefield.getUnmatchedResidues(modeller.topology)
    if unmatched:
        raise ProteinError("no force-field template for: " + ", ".join(sorted({_label(r) for r in unmatched})[:12])
                           + ". Check termini, protonation and missing atoms of these residues.")
    return PreparedProtein(modeller.topology, modeller.positions, report)


def _int_id(res_id: str) -> int:
    try:
        return int(res_id)
    except ValueError:
        return 0
