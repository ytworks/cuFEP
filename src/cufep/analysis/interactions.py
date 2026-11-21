"""The interactions between protein and ligand in the frames of an MD, by ProLIF.

``restraints.selection: interactions`` chooses the anchors of the Boresch restraint from the
interactions that last through an unrestrained MD of the complex (Ding et al., arXiv:2603.22274
v3, sec. 2.1: "ProLIF is used to identify protein-ligand interactions"). This module finds them,
frame by frame; :mod:`cufep.alchemy.interaction_anchors` counts them and chooses.

What ProLIF is given, as in the source of that work (``felis/external/prolif_tools.py`` at commit
d6dff91) unless said otherwise:

* the interactions of :data:`cufep.alchemy.interaction_anchors.INTERACTION_WEIGHTS` (hydrogen
  bond, ionic, halogen bond, cation-pi, aromatic stacking; each as ProLIF names its two
  directions), every occurrence of each (``count=True``), for the residues within 6.0 A of the
  ligand, with ProLIF's own geometric parameters. The source patches ProLIF 2.0.3 so that a
  charged atom bonded to an atom of the opposite charge (a nitro group) is no ion; the ProLIF of
  the image (``docker/environment.yml``) has that in its own patterns;
* the whole protein, its chemistry perceived by MDAnalysis from the atoms and bonds of the
  topology (all hydrogens are there). Its residues are numbered by their index in the topology,
  not by their names and numbers in the input: two residues of one number (insertion codes, two
  chains) stay two residues (the source keys them by name and number);
* the ligand with the chemistry of the input SDF, as the molecule that was parametrised has it,
  and RDKit's model of aromaticity, which ProLIF's patterns are written for (the molecule of the
  preparation carries the MDL model, in which the atoms of a five-membered ring are not
  aromatic). The source reads the SDF again for the same purpose.

The molecules are made once and take the coordinates of each frame (:class:`InteractionFinder`);
a test holds this to ProLIF's own run over a trajectory, which makes them again for every frame.

The coordinates must be those of neighbours: whole molecules with the ligand in the periodic
image next to the protein (:func:`cufep.alchemy.restraints.ligand_next_to_protein`). ProLIF
knows no box.
"""
from __future__ import annotations

import numpy as np

from cufep.alchemy.interaction_anchors import INTERACTION_WEIGHTS

#: the interactions that are looked for, as ProLIF names them
INTERACTIONS = tuple(INTERACTION_WEIGHTS)
#: the residues with an atom within this distance of the ligand are looked at (A)
VICINITY_CUTOFF_A = 6.0


def packages_problem() -> str | None:
    """Why the interactions of an MD cannot be found here, or None: ProLIF and MDAnalysis are
    packages of the official image, not of every environment."""
    missing = []
    for name in ("prolif", "MDAnalysis"):
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - a package that is there and cannot be loaded is as missing
            missing.append(f"{name} ({type(exc).__name__}: {str(exc)[:80]})")
    if missing:
        return ("restraints.selection: interactions needs ProLIF and MDAnalysis, which cannot be imported here: "
                + ", ".join(missing) + ". They are in the official image (docker/environment.yml); or use "
                "restraints.selection: auto")
    return None


def versions() -> dict[str, str]:
    """The versions of the packages that found the interactions, for the record of an analysis."""
    import MDAnalysis
    import prolif
    import rdkit

    return {"prolif": str(prolif.__version__), "MDAnalysis": str(MDAnalysis.__version__),
            "rdkit": str(rdkit.__version__)}


def with_rdkit_aromaticity(rdmol):
    """A copy of the ligand with aromaticity as RDKit perceives it. The atoms, their order, the
    bonds, the charges and the coordinates are those of ``rdmol``."""
    from rdkit import Chem

    mol = Chem.Mol(rdmol)
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
    return mol


def protein_atoms_of(topology, among=None) -> np.ndarray:
    """The atoms of the protein (amino-acid residues and caps), those of ``among`` if given."""
    from cufep.prep.protein import AMINO_ACIDS, CAPS

    allowed = None if among is None else {int(i) for i in among}
    return np.array([a.index for a in topology.atoms()
                     if (a.residue.name in AMINO_ACIDS or a.residue.name in CAPS)
                     and (allowed is None or a.index in allowed)], dtype=int)


def _protein_topology(topology, protein_atoms):
    """(a topology of the protein atoms alone, the index in ``topology`` of each of its residues)."""
    import openmm.app as app

    keep = {int(i) for i in protein_atoms}
    sub = app.Topology()
    chains, residues, atoms, residue_index = {}, {}, {}, []
    for atom in topology.atoms():
        if atom.index not in keep:
            continue
        residue = atom.residue
        if residue.chain.index not in chains:
            chains[residue.chain.index] = sub.addChain(residue.chain.id)
        if residue.index not in residues:
            # its number is its index in the whole topology: the names and numbers of the input can repeat
            residues[residue.index] = sub.addResidue(residue.name, chains[residue.chain.index], str(residue.index))
            residue_index.append(residue.index)
        atoms[atom.index] = sub.addAtom(atom.name, atom.element, residues[residue.index])
    for bond in topology.bonds():
        if bond[0].index in atoms and bond[1].index in atoms:
            sub.addBond(atoms[bond[0].index], atoms[bond[1].index])
    return sub, residue_index


class InteractionFinder:
    """The molecules of ProLIF for one complex, made once; :meth:`frame` gives the interactions of
    one set of coordinates."""

    def __init__(self, topology, ligand_atoms, rdmol, protein_atoms=None, protein_nm=None) -> None:
        import MDAnalysis as mda
        import prolif

        self.ligand_atoms = np.asarray(ligand_atoms, dtype=int)
        # in the order of the topology, which is the order of the atoms of the molecule made below
        self.protein_atoms = np.sort(protein_atoms_of(topology) if protein_atoms is None
                                     else np.asarray(protein_atoms, dtype=int))
        if len(self.protein_atoms) == 0:
            raise ValueError("the topology holds no protein atoms: there are no interactions to look for")
        sub, residue_index = _protein_topology(topology, self.protein_atoms)
        # the chemistry is read from the atoms and bonds; MDAnalysis gives a molecule coordinates only if
        # it has some that are not all zero, so it gets those of a frame, or a place holder
        first = np.arange(3 * len(self.protein_atoms), dtype=np.float32).reshape(-1, 3) if protein_nm is None \
            else np.asarray(protein_nm, dtype=np.float32) * 10.0
        universe = mda.Universe(sub, first[None], format="MEMORY")
        universe.residues.resids = np.asarray(residue_index, dtype=int)
        self.protein = prolif.Molecule.from_mda(universe.atoms)
        order = [atom.GetIntProp("_MDAnalysis_index") for atom in self.protein.GetAtoms()]
        if order != list(range(len(self.protein_atoms))):
            raise RuntimeError("the protein molecule of ProLIF does not keep the order of the atoms it was given")
        self.ligand = prolif.Molecule.from_rdkit(with_rdkit_aromaticity(rdmol), resname="LIG", resnumber=1, chain="")
        if self.ligand.GetNumAtoms() != len(self.ligand_atoms):
            raise ValueError(f"the ligand molecule has {self.ligand.GetNumAtoms()} atoms and the ligand of the "
                             f"System {len(self.ligand_atoms)}")
        self._protein_parts = self._parts(self.protein)
        self._ligand_parts = self._parts(self.ligand)
        self.fingerprint = prolif.Fingerprint(list(INTERACTIONS), count=True, vicinity_cutoff=VICINITY_CUTOFF_A)

    @staticmethod
    def _parts(molecule) -> list[np.ndarray]:
        """For every residue of a molecule of ProLIF, which atoms of the molecule its atoms are:
        ProLIF keeps a molecule and a copy of each of its residues, each with coordinates of its own."""
        return [np.array([atom.GetUnsignedProp("mapindex") for atom in residue.GetAtoms()], dtype=int)
                for residue in molecule.residues.values()]

    @staticmethod
    def _place(molecule, parts, xyz_A: np.ndarray) -> None:
        molecule.GetConformer().SetPositions(xyz_A)
        for residue, part in zip(molecule.residues.values(), parts):
            residue.GetConformer().SetPositions(xyz_A[part])

    def frame(self, protein_nm: np.ndarray, ligand_nm: np.ndarray) -> list[tuple[int, int, str]]:
        """The interactions of one frame as ``(ligand atom, residue, type)``: the first ligand atom
        of the interaction as its number in the ligand's molecule (for a hydrogen bond that the
        ligand donates the donor, not its hydrogen), the residue as its index in the topology, the
        type as ProLIF names it; one entry for every occurrence. ``protein_nm`` and ``ligand_nm``
        are the positions of :attr:`protein_atoms` and :attr:`ligand_atoms` in nm."""
        self._place(self.protein, self._protein_parts, np.ascontiguousarray(protein_nm, dtype=float) * 10.0)
        self._place(self.ligand, self._ligand_parts, np.ascontiguousarray(ligand_nm, dtype=float) * 10.0)
        found = self.fingerprint.generate(self.ligand, self.protein, metadata=True)
        return [(int(occurrence["parent_indices"]["ligand"][0]), int(residue.number), str(name))
                for (_ligand, residue), by_type in found.items()
                for name, occurrences in by_type.items() for occurrence in occurrences]


def interactions_of_frames(topology, frames, frame_atoms, ligand_atoms, rdmol) -> list[list[tuple[int, int, str]]]:
    """The interactions of every frame (:meth:`InteractionFinder.frame`). ``frames`` (frames x
    frame atoms x 3, nm) hold the atoms ``frame_atoms`` (numbers of the System) as whole molecules
    with the ligand next to the protein; ``ligand_atoms`` are the ligand's atoms in the order of
    ``rdmol``."""
    row = {int(a): i for i, a in enumerate(np.asarray(frame_atoms, dtype=int))}
    xyz = np.asarray(frames)
    among = protein_atoms_of(topology, among=row)
    finder = InteractionFinder(topology, ligand_atoms, rdmol, among,
                               protein_nm=xyz[0, [row[int(a)] for a in among], :] if len(xyz) else None)
    protein = [row[int(a)] for a in finder.protein_atoms]
    ligand = [row[int(a)] for a in finder.ligand_atoms]
    return [finder.frame(xyz[f, protein, :], xyz[f, ligand, :]) for f in range(xyz.shape[0])]


def occupancies(records, n_frames: int | None = None) -> list[dict]:
    """How often each residue has each interaction with the ligand, whichever ligand atom it is
    with: ``[{residue, interaction, frames, fraction}]``, most frequent first. A number of the
    analysis, without a verdict."""
    n = len(records) if n_frames is None else int(n_frames)
    counts: dict[tuple[int, str], int] = {}
    for frame in records:
        for residue, kind in {(int(r), str(k)) for _atom, r, k in frame}:
            counts[(residue, kind)] = counts.get((residue, kind), 0) + 1
    return [{"residue": residue, "interaction": kind, "frames": int(count),
             "fraction": float(count / n) if n else 0.0}
            for (residue, kind), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
