"""Hybrid topology of two ligands, for relative binding free energies.

The energy of every state is that of the HybridTopologyFactory of OpenFE's relative protocol
(openfe 1.13, ``openfe/protocols/openmm_rfe/_rfe_utils/relative.py``, from perses) with
``interpolate_old_and_new_14s=True`` (OpenFE's ``turn_off_core_unique_exceptions``, its default since
1.11) and the soft-core Lennard-Jones of Gapsys, Seeliger and de Groot (J. Chem. Theory Comput. 2012,
8, 2373) with alpha 0.85 (OpenFE's ``softcore_LJ: gapsys``, ``softcore_alpha``).

Atom classes (OpenFE's words). The ligand of state A is the *old* ligand, that of state B the *new* one.
    environment   everything that is not a ligand atom: the same atoms, with the same parameters, in both
    core          ligand atoms mapped onto each other: the parameters of A at lambda 0, those of B at 1
    unique old    ligand atoms of A that are not mapped: interacting at lambda 0, not at 1
    unique new    ligand atoms of B that are not mapped: not interacting at lambda 0, interacting at 1

The hybrid System has the atoms of System A in their order, then the unique new atoms.

Bonded terms (OpenFE's rules)
    * bonds and angles of core atoms only: force constant and length (angle) interpolated linearly with
      ``lambda_bonds`` (``lambda_angles``); a term that exists in one end state only has the force
      constant 0 in the other, with the length of the one it has;
    * torsions of core atoms only: a term present in both end states (same atoms, periodicity, phase and
      force constant) stays as it is; one present in one end state only goes off (or on) linearly with
      ``lambda_torsions``;
    * every term with a unique atom, and every environment term, stays on at full strength throughout:
      the unique atoms are dummies that keep their bonded terms at the end where they do not interact,
      in both legs alike.

Nonbonded terms (OpenFE's rules)
    * charges: core q_A + lambda_electrostatics_core (q_B - q_A); unique old q_A (1 -
      lambda_electrostatics_delete); unique new q_B lambda_electrostatics_insert;
    * Lennard-Jones with the environment and between ligand atoms: epsilon and sigma of a pair
      interpolated linearly between its values in A and in B (Lorentz-Berthelot mixing of the
      parameters of each end; a unique atom has epsilon 0 at the end where it does not exist) with
      ``lambda_sterics_core`` for a pair of core (and environment) atoms, ``lambda_sterics_delete`` for a
      pair with a unique old atom and ``lambda_sterics_insert`` for one with a unique new atom; for the
      last two the soft-core of Gapsys: below r_LJ = alpha ((26/7) sigma^6 lambda')^(1/6), with lambda'
      the lambda_sterics_delete of an old pair and 1 - lambda_sterics_insert of a new one, the potential is
      continued as OpenFE writes it (``_nonbonded_custom``);
    * pairs inside the ligand that are exceptions of the force field (1-4 pairs and those cuFEP's force
      field excludes): a core pair has its charge product and its sigma and epsilon interpolated linearly
      with lambda_electrostatics_core and lambda_sterics_core (the values of an end state where the pair is
      not an exception are those of the normal pair); a pair with a unique old atom has charge product
      times (1 - lambda_electrostatics_delete) and its Lennard-Jones switched off by
      lambda_sterics_delete with the soft-core; a pair with a unique new atom the same with insert;
    * other pairs inside the ligand: the product of the two charges, Lennard-Jones as above;
    * a unique old and a unique new atom never interact.

Layout (cuFEP's own, so that the cuPotential plugin, which replaces only the ``NonbondedForce``, carries
only charges; ``docs/plans/2026-11-25-rbfe.md``, ``spikes/m36``):
    ``NonbondedForce`` (group 0)
        the environment as in System A; every ligand atom with its charge through particle parameter
        offsets under the three electrostatic lambdas and no Lennard-Jones; every pair of ligand atoms an
        exception without charge and Lennard-Jones.
    ``CustomBondForce`` "cufep_ligand_coulomb" (group 0)
        the Coulomb energy of every pair of ligand atoms (unique old x unique new excepted) as the rules
        give it. No cutoff: the pair at its nearest image, and the NonbondedForce keeps its other images
        in reciprocal space, so that this is the Ewald energy of the pair, as in the ABFE
        (``docs/protocol.md``, "Interactions inside the ligand").
    ``CustomNonbondedForce`` "cufep_hybrid_lj" (group 1)
        Lennard-Jones of the ligand with the environment, OpenFE's expression, interaction groups (core,
        environment), (unique old, environment), (unique new, environment).
    ``CustomBondForce`` "cufep_ligand_lj" (group 5)
        Lennard-Jones of every pair of ligand atoms (unique old x unique new excepted), no cutoff (OpenFE
        cuts the pairs that are not exceptions at the cutoff; the ABFE keeps them all, ``docs/protocol.md``).
    core bonds, angles and improper torsions (group 4), core proper torsions (group 5).

Groups: 0 is an exact quadratic in the charges, so in each half of lambda an exact quadratic in lambda
(and with REST2 in (lambda sqrt(s), sqrt(s))); 1, 4 and 5 are evaluated per lambda value
(:mod:`cufep.rbfe.states`).

REST2 (Wang, Friesner and Berne 2011) as in the ABFE (``cufep.alchemy.factory``): the hot region is the
whole hybrid ligand and the environment atoms ``rest2_atoms``. Charges of hot atoms carry sqrt(s): the
three electrostatic lambdas are lambda sqrt(s) (the ABFE folds sqrt(s) into lambda_electrostatics the same
way) and the base charges of the ligand carry an offset under rest2_dq; the Lennard-Jones of the ligand
with the environment carries sqrt(s) per hot atom; the pairs inside the ligand and the proper torsions of
the hot region carry s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import openmm as mm
import openmm.unit as unit

from cufep.alchemy.factory import (
    MAIN_GROUP,
    ONE_4PI_EPS0,
    REST2_DQ,
    REST2_DS,
    SOFTCORE_GROUP,
    AlchemyError,
    _move_hot_torsions,
)

LAMBDA_ELECTROSTATICS_CORE = "lambda_electrostatics_core"
LAMBDA_ELECTROSTATICS_DELETE = "lambda_electrostatics_delete"
LAMBDA_ELECTROSTATICS_INSERT = "lambda_electrostatics_insert"
LAMBDA_STERICS_CORE = "lambda_sterics_core"
LAMBDA_STERICS_INSERT = "lambda_sterics_insert"
LAMBDA_STERICS_DELETE = "lambda_sterics_delete"
LAMBDA_BONDS = "lambda_bonds"
LAMBDA_ANGLES = "lambda_angles"
LAMBDA_TORSIONS = "lambda_torsions"
ELECTROSTATIC_LAMBDAS = (LAMBDA_ELECTROSTATICS_CORE, LAMBDA_ELECTROSTATICS_DELETE, LAMBDA_ELECTROSTATICS_INSERT)
#: the electrostatic lambdas of the charges that REST2 does not scale (the cold atoms of a ligand that is hot only in
#: part, the alchemical water): those of the hot atoms carry sqrt(s), these do not
COLD_ELECTROSTATIC_LAMBDAS = tuple(name + "_cold" for name in ELECTROSTATIC_LAMBDAS)
STERIC_LAMBDAS = (LAMBDA_STERICS_CORE, LAMBDA_STERICS_INSERT, LAMBDA_STERICS_DELETE)
BONDED_LAMBDAS = (LAMBDA_BONDS, LAMBDA_ANGLES, LAMBDA_TORSIONS)

#: OpenFE's default (``softcore_alpha``, Gapsys et al. 2012)
SOFTCORE_ALPHA = 0.85
#: groups of the hybrid System beyond those of the ABFE (cufep.alchemy.factory)
HYBRID_GROUP = 4          # core bonds, angles and improper torsions (not scaled by REST2)
HYBRID_HOT_GROUP = 5      # Lennard-Jones inside the ligand and core proper torsions (times s with REST2)

HYBRID_LJ_FORCE_NAME = "cufep_hybrid_lj"
LIGAND_COULOMB_FORCE_NAME = "cufep_ligand_coulomb"
LIGAND_LJ_FORCE_NAME = "cufep_ligand_lj"
CORE_BOND_FORCE_NAME = "cufep_core_bonds"
CORE_ANGLE_FORCE_NAME = "cufep_core_angles"
CORE_TORSION_FORCE_NAME = "cufep_core_torsions"
CORE_PROPER_TORSION_FORCE_NAME = "cufep_core_proper_torsions"

_E = unit.elementary_charge
_NM = unit.nanometer
_KJ = unit.kilojoule_per_mole
_RAD = unit.radian

#: OpenFE's soft-core Lennard-Jones (``_nonbonded_custom(v2=True)`` and ``_nonbonded_custom_sterics_common``),
#: with ``lambda_sterics`` and the flags of a pair defined by the caller
_GAPSYS = (
    "U_sterics = select(step(r - r_LJ), 4*epsilon*x*(x-1.0), U_sterics_quad);"
    "U_sterics_quad = Force*(((r - r_LJ)^2)/2 - (r - r_LJ)) + U_sterics_cut;"
    "U_sterics_cut = 4*epsilon*((sigma/r_LJ)^6)*(((sigma/r_LJ)^6) - 1.0);"
    "Force = -4*epsilon*((-12*sigma^12)/(r_LJ^13) + (6*sigma^6)/(r_LJ^7));"
    "x = (sigma/r)^6;"
    "r_LJ = softcore_alpha*((26/7)*(sigma^6)*lambda_sterics_deprecated)^(1/6);"
    "lambda_sterics_deprecated = new_interaction*(1.0 - lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
    "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
    "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
    "lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert"
    " + old_interaction*lambda_sterics_delete;"
)
#: the ligand with the environment: per-particle parameters and the mixing of each end (OpenFE's)
HYBRID_LJ_ENERGY = (
    "U_sterics;" + _GAPSYS
    + "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);"
      "new_interaction = max(unique_new1, unique_new2); old_interaction = max(unique_old1, unique_old2);"
      "epsilonA = sqrt(epsilonA1*epsilonA2); epsilonB = sqrt(epsilonB1*epsilonB2);"
      "sigmaA = 0.5*(sigmaA1 + sigmaA2); sigmaB = 0.5*(sigmaB1 + sigmaB2);"
)
#: the same with the REST2 factor of each atom
HYBRID_LJ_ENERGY_REST2 = "(1+hot1*rest2_dq)*(1+hot2*rest2_dq)*" + HYBRID_LJ_ENERGY
#: a pair inside the ligand: parameters of the pair at each end, and its flags
LIGAND_LJ_ENERGY = (
    "U_sterics;" + _GAPSYS
    + "core_interaction = delta(unique_old+unique_new); new_interaction = unique_new; old_interaction = unique_old;"
)
_CHARGE = "q{k}0{dq} + q{k}c*" + LAMBDA_ELECTROSTATICS_CORE + " + q{k}d*" + LAMBDA_ELECTROSTATICS_DELETE \
    + " + q{k}i*" + LAMBDA_ELECTROSTATICS_INSERT


def ligand_coulomb_energy_by_atom() -> str:
    """Coulomb energy of a pair inside a ligand that is hot only in part: the charges under the unscaled
    electrostatic lambdas, each times sqrt(s) if its atom is hot (``h1``, ``h2``)."""
    cold = dict(zip(ELECTROSTATIC_LAMBDAS, COLD_ELECTROSTATIC_LAMBDAS))
    qq = ("(qq0 + qqc*" + cold[LAMBDA_ELECTROSTATICS_CORE] + " + qqd*" + cold[LAMBDA_ELECTROSTATICS_DELETE]
          + " + qqi*" + cold[LAMBDA_ELECTROSTATICS_INSERT] + ")*(1+h1*rest2_dq)*(1+h2*rest2_dq)")

    def charge(k: int) -> str:
        return (f"(q{k}0 + q{k}c*{cold[LAMBDA_ELECTROSTATICS_CORE]} + q{k}d*{cold[LAMBDA_ELECTROSTATICS_DELETE]}"
                f" + q{k}i*{cold[LAMBDA_ELECTROSTATICS_INSERT]})*(1+h{k}*rest2_dq)")

    return (f"{ONE_4PI_EPS0!r}*(product*q1*q2 + (1-product)*qq)/r;"
            f"qq = {qq}; q1 = {charge(1)}; q2 = {charge(2)}")


def ligand_coulomb_energy(rest2: bool) -> str:
    """Coulomb energy of a pair inside the ligand: ``product`` 1 for the product of the two charges, 0
    for the charge product of an exception, both as offsets under the electrostatic lambdas."""
    dq = "*(1+rest2_dq)" if rest2 else ""
    qq = ("(qq0" + dq + " + qqc*" + LAMBDA_ELECTROSTATICS_CORE + " + qqd*" + LAMBDA_ELECTROSTATICS_DELETE
          + " + qqi*" + LAMBDA_ELECTROSTATICS_INSERT + ")" + dq)
    return (f"{ONE_4PI_EPS0!r}*(product*q1*q2 + (1-product)*qq)/r;"
            f"qq = {qq};"
            f"q1 = {_CHARGE.format(k=1, dq=dq)};"
            f"q2 = {_CHARGE.format(k=2, dq=dq)}")


CORE_BOND_ENERGY = ("(K/2)*(r-length)^2; K = (1-lambda_bonds)*K1 + lambda_bonds*K2;"
                    "length = (1-lambda_bonds)*length1 + lambda_bonds*length2")
CORE_ANGLE_ENERGY = ("(K/2)*(theta-theta0)^2; K = (1.0-lambda_angles)*K_1 + lambda_angles*K_2;"
                     "theta0 = (1.0-lambda_angles)*theta0_1 + lambda_angles*theta0_2")
CORE_TORSION_ENERGY = ("(1-lambda_torsions)*U1 + lambda_torsions*U2; U1 = K1*(1+cos(periodicity1*theta-phase1));"
                       "U2 = K2*(1+cos(periodicity2*theta-phase2))")

_KNOWN_FORCES = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce",
                 "CMMotionRemover"}


@dataclass
class HybridSystem:
    system: mm.System
    #: positions of the hybrid System (nm): those of A, then the unique new atoms where B has them
    positions: np.ndarray
    core: tuple[int, ...]
    unique_old: tuple[int, ...]
    unique_new: tuple[int, ...]
    #: atom of System A (B) -> atom of the hybrid System
    a_to_hybrid: dict[int, int]
    b_to_hybrid: dict[int, int]
    #: (charge e, sigma nm, epsilon kJ/mol) of the atoms of ligand A (B), by hybrid index
    parameters_a: dict[int, tuple[float, float, float]]
    parameters_b: dict[int, tuple[float, float, float]]
    softcore_alpha: float = SOFTCORE_ALPHA
    rest2: bool = False
    rest2_atoms: tuple[int, ...] = ()
    rest2_torsions: int = 0
    #: environment atoms whose parameters go from those of A to those of B (the water that is an ion in B,
    #: cufep.rbfe.charge)
    alchemical_water: tuple[int, ...] = ()
    #: the atoms of the ligand in the hot region of REST2 (all of them unless region: perturbed)
    rest2_ligand_atoms: tuple[int, ...] = ()
    #: pairs of bonded atoms of the hybrid System (bonds and constraints of both end states)
    bonds: frozenset = field(default_factory=frozenset)

    @property
    def ligand_atoms(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.core) | set(self.unique_old) | set(self.unique_new)))

    @property
    def atoms_a(self) -> tuple[int, ...]:
        """The atoms of ligand A in the hybrid System."""
        return tuple(sorted(set(self.core) | set(self.unique_old)))

    @property
    def atoms_b(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.core) | set(self.unique_new)))


def _forces(system: mm.System, name: str) -> dict:
    forces = {}
    for force in system.getForces():
        kind = type(force).__name__
        if kind not in _KNOWN_FORCES:
            raise AlchemyError(f"System {name} has a {kind}, which the hybrid topology does not take")
        if kind in forces:
            raise AlchemyError(f"System {name} has more than one {kind}")
        forces[kind] = force
    if "NonbondedForce" not in forces:
        raise AlchemyError(f"System {name} has no NonbondedForce")
    return forces


def _nonbonded_terms(nb: mm.NonbondedForce) -> tuple[list, dict]:
    particles = []
    for i in range(nb.getNumParticles()):
        q, s, e = nb.getParticleParameters(i)
        particles.append((q.value_in_unit(_E), s.value_in_unit(_NM), e.value_in_unit(_KJ)))
    exceptions = {}
    for k in range(nb.getNumExceptions()):
        i, j, qq, s, e = nb.getExceptionParameters(k)
        exceptions[(min(i, j), max(i, j))] = (qq.value_in_unit(_E * _E), s.value_in_unit(_NM), e.value_in_unit(_KJ))
    return particles, exceptions


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _same_environment_term(what: str, key, in_a, in_b) -> None:
    """The bonded terms of the environment are those of A in the hybrid; B must have the same ones."""
    if in_a is None:
        raise AlchemyError(f"the environment {what} {key} is in B but not in A")
    if not np.allclose(in_a, in_b, rtol=1e-9, atol=0.0):
        raise AlchemyError(f"the environment {what} {key} has other parameters in B than in A: {in_b} and {in_a}")


def make_hybrid_system(system_a: mm.System, positions_a, ligand_a, system_b: mm.System, positions_b, ligand_b,
                       core_map: dict[int, int], softcore_alpha: float = SOFTCORE_ALPHA, rest2: bool = False,
                       rest2_atoms=(), alchemical_water=(), rest2_ligand_atoms=None) -> HybridSystem:
    """The hybrid System of two end states.

    ``system_a`` and ``system_b`` hold the same environment, atom for atom in the same order, and the
    ligands ``ligand_a`` and ``ligand_b`` (atom indices of each System); ``core_map`` maps atoms of ligand A
    to atoms of ligand B (indices of the Systems). ``positions_*`` are in nm; the hybrid has the positions
    of A and, for the unique new atoms, those of B. ``rest2_atoms`` are environment atoms (indices of A,
    which are those of the hybrid) of the hot region; the ligand is always in it. ``alchemical_water`` are
    environment atoms (indices of A) whose nonbonded parameters differ in B, the water that is an ion in B
    (:mod:`cufep.rbfe.charge`): their charges go with ``lambda_electrostatics_core`` (offsets in the
    NonbondedForce), their Lennard-Jones with everything with ``lambda_sterics_core`` (the force of the ligand
    with its environment; none in the NonbondedForce), as those of a core atom. ``rest2_ligand_atoms`` (hybrid
    indices; None: the whole ligand) are the atoms of the ligand in the hot region of REST2 (``region: perturbed``,
    :mod:`cufep.rbfe.hot_region`); the others are cold: their charges go with electrostatic lambdas that do not carry
    sqrt(s), and every pair or torsion of the ligand carries the factor of its hot atoms (s for two, sqrt(s) for one).
    The charges of the alchemical water are cold too. With the whole ligand hot and no alchemical water the System is
    the one of before.
    """
    lig_a, lig_b = sorted(int(i) for i in ligand_a), sorted(int(i) for i in ligand_b)
    set_a, set_b = set(lig_a), set(lig_b)
    core_map = {int(i): int(j) for i, j in core_map.items()}
    if not core_map:
        raise AlchemyError("the atom mapping is empty")
    if not set(core_map) <= set_a or not set(core_map.values()) <= set_b:
        raise AlchemyError("the atom mapping has atoms that are not atoms of the ligands")
    if len(set(core_map.values())) != len(core_map):
        raise AlchemyError("the atom mapping maps two atoms of A onto one atom of B")
    env_a = [i for i in range(system_a.getNumParticles()) if i not in set_a]
    env_b = [i for i in range(system_b.getNumParticles()) if i not in set_b]
    if len(env_a) != len(env_b):
        raise AlchemyError(f"the environments differ: {len(env_a)} atoms in A, {len(env_b)} in B")
    forces_a, forces_b = _forces(system_a, "A"), _forces(system_b, "B")
    nb_a, nb_b = forces_a["NonbondedForce"], forces_b["NonbondedForce"]
    if nb_a.getNonbondedMethod() not in (mm.NonbondedForce.PME, mm.NonbondedForce.LJPME):
        raise AlchemyError("the NonbondedForce must use PME")
    if nb_a.getUseSwitchingFunction():
        raise AlchemyError("a switching function is not supported")
    particles_a, exceptions_a = _nonbonded_terms(nb_a)
    particles_b, exceptions_b = _nonbonded_terms(nb_b)

    # ---- atoms of the hybrid ------------------------------------------------------------
    unique_old = [i for i in lig_a if i not in core_map]
    unique_b = [j for j in lig_b if j not in set(core_map.values())]
    n_a = system_a.getNumParticles()
    a_to_h = {i: i for i in range(n_a)}
    b_to_h = {j: i for i, j in zip(env_a, env_b)}
    b_to_h.update({j: i for i, j in core_map.items()})
    for k, j in enumerate(unique_b):
        b_to_h[j] = n_a + k
    h_to_b = {h: j for j, h in b_to_h.items()}
    core = sorted(core_map)
    unique_new = [n_a + k for k in range(len(unique_b))]
    env = env_a
    core_set, old_set, new_set = set(core), set(unique_old), set(unique_new)
    ligand = sorted(core_set | old_set | new_set)
    ligand_set = set(ligand)
    alchemical = {int(i) for i in alchemical_water}
    if any(i in set_a or i < 0 or i >= n_a for i in alchemical):
        raise AlchemyError("alchemical_water must be environment atoms")
    hot_ligand = set(ligand) if rest2_ligand_atoms is None else {int(i) for i in rest2_ligand_atoms}
    if rest2_ligand_atoms is not None and not rest2:
        raise AlchemyError("rest2_ligand_atoms given without rest2")
    if not hot_ligand <= ligand_set:
        raise AlchemyError("rest2_ligand_atoms must be atoms of the ligand")
    # a ligand hot only in part: its pairs and torsions carry the factor of their hot atoms
    cold_ligand = bool(rest2) and hot_ligand != ligand_set
    # charges that REST2 does not scale (cold ligand atoms, the alchemical water) need lambdas without sqrt(s)
    uses_cold = bool(rest2) and (cold_ligand or bool(alchemical))
    env_b_of = dict(zip(env_a, env_b))
    for i, j in zip(env_a, env_b):
        if particles_a[i] != particles_b[j] and i not in alchemical:
            raise AlchemyError(f"environment atom {i} of A and {j} of B have other parameters")

    out = mm.System()
    for i in range(n_a):
        mass = system_a.getParticleMass(i)
        if i in core_map:
            mass = (mass + system_b.getParticleMass(core_map[i])) / 2   # OpenFE: the mean of the two
        out.addParticle(mass)
    for j in unique_b:
        out.addParticle(system_b.getParticleMass(j))
    out.setDefaultPeriodicBoxVectors(*system_a.getDefaultPeriodicBoxVectors())
    for i in range(n_a):
        if system_a.isVirtualSite(i):
            if i in set_a:
                raise AlchemyError("a ligand atom of A is a virtual site")
            out.setVirtualSite(i, _copy_virtual_site(system_a.getVirtualSite(i)))
    if any(system_b.isVirtualSite(j) for j in lig_b):
        raise AlchemyError("a ligand atom of B is a virtual site")

    # ---- constraints: those of A, then those of B that A does not have (OpenFE) ----------------
    lengths: dict[tuple[int, int], float] = {}
    for system, to_h in ((system_a, a_to_h), (system_b, b_to_h)):
        for k in range(system.getNumConstraints()):
            i, j, d = system.getConstraintParameters(k)
            key, d = _pair(to_h[i], to_h[j]), d.value_in_unit(_NM)
            if key not in lengths:
                out.addConstraint(key[0], key[1], d)
                lengths[key] = d
            elif abs(lengths[key] - d) > 1e-12:
                raise AlchemyError(f"the length of the constraint of atoms {key} changes between A and B; "
                                   "the mapping must leave out such atoms (constrained_bond_rule)")

    def kind(atoms) -> str:
        s = set(atoms)
        if s & old_set and s & new_set:
            raise AlchemyError(f"a term joins unique atoms of both ligands: {sorted(s)}")
        if s & old_set:
            return "old"
        if s & new_set:
            return "new"
        if s <= core_set:
            return "core"
        if s & ligand_set:
            raise AlchemyError(f"a term joins the ligand to the environment: {sorted(s)}")
        return "env"

    bonds_hybrid: set[tuple[int, int]] = set(lengths)

    # ---- bonds ---------------------------------------------------------------------------
    standard_bonds = mm.HarmonicBondForce()
    core_bonds = mm.CustomBondForce(CORE_BOND_ENERGY)
    core_bonds.setName(CORE_BOND_FORCE_NAME)
    for p in ("length1", "K1", "length2", "K2"):
        core_bonds.addPerBondParameter(p)
    core_bonds.addGlobalParameter(LAMBDA_BONDS, 0.0)
    bonds_b = {}
    if "HarmonicBondForce" in forces_b:
        f = forces_b["HarmonicBondForce"]
        for k in range(f.getNumBonds()):
            i, j, r0, kk = f.getBondParameters(k)
            bonds_b[_pair(b_to_h[i], b_to_h[j])] = (r0.value_in_unit(_NM), kk.value_in_unit(_KJ / _NM ** 2))
    done, env_bonds = set(), {}
    if "HarmonicBondForce" in forces_a:
        f = forces_a["HarmonicBondForce"]
        for k in range(f.getNumBonds()):
            i, j, r0, kk = f.getBondParameters(k)
            r0, kk = r0.value_in_unit(_NM), kk.value_in_unit(_KJ / _NM ** 2)
            key = _pair(i, j)
            bonds_hybrid.add(key)
            which = kind(key)
            if which == "core":
                r0_b, k_b = bonds_b.get(key, (r0, 0.0))
                core_bonds.addBond(key[0], key[1], [r0, kk, r0_b, k_b])
                done.add(key)
            else:
                standard_bonds.addBond(key[0], key[1], r0, kk)
                if which == "env":
                    env_bonds[key] = (r0, kk)
    for key, (r0, kk) in bonds_b.items():
        bonds_hybrid.add(key)
        which = kind(key)
        if which == "new":
            standard_bonds.addBond(key[0], key[1], r0, kk)
        elif which == "core" and key not in done:
            core_bonds.addBond(key[0], key[1], [r0, 0.0, r0, kk])
        elif which == "env":
            _same_environment_term("bond", key, env_bonds.pop(key, None), (r0, kk))
    if env_bonds:
        raise AlchemyError(f"the environment bond {next(iter(env_bonds))} is in A but not in B")

    # ---- angles --------------------------------------------------------------------------
    standard_angles = mm.HarmonicAngleForce()
    core_angles = mm.CustomAngleForce(CORE_ANGLE_ENERGY)
    core_angles.setName(CORE_ANGLE_FORCE_NAME)
    for p in ("theta0_1", "K_1", "theta0_2", "K_2"):
        core_angles.addPerAngleParameter(p)
    core_angles.addGlobalParameter(LAMBDA_ANGLES, 0.0)

    def angle_key(a, b, c):
        return (a, b, c) if a < c else (c, b, a)

    angles_b = {}
    if "HarmonicAngleForce" in forces_b:
        f = forces_b["HarmonicAngleForce"]
        for k in range(f.getNumAngles()):
            i, j, m, t0, kk = f.getAngleParameters(k)
            angles_b[angle_key(b_to_h[i], b_to_h[j], b_to_h[m])] = (t0.value_in_unit(_RAD),
                                                                   kk.value_in_unit(_KJ / _RAD ** 2))
    done, env_angles = set(), {}
    if "HarmonicAngleForce" in forces_a:
        f = forces_a["HarmonicAngleForce"]
        for k in range(f.getNumAngles()):
            i, j, m, t0, kk = f.getAngleParameters(k)
            t0, kk = t0.value_in_unit(_RAD), kk.value_in_unit(_KJ / _RAD ** 2)
            key = angle_key(i, j, m)
            which = kind(key)
            if which == "core":
                t0_b, k_b = angles_b.get(key, (t0, 0.0))
                core_angles.addAngle(*key, [t0, kk, t0_b, k_b])
                done.add(key)
            else:
                standard_angles.addAngle(*key, t0, kk)
                if which == "env":
                    env_angles[key] = (t0, kk)
    for key, (t0, kk) in angles_b.items():
        which = kind(key)
        if which == "new":
            standard_angles.addAngle(*key, t0, kk)
        elif which == "core" and key not in done:
            core_angles.addAngle(*key, [t0, 0.0, t0, kk])
        elif which == "env":
            _same_environment_term("angle", key, env_angles.pop(key, None), (t0, kk))
    if env_angles:
        raise AlchemyError(f"the environment angle {next(iter(env_angles))} is in A but not in B")

    # ---- torsions ------------------------------------------------------------------------
    standard_torsions = mm.PeriodicTorsionForce()
    core_torsions = []           # (atoms, [periodicity1, phase1, K1, periodicity2, phase2, K2])

    def torsion_terms(f, to_h, unique: str):
        unique_terms, interpolated = [], {}
        for k in range(f.getNumTorsions()):
            i, j, m, n, per, phase, kk = f.getTorsionParameters(k)
            atoms = (to_h[i], to_h[j], to_h[m], to_h[n])
            phase, kk = phase.value_in_unit(_RAD), kk.value_in_unit(_KJ)
            which = kind(atoms)
            if which == unique:
                unique_terms.append((atoms, int(per), phase, kk))
            elif which in ("core", "env"):
                # OpenFE's key: the atoms and the parameters rounded to 12 digits; a term that a System has twice
                # counts twice (OpenFE keeps one of them, and the end would not be that System)
                interpolated.setdefault((atoms, int(per), round(phase, 12), round(kk, 12)), []).append(
                    (atoms, int(per), phase, kk))
        return unique_terms, interpolated

    old_unique, old_terms = torsion_terms(forces_a["PeriodicTorsionForce"], a_to_h, "old") \
        if "PeriodicTorsionForce" in forces_a else ([], {})
    new_unique, new_terms = torsion_terms(forces_b["PeriodicTorsionForce"], b_to_h, "new") \
        if "PeriodicTorsionForce" in forces_b else ([], {})
    for atoms, per, phase, kk in old_unique + new_unique:
        standard_torsions.addTorsion(*atoms, per, phase, kk)
    # a term both ends have (as many times as both have it) is standard; the others go from one end to the other
    for key in sorted(old_terms.keys() & new_terms.keys()):
        for atoms, per, phase, kk in new_terms[key][:len(old_terms[key])]:
            standard_torsions.addTorsion(*atoms, per, phase, kk)
    for key in sorted(old_terms):
        for atoms, per, phase, kk in old_terms[key][len(new_terms.get(key, ())):]:
            if kind(atoms) == "env":
                raise AlchemyError(f"the environment torsion {atoms} differs between A and B")
            core_torsions.append((atoms, [per, phase, kk, 0.0, 0.0, 0.0]))
    for key in sorted(new_terms):
        for atoms, per, phase, kk in new_terms[key][len(old_terms.get(key, ())):]:
            if kind(atoms) == "env":
                raise AlchemyError(f"the environment torsion {atoms} differs between A and B")
            core_torsions.append((atoms, [0.0, 0.0, 0.0, per, phase, kk]))

    # ---- NonbondedForce: charges through offsets, no ligand Lennard-Jones ------------------------
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(nb_a.getNonbondedMethod())
    nb.setCutoffDistance(nb_a.getCutoffDistance())
    nb.setEwaldErrorTolerance(nb_a.getEwaldErrorTolerance())
    nb.setPMEParameters(*nb_a.getPMEParameters())
    nb.setLJPMEParameters(*nb_a.getLJPMEParameters())
    nb.setUseDispersionCorrection(nb_a.getUseDispersionCorrection())
    nb.setUseSwitchingFunction(False)
    nb.setExceptionsUsePeriodicBoundaryConditions(nb_a.getExceptionsUsePeriodicBoundaryConditions())
    for name in ELECTROSTATIC_LAMBDAS + (COLD_ELECTROSTATIC_LAMBDAS if uses_cold else ()):
        nb.addGlobalParameter(name, 0.0)
    #: charges of the ligand: base, and the offsets under core, delete and insert
    charge = {}
    params_a, params_b = {}, {}
    for h in range(out.getNumParticles()):
        if h in core_set:
            qa, sa, ea = particles_a[h]
            qb, sb, eb = particles_b[core_map[h]]
            charge[h] = (qa, qb - qa, 0.0, 0.0)
            params_a[h], params_b[h] = (qa, sa, ea), (qb, sb, eb)
            nb.addParticle(qa, sa, 0.0)
        elif h in old_set:
            qa, sa, ea = particles_a[h]
            charge[h] = (qa, 0.0, -qa, 0.0)
            params_a[h] = (qa, sa, ea)
            nb.addParticle(qa, sa, 0.0)
        elif h in new_set:
            qb, sb, eb = particles_b[h_to_b[h]]
            charge[h] = (0.0, 0.0, 0.0, qb)
            params_b[h] = (qb, sb, eb)
            nb.addParticle(0.0, sb, 0.0)
        elif h in alchemical:
            # the water that is an ion in B: its charge as an offset, its Lennard-Jones in the custom force
            qa, sa, _ea = particles_a[h]
            qb = particles_b[env_b_of[h]][0]
            nb.addParticle(qa, sa, 0.0)
            if qb != qa:
                # not hot: without sqrt(s) where REST2 scales the charges of the ligand
                core_lambda = COLD_ELECTROSTATIC_LAMBDAS[0] if uses_cold else LAMBDA_ELECTROSTATICS_CORE
                nb.addParticleParameterOffset(core_lambda, h, qb - qa, 0.0, 0.0)
        else:
            nb.addParticle(*particles_a[h])
    for h in ligand:
        _q0, dc, dd, di = charge[h]
        names = ELECTROSTATIC_LAMBDAS if h in hot_ligand or not cold_ligand else COLD_ELECTROSTATIC_LAMBDAS
        for name, d in zip(names, (dc, dd, di)):
            if d != 0.0:
                nb.addParticleParameterOffset(name, h, d, 0.0, 0.0)
    for (i, j), (qq, s, e) in exceptions_a.items():
        if i in set_a or j in set_a:
            if not (i in set_a and j in set_a):
                raise AlchemyError("an exception joins ligand A to the environment (covalent ligand?)")
            continue
        nb.addException(i, j, qq, s, e)
    for (i, j) in exceptions_b:
        if (i in set_b) != (j in set_b):
            raise AlchemyError("an exception joins ligand B to the environment (covalent ligand?)")
    for a, i in enumerate(ligand):
        for j in ligand[a + 1:]:
            nb.addException(i, j, 0.0, 0.3, 0.0)

    # ---- pairs inside the ligand -------------------------------------------------------------
    coulomb = mm.CustomBondForce(ligand_coulomb_energy_by_atom() if cold_ligand else ligand_coulomb_energy(rest2))
    coulomb.setName(LIGAND_COULOMB_FORCE_NAME)
    for name in ("product", "q10", "q1c", "q1d", "q1i", "q20", "q2c", "q2d", "q2i", "qq0", "qqc", "qqd", "qqi") \
            + (("h1", "h2") if cold_ligand else ()):
        coulomb.addPerBondParameter(name)
    lj = mm.CustomBondForce(("(1+h1*rest2_dq)*(1+h2*rest2_dq)*" if cold_ligand else "(1+rest2_ds)*" if rest2 else "")
                            + LIGAND_LJ_ENERGY)
    lj.setName(LIGAND_LJ_FORCE_NAME)
    for name in ("sigmaA", "epsilonA", "sigmaB", "epsilonB", "unique_old", "unique_new") \
            + (("h1", "h2") if cold_ligand else ()):
        lj.addPerBondParameter(name)
    for name in (COLD_ELECTROSTATIC_LAMBDAS if cold_ligand else ELECTROSTATIC_LAMBDAS):
        coulomb.addGlobalParameter(name, 0.0)
    lj.addGlobalParameter("softcore_alpha", float(softcore_alpha))
    for name in STERIC_LAMBDAS:
        lj.addGlobalParameter(name, 0.0)
    if rest2:
        coulomb.addGlobalParameter(REST2_DQ, 0.0)
        lj.addGlobalParameter(REST2_DQ if cold_ligand else REST2_DS, 0.0)

    def hot_flags(i: int, j: int) -> list[float]:
        """The flags of a pair of a ligand hot only in part (none otherwise)."""
        return [float(i in hot_ligand), float(j in hot_ligand)] if cold_ligand else []

    def lj_end(h: int, end: str):
        """sigma and epsilon of a ligand atom at one end (a unique atom: its sigma, epsilon 0, as OpenFE)."""
        if end == "a":
            p = params_a.get(h) or (0.0, params_b[h][1], 0.0)
        else:
            p = params_b.get(h) or (0.0, params_a[h][1], 0.0)
        return p[1], p[2]

    def end_pair(i: int, j: int, end: str):
        """(charge product, sigma, epsilon, exception) of a pair at one end; None if it does not exist there."""
        if end == "a":
            if i not in params_a or j not in params_a:
                return None
            key, ex, pp = _pair(i, j), exceptions_a, particles_a
        else:
            if i not in params_b or j not in params_b:
                return None
            key, ex, pp = _pair(h_to_b[i], h_to_b[j]), exceptions_b, particles_b
        if key in ex:
            qq, s, e = ex[key]
            return qq, s, e, True
        qi, si, ei = pp[key[0]]
        qj, sj, ej = pp[key[1]]
        return qi * qj, 0.5 * (si + sj), math.sqrt(ei * ej), False

    for a, i in enumerate(ligand):
        for j in ligand[a + 1:]:
            if {i, j} & old_set and {i, j} & new_set:
                continue                                            # never interact
            pair_kind = kind((i, j))
            end_a, end_b = end_pair(i, j, "a"), end_pair(i, j, "b")
            if pair_kind == "core":
                exception = end_a[3] or end_b[3]
                if exception:
                    # an exception at one end at least: charge product and LJ interpolated linearly
                    qq0, qqc = end_a[0], end_b[0] - end_a[0]
                    coulomb_terms = [0.0, *[0.0] * 8, qq0, qqc, 0.0, 0.0]
                else:
                    coulomb_terms = [1.0, *charge[i], *charge[j], 0.0, 0.0, 0.0, 0.0]
                sig_a, eps_a, sig_b, eps_b = end_a[1], end_a[2], end_b[1], end_b[2]
                flags = (0.0, 0.0)
            else:
                old = pair_kind == "old"
                here = end_a if old else end_b
                exception = here[3]
                if exception:
                    qq = here[0]
                    coulomb_terms = [0.0, *[0.0] * 8, qq if old else 0.0, 0.0, -qq if old else 0.0,
                                     0.0 if old else qq]
                    sig_a = sig_b = here[1]
                    eps_a, eps_b = (here[2], 0.0) if old else (0.0, here[2])
                else:
                    coulomb_terms = [1.0, *charge[i], *charge[j], 0.0, 0.0, 0.0, 0.0]
                    (sai, eai), (saj, eaj) = lj_end(i, "a"), lj_end(j, "a")
                    (sbi, ebi), (sbj, ebj) = lj_end(i, "b"), lj_end(j, "b")
                    sig_a, eps_a = 0.5 * (sai + saj), math.sqrt(eai * eaj)
                    sig_b, eps_b = 0.5 * (sbi + sbj), math.sqrt(ebi * ebj)
                flags = (1.0, 0.0) if old else (0.0, 1.0)
            if any(v != 0.0 for v in coulomb_terms[1:]):
                coulomb.addBond(i, j, coulomb_terms + hot_flags(i, j))
            if eps_a != 0.0 or eps_b != 0.0:
                lj.addBond(i, j, [sig_a, eps_a, sig_b, eps_b, *flags] + hot_flags(i, j))
    periodic = nb_a.getExceptionsUsePeriodicBoundaryConditions()
    coulomb.setUsesPeriodicBoundaryConditions(periodic)
    lj.setUsesPeriodicBoundaryConditions(periodic)

    # ---- REST2: hot environment atoms and environment exceptions --------------------------------
    hot_env = tuple(sorted({int(i) for i in rest2_atoms}))
    if hot_env and not rest2:
        raise AlchemyError("rest2_atoms given without rest2")
    if any(i in ligand_set or i < 0 or i >= n_a for i in hot_env):
        raise AlchemyError("rest2_atoms must be environment atoms")
    if alchemical & set(hot_env):
        raise AlchemyError("the alchemical water is in the hot region of REST2")
    region = hot_ligand | set(hot_env)
    if rest2:
        nb.addGlobalParameter(REST2_DQ, 0.0)
        nb.addGlobalParameter(REST2_DS, 0.0)
        for h in sorted(hot_ligand):
            q0 = charge[h][0]
            if q0 != 0.0:
                nb.addParticleParameterOffset(REST2_DQ, h, q0, 0.0, 0.0)
        for h in hot_env:
            q, _s, e = particles_a[h]
            if q != 0.0:
                nb.addParticleParameterOffset(REST2_DQ, h, q, 0.0, 0.0)
            if e != 0.0:
                nb.addParticleParameterOffset(REST2_DS, h, 0.0, 0.0, e)
        for k in range(nb.getNumExceptions()):
            i, j, qq, _s, e = nb.getExceptionParameters(k)
            n_hot = (i in region) + (j in region)
            qq, e = qq.value_in_unit(_E * _E), e.value_in_unit(_KJ)
            if n_hot and (qq != 0.0 or e != 0.0):
                nb.addExceptionParameterOffset(REST2_DS if n_hot == 2 else REST2_DQ, k, qq, 0.0, e)

    # ---- Lennard-Jones of the ligand with the environment ------------------------------------
    sc = mm.CustomNonbondedForce(HYBRID_LJ_ENERGY_REST2 if rest2 else HYBRID_LJ_ENERGY)
    sc.setName(HYBRID_LJ_FORCE_NAME)
    sc.addGlobalParameter("softcore_alpha", float(softcore_alpha))
    for name in STERIC_LAMBDAS:
        sc.addGlobalParameter(name, 0.0)
    for name in ("sigmaA", "epsilonA", "sigmaB", "epsilonB", "unique_old", "unique_new"):
        sc.addPerParticleParameter(name)
    if rest2:
        sc.addGlobalParameter(REST2_DQ, 0.0)
        sc.addPerParticleParameter("hot")
    for h in range(out.getNumParticles()):
        if h in core_set:
            row = [params_a[h][1], params_a[h][2], params_b[h][1], params_b[h][2], 0.0, 0.0]
        elif h in old_set:
            row = [params_a[h][1], params_a[h][2], params_a[h][1], 0.0, 1.0, 0.0]
        elif h in new_set:
            row = [params_b[h][1], 0.0, params_b[h][1], params_b[h][2], 0.0, 1.0]
        elif h in alchemical:
            _qa, sa, ea = particles_a[h]
            _qb, sb, eb = particles_b[env_b_of[h]]
            row = [sa, ea, sb, eb, 0.0, 0.0]
        else:
            _q, s, e = particles_a[h]
            row = [s, e, s, e, 0.0, 0.0]
        sc.addParticle(row + ([1.0 if h in region else 0.0] if rest2 else []))
    sc.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    sc.setCutoffDistance(nb.getCutoffDistance())
    sc.setUseSwitchingFunction(False)
    # no correction of the ligand-environment Lennard-Jones beyond the cutoff, as OpenFE's relative protocol
    # (`use_dispersion_correction: False` for the hybrid; analysis.lrc: none)
    sc.setUseLongRangeCorrection(False)
    environment = set(env)
    for group in (core_set, old_set, new_set):
        if group:
            sc.addInteractionGroup(group, environment)
    if alchemical:
        # the Lennard-Jones of the alchemical water with the rest of the environment (with the ligand: above)
        sc.addInteractionGroup(alchemical, environment - alchemical)
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        sc.addExclusion(i, j)

    # ---- assemble -------------------------------------------------------------------------------
    nb.setForceGroup(MAIN_GROUP)
    out.addForce(nb)
    for force in (standard_bonds, standard_angles, standard_torsions, coulomb):
        force.setForceGroup(MAIN_GROUP)
        out.addForce(force)
    sc.setForceGroup(SOFTCORE_GROUP)
    out.addForce(sc)
    for force in (core_bonds, core_angles):
        force.setForceGroup(HYBRID_GROUP)
        out.addForce(force)
    lj.setForceGroup(HYBRID_HOT_GROUP)
    out.addForce(lj)

    def proper(atoms) -> bool:
        i, j, m, n = atoms
        return _pair(i, j) in bonds_hybrid and _pair(j, m) in bonds_hybrid and _pair(m, n) in bonds_hybrid

    impropers = mm.CustomTorsionForce(CORE_TORSION_ENERGY)
    impropers.setName(CORE_TORSION_FORCE_NAME)
    # the REST2 factor multiplies the whole torsion, the term of A and the term of B (review of PR #19: prefixed to the
    # expression it multiplied the term of A alone, so a term only B has was never scaled)
    factor = "(1+hot*rest2_ds)" if cold_ligand else "(1+rest2_ds)" if rest2 else None
    propers = mm.CustomTorsionForce(CORE_TORSION_ENERGY if factor is None
                                    else f"{factor}*U_torsion; U_torsion = {CORE_TORSION_ENERGY}")
    propers.setName(CORE_PROPER_TORSION_FORCE_NAME)
    for force in (impropers, propers):
        for p in ("periodicity1", "phase1", "K1", "periodicity2", "phase2", "K2"):
            force.addPerTorsionParameter(p)
        force.addGlobalParameter(LAMBDA_TORSIONS, 0.0)
    if cold_ligand:
        propers.addPerTorsionParameter("hot")
    if rest2:
        propers.addGlobalParameter(REST2_DS, 0.0)
    for atoms, values in core_torsions:
        if proper(atoms):
            # a ligand hot only in part: a proper torsion is scaled if its four atoms are hot (the rule of the ABFE)
            propers.addTorsion(*atoms, values + ([float(all(x in hot_ligand for x in atoms))] if cold_ligand else []))
        else:
            impropers.addTorsion(*atoms, values)
    impropers.setForceGroup(HYBRID_GROUP)
    propers.setForceGroup(HYBRID_HOT_GROUP)
    out.addForce(impropers)
    out.addForce(propers)
    if "CMMotionRemover" in forces_a:
        out.addForce(mm.CMMotionRemover(forces_a["CMMotionRemover"].getFrequency()))
    n_rest2_torsions = _move_hot_torsions(out, region, bonds_hybrid) if rest2 else 0
    if not rest2:
        # without REST2 the core proper torsions are a part of the hybrid group like the impropers
        propers.setForceGroup(HYBRID_GROUP)

    pos_a = np.asarray(positions_a, dtype=float)
    pos_b = np.asarray(positions_b, dtype=float)
    positions = np.vstack([pos_a, pos_b[unique_b]]) if unique_b else pos_a.copy()
    return HybridSystem(system=out, positions=positions, core=tuple(core), unique_old=tuple(unique_old),
                        unique_new=tuple(unique_new), a_to_hybrid=a_to_h, b_to_hybrid=b_to_h,
                        parameters_a=params_a, parameters_b=params_b, softcore_alpha=float(softcore_alpha),
                        rest2=bool(rest2), rest2_atoms=hot_env, rest2_torsions=n_rest2_torsions,
                        bonds=frozenset(bonds_hybrid), alchemical_water=tuple(sorted(alchemical)),
                        rest2_ligand_atoms=tuple(sorted(hot_ligand)) if rest2 else ())


def _copy_virtual_site(site):
    """A copy of a virtual site of an environment atom (its atoms keep their indices in the hybrid)."""
    if isinstance(site, mm.ThreeParticleAverageSite):
        return mm.ThreeParticleAverageSite(site.getParticle(0), site.getParticle(1), site.getParticle(2),
                                           site.getWeight(0), site.getWeight(1), site.getWeight(2))
    if isinstance(site, mm.TwoParticleAverageSite):
        return mm.TwoParticleAverageSite(site.getParticle(0), site.getParticle(1), site.getWeight(0), site.getWeight(1))
    if isinstance(site, mm.OutOfPlaneSite):
        return mm.OutOfPlaneSite(site.getParticle(0), site.getParticle(1), site.getParticle(2),
                                 site.getWeight12(), site.getWeight13(), site.getWeightCross())
    raise AlchemyError(f"virtual site {type(site).__name__} is not supported")
