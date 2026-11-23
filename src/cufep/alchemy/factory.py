"""Alchemical System construction for absolute binding free energies.

The alchemical System is built only from stock OpenMM classes, in a layout that both
backends can run unchanged ("split" layout):

``NonbondedForce`` (PME)
    * ligand charges are set to zero and restored through particle parameter offsets
      scaled by ``lambda_electrostatics`` (ligand-environment electrostatics, including
      the reciprocal-space part, scale with lambda);
    * ligand Lennard-Jones epsilons are set to zero (ligand-environment LJ moves to the
      soft-core force below);
    * every ligand-ligand pair becomes an explicit exception carrying its full Coulomb
      and LJ interaction, so intramolecular nonbonded interactions are kept at all
      lambda values. The decoupled end state is therefore the same non-interacting,
      internally intact ligand in the complex leg and in the solvent leg, and no
      vacuum leg is needed.

``annihilate_electrostatics=True`` (the rule of openmmtools' ``AbsoluteAlchemicalFactory`` with
``alchemical_pme_treatment='exact'``, which OpenFE's ABFE protocol uses)
    The electrostatics inside the ligand are switched off with ``lambda_electrostatics`` too. In
    openmmtools the charges of the alchemical atoms are offsets scaled by lambda, so pairs of them
    that are not exceptions, and their periodic images, interact through PME with lambda^2; the
    charge product of an exception (the 1-4 pairs) gets an offset scaled by lambda, so it is linear
    in lambda. Here every ligand-ligand pair is an exception, so the same rule is written out: the
    charge product of every ligand-ligand exception is zero, and a ``CustomBondForce`` of the main
    force group gives each pair its Coulomb energy, times lambda for a pair that is an exception
    of the force field and times lambda^2 for a pair that cuFEP made one. The Lennard-Jones
    interactions inside the ligand are kept, as openmmtools keeps them with
    ``annihilate_sterics=False``. The decoupled end state is then a ligand without electrostatics
    of its own, again the same in both legs.

``CustomNonbondedForce`` (soft-core LJ, Beutler form with a=1, b=1, c=6)
    ligand x environment only (one interaction group), scaled by ``lambda_sterics``. In the
    names of openmmtools, U = lambda^a 4 eps x (x - 1) with x = 1 / (alpha (1 - lambda)^b +
    (r / sigma)^c). ``softcore_b=2`` gives r_eff^6 = r^6 + alpha (1 - lambda)^2 sigma^6, the form
    of Ding et al. (arXiv:2603.22274 v3, who use alpha = 1): the repulsion of an overlapping pair
    then falls faster as lambda falls.

The cuPotential backend replaces only the NonbondedForce; the soft-core force stays on
the stock CUDA kernels in the same System.

REST2 (``rest2=True``)
    Replica exchange with solute scaling (Wang, Friesner, Berne, J. Phys. Chem. B 2011, 115,
    9431): the energy inside a hot region is scaled by s = T / T_hot and its interaction with
    the rest by sqrt(s). The region is the ligand plus the environment atoms in
    ``rest2_atoms``. As in Sire / SOMD2 this is done by

    * charges of hot atoms times sqrt(s), their Lennard-Jones epsilon times s (pairs of two
      hot atoms then scale by s, pairs of a hot and a cold atom by sqrt(s));
    * 1-4 and other exception pairs times s if both atoms are hot, sqrt(s) if one is;
    * proper torsions whose four atoms are all hot times s. Impropers, bonds and angles
      are not scaled.

    Two global parameters carry it, ``rest2_dq`` = sqrt(s) - 1 and ``rest2_ds`` = s - 1; both
    are zero in the physical state, so the base parameters of the System stay the physical
    ones. The ligand charges, which are already offsets scaled by ``lambda_electrostatics``,
    get the factor sqrt(s) through that parameter (see ``sampling/states.py``). With
    ``annihilate_electrostatics`` the Coulomb energy of a pair inside the ligand carries s like
    every pair of two hot atoms: lambda^2 * s is the square of that parameter, and lambda * s
    its product with sqrt(s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import openmm as mm
import openmm.unit as unit

from cufep.backends.base import clone_system, find_forces

LAMBDA_ELECTROSTATICS = "lambda_electrostatics"
LAMBDA_STERICS = "lambda_sterics"
LAMBDA_RESTRAINTS = "lambda_restraints"
SOFTCORE_FORCE_NAME = "cufep_softcore_lj"

#: Force groups. Keeping the three lambda-dependent terms in separate groups lets the
#: sampler evaluate the energy of every thermodynamic state cheaply:
#:   group 0  everything else, including the NonbondedForce. Its energy is an exact
#:            quadratic in lambda_electrostatics (charges scale linearly), so three
#:            evaluations give it for all values.
#:   group 1  soft-core LJ (depends only on lambda_sterics; a small ligand x environment
#:            kernel that is cheap to re-evaluate)
#:   group 2  restraints (linear in lambda_restraints)
#:   group 3  REST2: the proper torsions inside the hot region (linear in s)
MAIN_GROUP = 0
SOFTCORE_GROUP = 1
RESTRAINT_GROUP = 2
REST2_GROUP = 3

REST2_DQ = "rest2_dq"            # sqrt(s) - 1
REST2_DS = "rest2_ds"            # s - 1
REST2_TORSION_FORCE_NAME = "cufep_rest2_torsions"
LIGAND_COULOMB_FORCE_NAME = "cufep_ligand_coulomb"
#: 1 / (4 pi epsilon_0) in kJ nm / (mol e^2), the value of OpenMM (ONE_4PI_EPS0)
ONE_4PI_EPS0 = 138.93545764438198

_E = unit.elementary_charge
_NM = unit.nanometer
_KJ = unit.kilojoule_per_mole

SOFTCORE_ENERGY = (
    "lambda_sterics*4*epsilon*x*(x-1);"
    "x=1/(softcore_alpha*(1-lambda_sterics)+reff^6);"
    "reff=r/sigma;"
    "sigma=0.5*(sigma1+sigma2);"
    "epsilon=sqrt(epsilon1*epsilon2)"
)


#: the same with the REST2 factor of each atom: sqrt(s) for a hot atom, 1 otherwise
SOFTCORE_ENERGY_REST2 = "(1+hot1*rest2_dq)*(1+hot2*rest2_dq)*" + SOFTCORE_ENERGY
#: the powers of (1 - lambda) in the soft-core radius that can be chosen (openmmtools' softcore_b)
SOFTCORE_POWERS = (1, 2)


def softcore_energy(softcore_b: int = 1, rest2: bool = False) -> str:
    """The energy expression of the soft-core force. With ``softcore_b`` 1 it is the text this
    module always used (the checksum of the System of a stored leg is taken of it)."""
    if softcore_b not in SOFTCORE_POWERS:
        raise AlchemyError(f"softcore_b is one of {SOFTCORE_POWERS}, not {softcore_b!r}")
    energy = SOFTCORE_ENERGY_REST2 if rest2 else SOFTCORE_ENERGY
    if softcore_b == 1:
        return energy
    radius = "softcore_alpha*(1-lambda_sterics)"
    if energy.count(radius) != 1:
        raise AlchemyError("the soft-core expression has changed; softcore_energy must follow it")
    return energy.replace(radius, f"{radius}^{int(softcore_b)}")
REST2_TORSION_ENERGY = "(1+rest2_ds)*k*(1+cos(periodicity*theta-phase))"


class AlchemyError(ValueError):
    """The System cannot be made alchemical as requested."""


@dataclass
class AlchemicalSystem:
    system: mm.System
    ligand_atoms: tuple[int, ...]
    softcore_alpha: float
    #: original (charge [e], sigma [nm], epsilon [kJ/mol]) of the ligand atoms
    ligand_parameters: tuple[tuple[float, float, float], ...]
    #: REST2: whether the System carries the scaling, the hot environment atoms (the ligand is
    #: always hot) and the number of torsion terms that are scaled
    rest2: bool = False
    rest2_atoms: tuple[int, ...] = ()
    rest2_torsions: int = 0
    #: whether the electrostatics inside the ligand are switched off with lambda_electrostatics
    annihilate_electrostatics: bool = False
    #: the power of (1 - lambda) in the soft-core radius (SOFTCORE_POWERS)
    softcore_b: int = 1

    @property
    def lambda_parameters(self) -> tuple[str, ...]:
        return (LAMBDA_ELECTROSTATICS, LAMBDA_STERICS)


def bonded_pairs(system: mm.System) -> set[tuple[int, int]]:
    """Pairs of atoms joined by a bond term or a constraint (bonds to hydrogen are constraints)."""
    pairs: set[tuple[int, int]] = set()
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for k in range(force.getNumBonds()):
                i, j, _r, _k = force.getBondParameters(k)
                pairs.add((min(i, j), max(i, j)))
    for k in range(system.getNumConstraints()):
        i, j, _d = system.getConstraintParameters(k)
        pairs.add((min(i, j), max(i, j)))
    return pairs


def make_alchemical_system(system: mm.System, ligand_atoms, softcore_alpha: float = 0.5, rest2: bool = False,
                           rest2_atoms=(), bonds=None, annihilate_electrostatics: bool = False,
                           softcore_b: int = 1) -> AlchemicalSystem:
    """Return an alchemical copy of ``system`` in which ``ligand_atoms`` can be decoupled.

    With ``rest2`` the System also carries the REST2 scaling of the hot region: the ligand and
    the environment atoms ``rest2_atoms``. ``bonds`` (pairs of atom indices) tells proper
    torsions from impropers; by default it is taken from the bond terms and constraints. With
    ``annihilate_electrostatics`` the electrostatics inside the ligand go with
    ``lambda_electrostatics`` too (module docstring); without it the System is the one this
    function always made. ``softcore_b`` is the power of (1 - lambda) in the soft-core radius.
    """
    ligand = tuple(sorted(int(i) for i in ligand_atoms))
    if not ligand:
        raise AlchemyError("no ligand atoms given")
    out = clone_system(system)
    n = out.getNumParticles()
    if ligand[-1] >= n:
        raise AlchemyError("ligand atom index out of range")
    found = find_forces(out, mm.NonbondedForce)
    if len(found) != 1:
        raise AlchemyError(f"expected exactly one NonbondedForce, found {len(found)}")
    nb: mm.NonbondedForce = found[0][1]
    if nb.getNonbondedMethod() not in (mm.NonbondedForce.PME, mm.NonbondedForce.LJPME):
        raise AlchemyError("the NonbondedForce must use PME")
    if nb.getNumParticleParameterOffsets() or nb.getNumExceptionParameterOffsets():
        raise AlchemyError("the NonbondedForce already has parameter offsets; pass a non-alchemical System")
    if any(type(f) is mm.CustomNonbondedForce for f in out.getForces()):
        raise AlchemyError("the System already has a CustomNonbondedForce (e.g. NBFIX); not supported")
    ligand_set = set(ligand)
    environment = [i for i in range(n) if i not in ligand_set]
    if not environment:
        raise AlchemyError("the System contains only ligand atoms")

    # -- original parameters --------------------------------------------------
    params = []
    for i in range(n):
        q, sig, eps = nb.getParticleParameters(i)
        params.append((q.value_in_unit(_E), sig.value_in_unit(_NM), eps.value_in_unit(_KJ)))

    # -- ligand-ligand pairs: make every one an explicit, constant exception ---
    # An exception has no cutoff. Pairs of ligand atoms four or more bonds apart are ordinary
    # pairs in the original System and cut off there; here their Lennard-Jones energy beyond the
    # cutoff is kept (hundredths of a kcal/mol for drug-like ligands, the same at both ends of
    # both legs; OpenFE keeps the cutoff). See protocol.md, "Interactions inside the ligand".
    # With annihilate_electrostatics the charge products of these exceptions are zero and their
    # Coulomb energies go to a force scaled by lambda (``coulomb``: atoms, charge product, and
    # whether the pair is an exception of the force field, whose energy openmmtools scales by
    # lambda, or one made here, a pair that PME would scale by lambda^2).
    existing: set[tuple[int, int]] = set()
    coulomb: list[tuple[int, int, float, float]] = []
    for k in range(nb.getNumExceptions()):
        i, j, qq, sig, eps = nb.getExceptionParameters(k)
        in_i, in_j = i in ligand_set, j in ligand_set
        if in_i != in_j:
            raise AlchemyError("an exception connects the ligand to the environment (covalent ligand?)")
        if in_i:
            existing.add((min(i, j), max(i, j)))
            if annihilate_electrostatics and qq.value_in_unit(_E * _E) != 0.0:
                coulomb.append((i, j, qq.value_in_unit(_E * _E), 1.0))
                nb.setExceptionParameters(k, i, j, 0.0, sig, eps)
    for a, i in enumerate(ligand):
        qi, si, ei = params[i]
        for j in ligand[a + 1:]:
            if (i, j) in existing:
                continue
            qj, sj, ej = params[j]
            if annihilate_electrostatics:
                nb.addException(i, j, 0.0, 0.5 * (si + sj), math.sqrt(ei * ej))
                if qi * qj != 0.0:
                    coulomb.append((i, j, qi * qj, 0.0))
            else:
                nb.addException(i, j, qi * qj, 0.5 * (si + sj), math.sqrt(ei * ej))

    # -- NonbondedForce: charges through offsets, no ligand LJ -------------------
    nb.addGlobalParameter(LAMBDA_ELECTROSTATICS, 1.0)
    for i in ligand:
        q, sig, _eps = params[i]
        nb.setParticleParameters(i, 0.0, sig, 0.0)
        if q != 0.0:
            nb.addParticleParameterOffset(LAMBDA_ELECTROSTATICS, i, q, 0.0, 0.0)

    # -- REST2: hot environment atoms and exception pairs in the NonbondedForce ----
    hot_env = tuple(sorted({int(i) for i in rest2_atoms}))
    if hot_env and not rest2:
        raise AlchemyError("rest2_atoms given without rest2")
    if any(i in ligand_set for i in hot_env) or (hot_env and (hot_env[0] < 0 or hot_env[-1] >= n)):
        raise AlchemyError("rest2_atoms must be environment atoms of the System")
    region = ligand_set | set(hot_env)
    n_rest2_torsions = 0
    if rest2:
        nb.addGlobalParameter(REST2_DQ, 0.0)
        nb.addGlobalParameter(REST2_DS, 0.0)
        for i in hot_env:
            q, _sig, eps = params[i]
            if q != 0.0:
                nb.addParticleParameterOffset(REST2_DQ, i, q, 0.0, 0.0)
            if eps != 0.0:
                nb.addParticleParameterOffset(REST2_DS, i, 0.0, 0.0, eps)
        for k in range(nb.getNumExceptions()):
            i, j, qq, _sig, eps = nb.getExceptionParameters(k)
            n_hot = (i in region) + (j in region)
            qq, eps = qq.value_in_unit(_E * _E), eps.value_in_unit(_KJ)
            if n_hot and (qq != 0.0 or eps != 0.0):
                nb.addExceptionParameterOffset(REST2_DS if n_hot == 2 else REST2_DQ, k, qq, 0.0, eps)
        n_rest2_torsions = _move_hot_torsions(out, region, bonded_pairs(out) if bonds is None else
                                              {(min(i, j), max(i, j)) for i, j in bonds})

    # -- soft-core LJ: ligand x environment --------------------------------------
    sc = mm.CustomNonbondedForce(softcore_energy(softcore_b, rest2))
    sc.setName(SOFTCORE_FORCE_NAME)
    sc.addGlobalParameter(LAMBDA_STERICS, 1.0)
    sc.addGlobalParameter("softcore_alpha", float(softcore_alpha))
    sc.addPerParticleParameter("sigma")
    sc.addPerParticleParameter("epsilon")
    if rest2:
        sc.addGlobalParameter(REST2_DQ, 0.0)
        sc.addPerParticleParameter("hot")
    for i, (_q, sig, eps) in enumerate(params):
        # sigma is irrelevant when epsilon is zero; keep it finite to avoid r/0.
        sc.addParticle([sig if sig > 0.0 else 0.3, eps] + ([1.0 if i in region else 0.0] if rest2 else []))
    sc.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    sc.setCutoffDistance(nb.getCutoffDistance())
    sc.setUseSwitchingFunction(nb.getUseSwitchingFunction())
    if nb.getUseSwitchingFunction():
        sc.setSwitchingDistance(nb.getSwitchingDistance())
    # The homogeneous tail of the ligand-environment LJ is not sampled; it is restored
    # in analysis by reweighting the coupled end state to a long cutoff (alchemy/lrc.py).
    sc.setUseLongRangeCorrection(False)
    sc.addInteractionGroup(set(ligand), set(environment))
    for k in range(nb.getNumExceptions()):
        i, j, _qq, _sig, _eps = nb.getExceptionParameters(k)
        sc.addExclusion(i, j)
    nb.setForceGroup(MAIN_GROUP)
    sc.setForceGroup(SOFTCORE_GROUP)
    out.addForce(sc)

    # -- electrostatics inside the ligand, switched off with the charges ------------
    if annihilate_electrostatics:
        # lambda_electrostatics is lambda * sqrt(s) with REST2 (s = 1 without): its square is
        # lambda^2 * s, and times sqrt(s) = 1 + rest2_dq it is lambda * s. Both atoms of a pair are
        # ligand atoms and so hot: the pair carries s like every pair of two hot atoms.
        once = f"{LAMBDA_ELECTROSTATICS}*(1+{REST2_DQ})" if rest2 else LAMBDA_ELECTROSTATICS
        inside = mm.CustomBondForce(f"{ONE_4PI_EPS0!r}*qq/r*(linear*{once}+(1-linear)*{LAMBDA_ELECTROSTATICS}^2)")
        inside.setName(LIGAND_COULOMB_FORCE_NAME)
        inside.addGlobalParameter(LAMBDA_ELECTROSTATICS, 1.0)
        if rest2:
            inside.addGlobalParameter(REST2_DQ, 0.0)
        inside.addPerBondParameter("qq")
        inside.addPerBondParameter("linear")
        for i, j, qq, linear in coulomb:
            inside.addBond(i, j, [qq, linear])
        inside.setUsesPeriodicBoundaryConditions(nb.getExceptionsUsePeriodicBoundaryConditions())
        inside.setForceGroup(MAIN_GROUP)
        out.addForce(inside)

    return AlchemicalSystem(system=out, ligand_atoms=ligand, softcore_alpha=float(softcore_alpha),
                            ligand_parameters=tuple(params[i] for i in ligand), rest2=bool(rest2),
                            rest2_atoms=hot_env, rest2_torsions=n_rest2_torsions,
                            annihilate_electrostatics=bool(annihilate_electrostatics), softcore_b=int(softcore_b))


def _move_hot_torsions(system: mm.System, region: set[int], bonds: set[tuple[int, int]]) -> int:
    """Move the proper torsions inside ``region`` into a force that REST2 scales by s.

    A torsion is proper if its four atoms form a chain of three bonds; impropers stay as they
    are. Returns the number of torsion terms moved.
    """
    def bonded(i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in bonds

    hot = mm.CustomTorsionForce(REST2_TORSION_ENERGY)
    hot.setName(REST2_TORSION_FORCE_NAME)
    hot.addGlobalParameter(REST2_DS, 0.0)
    for name in ("k", "periodicity", "phase"):
        hot.addPerTorsionParameter(name)
    hot.setForceGroup(REST2_GROUP)
    for force in system.getForces():
        if type(force) is not mm.PeriodicTorsionForce:
            continue
        hot.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for t in range(force.getNumTorsions()):
            i, j, k, l, periodicity, phase, kval = force.getTorsionParameters(t)
            kval = kval.value_in_unit(_KJ)
            if kval == 0.0 or not all(x in region for x in (i, j, k, l)):
                continue
            if not (bonded(i, j) and bonded(j, k) and bonded(k, l)):
                continue
            hot.addTorsion(i, j, k, l, [kval, float(periodicity), phase.value_in_unit(unit.radian)])
            force.setTorsionParameters(t, i, j, k, l, periodicity, phase, 0.0)
    system.addForce(hot)
    return hot.getNumTorsions()


def set_lambdas(context: mm.Context, lambdas: dict[str, float]) -> None:
    """Set alchemical parameters on a Context (only those the Context knows)."""
    known = context.getParameters()
    for name, value in lambdas.items():
        if name in known:
            context.setParameter(name, float(value))
