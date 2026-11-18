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

``CustomNonbondedForce`` (soft-core LJ, Beutler form with a=1, b=1, c=6)
    ligand x environment only (one interaction group), scaled by ``lambda_sterics``.

The cuPotential backend replaces only the NonbondedForce; the soft-core force stays on
the stock CUDA kernels in the same System.
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
MAIN_GROUP = 0
SOFTCORE_GROUP = 1
RESTRAINT_GROUP = 2

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


class AlchemyError(ValueError):
    """The System cannot be made alchemical as requested."""


@dataclass
class AlchemicalSystem:
    system: mm.System
    ligand_atoms: tuple[int, ...]
    softcore_alpha: float
    #: original (charge [e], sigma [nm], epsilon [kJ/mol]) of the ligand atoms
    ligand_parameters: tuple[tuple[float, float, float], ...]

    @property
    def lambda_parameters(self) -> tuple[str, ...]:
        return (LAMBDA_ELECTROSTATICS, LAMBDA_STERICS)


def make_alchemical_system(system: mm.System, ligand_atoms, softcore_alpha: float = 0.5) -> AlchemicalSystem:
    """Return an alchemical copy of ``system`` in which ``ligand_atoms`` can be decoupled."""
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
    existing: set[tuple[int, int]] = set()
    for k in range(nb.getNumExceptions()):
        i, j, _qq, _sig, _eps = nb.getExceptionParameters(k)
        in_i, in_j = i in ligand_set, j in ligand_set
        if in_i != in_j:
            raise AlchemyError("an exception connects the ligand to the environment (covalent ligand?)")
        if in_i:
            existing.add((min(i, j), max(i, j)))
    for a, i in enumerate(ligand):
        qi, si, ei = params[i]
        for j in ligand[a + 1:]:
            if (i, j) in existing:
                continue
            qj, sj, ej = params[j]
            nb.addException(i, j, qi * qj, 0.5 * (si + sj), math.sqrt(ei * ej))

    # -- NonbondedForce: charges through offsets, no ligand LJ -------------------
    nb.addGlobalParameter(LAMBDA_ELECTROSTATICS, 1.0)
    for i in ligand:
        q, sig, _eps = params[i]
        nb.setParticleParameters(i, 0.0, sig, 0.0)
        if q != 0.0:
            nb.addParticleParameterOffset(LAMBDA_ELECTROSTATICS, i, q, 0.0, 0.0)

    # -- soft-core LJ: ligand x environment --------------------------------------
    sc = mm.CustomNonbondedForce(SOFTCORE_ENERGY)
    sc.setName(SOFTCORE_FORCE_NAME)
    sc.addGlobalParameter(LAMBDA_STERICS, 1.0)
    sc.addGlobalParameter("softcore_alpha", float(softcore_alpha))
    sc.addPerParticleParameter("sigma")
    sc.addPerParticleParameter("epsilon")
    for q, sig, eps in params:
        # sigma is irrelevant when epsilon is zero; keep it finite to avoid r/0.
        sc.addParticle([sig if sig > 0.0 else 0.3, eps])
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

    return AlchemicalSystem(system=out, ligand_atoms=ligand, softcore_alpha=float(softcore_alpha),
                            ligand_parameters=tuple(params[i] for i in ligand))


def set_lambdas(context: mm.Context, lambdas: dict[str, float]) -> None:
    """Set alchemical parameters on a Context (only those the Context knows)."""
    known = context.getParameters()
    for name, value in lambdas.items():
        if name in known:
            context.setParameter(name, float(value))
