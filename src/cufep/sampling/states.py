"""Thermodynamic states and cheap evaluation of all state energies.

A state is the triple (lambda_restraints, lambda_electrostatics, lambda_sterics), and with
REST2 the effective temperature of the hot region relative to that of the run.
The path for the complex leg is

    A  restraints on      (0,1,1) -> (1,1,1)
    B  charges off        (1,1,1) -> (1,0,1)
    C  Lennard-Jones off  (1,0,1) -> (1,0,0)

and B, C for the solvent leg (no restraint). Electrostatics must be fully off before the
soft-core Lennard-Jones stage starts; there is no soft-core for Coulomb.

Energy decomposition (see ``alchemy/factory.py`` for the force groups):

    U(state) = G0(lambda_elec) + G1(lambda_sterics) + lambda_restraints * G2

G0 is an exact quadratic in lambda_elec (charges scale linearly, and Ewald energies are a
quadratic form in the charges), so three evaluations give it for every state. G1 is
evaluated once per distinct lambda_sterics. This avoids one full parameter update and
energy evaluation per state, which is the expensive path on the cuPotential plugin.

With REST2 (energy scale s = 1 / rest2_scale of the hot region, a = sqrt(s)):

    U(state) = G0(g, a) + G1(lambda_sterics, a) + lambda_restraints * G2 + s * G3

g = lambda_elec * a is the factor of the ligand charges. G0 is an exact quadratic polynomial
in (g, a): charges are linear in g (ligand) and a (hot environment atoms), Lennard-Jones
pairs carry a for each hot atom, and exception pairs s or a. Six evaluations give its six
coefficients. G1 = a * A(lambda_sterics) + a^2 * B(lambda_sterics): ligand atoms are hot, so a
pair with a cold atom carries a and a pair with a hot environment atom a^2; two evaluations
per lambda_sterics. G3 is the energy of the proper torsions inside the hot region.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import openmm as mm
import openmm.unit as unit

from cufep.alchemy.factory import (
    LAMBDA_ELECTROSTATICS,
    LAMBDA_RESTRAINTS,
    LAMBDA_STERICS,
    MAIN_GROUP,
    REST2_DQ,
    REST2_DS,
    REST2_GROUP,
    RESTRAINT_GROUP,
    SOFTCORE_FORCE_NAME,
    SOFTCORE_GROUP,
)

R_GAS = 8.31446261815324e-3            # kJ/mol/K
BAR_NM3_TO_KJ_MOL = 0.0602214076       # 1 bar nm^3 in kJ/mol
_KJ = unit.kilojoule_per_mole


@dataclass(frozen=True)
class AlchemicalState:
    lambda_restraints: float = 1.0
    lambda_electrostatics: float = 1.0
    lambda_sterics: float = 1.0
    #: REST2: effective temperature of the hot region relative to the temperature of the run
    #: (T_hot / T, the meaning of SOMD2's rest2_scale); 1 is the physical Hamiltonian
    rest2_scale: float = 1.0

    @property
    def rest2_energy_scale(self) -> float:
        """s = T / T_hot, the factor of the energy inside the hot region."""
        return 1.0 / self.rest2_scale

    def as_parameters(self) -> dict[str, float]:
        s = self.rest2_energy_scale
        a = math.sqrt(s)
        return {LAMBDA_RESTRAINTS: self.lambda_restraints,
                # the ligand is hot: its charges carry sqrt(s) on top of lambda
                LAMBDA_ELECTROSTATICS: self.lambda_electrostatics * a,
                LAMBDA_STERICS: self.lambda_sterics,
                REST2_DQ: a - 1.0, REST2_DS: s - 1.0}

    def to_dict(self) -> dict:
        return asdict(self)


#: Stages of a leg, in the order in which they are passed. The names are those of
#: A3FE: restrain (restraint off -> on), discharge (charges on -> off), vanish (LJ on -> off).
STAGES_COMPLEX = ("restrain", "discharge", "vanish")
STAGES_SOLVENT = ("discharge", "vanish")
#: names used by earlier versions (still accepted when reading files)
STAGE_ALIASES = {"restraints": "restrain", "electrostatics": "discharge", "sterics": "vanish"}


def canonical_stage(name: str) -> str:
    name = STAGE_ALIASES.get(name, name)
    if name not in STAGES_COMPLEX:
        raise ValueError(f"unknown stage '{name}'")
    return name


def rest2_scale_along_leg(stages, stage: str, lam: float, peak: float) -> float:
    """REST2 temperature scale at lambda ``lam`` of ``stage`` of a leg with the given stages.

    The leg is one path from 0 to 1 on which every stage takes an equal part. The scale is 1 at
    both ends and ``peak`` in the middle, linear in between: SOMD2's rule
    1 + (peak - 1) * (1 - 2 |x - 0.5|) with x the position on that path.
    """
    if peak == 1.0:
        return 1.0
    names = [canonical_stage(n) for n in stages]
    x = (names.index(canonical_stage(stage)) + float(lam)) / len(names)
    return 1.0 + (float(peak) - 1.0) * (1.0 - 2.0 * abs(x - 0.5))


def stage_state(stage: str, lam: float, stages=None, rest2_scale: float = 1.0) -> AlchemicalState:
    """State at lambda ``lam`` of one stage (0 = start of the stage, 1 = its end).

    With REST2 (``rest2_scale`` above 1, the scale in the middle of the leg) the stages of the
    leg are needed too, since the scale of a state depends on where it is along the leg.
    """
    lam = float(min(1.0, max(0.0, lam)))
    stage = canonical_stage(stage)
    if stage == "restrain":
        state = AlchemicalState(lam, 1.0, 1.0)
    elif stage == "discharge":
        state = AlchemicalState(1.0, 1.0 - lam, 1.0)
    else:
        state = AlchemicalState(1.0, 0.0, 1.0 - lam)
    if rest2_scale != 1.0:
        if stages is None:
            raise ValueError("the stages of the leg are needed to place a REST2 scale")
        state = replace(state, rest2_scale=rest2_scale_along_leg(stages, stage, lam, rest2_scale))
    return state


def validate_states(states: list[AlchemicalState]) -> None:
    if len(states) < 2:
        raise ValueError("at least two thermodynamic states are needed")
    for s in states:
        for v in (s.lambda_restraints, s.lambda_electrostatics, s.lambda_sterics):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"lambda outside [0, 1] in {s}")
        if s.lambda_sterics < 1.0 and s.lambda_electrostatics > 0.0:
            raise ValueError(f"{s}: electrostatics must be fully off before sterics are scaled "
                             "(the soft-core potential covers Lennard-Jones only)")
        if s.rest2_scale < 1.0:
            raise ValueError(f"{s}: rest2_scale must be at least 1")
    for s in (states[0], states[-1]):
        if abs(s.rest2_scale - 1.0) > 1e-9:
            raise ValueError("REST2 scaling must be off (rest2_scale 1) in the first and in the last state")


def set_state(context: mm.Context, state: AlchemicalState) -> None:
    known = context.getParameters()
    if state.rest2_scale != 1.0 and REST2_DS not in known:
        raise ValueError("the state has a REST2 scale but the System was built without REST2")
    for name, value in state.as_parameters().items():
        if name in known:
            context.setParameter(name, float(value))


#: (g, a) at which the main group is evaluated to get its quadratic polynomial with REST2
_MAIN_DESIGN = ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.0, 0.5), (1.0, 0.5), (0.0, 0.75))


def _main_basis(g, a) -> np.ndarray:
    g, a = np.asarray(g, dtype=float), np.asarray(a, dtype=float)
    return np.stack([np.ones_like(g), g, g * g, a, a * a, g * a], axis=-1)


_MAIN_SOLVE = np.linalg.inv(_main_basis(*np.array(_MAIN_DESIGN).T))


class StateEnergyEvaluator:
    """Reduced potentials of the Context's current configuration at every state."""

    def __init__(self, context: mm.Context, states: list[AlchemicalState], temperature_K: float,
                 pressure_bar: float | None) -> None:
        self.context = context
        self.states = list(states)
        self.beta = 1.0 / (R_GAS * temperature_K)
        self.pressure_bar = pressure_bar
        params = context.getParameters()
        self.has_restraint = LAMBDA_RESTRAINTS in params
        self.has_elec = LAMBDA_ELECTROSTATICS in params
        self.has_sterics = LAMBDA_STERICS in params
        self.has_rest2 = REST2_DS in params
        self._le = np.array([s.lambda_electrostatics for s in self.states])
        self._ls = np.array([s.lambda_sterics for s in self.states])
        self._lr = np.array([s.lambda_restraints for s in self.states])
        self._s = np.array([s.rest2_energy_scale for s in self.states])
        self._a = np.sqrt(self._s)
        self._le_unique = np.unique(self._le)
        self._ls_unique = np.unique(self._ls)
        #: REST2 changes the energies only if some state is scaled
        self.rest2_active = bool(np.any(np.abs(self._s - 1.0) > 1e-12))
        if self.rest2_active and not self.has_rest2:
            raise ValueError("states with a REST2 scale need a System built with REST2")
        self._hot_environment = self.rest2_active and self._has_hot_environment()

    def _has_hot_environment(self) -> bool:
        """Whether the soft-core force has hot atoms on the environment side."""
        system = self.context.getSystem()
        for force in system.getForces():
            if isinstance(force, mm.CustomNonbondedForce) and force.getName() == SOFTCORE_FORCE_NAME:
                names = [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())]
                if "hot" not in names:
                    return False
                column = names.index("hot")
                _ligand, environment = force.getInteractionGroupParameters(0)
                return any(force.getParticleParameters(i)[column] != 0.0 for i in environment)
        return False

    def _group(self, group: int) -> float:
        st = self.context.getState(getEnergy=True, groups={group})
        return st.getPotentialEnergy().value_in_unit(_KJ)

    def _set_rest2(self, a: float) -> None:
        self.context.setParameter(REST2_DQ, float(a) - 1.0)
        self.context.setParameter(REST2_DS, float(a) * float(a) - 1.0)

    def main_group(self) -> np.ndarray:
        """G0 at every state."""
        if self.rest2_active:
            values = []
            for g, a in _MAIN_DESIGN:
                self._set_rest2(a)
                if self.has_elec:
                    self.context.setParameter(LAMBDA_ELECTROSTATICS, g)
                values.append(self._group(MAIN_GROUP))
            coefficients = _MAIN_SOLVE @ np.array(values)
            return _main_basis(self._le * self._a, self._a) @ coefficients
        if self.has_rest2:
            self._set_rest2(1.0)
        if not self.has_elec:
            return np.full(len(self.states), self._group(MAIN_GROUP))
        if len(self._le_unique) <= 3:
            table = {}
            for le in self._le_unique:
                self.context.setParameter(LAMBDA_ELECTROSTATICS, float(le))
                table[float(le)] = self._group(MAIN_GROUP)
            return np.array([table[float(le)] for le in self._le])
        g = []
        for le in (0.0, 0.5, 1.0):
            self.context.setParameter(LAMBDA_ELECTROSTATICS, le)
            g.append(self._group(MAIN_GROUP))
        a2 = 2.0 * (g[2] - 2.0 * g[1] + g[0])
        a1 = g[2] - g[0] - a2
        return g[0] + a1 * self._le + a2 * self._le ** 2

    def softcore_group(self) -> np.ndarray:
        if not self.has_sterics:
            return np.zeros(len(self.states))
        if self.has_rest2:
            self._set_rest2(1.0)
        table = {}
        for ls in self._ls_unique:
            self.context.setParameter(LAMBDA_STERICS, float(ls))
            table[float(ls)] = self._group(SOFTCORE_GROUP)
        full = np.array([table[float(ls)] for ls in self._ls])            # at a = 1: A + B
        if not self.rest2_active:
            return full
        if not self._hot_environment:                                      # B = 0: every pair carries a
            return self._a * full
        self._set_rest2(0.5)
        for ls in self._ls_unique:
            self.context.setParameter(LAMBDA_STERICS, float(ls))
            table[float(ls)] = self._group(SOFTCORE_GROUP)
        half = np.array([table[float(ls)] for ls in self._ls])            # at a = 1/2: A/2 + B/4
        self._set_rest2(1.0)
        cold = 4.0 * half - full                                           # A: pairs with a cold atom
        hot = full - cold                                                  # B: pairs with a hot environment atom
        return self._a * cold + self._s * hot

    def rest2_group(self) -> np.ndarray:
        """s * (energy of the proper torsions inside the hot region) at every state."""
        if not self.has_rest2:
            return np.zeros(len(self.states))
        self._set_rest2(1.0)
        return self._s * self._group(REST2_GROUP)

    def restraint_energy(self) -> float:
        if not self.has_restraint:
            return 0.0
        self.context.setParameter(LAMBDA_RESTRAINTS, 1.0)
        return self._group(RESTRAINT_GROUP)

    def volume_nm3(self) -> float:
        return self.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)

    def reduced_potentials(self, restore: AlchemicalState | None = None) -> tuple[np.ndarray, float]:
        """Return ``(u_k, volume)``; ``restore`` re-applies a state afterwards."""
        if self.has_sterics:
            # OpenMM rebuilds the neighbour list of an interaction-group CustomNonbondedForce (the
            # soft-core force) only in a computation that also runs the standard nonbonded kernel.
            # Evaluating the soft-core group on its own after the positions changed would use a
            # stale list (errors of several kT). One evaluation of all groups brings both lists
            # up to date for the current positions; the per-group evaluations below then agree.
            self.context.getState(getEnergy=True)
        energy = self.main_group() + self.softcore_group() + self.rest2_group()
        if self.has_restraint:
            energy = energy + self._lr * self.restraint_energy()
        volume = self.volume_nm3()
        if self.pressure_bar is not None:
            energy = energy + self.pressure_bar * volume * BAR_NM3_TO_KJ_MOL
        if restore is not None:
            set_state(self.context, restore)
        return self.beta * energy, volume
