"""Thermodynamic states and cheap evaluation of all state energies.

A state is the triple (lambda_restraints, lambda_electrostatics, lambda_sterics).
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
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import openmm as mm
import openmm.unit as unit

from cufep.alchemy.factory import (LAMBDA_ELECTROSTATICS, LAMBDA_RESTRAINTS, LAMBDA_STERICS, MAIN_GROUP,
                                   RESTRAINT_GROUP, SOFTCORE_GROUP)

R_GAS = 8.31446261815324e-3            # kJ/mol/K
BAR_NM3_TO_KJ_MOL = 0.0602214076       # 1 bar nm^3 in kJ/mol
_KJ = unit.kilojoule_per_mole


@dataclass(frozen=True)
class AlchemicalState:
    lambda_restraints: float = 1.0
    lambda_electrostatics: float = 1.0
    lambda_sterics: float = 1.0

    def as_parameters(self) -> dict[str, float]:
        return {LAMBDA_RESTRAINTS: self.lambda_restraints,
                LAMBDA_ELECTROSTATICS: self.lambda_electrostatics,
                LAMBDA_STERICS: self.lambda_sterics}

    def to_dict(self) -> dict:
        return asdict(self)


STAGES_COMPLEX = ("restraints", "electrostatics", "sterics")
STAGES_SOLVENT = ("electrostatics", "sterics")


def stage_state(stage: str, t: float) -> AlchemicalState:
    """State at progress ``t`` in [0, 1] along one stage."""
    t = float(min(1.0, max(0.0, t)))
    if stage == "restraints":
        return AlchemicalState(t, 1.0, 1.0)
    if stage == "electrostatics":
        return AlchemicalState(1.0, 1.0 - t, 1.0)
    if stage == "sterics":
        return AlchemicalState(1.0, 0.0, 1.0 - t)
    raise ValueError(f"unknown stage '{stage}'")


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


def set_state(context: mm.Context, state: AlchemicalState) -> None:
    known = context.getParameters()
    for name, value in state.as_parameters().items():
        if name in known:
            context.setParameter(name, float(value))


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
        self._le = np.array([s.lambda_electrostatics for s in self.states])
        self._ls = np.array([s.lambda_sterics for s in self.states])
        self._lr = np.array([s.lambda_restraints for s in self.states])
        self._le_unique = np.unique(self._le)
        self._ls_unique = np.unique(self._ls)

    def _group(self, group: int) -> float:
        st = self.context.getState(getEnergy=True, groups={group})
        return st.getPotentialEnergy().value_in_unit(_KJ)

    def main_group(self) -> np.ndarray:
        """G0 at every state's lambda_electrostatics."""
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
        table = {}
        for ls in self._ls_unique:
            self.context.setParameter(LAMBDA_STERICS, float(ls))
            table[float(ls)] = self._group(SOFTCORE_GROUP)
        return np.array([table[float(ls)] for ls in self._ls])

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
        energy = self.main_group() + self.softcore_group()
        if self.has_restraint:
            energy = energy + self._lr * self.restraint_energy()
        volume = self.volume_nm3()
        if self.pressure_bar is not None:
            energy = energy + self.pressure_bar * volume * BAR_NM3_TO_KJ_MOL
        if restore is not None:
            set_state(self.context, restore)
        return self.beta * energy, volume
