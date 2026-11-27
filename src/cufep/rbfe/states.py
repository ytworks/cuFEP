"""States of a hybrid System and the cheap evaluation of all their energies.

A state is the position ``lambda_hybrid`` (x) on the one alchemical coordinate of a relative transformation,
from ligand A (0) to ligand B (1), and with REST2 the temperature scale of the hot region. The leg has one
stage, ``morph`` (the name of the stage of Sire's standard_morph schedule, which SOMD2 uses for relative
transformations). Each lambda of the hybrid System is OpenFE's default function of x (the
``LambdaProtocol`` of perses, ``openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py``):

    lambda_sterics_core = lambda_electrostatics_core = lambda_bonds = lambda_angles = lambda_torsions = x
    lambda_electrostatics_delete = 2x for x < 0.5, else 1      (charges of the unique old atoms off)
    lambda_sterics_insert        = 2x for x < 0.5, else 1      (Lennard-Jones of the unique new atoms on)
    lambda_sterics_delete        = 0 for x < 0.5, else 2(x - 0.5)
    lambda_electrostatics_insert = 0 for x < 0.5, else 2(x - 0.5)

and, cuFEP's own, the lambdas of the bonded terms of the dummy atoms that ``cufep.rbfe.junction`` changes (a mapped
atom with unmapped atoms of both ligands):

    lambda_junction_delete = 0 for x <= 0.4, 10 (x - 0.4) up to 0.5, then 1   (those of A: the force field's -> dummy's)
    lambda_junction_insert = 0 for x <= 0.5, 10 (x - 0.5) up to 0.6, then 1   (those of B: dummy's -> force field's)

so that from 0 to 0.4 ligand A has its own terms and the dummy atoms of B theirs (A turns about the bonds of such an
atom as it would alone), and from 0.6 to 1 the same of B, with the hot windows of REST2 next to the middle: a
conformer that A or B reaches only in a hot window is carried to its end by the replica exchange. The two kinds of
terms of a group, which anchor it to different neighbours, are both partly on only in 0.4-0.5 and 0.5-0.6 (where they
change with the steric lambdas, as version 3 did, they were in most windows, and the urea C-N bond of ejm_54 in
ejm_42~ejm_54 could not turn in any window but the last: a torsional barrier of 220 to 640 kJ/mol with the group
turned with it, against 44 to 50 kJ/mol divided by the REST2 scale in the windows from 0.62 to 0.88 with the terms
of this rule, spikes/out/m44).

The three electrostatic lambdas of the System carry sqrt(s) with REST2 (``cufep.rbfe.hybrid``).

Energy of all states (``cufep.rbfe.hybrid`` for the groups):

    U(x) = G0(x) + G1(x) + G4(x) + s G5(x) + s G3

G0 holds the NonbondedForce and the Coulomb energy inside the ligand: in each half of x the charges are
linear in x (and with REST2 linear in g = x sqrt(s) and a = sqrt(s)), so G0 is an exact quadratic there:
three evaluations per half give it (with REST2 six per half, the quadratic in (g, a) of the ABFE). G1 (the
Lennard-Jones of the ligand with the environment), G4 (core bonds, angles and improper torsions) and G5 (the
Lennard-Jones inside the ligand and the core proper torsions) are evaluated once per distinct x; with REST2
G1 is a A(x) + s B(x) as in the ABFE. G3 holds the proper torsions of the hot region that do not change.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
import openmm as mm
import openmm.unit as unit

from cufep.alchemy.factory import MAIN_GROUP, REST2_DQ, REST2_DS, REST2_GROUP, SOFTCORE_GROUP
from cufep.engine.protocol import HYBRID, canonical_stage, rest2_scale_along_leg
from cufep.rbfe.hybrid import (
    COLD_ELECTROSTATIC_LAMBDAS,
    ELECTROSTATIC_LAMBDAS,
    HYBRID_GROUP,
    HYBRID_HOT_GROUP,
    HYBRID_LJ_FORCE_NAME,
    LAMBDA_ANGLES,
    LAMBDA_BONDS,
    LAMBDA_ELECTROSTATICS_CORE,
    LAMBDA_ELECTROSTATICS_DELETE,
    LAMBDA_ELECTROSTATICS_INSERT,
    LAMBDA_JUNCTION_DELETE,
    LAMBDA_JUNCTION_INSERT,
    LAMBDA_STERICS_CORE,
    LAMBDA_STERICS_DELETE,
    LAMBDA_STERICS_INSERT,
    LAMBDA_TORSIONS,
    STERIC_LAMBDAS,
)
from cufep.sampling.states import BAR_NM3_TO_KJ_MOL, R_GAS

_KJ = unit.kilojoule_per_mole
#: the stage of a relative transformation
MORPH = "morph"


#: the ranges of x in which the bonded terms of the dummy atoms of A and of B change (module docstring)
JUNCTION_DELETE_RANGE = (0.4, 0.5)
JUNCTION_INSERT_RANGE = (0.5, 0.6)


def _ramp(x: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def lambda_functions(x: float) -> dict[str, float]:
    """OpenFE's default lambda functions at ``x``, and the lambdas of the bonded terms of the dummy atoms."""
    x = float(x)
    first = x < 0.5
    return {LAMBDA_STERICS_CORE: x, LAMBDA_ELECTROSTATICS_CORE: x,
            LAMBDA_STERICS_INSERT: 2.0 * x if first else 1.0,
            LAMBDA_STERICS_DELETE: 0.0 if first else 2.0 * (x - 0.5),
            LAMBDA_ELECTROSTATICS_INSERT: 0.0 if first else 2.0 * (x - 0.5),
            LAMBDA_ELECTROSTATICS_DELETE: 2.0 * x if first else 1.0,
            LAMBDA_BONDS: x, LAMBDA_ANGLES: x, LAMBDA_TORSIONS: x,
            LAMBDA_JUNCTION_DELETE: _ramp(x, *JUNCTION_DELETE_RANGE),
            LAMBDA_JUNCTION_INSERT: _ramp(x, *JUNCTION_INSERT_RANGE)}


@dataclass(frozen=True)
class HybridState:
    #: the protocol of the state (cufep.engine.protocol)
    protocol: ClassVar[str] = HYBRID
    lambda_hybrid: float = 0.0
    #: REST2: effective temperature of the hot region relative to that of the run (as AlchemicalState)
    rest2_scale: float = 1.0

    @property
    def rest2_energy_scale(self) -> float:
        return 1.0 / self.rest2_scale

    def as_parameters(self) -> dict[str, float]:
        s = self.rest2_energy_scale
        a = math.sqrt(s)
        out = lambda_functions(self.lambda_hybrid)
        for name, cold in zip(ELECTROSTATIC_LAMBDAS, COLD_ELECTROSTATIC_LAMBDAS):
            out[cold] = out[name]              # the charges REST2 does not scale (a System without them ignores it)
            out[name] = out[name] * a          # the hot atoms of the ligand: their charges carry sqrt(s)
        out[REST2_DQ] = a - 1.0
        out[REST2_DS] = s - 1.0
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def validate_hybrid_states(states: list[HybridState]) -> None:
    if not all(isinstance(s, HybridState) for s in states):
        raise ValueError("the states are not all hybrid states")
    if len(states) < 2:
        raise ValueError("at least two thermodynamic states are needed")
    for s in states:
        if not 0.0 <= s.lambda_hybrid <= 1.0:
            raise ValueError(f"lambda outside [0, 1] in {s}")
        if s.rest2_scale < 1.0:
            raise ValueError(f"{s}: rest2_scale must be at least 1")
    for s in (states[0], states[-1]):
        if abs(s.rest2_scale - 1.0) > 1e-9:
            raise ValueError("REST2 scaling must be off (rest2_scale 1) in the first and in the last state")


def _electrostatics(half: int, g: float, a: float) -> dict[str, float]:
    """The three electrostatic parameters as affine functions of (g, a) = (x sqrt(s), sqrt(s)) in one half:
    the first half x <= 0.5, the second x >= 0.5 (both give the same values at x = 0.5)."""
    if half == 0:
        return {LAMBDA_ELECTROSTATICS_CORE: g, LAMBDA_ELECTROSTATICS_DELETE: 2.0 * g, LAMBDA_ELECTROSTATICS_INSERT: 0.0}
    return {LAMBDA_ELECTROSTATICS_CORE: g, LAMBDA_ELECTROSTATICS_DELETE: a, LAMBDA_ELECTROSTATICS_INSERT: 2.0 * g - a}


def _charges(half: int, x: float, a: float) -> dict[str, float]:
    """The electrostatic lambdas of the hot charges (times a = sqrt(s)) and of the cold ones at ``x`` in one half."""
    values = _electrostatics(half, x, 1.0)
    out = {name: values[name] * a for name in ELECTROSTATIC_LAMBDAS}
    out.update({cold: values[name] for name, cold in zip(ELECTROSTATIC_LAMBDAS, COLD_ELECTROSTATIC_LAMBDAS)})
    return out


#: the values of a = sqrt(s) at which a group whose terms carry 1, a or a^2 is evaluated (a System with cold charges)
_A_DESIGN = (1.0, 0.75, 0.5)


#: (g, a) at which the main group is evaluated in a half with REST2 (those of the ABFE, cufep.sampling.states)
_MAIN_DESIGN = ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.0, 0.5), (1.0, 0.5), (0.0, 0.75))


def _main_basis(g, a) -> np.ndarray:
    g, a = np.asarray(g, dtype=float), np.asarray(a, dtype=float)
    return np.stack([np.ones_like(g), g, g * g, a, a * a, g * a], axis=-1)


_MAIN_SOLVE = np.linalg.inv(_main_basis(*np.array(_MAIN_DESIGN).T))


class HybridEnergyEvaluator:
    """Reduced potentials of the Context's current configuration at every hybrid state."""

    def __init__(self, context: mm.Context, states: list[HybridState], temperature_K: float,
                 pressure_bar: float | None) -> None:
        self.context = context
        self.states = list(states)
        self.beta = 1.0 / (R_GAS * temperature_K)
        self.pressure_bar = pressure_bar
        params = context.getParameters()
        self._known = set(params)
        self.has_rest2 = REST2_DS in params
        self._x = np.array([s.lambda_hybrid for s in self.states])
        self._s = np.array([s.rest2_energy_scale for s in self.states])
        self._a = np.sqrt(self._s)
        self._x_unique = np.unique(self._x)
        self.rest2_active = bool(np.any(np.abs(self._s - 1.0) > 1e-12))
        if self.rest2_active and not self.has_rest2:
            raise ValueError("states with a REST2 scale need a System built with REST2")
        self._hot_environment = self.rest2_active and self._has_hot_environment()
        groups = {f.getForceGroup() for f in context.getSystem().getForces()}
        self._has = {g: g in groups for g in (SOFTCORE_GROUP, REST2_GROUP, HYBRID_GROUP, HYBRID_HOT_GROUP)}
        # a System with charges that REST2 does not scale (a ligand hot in part, an alchemical water): every group
        # is a polynomial in a = sqrt(s) of terms of 1, a and a^2, and the main group in (x, a) of degree two in each
        self.by_atom = COLD_ELECTROSTATIC_LAMBDAS[0] in params

    def _has_hot_environment(self) -> bool:
        for force in self.context.getSystem().getForces():
            if isinstance(force, mm.CustomNonbondedForce) and force.getName() == HYBRID_LJ_FORCE_NAME:
                names = [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())]
                if "hot" not in names:
                    return False
                column = names.index("hot")
                environment = force.getInteractionGroupParameters(0)[1]
                return any(force.getParticleParameters(i)[column] != 0.0 for i in environment)
        return False

    def _group(self, group: int) -> float:
        return self.context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(_KJ)

    def _set(self, values: dict[str, float]) -> None:
        # a System has the lambdas of its forces only (the junction lambdas where cufep.rbfe.junction changed terms)
        for name, value in values.items():
            if name in self._known:
                self.context.setParameter(name, float(value))

    def _set_rest2(self, a: float) -> None:
        if self.has_rest2:
            self._set({REST2_DQ: float(a) - 1.0, REST2_DS: float(a) * float(a) - 1.0})

    def _set_x(self, x: float) -> None:
        """Every lambda at ``x`` without REST2 scaling (the charges included)."""
        values = lambda_functions(x)
        if self.by_atom:
            values.update({cold: values[name] for name, cold in zip(ELECTROSTATIC_LAMBDAS, COLD_ELECTROSTATIC_LAMBDAS)})
        self._set(values)

    def _in_a(self, group: int) -> np.ndarray:
        """A group whose terms carry 1, a or a^2 at the a of each state: three evaluations per distinct x."""
        table = {}
        for x in self._x_unique:
            values = []
            for a in _A_DESIGN:
                self._set_rest2(a)
                self._set_x(float(x))
                values.append(self._group(group))
            table[float(x)] = np.polyfit(_A_DESIGN, values, 2)
        self._set_rest2(1.0)
        return np.array([np.polyval(table[float(x)], a) for x, a in zip(self._x, self._a)])

    def _main_by_atom(self, half: int, members: np.ndarray) -> np.ndarray:
        """The main group with cold charges: of degree two in x (in one half) and in a, nine evaluations."""
        xs = (0.0, 0.25, 0.5) if half == 0 else (0.5, 0.75, 1.0)
        values = np.empty((3, 3))
        for i, x in enumerate(xs):
            for j, a in enumerate(_A_DESIGN):
                self._set_rest2(a)
                self._set(_charges(half, x, a))
                values[i, j] = self._group(MAIN_GROUP)
        vx, va = np.vander(xs, 3, increasing=True), np.vander(_A_DESIGN, 3, increasing=True)
        coefficients = np.linalg.solve(vx, np.linalg.solve(va, values.T).T)
        x, a = self._x[members], self._a[members]
        return np.einsum("kp,pq,kq->k", np.vander(x, 3, increasing=True), coefficients,
                         np.vander(a, 3, increasing=True))

    def main_group(self) -> np.ndarray:
        out = np.empty(len(self.states))
        halves = (self._x <= 0.5, self._x > 0.5)
        for half, members in enumerate(halves):
            if not np.any(members):
                continue
            if self.by_atom:
                out[members] = self._main_by_atom(half, members)
            elif self.rest2_active:
                values = []
                for g, a in _MAIN_DESIGN:
                    self._set_rest2(a)
                    self._set(_electrostatics(half, g, a))
                    values.append(self._group(MAIN_GROUP))
                coefficients = _MAIN_SOLVE @ np.array(values)
                out[members] = _main_basis(self._x[members] * self._a[members], self._a[members]) @ coefficients
            else:
                self._set_rest2(1.0)
                xs = (0.0, 0.25, 0.5) if half == 0 else (0.5, 0.75, 1.0)
                values = []
                for x in xs:
                    self._set(_electrostatics(half, x, 1.0))
                    values.append(self._group(MAIN_GROUP))
                coefficients = np.polyfit(xs, values, 2)
                out[members] = np.polyval(coefficients, self._x[members])
        return out

    def _per_x(self, group: int, a: float = 1.0) -> np.ndarray:
        self._set_rest2(a)
        table = {}
        for x in self._x_unique:
            self._set_x(float(x))
            table[float(x)] = self._group(group)
        return np.array([table[float(x)] for x in self._x])

    def softcore_group(self) -> np.ndarray:
        if not self._has[SOFTCORE_GROUP]:
            return np.zeros(len(self.states))
        if self.by_atom and self.rest2_active:
            return self._in_a(SOFTCORE_GROUP)
        full = self._per_x(SOFTCORE_GROUP)                       # at a = 1: A + B
        if not self.rest2_active:
            return full
        if not self._hot_environment:                            # every pair carries a
            return self._a * full
        half = self._per_x(SOFTCORE_GROUP, 0.5)                  # A/2 + B/4
        self._set_rest2(1.0)
        cold = 4.0 * half - full
        return self._a * cold + self._s * (full - cold)

    def hybrid_groups(self) -> np.ndarray:
        out = np.zeros(len(self.states))
        if self._has[HYBRID_GROUP]:
            out += self._per_x(HYBRID_GROUP)
        if self._has[HYBRID_HOT_GROUP]:
            out += self._in_a(HYBRID_HOT_GROUP) if self.by_atom and self.rest2_active \
                else self._s * self._per_x(HYBRID_HOT_GROUP)
        return out

    def rest2_group(self) -> np.ndarray:
        if not self.has_rest2 or not self._has[REST2_GROUP]:
            return np.zeros(len(self.states))
        self._set_rest2(1.0)
        return self._s * self._group(REST2_GROUP)

    def volume_nm3(self) -> float:
        return self.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)

    def reduced_potentials(self, restore: HybridState | None = None) -> tuple[np.ndarray, float]:
        """Return ``(u_k, volume)``; ``restore`` re-applies a state afterwards."""
        # one evaluation of all groups first: the interaction-group lists of the CustomNonbondedForce are
        # rebuilt only with the standard nonbonded kernel (cufep.sampling.states)
        self.context.getState(getEnergy=True)
        energy = self.main_group() + self.softcore_group() + self.hybrid_groups() + self.rest2_group()
        volume = self.volume_nm3()
        if self.pressure_bar is not None:
            energy = energy + self.pressure_bar * volume * BAR_NM3_TO_KJ_MOL
        if restore is not None:
            for name, value in restore.as_parameters().items():
                if name in self.context.getParameters():
                    self.context.setParameter(name, float(value))
        return self.beta * energy, volume


class _Hybrid:
    """The protocol of a relative transformation through a hybrid topology (:mod:`cufep.engine.protocol`)."""

    name = HYBRID
    stages = (MORPH,)
    stage_parameter = {MORPH: LAMBDA_STERICS_CORE}

    @staticmethod
    def leg_stages(leg: str) -> tuple[str, ...]:
        return (MORPH,)

    @staticmethod
    def stage_state(stage: str, lam: float, stages=None, rest2_scale: float = 1.0) -> HybridState:
        if canonical_stage(stage) != MORPH:
            raise ValueError(f"'{stage}' is not the stage of a relative transformation")
        lam = float(min(1.0, max(0.0, lam)))
        scale = 1.0 if rest2_scale == 1.0 else rest2_scale_along_leg(stages or [MORPH], MORPH, lam, rest2_scale)
        return HybridState(lam, scale)

    @staticmethod
    def state_from_dict(d: dict) -> HybridState:
        return HybridState(**d)

    @staticmethod
    def evaluator(context, states, temperature_K: float, pressure_bar: float | None) -> "HybridEnergyEvaluator":
        return HybridEnergyEvaluator(context, states, temperature_K, pressure_bar)

    @staticmethod
    def validate(states) -> None:
        validate_hybrid_states(states)


PROTOCOL = _Hybrid()

__all__ = ["HybridState", "HybridEnergyEvaluator", "lambda_functions", "validate_hybrid_states", "MORPH",
           "STERIC_LAMBDAS", "PROTOCOL"]
