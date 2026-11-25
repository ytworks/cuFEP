"""The protocols of a leg: the lambda states a leg passes through, its stages, and how the energies of all its
states are evaluated.

A leg goes from one end state to the other through lambda windows, and the engine of a leg samples them by
Hamiltonian replica exchange whatever its states are. Two protocols:

- ``abfe``: the double decoupling of a ligand (:mod:`cufep.sampling.states`): the stages restrain, discharge and
  vanish in the complex leg, discharge and vanish in the solvent leg;
- ``hybrid``: the transformation of ligand A into ligand B through a hybrid topology (:mod:`cufep.rbfe.states`):
  the one stage morph.

A stage belongs to one protocol, and a state knows its protocol (``state.protocol``). The protocol of a leg is
named in its spec (``LegSpec.protocol``, with the conditions of its samples when it is not the ABFE) and in its
schedule file when it is not the ABFE; a schedule file that names none is of the protocol of its stages, and a
spec that names none is of the ABFE (every file the ABFE writes is as it was before the engine).
"""
from __future__ import annotations

import importlib
from typing import Protocol as _Interface

ABFE = "abfe"
HYBRID = "hybrid"
#: the module that defines each protocol (as ``PROTOCOL``)
_MODULES = {ABFE: "cufep.sampling.states", HYBRID: "cufep.rbfe.states"}
#: the stages of each protocol, in the order of a leg; a stage belongs to one protocol
STAGES = {ABFE: ("restrain", "discharge", "vanish"), HYBRID: ("morph",)}
#: names of stages that earlier versions used (accepted when files are read)
STAGE_ALIASES = {"restraints": "restrain", "electrostatics": "discharge", "sterics": "vanish"}


class LegProtocol(_Interface):
    """What a protocol defines."""

    name: str
    #: its stages, in the order of a leg
    stages: tuple[str, ...]
    #: per stage, a global parameter that a System has if it can pass through the stage (adaptive lambda
    #: scheduling leaves out a stage whose parameter the Context does not have)
    stage_parameter: dict[str, str]

    def leg_stages(self, leg: str) -> tuple[str, ...]: ...

    def stage_state(self, stage: str, lam: float, stages=None, rest2_scale: float = 1.0): ...

    def state_from_dict(self, d: dict): ...

    def evaluator(self, context, states, temperature_K: float, pressure_bar: float | None): ...

    def validate(self, states) -> None: ...


def protocol(name: str | None = None) -> LegProtocol:
    """The protocol of this name; None is the ABFE (a file that names no protocol)."""
    name = ABFE if name is None else str(name)
    if name not in _MODULES:
        raise ValueError(f"unknown protocol '{name}'")
    return importlib.import_module(_MODULES[name]).PROTOCOL


def canonical_stage(name: str) -> str:
    """The name of a stage, also from the name an earlier version gave it."""
    name = STAGE_ALIASES.get(name, name)
    if not any(name in own for own in STAGES.values()):
        raise ValueError(f"unknown stage '{name}'")
    return name


def protocol_of_stages(stages) -> str:
    """The protocol whose stages these are."""
    names = {next(p for p, own in STAGES.items() if canonical_stage(s) in own) for s in stages}
    if len(names) != 1:
        raise ValueError(f"the stages {', '.join(map(str, stages))} are not those of one protocol")
    return names.pop()


def protocol_of_states(states) -> str:
    """The protocol of these states (all of one)."""
    names = {getattr(s, "protocol", None) for s in states}
    if len(names) != 1 or None in names:
        raise ValueError("the states are not those of one protocol")
    return names.pop()


def stage_state(stage: str, lam: float, stages=None, rest2_scale: float = 1.0):
    """The state at lambda ``lam`` of ``stage`` (0 its start, 1 its end), of the protocol of the stage; with
    REST2 (``rest2_scale`` above 1, the scale in the middle of the leg) at its place along the leg's ``stages``."""
    return protocol(protocol_of_stages([stage])).stage_state(stage, lam, stages, rest2_scale)


def state_from_dict(d: dict, name: str | None = None):
    """The state of protocol ``name`` (None: the ABFE) that a dictionary of ``state.to_dict()`` describes."""
    return protocol(name).state_from_dict(d)


def make_state_evaluator(context, states, temperature_K: float, pressure_bar: float | None):
    """The evaluator of the energies of all ``states`` of one protocol."""
    return protocol(protocol_of_states(states)).evaluator(context, states, temperature_K, pressure_bar)


def validate_states(states) -> None:
    """ValueError unless ``states`` are a valid sequence of the states of one protocol."""
    if len(states) < 2:
        raise ValueError("at least two thermodynamic states are needed")
    protocol(protocol_of_states(states)).validate(states)


def stage_parameter(stage: str) -> str:
    """The global parameter a System has if it can pass through ``stage``."""
    stage = canonical_stage(stage)
    return protocol(protocol_of_stages([stage])).stage_parameter[stage]


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
