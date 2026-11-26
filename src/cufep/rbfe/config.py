"""Configuration of a relative binding free energy run (``cufep rbfe``): its own YAML.

Only ``system.protein`` and ``system.ligands`` are required. The sections that mean what they mean in the
ABFE (``membrane``, ``forcefield``, ``backend``, ``thermodynamics``, ``dynamics``, ``solvation``,
``resources``) are those the two configurations share (:class:`cufep.config.schema.CommonConfig`);
``sampling`` and ``analysis`` too, with the defaults of a relative calculation. The rest is of the relative
calculation alone: the ligands and their network, the mapping, the hybrid topology and the lambda windows of its
one coordinate.
``docs/rbfe.md`` gives every setting with the basis of its default; a unit test keeps its YAML block equal to
the defaults here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from cufep.config.schema import (
    AnalysisConfig,
    CommonConfig,
    OutputConfig,
    PrecalculationLengths,
    SamplingConfig,
    _of_leg,
    _Section,
    common_problems,
)


class RbfeSystem(_Section):
    protein: Path
    #: every ligand of the network, each with its pose in the pocket, all in one frame (an SDF)
    ligands: Path
    #: the ligands of the run, by name (default: every record of the SDF)
    ligand_names: list[str] | None = None
    #: experimental binding free energies (kcal/mol): a CSV with the columns ``ligand`` and ``dg``; without it
    #: the tags ``r_exp_dg`` or ``exp_dg`` of the SDF records
    experiment: Path | None = None


class NetworkConfig(_Section):
    #: the edges [ligand A, ligand B]; not given, the LOMAP network of the ligands (lomap2's
    #: generate_lomap_network with the default LOMAP score and Kartograf, as the OpenFE benchmark planned)
    edges: list[tuple[str, str]] | None = None
    #: LOMAP (lomap2's distance_cutoff): pairs of a score below 1 - distance_cutoff are left out, except those that
    #: join the parts the others leave (docs/rbfe.md, "The network")
    distance_cutoff: float = Field(default=0.4, ge=0.0, le=1.0)
    #: LOMAP: the longest path between two ligands, in edges
    max_path_length: int = Field(default=6, ge=1)
    #: LOMAP: every ligand in a cycle, so that every ddG has a cycle closure
    require_cycle_covering: bool = True

    @field_validator("edges")
    @classmethod
    def _edges_join_two_ligands(cls, v):
        if v is not None:
            for a, b in v:
                if a == b:
                    raise ValueError(f"an edge joins {a} to itself")
            seen = set()
            for a, b in v:
                key = frozenset((a, b))
                if key in seen:
                    raise ValueError(f"the edge {a} - {b} is given twice")
                seen.add(key)
        return v


class MappingConfig(_Section):
    """Kartograf's settings (its defaults, OpenFE's)."""

    atom_max_distance_A: float = Field(default=0.95, gt=0.0)
    map_hydrogens_on_hydrogens_only: bool = True
    map_exact_ring_matches_only: bool = True
    allow_partial_fused_rings: bool = True
    allow_bond_breaks: bool = False


class HybridAlchemyConfig(_Section):
    #: alpha of the soft-core Lennard-Jones of Gapsys et al. 2012 (OpenFE's softcore_alpha)
    softcore_alpha: float = Field(default=0.85, gt=0.0, le=1.0)


class HybridAdaptiveConfig(_Section):
    """Adaptive lambda scheduling as Midgley et al. (J. Chem. Inf. Model. 2025, 65, 512) set it for relative
    transformations: a precalculation of the ligand in solution decides the windows of both legs."""

    overlap_threshold: float = Field(default=0.04, gt=0.0, lt=0.5)
    overlap_matrix: Literal["stage", "pair"] = "stage"
    candidates: int = Field(default=101, ge=3)
    #: the windows of the first iteration: a number of evenly spaced windows or the lambda values
    initial_windows: int | list[float] = 9
    #: MD per new window of the precalculation, recorded (one value, or one per leg): the paper's 50 ps in the
    #: solvent leg; in the complex leg those of the ABFE (500 ps after 500 ps of equilibration), because the pocket
    #: takes hundreds of picoseconds to adapt to a changed ligand (docs/protocol.md, "What was measured on TYK2")
    window_ps: float | PrecalculationLengths = Field(
        default_factory=lambda: PrecalculationLengths(complex=500.0, solvent=50.0))
    equilibration_ps: float | PrecalculationLengths = Field(
        default_factory=lambda: PrecalculationLengths(complex=500.0, solvent=0.0))
    sample_interval_ps: float = Field(default=0.5, gt=0.0)
    start_configurations: Literal["walk", "sequential"] = "walk"
    walk_ps: float = Field(default=5.0, gt=0.0)
    max_windows: int = Field(default=100, ge=2)
    #: each: each leg has a precalculation of its own (as the ABFE; the owner's choice of 2026-11-25, after the
    #: windows of the solvent leg left a pair of windows of a complex leg of TYK2 at an overlap of 0.013); solvent:
    #: the precalculation of the solvent leg decides the windows of both legs (the paper's)
    legs: Literal["solvent", "each"] = "each"

    def window(self, leg: str) -> float:
        return float(_of_leg(self.window_ps, leg))

    def equilibration(self, leg: str) -> float:
        return float(_of_leg(self.equilibration_ps, leg))


class HybridLambdaSchedule(_Section):
    #: adaptive: adaptive lambda scheduling; fixed: the windows below
    mode: Literal["adaptive", "fixed"] = "adaptive"
    #: fixed mode: a number of evenly spaced windows (OpenFE's 11) or the lambda values
    windows: int | list[float] = 11
    adaptive: HybridAdaptiveConfig = Field(default_factory=HybridAdaptiveConfig)

    def precalculated_legs(self) -> list[str]:
        """The legs that run a precalculation of adaptive lambda scheduling."""
        if self.mode != "adaptive":
            return []
        return ["complex", "solvent"] if self.adaptive.legs == "each" else ["solvent"]

    @field_validator("windows")
    @classmethod
    def _windows(cls, v):
        from cufep.config.schema import _check_stage

        return _check_stage(v)


def initial_window_problems(schedule: HybridLambdaSchedule) -> list[str]:
    """Why adaptive lambda scheduling could not start from ``initial_windows`` (review of PR #17: nine windows to
    start with and ``max_windows: 2`` were accepted, and the precalculation ran nine): the windows must be
    candidates, and no more than ``max_windows``. The rule of the ABFE
    (:func:`cufep.config.schema.initial_window_problems`)."""
    from cufep.alchemy.candidates import initial_window_indices

    if schedule.mode != "adaptive":
        return []
    a = schedule.adaptive
    try:
        n = len(initial_window_indices(a.initial_windows, a.candidates))
    except ValueError as exc:
        return [f"lambda_schedule.adaptive.initial_windows: {exc} (lambda_schedule.adaptive.candidates = "
                f"{a.candidates})"]
    if n > a.max_windows:
        return [f"lambda_schedule.adaptive.initial_windows gives {n} lambda windows to start with, more than "
                f"lambda_schedule.adaptive.max_windows = {a.max_windows}"]
    return []


class RbfeSampling(SamplingConfig):
    #: per lambda window, used for the estimate: 5 ns, the default of OpenFE's relative protocol
    production_ns: float = 5.0

    @model_validator(mode="after")
    def _no_water_moves(self):
        if self.relocation_mc.enabled or self.gcmc.enabled:
            raise ValueError("sampling.relocation_mc and sampling.gcmc are moves of the ABFE; a relative "
                             "calculation does not have them")
        if not isinstance(self.equilibration_ns, (int, float)):
            raise ValueError("sampling.equilibration_ns is one value for both legs")
        return self


class RbfeAnalysis(AnalysisConfig):
    #: the Lennard-Jones of the ligand with its surroundings beyond the cutoff is not corrected (OpenFE's
    #: use_dispersion_correction: False for the hybrid; the NonbondedForce of the environment keeps its own)
    lrc: Literal["none"] = "none"


class RbfeOutput(OutputConfig):
    dir: Path = Path("runs/cufep_rbfe")


class RbfeConfig(CommonConfig):
    """The configuration of a relative calculation: the sections it shares with the ABFE
    (:class:`cufep.config.schema.CommonConfig`, ``sampling``, ``analysis`` and ``output`` with the defaults of a
    relative calculation) and its own."""

    system: RbfeSystem
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    alchemy: HybridAlchemyConfig = Field(default_factory=HybridAlchemyConfig)
    lambda_schedule: HybridLambdaSchedule = Field(default_factory=HybridLambdaSchedule)
    sampling: RbfeSampling = Field(default_factory=RbfeSampling)
    analysis: RbfeAnalysis = Field(default_factory=RbfeAnalysis)
    output: RbfeOutput = Field(default_factory=RbfeOutput)

    @model_validator(mode="after")
    def _agree(self):
        problems = common_problems(self)
        if self.membrane.enabled and self.membrane.complex_leg != "bilayer":
            problems.append("membrane.complex_leg: water is a setting of the ABFE; a relative calculation runs the "
                            "complex leg in the bilayer")
        problems += initial_window_problems(self.lambda_schedule)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def holds_solute(self, leg: str) -> bool:
        """The complex leg: the pose that was given, held through NVT and NPT (and through the equilibration of a
        new bilayer), without free MD, as OpenFE starts its relative legs from the minimised input pose; the
        equilibration of the replica exchange follows."""
        return leg == "complex"


def load_rbfe_config(path: str | Path) -> RbfeConfig:
    """Read a YAML; relative paths of the system are relative to the file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    system = data.get("system", {})
    for key in ("protein", "ligands", "experiment"):
        if system.get(key) is not None and not Path(system[key]).is_absolute():
            system[key] = str((path.parent / system[key]).resolve())
    return RbfeConfig.model_validate(data)
