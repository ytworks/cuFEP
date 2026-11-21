"""Configuration schema (YAML -> validated settings).

Principles (docs/configuration.md lists every key with the basis of its default):

* every setting can be changed in the YAML;
* every setting has a default, and the default is the recommended value;
* ``auto`` is used only where the value has to be computed from the input (restraint
  atoms, GPUs, box shape, membrane orientation); everything else is a plain value.

Only ``system.protein`` and ``system.ligand`` are required. Keys set to ``auto`` are
resolved by :mod:`cufep.config.auto`; the resolved configuration is written next to the
results.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Auto = Literal["auto"]

#: Recommended lambda values of the vanish stage (0 = Lennard-Jones on, 1 = off). These and the
#: evenly spaced restrain and discharge stages are the defaults of OpenFE's ABFE protocol, which
#: uses the same soft-core form and replica exchange: 30 windows for the complex leg and 14 for
#: the solvent leg (windows at stage boundaries are shared).
RECOMMENDED_VANISH_COMPLEX = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
RECOMMENDED_VANISH_SOLVENT = [0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0]

#: A stage is given either as a number of lambda windows or as the lambda values themselves.
StageWindows = int | list[float]


def _check_stage(value):
    if isinstance(value, bool):
        raise ValueError("give a number of lambda windows or a list of lambda values")
    if isinstance(value, int):
        if value < 2:
            raise ValueError("a stage needs at least two lambda windows (its start and its end)")
        return value
    values = [float(x) for x in value]
    if len(values) < 2 or abs(values[0]) > 1e-9 or abs(values[-1] - 1.0) > 1e-9 \
            or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("the lambda values of a stage must increase from 0.0 (start of the stage) to 1.0 (its end)")
    return values


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SystemConfig(_Section):
    protein: Path
    ligand: Path
    #: optional cofactors as SDF files (kept in both end states of the complex leg)
    cofactors: list[Path] = Field(default_factory=list)
    #: restrict a multi-record SDF to these ligand names (default: all records)
    ligand_names: list[str] | None = None


class MembraneConfig(_Section):
    #: build a lipid bilayer around the protein? The user decides; cuFEP does not guess.
    enabled: bool = False
    #: how the membrane plane is found when enabled: auto = at least two of the built-in
    #: methods must agree; input = the PDB is already oriented (membrane normal along z,
    #: membrane centre at z = 0); or one method by name
    orientation: Literal["auto", "input", "transfer", "slab", "memembed"] = "auto"
    lipid: str = "POPC"
    padding_nm: float = 1.5
    #: unrestrained MD of the bilayer before the alchemical run (replaces pre_equilibration md_ns)
    equilibration_ns: float = 10.0


class ForceFieldConfig(_Section):
    protein: str = "amber14/protein.ff14SB.xml"
    water: str = "amber14/tip3p.xml"
    lipid: str = "amber19/lipid21.xml"
    ligand: str = "openff-2.2.1"
    ligand_charges: Literal["am1bcc", "am1bcc-nagl"] = "am1bcc"


class BackendConfig(_Section):
    name: Literal["cupotential", "openmm"] = "cupotential"
    #: backend used when the default one is unavailable or fails for a leg
    fallback: Literal["openmm", "none"] = "openmm"
    precision: Literal["mixed", "single", "double"] = "mixed"
    platform: Literal["CUDA", "CPU", "Reference", "OpenCL"] = "CUDA"


class ThermodynamicsConfig(_Section):
    temperature_K: float = 298.15
    pressure_bar: float = 1.0
    ionic_strength_M: float = 0.15
    ph: float = 7.4


class DynamicsConfig(_Section):
    #: used together with hydrogen mass repartitioning; if the plain MD before the alchemical
    #: run is unstable it is repeated at 2 fs and the fact is recorded
    timestep_fs: float = 4.0
    constraints: Literal["hbonds", "none"] = "hbonds"
    hmr_mass_amu: float = 3.0
    friction_per_ps: float = 1.0
    cutoff_nm: float = 0.9
    #: MD steps between the moves of the Monte Carlo barostat (OpenMM's and OpenFE's default). The
    #: replicas that share a GPU take turns in one OpenMM Context whose barostat counts the steps
    #: of the Context, so every stretch of MD that one configuration gets in one go (an iteration
    #: of a replica, a window of the precalculation) must have at least this many steps
    barostat_frequency: int = Field(default=25, ge=1)


class SolvationConfig(_Section):
    box_shape: Literal["auto", "cube", "dodecahedron"] = "auto"
    padding_complex_nm: float = 1.0
    padding_solvent_nm: float = 1.5


class InteractionsConfig(_Section):
    """``restraints.selection: interactions``: the MD from which the anchors are chosen and the
    thresholds of the rule, with the numbers of Ding et al. (arXiv:2603.22274 v3, sec. 2.1 and
    2.4; ``docs/configuration.md`` quotes them)."""

    #: unrestrained NPT trajectories of the complex, each from the end of the pre-equilibration
    trajectories: int = Field(default=4, ge=1)
    length_ns: float = Field(default=3.0, gt=0)
    #: the first part of every trajectory that is not analysed
    discard_ns: float = Field(default=1.0, ge=0)
    frame_interval_ps: float = Field(default=10.0, gt=0)
    #: an interaction is kept if it is there in this fraction of the analysed frames; the angles
    #: of a set of anchors must be in range in as many
    min_frame_fraction: float = Field(default=0.40, gt=0, le=1)
    #: the range of the angles L1-L2-L3 and P1-L1-L2 (and of the frames that give reference values)
    angle_range_deg: tuple[float, float] = (45.0, 135.0)
    #: a residue counts as of stable secondary structure if it is in more than this fraction of the frames
    stable_structure_fraction: float = Field(default=0.5, ge=0, le=1)
    #: a residue with two interactions: the second in more than this fraction of the frames of the first
    second_interaction_ratio: float = Field(default=0.6, ge=0, le=1)
    #: without anchors from interactions: the backbone nearest to the ligand, within this distance
    #: in this fraction of the frames
    fallback_frame_fraction: float = Field(default=0.15, gt=0, le=1)
    fallback_cutoff_nm: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _something_is_analysed(self):
        if not self.discard_ns < self.length_ns:
            raise ValueError(f"restraints.interactions.discard_ns = {self.discard_ns:g} ns leaves nothing of a "
                             f"trajectory of length_ns = {self.length_ns:g} ns to analyse")
        low, high = self.angle_range_deg
        if not 0.0 <= low < high <= 180.0:
            raise ValueError(f"restraints.interactions.angle_range_deg must be two angles between 0 and 180 degrees, "
                             f"the lower first, not {list(self.angle_range_deg)}")
        return self


class RestraintConfig(_Section):
    type: Literal["boresch"] = "boresch"
    #: interactions: anchors from the protein-ligand interactions of unrestrained MD of the complex
    #: (the section ``interactions``); auto: anchors with the smallest fluctuations in the
    #: pre-equilibration (the search of OpenFE); manual: host_atoms and guest_atoms
    selection: Literal["interactions", "auto", "manual"] = "interactions"
    #: openfe or custom: the two values below; auto: kT / sigma^2 of each restrained coordinate over
    #: the frames the restraint is made from (the pre-equilibration; with selection interactions the
    #: analysed frames of the MD for the restraint), kept within 0.25-4 times the two values' defaults
    force_constants: Literal["openfe", "auto", "custom"] = "openfe"
    k_r_kcal_per_A2: float = 10.0
    k_angle_kcal_per_rad2: float = 80.0
    #: manual selection: atom indices in the prepared complex (H0,H1,H2 / G0,G1,G2)
    host_atoms: list[int] | None = None
    guest_atoms: list[int] | None = None
    #: the four settings of the search of selection auto
    rmsf_cutoff_nm: float = 0.1
    host_min_distance_nm: float = 0.5
    #: also, for every selection: the backbone atoms within this distance + 0.5 nm of the ligand are
    #: the binding site in which ``ligand_rmsd_from_input_A`` is measured
    host_max_distance_nm: float = 1.5
    dssp_filter: bool = True
    #: selection interactions only
    interactions: InteractionsConfig = Field(default_factory=InteractionsConfig)

    @model_validator(mode="after")
    def _manual_anchors(self):
        if self.selection == "manual":
            for key in ("host_atoms", "guest_atoms"):
                atoms = getattr(self, key)
                if not atoms or len(atoms) != 3 or len(set(atoms)) != 3 or min(atoms) < 0:
                    raise ValueError(f"restraints.selection is 'manual': restraints.{key} must be three distinct atom "
                                     f"indices of the prepared complex (from 0), not {atoms}")
            if set(self.host_atoms) & set(self.guest_atoms):
                raise ValueError(f"restraints.host_atoms {self.host_atoms} and guest_atoms {self.guest_atoms} share "
                                 "atoms; the host atoms belong to the protein and the guest atoms to the ligand")
        return self


class AlchemyConfig(_Section):
    softcore_alpha: float = 0.5


class ComplexWindows(_Section):
    restrain: StageWindows = 6
    discharge: StageWindows = 11
    vanish: StageWindows = Field(default_factory=lambda: list(RECOMMENDED_VANISH_COMPLEX))

    _check = field_validator("restrain", "discharge", "vanish")(_check_stage)


class SolventWindows(_Section):
    discharge: StageWindows = 5
    vanish: StageWindows = Field(default_factory=lambda: list(RECOMMENDED_VANISH_SOLVENT))

    _check = field_validator("discharge", "vanish")(_check_stage)


class WindowsConfig(_Section):
    """Lambda windows per leg and stage: a number of windows or the lambda values.

    Lambda runs from 0 at the start of a stage to 1 at its end (restrain: restraint off to
    on; discharge: charges on to off; vanish: Lennard-Jones on to off), as OpenFE's
    ``lambda_restraints``, ``lambda_elec`` and ``lambda_vdw``.
    """

    complex: ComplexWindows = Field(default_factory=ComplexWindows)
    solvent: SolventWindows = Field(default_factory=SolventWindows)

    def for_leg(self, leg: str) -> dict[str, StageWindows]:
        section = self.complex if leg == "complex" else self.solvent
        return {name: getattr(section, name) for name in type(section).model_fields}


class ComplexInitialWindows(_Section):
    """The start of the complex leg: the recommended windows of mode fixed (``windows.complex``),
    30 in all. The owner's choice of 2026-11-22 from three runs of each of three starts on the
    TYK2 example (3, 5 and 9 evenly spaced windows until then; docs/configuration.md)."""

    restrain: StageWindows = 6
    discharge: StageWindows = 11
    vanish: StageWindows = Field(default_factory=lambda: list(RECOMMENDED_VANISH_COMPLEX))

    _check = field_validator("restrain", "discharge", "vanish")(_check_stage)


class SolventInitialWindows(_Section):
    """The start of the solvent leg: 9 evenly spaced windows are the paper's number for one lambda
    coordinate; the discharge stage starts with fewer (cuFEP's own)."""

    discharge: StageWindows = 5
    vanish: StageWindows = 9

    _check = field_validator("discharge", "vanish")(_check_stage)


class InitialWindows(_Section):
    """Windows per leg and stage with which adaptive lambda scheduling starts: a number of evenly
    spaced windows (the start of the paper), or the lambda values, which need not be evenly
    spaced and must be candidates (cuFEP's own)."""

    complex: ComplexInitialWindows = Field(default_factory=ComplexInitialWindows)
    solvent: SolventInitialWindows = Field(default_factory=SolventInitialWindows)

    def for_leg(self, leg: str) -> dict[str, StageWindows]:
        section = self.complex if leg == "complex" else self.solvent
        return {name: getattr(section, name) for name in type(section).model_fields}


class WindowLimits(_Section):
    """Largest number of lambda windows of a leg in adaptive lambda scheduling.

    100 in both legs is the owner's choice of 2026-11-21 (48 and 32 until then; the paper gives
    no limit).
    """

    complex: int = 100
    solvent: int = 100


class PrecalculationLengths(_Section):
    """A length of the precalculation of adaptive lambda scheduling for each leg, in ps."""

    complex: float = Field(ge=0.0)
    solvent: float = Field(ge=0.0)


class LegThresholds(_Section):
    """An overlap threshold of adaptive lambda scheduling for each leg."""

    complex: float = Field(gt=0.0, lt=0.5)
    solvent: float = Field(gt=0.0, lt=0.5)


class LegOverlapMatrices(_Section):
    """The overlap matrix that adaptive lambda scheduling reads, for each leg."""

    complex: Literal["stage", "pair"]
    solvent: Literal["stage", "pair"]


class LegCandidates(_Section):
    """A number of candidate lambda values per stage for each leg."""

    complex: int = Field(ge=3)
    solvent: int = Field(ge=3)


class LegModes(_Section):
    """How the lambda windows are chosen, for each leg."""

    complex: Literal["adaptive", "fixed"]
    solvent: Literal["adaptive", "fixed"]


def _of_leg(value, leg: str):
    """The value of a setting that is given once for both legs or once per leg."""
    return getattr(value, leg) if isinstance(value, _Section) else value


#: The threshold of a leg whose ``overlap_threshold`` is not given, by the overlap matrix that the
#: leg reads. A threshold means another spacing of the windows under each matrix, so the default
#: belongs to the matrix. stage: 0.18, the owner's choice of 2026-11-20 from the examples at 5 ns
#: (the paper: 0.04; at 0.20 the precalculation of the TYK2 solvent leg did not converge before
#: max_windows, 32 then). pair: 0.30, the owner's choice of 2026-11-22 for the complex leg with
#: its start. In the matrix of a stage nearly every pair of that start reads at or below 0.30 (26
#: or 27 of 29 in the three runs of the TYK2 example, 6 by the pair alone), so with
#: ``overlap_matrix: stage`` alone the rule kept adding windows (harmonic windows: to max_windows)
DEFAULT_OVERLAP_THRESHOLD = {"stage": 0.18, "pair": 0.30}


class AdaptiveConfig(_Section):
    """Adaptive lambda scheduling (Midgley et al., J. Chem. Inf. Model. 2025, 65, 512).

    ``overlap_threshold``, ``overlap_matrix`` and ``candidates`` are one value for both legs or
    one per leg (``{complex: ..., solvent: ...}``), as the lengths of the precalculation are;
    ``initial_windows`` and ``max_windows`` are always given per leg.
    """

    #: a pair of neighbouring windows is deficient if its overlap is at or below this value. Not
    #: given, it is for each leg the threshold of the matrix that the leg reads
    #: (DEFAULT_OVERLAP_THRESHOLD): with the default matrices 0.30 in the complex leg and 0.18 in
    #: the solvent leg
    overlap_threshold: float | LegThresholds = Field(
        default_factory=lambda: LegThresholds(complex=DEFAULT_OVERLAP_THRESHOLD["pair"],
                                              solvent=DEFAULT_OVERLAP_THRESHOLD["stage"]))
    #: the overlap matrix whose element is that overlap. stage: the matrix of all windows of the
    #: stage (the paper; the solvent leg). pair: the matrix of the two neighbouring windows alone
    #: (cuFEP's own, the owner's decision of 2026-11-21; the complex leg since 2026-11-22). An
    #: element of the stage matrix is shared among all windows that overlap, so it falls when
    #: windows get closer and windows that are close enough read as deficient; the overlap of a
    #: pair alone rises to 0.5 as its windows meet
    overlap_matrix: Literal["stage", "pair"] | LegOverlapMatrices = Field(
        default_factory=lambda: LegOverlapMatrices(complex="pair", solvent="stage"))
    #: number of equally spaced candidate lambda values per stage
    candidates: int | LegCandidates = 101
    #: windows per leg and stage in the first iteration (the schedule never gets smaller than this):
    #: a number of evenly spaced windows, or the lambda values
    initial_windows: InitialWindows = Field(default_factory=InitialWindows)
    #: MD per new window in the precalculation during which energies are recorded: one number
    #: for both legs, or one per leg. The solvent leg has the paper's 50 ps. A complex leg needs
    #: longer: the pocket adapts to a partly decoupled ligand over hundreds of picoseconds, and
    #: 50 ps read the overlap there several times too high
    window_ps: float | PrecalculationLengths = Field(
        default_factory=lambda: PrecalculationLengths(complex=500.0, solvent=50.0))
    #: MD in a new window before that, not recorded; one number or one per leg
    equilibration_ps: float | PrecalculationLengths = Field(
        default_factory=lambda: PrecalculationLengths(complex=500.0, solvent=0.0))
    #: interval at which energies are recorded during that MD
    sample_interval_ps: float = Field(default=0.5, gt=0.0)
    #: the windows of an iteration run at the same time on the available GPUs; before the first
    #: iteration a walk through the windows with this much MD each gives them start configurations
    walk_ps: float = Field(default=5.0, gt=0.0)
    #: largest number of windows per leg; reaching it ends the precalculation with a warning
    max_windows: WindowLimits = Field(default_factory=WindowLimits)

    @field_validator("overlap_threshold")
    @classmethod
    def _threshold_is_an_overlap(cls, v):
        # the overlap of two windows lies between 0 and 0.5 (the limits of a leg are in LegThresholds)
        if not isinstance(v, LegThresholds) and not 0.0 < v < 0.5:
            raise ValueError("overlap_threshold must be above 0 and below 0.5")
        return v

    @field_validator("candidates")
    @classmethod
    def _a_stage_has_candidates_between_its_ends(cls, v):
        if not isinstance(v, LegCandidates) and v < 3:
            raise ValueError("candidates must be at least 3")
        return v

    @model_validator(mode="after")
    def _a_threshold_that_is_not_given_is_that_of_the_matrix(self):
        if "overlap_threshold" not in self.model_fields_set:
            # written past __setattr__: the setting stays one that was not given
            self.__dict__["overlap_threshold"] = LegThresholds(**{
                leg: DEFAULT_OVERLAP_THRESHOLD[self.matrix(leg)] for leg in ("complex", "solvent")})
        return self

    def threshold(self, leg: str) -> float:
        """The overlap at or below which a pair of neighbouring windows of this leg is deficient."""
        return float(_of_leg(self.overlap_threshold, leg))

    def matrix(self, leg: str) -> str:
        """The overlap matrix read in this leg: ``stage`` or ``pair``."""
        return str(_of_leg(self.overlap_matrix, leg))

    def candidates_of(self, leg: str) -> int:
        """The number of candidate lambda values per stage of this leg."""
        return int(_of_leg(self.candidates, leg))

    @field_validator("window_ps")
    @classmethod
    def _recorded_md_is_not_empty(cls, v):
        values = (v.complex, v.solvent) if isinstance(v, PrecalculationLengths) else (v,)
        if min(values) <= 0.0:
            raise ValueError("window_ps must be above 0 in both legs")
        return v

    @field_validator("equilibration_ps")
    @classmethod
    def _equilibration_is_not_negative(cls, v):
        if not isinstance(v, PrecalculationLengths) and v < 0.0:
            raise ValueError("equilibration_ps must not be negative")
        return v

    def window(self, leg: str) -> float:
        """Recorded MD per new window of the precalculation in this leg, ps."""
        v = self.window_ps
        return float(getattr(v, leg)) if isinstance(v, PrecalculationLengths) else float(v)

    def equilibration(self, leg: str) -> float:
        """MD before that in this leg, ps."""
        v = self.equilibration_ps
        return float(getattr(v, leg)) if isinstance(v, PrecalculationLengths) else float(v)


class LambdaScheduleConfig(_Section):
    #: adaptive: the precalculation decides the number and the values of the lambda windows;
    #: fixed: the windows below are used as they are. One value for both legs, or one per leg
    #: (``{complex: adaptive, solvent: fixed}``)
    mode: Literal["adaptive", "fixed"] | LegModes = "adaptive"
    windows: WindowsConfig = Field(default_factory=WindowsConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    #: fixed mode: MD in each window, one window after the other, that gives the replicas their
    #: start configurations
    fixed_walk_ps: float = Field(default=50.0, gt=0.0)
    #: a schedule.yaml written by an earlier run (or a directory of them, as written by
    #: `cufep benchmark freeze`); if given it is used as it is, whatever the mode
    file: Path | None = None

    def mode_of(self, leg: str) -> str:
        """How the windows of this leg are chosen: ``adaptive`` or ``fixed``."""
        return str(_of_leg(self.mode, leg))

    def precalculated_legs(self) -> list[str]:
        """The legs whose windows a precalculation of adaptive lambda scheduling decides."""
        if self.file is not None:
            return []
        return [leg for leg in ("complex", "solvent") if self.mode_of(leg) == "adaptive"]


class PreEquilibrationLeg(_Section):
    nvt_ns: float = Field(default=0.25, ge=0.0)
    npt_ns: float = Field(default=0.5, ge=0.0)
    md_ns: float = Field(default=5.0, gt=0.0)


class SolventPreEquilibration(PreEquilibrationLeg):
    nvt_ns: float = Field(default=0.1, ge=0.0)
    npt_ns: float = Field(default=0.2, ge=0.0)
    md_ns: float = Field(default=0.5, gt=0.0)


class PreEquilibrationConfig(_Section):
    """Plain MD of the fully interacting system before the alchemical run (OpenFE's defaults)."""

    complex: PreEquilibrationLeg = Field(default_factory=PreEquilibrationLeg)
    solvent: SolventPreEquilibration = Field(default_factory=SolventPreEquilibration)


class LegLengths(_Section):
    complex: float = Field(gt=0.0)
    solvent: float = Field(gt=0.0)


class Rest2Config(_Section):
    """REST2, replica exchange with solute scaling (Wang, Friesner, Berne 2011), as SOMD2 uses it.

    The energy inside a hot region is scaled so that the region behaves as if it were at a
    higher temperature in the lambda windows in the middle of a leg; the first and the last
    window are unscaled. The ligand is always in the hot region.
    """

    #: temperature of the hot region relative to that of the run in the middle of a leg (the
    #: meaning of SOMD2's rest2_scale); 1 switches REST2 off. Kept low enough that amide bonds
    #: do not isomerise (docs/configuration.md)
    scale: float = Field(default=1.5, ge=1.0)
    #: complex leg: ligand = the ligand only; pocket = also the protein residues that have a
    #: heavy atom within pocket_cutoff_nm of a heavy atom of the ligand in the prepared structure.
    #: pocket is the owner's choice of 2026-11-20 (the default of SOMD2 is the ligand)
    region: Literal["ligand", "pocket"] = "pocket"
    pocket_cutoff_nm: float = Field(default=0.4, gt=0.0)
    #: protein residues added to the hot region, named as in the input PDB: "960", with the chain
    #: of the input "B:960", with an insertion code "B:898A"
    residues: list[str] = Field(default_factory=list)


class SamplingConfig(_Section):
    pre_equilibration: PreEquilibrationConfig = Field(default_factory=PreEquilibrationConfig)
    rest2: Rest2Config = Field(default_factory=Rest2Config)
    #: hrex = Hamiltonian replica exchange between the lambda windows; none = independent windows
    exchange: Literal["hrex", "none"] = "hrex"
    time_per_iteration_ps: float = Field(default=2.5, gt=0.0)
    #: per lambda window, discarded; one number or {complex: ..., solvent: ...}
    equilibration_ns: float | LegLengths = 1.0
    #: per lambda window, used for the estimate; one number or {complex: ..., solvent: ...}.
    #: 10 ns is the default of OpenFE's ABFE protocol: with 5 ns the lambda
    #: window with the fewest independent samples of a TYK2 complex leg held about 29, where
    #: 50 are recommended for MBAR (docs/protocol.md, "Checks of the analysis")
    production_ns: float | LegLengths = 10.0
    #: number of completely independent repeats of the whole calculation (as OpenFE's protocol_repeats)
    repeats: int = Field(default=1, ge=1)
    seed: int = 20261118
    checkpoint_interval_ps: float = 250.0

    @field_validator("equilibration_ns", "production_ns")
    @classmethod
    def _positive(cls, v):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0:
            raise ValueError("the length must be positive")
        return v

    def equilibration(self, leg: str) -> float:
        v = self.equilibration_ns
        return float(getattr(v, leg)) if isinstance(v, LegLengths) else float(v)

    def production(self, leg: str) -> float:
        v = self.production_ns
        return float(getattr(v, leg)) if isinstance(v, LegLengths) else float(v)


class AnalysisConfig(_Section):
    estimator: Literal["mbar"] = "mbar"
    #: long-range dispersion correction for the ligand-environment LJ at the coupled end state
    #: reweight = explicit tail by reweighting (default); isotropic = homogeneous tail, as in
    #: OpenMM's long-range correction and OpenFE; none = truncated interaction
    lrc: Literal["reweight", "isotropic", "none"] = "reweight"
    lrc_cutoff_nm: float = 2.0
    #: a result is NOT RELIABLE if the overlap between two neighbouring lambda windows in the
    #: production is below this
    min_overlap: float = 0.03
    #: number of points of the forward and reverse series (free energy from growing parts of the
    #: production, counted from its start and from its end)
    convergence_points: int = Field(default=10, ge=2)
    #: number of consecutive blocks of the production that get a free energy estimate of their
    #: own, to show a free energy that keeps moving in one direction
    convergence_blocks: int = Field(default=5, ge=2)
    #: the result carries a warning if the lambda window with the fewest uncorrelated samples has
    #: fewer than this (recommended for MBAR by Mey et al. 2020). With min_overlap the only levels
    #: of a verdict on the analysis: cuFEP has no criteria of its own (owner, 2026-11-20)
    min_effective_samples: float = Field(default=50.0, gt=0.0)
    #: numbers of time blocks, common to all lambda windows, with which the standard error of a
    #: leg is computed (delete-one-block jackknife); the largest of the errors they give is the
    #: one reported, unless the analytical one is larger
    error_blocks: list[Annotated[int, Field(ge=2)]] = Field(default=[40, 20, 10, 5], min_length=1)
    #: number of time blocks for the standard error of every part of the production: each half,
    #: each point of the forward and reverse series, each consecutive block
    part_error_blocks: int = Field(default=5, ge=2)


class ResourcesConfig(_Section):
    #: "auto" = every visible GPU that is not busy; or an explicit list of device indices
    gpus: Auto | list[int] = "auto"
    #: worker processes that share one GPU at the same time (needs NVIDIA MPS, which cuFEP starts
    #: itself); "auto" = from the size of the system, 1 if MPS is not available
    oversubscription_factor: Auto | int = "auto"
    cpu_threads_per_worker: int = 2

    @field_validator("oversubscription_factor")
    @classmethod
    def _factor(cls, v):
        if v != "auto" and int(v) < 1:
            raise ValueError("oversubscription_factor must be 'auto' or at least 1")
        return v


class OutputConfig(_Section):
    dir: Path = Path("runs/cufep")


class Config(_Section):
    version: Literal[1] = 1
    system: SystemConfig
    membrane: MembraneConfig = Field(default_factory=MembraneConfig)
    forcefield: ForceFieldConfig = Field(default_factory=ForceFieldConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    thermodynamics: ThermodynamicsConfig = Field(default_factory=ThermodynamicsConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    solvation: SolvationConfig = Field(default_factory=SolvationConfig)
    restraints: RestraintConfig = Field(default_factory=RestraintConfig)
    alchemy: AlchemyConfig = Field(default_factory=AlchemyConfig)
    lambda_schedule: LambdaScheduleConfig = Field(default_factory=LambdaScheduleConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _settings_agree(self):
        problems = configuration_problems(self)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def md_segments(cfg: "Config") -> list[tuple[str, int]]:
    """Every stretch of MD that one configuration gets in one go in an OpenMM Context it shares
    with other configurations, as (what it is in the words of the configuration, MD steps), at
    the configured time step. A run whose pre-equilibration falls back to a shorter time step
    makes each of them longer. Only the stretches the configured lambda schedule uses."""
    from cufep.steps import md_steps, precalculation_steps

    dt = float(cfg.dynamics.timestep_fs)
    ls = cfg.lambda_schedule
    out = [("sampling.time_per_iteration_ps (the MD of one replica in one iteration)",
            md_steps(cfg.sampling.time_per_iteration_ps, dt))]
    # the walks through the windows (fixed_walk_ps, adaptive.walk_ps) are one trajectory that goes from
    # window to window: no other configuration shares its Context, so it is not listed here
    a = ls.adaptive
    for leg in ls.precalculated_legs():
        eq_steps, n_samples, per_sample = precalculation_steps(a.window(leg), a.equilibration(leg),
                                                               a.sample_interval_ps, dt)
        out.append(
            (
                f"lambda_schedule.adaptive.window_ps + equilibration_ps of the {leg} leg (one window of "
                "the precalculation, which starts from a configuration of its own)",
                eq_steps + n_samples * per_sample,
            )
        )
    return out


class ConfigurationError(ValueError):
    """Settings that contradict each other."""


def check_configuration(cfg: "Config") -> None:
    """Raise :class:`ConfigurationError` with every problem of :func:`configuration_problems`.
    The commands call it before they write anything, also after options of the command line have
    changed settings of a configuration that was valid when it was read."""
    problems = configuration_problems(cfg)
    if problems:
        raise ConfigurationError("; ".join(problems))


def configuration_problems(cfg: "Config") -> list[str]:
    """Settings that contradict each other, found without any input file: every problem in the
    words of the configuration. :class:`Config` refuses to exist with one of them, so no command
    gets as far as writing anything."""
    from cufep.prep.water import water_problem
    from cufep.steps import barostat_segment_problem

    problems = []
    # every configuration that shares a Context must get moves of the barostat
    for what, steps in md_segments(cfg):
        problem = barostat_segment_problem(steps, cfg.dynamics.barostat_frequency, what)
        if problem:
            problems.append(problem)
    # the explicit tail of the dispersion starts where the real-space cutoff ends: an outer radius
    # inside the cutoff would count the interactions between them, which the MD has, a second time
    if cfg.analysis.lrc != "none" and not cfg.analysis.lrc_cutoff_nm > cfg.dynamics.cutoff_nm:
        problems.append(f"analysis.lrc_cutoff_nm = {cfg.analysis.lrc_cutoff_nm:g} nm must be larger than "
                        f"dynamics.cutoff_nm = {cfg.dynamics.cutoff_nm:g} nm: the tail correction (analysis.lrc = "
                        f"{cfg.analysis.lrc}) adds the dispersion beyond the cutoff, from the cutoff outwards")
    water = water_problem(cfg.forcefield.water, membrane=bool(cfg.membrane.enabled))
    if water:
        problems.append(water)
    problems += initial_window_problems(cfg)
    if cfg.restraints.selection == "interactions":
        # the thresholds of the rule are numbers of frames, rounded down: none of them may be no frame
        n = analysed_frames(cfg)
        ri = cfg.restraints.interactions
        for name, fraction in (("min_frame_fraction", ri.min_frame_fraction),
                               ("fallback_frame_fraction", ri.fallback_frame_fraction)):
            if frames_needed(n, fraction) < 1:
                problems.append(
                    f"restraints.interactions: {n} frame(s) are analysed ({ri.trajectories} trajectories of "
                    f"{ri.length_ns:g} ns, the first {ri.discard_ns:g} ns left out, a frame every "
                    f"{ri.frame_interval_ps:g} ps) and {name} = {fraction:g} of them is less than one frame: an "
                    "interaction in no frame would be kept. Run longer trajectories or keep frames more often")
    return problems


def initial_window_problems(cfg: "Config") -> list[str]:
    """Why adaptive lambda scheduling could not start from ``initial_windows``: the windows of a
    stage are candidates (a number of them must fit on the candidates, given values must be
    candidates), and a leg does not start with more windows than ``max_windows``. Asked of the
    rule the precalculation starts with, and only of the legs that run a precalculation."""
    from cufep.alchemy.candidates import initial_window_indices

    ls = cfg.lambda_schedule
    a, problems = ls.adaptive, []
    for leg in ls.precalculated_legs():
        total, stages = 0, a.initial_windows.for_leg(leg)
        # a start that the file did not give is refused for a setting it did give: say so
        default = ("" if stages != InitialWindows().for_leg(leg) else
                   f"; this is the default start of the {leg} leg: give "
                   f"lambda_schedule.adaptive.initial_windows.{leg} for these settings as well")
        for stage, spec in stages.items():
            try:
                total += len(initial_window_indices(spec, a.candidates_of(leg)))
            except ValueError as exc:
                problems.append(f"lambda_schedule.adaptive.initial_windows.{leg}.{stage}: {exc} "
                                f"(lambda_schedule.adaptive.candidates = {a.candidates_of(leg)}){default}")
                total = None
                break
        limit = getattr(a.max_windows, leg)
        if total is not None and total - (len(stages) - 1) > limit:
            problems.append(f"lambda_schedule.adaptive.initial_windows.{leg} gives the {leg} leg "
                            f"{total - (len(stages) - 1)} lambda windows to start with (windows at stage boundaries "
                            f"are shared), more than lambda_schedule.adaptive.max_windows.{leg} = {limit}{default}")
    return problems


def unpaired_threshold_notices(data: dict, cfg: "Config") -> list[str]:
    """A notice for a file that gives ``lambda_schedule.adaptive.overlap_threshold`` and not the
    overlap matrix it is compared with, where a leg then reads the overlap of a pair alone: files
    written before ``overlap_matrix`` existed meant the matrix of the stage in both legs (review of
    the defaults of 2026-11-22)."""
    block = data.get("lambda_schedule")
    adaptive = block.get("adaptive") if isinstance(block, dict) else None
    if not isinstance(adaptive, dict) or "overlap_threshold" not in adaptive or "overlap_matrix" in adaptive:
        return []
    a = cfg.lambda_schedule.adaptive
    return [f"lambda_schedule.adaptive.overlap_threshold is given without overlap_matrix: in the {leg} leg the "
            f"threshold ({a.threshold(leg):g}) is compared with the overlap of the two windows of a pair alone, the "
            "default of that leg. Earlier versions compared it with the overlap matrix of the stage; "
            "overlap_matrix: stage beside the threshold does that."
            for leg in cfg.lambda_schedule.precalculated_legs() if a.matrix(leg) == "pair"]


def frames_needed(n_frames: int, fraction: float) -> int:
    """The number of frames that ``fraction`` of ``n_frames`` is, rounded down ("frame-count
    thresholds are rounded down to integers in both selection methods"): of the fraction as it is
    written. 0.29 of 800 frames is 232; the product of the two binary numbers is
    231.99999999999997, and its integer part kept an interaction of 231 frames (review of
    eb6f89f). One rounding for the rule, the check of the configuration and ``cufep plan``."""
    from fractions import Fraction

    return int(Fraction(repr(float(fraction))) * int(n_frames))


def analysed_frames(cfg: "Config") -> int:
    """The frames that ``restraints.selection: interactions`` analyses at the time step of the
    configuration: those of every trajectory that are later than ``discard_ns`` (a frame after
    every interval and one at the end, in whole MD steps)."""
    from cufep.steps import md_steps

    ri = cfg.restraints.interactions
    dt = float(cfg.dynamics.timestep_fs)
    steps, per_frame = md_steps(ri.length_ns * 1000.0, dt), md_steps(ri.frame_interval_ps, dt)
    times = [min((i + 1) * per_frame, steps) * dt / 1000.0 for i in range(-(-steps // per_frame))]
    return int(ri.trajectories) * sum(1 for t in times if t > ri.discard_ns * 1000.0 + 1e-9)


#: Settings that earlier versions wrote (``cufep init --full``, ``resolved_config.yaml``) and
#: this one does not have: the value a file holds if the user never touched the setting, and
#: what replaced it. A file with that value is read and the line is ignored, with a notice. Any
#: other value was a choice of the user that cannot be carried over, so reading stops there.
REMOVED_SETTINGS = {
    ("analysis", "convergence_tolerance_sigma"): (
        2.0,
        "The checks of the analysis no longer use a multiple of the standard error. Their levels are the "
        "literature's: analysis.min_overlap (0.03) and analysis.min_effective_samples (50 uncorrelated samples "
        "in the lambda window with the fewest)",
        "A multiple of the standard error is no level of the literature: remove the line"),
    ("membrane", "area_tolerance"): (
        0.01,
        "Whether the bilayer has settled is no longer judged (a level of cuFEP's own): how its area and the "
        "water near the ligand changed at the end of its equilibration are numbers of the result (bilayer)",
        "Remove the line"),
    ("membrane", "pocket_water_tolerance"): (
        1.5,
        "Whether the bilayer has settled is no longer judged (a level of cuFEP's own): how its area and the "
        "water near the ligand changed at the end of its equilibration are numbers of the result (bilayer)",
        "Remove the line"),
    ("restraints", "pose_rmsd_warning_nm"): (
        0.3,
        "The distance of the ligand from its input pose is no longer judged (a level of cuFEP's own): it is a "
        "number of the result (ligand_rmsd_from_input_A), as OpenFE reports it without a limit",
        "Remove the line"),
    ("analysis", "halves_tolerance_kT"): (
        1.0,
        "The two halves of the production are no longer a check: their difference is shown with the analysis, "
        "without a verdict. The only checks are analysis.min_overlap and analysis.min_effective_samples, the "
        "levels of the literature",
        "Remove the line"),
    ("analysis", "estimators_tolerance_kcal_mol"): (
        1.0,
        "BAR against MBAR is no longer a check: both values are shown with the analysis, without a verdict. "
        "The only checks are analysis.min_overlap and analysis.min_effective_samples, the levels of the literature",
        "Remove the line"),
    ("lambda_schedule", "adaptive", "equilibration_check_ps"): (
        250.0,
        "Adaptive lambda scheduling chooses the lambda windows in its precalculation only and no longer adds "
        "windows during the equilibration",
        "Remove the line; lambda_schedule.adaptive.window_ps and equilibration_ps set the lengths of the "
        "precalculation"),
}


#: settings of earlier versions that have another name now: (old path) -> (new path, what changed).
#: Applied to configuration files and to the records of the settings stored with results and
#: samples alike (:func:`renamed_settings`)
RENAMED_SETTINGS = {
    ("analysis", "halves_error_blocks"): (
        ("analysis", "part_error_blocks"),
        "the number of time blocks for the standard error of a half of the production is now that of every "
        "part of the production (the halves, the forward and reverse series, the consecutive blocks)"),
}


def renamed_settings(section: str, record: dict) -> tuple[dict, list[str]]:
    """A flat record of the settings of one section (``record``, as results store it) with the
    names of this version, and what was renamed."""
    record, notices = dict(record), []
    for (old_section, old), ((_new_section, new), what) in RENAMED_SETTINGS.items():
        if old_section == section and old in record:
            value = record.pop(old)
            if new not in record:
                record[new] = value
            notices.append(f"{section}.{old}: {value} is {section}.{new} in this version ({what})")
    return record, notices


def migrate_settings(data: dict) -> tuple[dict, list[str]]:
    """``data`` of a configuration file as this version reads it, and notices about what was changed.

    Raises ValueError for a setting of an earlier version that the user had changed and that has
    no equivalent now.
    """
    data = copy.deepcopy(data)
    notices: list[str] = []
    for section in {path[0] for path in RENAMED_SETTINGS}:
        if isinstance(data.get(section), dict):
            data[section], renamed = renamed_settings(section, data[section])
            notices += renamed
    for path, (default, now, to_do) in REMOVED_SETTINGS.items():
        block = data
        for name in path[:-1]:
            block = block.get(name) if isinstance(block, dict) else None
        if not isinstance(block, dict) or path[-1] not in block:
            continue
        key, value = ".".join(path), block.pop(path[-1])
        untouched = isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == default
        if not untouched:
            raise ValueError(f"{key}: {value} is a setting of an earlier version of cuFEP that this version does "
                             f"not have. {now}. {to_do}.")
        notices.append(f"{key}: {value} is a setting of an earlier version and is not used; it holds the default "
                       f"of that version and is ignored. {now}.")
    return data, notices


def read_settings(path: str | Path) -> dict | None:
    """The mapping in a YAML file as it is written, without validation; None if it cannot be read.

    For records of earlier runs (``resolved_config.yaml``), which may come from another version.
    """
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def load_config_with_notices(path: str | Path) -> tuple[Config, list[str]]:
    """:func:`load_config`, and the notices about settings of earlier versions in the file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level of the configuration must be a mapping")
    data, notices = migrate_settings(data)
    cfg = Config.model_validate(data)
    notices += unpaired_threshold_notices(data, cfg)
    base = path.resolve().parent

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else (base / p).resolve()

    cfg.system.protein = resolve(cfg.system.protein)
    cfg.system.ligand = resolve(cfg.system.ligand)
    cfg.system.cofactors = [resolve(p) for p in cfg.system.cofactors]
    if cfg.lambda_schedule.file is not None:
        cfg.lambda_schedule.file = resolve(cfg.lambda_schedule.file)
    return cfg, notices


def load_config(path: str | Path) -> Config:
    """Read a YAML file. Relative input paths are resolved against the YAML's directory.

    A file written by an earlier version is read as far as its settings still exist or held their
    defaults (:func:`migrate_settings`); :func:`load_config_with_notices` also returns what was
    ignored or converted.
    """
    return load_config_with_notices(path)[0]


def dump_config(cfg: Config, path: str | Path | None = None) -> str:
    """Serialize a configuration to YAML (and write it if ``path`` is given)."""
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
    if path is not None:
        from cufep.files import publish_text

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        publish_text(path, text)
    return text
