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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    #: the result carries a warning if the bilayer has not settled by the end of that MD: if,
    #: between its last two quarters, the area in the membrane plane changes by more than this
    #: fraction, or the number of water molecules within 0.6 nm of the ligand by more than the
    #: number below. The run is not made longer
    area_tolerance: float = Field(default=0.01, gt=0.0)
    pocket_water_tolerance: float = Field(default=1.5, gt=0.0)


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


class SolvationConfig(_Section):
    box_shape: Literal["auto", "cube", "dodecahedron"] = "auto"
    padding_complex_nm: float = 1.0
    padding_solvent_nm: float = 1.5


class RestraintConfig(_Section):
    type: Literal["boresch"] = "boresch"
    selection: Literal["auto", "manual"] = "auto"
    #: openfe or custom: the two values below; auto: kT / sigma^2 of each restrained coordinate in
    #: the pre-equilibration, kept within 0.25-4 times the two values' defaults
    force_constants: Literal["openfe", "auto", "custom"] = "openfe"
    k_r_kcal_per_A2: float = 10.0
    k_angle_kcal_per_rad2: float = 80.0
    #: manual selection: atom indices in the prepared complex (H0,H1,H2 / G0,G1,G2)
    host_atoms: list[int] | None = None
    guest_atoms: list[int] | None = None
    rmsf_cutoff_nm: float = 0.1
    host_min_distance_nm: float = 0.5
    host_max_distance_nm: float = 1.5
    dssp_filter: bool = True
    #: the result carries a warning when the ligand's heavy atoms are on average further than
    #: this from the input pose over the second half of the pre-equilibration MD
    pose_rmsd_warning_nm: float = Field(0.3, gt=0)


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
    restrain: int = Field(default=3, ge=2)
    discharge: int = Field(default=5, ge=2)
    vanish: int = Field(default=9, ge=2)


class SolventInitialWindows(_Section):
    discharge: int = Field(default=5, ge=2)
    vanish: int = Field(default=9, ge=2)


class InitialWindows(_Section):
    """Equally spaced windows per leg and stage with which adaptive lambda scheduling starts."""

    complex: ComplexInitialWindows = Field(default_factory=ComplexInitialWindows)
    solvent: SolventInitialWindows = Field(default_factory=SolventInitialWindows)

    def for_leg(self, leg: str) -> dict[str, int]:
        section = self.complex if leg == "complex" else self.solvent
        return {name: getattr(section, name) for name in type(section).model_fields}


class WindowLimits(_Section):
    complex: int = 48
    solvent: int = 32


class PrecalculationLengths(_Section):
    """A length of the precalculation of adaptive lambda scheduling for each leg, in ps."""

    complex: float = Field(ge=0.0)
    solvent: float = Field(ge=0.0)


class AdaptiveConfig(_Section):
    """Adaptive lambda scheduling (Midgley et al., J. Chem. Inf. Model. 2025, 65, 512)."""

    #: a pair of neighbouring windows is deficient if its overlap is at or below this value
    overlap_threshold: float = Field(default=0.04, gt=0.0, lt=0.5)
    #: number of equally spaced candidate lambda values per stage
    candidates: int = Field(default=101, ge=3)
    #: windows per leg and stage in the first iteration (the schedule never gets smaller than this)
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
    #: fixed: the windows below are used as they are
    mode: Literal["adaptive", "fixed"] = "adaptive"
    windows: WindowsConfig = Field(default_factory=WindowsConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    #: fixed mode: MD in each window, one window after the other, that gives the replicas their
    #: start configurations
    fixed_walk_ps: float = Field(default=50.0, gt=0.0)
    #: a schedule.yaml written by an earlier run (or a directory of them, as written by
    #: `cufep benchmark freeze`); if given it is used as it is, whatever the mode
    file: Path | None = None


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
    """REST2, replica exchange with solute scaling (Wang, Friesner, Berne 2011), as FEP+ and SOMD2 use it.

    The energy inside a hot region is scaled so that the region behaves as if it were at a
    higher temperature in the lambda windows in the middle of a leg; the first and the last
    window are unscaled. The ligand is always in the hot region.
    """

    #: temperature of the hot region relative to that of the run in the middle of a leg (the
    #: meaning of SOMD2's rest2_scale); 1 switches REST2 off. Kept low enough that amide bonds
    #: do not isomerise (docs/configuration.md)
    scale: float = Field(default=1.5, ge=1.0)
    #: complex leg: ligand = the ligand only; pocket = also the protein residues that have a
    #: heavy atom within pocket_cutoff_nm of a heavy atom of the ligand in the prepared structure
    region: Literal["ligand", "pocket"] = "ligand"
    pocket_cutoff_nm: float = Field(default=0.4, gt=0.0)
    #: protein residues added to the hot region: residue numbers as in the PDB file, "960" or
    #: with the chain "A:960"
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
    #: 10 ns is the default of OpenFE's ABFE protocol (FEP+ uses 5 ns): with 5 ns the lambda
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
    #: the result carries a warning if the lambda window with the fewest independent samples has
    #: fewer than this (recommended for MBAR by Mey et al. 2020)
    min_effective_samples: float = Field(default=50.0, gt=0.0)
    #: ... if the free energies from the first and the second half of the production differ by
    #: more than this many kT ("differences over 1 kBT indicating poor convergence", Mey et al. 2020)
    halves_tolerance_kT: float = Field(default=1.0, gt=0.0)
    #: ... if BAR between neighbouring windows and MBAR differ by more than this (Mey et al. 2020)
    estimators_tolerance_kcal_mol: float = Field(default=1.0, gt=0.0)
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


#: Settings that earlier versions wrote (``cufep init --full``, ``resolved_config.yaml``) and
#: this one does not have: the value a file holds if the user never touched the setting, and
#: what replaced it. A file with that value is read and the line is ignored, with a notice. Any
#: other value was a choice of the user that cannot be carried over, so reading stops there.
REMOVED_SETTINGS = {
    ("analysis", "convergence_tolerance_sigma"): (
        2.0,
        "The checks of the analysis no longer use a multiple of the standard error. Their levels are "
        "analysis.min_effective_samples (50 independent samples in the lambda window with the fewest), "
        "analysis.halves_tolerance_kT (the two halves of the production within 1 kT) and "
        "analysis.estimators_tolerance_kcal_mol (BAR and MBAR within 1 kcal/mol)",
        "A multiple of the standard error cannot be turned into kT or kcal/mol: remove the line and set those "
        "three where their defaults do not suit (docs/configuration.md)"),
    ("analysis", "halves_error_blocks"): (
        5.0,
        "The number of time blocks for the standard error of a half of the production is now that of every "
        "part of the production (the halves, the forward and reverse series, the consecutive blocks)",
        "Use analysis.part_error_blocks"),
    ("lambda_schedule", "adaptive", "equilibration_check_ps"): (
        250.0,
        "Adaptive lambda scheduling chooses the lambda windows in its precalculation only and no longer adds "
        "windows during the equilibration",
        "Remove the line; lambda_schedule.adaptive.window_ps and equilibration_ps set the lengths of the "
        "precalculation"),
}


def migrate_settings(data: dict) -> tuple[dict, list[str]]:
    """``data`` of a configuration file as this version reads it, and notices about what was changed.

    Raises ValueError for a setting of an earlier version that the user had changed and that has
    no equivalent now.
    """
    data = copy.deepcopy(data)
    notices: list[str] = []
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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text
