"""Configuration schema (YAML -> validated settings).

Only ``system.protein`` and ``system.ligand`` are required. Every other key has a
default; keys set to ``auto`` are resolved by :mod:`cufep.config.auto` and the resolved
configuration is written next to the results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Auto = Literal["auto"]


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
    enabled: bool | Auto = "auto"
    #: auto = consensus of the built-in methods; input = the PDB is already oriented
    #: (normal along z, membrane centre at z = 0); or force one method
    orientation: Literal["auto", "input", "transfer", "slab", "memembed"] = "auto"
    lipid: str = "POPC"
    padding_nm: float = 1.5
    equilibration_ns: float = 10.0
    max_equilibration_ns: float = 30.0


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
    timestep_fs: float | Auto = "auto"
    constraints: Literal["auto", "hbonds", "none"] = "auto"
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
    force_constants: Literal["openfe", "custom"] = "openfe"
    k_r_kcal_per_A2: float = 10.0
    k_angle_kcal_per_rad2: float = 80.0
    #: manual selection: atom indices in the prepared complex (H0,H1,H2 / G0,G1,G2)
    host_atoms: list[int] | None = None
    guest_atoms: list[int] | None = None
    rmsf_cutoff_nm: float = 0.1
    host_min_distance_nm: float = 0.5
    host_max_distance_nm: float = 1.5
    dssp_filter: bool = True


class AlchemyConfig(_Section):
    softcore_alpha: float = 0.5


class WindowLimits(_Section):
    complex: int = 48
    solvent: int = 32


class LambdaScheduleConfig(_Section):
    mode: Literal["adaptive", "fixed"] = "adaptive"
    target_std_kT: float = 1.0
    min_overlap: float = 0.03
    max_windows: WindowLimits = Field(default_factory=WindowLimits)
    pilot_ps_per_window: float = 40.0
    #: for mode "fixed": path to a schedule.yaml written by an earlier run
    file: Path | None = None


class SamplingConfig(_Section):
    #: plain MD of the fully interacting system before the alchemical run (OpenFE: 0.25 + 0.5 + 5 ns)
    nvt_ns: float = 0.25
    npt_ns: float = 0.5
    pre_equilibration_ns: float = 5.0
    solvent_nvt_ns: float = 0.1
    solvent_npt_ns: float = 0.2
    solvent_pre_equilibration_ns: float = 0.5
    #: replica exchange
    iteration_ps: float = 2.5
    equilibration_ns: float = 1.0
    #: production length per state; checked only at these lengths (ns), the last one is the cap
    production_ns: list[float] = Field(default_factory=lambda: [5.0, 10.0])
    min_effective_samples: int = 50
    min_round_trips: int = 2
    exchange: Literal["hrex", "none"] = "hrex"
    repeats: int = 3
    seed: int = 20261118
    checkpoint_interval_ps: float = 250.0

    @field_validator("production_ns")
    @classmethod
    def _increasing(cls, v: list[float]) -> list[float]:
        if not v or any(b <= a for a, b in zip(v, v[1:])) or v[0] <= 0:
            raise ValueError("production_ns must be a non-empty, strictly increasing list of positive numbers")
        return v


class AnalysisConfig(_Section):
    estimator: Literal["mbar"] = "mbar"
    #: long-range dispersion correction for the ligand-environment LJ at the coupled end state
    lrc: Literal["reweight", "none"] = "reweight"
    lrc_cutoff_nm: float = 2.0


class ResourcesConfig(_Section):
    #: "auto" = every visible GPU that is not busy; or an explicit list of device indices
    gpus: Auto | list[int] = "auto"
    cpu_threads_per_worker: int = 2


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


def load_config(path: str | Path) -> Config:
    """Read a YAML file. Relative input paths are resolved against the YAML's directory."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level of the configuration must be a mapping")
    cfg = Config.model_validate(data)
    base = path.resolve().parent

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else (base / p).resolve()

    cfg.system.protein = resolve(cfg.system.protein)
    cfg.system.ligand = resolve(cfg.system.ligand)
    cfg.system.cofactors = [resolve(p) for p in cfg.system.cofactors]
    if cfg.lambda_schedule.file is not None:
        cfg.lambda_schedule.file = resolve(cfg.lambda_schedule.file)
    return cfg


def dump_config(cfg: Config, path: str | Path | None = None) -> str:
    """Serialize a configuration to YAML (and write it if ``path`` is given)."""
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text
