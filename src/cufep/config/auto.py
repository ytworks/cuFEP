"""Automatic settings: turn every ``auto`` into a concrete value and explain the choice.

Static choices are made here (before any simulation). Dynamic ones are made later by the
step that has the necessary data and are recorded in the leg directories:
restraint atoms (``restraint.json``), the lambda schedule (``schedule.yaml``), the time
step fallback and the production length (``result.json``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cufep.config.schema import Config


@dataclass
class AutoReport:
    decisions: list[tuple[str, str, str]] = field(default_factory=list)   # (key, value, reason)
    warnings: list[str] = field(default_factory=list)

    def add(self, key: str, value, reason: str) -> None:
        self.decisions.append((key, str(value), reason))

    def to_markdown(self) -> str:
        lines = ["# cuFEP automatic settings", "", "| setting | value | reason |", "|---|---|---|"]
        lines += [f"| `{k}` | {v} | {r} |" for k, v, r in self.decisions]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        lines += ["", "Decided during the run and recorded per leg: restraint atoms (`restraint.json`), "
                  "lambda schedule (`schedule.yaml`), time-step fallback and production length (`result.json`)."]
        return "\n".join(lines) + "\n"


def resolve(cfg: Config, protein=None) -> tuple[Config, AutoReport]:
    """Return a copy of ``cfg`` without static ``auto`` values, and the reasons."""
    from cufep.scheduler import detect_gpus

    out = cfg.model_copy(deep=True)
    report = AutoReport()

    # ---- membrane ---------------------------------------------------------------------
    if out.membrane.enabled == "auto":
        if protein is None:
            out.membrane.enabled = False
            report.add("membrane.enabled", False, "no protein given to the resolver")
        else:
            from cufep.prep.orient import detect_membrane
            verdict = detect_membrane(protein.topology, protein.positions_nm())
            if verdict.confident:
                out.membrane.enabled = verdict.is_membrane
                report.add("membrane.enabled", verdict.is_membrane, verdict.reason)
            else:
                raise ValueError(
                    "cuFEP could not decide with confidence whether this is a membrane protein "
                    f"({verdict.reason}). Set `membrane.enabled: true` or `false` in the configuration.")
    else:
        report.add("membrane.enabled", out.membrane.enabled, "set by the user")
    membrane = bool(out.membrane.enabled)

    # ---- box --------------------------------------------------------------------------
    if out.solvation.box_shape == "auto":
        out.solvation.box_shape = "cube" if membrane else "dodecahedron"
        report.add("solvation.box_shape", out.solvation.box_shape,
                   "bilayers need a rectangular box" if membrane else
                   "fewest atoms for a soluble system (OpenFE's default)")

    # ---- dynamics -----------------------------------------------------------------------
    if out.dynamics.constraints == "auto":
        out.dynamics.constraints = "hbonds"
        report.add("dynamics.constraints", "hbonds", "bonds to hydrogen constrained, rigid water")
    if out.dynamics.timestep_fs == "auto":
        report.add("dynamics.timestep_fs", "4 (2 on instability)",
                   f"hydrogen mass {out.dynamics.hmr_mass_amu:g} amu; lowered automatically if the "
                   "pre-equilibration of the bound state is unstable")

    # ---- backend --------------------------------------------------------------------------
    if out.backend.name == "cupotential":
        from cufep.backends.cupotential import CuPotentialBackend
        ok, why = CuPotentialBackend.available() if out.backend.platform == "CUDA" else (False, "platform is not CUDA")
        if ok:
            report.add("backend.name", "cupotential", why)
        elif out.backend.fallback == "openmm":
            out.backend.name = "openmm"
            report.add("backend.name", "openmm", f"cuPotential unavailable: {why}")
            report.warnings.append(f"cuPotential is not available ({why}); using stock OpenMM")
        else:
            raise ValueError(f"cuPotential is not available ({why}) and backend.fallback is 'none'")

    # ---- GPUs -----------------------------------------------------------------------------
    if out.backend.platform != "CUDA":
        out.resources.gpus = []
        report.add("resources.gpus", "none", f"platform {out.backend.platform}")
    elif out.resources.gpus == "auto":
        gpus = detect_gpus()
        out.resources.gpus = gpus
        report.add("resources.gpus", gpus or "none", "every visible GPU that is not busy computing")
        if not gpus:
            report.warnings.append("no GPU found; the run will be extremely slow")
    return out, report
