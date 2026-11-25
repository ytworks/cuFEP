"""A leg as the scheduler runs it (:mod:`cufep.scheduler`): what it needs to place the leg on GPUs, and the function
that runs it in the job's process. A kind of job (a leg of a ligand of the ABFE, :class:`cufep.workflow.LegJob`; a
leg of an edge of a relative calculation, :class:`cufep.rbfe.leg.EdgeLegJob`) says what the leg is of
(:meth:`Job.identity`, as its files and results name it), where its directory is and which function runs it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(kw_only=True)
class Job:
    leg: str                       # complex | solvent
    repeat: int
    devices: list
    config: dict
    production_ns: float
    reschedule: bool = False
    inline: bool = False
    #: size of the system, and how many workers of this job one GPU runs at the same time
    n_atoms: int = 0
    oversubscription: int = 1
    #: set in the job's own process when a scheduler is in charge of the GPUs (cufep.workflow.JobControl)
    control: object | None = None

    #: what the leg is of, as its files and results name it ("ligand", "edge")
    unit_key: ClassVar[str] = ""

    @property
    def name(self) -> str:
        """The name of what the leg is of (the ligand, the edge)."""
        raise NotImplementedError

    @property
    def unit_dir(self) -> Path:
        """The directory of what the leg is of; the leg is in ``repeat_<n>/<leg>`` of it."""
        raise NotImplementedError

    def runner(self):
        """The function that runs the leg in the job's process and records a failure instead of raising."""
        raise NotImplementedError

    def identity(self) -> dict:
        """What the leg is of, its leg and its repeat, as its files and results name them."""
        return {self.unit_key: self.name, "leg": self.leg, "repeat": self.repeat}

    @property
    def leg_dir(self) -> Path:
        return Path(self.unit_dir) / f"repeat_{self.repeat}" / self.leg

    @property
    def tag(self) -> str:
        return f"{self.name}/r{self.repeat}/{self.leg}"
