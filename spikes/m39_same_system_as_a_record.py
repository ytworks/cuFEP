"""Whether this tree, with the settings a recorded relative run was made with, builds the hybrid Systems and the
conditions the samples of that run record: the check a run makes before it goes on from a leg
(:func:`cufep.rbfe.leg.check_leg_reuse`: the settings, the lambda windows, every condition of the samples and the
checksum of the System), made for the legs of a run directory without running anything. Reads only.

Before a comparison of a change with a record: with the change switched off, the code of the comparison is the
code of the record as far as the samples can tell (plan of 2026-11-28, issue #27).

    scripts/in_docker.sh python spikes/m39_same_system_as_a_record.py runs/rbfe_bench_tyk2_957a9cf [EDGE ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main(run_dir: str, *edges: str) -> int:
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import check_leg_reuse

    run = Path(run_dir)
    rcfg = RbfeConfig.model_validate(yaml.safe_load((run / "resolved_config.yaml").read_text()))
    different = 0
    for edge_dir in sorted(path for path in (run / "edges").iterdir() if path.is_dir()):
        if edges and edge_dir.name not in edges:
            continue
        for leg in ("complex", "solvent"):
            try:
                check_leg_reuse(rcfg, edge_dir, leg, 0)
            except Exception as exc:  # noqa: BLE001  every refusal is a difference to report
                different += 1
                print(f"{edge_dir.name} {leg}: DIFFERENT: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
            else:
                print(f"{edge_dir.name} {leg}: the System and the conditions of the samples", flush=True)
    print(f"legs that differ: {different}")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
