"""The ligand table of a benchmark run so far, for reports during the run. Reads only.

Every ligand of the set, in the order of experiment: experiment and FEP+; where the binding free
energy exists (<run>/<ligand>/result.json) cuFEP's value, its standard error, both signed errors,
the legs, the lambda windows (and those adaptive lambda scheduling added), the lowest overlaps,
the scheduling mode and threshold each leg recorded in its schedule.yaml, and the flags; else the
state of each leg (its free energy of decoupling if finished, the phase or the iteration of the
replica-exchange run from the log if running, waiting otherwise). Below, the statistics of the
ligands with a result.

    scripts/in_docker.sh python spikes/benchmark_progress.py tyk2 runs/benchmark_tyk2_full
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from cufep.benchmarks.runner import flags_of, read_reference
from cufep.benchmarks.stats import metrics
from cufep.config.schema import load_config
from cufep.workflow import combine_ligand

KCAL = 4.184
LINE = re.compile(r"^\S+ \[(?P<lig>[^/\]]+)/r0/(?P<leg>complex|solvent)\] (?P<text>.*)$")
STARTED = re.compile(r"^\S+ started (?P<lig>[^/ ]+)/r0/(?P<leg>complex|solvent) on GPU")


def schedule_of(leg_dir: Path) -> str:
    path = leg_dir / "schedule.yaml"
    if not path.exists():
        return "-"
    data = yaml.safe_load(path.read_text()) or {}
    provenance = data.get("provenance") or {}
    matrix = provenance.get("overlap_matrix")                  # which overlap the threshold was compared with
    return f"{data.get('mode', '?')} {provenance.get('overlap_threshold')}" + (f" ({matrix})" if matrix else "")


def last_states(run: Path) -> dict[tuple[str, str], str]:
    """The last thing the log of the run says about each leg."""
    states: dict[tuple[str, str], str] = {}
    log = run.parent / f"{run.name}.log"
    if not log.exists():
        return states
    for line in log.read_text(errors="replace").splitlines():
        started = STARTED.match(line)
        if started:
            states.setdefault((started.group("lig"), started.group("leg")), "started (pre-equilibration)")
            continue
        m = LINE.match(line)
        if not m:
            continue
        text, key = m.group("text"), (m.group("lig"), m.group("leg"))
        it = re.match(r"iteration (\d+)", text)
        if it:
            states[key] = f"iteration {it.group(1)}"
        elif text.startswith("lambda schedule"):
            states[key] = "lambda schedule (ALS)"
        elif not states.get(key, "").startswith(("iteration", "lambda")):
            states[key] = "pre-equilibration"
    return states


def leg_state(run: Path, lig: str, leg: str, states: dict, total: int) -> str:
    result = run / lig / "repeat_0" / leg / "result.json"
    if result.exists():
        r = json.loads(result.read_text())
        if "delta_g_decouple" in r:
            return f"done {r['delta_g_decouple'] / KCAL:.2f} ({r.get('n_states')} windows)"
        return "FAILED"
    state = states.get((lig, leg))
    if state is None:
        return "waiting"
    return f"{state} / {total}" if state.startswith("iteration") else state


def main(name: str, run_dir: str) -> None:
    run = Path(run_dir)
    ref = read_reference(name)
    cfg = load_config(run / "resolved_config.yaml")
    s = yaml.safe_load((run / "resolved_config.yaml").read_text())["sampling"]
    total = round((float(s["equilibration_ns"]) + float(s["production_ns"])) * 1000 / float(s["time_per_iteration_ps"]))
    states = last_states(run)
    print("| ligand | experiment | FEP+ | FEP+-exp | cuFEP | SE | cuFEP-exp | dG complex | dG solvent | restraint "
          "| windows c (added) | windows s (added) | lowest overlap c / s | schedule c / s | flags |")
    print("|" + "---|" * 15)
    done = []
    for lig in sorted(ref, key=lambda n: ref[n]["exp"]):
        exp, fep = ref[lig]["exp"], ref[lig]["fepplus"]
        path = run / lig / "result.json"
        r = json.loads(path.read_text()) if path.exists() else {}
        if not r and all((run / lig / "repeat_0" / leg / "result.json").exists() for leg in ("complex", "solvent")):
            # both legs are finished; the run writes the ligand's result only at its end: combine
            # them as the run will, without writing anything
            r = combine_ligand(run / lig, lig, 1, cfg.analysis, publish=False)
        head = f"| {lig} | {exp:.2f} | {fep:.2f} | {fep - exp:+.2f} |"
        if r.get("dg_bind") is None:
            c, sv = (leg_state(run, lig, leg, states, total) for leg in ("complex", "solvent"))
            why = "; ".join(flags_of(r)) if r else ""
            cells = ["", "", "", f"complex: {c}", f"solvent: {sv}", "", "", "", "", "", why]
            print(head + "".join(f" {cell} |" for cell in cells))
            continue
        rep = (r.get("repeats") or [{}])[0]
        added, windows, low = (rep.get(k) or {} for k in ("lambda_windows_added", "lambda_windows", "min_overlap"))
        legs = {leg: run / lig / "repeat_0" / leg for leg in ("complex", "solvent")}

        def w(leg: str, added=added, windows=windows) -> str:
            extra = added.get(leg)
            return f"{windows.get(leg)} ({sum(extra.values()) if extra else 0})"

        print(f"{head} {r['dg_bind']:.2f} | {r['dg_bind_error']:.2f} | {r['dg_bind'] - exp:+.2f} | "
              f"{rep['dg_complex']:.2f} | {rep['dg_solvent']:.2f} | {rep['restraint_release']:.2f} | {w('complex')} | "
              f"{w('solvent')} | {low.get('complex', float('nan')):.3f} / {low.get('solvent', float('nan')):.3f} | "
              f"{schedule_of(legs['complex'])} / {schedule_of(legs['solvent'])} | {'; '.join(flags_of(r)) or '-'} |")
        done.append((exp, r["dg_bind"], fep))
    print(f"\n{len(done)} of {len(ref)} ligands have a binding free energy.")
    if len(done) >= 2:
        exps = [d[0] for d in done]
        for method, values in (("cuFEP", [d[1] for d in done]), ("FEP+", [d[2] for d in done])):
            m = metrics(values, exps)
            shown = ", ".join(f"{k} {'N/A' if m[k] is None else f'{m[k]:.2f}'}" for k in
                              ("mean_signed_error", "rmse", "mue", "centred_rmse", "pearson_r", "kendall_tau",
                               "spearman"))
            print(f"{method}, these {len(done)} ligands: {shown}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
