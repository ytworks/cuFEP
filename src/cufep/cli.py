"""Command line interface."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from cufep import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def main() -> None:
    """cuFEP: absolute binding free energies on OpenMM (cuPotential backend by default).

    The only required inputs are a protein PDB and a ligand SDF with the ligand placed in
    the pocket:

    \b
        cufep init protein.pdb ligand.sdf -o config.yaml
        cufep plan config.yaml      # show what will be done, without simulating
        cufep run config.yaml       # run (or resume) everything
    """


@main.command()
@click.argument("protein", type=click.Path(exists=True, dir_okay=False))
@click.argument("ligand", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default="config.yaml", show_default=True, help="YAML file to write.")
@click.option("--full", is_flag=True, help="Write every setting with its default instead of the minimal file.")
def init(protein: str, ligand: str, output: str, full: bool) -> None:
    """Write a configuration for PROTEIN (PDB) and LIGAND (SDF)."""
    import yaml

    from cufep.config.schema import Config, dump_config

    out = Path(output)

    def rel(p):
        if Path(p).resolve().is_relative_to(out.resolve().parent):
            return str(Path(p).resolve().relative_to(out.resolve().parent))
        return str(Path(p).resolve())

    data = {"system": {"protein": rel(protein), "ligand": rel(ligand)},
            "output": {"dir": f"runs/{Path(ligand).stem}"}}
    if full:
        cfg = Config.model_validate(data)
        text = dump_config(cfg)
    else:
        text = ("# cuFEP configuration. Everything not listed here uses its default or is chosen\n"
                "# automatically; `cufep init --full` writes all settings.\n" + yaml.safe_dump(data, sort_keys=False))
    from cufep.files import publish_text

    publish_text(out, text)
    click.echo(f"wrote {out}")


def _load(config: str):
    from cufep.config.schema import load_config_with_notices
    try:
        cfg, notices = load_config_with_notices(config)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"{config}: {exc}") from exc
    for notice in notices:
        click.echo(f"note: {config}: {notice}", err=True)
    return cfg


def _apply_overrides(cfg, repeats, production_ns, gpus, output, backend):
    if repeats is not None:
        cfg.sampling.repeats = repeats
    if production_ns is not None:
        cfg.sampling.production_ns = float(production_ns)
    if gpus:
        cfg.resources.gpus = "auto" if gpus == "auto" else [int(x) for x in gpus.split(",")]
    if output:
        cfg.output.dir = Path(output)
    if backend:
        cfg.backend.name = backend
    return cfg


def lambda_window_table(cfg) -> str:
    """The lambda windows a run with this configuration uses, per leg and stage."""
    from cufep.alchemy.schedule import stage_values
    from cufep.config.schema import WindowsConfig

    ls = cfg.lambda_schedule
    lines = ["# lambda windows"]
    if ls.file is not None:
        return "\n".join(lines + [f"taken from {ls.file}"])
    adaptive = ls.precalculated_legs()
    if adaptive:
        a = ls.adaptive
        fixed = [leg for leg in ("complex", "solvent") if leg not in adaptive]

        def per_leg(text) -> str:
            """What ``text`` says of a leg: once if the legs agree, else leg by leg."""
            texts = {leg: text(leg) for leg in adaptive}
            if len(set(texts.values())) == 1:
                return texts[adaptive[0]]
            return "; ".join(f"{leg} leg {t}" for leg, t in texts.items())

        def threshold(leg: str) -> str:
            alone = " (the overlap of the two windows alone)" if a.matrix(leg) == "pair" else ""
            return f"at or below {a.threshold(leg):g}{alone}"

        lines += [
            "mode adaptive" + (f" in the {adaptive[0]} leg" if fixed else "")
            + ": adaptive lambda scheduling decides the windows in a precalculation.",
            f"  a window is added where the overlap of neighbouring windows is {per_leg(threshold)};",
            f"  {per_leg(lambda leg: str(a.candidates_of(leg)))} candidate lambda values per stage; "
            "MD per new window: "
            + "; ".join(
                f"{leg} leg {a.window(leg):g} ps"
                + (f" after {a.equilibration(leg):g} ps of equilibration" if a.equilibration(leg) > 0 else "")
                for leg in adaptive
            )
            + ";",
            "  start: "
            + "; ".join(
                f"{leg} leg " + ", ".join(
                    # a number of evenly spaced windows, or the lambda values that were given
                    f"{k} {v}" if isinstance(v, int) else f"{k} {len(v)} ({' '.join(f'{x:g}' for x in v)})"
                    for k, v in a.initial_windows.for_leg(leg).items())
                for leg in adaptive
            )
            + "; at most " + " / ".join(f"{getattr(a.max_windows, leg)} ({leg})" for leg in adaptive) + " windows.",
            (f"mode fixed in the {fixed[0]} leg: its windows below are used (those of the {adaptive[0]} leg are "
             "for reference)." if fixed else "The windows of mode fixed, for reference:"),
        ]
    else:
        lines.append("mode fixed: the windows below are used.")
    recommended = WindowsConfig()
    for leg in ("complex", "solvent"):
        total = 0
        for stage, spec in ls.windows.for_leg(leg).items():
            values = stage_values(spec, recommended.for_leg(leg)[stage])
            total += len(values)
            lines.append(f"  {leg:8s} {stage:10s} {len(values):2d}: " + " ".join(f"{v:g}" for v in values))
        lines.append(
            f"  {leg:8s} {'total':10s} {total - (len(ls.windows.for_leg(leg)) - 1):2d} (stage boundaries are shared)"
        )
    r2 = cfg.sampling.rest2
    if r2.scale > 1.0:
        region = (
            "the ligand"
            if r2.region == "ligand" and not r2.residues
            else (
                "the ligand and protein residues"
                + (f" within {r2.pocket_cutoff_nm:g} nm of it" if r2.region == "pocket" else "")
                + (f" ({', '.join(r2.residues)})" if r2.residues else "")
            )
        )
        lines += ["# REST2", f"hot region: {region}; {r2.scale:g} times the temperature in the middle of each leg, "
                             "unscaled in the first and the last lambda window."]
    else:
        lines += ["# REST2", "off (sampling.rest2.scale is 1)."]
    return "\n".join(lines)


def restraint_selection_lines(cfg) -> str:
    """How a run with this configuration chooses the Boresch restraint of the complex leg."""
    from cufep.config.schema import analysed_frames, frames_needed

    rc = cfg.restraints
    lines = ["# restraint"]
    if rc.selection == "interactions":
        ri = rc.interactions
        n = analysed_frames(cfg)
        lines += [
            "selection interactions: the anchors are chosen from the protein-ligand interactions of unrestrained MD.",
            f"  {ri.trajectories} trajectories of {ri.length_ns:g} ns from the pose that was given (held while the "
            "solvent settles), a frame "
            f"every {ri.frame_interval_ps:g} ps, the first {ri.discard_ns:g} ns of each left out: {n} frames;",
            f"  on the backend of the leg ({cfg.backend.name}, fallback {cfg.backend.fallback});",
            *(["  of the complex in the bilayer; the alchemical run is made on the complex in water "
               "(membrane.complex_leg: water);"] if cfg.membrane.for_restraint_only else []),
            f"  an interaction is kept in {frames_needed(n, ri.min_frame_fraction)} frames or more "
            f"({ri.min_frame_fraction:g}), anchor angles within {ri.angle_range_deg[0]:g} to "
            f"{ri.angle_range_deg[1]:g} degrees;",
            f"  without one: the backbone within {ri.fallback_cutoff_nm:g} nm of the ligand in "
            f"{frames_needed(n, ri.fallback_frame_fraction)} frames or more ({ri.fallback_frame_fraction:g}).",
        ]
    elif rc.selection == "manual":
        lines.append(f"selection manual: host atoms {rc.host_atoms}, guest atoms {rc.guest_atoms}.")
    else:
        lines.append("selection auto: the anchors with the smallest fluctuations over the frames of the "
                     "pre-equilibration.")
    return "\n".join(lines)


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def plan(config: str) -> None:
    """Resolve automatic settings and check the inputs, without simulating."""
    from cufep import run as driver
    from cufep.config.schema import dump_config

    cfg = _load(config)
    try:
        resolved, report, _protein, ligands = driver.plan(cfg, click.echo)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    click.echo("")
    click.echo(report.to_markdown())
    click.echo(f"ligands ({len(ligands)}): " + ", ".join(f"{lig.name} [{lig.formal_charge:+d}]" for lig in ligands))
    click.echo("\n" + lambda_window_table(resolved))
    click.echo("\n" + restraint_selection_lines(resolved))
    click.echo("\n# resolved configuration\n" + dump_config(resolved))


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--repeats", type=int, help="Override sampling.repeats.")
@click.option("--production-ns", type=float, help="Override sampling.production_ns (both legs).")
@click.option("--gpus", help="Override resources.gpus: 'auto' or a comma separated list of CUDA device numbers "
                             "(as CUDA_VISIBLE_DEVICES leaves them).")
@click.option("--backend", type=click.Choice(["cupotential", "openmm"]), help="Override backend.name.")
@click.option("-o", "--output", help="Override output.dir.")
@click.option("--ligand", "ligands", multiple=True,
              help="Run only this ligand (may be given more than once). The other ligands keep their results.")
@click.option("--leg", "legs", multiple=True, type=click.Choice(["complex", "solvent"]),
              help="Run only this leg (may be given more than once). The other leg keeps its result.")
@click.option("--reschedule", is_flag=True,
              help="Determine the lambda windows again and start the legs anew; the windows and samples a leg "
                   "already has are moved to previous_<n>/ in its directory.")
def run(config: str, repeats, production_ns, gpus, backend, output, ligands, legs, reschedule: bool) -> None:
    """Run (or resume) the calculation described by CONFIG.

    Run again in the same output directory, a leg goes on from where it stopped: an unfinished
    leg resumes, and a finished leg runs on if the production is now longer
    (sampling.production_ns or --production-ns). To run one flagged leg again, combine --ligand
    and --leg with --production-ns (longer) or --reschedule (lambda windows determined again).
    """
    from cufep import run as driver

    cfg = _apply_overrides(_load(config), repeats, production_ns, gpus, output, backend)
    try:
        summary = driver.run(cfg, reschedule=reschedule, only_ligands=ligands, only_legs=legs)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    for note in driver.waiting_for_partner(summary):
        click.echo(f"note: {note}")
    failed = driver.failed_legs(summary)
    if failed:
        click.echo("failed: " + "; ".join(failed), err=True)
        sys.exit(1)


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--config", "config", type=click.Path(exists=True, dir_okay=False),
              help="Take the settings of the analysis from this configuration instead of those the leg was "
                   "analysed with (its analysis section and the levels for the notes on the bilayer and the pose; "
                   "nothing else of it is used).")
@click.option("--equilibration-ns", type=float,
              help="Length of the equilibration per lambda window. Needed only for a run of an earlier version "
                   "that was interrupted: it did not record where its production begins.")
@click.option("--equilibration-iterations", type=click.IntRange(min=0),
              help="The same as a number of iterations, exactly (where the length of an iteration that run asked "
                   "for is not recorded and the one it had differs from it by a fraction of a step).")
def analyze(directory: str, config, equilibration_ns, equilibration_iterations) -> None:
    """Analyse a finished, running or interrupted leg directory (the one containing hrex/).

    The analysis uses the settings of the leg's own run (overlap level, numbers of points and
    blocks, tail correction ...), so that it says what the run said: those its result records,
    for a leg without a result those its run stored with the samples before it began, else
    those of the result it had before. --config gives others. Temperature and the length of an
    iteration are always those of the samples. Only prints; no result file is written or changed.
    """
    from cufep import jsonio
    from cufep.workflow import WorkflowError, analyse_leg_directory

    try:
        done = analyse_leg_directory(directory, _load(config) if config is not None else None, equilibration_ns,
                                     equilibration_iterations)
    except WorkflowError as exc:
        raise click.ClickException(str(exc)) from exc
    result, samples, n_equilibration = done["result"], done["samples"], done["n_equilibration"]
    attrs = samples["attrs"]
    source = f"--config {config}" if config is not None else done["source"]
    out = {
        k: v
        for k, v in result["analysis"].items()
        if k not in ("f_k", "df_k", "overlap_matrix", "overlap_used_matrix", "overlap_used_weighted_by_number")
    }
    out.update(n_equilibration_iterations=n_equilibration, n_iterations_stored=int(samples["u"].shape[0]))
    if result.get("equilibration_detected") is not None:
        # analysis.detect_equilibration: what the detection left out after the equilibration of the run
        out["equilibration_detected"] = result["equilibration_detected"]
    if "n_planned_iterations" in attrs:
        out["n_planned_iterations"] = int(attrs["n_planned_iterations"])
    # the radii of the tail correction that were really used: the outer one the samples recorded
    # their tail energies to, the inner one their interactions were cut off at
    out.update(lrc_long_cutoff_nm=result.get("lrc_long_cutoff_nm"), lrc_cutoff_nm=attrs.get("lrc_cutoff_nm"))
    out.update(delta_g_decouple=result["delta_g_decouple"], lrc=result["lrc"], lrc_method=result["lrc_method"],
               quality=result["quality"], notes=result["notes"], analysis_record=result["analysis_record"],
               analysis_settings_from=source)
    if result.get("restraint_choice") is not None:
        # what a restraint chosen from interactions rests on, as the result of the leg has it
        out["restraint_choice"] = result["restraint_choice"]
    click.echo(jsonio.dumps(out, indent=1))


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--ligand", help="Ligand name (default: the first one in the SDF).")
@click.option("--device", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Override output.dir.")
def validate(config: str, ligand, device: int, output) -> None:
    """Compare the two backends' alchemical energies for the system in CONFIG.

    Prints the differences between the states as numbers, without a verdict; ends with an error
    if a leg could not be compared."""
    from cufep.validate import validate as run_validation

    cfg = _apply_overrides(_load(config), None, None, None, output, None)
    try:
        ok = run_validation(cfg, ligand, device, click.echo)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    sys.exit(0 if ok else 1)


@main.group()
def rbfe() -> None:
    """Relative binding free energies of a network of ligands (its own YAML; docs/rbfe.md).

    \b
        cufep rbfe plan config.yaml     # ligands, edges and mappings, without simulating
        cufep rbfe run config.yaml      # run (or resume) every edge
        cufep rbfe analyze OUTPUT_DIR   # the ddG and dG from the results the legs have
    """


def _load_rbfe(config: str):
    from cufep.rbfe.config import load_rbfe_config

    try:
        return load_rbfe_config(config)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"{config}: {exc}") from exc


@rbfe.command("plan")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def rbfe_plan(config: str) -> None:
    """Show the ligands, the edges and their atom mappings of CONFIG, without simulating."""
    from cufep.rbfe.mapping import kartograf_mapping, lomap_score, mapping_settings
    from cufep.rbfe.run import edges_of, experiment_of, ligands_of

    rcfg = _load_rbfe(config)
    try:
        ligands = ligands_of(rcfg)
        edges, _how = edges_of(rcfg, ligands, click.echo)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    from cufep.rbfe.charge import charge_change_problems

    # what `cufep rbfe run` refuses, the plan refuses (review of PR #19: a change of two charges was shown as possible)
    problems = charge_change_problems(ligands, edges)
    if problems:
        raise click.ClickException("; ".join(problems))
    experiment = experiment_of(rcfg, ligands)
    by_name = {lig.name: lig for lig in ligands}
    m = rcfg.mapping
    settings = mapping_settings(m)
    for lig in ligands:
        exp = experiment.get(lig.name)
        click.echo(f"ligand {lig.name}: {lig.n_atoms} atoms" + (f", experiment {exp:.2f} kcal/mol" if exp else ""))
    for a, b in edges:
        mapping = kartograf_mapping(by_name[a], by_name[b], settings)
        dq = by_name[b].formal_charge - by_name[a].formal_charge
        click.echo(f"edge {a} -> {b}: {len(mapping)} atoms mapped, {by_name[a].n_atoms - len(mapping)} unique to {a}, "
                   f"{by_name[b].n_atoms - len(mapping)} unique to {b}; LOMAP score "
                   f"{lomap_score(by_name[a], by_name[b], mapping):.2f}"
                   + (f"; the net charge changes by {dq:+d} (an alchemical water)" if dq else ""))
    ls = rcfg.lambda_schedule
    click.echo(f"lambda windows: {ls.mode}" + (f" from {ls.adaptive.initial_windows}, overlap threshold "
                                                 f"{ls.adaptive.overlap_threshold:g} ({ls.adaptive.legs} leg decides)"
                                                 if ls.mode == "adaptive" else f", {ls.windows}"))
    # the reference the run takes: of the compounds where the ligands are states of them (system.states, read and
    # checked as the run reads it; eleventh review of PR #19)
    from cufep.rbfe.run import compound_experiment, states_of

    try:
        states = states_of(rcfg, ligands)
        exp = experiment_of(rcfg, ligands, states)
        of_compounds = compound_experiment(states, exp) if states is not None else None
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    reference = (("mean of the experimental values of the compounds " + ", ".join(of_compounds) if of_compounds
                  else "mean of the network") if states is not None
                 else ("mean of the experimental values of " + ", ".join(exp) if exp else "mean of the network"))
    click.echo(f"per window {rcfg.sampling.equilibration('complex'):g} ns equilibration and "
               f"{rcfg.sampling.production_ns:g} ns production; dG of every "
               + ("compound" if states is not None else "ligand") + f" from the edges, relative to the {reference}")


@rbfe.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--production-ns", type=float, help="Override sampling.production_ns.")
@click.option("--gpus", help="Override resources.gpus: 'auto' or a comma separated list of CUDA device numbers.")
@click.option("--backend", type=click.Choice(["cupotential", "openmm"]), help="Override backend.name.")
@click.option("-o", "--output", help="Override output.dir.")
@click.option("--edge", "edges", multiple=True, help="Run only this edge (A~B); repeat for several.")
@click.option("--leg", "legs", multiple=True, type=click.Choice(["complex", "solvent"]),
              help="Run only this leg of the edges; the other keeps the result it has.")
@click.option("--repeat", "repeats", multiple=True, type=int,
              help="Run only this repeat of the edges (0, 1, ...); repeat for several.")
@click.option("--reschedule", is_flag=True,
              help="Start the legs that run anew: their windows, samples and result (and a pre-equilibration that "
                   "cannot be used again) are set aside in previous_<n>/.")
def rbfe_run(config: str, production_ns, gpus, backend, output, edges, legs, repeats, reschedule: bool) -> None:
    """Run (or resume) the relative calculation described by CONFIG."""
    from cufep.rbfe.run import failed_edges, run_rbfe

    rcfg = _load_rbfe(config)
    if production_ns is not None:
        rcfg.sampling.production_ns = float(production_ns)
    if gpus:
        rcfg.resources.gpus = "auto" if gpus == "auto" else [int(x) for x in gpus.split(",")]
    if output:
        rcfg.output.dir = Path(output)
    if backend:
        rcfg.backend.name = backend
    try:
        summary = run_rbfe(rcfg, reschedule=reschedule, only_edges=list(edges) or None, only_legs=list(legs) or None,
                           only_repeats=list(repeats) or None)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    missing = failed_edges(summary)
    if missing:
        click.echo("no ddG: " + ", ".join(missing), err=True)
    # the exit code is that of the jobs of this run: a run of some legs that all succeeded is not a failure, a
    # failed repeat next to one that gave the ddG is (review of PR #17)
    failed = summary.get("failed_in_this_run") or []
    if failed:
        click.echo("failed in this run: " + ", ".join(failed), err=True)
        sys.exit(1)


@rbfe.command("analyze")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def rbfe_analyze(directory: str) -> None:
    """Print the ddG of the edges and the dG of the ligands of the output DIRECTORY of a relative run, from the
    results its legs have. Writes nothing."""
    from cufep.rbfe.run import analyse_directory, result_lines

    try:
        summary = analyse_directory(Path(directory))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    for line in result_lines(summary):
        click.echo(line)


@main.group()
def benchmark() -> None:
    """Built-in benchmarks against experiment and FEP+ (TYK2 kinase, A2A receptor in a bilayer)."""


@benchmark.command("list")
def benchmark_list() -> None:
    """Show the benchmark sets."""
    from cufep.benchmarks.runner import load_registry, read_reference
    for name, entry in load_registry().items():
        click.echo(
            f"{name}: {entry['title']} ({len(read_reference(name))} ligands; subset: {', '.join(entry['subset'])})"
        )


@benchmark.command("run")
@click.argument("name")
@click.option("--tier", type=click.Choice(["smoke", "subset", "full"]), default="smoke", show_default=True,
              help="smoke: one ligand, very short; subset: five ligands; full: every ligand.")
@click.option("-o", "--output", help="Output directory (default: runs/benchmark_<name>_<tier>).")
@click.option("--gpus", help="'auto' or a comma separated list of CUDA device numbers.")
@click.option("--repeats", type=int, help="Override the number of repeats.")
@click.option("--protocol", type=click.Choice(["abfe", "rbfe"]), default="abfe", show_default=True,
              help="abfe: absolute binding free energies of every ligand; rbfe: relative calculations on the network "
                   "of the ligands (smoke: the first edge of the relative example).")
def benchmark_run(name: str, tier: str, output, gpus, repeats, protocol: str) -> None:
    """Run benchmark NAME (resumable)."""
    from cufep import run as driver
    from cufep.benchmarks.runner import benchmark_config, example_notices, report

    if protocol == "rbfe":
        _benchmark_rbfe(name, tier, output, gpus, repeats)
        return
    try:
        example, notices = example_notices(name)
        for notice in notices:
            click.echo(f"note: {example}: {notice}", err=True)
        cfg = _apply_overrides(benchmark_config(name, tier, output), repeats, None, gpus, None, None)
        summary = driver.run(cfg)
        click.echo(report(name, cfg.output.dir))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    for note in driver.waiting_for_partner(summary):
        click.echo(f"note: {note}")
    failed = driver.failed_legs(summary)
    if failed:
        click.echo("failed: " + "; ".join(failed), err=True)
        sys.exit(1)


def _benchmark_rbfe(name: str, tier: str, output, gpus, repeats) -> None:
    """``cufep benchmark run NAME --protocol rbfe``: the relative run of the set and its report; the exit code is that
    of ``cufep rbfe run``."""
    from cufep.benchmarks.rbfe import rbfe_benchmark_config
    from cufep.benchmarks.rbfe import report as rbfe_report
    from cufep.rbfe.run import failed_edges, run_rbfe

    try:
        rcfg = rbfe_benchmark_config(name, tier, output)
        if gpus:
            rcfg.resources.gpus = "auto" if gpus == "auto" else [int(x) for x in gpus.split(",")]
        if repeats is not None:
            rcfg.sampling.repeats = int(repeats)
        summary = run_rbfe(rcfg)
        click.echo(rbfe_report(name, rcfg.output.dir))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    missing = failed_edges(summary)
    if missing:
        click.echo("no ddG: " + ", ".join(missing), err=True)
    failed = summary.get("failed_in_this_run") or []
    if failed:
        click.echo("failed in this run: " + ", ".join(failed), err=True)
        sys.exit(1)


@benchmark.command("report")
@click.argument("name")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def benchmark_report(name: str, directory: str) -> None:
    """Compare a benchmark run in DIRECTORY with experiment and FEP+ (an absolute or a relative run)."""
    from cufep.benchmarks.rbfe import is_relative_run
    from cufep.benchmarks.rbfe import report as rbfe_report
    from cufep.benchmarks.runner import report
    try:
        click.echo(rbfe_report(name, directory) if is_relative_run(directory) else report(name, directory))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc


@benchmark.command("freeze")
@click.argument("name")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--dest", help="Where to write the schedules (default: DIRECTORY/frozen_schedules).")
def benchmark_freeze(name: str, directory: str, dest) -> None:
    """Save the lambda schedules of a finished benchmark run for exact re-runs."""
    from cufep.benchmarks.rbfe import is_relative_run
    from cufep.benchmarks.runner import freeze_schedules
    if is_relative_run(directory):
        raise click.ClickException(f"{directory} is a relative run: its lambda windows cannot be given to a later run "
                                   "as a file (a relative calculation has no lambda_schedule.file)")
    try:
        click.echo(f"schedules written to {freeze_schedules(name, directory, dest)}")
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc


@main.command("check-env")
@click.option("--device", type=int, default=0, show_default=True)
def check_env(device: int) -> None:
    """Report the software environment and test both backends on a small water box."""
    from cufep.envcheck import check_environment

    ok = check_environment(device, click.echo)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
