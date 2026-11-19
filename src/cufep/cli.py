"""Command line interface."""
from __future__ import annotations

import json
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
    rel = lambda p: str(Path(p).resolve().relative_to(out.resolve().parent)) if Path(p).resolve().is_relative_to(out.resolve().parent) else str(Path(p).resolve())  # noqa: E731
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
    if ls.mode == "adaptive":
        a = ls.adaptive
        lines += ["mode adaptive: adaptive lambda scheduling decides the windows in a precalculation.",
                  f"  a window is added where the overlap of neighbouring windows is at or below {a.overlap_threshold:g};",
                  f"  {a.candidates} candidate lambda values per stage; MD per new window: " + "; ".join(
                      f"{leg} leg {a.window(leg):g} ps"
                      + (f" after {a.equilibration(leg):g} ps of equilibration" if a.equilibration(leg) > 0 else "")
                      for leg in ("complex", "solvent")) + ";",
                  "  start: " + "; ".join(
                      f"{leg} leg " + ", ".join(f"{k} {v}" for k, v in a.initial_windows.for_leg(leg).items())
                      for leg in ("complex", "solvent"))
                  + f"; at most {a.max_windows.complex} (complex) / {a.max_windows.solvent} (solvent) windows.",
                  "The windows of mode fixed, for reference:"]
    else:
        lines.append("mode fixed: the windows below are used.")
    recommended = WindowsConfig()
    for leg in ("complex", "solvent"):
        total = 0
        for stage, spec in ls.windows.for_leg(leg).items():
            values = stage_values(spec, recommended.for_leg(leg)[stage])
            total += len(values)
            lines.append(f"  {leg:8s} {stage:10s} {len(values):2d}: " + " ".join(f"{v:g}" for v in values))
        lines.append(f"  {leg:8s} {'total':10s} {total - (len(ls.windows.for_leg(leg)) - 1):2d} (stage boundaries are shared)")
    r2 = cfg.sampling.rest2
    if r2.scale > 1.0:
        region = "the ligand" if r2.region == "ligand" and not r2.residues else (
            "the ligand and protein residues" + (f" within {r2.pocket_cutoff_nm:g} nm of it" if r2.region == "pocket" else "")
            + (f" ({', '.join(r2.residues)})" if r2.residues else ""))
        lines += ["# REST2", f"hot region: {region}; {r2.scale:g} times the temperature in the middle of each leg, "
                             "unscaled in the first and the last lambda window."]
    else:
        lines += ["# REST2", "off (sampling.rest2.scale is 1)."]
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
    failed = driver.failed_legs(summary)
    if failed:
        click.echo("failed: " + "; ".join(failed), err=True)
    if failed or not any("dg_bind" in s for s in summary):
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
    out = {k: v for k, v in result["analysis"].items() if k not in ("f_k", "df_k", "overlap_matrix", "overlap_used_matrix",
                                                                    "overlap_used_weighted_by_number")}
    out.update(n_equilibration_iterations=n_equilibration, n_iterations_stored=int(samples["u"].shape[0]))
    if "n_planned_iterations" in attrs:
        out["n_planned_iterations"] = int(attrs["n_planned_iterations"])
    # the radii of the tail correction that were really used: the outer one the samples recorded
    # their tail energies to, the inner one their interactions were cut off at
    out.update(lrc_long_cutoff_nm=result.get("lrc_long_cutoff_nm"), lrc_cutoff_nm=attrs.get("lrc_cutoff_nm"))
    out.update(delta_g_decouple=result["delta_g_decouple"], lrc=result["lrc"], lrc_method=result["lrc_method"],
               quality=result["quality"], notes=result["notes"], analysis_record=result["analysis_record"],
               analysis_settings_from=source)
    click.echo(jsonio.dumps(out, indent=1))


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--ligand", help="Ligand name (default: the first one in the SDF).")
@click.option("--device", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Override output.dir.")
def validate(config: str, ligand, device: int, output) -> None:
    """Compare the two backends' alchemical energies for the system in CONFIG."""
    from cufep.validate import validate as run_validation

    cfg = _apply_overrides(_load(config), None, None, None, output, None)
    try:
        ok = run_validation(cfg, ligand, device, click.echo)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    sys.exit(0 if ok else 1)


@main.group()
def benchmark() -> None:
    """Built-in benchmarks against experiment and FEP+ (TYK2 kinase, A2A receptor in a bilayer)."""


@benchmark.command("list")
def benchmark_list() -> None:
    """Show the benchmark sets."""
    from cufep.benchmarks.runner import load_registry, read_reference
    for name, entry in load_registry().items():
        click.echo(f"{name}: {entry['title']} ({len(read_reference(name))} ligands; subset: {', '.join(entry['subset'])})")


@benchmark.command("run")
@click.argument("name")
@click.option("--tier", type=click.Choice(["smoke", "subset", "full"]), default="smoke", show_default=True,
              help="smoke: one ligand, very short; subset: five ligands; full: every ligand.")
@click.option("-o", "--output", help="Output directory (default: runs/benchmark_<name>_<tier>).")
@click.option("--gpus", help="'auto' or a comma separated list of CUDA device numbers.")
@click.option("--repeats", type=int, help="Override the number of repeats.")
def benchmark_run(name: str, tier: str, output, gpus, repeats) -> None:
    """Run benchmark NAME (resumable)."""
    from cufep import run as driver
    from cufep.benchmarks.runner import benchmark_config, report

    try:
        cfg = _apply_overrides(benchmark_config(name, tier, output), repeats, None, gpus, None, None)
        summary = driver.run(cfg)
        click.echo(report(name, cfg.output.dir))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    failed = driver.failed_legs(summary)
    if failed:
        click.echo("failed: " + "; ".join(failed), err=True)
    if failed or not any("dg_bind" in s for s in summary):
        sys.exit(1)


@benchmark.command("report")
@click.argument("name")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def benchmark_report(name: str, directory: str) -> None:
    """Compare a benchmark run in DIRECTORY with experiment and FEP+."""
    from cufep.benchmarks.runner import report
    click.echo(report(name, directory))


@benchmark.command("freeze")
@click.argument("name")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--dest", help="Where to write the schedules (default: benchmarks/<name>/schedules).")
def benchmark_freeze(name: str, directory: str, dest) -> None:
    """Save the lambda schedules of a finished benchmark run for exact re-runs."""
    from cufep.benchmarks.runner import freeze_schedules
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
