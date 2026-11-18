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
    out.write_text(text)
    click.echo(f"wrote {out}")


def _load(config: str):
    from cufep.config.schema import load_config
    try:
        return load_config(config)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"{config}: {exc}") from exc


def _apply_overrides(cfg, repeats, production_ns, gpus, output, backend):
    if repeats is not None:
        cfg.sampling.repeats = repeats
    if production_ns:
        cfg.sampling.production_ns = sorted(float(x) for x in production_ns.split(","))
    if gpus:
        cfg.resources.gpus = "auto" if gpus == "auto" else [int(x) for x in gpus.split(",")]
    if output:
        cfg.output.dir = Path(output)
    if backend:
        cfg.backend.name = backend
    return cfg


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
    click.echo("\n# resolved configuration\n" + dump_config(resolved))


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--repeats", type=int, help="Override sampling.repeats.")
@click.option("--production-ns", help="Override sampling.production_ns, e.g. '5,10'.")
@click.option("--gpus", help="Override resources.gpus: 'auto' or a comma separated list of device indices.")
@click.option("--backend", type=click.Choice(["cupotential", "openmm"]), help="Override backend.name.")
@click.option("-o", "--output", help="Override output.dir.")
@click.option("--reschedule", is_flag=True, help="Search the lambda schedule again instead of reusing schedule.yaml.")
def run(config: str, repeats, production_ns, gpus, backend, output, reschedule: bool) -> None:
    """Run (or resume) the calculation described by CONFIG."""
    from cufep import run as driver

    cfg = _apply_overrides(_load(config), repeats, production_ns, gpus, output, backend)
    try:
        summary = driver.run(cfg, reschedule=reschedule)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    if not any("dg_bind" in s for s in summary):
        sys.exit(1)


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def analyze(directory: str) -> None:
    """Re-analyse a finished or running leg directory (the one containing hrex/)."""
    from cufep.analysis.mbar import analyze_leg
    from cufep.sampling.storage import read_samples

    d = Path(directory)
    hrex = d / "hrex" if (d / "hrex").exists() else d
    samples = read_samples(hrex)
    temperature = float(samples["attrs"].get("temperature_K", 298.15))
    res = analyze_leg(samples, temperature)
    click.echo(json.dumps({k: v for k, v in res.to_dict().items() if k not in ("f_k", "df_k")}, indent=1))


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
              help="smoke: one ligand, very short; subset: five ligands; full: every ligand (both with 3 repeats).")
@click.option("-o", "--output", help="Output directory (default: runs/benchmark_<name>_<tier>).")
@click.option("--gpus", help="'auto' or a comma separated list of device indices.")
@click.option("--repeats", type=int, help="Override the number of repeats.")
def benchmark_run(name: str, tier: str, output, gpus, repeats) -> None:
    """Run benchmark NAME (resumable)."""
    from cufep import run as driver
    from cufep.benchmarks.runner import benchmark_config, report

    try:
        cfg = _apply_overrides(benchmark_config(name, tier, output), repeats, None, gpus, None, None)
        driver.run(cfg)
        click.echo(report(name, cfg.output.dir))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc


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
