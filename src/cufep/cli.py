"""Command line interface (placeholder; filled in during M1)."""
import click

from cufep import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """cuFEP: absolute binding free energy calculations on OpenMM."""


if __name__ == "__main__":
    main()
