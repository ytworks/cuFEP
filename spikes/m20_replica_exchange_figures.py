"""How the replicas of a leg moved between its lambda windows, for several runs: a figure or text.

    scripts/in_docker.sh python spikes/m20_replica_exchange_figures.py --ligand ejm_31 --leg complex \
        now_1=runs/tyk2_spread_1b770da_1 A1=runs/tyk2_windows_a_<commit>_1 --out spikes/out/exchange/complex.png
    scripts/in_docker.sh python spikes/m20_replica_exchange_figures.py --ligand ejm_31 --leg solvent \
        now_1=runs/tyk2_spread_1b770da_1 A1=runs/tyk2_windows_a_<commit>_1 --text

Every argument ``label=run`` is one run. For each of them, from the stored samples of the leg
(``hrex/``, the production only):

- the matrix of the moves of the replicas between lambda windows: the cell in row i and column j
  is the share of the iterations after which a replica that was in window i is in window j
  (every row adds up to 1). The exchange of an iteration tries many pairs of replicas, so a
  replica can move by more than one window in an iteration. Counted from the state of every
  replica in every iteration. The stored state of an iteration is the one it was sampled in,
  before its exchange, so the exchange of the last iteration is read from the checkpoint; where
  the checkpoint is not of the last stored iteration, that one exchange is not counted;
- the acceptance of the exchanges between the replicas of neighbouring windows, the only
  acceptances that a run records, and the number of round trips.

A leg that has a result gives the acceptances and the round trips of its result (a result of an
earlier version too: where the production begins is read from the samples). A leg that is
still being sampled is read up to its last checkpoint: its windows are those of ``schedule.yaml``
and the two numbers are computed by the functions of the analysis from the iterations stored so
far; the number of those iterations is shown. A leg without iterations of the production is named
and left out.

``--out``: a figure. The colour of a cell is on a logarithmic scale from 0.001 to 1, a move that
is rarer has the colour of 0.001, and a cell without a move has the colour of the surface; thin
lines mark where one stage ends and the next begins. ``--text``: the same matrices as characters
(' ' no move, '·' below 0.003, '░' below 0.03, '▒' below 0.1, '▓' below 0.3, '█' from 0.3), each
row with the first letter of its stage and the number of its window, and under them the
acceptances in percent. Nothing is written to the runs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

STAGES = {"complex": ["restrain", "discharge", "vanish"], "solvent": ["discharge", "vanish"]}
# one hue, light to dark (a move that is rare recedes towards the surface)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK_2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
LINE = "#256abf"
GLYPHS = ((0.3, "█"), (0.1, "▓"), (0.03, "▒"), (0.003, "░"))


def moves_of(run: Path, ligand: str, leg: str) -> dict | None:
    """The moves of the replicas of a leg in its production; None without iterations of it."""
    import yaml

    from cufep.analysis.mbar import count_round_trips, swap_acceptance
    from cufep.sampling.storage import StorageError, equilibration_boundary, read_samples
    from cufep.workflow import ligand_dir_name

    leg_dir = run / ligand_dir_name(ligand) / "repeat_0" / leg
    try:
        samples = read_samples(leg_dir / "hrex")
    except (StorageError, OSError):
        return None
    attrs, stored = samples["attrs"], np.asarray(samples["states"])
    # where the production begins: from the samples (a result of an earlier version does not say)
    first = equilibration_boundary(attrs, leg_dir)
    if first is None or len(stored) < first + 2:
        return None
    states, sampled = stored[first:], None
    if (leg_dir / "result.json").exists():
        result = json.loads((leg_dir / "result.json").read_text())
        windows = result.get("lambda_windows") or {}
        acceptance = result["analysis"].get("swap_acceptance_neighbour") or []
        round_trips = result["analysis"].get("round_trips")
    else:                                                  # being sampled: up to the last checkpoint
        windows = yaml.safe_load((leg_dir / "schedule.yaml").read_text())["windows"]
        acceptance = swap_acceptance(samples, first, len(states)) or []
        round_trips = count_round_trips(states)
        sampled = (len(states), int(attrs["n_planned_iterations"]) - first)
    after = after_the_last_exchange(leg_dir, len(stored))
    if after is not None:
        states = np.vstack([states, after[None, :]])
    k = len(samples["attrs"]["states"])
    counts = np.zeros((k, k))
    np.add.at(counts, (states[:-1].ravel(), states[1:].ravel()), 1.0)
    stages = [len(windows.get(stage, [])) for stage in STAGES[leg]]
    # neighbouring stages share a window: every stage but the first adds its windows but one
    tags = "".join(stage[0].upper() * (n if s == 0 else n - 1) for s, (stage, n) in enumerate(zip(STAGES[leg], stages)))
    visits = counts.sum(axis=1, keepdims=True)
    return {"matrix": counts / np.where(visits > 0, visits, 1.0), "k": k, "stages": stages, "tags": tags,
            "stage_ends": list(itertools.accumulate(n - 1 for n in stages[:-1])),
            "acceptance": [float("nan") if v is None else float(v) for v in acceptance],
            "round_trips": round_trips, "sampled": sampled}


def after_the_last_exchange(leg_dir: Path, n_stored: int) -> np.ndarray | None:
    """The windows of the replicas after the exchange of the last stored iteration: those of the
    checkpoint, if it is of that iteration; None otherwise."""
    try:
        with np.load(leg_dir / "hrex" / "checkpoint.npz") as checkpoint:
            if int(checkpoint["iteration"]) != n_stored:
                return None
            return np.asarray(checkpoint["state_of_replica"], dtype=int)
    except (OSError, KeyError, ValueError):
        return None


def headline(label: str, d: dict) -> str:
    text = f"{label}: {d['k']} windows ({' / '.join(str(s) for s in d['stages'])}), round trips {d['round_trips']}"
    if d["sampled"] is not None:
        text += f", being sampled ({d['sampled'][0]} of {d['sampled'][1]} iterations)"
    return text


def lowest_of(d: dict) -> tuple[int, float] | None:
    if not d["acceptance"] or np.all(np.isnan(d["acceptance"])):
        return None
    lowest = int(np.nanargmin(d["acceptance"]))
    return lowest, d["acceptance"][lowest]


def glyph(share: float) -> str:
    if share <= 0:
        return " "
    return next((character for level, character in GLYPHS if share >= level), "·")


def print_text(data: list[tuple[str, dict]], width: int) -> None:
    """The matrices as characters, side by side as far as ``width`` characters allow."""
    def rows_of(d: dict) -> list[str]:
        return [f"{d['tags'][i]}{i:2d} " + "".join(glyph(v) for v in d["matrix"][i]) for i in range(d["k"])]

    def flush(group: list[tuple[str, dict, list[str]]]) -> None:
        for label, d, _rows in group:
            lowest = lowest_of(d)
            print(headline(label, d) + ("" if lowest is None else
                                        f", lowest acceptance {lowest[1]:.2f} ({lowest[0]}-{lowest[0] + 1})"))
        for line in itertools.zip_longest(*([row.ljust(len(rows[0])) for row in rows] for _l, _d, rows in group)):
            print("    ".join(part if part is not None else " " * len(rows[0])
                              for part, (_l, _d, rows) in zip(line, group)).rstrip())
        for label, d, _rows in group:
            print(f"{label} acc% " + " ".join("-" if math.isnan(v) else f"{round(100 * v):d}" for v in d["acceptance"]))
        print()

    group: list[tuple[str, dict, list[str]]] = []
    for label, d in data:
        rows = rows_of(d)
        if group and sum(len(r[0]) + 4 for _l, _d, r in group) + len(rows[0]) > width:
            flush(group)
            group = []
        group.append((label, d, rows))
    if group:
        flush(group)


def write_figure(data: list[tuple[str, dict]], ligand: str, leg: str, columns: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import MaxNLocator

    cmap = LinearSegmentedColormap.from_list("one_hue", RAMP)
    cmap.set_bad(SURFACE)
    cmap.set_under(SURFACE)
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    columns = min(columns, len(data))
    rows = math.ceil(len(data) / columns)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": AXIS, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(3.6 * columns + 0.9, 4.5 * rows + 1.2))
    grid = fig.add_gridspec(2 * rows, columns + 1, height_ratios=[4.0, 1.0] * rows,
                            width_ratios=[1.0] * columns + [0.04], hspace=0.30, wspace=0.22,
                            left=0.05, right=0.95, top=1.0 - 1.0 / (4.5 * rows + 1.2), bottom=0.06)
    image = None
    for n, (label, d) in enumerate(data):
        r, c = divmod(n, columns)
        ax = fig.add_subplot(grid[2 * r, c])
        # a move below the scale keeps the colour of its lower end: only no move is the surface
        matrix = np.where(d["matrix"] > 0, np.maximum(d["matrix"], norm.vmin), np.nan)
        # aspect auto: the matrix is as wide as the acceptances under it, so that the windows line up
        image = ax.imshow(matrix, cmap=cmap, norm=norm, origin="upper", interpolation="nearest", aspect="auto",
                          extent=(-0.5, d["k"] - 0.5, d["k"] - 0.5, -0.5))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        for end in d["stage_ends"]:
            ax.axhline(end, color=INK_2, lw=0.6)
            ax.axvline(end, color=INK_2, lw=0.6)
        title = headline(label, d)
        if d["sampled"] is not None:                       # two lines: the title is as wide as the panel
            title = title.replace(", being sampled", ",\nbeing sampled")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylabel("window before" if c == 0 else "")
        ax.tick_params(length=2, labelsize=8, labelbottom=False)
        low = fig.add_subplot(grid[2 * r + 1, c], sharex=ax)
        lowest = lowest_of(d)
        if lowest is not None:
            x = np.arange(len(d["acceptance"])) + 0.5
            low.plot(x, d["acceptance"], color=LINE, lw=2.0, solid_capstyle="round")
            low.plot([x[lowest[0]]], [lowest[1]], "o", ms=5, color=LINE, mec=SURFACE, mew=1.5)
            low.text(0.99, 0.97, f"lowest {lowest[1]:.2f} (windows {lowest[0]} and {lowest[0] + 1}, marked)",
                     transform=low.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
        for end in d["stage_ends"]:
            low.axvline(end, color=INK_2, lw=0.6)
        low.set_xlim(-0.5, d["k"] - 0.5)
        low.set_ylim(0.0, 1.25)
        low.set_yticks([0.0, 0.5, 1.0])
        low.grid(axis="y", color=AXIS, lw=0.5, alpha=0.6)
        if n + columns >= len(data):                      # the last panel of its column
            low.set_xlabel("window after (matrix); neighbouring windows (acceptance)", fontsize=8)
        low.set_ylabel("acceptance" if c == 0 else "")
        low.tick_params(length=2, labelsize=8)
        for side in ("top", "right"):
            low.spines[side].set_visible(False)
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, columns]))
    bar.set_label("share of the iterations (logarithmic)", fontsize=8, color=INK_2)
    bar.outline.set_edgecolor(AXIS)
    fig.suptitle(f"{ligand}, {leg} leg. Matrix: share of the iterations after which a replica of the "
                 "row's window is in the column's window.\nBelow each matrix: acceptance of the exchanges between "
                 "neighbouring windows. Lines: ends of the stages.", fontsize=10, color=INK, x=0.05, ha="left",
                 y=1.0 - 0.12 / (4.5 * rows + 1.2))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"written: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=run")
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--leg", default="complex", choices=list(STAGES))
    ap.add_argument("--columns", type=int, default=4, help="panels in a row of the figure")
    ap.add_argument("--out", help="the figure to write")
    ap.add_argument("--text", action="store_true", help="print the matrices as characters")
    ap.add_argument("--width", type=int, default=100, help="characters in a line of the text")
    args = ap.parse_args()
    if not args.out and not args.text:
        ap.error("give --out, --text or both")
    data = []
    for item in args.runs:
        label, _sep, run = item.partition("=")
        moves = moves_of(Path(run), args.ligand, args.leg)
        if moves is None:
            print(f"{label}: the {args.leg} leg has no iterations of the production stored ({run})")
        else:
            data.append((label, moves))
    if not data:
        return
    if args.text:
        print_text(data, args.width)
    if args.out:
        write_figure(data, args.ligand, args.leg, args.columns, Path(args.out))


if __name__ == "__main__":
    main()
