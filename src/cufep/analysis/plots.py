"""Figures of the analysis of a finished calculation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KCAL = 4.184


def plot_repeat(repeat_dir: str | Path, path: str | Path | None = None) -> Path | None:
    """Convergence and overlap of the legs of one repeat, written to ``analysis.png``.

    One row per finished leg. Left: the free energy of the leg from growing parts of the
    production, counted from its start (forward) and from its end (reverse), with the two
    halves of the production and a band of the tolerance of the convergence check around the
    value from all of it. Right: the overlap matrix of the lambda windows. Returns the path, or
    None if no leg has a result or matplotlib is missing.
    """
    repeat_dir = Path(repeat_dir)
    out = Path(path) if path is not None else repeat_dir / "analysis.png"
    legs = []
    for leg in ("complex", "solvent"):
        f = repeat_dir / leg / "result.json"
        if f.exists():
            legs.append((leg, json.loads(f.read_text())))
    if not legs:
        # no leg of this repeat has a result now: a figure of an earlier state would be read as the
        # current one (review of PR 1)
        out.unlink(missing_ok=True)
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker  # noqa: F401
    except Exception:  # noqa: BLE001
        return None

    fig, axes = plt.subplots(len(legs), 2, figsize=(10.0, 4.0 * len(legs)), dpi=130, squeeze=False)
    for row, (leg, result) in enumerate(legs):
        a = result["analysis"]
        quality = result.get("quality") or {}
        if a.get("delta_g") is None:                  # the analysis gave no free energy for this leg
            axes[row][0].set_title(f"{leg} leg: no free energy", fontsize=9)
            axes[row][1].set_axis_off()
            continue
        final, error = a["delta_g"] / KCAL, (a.get("delta_g_error") or 0.0) / KCAL
        # since version 5 of the analysis every value is of the free energy with the tail correction;
        # before, of MBAR alone, and the figure says so
        version = (result.get("analysis_record") or {}).get("version")
        corrected = version is not None and version >= 5
        ax = axes[row][0]
        ax.axhspan(final - error, final + error, color="0.88", zorder=0,
                   label=f"all of the production ± {error:.2f} (standard error)")
        ax.axhline(final, color="0.4", lw=1)
        for name, colour, marker in (("forward", "#1f6f8b", "o"), ("reverse", "#c0392b", "s")):
            series = np.array(a.get(name, []), dtype=float)
            if series.size:
                ax.errorbar(series[:, 0], series[:, 1] / KCAL, yerr=series[:, 2] / KCAL, color=colour, marker=marker,
                            ms=4, lw=1, capsize=2, label=name)
        for name, colour, x in (("first_half", "#1f6f8b", 0.48), ("second_half", "#c0392b", 0.52)):
            half = a.get(name)
            if half is not None and half[0] is not None and np.isfinite(half[0]):
                ax.plot([x], [half[0] / KCAL], marker="D", ms=8, mfc="none", color=colour)
        # consecutive blocks that do not overlap: a free energy that keeps moving shows as a staircase
        blocks = [b for b in a.get("blocks", []) if b[2] is not None and np.isfinite(b[2])]
        for k, (start, end, value, _err) in enumerate(blocks):
            ax.plot([start, end], [value / KCAL, value / KCAL], color="0.25", lw=2.5, solid_capstyle="butt",
                    label="consecutive blocks" if k == 0 else None)
        # the range of the values, the whole and its band: an error bar that reaches further is cut by the
        # axes (review of PR 1: a part with an error of thousands of kcal/mol flattened every other value)
        shown = [final - error, final + error] + [
            p[1] / KCAL for name in ("forward", "reverse") for p in a.get(name, [])
        ]
        shown += [
            h[0] / KCAL for h in (a.get("first_half"), a.get("second_half")) if h is not None and h[0] is not None
        ]
        shown += [b[2] / KCAL for b in blocks]
        shown = [v for v in shown if v is not None and np.isfinite(v)]
        low, high = min(shown), max(shown)
        margin = max(0.5, 0.15 * (high - low))
        ax.set_ylim(low - margin, high + margin)
        # the forward and reverse estimates, the halves and the blocks are shown as numbers, without a verdict
        # (OpenFE shows its forward and reverse series without one); the verdicts are the two checks of quality_checks
        verdict = ""
        samples = quality.get("samples") or {}
        if samples.get("fewest") is not None:
            verdict += (
                ", " if verdict else ""
            ) + f"fewest independent samples {samples['fewest']:.0f} (window {samples['window']})"
        what = "" if corrected else f" (MBAR without the tail correction, analysis version {version or 'not recorded'})"
        ax.set_title(f"{leg} leg: {final:.2f} ± {error:.2f} kcal/mol{what}" + (f"\n{verdict}" if verdict else ""),
                     fontsize=9)
        ax.set_xlabel("fraction of the production (diamonds: its two halves)")
        ax.set_ylabel("free energy of decoupling, with the tail correction (kcal/mol)" if corrected
                      else "free energy of the leg by MBAR, without the tail correction (kcal/mol)")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[row][1]
        matrix = np.array(a.get("overlap_matrix", []), dtype=float)
        if matrix.size:
            image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(0.3, float(matrix.max())))
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            n = matrix.shape[0]
            if n <= 24:
                for k in range(n - 1):
                    low = matrix[k, k + 1] < (quality.get("overlap") or {}).get("threshold", 0.03)
                    ax.text(k + 1, k, f"{matrix[k, k + 1]:.2f}", ha="center", va="center", fontsize=5,
                            color="#c0392b" if low else "black")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel("lambda window")
            ax.set_ylabel("lambda window")
        used = a.get("min_overlap_used")
        ax.set_title(f"overlap matrix, lowest overlap of neighbours {a['min_overlap']:.3f}"
                     + ("" if used is None else f"\n(among the samples of the estimate {used:.3f})"), fontsize=9)
    fig.suptitle(f"{legs[0][1].get('ligand', '')}  (repeat {legs[0][1].get('repeat', 0)})", fontsize=11)
    fig.tight_layout()
    out = Path(path) if path is not None else repeat_dir / "analysis.png"
    from cufep.files import publish_with

    publish_with(out, lambda fh: fig.savefig(fh, format=Path(out).suffix.lstrip(".") or "png"), mode="wb")
    plt.close(fig)
    return out


#: the colours of the figure of the interactions: one hue for the data, grey for what is context,
#: ink for text and for the reference values
_SURFACE, _INK, _INK_2, _MUTED, _GRID, _AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
_DATA, _CONTEXT = "#2a78d6", "#c3c2b7"
#: the rows of interactions the figure shows at most (the most frequent ones; the table holds all)
_ROWS_SHOWN = 12


def _is_of_the_restraint(report_file: Path, restraint_file: Path) -> bool:
    """Whether ``interactions.json`` is the analysis the restraint of the leg was chosen from: the leg has a
    restraint of ``restraints.selection: interactions`` with the anchors the analysis chose. A run that was
    stopped between the two files and went on with another selection leaves the analysis of a restraint that
    was never used (review of eb6f89f)."""
    try:
        stored = json.loads(restraint_file.read_text())
        report = json.loads(report_file.read_text())
        chosen = report["chosen"]
        # the same anchors can be chosen from two analyses (of another start): the record of the start tells them
        # apart, and so do the reference values, which the restraint has from this analysis
        reference = [report["reference_values"][name] for name in ("r", "theta_A", "theta_B")]
        of_restraint = [stored["restraint"]["r0"], *np.degrees([stored["restraint"]["theta_a0"],
                                                                stored["restraint"]["theta_b0"]])]
        return (stored["info"].get("selection") == "interactions"
                and (report.get("start") is None or stored["info"].get("start") == report.get("start"))
                and bool(np.allclose(reference, of_restraint, atol=1e-6))
                and [int(i) for i in stored["restraint"]["host_atoms"]] == [int(i) for i in chosen["host_atoms"]]
                and [int(i) for i in stored["restraint"]["guest_atoms"]] == [int(i) for i in chosen["guest_atoms"]])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plot_interactions(leg_dir: str | Path, path: str | Path | None = None) -> Path | None:
    """What the restraint of a complex leg was chosen from with ``restraints.selection: interactions``,
    from ``interactions.json`` of the leg, written to ``interactions.png`` next to it.

    Top left: in what fraction of the analysed frames each interaction (ligand atom, residue, type)
    is there, with the fraction from which an interaction is kept; the interactions of the ligand
    atom and residue that the anchors were chosen from are in colour, the others in grey. Top
    right: the frames in which each of them is there, trajectory after trajectory. Below: the six
    coordinates of the chosen anchors over the frames that give the reference values, in the bins
    of the rule, with the reference value of each. Numbers of the analysis, without a verdict.
    Returns the path, or None if the leg has no such analysis of the restraint it has
    (``restraint.json``) or matplotlib is missing.
    """
    leg_dir = Path(leg_dir)
    out = Path(path) if path is not None else leg_dir / "interactions.png"
    source = leg_dir / "interactions.json"
    if not source.exists() or not _is_of_the_restraint(source, leg_dir / "restraint.json"):
        out.unlink(missing_ok=True)                  # a figure of an earlier state would be read as the current one
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:  # noqa: BLE001
        return None
    report = json.loads(source.read_text())
    n = int(report["frames_analysed"])
    chosen = report["chosen"]
    rows = report["interactions"][:_ROWS_SHOWN]
    basis = [row["ligand_atom"] == chosen["guest_atoms"][0] and row["residue"] == chosen["residue_index"]
             and report["method"] == "interactions" for row in rows]
    colours = [_DATA if b else _CONTEXT for b in basis]
    labels = [f"{row['ligand_atom_name'].split(':')[-1]} - {row['residue_name']}  {row['interaction']}" for row in rows]

    fig = plt.figure(figsize=(11.5, 8.4), dpi=130, facecolor=_SURFACE)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.55,
                            left=0.215, right=0.975, top=0.86, bottom=0.085)
    bars, code = fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:])

    def quiet(ax, grid_axis=None):
        ax.set_facecolor(_SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_AXIS)
        ax.tick_params(colors=_INK_2, labelsize=8, length=0)
        if grid_axis:
            ax.grid(axis=grid_axis, color=_GRID, linewidth=0.8)
            ax.set_axisbelow(True)

    # ---- how often each interaction is there
    y = np.arange(len(rows))
    bars.barh(y, [row["fraction"] for row in rows], height=0.56, color=colours, edgecolor="none")
    needed = report["frames_needed"] / n if n else 0.0
    bars.axvline(needed, color=_INK_2, linewidth=1.0)
    bars.text(needed + (0.015 if needed < 0.5 else -0.015), -0.85,
              f"kept from {report['frames_needed']} of {n} frames", color=_INK_2, fontsize=7.5,
              ha="left" if needed < 0.5 else "right", va="center")
    for k, row in enumerate(rows):
        if basis[k]:
            # the value at the tip of the bar; inside it, in white, where there is no room beyond the tip
            outside = row["fraction"] < 0.86
            bars.text(row["fraction"] + (0.02 if outside else -0.02), k, f"{row['fraction']:.2f}",
                      color=_INK if outside else "#ffffff", fontsize=8, va="center", ha="left" if outside else "right")
    bars.set_yticks(y)
    bars.set_yticklabels(labels, fontsize=8, color=_INK)
    bars.set_ylim(_ROWS_SHOWN - 0.4, -1.3)             # the same slots whatever the number of rows: thin bars
    bars.set_xlim(0.0, 1.0)
    bars.set_xlabel("fraction of the analysed frames", fontsize=8, color=_INK_2, loc="left")
    bars.set_title("interactions (ligand atom - residue, type)", fontsize=9, color=_INK, loc="left")
    quiet(bars, "x")
    if any(basis):
        bars.legend(handles=[Patch(color=_DATA, label="of the anchors' atom and residue"),
                             Patch(color=_CONTEXT, label="other interactions")],
                    frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.02, -0.13), ncols=1,
                    labelcolor=_INK_2, handlelength=1.2, borderaxespad=0.0)

    # ---- the frames in which each is there, trajectory after trajectory
    of_frame = np.asarray(report.get("trajectory_of_frame") or [0] * n, dtype=int)
    for k, row in enumerate(rows):
        code.vlines(row["in_frames"], k - 0.3, k + 0.3, color=colours[k], linewidth=0.9)
    starts = [0, *[int(f) for f in np.nonzero(np.diff(of_frame))[0] + 1]]
    for start, trajectory in zip(starts, of_frame[starts] if n else []):
        if start:
            code.axvline(start - 0.5, color=_AXIS, linewidth=0.8)
        code.text(start + 0.01 * n, -0.85, f"trajectory {int(trajectory)}", color=_INK_2, fontsize=7.5, va="center")
    code.set_xlim(-0.5, max(n, 1) - 0.5)
    code.set_ylim(_ROWS_SHOWN - 0.4, -1.3)
    code.set_yticks([])
    code.set_xlabel("analysed frame", fontsize=8, color=_INK_2)
    code.set_title("the frames in which each interaction is there", fontsize=9, color=_INK, loc="left")
    quiet(code)

    # ---- the six coordinates of the chosen anchors, in the bins of the rule
    series = {name: np.asarray(values, dtype=float) for name, values in report["coordinates"].items()}
    low, high = 45.0, 135.0
    made_with = (report.get("made_with") or {}).get("interactions") or {}
    if made_with.get("angle_range_deg"):
        low, high = (float(x) for x in made_with["angle_range_deg"])
    inside = (series["theta_A"] >= low) & (series["theta_A"] <= high) \
        & (series["theta_B"] >= low) & (series["theta_B"] <= high)
    spans = {"r": None, "theta_A": (0.0, 180.0), "theta_B": (0.0, 180.0), "phi_A": (-180.0, 180.0),
             "phi_B": (-180.0, 180.0), "phi_C": (-180.0, 180.0)}
    for k, (name, span) in enumerate(spans.items()):
        ax = fig.add_subplot(grid[1, k])
        values = series[name][inside] if inside.any() else series[name]
        counts, edges = np.histogram(values, bins=20, range=span)
        ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=0.86 * (edges[1] - edges[0]), color=_DATA,
               edgecolor="none")
        reference = float(report["reference_values"][name])
        ax.axvline(reference, color=_INK, linewidth=1.2)
        unit = "nm" if name == "r" else "deg"
        ax.set_title(f"{name}\nreference {reference:.3f} {unit}" if name == "r"
                     else f"{name}\nreference {reference:.1f} {unit}", fontsize=8, color=_INK, loc="left")
        if name in ("theta_A", "theta_B"):
            for bound in (low, high):
                ax.axvline(bound, color=_AXIS, linewidth=0.8)
        if span is not None:
            ax.set_xlim(*span)
            ax.set_xticks(np.linspace(span[0], span[1], 3))
        ax.set_xlabel(unit, fontsize=7.5, color=_INK_2)
        if k == 0:
            ax.set_ylabel("frames", fontsize=8, color=_INK_2)
        quiet(ax, "y")
        ax.tick_params(labelsize=7)
    how = ("chosen from an interaction" if report["method"] == "interactions"
           else "chosen by the distance method (no interaction gave anchors)")
    fig.suptitle(f"restraint {how}: host {', '.join(chosen['host'])} | guest {', '.join(chosen['guest'])}",
                 fontsize=10, color=_INK, x=0.02, ha="left", y=0.965)
    fig.text(0.02, 0.925, f"{n} frames of {report['md']['trajectories']} unrestrained trajectories of the complex; "
             f"below: the coordinates of these anchors over the {int(inside.sum())} frames with both anchor angles "
             f"within {low:g} to {high:g} degrees", fontsize=8, color=_INK_2, ha="left")
    from cufep.files import publish_with

    publish_with(out, lambda fh: fig.savefig(fh, format=Path(out).suffix.lstrip(".") or "png", facecolor=_SURFACE),
                 mode="wb")
    plt.close(fig)
    return out
