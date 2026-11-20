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
        shown = [final - error, final + error] + [p[1] / KCAL for name in ("forward", "reverse") for p in a.get(name, [])]
        shown += [h[0] / KCAL for h in (a.get("first_half"), a.get("second_half")) if h is not None and h[0] is not None]
        shown += [b[2] / KCAL for b in blocks]
        shown = [v for v in shown if v is not None and np.isfinite(v)]
        low, high = min(shown), max(shown)
        margin = max(0.5, 0.15 * (high - low))
        ax.set_ylim(low - margin, high + margin)
        # the forward and reverse estimates, the halves and the blocks are shown as numbers, without a verdict
        # (as FEP+ and OpenFE show them); the verdicts are the two checks of quality_checks
        verdict = ""
        samples = quality.get("samples") or {}
        if samples.get("fewest") is not None:
            verdict += (", " if verdict else "") + f"fewest independent samples {samples['fewest']:.0f} (window {samples['window']})"
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
            ax.set_xlabel("lambda window"); ax.set_ylabel("lambda window")
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
