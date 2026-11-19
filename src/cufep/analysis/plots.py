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
    legs = []
    for leg in ("complex", "solvent"):
        f = repeat_dir / leg / "result.json"
        if f.exists():
            legs.append((leg, json.loads(f.read_text())))
    if not legs:
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
        final, error = a["delta_g"] / KCAL, a["delta_g_error"] / KCAL
        ax = axes[row][0]
        conv = quality.get("halves") or {}
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
            if half is not None and np.isfinite(half[0]):
                ax.plot([x], [half[0] / KCAL], marker="D", ms=8, mfc="none", color=colour)
        # consecutive blocks that do not overlap: a free energy that keeps moving shows as a staircase
        blocks = [b for b in a.get("blocks", []) if np.isfinite(b[2])]
        for k, (start, end, value, _err) in enumerate(blocks):
            ax.plot([start, end], [value / KCAL, value / KCAL], color="0.25", lw=2.5, solid_capstyle="butt",
                    label="consecutive blocks" if k == 0 else None)
        verdict = ""
        if conv.get("difference_kT") is not None:
            verdict = f"halves differ by {conv['difference_kT']:.1f} kT"
        if conv.get("reason"):
            verdict = "halves not compared"
            ax.text(0.5, 0.04, conv["reason"], transform=ax.transAxes, ha="center", va="bottom", fontsize=6, wrap=True)
        samples = quality.get("samples") or {}
        if samples.get("fewest") is not None:
            verdict += (", " if verdict else "") + f"fewest independent samples {samples['fewest']:.0f} (window {samples['window']})"
        ax.set_title(f"{leg} leg: {final:.2f} ± {error:.2f} kcal/mol" + (f"\n{verdict}" if verdict else ""), fontsize=9)
        ax.set_xlabel("fraction of the production (diamonds: its two halves)")
        ax.set_ylabel("free energy of the leg (kcal/mol)")
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
        ax.set_title(f"overlap matrix, lowest overlap of neighbours {a['min_overlap']:.3f}", fontsize=10)
    fig.suptitle(f"{legs[0][1].get('ligand', '')}  (repeat {legs[0][1].get('repeat', 0)})", fontsize=11)
    fig.tight_layout()
    out = Path(path) if path is not None else repeat_dir / "analysis.png"
    fig.savefig(out)
    plt.close(fig)
    return out
