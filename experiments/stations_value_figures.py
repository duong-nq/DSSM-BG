"""Figures 4-8: VALUE indices at the nine stations, from stations_indices.xlsx (run stations_indices.py first)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import paths
import stations_indices as si
from stations_summary import STATIONS

# colours of the other figures; position and marker also tell the models apart
COLOR = {"GLM-BG": "#6F8FC6", "RF-BG": "#E39A6B", "MLP-BG": "#6DB77F", "DSSM": "#C8102E"}
MARKER = {"GLM-BG": "s", "RF-BG": "^", "MLP-BG": "D", "DSSM": "o"}
SHORT = {"GLM-BG": "GLM", "RF-BG": "RF", "MLP-BG": "MLP", "DSSM": "DSSM"}
LABEL = {"GLM-BG": "GLM–BG", "RF-BG": "RF–BG", "MLP-BG": "MLP–BG", "DSSM": "DSSM–BG"}
TITLE = {"Annual": "Whole period", "DJF": "Winter (DJF)", "MAM": "Spring (MAM)", "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}
NAMES = {"WetSpell": "Wet spell", "DrySpell": "Dry spell"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def seasons_figure(bias, path):
    """Median relative bias at the nine stations: one row per index, one column per season, one box per model."""
    fig, axes = plt.subplots(len(si.SUMMARY), len(si.SEASONS), figsize=(7.0, 9.4), sharey="row")
    for r, k in enumerate(si.SUMMARY):
        for c, season in enumerate(si.SEASONS):
            ax = axes[r, c]
            g = bias[bias.group == season].pivot_table(index="station", columns="model", values=k).reindex(list(STATIONS))
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            parts = ax.boxplot([g[m].to_numpy() for m in si.MODELS], positions=range(len(si.MODELS)), widths=0.6,
                               whis=1.5, patch_artist=True, zorder=2, medianprops={"color": INK, "lw": 1.0})
            for j, model in enumerate(si.MODELS):
                parts["boxes"][j].set(facecolor=COLOR[model], edgecolor=COLOR[model], linewidth=0.6)
                for line in parts["whiskers"][2 * j:2 * j + 2] + parts["caps"][2 * j:2 * j + 2]:
                    line.set(color=COLOR[model], linewidth=0.9)
                parts["fliers"][j].set(marker=MARKER[model], markersize=3.5, markerfacecolor=COLOR[model],
                                       markeredgecolor="white", markeredgewidth=0.4)
            ax.set_xticks(range(len(si.MODELS)), [SHORT[m] for m in si.MODELS] if r == len(si.SUMMARY) - 1 else [],
                          rotation=90)
            ax.set_xlim(-0.6, len(si.MODELS) - 0.4)
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
            ax.tick_params(length=2)
            if r == 0:
                ax.set_title(TITLE[season], fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f"{NAMES.get(k, k)} (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[m]) for m in si.MODELS]
    fig.legend(handles, [LABEL[m] for m in si.MODELS], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=0.6, h_pad=0.5)
    fig.savefig(path)
    plt.close(fig)


def members_figure(bias, season, path):
    """Relative bias of the 1000 members at each station in one season; the KS row marks the 5% critical value."""
    rows = si.SUMMARY + ["KS"]
    group = bias[bias.group == season].set_index(["station", "model"])
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.0, 9.6), sharex=True)
    width = 0.18
    for ax, k in zip(axes, rows):
        if k != "KS":
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        for s, station in enumerate(STATIONS):
            for j, model in enumerate(si.MODELS):
                row = group.loc[(station, model)]
                q05, q25, q50, q75, q95 = (row[f"{k}_q05"], row[f"{k}_q25"], row[k], row[f"{k}_q75"], row[f"{k}_q95"])
                x = s + (j - 1.5) * (width + 0.03)
                ax.vlines(x, q05, q95, color=COLOR[model], lw=0.9, zorder=2)
                ax.add_patch(plt.Rectangle((x - width / 2, q25), width, q75 - q25, facecolor=COLOR[model],
                                           edgecolor="white", linewidth=0.5, zorder=3))
                ax.hlines(q50, x - width / 2, x + width / 2, color=INK, lw=0.8, zorder=4)
            if k == "KS":
                ax.hlines(group.loc[station].loc[si.MODELS].KS_crit.mean(), s - 0.45, s + 0.45, color=INK, lw=1.2, zorder=5)
        ax.set_ylabel("KS" if k == "KS" else f"{NAMES.get(k, k)} (%)")
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.tick_params(length=2)
        ax.set_xlim(-0.6, len(STATIONS) - 0.4)
    axes[-1].set_xticks(range(len(STATIONS)), [v[0] for v in STATIONS.values()], rotation=30, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[m]) for m in si.MODELS]
    fig.legend(handles, [LABEL[m] for m in si.MODELS], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.56, 1.0))
    fig.text(0.01, 0.993, TITLE[season], ha="left", va="top", fontsize=9, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def main():
    bias = pd.read_excel(si.OUT / "stations_indices.xlsx", sheet_name="indices")
    bias = bias[bias.model.isin(si.MODELS)]
    seasons_figure(bias, paths.FIGURES / "value_stations_seasons.pdf")
    for season in si.SEASONS:
        members_figure(bias, season, paths.FIGURES / f"value_members_{season}.pdf")
    print("written:", paths.FIGURES / "value_stations_seasons.pdf", *(f"value_members_{s}.pdf" for s in si.SEASONS))


if __name__ == "__main__":
    main()
