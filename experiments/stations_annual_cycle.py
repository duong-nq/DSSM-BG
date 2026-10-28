"""Figure 9: monthly mean precipitation at the nine stations and the mean difference Delta mu (Keisler et al., 2026,
Eq. A.1) between the ensemble means and the observations, for the models trained at the Annual scale."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import make_tables_8vars as tables
import paths
import stations_indices as si
from stations_summary import STATIONS, precipitation
from stations_value_figures import LABEL, MARKER

CLIMATOLOGY = "1980-2017"                  # observed means of 1980-2017, scored like a model
PERIODS = tables.MONTHS + ["All"]
PANEL, STRIP, AXIS_TEXT, TICK = "#ebebeb", "#d9d9d9", "#4d4d4d", "#333333"     # ggplot2 theme_gray
# hues of the other figures, stepped in lightness to stay distinct under colour-vision deficiencies
PALETTE = {"GLM-BG": "#2a78d6", "RF-BG": "#dd5b25", "MLP-BG": "#149769", "DSSM": "#ae0f27"}
DASHED = {"color": "#666666", "lw": 1.0, "ls": (0, (4, 2)), "zorder": 2}
OBSERVED = {"color": "#1a1a1a", "lw": 1.7, "marker": "o", "ms": 3.6, "markeredgecolor": "white", "markeredgewidth": 0.5,
            "zorder": 4}


def model_style(model):
    return {"color": PALETTE[model], "lw": 1.3, "marker": MARKER[model], "ms": 3.6, "markeredgecolor": "white",
            "markeredgewidth": 0.5, "zorder": 3}


def ggplot_panel(ax, title, values):
    """ggplot2 look of one panel, with the y axis fitted to the values."""
    low, high = np.nanmin(values), np.nanmax(values)
    ax.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xticks(np.arange(1, 13), [m[0] for m in tables.MONTHS])
    ax.set_xlim(0.5, 12.5)
    ax.set_facecolor(PANEL)
    ax.grid(which="major", color="white", lw=0.9)
    ax.grid(which="minor", axis="y", color="white", lw=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(which="major", length=2.5, width=0.6, color=TICK, labelcolor=AXIS_TEXT)
    ax.tick_params(which="minor", length=0)
    strip = make_axes_locatable(ax).append_axes("top", size=0.17, pad=0)
    strip.set_facecolor(STRIP)
    strip.set_xticks([])
    strip.set_yticks([])
    strip.text(0.5, 0.5, title, ha="center", va="center", color="#1a1a1a", transform=strip.transAxes)
    for spine in [*ax.spines.values(), *strip.spines.values()]:
        spine.set_visible(False)


def means():
    """Mean precipitation (mm/day) of each month and of the test period (All): observed, 1980-2017 and ensemble means."""
    rows = []
    for station in STATIONS:
        series = precipitation(station)
        series = series.where(series > si.TAU, 0.0)
        test = series.loc["2018-01-01":"2025-12-31"]
        frames = {"observed": test, CLIMATOLOGY: series.loc[:"2017-12-31"]}
        for model in si.MODELS:
            e = si.ensemble(station, model)
            assert (pd.DatetimeIndex(e["dates"]) == test.index).all()
            members = e["rainfall"].astype(np.float64)
            frames[model] = pd.Series(np.where(members > si.TAU, members, 0.0).mean(1), index=test.index)
        for name, x in frames.items():
            monthly = x.groupby(x.index.month).mean().to_numpy()
            rows.append({"station": station, "series": name, **dict(zip(tables.MONTHS, monthly)), "All": x.mean()})
    return pd.DataFrame(rows).set_index(["station", "series"])[PERIODS]


def delta_mu(table):
    """Signed differences from the observed means at every station, and Delta mu over the nine stations."""
    scored = table.drop(index="observed", level="series")
    difference = scored - table.xs("observed", level="series").reindex(scored.index, level="station")
    result = difference.abs().groupby(level="series").mean()
    result.insert(len(tables.MONTHS), "Avg.", result[tables.MONTHS].mean(axis=1))
    return difference, result.loc[si.MODELS + [CLIMATOLOGY]]


def saved_delta_mu(station, model):
    """Delta_mu stored by the run of one model trained at the Annual scale."""
    key, values = f"{tables.PREFIX[model]}_Annual", {}
    for name in ("monthly", "overall"):
        saved = pd.read_excel(tables.runs_0606(station)[model] / f"point_metrics_{name}.xlsx")
        saved = saved[saved["variant" if "variant" in saved else "model"] == key].drop_duplicates("group")
        values.update(saved.set_index("group").Delta_mu.to_dict())
    return pd.Series(values).rename({"Overall": "All"})[PERIODS]


def cycle_figure(table, result, path):
    """Delta mu by month (top) and the monthly means at each station."""
    months = np.arange(1, 13)
    fig = plt.figure(figsize=(7.0, 8.6))
    grid = fig.add_gridspec(4, 3, height_ratios=(1.2, 1, 1, 1), left=0.075, right=0.985, top=0.955, bottom=0.035,
                            hspace=0.28, wspace=0.2)
    top = fig.add_subplot(grid[0, :])
    delta = result[tables.MONTHS]
    top.plot(months, delta.loc[CLIMATOLOGY], **DASHED)
    for model in si.MODELS:
        top.plot(months, delta.loc[model], **model_style(model))
    ggplot_panel(top, r"Mean difference $\Delta\mu$ over the nine stations", delta.to_numpy())
    top.set_ylabel(r"$\Delta\mu$ (mm day$^{-1}$)")
    for k, station in enumerate(STATIONS):
        ax = fig.add_subplot(grid[1 + k // 3, k % 3])
        rows = table.loc[station, tables.MONTHS]
        ax.plot(months, rows.loc[CLIMATOLOGY], **DASHED)
        for model in si.MODELS:
            ax.plot(months, rows.loc[model], **model_style(model))
        ax.plot(months, rows.loc["observed"], **OBSERVED)
        ggplot_panel(ax, STATIONS[station][0], rows.to_numpy())
        ax.tick_params(labelbottom=k >= 6)
        if k % 3 == 0:
            ax.set_ylabel("mm day$^{-1}$")
    handles = [plt.Line2D([], [], **OBSERVED), plt.Line2D([], [], **DASHED)]
    handles += [plt.Line2D([], [], **model_style(m)) for m in si.MODELS]
    fig.legend(handles, ["Observed 2018–2025", "Observed 1980–2017"] + [LABEL[m] for m in si.MODELS], loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.997), columnspacing=1.0, handlelength=1.8, handletextpad=0.5)
    fig.savefig(path)
    plt.close(fig)


def main():
    table = means()
    difference, result = delta_mu(table)
    with pd.ExcelWriter(si.OUT / "stations_annual_cycle.xlsx") as writer:
        table.to_excel(writer, sheet_name="means")
        result.to_excel(writer, sheet_name="delta_mu")
    cycle_figure(table, result, paths.FIGURES / "annual_cycle_stations.pdf")

    pd.set_option("display.width", 250)
    gap = max((difference.loc[(s, m)].abs() - saved_delta_mu(s, m)).abs().max() for s in STATIONS for m in si.MODELS)
    assert gap < 1e-4, gap
    print(f"largest difference from Delta_mu of the run folders: {gap:.1e} mm/day")
    print("Delta mu (mm/day) over the nine stations: models trained at the Annual scale, and observed means of 1980-2017\n",
          result.round(3).to_string())
    models = result.loc[si.MODELS, tables.MONTHS]
    print("months where Delta mu is lower than that of the observed means of 1980-2017:",
          (models < result.loc[CLIMATOLOGY, tables.MONTHS]).sum(axis=1).to_dict())
    print("largest Delta mu of each model:", models.idxmax(axis=1).to_dict(), models.max(axis=1).round(3).to_dict())
    print("ratio of the average over the months to the whole test period:", (result["Avg."] / result["All"]).round(1).to_dict())
    relative = 100 * difference / table.xs("observed", level="series").reindex(difference.index, level="station")
    print("whole test period: signed difference (mm/day) and relative difference (%) at each station\n",
          pd.concat({"mm/day": difference["All"].unstack(), "%": relative["All"].unstack()}, axis=1)
          .reindex(list(STATIONS)).round(2).to_string())
    print("share of each station in Delta mu of DSSM-BG over the whole test period:",
          (difference["All"].xs("DSSM", level="series").abs() / difference["All"].xs("DSSM", level="series").abs().sum())
          .round(2).to_dict())
    print("month of the largest mean at each station (observed 2018-2025, 1980-2017 and models)\n",
          table[tables.MONTHS].idxmax(axis=1).unstack().reindex(list(STATIONS)).to_string())
    print("relative difference (%) in October at Perpignan:", relative.loc["perpignan", "Oct"].round(1).to_dict())


if __name__ == "__main__":
    main()
