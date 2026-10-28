"""Table 9 (training window) and the numbers of the appendix on the sensitivity of DSSM-BG to the training choices."""

import numpy as np
import pandas as pd

import paths
from stations_summary import STATIONS, read_crps

RESULTS = paths.RESULTS
OUT = RESULTS / "model_comparison_stations"
WINDOWS = ["128", "256", "512", "1024", "full"]
VERSIONS = [("DSSM_BiGRU", "DSSM--BG", "Filtering"), ("DSSM_BiGRU_SmoothEnc", "DSSM--BG", "Smoothing"),
            ("DSSM_GRU", "Causal context", "Filtering"), ("DSSM_GRU_SmoothEnc", "Causal context", "Smoothing")]
TABLE_VERSIONS = VERSIONS[:2]
SMOOTHER_RUN = paths.SMOOTHER_RUN


def main_run(station):
    return paths.run_of(paths.DSSM_RUNS, station) / station


def window_crps():
    """CRPS of every version and window at the nine stations: rows (version, window), columns stations."""
    long = pd.read_excel(RESULTS / "latex_tables_window_sensitivity" / "window_metrics_long.xlsx")
    long = long[long.station.isin(STATIONS)]
    long["window"] = long.window.astype(str)
    table = long.pivot_table(index=["arch", "window"], columns="station", values="CRPS")[list(STATIONS)]
    return table, long


def scale_crps(folder, arch):
    """Annual-period CRPS of the three training scales of one run folder."""
    scores = pd.read_excel(folder / arch / "scores.xlsx")
    scores = scores[scores.period.astype(str) == "Annual"].drop_duplicates("variant").set_index("variant")
    return scores.CRPS


def body(table):
    """Rows of the window table; bold: lowest printed value of each column within each version."""
    lines = []
    for i, (arch, _, encoder) in enumerate(TABLE_VERSIONS):
        block = table.loc[arch].reindex(WINDOWS).copy()
        block["Mean"] = block.mean(axis=1)
        printed = block.round(3)
        best = printed.min(axis=0)
        if i:
            lines.append("\\midrule")
        for j, window in enumerate(WINDOWS):
            cells = [f"\\textbf{{{v:.3f}}}" if v == best[c] else f"{v:.3f}" for c, v in printed.loc[window].items()]
            lead = f"\\multirow{{5}}{{*}}{{{encoder}}}" if j == 0 else ""
            label = "none" if window == "full" else window
            lines.append(f"{lead} & {label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines) + "\n"


def main():
    table, long = window_crps()
    (OUT / "table_window.tex").write_text(body(table))

    # the windows of 512 days with the filtering encoder are the main runs
    for arch in ("DSSM_GRU", "DSSM_BiGRU"):
        main_crps = pd.Series({s: scale_crps(main_run(s), arch)["DSSM_Annual"] for s in STATIONS})
        assert np.allclose(main_crps, table.loc[(arch, "512")]), arch

    mean = table.mean(axis=1).unstack()[WINDOWS]
    print("mean CRPS over the nine stations\n", mean.round(4).to_string())
    for arch in ("DSSM_GRU", "DSSM_BiGRU"):
        relative = 100 * (table.loc[arch].T.div(table.loc[(arch, "512")], axis=0) - 1)
        within = relative[["128", "256", "1024"]]
        print(f"{arch}: mean change against 512 (%) {(100 * (mean.loc[arch] / mean.loc[arch, '512'] - 1)).round(2).to_dict()}"
              f" | per station, 128-1024: {within.min().min():.1f} to {within.max().max():.1f}"
              f" | without windows: {relative['full'].min():.1f} to {relative['full'].max():.1f}")
    updates = long.groupby(["window", "arch"]).updates_to_best.agg(["min", "median", "max"])
    print("updates to the selected checkpoint\n", updates.to_string())

    # DSSM-BG at each window against the best baseline at any scale and BiGRU-BG
    crps = read_crps()
    best = crps[crps.model.isin(["GLM-BG", "RF-BG", "MLP-BG"])].groupby("station").CRPS.min()[list(STATIONS)]
    bigru = pd.read_excel(paths.BIGRU_RUN / "all_stations_scores.xlsx")
    bigru = bigru[(bigru.period == "Annual") & (bigru.time_scale == "Annual")].set_index("station").CRPS_ensemble_fair[list(STATIONS)]
    for window in WINDOWS[:4]:
        d = table.loc[("DSSM_BiGRU", window)]
        print(f"window {window}: DSSM-BG lower than the best baseline at {(d < best).sum()} stations, mean {d.mean():.4f} "
              f"against BiGRU-BG {bigru.mean():.4f}; smoothing mean {table.loc[('DSSM_BiGRU_SmoothEnc', window)].mean():.4f}")
    for filtering, smoothing in (("DSSM_GRU", "DSSM_GRU_SmoothEnc"), ("DSSM_BiGRU", "DSSM_BiGRU_SmoothEnc")):
        change = 100 * (table.loc[(smoothing, "512")] / table.loc[(filtering, "512")] - 1)
        better = change[change < 0]
        print(f"{smoothing} vs {filtering}, window 512: mean change {100 * (table.loc[(smoothing, '512')].mean() / table.loc[(filtering, '512')].mean() - 1):.2f}%,"
              f" better at {len(better)} stations by at most {-better.min():.2f}%")
        for variant in ("DSSM_SeasonalConcat", "DSSM_MonthConcat"):
            f = np.mean([scale_crps(main_run(s), filtering)[variant] for s in STATIONS])
            m = np.mean([scale_crps(SMOOTHER_RUN / s, smoothing)[variant] for s in STATIONS])
            print(f"    {variant}: filtering {f:.4f}, smoothing {m:.4f}, change {100 * (m / f - 1):+.2f}%")


if __name__ == "__main__":
    main()
