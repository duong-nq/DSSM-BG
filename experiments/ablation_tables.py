"""Ablation of DSSM-BG (Tables 7 and 8): CRPS, scores and VALUE indices of six variants at the nine stations."""

from pathlib import Path

import numpy as np
import pandas as pd

import make_tables_8vars as tables
import paths
import sensitivity_tables as sens
import stations_indices as si
from precipitation_metrics import crps_ensemble_fair
from stations_summary import STATIONS, read_crps

OUT = paths.RESULTS / "model_comparison_stations"
# key, label, context, inference network ("--": no latent state)
ROWS = [("DSSM-BG", "DSSM--BG", "Bidirectional", "Filtering"),
        ("smoothing", "DSSM--BG", "Bidirectional", "Smoothing"),
        ("causal", "DSSM--BG", "Causal", "Filtering"),
        ("BiGRU-BG", "BiGRU--BG", "Bidirectional", "--"),
        ("GRU-BG", "GRU--BG", "Causal", "--"),
        ("MLP-BG", "MLP--BG", "None", "--")]
# column, header, decimals, "low" or "high" is better
SCORES = [("CRPS", "CRPS$\\downarrow$", 3, "low"), ("NLL", "NLL$\\downarrow$", 4, "low"),
          ("Brier", "Brier$\\downarrow$", 4, "low"), ("AUC", "AUC$\\uparrow$", 4, "high"),
          ("COR", "COR$\\uparrow$", 4, "high"), ("RMSE", "RMSE$\\downarrow$", 4, "low"),
          ("WW", "WW$\\downarrow$", 1, "low"), ("DW", "DW$\\downarrow$", 1, "low"),
          ("WetSpell", "Wet spell$\\downarrow$", 1, "low"), ("DrySpell", "Dry spell$\\downarrow$", 1, "low"),
          ("KS", "KS$\\downarrow$", 3, "low")]
SEASONAL = ["R01", "SDII", "P98", "WW", "DW", "WetSpell", "DrySpell", "KS"]
tables.PREFIX.update({"GRU-BG": "GRU_BG", "BiGRU-BG": "BiGRU_BG"})


def folders(station):
    """Run folder (read by make_tables_8vars.read_model), model key of that reader and ensemble folder."""
    main = tables.runs_0606(station)
    mlp = paths.run_of(paths.MLP_RUNS, station) / station
    smooth = paths.SMOOTHER_RUN / station / "DSSM_BiGRU_SmoothEnc"
    return {"DSSM-BG": (main["DSSM"], "DSSM", main["DSSM"] / "DSSM_Annual"),
            "smoothing": (smooth, "DSSM", smooth / "DSSM_Annual"),
            "BiGRU-BG": (paths.BIGRU_RUN / station, "BiGRU-BG", paths.BIGRU_RUN / station / "BiGRU_BG_Annual"),
            "causal": (main["DSSM-GRU"], "DSSM-GRU", main["DSSM-GRU"] / "DSSM_Annual"),
            "GRU-BG": (mlp, "GRU-BG", mlp / "GRU_BG_Annual"),
            "MLP-BG": (main["MLP-BG"], "MLP-BG", main["MLP-BG"] / "MLP_BG_Annual")}


def ensemble(folder):
    files = sorted(folder.glob("seed*/test_ensemble.npz"))
    assert len(files) == 1, folder
    with np.load(files[0]) as saved:
        return {k: saved[k] for k in saved.files}


def collect():
    """Per station and variant: stored scores (three scales) and scores computed from the Annual test ensemble."""
    rows = []
    for station in STATIONS:
        series = si.precipitation(station)
        fit = series.loc[:"2017-12-31"]
        q99 = np.percentile(fit[fit > si.TAU], 99)
        for key, (folder, model, ens_folder) in folders(station).items():
            stored = {r["scale"]: r for r in tables.read_model(Path(folder), model) if r["period"] == "Annual"}
            e = ensemble(ens_folder)
            dates = pd.DatetimeIndex(e["dates"])
            days = dates.values.astype("datetime64[D]").astype(np.int64)
            observed = series.loc[dates].to_numpy()
            observed = np.where(observed > si.TAU, observed, 0.0)
            wet, top = observed > si.TAU, observed > q99
            members = e["rainfall"].astype(np.float64)
            crps = crps_ensemble_fair(observed, members).mean()
            assert np.isclose(crps, stored["Annual"]["CRPS"], rtol=1e-6), (station, key)
            groups = si.masks(dates)
            bias = si.median_bias(members, observed, days, groups.pop("Annual"))
            seasons = pd.DataFrame([si.median_bias(members, observed, days, m) for m in groups.values()],
                                   index=list(groups))
            rows.append({"station": station, "variant": key,
                         **{k: stored["Annual"][k] for k in ("CRPS", "NLL", "Brier", "AUC", "COR", "RMSE")},
                         "CRPS_Season": stored["Season"]["CRPS"] if "Season" in stored else np.nan,
                         "CRPS_Month": stored["Month"]["CRPS"] if "Month" in stored else np.nan,
                         **{f"{k}_{s}": seasons.loc[s, k] for k in SEASONAL for s in seasons.index},
                         **{f"{k}_Annual": bias[k] for k in SEASONAL},
                         "alpha_wet": e["alpha"][wet].mean(),
                         "q99_share_above_q95": (observed[top] > np.percentile(members[top], 95, axis=1)).mean()})
    return pd.DataFrame(rows)


def seasonal_medians(frame):
    """Median over the stations of the absolute median bias: one frame (variants x seasons) per index."""
    return {k: pd.DataFrame({s: frame.assign(v=frame[f"{k}_{s}"].abs()).groupby("variant").v.median() for s in si.SEASONS})
            for k in SEASONAL}


def crps_body(table):
    printed = table.round(3)
    best = printed.min(axis=0)
    lines = []
    for key, label, context, inference in ROWS:
        cells = [f"\\textbf{{{v:.3f}}}" if v == best[c] else f"{v:.3f}" for c, v in printed.loc[key].items()]
        lines.append(f"{label} & {context} & {inference} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines) + "\n"


def scores_body(mean):
    lines = []
    cells = {}
    for column, _, decimals, rule in SCORES:
        printed = mean[column].round(decimals)
        best = printed.min() if rule == "low" else printed.max()
        cells[column] = [f"\\textbf{{{v:.{decimals}f}}}" if v == best else f"{v:.{decimals}f}" for v in printed]
    for i, (_, label, context, inference) in enumerate(ROWS):
        lines.append(f"{label} & {context} & {inference} & " + " & ".join(cells[c][i] for c, *_ in SCORES) + "\\\\")
    return "\n".join(lines) + "\n"


def main():
    frame = collect()
    order = [key for key, *_ in ROWS]
    table = frame.pivot(index="variant", columns="station", values="CRPS").reindex(order)[list(STATIONS)]
    # the main runs are the 512-day windows of the window study
    window, _ = sens.window_crps()
    assert np.allclose(table.loc["DSSM-BG"], window.loc[("DSSM_BiGRU", "512")])
    assert np.allclose(table.loc["smoothing"], window.loc[("DSSM_BiGRU_SmoothEnc", "512")])
    main_crps = read_crps().query("scale == 'Annual'").pivot_table(index="model", columns="station", values="CRPS")
    for key, model in (("DSSM-BG", "b-DSSM"), ("causal", "u-DSSM"), ("MLP-BG", "MLP-BG")):
        assert np.allclose(table.loc[key], main_crps.loc[model, list(STATIONS)]), key
    table.columns = [STATIONS[s][0] for s in table.columns]
    table["Mean"] = table.mean(axis=1)
    by_season = seasonal_medians(frame)
    mean = frame.groupby("variant")[["CRPS", "NLL", "Brier", "AUC", "COR", "RMSE"]].mean()
    for k in ("WW", "DW", "WetSpell", "DrySpell", "KS"):      # mean over the seasons of the median over the stations
        mean[k] = by_season[k].mean(axis=1)
    mean = mean.reindex(order)

    table.to_excel(OUT / "ablation_crps.xlsx")
    with pd.ExcelWriter(OUT / "ablation_scores.xlsx") as writer:
        mean.to_excel(writer, sheet_name="mean")
        frame.to_excel(writer, sheet_name="stations", index=False)
    (OUT / "table_ablation.tex").write_text(crps_body(table))
    (OUT / "table_ablation_scores.tex").write_text(scores_body(mean))

    pd.set_option("display.width", 220)
    print(table.round(3).to_string())
    print(mean.round(4).to_string())
    for k in SEASONAL:
        print(f"{k}: median over the stations of the absolute median bias in each season, and its mean over the seasons\n",
              by_season[k].reindex(order).assign(mean=lambda d: d.mean(axis=1)).round(3 if k == "KS" else 1).to_string())
    crps = table.drop(columns="Mean")

    def lower(with_, without):
        gain = 100 * (1 - crps.loc[with_] / crps.loc[without])
        overall = 100 * (1 - table.loc[with_, "Mean"] / table.loc[without, "Mean"])
        return gain, overall

    steps = [("bidirectional context, no latent state (MLP-BG -> BiGRU-BG)", "BiGRU-BG", "MLP-BG"),
             ("latent state, bidirectional context (BiGRU-BG -> DSSM-BG)", "DSSM-BG", "BiGRU-BG"),
             ("bidirectional context, with latent state (causal -> DSSM-BG)", "DSSM-BG", "causal"),
             ("bidirectional context, no latent state (GRU-BG -> BiGRU-BG)", "BiGRU-BG", "GRU-BG"),
             ("latent state, causal context (GRU-BG -> causal)", "causal", "GRU-BG"),
             ("causal recurrent context (MLP-BG -> GRU-BG)", "GRU-BG", "MLP-BG"),
             ("smoothing instead of filtering (smoothing -> DSSM-BG)", "DSSM-BG", "smoothing")]
    for name, with_, without in steps:
        gain, overall = lower(with_, without)
        print(f"{name}: mean CRPS lower by {overall:.2f}% | per station {gain.min():+.2f} to {gain.max():+.2f}% | lower at {(gain > 0).sum()} stations")
    gain, _ = lower("DSSM-BG", "BiGRU-BG")
    print("latent state with the bidirectional context, per station (%):", gain.round(2).to_dict())
    alpha = frame.pivot(index="variant", columns="station", values="alpha_wet")
    print("mean alpha on observed wet days:", {k: (round(alpha.loc[k].min(), 2), round(alpha.loc[k].max(), 2)) for k in ("MLP-BG", "BiGRU-BG", "DSSM-BG")})
    q99 = frame.pivot(index="variant", columns="station", values="q99_share_above_q95")
    print("share of the days above q99 with the observation above the member q95:", q99.min(axis=1).round(2).to_dict())
    perpignan = frame[frame.station == "perpignan"].set_index("variant")
    for k in ("WW", "DW"):
        print(f"{k} at Perpignan (whole period and seasons):\n",
              perpignan[[f"{k}_{g}" for g in ("Annual", "DJF", "MAM", "JJA", "SON")]].round(1).to_string())
    for scale in ("Season", "Month"):
        s = frame.pivot(index="variant", columns="station", values=f"CRPS_{scale}")
        print(f"{scale}: mean CRPS BiGRU-BG {s.loc['BiGRU-BG'].mean():.3f}, DSSM-BG {s.loc['DSSM-BG'].mean():.3f}; "
              f"BiGRU-BG lower at {(s.loc['BiGRU-BG'] < s.loc['DSSM-BG']).sum()} stations")


if __name__ == "__main__":
    main()
