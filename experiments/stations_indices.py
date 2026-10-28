"""VALUE indices, heavy precipitation and predictive distributions at the nine stations (Tables 10-12, Section 4.3),
from the saved test ensembles. Pairs of days and spells only join consecutive calendar days."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import make_tables_8vars as tables
import paths
from precipitation_metrics import crps_ensemble_fair, twcrps_ensemble_fair
from stations_summary import STATIONS, precipitation

OUT = paths.RESULTS / "model_comparison_stations"
TAU = 1.0
MODELS = ["GLM-BG", "RF-BG", "MLP-BG", "DSSM"]
TEX = {"GLM-BG": "GLM--BG", "RF-BG": "RF--BG", "MLP-BG": "MLP--BG", "DSSM": "DSSM--BG", "DSSM-GRU": "Causal context"}
FOLDER = {"GLM-BG": "GLM_Annual", "RF-BG": "RF_Annual", "MLP-BG": "MLP_BG_Annual", "DSSM": "DSSM_Annual", "DSSM-GRU": "DSSM_Annual"}
INDICES = ["R01", "SDII", "SD", "P95", "P98", "P99", "WW", "DW", "WetSpell", "DrySpell"]
SUMMARY = ["R01", "SDII", "P98", "WW", "DW", "WetSpell", "DrySpell"]       # indices of the seasonal table (tab:value_summary)
NAME = {"WetSpell": "Wet spell", "DrySpell": "Dry spell", "KS": "KS"}
QUANTILES = (5, 25, 75, 95)        # over the members, for the box plots
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
GROUPS = ["Annual"] + list(SEASONS)


def ensemble(station, model):
    files = sorted((tables.runs_0606(station)[model] / FOLDER[model]).glob("seed*/test_ensemble.npz"))
    assert len(files) == 1, (station, model, files)
    with np.load(files[0], allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def indices(values, days, reference_wet=None):
    """Indices of every column of values (days x simulations); values at or below TAU are dry."""
    values = np.where(values > TAU, values, 0.0)
    consecutive = np.diff(days) == 1
    wet = values > TAU
    amounts = np.where(wet, values, np.nan)
    previous, current = wet[:-1][consecutive], wet[1:][consecutive]
    # a spell starts on the first day, after a gap and when the state changes; mean length = days / spells
    start = np.vstack([np.ones((1, wet.shape[1]), bool), (wet[1:] != wet[:-1]) | ~consecutive[:, None]])
    result = {"R01": wet.mean(0), "SDII": np.nanmean(amounts, 0), "SD": values.std(0),
              "P95": np.nanpercentile(amounts, 95, 0), "P98": np.nanpercentile(amounts, 98, 0),
              "P99": np.nanpercentile(amounts, 99, 0), "WW": (previous & current).sum(0) / previous.sum(0),
              "DW": (~previous & current).sum(0) / (~previous).sum(0),
              "WetSpell": wet.sum(0) / (start & wet).sum(0), "DrySpell": (~wet).sum(0) / (start & ~wet).sum(0)}
    if reference_wet is not None:
        result["KS"] = np.array([ks_2samp(values[wet[:, k], k], reference_wet).statistic for k in range(values.shape[1])])
    return pd.DataFrame(result)


def median_bias(simulated, observed, days, mask, ks=True):
    """Median over the members of the relative bias (%) of every index, with its mean and quantiles, the median KS and
    its 5% critical value."""
    obs = indices(observed[mask][:, None], days[mask]).iloc[0]
    sim = indices(simulated[mask], days[mask], observed[mask][observed[mask] > TAU] if ks else None)
    row = {}
    for k in INDICES:
        rb = 100 * (sim[k] - obs[k]) / obs[k]
        row[k], row[f"{k}_mean"] = np.median(rb), np.mean(rb)
        row.update({f"{k}_q{q:02d}": np.percentile(rb, q) for q in QUANTILES})
    if ks:
        n, m = (observed[mask] > TAU).sum(), np.median((simulated[mask] > TAU).sum(0))
        row["KS"], row["KS_crit"] = np.median(sim.KS), 1.358 * np.sqrt((n + m) / (n * m))
        row.update({f"KS_q{q:02d}": np.percentile(sim.KS, q) for q in QUANTILES})
    return row


def masks(dates):
    """Whole test period and the four seasons."""
    return {"Annual": np.ones(len(dates), bool), **{s: dates.month.isin(m) for s, m in SEASONS.items()}}


def observed_wet_days():
    """Observed wet days of each station and season, and how many exceed the seasonal P98."""
    rows = []
    for station in STATIONS:
        series = precipitation(station).loc["2018-01-01":"2025-12-31"]
        for season, months in SEASONS.items():
            x = series[series.index.month.isin(months)].to_numpy()
            wet = x[x > TAU]
            rows.append({"station": station, "season": season, "wet_days": len(wet),
                         "above_P98": int((wet > np.percentile(wet, 98)).sum())})
    return pd.DataFrame(rows)


def compute():
    bias, tw, mech = [], [], []
    for station in STATIONS:
        series = precipitation(station)
        fit = series.loc[:"2017-12-31"]
        level = np.percentile(fit[fit > TAU], 95)
        q99 = np.percentile(fit[fit > TAU], 99)
        ens = {m: ensemble(station, m) for m in MODELS + ["DSSM-GRU"]}
        dates = pd.DatetimeIndex(ens["DSSM"]["dates"])
        days = dates.values.astype("datetime64[D]").astype(np.int64)
        observed = series.loc[dates].to_numpy()
        observed = np.where(observed > TAU, observed, 0.0)
        wet = observed > TAU
        for model, e in ens.items():
            assert (pd.DatetimeIndex(e["dates"]) == dates).all()
            members = e["rainfall"].astype(np.float64)
            for group, mask in masks(dates).items():
                bias.append({"station": station, "group": group, "model": model,
                             **median_bias(members, observed, days, mask)})
            top = observed > q99
            above = (observed[top] > np.percentile(members[top], 95, axis=1)).mean()
            tw.append({"station": station, "model": model, "level_mm": level,
                       "twCRPS": twcrps_ensemble_fair(observed, members, level).mean(),
                       "days_above_q99": int(top.sum()), "share_obs_above_member_q95": above})
        d, m = ens["DSSM"], ens["MLP-BG"]
        crps_d = crps_ensemble_fair(observed, d["rainfall"].astype(np.float64))
        crps_m = crps_ensemble_fair(observed, m["rainfall"].astype(np.float64))
        pi = d["wet_probability"].astype(np.float64)
        band = np.percentile(pi, 95, axis=1) - np.percentile(pi, 5, axis=1)
        median_pi = np.median(pi, axis=1)
        mech.append({"station": station, "band": np.median(band), "band_low": np.median(band[median_pi < 0.1]),
                     "band_mid": np.median(band[(median_pi >= 0.3) & (median_pi <= 0.7)]),
                     "crps_dry_dssm": crps_d[~wet].mean(), "crps_dry_mlp": crps_m[~wet].mean(),
                     "crps_wet_dssm": crps_d[wet].mean(), "crps_wet_mlp": crps_m[wet].mean(),
                     "pi_dry_dssm": pi[~wet].mean(), "pi_dry_mlp": m["wet_probability"][~wet].mean(),
                     "mu_dry_dssm": np.median(d["mu"][~wet]), "mu_dry_mlp": np.median((m["alpha"] / m["beta"])[~wet]),
                     "alpha_wet_dssm": d["alpha"][wet].mean(), "alpha_wet_mlp": m["alpha"][wet].mean()})
    return pd.DataFrame(bias), pd.DataFrame(tw), pd.DataFrame(mech).set_index("station")


def tex_name(station):
    return "Mean" if station == "Mean" else STATIONS[station][0].replace("é", "\\'e")


def bolded(text, flag):
    return f"\\textbf{{{text}}}" if flag else text


def over_stations(bias, how):
    """Median or mean over the stations of the absolute median bias: one frame (models x groups) per index."""
    frame = bias[bias.model.isin(MODELS)].copy()
    frame[INDICES] = frame[INDICES].abs()
    stat = frame.groupby(["model", "group"])[INDICES + ["KS"]].agg(how)
    return {k: stat[k].unstack("group")[GROUPS].loc[MODELS] for k in INDICES + ["KS"]}


def summary_blocks(bias, columns):
    """Median and mean over the stations, one block each; bold: lowest value of each column, before rounding."""
    lines = []
    for b, (name, how) in enumerate((("Median", "median"), ("Mean", "mean"))):
        stats = over_stations(bias, how)
        block = pd.DataFrame({c: stats[c[0]][c[1]] for c in columns})
        best = block.idxmin()
        if b:
            lines.append("\\midrule")
        for i, model in enumerate(MODELS):
            cells = [bolded(f"{block.loc[model, c]:.{3 if c[0] == 'KS' else 1}f}", model == best[c]) for c in columns]
            first = f"\\multirow{{{len(MODELS)}}}{{*}}{{{name}}}" if i == 0 else ""
            lines.append(f"{first} & {TEX[model]} & " + " & ".join(cells) + "\\\\")
    return lines


def summary_table(bias):
    """Seasonal table: one block per index, median and mean over the stations with their average over the seasons."""
    stats = {how: over_stations(bias, how) for how in ("median", "mean")}
    lines = []
    for j, k in enumerate(SUMMARY + ["KS"]):
        if j:
            lines.append("\\midrule")
        parts = [stats[how][k][list(SEASONS)] for how in stats]
        block = pd.concat([p.assign(Avg=p.mean(axis=1)) for p in parts], axis=1, keys=list(stats))
        best, digits = block.idxmin(), 3 if k == "KS" else 1
        for i, model in enumerate(MODELS):
            cells = [bolded(f"{block.loc[model, c]:.{digits}f}", model == best[c]) for c in block.columns]
            name = f"\\multirow{{{len(MODELS)}}}{{*}}{{{NAME.get(k, k)}}}" if i == 0 else ""
            lines.append(f"{name} & {TEX[model]} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines) + "\n"


def whole_table(bias):
    """Whole test period: median and mean over the stations of the absolute biases."""
    return "\n".join(summary_blocks(bias, [(c, "Annual") for c in INDICES + ["KS"]])) + "\n"


def predictive_table(mech):
    """Per station: width of the pi band by situation, then MLP-BG and DSSM-BG on observed dry and wet days."""
    columns = [("band_low", 2), ("band_mid", 2), ("crps_dry_mlp", 3), ("crps_dry_dssm", 3), ("crps_wet_mlp", 2),
               ("crps_wet_dssm", 2), ("pi_dry_mlp", 3), ("pi_dry_dssm", 3), ("mu_dry_mlp", 1), ("mu_dry_dssm", 1),
               ("alpha_wet_mlp", 2), ("alpha_wet_dssm", 2)]
    lines = []
    for station, row in mech.iterrows():
        cells = []
        for name, d in columns:
            value = f"{row[name]:.{d}f}"
            if name.endswith("_dssm") and name.startswith("crps"):   # bold: lower CRPS of the two models
                other = row[name.replace("_dssm", "_mlp")]
                cells[-1] = f"\\textbf{{{cells[-1]}}}" if round(other, d) < round(row[name], d) else cells[-1]
                value = f"\\textbf{{{value}}}" if round(row[name], d) < round(other, d) else value
            cells.append(value)
        lines.append(f"{tex_name(station)} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines) + "\n"


def main():
    bias, tw, mech = compute()
    with pd.ExcelWriter(OUT / "stations_indices.xlsx") as writer:
        bias.to_excel(writer, sheet_name="indices", index=False)
        tw.to_excel(writer, sheet_name="heavy", index=False)
        mech.to_excel(writer, sheet_name="predictive")
    (OUT / "table_stations_value_summary.tex").write_text(summary_table(bias))
    (OUT / "table_stations_indices_whole.tex").write_text(whole_table(bias))
    (OUT / "table_stations_predictive.tex").write_text(predictive_table(mech))

    pd.set_option("display.width", 250)
    four = bias[bias.model.isin(MODELS)]
    seasonal, annual = four[four.group.isin(SEASONS)], four[four.group == "Annual"]
    for k in ("SD", "P98"):
        x = seasonal.pivot_table(index=["station", "group"], columns="model", values=k)[MODELS]
        print(f"{k}, 36 station-seasons: bias > 0", (x > 0).sum().to_dict(), "| median bias", x.median().round(1).to_dict())
    best = {st: sum(g[g.group == s].set_index("model")[k].abs().idxmin() == "DSSM" for s in SEASONS for k in SUMMARY + ["KS"])
            for st, g in seasonal.groupby("station")}
    print("combinations (of 32) of the main-text indices and KS with the seasons where DSSM-BG has the smallest absolute "
          "bias or the lowest KS, by station:", best)
    medians, means = over_stations(bias, "median"), over_stations(bias, "mean")
    for how, stats in (("median", medians), ("mean", means)):
        avg = pd.DataFrame({k: stats[k][list(SEASONS)].mean(axis=1) for k in SUMMARY + ["KS"]})
        print(f"average over the four seasons of the {how} over the stations (column Avg.)\n", avg.round(4).to_string())
    for k in SUMMARY + ["KS"]:
        d = 3 if k == "KS" else 1
        print(f"{k}: median over the stations of the absolute median bias\n", medians[k].round(d).to_string())
        print(f"{k}: mean over the stations of the absolute median bias\n", means[k].round(d).to_string())
    for k in SUMMARY:
        v = seasonal.groupby(["model", "group"])[k]
        box = pd.concat({"median": v.median(), "q25": v.quantile(0.25), "q75": v.quantile(0.75)}, axis=1)
        print(f"{k}: median and quartiles over the stations of the median bias\n", box.unstack("group").round(1).to_string())
    for k in ("WW", "DW", "WetSpell", "DrySpell"):
        for group in GROUPS:
            g = four[four.group == group].pivot_table(index="station", columns="model", values=k)[MODELS].abs().round(1)
            lowest = g.eq(g.min(axis=1), axis=0).sum()                       # printed values, ties count for both
            print(f"{k} {group}: stations with the smallest absolute bias", lowest.to_dict())
    for k in SUMMARY:
        x = seasonal.pivot_table(index=["station", "group"], columns="model", values=k)[MODELS]
        print(f"{k}, 36 station-seasons: |bias| > 15%", (x.abs() > 15).sum().to_dict(), "| bias > 0", (x > 0).sum().to_dict(),
              "| largest |bias|", x.abs().max().round(1).to_dict())
        print(f"{k} at each station in each season\n",
              seasonal.pivot_table(index="station", columns=["model", "group"], values=k).reindex(list(STATIONS))[MODELS].round(1).to_string())
    outside = four.assign(**{k: (four[f"{k}_q05"] > 0) | (four[f"{k}_q95"] < 0) for k in INDICES})
    print("station-seasons (of 36) where the observed index lies outside the 5-95% range of the members\n",
          outside[outside.group.isin(SEASONS)].groupby("model")[INDICES].sum().loc[MODELS].to_string())
    print("stations (of 9) where it lies outside, whole test period\n",
          outside[outside.group == "Annual"].groupby("model")[INDICES].sum().loc[MODELS].to_string())
    width = four.assign(**{k: four[f"{k}_q95"] - four[f"{k}_q05"] for k in INDICES})
    for name, part in (("within a season, median over the stations and seasons", width[width.group.isin(SEASONS)]),
                       ("whole test period, median over the stations", width[width.group == "Annual"])):
        print(f"width of the 5-95% member range of the relative bias (percentage points), {name}\n",
              part.groupby("model")[["R01", "WW", "DW", "P98"]].median().loc[MODELS].round(1).to_string())
    crit = four.groupby(["group", "station"]).KS_crit.mean()                 # mean over the models at each station
    print("5% critical value of the KS test, range over the stations\n",
          crit.groupby("group").agg(["min", "max"]).loc[GROUPS].round(3).to_string(),
          "\nall seasons:", crit.drop("Annual").agg(["min", "max"]).round(3).to_dict())
    print("station-seasons (of 36) where the median KS statistic exceeds the critical value\n",
          (seasonal.KS > seasonal.KS_crit).groupby(seasonal.model).sum().loc[MODELS].to_dict())
    wet_days = observed_wet_days()
    print("observed wet days in each season of 2018-2025, and how many exceed the seasonal P98\n",
          wet_days.pivot(index="station", columns="season", values=["wet_days", "above_P98"]).to_string())
    for k in ("WW", "DW"):
        v = four.pivot_table(index=["station", "model"], columns="group", values=k)[GROUPS]
        print(f"{k} by season at Clermont-Ferrand and Perpignan\n", v.loc[["clermont_ferrand", "perpignan"]].round(1).to_string())
    for k in INDICES:
        v = annual.pivot_table(index="station", columns="model", values=k).reindex(list(STATIONS))
        print(f"{k:8s} min/max per model:", {m: (round(v[m].min(), 1), round(v[m].max(), 1)) for m in MODELS},
              "| mean |.|:", v.abs().mean().round(1).to_dict())
    gap = max((four[f"{k}_mean"] - four[k]).abs().max() for k in SUMMARY)
    print(f"largest difference between the mean and the median relative bias over the members (main-text indices): {gap:.2f}")
    print("5% critical value of the KS test, whole period:", annual.groupby("station").KS_crit.mean().round(3).to_dict())
    ks = annual.pivot_table(index="station", columns="model", values="KS").reindex(list(STATIONS))
    print("median KS\n", ks.round(3).to_string(), "\nmean", ks.mean().round(3).to_dict())
    print("lowest KS at:", ks[MODELS].round(3).idxmin(axis=1).value_counts().to_dict())
    two = tw.pivot_table(index="station", columns="model", values="twCRPS").reindex(list(STATIONS))
    print("twCRPS\n", two.round(4).to_string(), "\nlowest at:", two[MODELS].round(4).idxmin(axis=1).value_counts().to_dict())
    print("days above q99 and share above member q95\n",
          tw.pivot_table(index="station", columns="model", values="share_obs_above_member_q95").round(2).to_string(),
          tw.groupby("station").days_above_q99.first().to_dict())
    print("predictive distribution\n", mech.round(3).to_string())
    dry = 100 * (mech.crps_dry_dssm / mech.crps_dry_mlp - 1)
    wet = 100 * (1 - mech.crps_wet_dssm / mech.crps_wet_mlp)
    print("dry days: DSSM-BG higher by", dry.round(1).to_dict())
    print("wet days: DSSM-BG lower by", wet.round(1).to_dict())


if __name__ == "__main__":
    main()
