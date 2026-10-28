"""Figures 10-13 and the numbers of the comparison of the models at Tours (Appendix F), from the saved test ensembles."""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print
from scipy.stats import ks_2samp

import paths
from precipitation_metrics import (MONTH_LABELS, PERIOD_MONTHS, SEASON_ORDER, auc_score,
                                   bernoulli_gamma_log_likelihood, crps_ensemble_fair,
                                   pearson_correlation, threshold_precipitation,
                                   twcrps_ensemble_fair)

TAU = 1.0
STATION = "tours"
RESULTS = paths.RESULTS
DATA = paths.DATA
OUT = RESULTS / "model_comparison_tours"
FIGURES = paths.FIGURES
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SCALES = {"Annual": "Annual", "Season": "SeasonalConcat", "Month": "MonthConcat"}
PREDICTOR_SETS = ("8",)
# "DSSM": DSSM-BG; DSSM-GRU and GRU-BG: causal-context variants of the ablation
ALL_MODELS = ["GLM-BG", "RF-BG", "MLP-BG", "GRU-BG", "DSSM-GRU", "DSSM"]
MODELS = ["GLM-BG", "RF-BG", "MLP-BG", "DSSM"]
TABLE_MODELS = MODELS
ABLATION_MODELS = ["DSSM-GRU", "GRU-BG"]
EXAMPLE_BASELINE = "MLP-BG"  # baseline of the example figure
LABEL = {"GLM-BG": "GLM–BG", "RF-BG": "RF–BG", "MLP-BG": "MLP–BG", "GRU-BG": "GRU–BG", "DSSM-GRU": "causal context", "DSSM": "DSSM–BG",
         "Climatology": "Climatology"}


def sources(predictors):
    """Folder template of every model for one predictor set; {scale} is filled in later."""
    runs = {"glm_rf": paths.run_of(paths.GLM_RUNS, STATION), "rf": paths.RF_RUN,
            "mlp_gru": paths.run_of(paths.MLP_RUNS, STATION), "dssm": paths.run_of(paths.DSSM_RUNS, STATION)}
    layout = {"GLM-BG": ("glm_rf", "GLM_{scale}/seed42"), "RF-BG": ("rf", "RF_{scale}/seed42"),
              "MLP-BG": ("mlp_gru", "MLP_BG_{scale}/seed42"), "GRU-BG": ("mlp_gru", "GRU_BG_{scale}/seed42"),
              "DSSM-GRU": ("dssm", "DSSM_GRU/DSSM_{scale}/seed456"), "DSSM": ("dssm", "DSSM_BiGRU/DSSM_{scale}/seed456")}
    return {model: str(runs[run] / STATION / template) for model, (run, template) in layout.items() if runs[run] is not None}


catalogue = []
for predictors in PREDICTOR_SETS:
    for model, template in sources(predictors).items():
        for scale, folder_scale in SCALES.items():
            path = Path(template.format(scale=folder_scale)) / "test_ensemble.npz"
            if path.exists():
                catalogue.append({"predictors": predictors, "model": model, "scale": scale, "path": path})
catalogue = pd.DataFrame(catalogue)
print(f"{len(catalogue)} fitted systems found")
display(catalogue.assign(run=catalogue.path.map(lambda p: next(q.parent.name for q in p.parents if q.name == STATION)))
        .pivot_table(index=["predictors", "model"], columns="scale", values="run", aggfunc="first")[list(SCALES)])


# ---- Observations, levels of the twCRPS (wet-day quantiles of 1980-2017) and the monthly climatology


with np.load(DATA / f"era5_daily_cell_{STATION}.npz", allow_pickle=False) as data:
    dates = pd.DatetimeIndex(data["dates"].astype("datetime64[ns]"))
    predictor_names = [str(v) for v in data["variables"]]
    cape = pd.Series(data["data"][:, predictor_names.index("cape_max")].astype(np.float64), index=dates)
raw_rainfall = np.load(DATA / f"ecad_rr_{STATION}_1980_2025.npy").astype(np.float64)

annual_totals = pd.Series(raw_rainfall, index=dates).groupby(dates.year).sum()
print(f"record {dates[0].date()} to {dates[-1].date()}: {len(dates)} days, {int(np.isnan(raw_rainfall).sum())} missing | "
      f"mean annual precipitation {annual_totals.mean():.0f} mm | days above {TAU:g} mm: {100 * (raw_rainfall > TAU).mean():.1f}% | "
      f"mean on these days {raw_rainfall[raw_rainfall > TAU].mean():.1f} mm | maximum {raw_rainfall.max():.0f} mm")
print(f"{len(predictor_names)} ERA5 predictors (Table 1):", ", ".join(predictor_names))
y_all = np.where(raw_rainfall <= TAU, 0.0, raw_rainfall)

reference_rows = np.flatnonzero((dates >= "1980-01-01") & (dates <= "2017-12-31"))
test_rows = np.flatnonzero((dates >= "2018-01-01") & (dates <= "2025-12-31"))
observed = y_all[test_rows]
test_dates = dates[test_rows]
months = test_dates.month.to_numpy()
wet_observed = observed > TAU

reference_wet = y_all[reference_rows][y_all[reference_rows] > TAU]
candidates = pd.DataFrame([{"quantile": q, "level_mm": float(np.quantile(reference_wet, q)),
                            "test_days_above": int((observed > np.quantile(reference_wet, q)).sum())}
                           for q in (0.90, 0.95, 0.98, 0.99)])
display(candidates.round(3))
TW_LEVELS = {"twCRPS_q95": float(np.quantile(reference_wet, 0.95)),
             "twCRPS_q98": float(np.quantile(reference_wet, 0.98))}
print(f"test days {len(observed)} | wet fraction {wet_observed.mean():.3f} | "
      + " | ".join(f"{name}: {level:.2f} mm, {int((observed > level).sum())} days above" for name, level in TW_LEVELS.items()))

# monthly climatology: 1000 draws from the 1980-2017 observations of the same calendar month
N_MEMBERS = 1000
climatology_rng = np.random.default_rng(0)
reference_months = dates[reference_rows].month.to_numpy()
climatology_pool = {m: y_all[reference_rows][reference_months == m] for m in range(1, 13)}
climatology_draw = {m: climatology_rng.choice(pool, size=N_MEMBERS, replace=True) for m, pool in climatology_pool.items()}
climatology = np.stack([climatology_draw[m] for m in months])
climatology_crps = crps_ensemble_fair(observed, climatology, TAU)
print(f"monthly climatology: CRPS {climatology_crps.mean():.4f} mm/day")


# ---- Scores of every model and scale, by group of days (whole period, seasons, months)


GROUPS = ([("Overall", np.ones(len(observed), dtype=bool))]
          + [(season, np.isin(months, PERIOD_MONTHS[season])) for season in SEASON_ORDER]
          + [(MONTH_LABELS[m - 1], months == m) for m in range(1, 13)])


def load_system(path):
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved["dates"].astype("datetime64[D]"), test_dates.to_numpy().astype("datetime64[D]")), path
        system = {name: saved[name] for name in ("rainfall", "wet_probability", "alpha", "beta")}
    system["rainfall"] = threshold_precipitation(system["rainfall"].astype(np.float64), TAU)
    return system


def daily_scores(system):
    ensemble = system["rainfall"]
    wet_probability = system["wet_probability"].astype(np.float64)
    daily = {"CRPS": crps_ensemble_fair(observed, ensemble, TAU),
             **{name: twcrps_ensemble_fair(observed, ensemble, level, TAU) for name, level in TW_LEVELS.items()},
             "NLL": -bernoulli_gamma_log_likelihood(observed, wet_probability, system["alpha"].astype(np.float64), system["beta"].astype(np.float64), TAU)}
    probability = wet_probability if wet_probability.ndim == 1 else wet_probability.mean(axis=1)
    daily["Brier"] = (np.clip(probability, 0.0, 1.0) - wet_observed) ** 2
    return daily, probability, ensemble.mean(axis=1)


def group_rows(daily, probability, mean, labels):
    rows = []
    for group, mask in GROUPS:
        target, forecast = observed[mask], mean[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = auc_score(target, probability[mask], TAU) if probability is not None and wet_observed[mask].any() else np.nan
        rows.append({**labels, "group": group, "n_days": int(mask.sum()),
                     **{name: float(np.mean(values[mask])) for name, values in daily.items()},
                     "AUC": float(auc), "COR": pearson_correlation(target, forecast),
                     "MAE": float(np.mean(np.abs(forecast - target))),
                     "RMSE": float(np.sqrt(np.mean((forecast - target) ** 2)))})
    return rows


rows, DAILY, ANNUAL = [], {}, {}
for entry in catalogue.itertuples():
    system = load_system(entry.path)
    daily, probability, mean = daily_scores(system)
    rows += group_rows(daily, probability, mean, {"predictors": entry.predictors, "model": entry.model, "scale": entry.scale})
    DAILY[(entry.predictors, entry.model, entry.scale)] = daily
    if entry.scale == "Annual":
        ANNUAL[(entry.predictors, entry.model)] = {"rainfall": system["rainfall"].astype(np.float32), "wet_probability": system["wet_probability"]}
    del system
rows += group_rows({"CRPS": climatology_crps, **{n: twcrps_ensemble_fair(observed, climatology, v, TAU) for n, v in TW_LEVELS.items()}},
                   None, climatology.mean(axis=1), {"predictors": "-", "model": "Climatology", "scale": "-"})

METRICS = ["CRPS", *TW_LEVELS, "NLL", "Brier", "AUC", "COR", "MAE", "RMSE"]
metrics = pd.DataFrame(rows)[["predictors", "model", "scale", "group", "n_days", *METRICS]]
metrics["model"] = pd.Categorical(metrics["model"], list(LABEL), ordered=True)
metrics["scale"] = pd.Categorical(metrics["scale"], [*SCALES, "-"], ordered=True)
metrics = metrics.sort_values(["predictors", "model", "scale"]).reset_index(drop=True)
print(f"scored {len(metrics)} rows")


# ---- Tables and bootstrap of the CRPS differences


overall = metrics[metrics.group == "Overall"].drop(columns="group")
for name, groups in (("overall", ["Overall"]), ("seasonal", list(SEASON_ORDER)), ("monthly", list(MONTH_LABELS))):
    metrics[metrics.group.isin(groups)].to_excel(OUT / f"comparison_{name}.xlsx", index=False)

TABLE_METRICS = ["CRPS", "twCRPS_q95", "NLL", "Brier", "AUC", "COR", "RMSE"]
HIGHER_IS_BETTER = {"AUC", "COR"}


TABLE_LABEL = {"8": {"GLM-BG": "GLM--BG \\citep{Legasa_PP_2023}", "RF-BG": "RF--BG \\citep{Legasa_2022_RF}",
                     "MLP-BG": "MLP--BG \\citep{Alex_NN_BG_2008}", "GRU-BG": "GRU--BG",
                     "DSSM-GRU": "Causal context", "DSSM": "DSSM--BG"}}


def latex_rows(frame, models, labels):
    """Body of a booktabs table: one block of three scales per model, best value in bold."""
    frame = frame[frame.model.isin(models)]
    best = {m: (frame[m].max() if m in HIGHER_IS_BETTER else frame[m].min()) for m in TABLE_METRICS}
    lines = []
    for model in models:
        block = frame[frame.model == model]
        for i, row in enumerate(block.itertuples()):
            cells = []
            for m in TABLE_METRICS:
                value = getattr(row, m)
                cells.append(f"\\textbf{{{value:.4f}}}" if np.isclose(value, best[m]) else f"{value:.4f}")
            name = f"\\multirow{{{len(block)}}}{{*}}{{{labels[model]}}}" if i == 0 else ""
            lines.append(f"{name} & {row.scale} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


for predictors in PREDICTOR_SETS:
    frame = overall[(overall.predictors == predictors) & overall.model.isin(TABLE_MODELS)]
    (OUT / f"table_timescales_{predictors}vars.tex").write_text(latex_rows(frame, TABLE_MODELS, TABLE_LABEL[predictors]))
    gru_rows = overall[(overall.predictors == predictors) & (overall.model == "GRU-BG")]
    (OUT / f"table_grubg_{predictors}vars.tex").write_text(latex_rows(gru_rows, ["GRU-BG"], TABLE_LABEL[predictors]))
    print(f"\n{predictors} predictors")
    display(frame.set_index(["model", "scale"])[TABLE_METRICS].dropna(how="all").round(4))

seasonal = metrics[(metrics.predictors == "8") & (metrics.scale == "Annual") & metrics.group.isin(SEASON_ORDER) & metrics.model.isin(TABLE_MODELS)]
# seasonal table: CRPS, AUC and COR of the Annual models, best value in bold
blocks = {metric: seasonal.astype({"model": str}).pivot(index="model", columns="group", values=metric).reindex(index=TABLE_MODELS, columns=list(SEASON_ORDER))
          for metric in ("CRPS", "AUC", "COR")}
lines = []
for model in TABLE_MODELS:
    cells = []
    for metric, block in blocks.items():
        best = block.min() if metric == "CRPS" else block.max()
        cells += [f"\\textbf{{{block.loc[model, s]:.3f}}}" if round(block.loc[model, s], 3) == round(best[s], 3) else f"{block.loc[model, s]:.3f}" for s in SEASON_ORDER]
    lines.append(LABEL[model].replace("–", "--") + " & " + " & ".join(cells) + "\\\\")
(OUT / "table_seasons.tex").write_text("\n".join(lines))
display(seasonal.pivot_table(index="model", columns="group", values=["CRPS", "AUC", "COR"], observed=True).round(3))

# moving-block bootstrap (blocks of 30 days) of the CRPS differences to DSSM-BG
BLOCK, N_BOOT = 30, 2000
boot_rng = np.random.default_rng(1)
n_days = len(observed)
starts = boot_rng.integers(0, n_days - BLOCK + 1, size=(N_BOOT, int(np.ceil(n_days / BLOCK))))
BOOT_INDEX = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :n_days]
boot_rows = []
for predictors in PREDICTOR_SETS:
    reference = DAILY[(predictors, "DSSM", "Annual")]["CRPS"]
    for model in ["GLM-BG", "RF-BG", "MLP-BG", "GRU-BG", "DSSM-GRU"]:
        difference = DAILY[(predictors, model, "Annual")]["CRPS"] - reference
        samples = difference[BOOT_INDEX].mean(axis=1)
        boot_rows.append({"predictors": predictors, "model": model, "dCRPS": difference.mean(),
                          "ci_low": np.quantile(samples, 0.025), "ci_high": np.quantile(samples, 0.975)})
bootstrap = pd.DataFrame(boot_rows)
bootstrap.to_excel(OUT / "bootstrap_crps_vs_dssm.xlsx", index=False)
display(bootstrap.round(4))


# ---- Numbers quoted in the text


crps_table = overall.astype({"model": str, "scale": str}).pivot_table(index="model", columns=["predictors", "scale"], values="CRPS")
for predictors in PREDICTOR_SETS:
    dssm = crps_table.loc["DSSM", (predictors, "Annual")]
    print(f"{predictors} predictors, Annual: DSSM CRPS {dssm:.3f} mm/day, lower than "
          + ", ".join(f"{LABEL[m]} by {100 * (1 - dssm / crps_table.loc[m, (predictors, 'Annual')]):.1f}%" for m in MODELS[:-1]))
print("CRPSS against the monthly climatology (Annual, 8 predictors): "
      + ", ".join(f"{LABEL[m]} {1 - crps_table.loc[m, ('8', 'Annual')] / climatology_crps.mean():.2f}" for m in MODELS))
print("CRPS from Annual to Month (8 predictors): "
      + ", ".join(f"{LABEL[m]} {crps_table.loc[m, ('8', 'Annual')]:.3f} -> {crps_table.loc[m, ('8', 'Month')]:.3f}" for m in MODELS))
for predictors in PREDICTOR_SETS:
    ranking = crps_table[(predictors, "Month")].sort_values()
    print(f"lowest CRPS at the Month scale, {predictors} predictors: " + ", ".join(f"{LABEL[m]} {v:.3f}" for m, v in ranking.iloc[:2].items()))

seasonal_crps = blocks["CRPS"]
print("DSSM CRPS lower than MLP–BG by season: "
      + ", ".join(f"{s} {100 * (1 - seasonal_crps.loc['DSSM', s] / seasonal_crps.loc['MLP-BG', s]):.1f}%" for s in SEASON_ORDER))

for predictors in PREDICTOR_SETS:
    ensemble_mean = ANNUAL[(predictors, "GLM-BG")]["rainfall"].astype(np.float64).mean(axis=1)
    print(f"GLM–BG, {predictors} predictors: ensemble mean above 20 mm on {(ensemble_mean > 20).sum()} test days, maximum {ensemble_mean.max():.0f} mm")
print(f"largest observation of the test period: {observed.max():.1f} mm on {test_dates[observed.argmax()].date()}")
for m in ("DSSM-GRU", "GRU-BG"):
    value = crps_table.loc[m, ("8", "Annual")]
    print(f"{LABEL[m]}: CRPS {value:.3f} (Annual) -> {crps_table.loc[m, ('8', 'Month')]:.3f} (Month); DSSM (BiGRU) lower by {100 * (1 - crps_table.loc['DSSM', ('8', 'Annual')] / value):.1f}%; "
          f"CRPSS {1 - value / climatology_crps.mean():.2f}; lower than MLP-BG by {100 * (1 - value / crps_table.loc['MLP-BG', ('8', 'Annual')]):.1f}%")
seasonal_all = metrics[(metrics.predictors == "8") & (metrics.scale == "Annual") & metrics.group.isin(SEASON_ORDER)].astype({"model": str}).pivot(index="model", columns="group", values="CRPS")[list(SEASON_ORDER)]
display(seasonal_all.round(3))
print("DSSM (BiGRU) CRPS lower than GRU-BG by season: " + ", ".join(f"{s} {100 * (1 - seasonal_all.loc['DSSM', s] / seasonal_all.loc['GRU-BG', s]):.1f}%" for s in SEASON_ORDER))
print("DSSM (BiGRU) CRPS lower than DSSM (GRU) by season: " + ", ".join(f"{s} {100 * (1 - seasonal_all.loc['DSSM', s] / seasonal_all.loc['DSSM-GRU', s]):.1f}%" for s in SEASON_ORDER))


# ---- Figures: one colour and one marker per model
PALETTE = {"GLM-BG": "#6F8FC6", "RF-BG": "#E39A6B", "MLP-BG": "#6DB77F", "GRU-BG": "#8C8C8C", "DSSM-GRU": "#5B2A86", "DSSM": "#C8102E"}
WIDTH = {model: 1.2 for model in PALETTE}
MARKERS = {"GLM-BG": "o", "RF-BG": "s", "MLP-BG": "^", "GRU-BG": "D", "DSSM-GRU": "P", "DSSM": "v"}
mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def save(figure, name, manuscript=True):
    """PNG in OUT; PDF in manuscript/Figures for a figure of the manuscript, else in OUT."""
    pdf = (FIGURES if manuscript else OUT) / f"{name}.pdf"
    figure.savefig(OUT / f"{name}.png", bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"written: {OUT / f'{name}.png'} and {pdf}")


def panel_label(axis, text):
    axis.text(-0.02, 1.02, text, transform=axis.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")


# ---- Monthly CRPS skill score against the monthly climatology, models trained at the Annual scale
figure, axis = plt.subplots(figsize=(7.2, 2.9))
monthly_clim = np.array([climatology_crps[months == m].mean() for m in range(1, 13)])
monthly_skill = pd.DataFrame({LABEL[m]: [1 - DAILY[("8", m, "Annual")]["CRPS"][months == k].mean() / monthly_clim[k - 1] for k in range(1, 13)] for m in TABLE_MODELS}, index=MONTH_LABELS)
for model in TABLE_MODELS:
    axis.plot(range(12), monthly_skill[LABEL[model]], marker=MARKERS[model], color=PALETTE[model], markersize=4,
              linewidth=WIDTH[model], label=LABEL[model], zorder=3 if model.startswith("DSSM") else 2)
values = monthly_skill.to_numpy()
axis.axhline(0, color="0.3", linestyle="--", linewidth=0.8, zorder=1)
axis.set_ylim(min(0.0, values.min()) - 0.03, 1.05 * values.max())
axis.set_xticks(range(12), MONTH_LABELS)
axis.set_ylabel("CRPSS vs monthly climatology")
axis.set_xlabel("calendar month (Annual scale)")
axis.legend(ncol=len(TABLE_MODELS), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
figure.tight_layout()
save(figure, "skill_scale_month")
monthly_skill.to_excel(OUT / "monthly_crpss_annual_8vars.xlsx")
display(monthly_skill.round(3))


# ---- VALUE indices of each member (relative bias to the observed index, %) and KS statistic of the wet-day amounts


RELATIVE_INDICES = ["R01", "SDII", "SD", "P95", "P98", "P99", "WW", "DW"]
day_numbers = test_dates.to_numpy().astype("datetime64[D]").astype(np.int64)


def indices(values, mask, reference_wet_amounts=None):
    """Indices for every column of values (days x simulations), on the days selected by mask."""
    values, days = values[mask], day_numbers[mask]
    consecutive = np.diff(days) == 1
    wet = values > TAU
    wet_amounts = np.where(wet, values, np.nan)
    previous, current = wet[:-1][consecutive], wet[1:][consecutive]
    result = {"R01": wet.mean(axis=0), "SDII": np.nanmean(wet_amounts, axis=0), "SD": values.std(axis=0, ddof=0),
              "P95": np.nanpercentile(wet_amounts, 95, axis=0), "P98": np.nanpercentile(wet_amounts, 98, axis=0),
              "P99": np.nanpercentile(wet_amounts, 99, axis=0),
              "WW": (previous & current).sum(axis=0) / previous.sum(axis=0),
              "DW": (~previous & current).sum(axis=0) / (~previous).sum(axis=0)}
    if reference_wet_amounts is not None:
        result["KS"] = np.array([ks_2samp(values[wet[:, k], k], reference_wet_amounts).statistic for k in range(values.shape[1])])
    return pd.DataFrame(result)


INDEX_GROUPS = [("Annual", np.ones(len(observed), dtype=bool))] + [(season, np.isin(months, PERIOD_MONTHS[season])) for season in SEASON_ORDER]
index_rows = []
observed_indices = {}
for group, mask in INDEX_GROUPS:
    reference = indices(observed[:, None], mask).iloc[0]
    observed_indices[group] = reference
    observed_wet = observed[mask][observed[mask] > TAU]
    for model in TABLE_MODELS + ABLATION_MODELS:
        simulated = indices(ANNUAL[("8", model)]["rainfall"].astype(np.float64), mask, observed_wet)
        for name in RELATIVE_INDICES:
            simulated[name] = 100 * (simulated[name] - reference[name]) / reference[name]
        simulated.insert(0, "member", np.arange(len(simulated)))
        simulated.insert(0, "model", model)
        simulated.insert(0, "group", group)
        index_rows.append(simulated)
index_bias = pd.concat(index_rows, ignore_index=True)
index_bias.to_excel(OUT / "value_indices_relative_bias.xlsx", index=False)
pd.DataFrame(observed_indices).T.to_excel(OUT / "value_indices_observed.xlsx")
display(pd.DataFrame(observed_indices).T.round(3))

summary = index_bias.groupby(["group", "model"], sort=False)[RELATIVE_INDICES + ["KS"]].median()
summary.to_excel(OUT / "value_indices_median.xlsx")
print("median relative bias (%) over the 1000 simulations; KS: median statistic")
display(summary.loc["Annual"].round(2))
display(summary.round(1))

# mean absolute median bias over the eight indices
mean_abs_bias = summary[RELATIVE_INDICES].abs().mean(axis=1).unstack("group").reindex(index=TABLE_MODELS + ABLATION_MODELS, columns=[g for g, _ in INDEX_GROUPS])
mean_abs_bias.to_excel(OUT / "value_indices_mean_abs_bias.xlsx")
print("mean absolute median relative bias over the eight indices (%)")
display(mean_abs_bias.round(1))
print("median KS statistic by group")
display(summary["KS"].unstack("group").reindex(index=TABLE_MODELS + ABLATION_MODELS, columns=[g for g, _ in INDEX_GROUPS]).round(3))
print(f"observed wet days in the test period: {wet_observed.sum()}, so the observed P99 rests on about {wet_observed.sum() / 100:.0f} values")

# LaTeX rows: median relative bias and median KS, whole test period
lines = []
for model in TABLE_MODELS:
    row = summary.loc[("Annual", model)]
    lines.append(LABEL[model].replace("–", "--") + " & " + " & ".join(f"{row[n]:+.1f}".replace("-", "$-$") for n in RELATIVE_INDICES)
                 + f" & {row['KS']:.3f}\\\\")
(OUT / "table_indices.tex").write_text("\n".join(lines))

INDEX_TITLE = {"R01": "R01", "SDII": "SDII", "SD": "SD", "P95": "P95", "P98": "P98", "P99": "P99", "WW": "WW", "DW": "DW",
               "KS": "KS"}


def boxes(axis, data, positions, model):
    parts = axis.boxplot(data, positions=positions, widths=0.7, whis=(5, 95), showfliers=False, patch_artist=True,
                         medianprops=dict(color="k", linewidth=0.8))
    for patch in parts["boxes"]:
        patch.set(facecolor=PALETTE[model], alpha=0.75, edgecolor=PALETTE[model], linewidth=0.6)
    for part in ("whiskers", "caps"):
        for line in parts[part]:
            line.set(color=PALETTE[model], linewidth=0.7)


figure, axes = plt.subplots(2, 5, figsize=(7.2, 3.8))
annual = index_bias[(index_bias.group == "Annual") & index_bias.model.isin(TABLE_MODELS)]
for axis, name in zip(axes.flat, RELATIVE_INDICES + ["KS"]):
    for k, model in enumerate(TABLE_MODELS):
        boxes(axis, [annual.loc[annual.model == model, name].to_numpy()], [k], model)
    if name != "KS":
        axis.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
    axis.set_xticks([])
    axis.set_title(INDEX_TITLE[name], fontsize=8)
    axis.tick_params(axis="y", labelsize=6.5)
axes[0, 0].set_ylabel("relative bias (%)")
axes[1, 0].set_ylabel("relative bias (%)")
axes[1, 3].set_ylabel("statistic")
axes[1, 4].axis("off")
for model in TABLE_MODELS:
    axes[1, 4].plot([], [], "s", color=PALETTE[model], markersize=7, alpha=0.75, label=LABEL[model])
axes[1, 4].legend(frameon=False, loc="center", fontsize=7.5)
figure.tight_layout()
save(figure, "value_indices", manuscript=False)

SEASON_INDICES = ["R01", "SDII", "SD", "P95", "P98", "P99", "WW", "DW", "KS"]
figure, axes = plt.subplots(3, 3, figsize=(7.2, 6.2))
season_colors = {"DJF": "#d9f0f7", "MAM": "#dff2df", "JJA": "#fbeccc", "SON": "#f7dfe4"}
for axis, name in zip(axes.flat, SEASON_INDICES):
    for s, season in enumerate(SEASON_ORDER):
        axis.axvspan(s * 6 - 0.6, s * 6 + 4.6, color=season_colors[season], zorder=0)
        block = index_bias[index_bias.group == season]
        for k, model in enumerate(TABLE_MODELS):
            boxes(axis, [block.loc[block.model == model, name].to_numpy()], [s * 6 + k], model)
    if name != "KS":
        axis.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
    axis.set_xticks([s * 6 + 2 for s in range(4)], list(SEASON_ORDER))
    axis.set_title(INDEX_TITLE[name], fontsize=8.5)
    axis.set_ylabel("statistic" if name == "KS" else "relative bias (%)", fontsize=7.5)
    axis.grid(axis="x", visible=False)
handles = [mpl.patches.Patch(facecolor=PALETTE[m], alpha=0.75, label=LABEL[m]) for m in TABLE_MODELS]
figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
figure.tight_layout(rect=(0, 0, 1, 0.97))
save(figure, "value_indices_seasonal", manuscript=False)


# ---- Heavy precipitation: twCRPSS and frequency ratio by threshold (thresholds exceeded on at least 20 test days)


SWEEP = np.arange(1.0, 31.0, 1.0)
MIN_EVENTS = 20
event_counts = np.array([int((observed > v).sum()) for v in SWEEP])
climatology_tw = np.array([twcrps_ensemble_fair(observed, climatology, v, TAU).mean() for v in SWEEP])
sweep_rows = []
for m in TABLE_MODELS + ABLATION_MODELS:
    ensemble = ANNUAL[("8", m)]["rainfall"].astype(np.float64)
    for level, count, reference in zip(SWEEP, event_counts, climatology_tw):
        probability = (ensemble > level).mean(axis=1)
        sweep_rows.append({"model": m, "threshold_mm": level, "events": count,
                           "twCRPSS": 1 - twcrps_ensemble_fair(observed, ensemble, level, TAU).mean() / reference,
                           "frequency_ratio": (ensemble > level).mean() / max((observed > level).mean(), 1e-12),
                           "AUC": auc_score(observed, probability, level) if 0 < count < len(observed) else np.nan})
sweep = pd.DataFrame(sweep_rows)
sweep.to_excel(OUT / "extreme_threshold_sweep.xlsx", index=False)
usable = sweep[sweep.events >= MIN_EVENTS]
print(f"thresholds with at least {MIN_EVENTS} events: {usable.threshold_mm.min():.0f}-{usable.threshold_mm.max():.0f} mm")
display(usable.groupby("model", sort=False)[["twCRPSS", "frequency_ratio", "AUC"]].mean().round(4))
print(f"GLM–BG twCRPSS below zero from {usable[(usable.model == 'GLM-BG') & (usable.twCRPSS < 0)].threshold_mm.min():.0f} mm")
display(usable[usable.threshold_mm >= 20].pivot(index="threshold_mm", columns="model", values="twCRPSS")[TABLE_MODELS].round(3))
below_20 = usable[(usable.threshold_mm < 20) & usable.model.isin(["RF-BG", "DSSM"])]
print(f"frequency ratio below 20 mm for RF–BG and DSSM–BG: {below_20.frequency_ratio.min():.2f} to {below_20.frequency_ratio.max():.2f}")
for m in TABLE_MODELS:
    block = usable[(usable.threshold_mm < 20) & (usable.model == m)]
    print(f"frequency ratio below 20 mm, {LABEL[m]}: {block.frequency_ratio.min():.2f} to {block.frequency_ratio.max():.2f}")

figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
for column, axis in zip(("twCRPSS", "frequency_ratio"), axes):
    for m in TABLE_MODELS:
        block = usable[usable.model == m]
        axis.plot(block.threshold_mm, block[column], color=PALETTE[m], marker=MARKERS[m], markersize=2.8, linewidth=1.1, label=LABEL[m])
    axis.set_xlabel("threshold v (mm/day)"); axis.set_xlim(usable.threshold_mm.min() - 0.5, usable.threshold_mm.max() + 0.5)
axes[0].set_ylabel("twCRPSS"); axes[1].set_ylabel("forecast / observed frequency")
axes[0].axhline(0, color="0.3", linestyle="--", linewidth=0.8)
axes[1].axhline(1, color="0.3", linestyle="--", linewidth=0.8)
axes[0].legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.6)
for axis, text in zip(axes, "ab"):
    axis.text(-0.075, 1.04, f"({text})", transform=axis.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")
figure.tight_layout()
save(figure, "extremes_sweep")

# predictive distributions on the test days above the 95th, 98th and 99th percentiles of the wet days of 1980-2017
EXTREME_PANELS = [(95, ["DSSM"]), (98, TABLE_MODELS), (99, TABLE_MODELS)]
extreme_rows, extreme_days = [], {}
for percentile, shown in EXTREME_PANELS:
    level = float(np.quantile(reference_wet, percentile / 100))
    days = np.flatnonzero(observed > level)
    extreme_days[percentile] = (level, days)
    for m in shown:
        members = ANNUAL[("8", m)]["rainfall"][days].astype(np.float64)
        probability = ANNUAL[("8", m)]["wet_probability"][days]
        probability = probability if probability.ndim == 1 else probability.mean(axis=1)
        extreme_rows.append({"percentile": percentile, "level_mm": level, "days": len(days), "model": m,
                             "obs_above_member_q90": np.mean(observed[days] > np.quantile(members, 0.90, axis=1)),
                             "obs_above_member_q95": np.mean(observed[days] > np.quantile(members, 0.95, axis=1)),
                             "median_P_exceed_level": np.median((members > level).mean(axis=1)),
                             "mean_wet_probability": probability.mean()})
extremes_pct = pd.DataFrame(extreme_rows)
extremes_pct.to_excel(OUT / "extremes_percentile_days_annual_8vars.xlsx", index=False)
display(extremes_pct.round(3))

level99, days99 = extreme_days[99]
wet_q99 = pd.DataFrame({LABEL[m]: (lambda p: p if p.ndim == 1 else p.mean(axis=1))(ANNUAL[("8", m)]["wet_probability"][days99])
                        for m in TABLE_MODELS}, index=[d.strftime("%Y-%m-%d") for d in test_dates[days99]])
wet_q99.insert(0, "observed_mm", observed[days99])
display(wet_q99.round(2))

# CAPE on 3 June 2021
reference_cape = cape.iloc[reference_rows]
summer_reference = reference_cape[reference_cape.index.month.isin(PERIOD_MONTHS["JJA"])]
print(f"maximum CAPE on 2021-06-03: {cape.loc[pd.Timestamp('2021-06-03')]:.0f} J/kg; "
      f"90th percentile of summer days in 1980-2017: {np.quantile(summer_reference, 0.90):.0f} J/kg")
summer99 = days99[np.isin(months[days99], PERIOD_MONTHS["JJA"])]
summer_dates = [d.strftime("%Y-%m-%d") for d in test_dates[summer99]]
figure, axes = plt.subplots(3, 1, figsize=(7.2, 6.6))
for axis, (percentile, shown), text in zip(axes, EXTREME_PANELS, "abc"):
    level, days = extreme_days[percentile]
    step = len(shown) + 1
    for i, day in enumerate(days):
        for k, m in enumerate(shown):
            parts = axis.boxplot([ANNUAL[("8", m)]["rainfall"][day]], positions=[i * step + k], widths=0.75, whis=(5, 95),
                                 showfliers=False, patch_artist=True, medianprops=dict(color="k", linewidth=0.8))
            parts["boxes"][0].set(facecolor=PALETTE[m], alpha=0.75, edgecolor=PALETTE[m], linewidth=0.6)
            for part in ("whiskers", "caps"):
                for line in parts[part]:
                    line.set(color=PALETTE[m], linewidth=0.7)
        axis.plot(i * step + shown.index("DSSM"), observed[day], "D", color="#d2601a", markersize=4.5, zorder=5)
    axis.axhline(level, color="0.3", linestyle="--", linewidth=0.8, label=f"q{percentile} = {level:.1f} mm")
    axis.set_xticks([i * step + (len(shown) - 1) / 2 for i in range(len(days))],
                    [test_dates[d].strftime("%Y-%m-%d") for d in days], rotation=55, ha="right",
                    fontsize=6.5 if len(days) > 20 else 7.5)
    axis.set_xlim(-1, len(days) * step - 1)
    axis.set_ylim(-2, 45)
    axis.set_ylabel("precipitation (mm/day)")
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, loc="upper right", fontsize=7)
    axis.text(-0.075, 1.04, f"({text})", transform=axis.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")
handles = [mpl.patches.Patch(facecolor=PALETTE[m], alpha=0.75, label=LABEL[m]) for m in TABLE_MODELS]
handles.append(mpl.lines.Line2D([], [], marker="D", color="#d2601a", linestyle="", markersize=4.5, label="observed"))
figure.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=7.5)
figure.tight_layout(rect=(0, 0, 1, 0.975))
save(figure, "extremes_top_days")


# ---- Example: the four-month period with the largest CRPS reduction of DSSM-BG against the best baseline


dssm_path = catalogue[(catalogue.predictors == "8") & (catalogue.model == "DSSM") & (catalogue.scale == "Annual")].path.iloc[0]
mlp_path = catalogue[(catalogue.predictors == "8") & (catalogue.model == EXAMPLE_BASELINE) & (catalogue.scale == "Annual")].path.iloc[0]
with np.load(dssm_path) as saved:
    dssm_full = {k: saved[k].astype(np.float64) for k in ("rainfall", "wet_probability", "mu", "alpha")}
with np.load(mlp_path) as saved:
    mlp_full = {k: saved[k].astype(np.float64) for k in ("rainfall", "wet_probability", "mu", "alpha")}

daily_crps = pd.DataFrame({m: DAILY[("8", m, "Annual")]["CRPS"] for m in MODELS}, index=test_dates)
period_rows = []
for start in pd.date_range(test_dates[0], test_dates[-1] - pd.DateOffset(months=4) + pd.Timedelta(days=1), freq="MS"):
    end = start + pd.DateOffset(months=4) - pd.Timedelta(days=1)
    scores = daily_crps.loc[start:end].mean()
    baselines = scores.drop("DSSM")
    period_rows.append({"start": start, "end": end, **scores.to_dict(), "best_baseline": baselines.idxmin(),
                        "gain_vs_best_baseline": 1 - scores["DSSM"] / baselines.min(),
                        "gain_vs_MLP-BG": 1 - scores["DSSM"] / scores["MLP-BG"]})
periods = pd.DataFrame(period_rows).sort_values("gain_vs_best_baseline", ascending=False)
periods.to_excel(OUT / "example_periods_annual_8vars.xlsx", index=False)
display(periods.head(5).round(3))
best_start, best_end = periods.iloc[0].start, periods.iloc[0].end
EXAMPLES = {f"example_{best_start.year}": (best_start, best_end)}


def plot_example(window, name):
    """Observations, ensembles and Bernoulli-Gamma parameters of MLP-BG and the DSSM over one period."""
    days = test_dates[window]
    line = {EXAMPLE_BASELINE: "#3B7A4B", "DSSM": "#8E0B20"}  # darker shades of PALETTE for the lines
    dssm = {k: v[window] for k, v in dssm_full.items()}
    mlp = {k: v[window] for k, v in mlp_full.items()}
    figure, axes = plt.subplots(5, 1, figsize=(7.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.25, 0.9, 0.9, 0.9]})
    top = 1.05 * max(observed[window].max(), *(np.quantile(v["rainfall"], 0.95, axis=1).max() for v in (mlp, dssm)))
    # (a, b) the ensembles, one panel per model
    for axis, (m, values), text in zip(axes[:2], ((EXAMPLE_BASELINE, mlp), ("DSSM", dssm)), "ab"):
        q05, q75, q95 = np.quantile(values["rainfall"], [0.05, 0.75, 0.95], axis=1)
        axis.fill_between(days, q05, q95, color=PALETTE[m], alpha=0.3, linewidth=0, step="mid", zorder=1, label="5–95% of the members")
        axis.bar(days, observed[window], width=0.75, color="0.3", zorder=2, label="observed")
        axis.plot(days, q75, color=line[m], linewidth=1.0, drawstyle="steps-mid", zorder=3, label="75th percentile of the members")
        axis.set_ylim(0, top)
        axis.set_title(f"({text}) {LABEL[m]}: precipitation (mm)", loc="left", fontsize=8.5)
        handles, labels = axis.get_legend_handles_labels()
        order = [labels.index(k) for k in ("observed", "5–95% of the members", "75th percentile of the members")]
        axis.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3, fontsize=7, frameon=False,
                    loc="lower right", bbox_to_anchor=(1.0, 0.98), borderaxespad=0, handlelength=1.6, columnspacing=1.2)
    # (c-e) the parameters
    titles = ("(c) Wet-day probability $\\pi_t$", "(d) Mean wet-day excess $\\mu_t$ (mm)", "(e) Shape $\\alpha_t$")
    for axis, key, title in zip(axes[2:], ("wet_probability", "mu", "alpha"), titles):
        q05, q50, q95 = np.quantile(dssm[key], [0.05, 0.5, 0.95], axis=1)
        axis.fill_between(days, q05, q95, color=PALETTE["DSSM"], alpha=0.25, linewidth=0, label=f"{LABEL['DSSM']}, 5–95% of the members")
        axis.plot(days, q50, color=line["DSSM"], linewidth=1.1, label=f"{LABEL['DSSM']}, median of the members")
        axis.plot(days, mlp[key], color=line[EXAMPLE_BASELINE], linewidth=0.9, label="MLP–BG")
        axis.set_title(title, loc="left", fontsize=8.5)
    handles, labels = axes[2].get_legend_handles_labels()
    order = [labels.index(k) for k in (LABEL[EXAMPLE_BASELINE], f"{LABEL['DSSM']}, median of the members", f"{LABEL['DSSM']}, 5–95% of the members")]
    axes[2].legend([handles[i] for i in order], [labels[i] for i in order], ncol=3, fontsize=7, frameon=False,
                   loc="lower right", bbox_to_anchor=(1.0, 0.98), borderaxespad=0, handlelength=1.6, columnspacing=1.2)
    axes[-1].xaxis.set_major_locator(mpl.dates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_minor_locator(mpl.dates.DayLocator(bymonthday=(8, 15, 22)))
    axes[-1].set_xlim(days[0] - pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=1))
    figure.tight_layout(h_pad=0.6)
    save(figure, name)

example_rows = []
for name, (start, end) in EXAMPLES.items():
    window = (test_dates >= start) & (test_dates <= end)
    plot_example(window, name)
    wet, dry = window & wet_observed, window & ~wet_observed
    example_rows.append({"figure": name, "start": start.date(), "end": end.date(),
                         **{f"CRPS {m}": daily_crps[window][m].mean() for m in MODELS},
                         "CRPS dry days, MLP-BG": daily_crps[dry][EXAMPLE_BASELINE].mean(), "CRPS dry days, DSSM": daily_crps[dry]["DSSM"].mean(),
                         "CRPS wet days, MLP-BG": daily_crps[wet][EXAMPLE_BASELINE].mean(), "CRPS wet days, DSSM": daily_crps[wet]["DSSM"].mean(),
                         "median mu on dry days, MLP-BG": np.median(mlp_full["mu"][dry]),
                         "median mu on dry days, DSSM": np.median(np.median(dssm_full["mu"][dry], axis=1)),
                         "mean alpha on wet days, MLP-BG": mlp_full["alpha"][wet].mean(),
                         "mean alpha on wet days, DSSM": dssm_full["alpha"][wet].mean()})
display(pd.DataFrame(example_rows).set_index("figure").T.round(3))


# numbers quoted in the text
width_pi = np.diff(np.quantile(dssm_full["wet_probability"], [0.05, 0.95], axis=1), axis=0)[0]
median_pi = np.median(dssm_full["wet_probability"], axis=1)
parameter_summary = {
    "DSSM pi 90% band width, median over days": np.median(width_pi),
    "DSSM alpha 90% band width, median over days": np.median(np.diff(np.quantile(dssm_full["alpha"], [0.05, 0.95], axis=1), axis=0)[0]),
    "DSSM pi band width, days with median pi < 0.1": np.median(width_pi[median_pi < 0.1]),
    "DSSM pi band width, days with median pi in [0.3, 0.7]": np.median(width_pi[(median_pi >= 0.3) & (median_pi <= 0.7)]),
    "median mu on observed dry days, DSSM": np.median(np.median(dssm_full["mu"], axis=1)[~wet_observed]),
    "median mu on observed dry days, MLP-BG": np.median(mlp_full["mu"][~wet_observed]),
    "mean alpha on observed wet days, DSSM": dssm_full["alpha"].mean(axis=1)[wet_observed].mean(),
    "mean alpha on observed wet days, MLP-BG": mlp_full["alpha"][wet_observed].mean(),
}
display(pd.Series(parameter_summary).round(3))

window = (test_dates >= best_start) & (test_dates <= best_end)
mlp_pi, dssm_pi = mlp_full["wet_probability"], dssm_full["wet_probability"].mean(axis=1)
dssm_mu_median = np.median(dssm_full["mu"], axis=1)
gain = (daily_crps[EXAMPLE_BASELINE] - daily_crps["DSSM"]).to_numpy()
dry_window = window & ~wet_observed
uncertain = dry_window & (mlp_pi > 0.3)
band = lambda values: np.diff(np.quantile(values, [0.05, 0.95], axis=1), axis=0)[0]
print("median 5-95% band over the members, whole test period: "
      + ", ".join(f"{k} {np.median(band(dssm_full[k])):.3f}" for k in ("wet_probability", "mu", "alpha")))
print(f"dry days of the period with MLP–BG pi > 0.3: {uncertain.sum()}, "
      f"share of the dry-day CRPS gain {gain[uncertain].sum() / gain[dry_window].sum():.2f}; "
      f"mean pi DSSM {dssm_pi[uncertain].mean():.2f}, MLP–BG {mlp_pi[uncertain].mean():.2f}; "
      f"median mu DSSM {np.median(dssm_mu_median[uncertain]):.1f} mm, MLP–BG {np.median(mlp_full['mu'][uncertain]):.1f} mm")
# longest dry spell of the example period
runs, start_day = [], None
for day, dry in zip(test_dates[window], ~wet_observed[window]):
    if dry and start_day is None:
        start_day = day
    if not dry and start_day is not None:
        runs.append((start_day, day - pd.Timedelta(days=1))); start_day = None
if start_day is not None:
    runs.append((start_day, test_dates[window][-1]))
spell_start, spell_end = max(runs, key=lambda r: r[1] - r[0])
spell = (test_dates >= spell_start) & (test_dates <= spell_end)
spell_dry = spell & ~wet_observed
print(f"longest dry spell of the period, {spell_start.date()} to {spell_end.date()}: {spell.sum()} days, {(spell & wet_observed).sum()} wet; on the dry days: "
      f"max pi DSSM {dssm_pi[spell_dry].max():.3f}, MLP–BG {mlp_pi[spell_dry].max():.3f}; "
      f"mean CRPS DSSM {daily_crps['DSSM'][spell_dry].mean():.3f}, MLP–BG {daily_crps[EXAMPLE_BASELINE][spell_dry].mean():.3f}; "
      f"mu MLP–BG {mlp_full['mu'][spell_dry].min():.1f}-{mlp_full['mu'][spell_dry].max():.1f} mm, DSSM median below {dssm_mu_median[spell_dry].max():.2f} mm")
wet_window = window & wet_observed
print(f"wet days of the period: mean pi DSSM {dssm_pi[wet_window].mean():.2f}, MLP–BG {mlp_pi[wet_window].mean():.2f}")
print(f"mean of 1/sqrt(alpha) on observed wet days: DSSM {(1 / np.sqrt(dssm_full['alpha'])).mean(axis=1)[wet_observed].mean():.2f}, "
      f"MLP–BG {(1 / np.sqrt(mlp_full['alpha']))[wet_observed].mean():.2f}")
print("whole test period, CRPS on observed dry days: "
      + ", ".join(f"{LABEL[m]} {daily_crps[m][~wet_observed].mean():.3f}" for m in ("DSSM", EXAMPLE_BASELINE))
      + "; on observed wet days: " + ", ".join(f"{LABEL[m]} {daily_crps[m][wet_observed].mean():.2f}" for m in ("DSSM", EXAMPLE_BASELINE)))
for m in (EXAMPLE_BASELINE, "DSSM"):
    print(f"{LABEL[m]}: ensemble median zero on {100 * (np.median(ANNUAL[('8', m)]['rainfall'], axis=1) == 0).mean():.0f}% of the test days")
print("Done. Tables and figures in", OUT, "and", FIGURES)
