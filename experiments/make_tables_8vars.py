"""Run folders and scores of the models, read from the scores.xlsx and point_metrics_overall.xlsx of each run."""

import pandas as pd

from paths import DSSM_RUNS, GLM_RUNS, MLP_RUNS, RF_RUN, run_of

MODELS = ["GLM-BG", "RF-BG", "MLP-BG", "DSSM-GRU", "DSSM"]
SCALES = {"Annual": "Annual", "Season": "SeasonalConcat", "Month": "MonthConcat"}
PREFIX = {"GLM-BG": "GLM", "RF-BG": "RF", "MLP-BG": "MLP_BG", "DSSM-GRU": "DSSM", "DSSM": "DSSM"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# column, header, decimals, rule ("low", "high" or "zero")
METRICS = [("CRPS", r"CRPS$\downarrow$", 3, "low"), ("Brier", r"Brier$\downarrow$", 3, "low"),
           ("AUC", r"AUC$\uparrow$", 3, "high"), ("Spearman", r"$\rho_s\uparrow$", 3, "high"),
           ("NLL", r"NLL$\downarrow$", 3, "low"),
           ("RMSE", r"RMSE$\downarrow$", 3, "low"), ("COR", r"COR$\uparrow$", 3, "high")]


def runs_0606(station):
    """Folders of the five models at one station (ERA5 06-06 UTC)."""
    dssm = run_of(DSSM_RUNS, station) / station
    return {"GLM-BG": run_of(GLM_RUNS, station) / station, "RF-BG": RF_RUN / station,
            "MLP-BG": run_of(MLP_RUNS, station) / station, "DSSM-GRU": dssm / "DSSM_GRU", "DSSM": dssm / "DSSM_BiGRU"}


def read_model(folder, model):
    """Rows (scale, period) with every metric of one model of one run folder."""
    scores = pd.read_excel(folder / "scores.xlsx")
    point = pd.read_excel(folder / "point_metrics_overall.xlsx")
    name = "variant" if "variant" in scores else "model"
    point_name = "variant" if "variant" in point else "model"
    crps = "CRPS" if "CRPS" in scores else "CRPS_ensemble_fair"
    rows = []
    for scale, folder_scale in SCALES.items():
        key = f"{PREFIX[model]}_{folder_scale}"
        block = scores[scores[name] == key].drop_duplicates("period").set_index("period")
        overall = point[(point[point_name] == key) & (point.group == "Overall")].iloc[0]
        for period, row in block.iterrows():
            period = str(period).zfill(2) if str(period).isdigit() else str(period)
            label = MONTHS[int(period) - 1] if period.isdigit() else period
            record = {"model": model, "scale": scale, "period": label, "CRPS": row[crps],
                      **{m: row[m] for m, *_ in METRICS if m in row.index and m != "CRPS"}}
            if label == "Annual":
                record.update({m: overall[m] for m in ("Delta_mu", "Delta_sigma", "MAE", "RMSE", "COR", "NLL")})
            rows.append(record)
    return rows
