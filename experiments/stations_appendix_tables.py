"""Tables 5 and 6: scores of the four models at the Annual scale and seasonal CRPS, at the nine stations."""

from pathlib import Path

import pandas as pd

import make_tables_8vars as tables
import paths
from stations_summary import STATIONS

OUT = paths.RESULTS / "model_comparison_stations"
MODELS = ["GLM-BG", "RF-BG", "MLP-BG", "DSSM"]
BASELINES = ["GLM-BG", "RF-BG", "MLP-BG"]
TEX = {"GLM-BG": "GLM--BG", "RF-BG": "RF--BG", "MLP-BG": "MLP--BG", "DSSM-GRU": "Causal context", "DSSM": "DSSM--BG"}
SEASONS = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)", "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}
# metric, header, decimals, "low" or "high" is better
METRICS = [("NLL", "NLL$\\downarrow$", 4, "low"), ("Brier", "Brier$\\downarrow$", 4, "low"),
           ("AUC", "AUC$\\uparrow$", 4, "high"), ("COR", "COR$\\uparrow$", 4, "high"),
           ("RMSE", "RMSE$\\downarrow$", 4, "low")]


def read_scores():
    rows = []
    for station in STATIONS:
        for model, folder in tables.runs_0606(station).items():
            rows += [{"station": station, **r} for r in tables.read_model(Path(folder), model)]
    return pd.DataFrame(rows)


def bold_best(block, decimals, rule, rows=None):
    """Formatted cells; the best printed value of each column (among `rows`) is bold."""
    printed = block.round(decimals)
    pool = printed if rows is None else printed.loc[rows]
    best = pool.min(axis=0) if rule == "low" else pool.max(axis=0)
    return printed.apply(lambda col: [f"\\textbf{{{v:.{decimals}f}}}" if v == best[col.name] else f"{v:.{decimals}f}"
                                      for v in col]).set_axis(block.index)


def metrics_table(frame):
    annual = frame[(frame.period == "Annual") & (frame.scale == "Annual")]
    lines = []
    for k, (metric, header, decimals, rule) in enumerate(METRICS):
        block = annual.pivot_table(index="model", columns="station", values=metric).reindex(MODELS)[list(STATIONS)]
        block["Mean"] = block.mean(axis=1)
        cells = bold_best(block, decimals, rule)
        if k:
            lines.append("\\midrule")
        for i, model in enumerate(MODELS):
            name = f"\\multirow{{{len(MODELS)}}}{{*}}{{{header}}}" if i == 0 else ""
            lines.append(f"{name} & {TEX[model]} & " + " & ".join(cells.loc[model]) + "\\\\")
    return "\n".join(lines) + "\n"


def seasons_table(frame):
    """Seasonal CRPS of the best baseline and of DSSM-BG at each station; no mean, as the best baseline varies."""
    seasonal = frame[frame.period.isin(SEASONS)]
    columns = ["best", "DSSM"]
    short = {"GLM-BG": "GLM", "RF-BG": "RF", "MLP-BG": "MLP"}
    cells, summary = {}, {}
    for season in SEASONS:
        data = seasonal[seasonal.period == season].pivot_table(index=["model", "scale"], columns="station",
                                                               values="CRPS")[list(STATIONS)]
        block = data.xs("Annual", level="scale").reindex(MODELS + ["DSSM-GRU"])
        baselines = data.loc[BASELINES]
        block.loc["best"] = baselines.min(axis=0)
        winner = baselines.idxmin(axis=0)
        summary[season] = (block, winner)
        printed = block.loc[columns].round(3)
        for station in printed.columns:
            low = printed[station].min()
            row = [f"\\textbf{{{v:.3f}}}" if v == low else f"{v:.3f}" for v in printed[station]]
            model, scale = winner[station]
            row.insert(1, f"{{\\scriptsize {short[model]}-{scale[0]}}}")
            cells.setdefault(station, []).extend(row)
    lines = []
    for station, row in cells.items():
        name = STATIONS[station][0].replace("é", "\\'e")
        lines.append(f"{name} & " + " & ".join(row) + "\\\\")
    return "\n".join(lines) + "\n", summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = read_scores()
    (OUT / "table_stations_metrics.tex").write_text(metrics_table(frame))
    body, summary = seasons_table(frame)
    (OUT / "table_stations_seasons.tex").write_text(body)

    every = frame[frame.period == "Annual"]
    for metric, _, _, rule in METRICS:
        values = every.pivot_table(index=["model", "scale"], columns="station", values=metric)
        best = values.idxmin() if rule == "low" else values.idxmax()
        annual = values.xs("Annual", level="scale")
        main = values.loc[[m for m in values.index if m[0] in MODELS]]
        best_main = main.idxmin() if rule == "low" else main.idxmax()
        annual_main = annual.loc[MODELS]
        best_annual_main = annual_main.idxmin() if rule == "low" else annual_main.idxmax()
        print(f"{metric:6s} best at any scale, not DSSM-BG Annual (with the causal variant): {dict(best[best != ('DSSM', 'Annual')])}")
        print(f"       best at any scale among DSSM-BG and the baselines, not DSSM-BG Annual: {dict(best_main[best_main != ('DSSM', 'Annual')])}")
        print(f"       best among the Annual models (DSSM-BG and baselines), not DSSM-BG: {dict(best_annual_main[best_annual_main != 'DSSM'])}")
    wins = cases = 0
    for season, (block, winner) in summary.items():
        gap = 100 * (block.loc["best"] / block.loc["DSSM"] - 1)
        causal = 100 * (block.loc["DSSM-GRU"] / block.loc["DSSM"] - 1)
        wins += (gap > 0).sum()
        cases += len(STATIONS)
        print(season, "DSSM-BG worse than best baseline at", {s: (round(g, 1), winner[s]) for s, g in gap.items() if g <= 0},
              "| causal variant better at", {s: round(g, 1) for s, g in causal.items() if g <= 0},
              f"| max gap {gap.max():.1f}%")
    print(f"DSSM-BG Annual beats the best baseline in {wins} of {cases} station-seasons")


if __name__ == "__main__":
    main()
