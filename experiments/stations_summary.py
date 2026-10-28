"""Table 1 (stations) and Table 4 (CRPS at the three training scales) at the nine stations, with the skill scores."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from precipitation_metrics import crps_ensemble_fair

RESULTS = paths.RESULTS
DATA = paths.DATA
OUT = RESULTS / "model_comparison_stations"
ECAD_STATIONS = paths.ECAD_STATIONS
TAU = 1.0

# slug: (label, STAID), from north-west to south-east
STATIONS = {"brest_guipavas": ("Brest", 434), "lille_lesquin": ("Lille", 737), "tours": ("Tours", 2190),
            "strasbourg_entzheim": ("Strasbourg", 323), "clermont_ferrand": ("Clermont-Ferrand", 750),
            "bordeaux_merignac": ("Bordeaux", 34), "montelimar": ("Montélimar", 786),
            "tarbes_ossun": ("Tarbes", 756), "perpignan": ("Perpignan", 36)}
# "b-DSSM": DSSM-BG; "u-DSSM": its causal-context variant (ablation)
MODELS = ["GLM-BG", "RF-BG", "MLP-BG", "u-DSSM", "b-DSSM"]
MAIN = ["GLM-BG", "RF-BG", "MLP-BG", "b-DSSM"]
LABEL = {"GLM-BG": "GLM–BG", "RF-BG": "RF–BG", "MLP-BG": "MLP–BG", "u-DSSM": "causal context", "b-DSSM": "DSSM–BG"}
TEX = {"GLM-BG": "GLM--BG", "RF-BG": "RF--BG", "MLP-BG": "MLP--BG", "u-DSSM": "Causal context", "b-DSSM": "DSSM--BG"}
SCALES = {"Annual": "Annual", "SeasonalConcat": "Season", "MonthConcat": "Month"}
# cited below the names of the baselines in the CRPS table
CITE = {"GLM-BG": "Legasa_PP_2023", "RF-BG": "Legasa_2022_RF", "MLP-BG": "Alex_NN_BG_2008"}


def read_crps():
    """CRPS over the whole test period of every model, scale and station."""
    rows = []
    for family, runs in [("GLM", paths.GLM_RUNS.values()), ("RF", [paths.RF_RUN])]:
        for run in runs:
            d = pd.read_excel(run / "all_stations_scores.xlsx")
            d = d[(d.period == "Annual") & (d.family == family)]
            rows += [(r.family + "-BG", r.time_scale, r.station, r.CRPS_ensemble_fair) for r in d.itertuples()]
    for run in paths.MLP_RUNS.values():
        d = pd.read_excel(run / "all_stations_scores.xlsx")
        d = d[(d.period == "Annual") & (d.architecture == "MLP_BG")]
        rows += [("MLP-BG", r.time_scale, r.station, r.CRPS_ensemble_fair) for r in d.itertuples()]
    for run in paths.DSSM_RUNS.values():
        d = pd.read_excel(run / "all_stations_scores.xlsx")
        d = d[d.period == "Annual"]
        name = {"DSSM_GRU": "u-DSSM", "DSSM_BiGRU": "b-DSSM"}
        rows += [(name[r.architecture], r.variant.replace("DSSM_", ""), r.station, r.CRPS) for r in d.itertuples()]
    frame = pd.DataFrame(rows, columns=["model", "scale", "station", "CRPS"])
    frame["scale"] = frame.scale.map(lambda s: SCALES.get(s, s))
    frame = frame[frame.station.isin(STATIONS)]
    assert not frame.duplicated(["model", "scale", "station"]).any()
    assert len(frame) == len(MODELS) * len(SCALES) * len(STATIONS)
    return frame


def precipitation(station):
    raw = np.load(DATA / f"ecad_rr_{station}_1980_2025.npy").astype(np.float64)
    return pd.Series(raw, index=pd.date_range("1980-01-01", "2025-12-31"))


def climatology_crps(station):
    """Fair CRPS of the monthly climatology, built as in tours_figures.py."""
    series = precipitation(station)
    y = np.where(series.to_numpy() <= TAU, 0.0, series.to_numpy())
    dates = series.index
    reference = (dates >= "1980-01-01") & (dates <= "2017-12-31")
    test = (dates >= "2018-01-01") & (dates <= "2025-12-31")
    rng = np.random.default_rng(0)
    months = dates[reference].month.to_numpy()
    draw = {m: rng.choice(y[reference][months == m], size=1000, replace=True) for m in range(1, 13)}
    ensemble = np.stack([draw[m] for m in dates[test].month])
    return float(crps_ensemble_fair(y[test], ensemble, TAU).mean())


def dms(value):
    sign = -1.0 if value.strip()[0] == "-" else 1.0
    d, m, s = map(float, value.strip()[1:].split(":"))
    return sign * (d + m / 60 + s / 3600)


def station_info():
    lines = ECAD_STATIONS.read_text(encoding="latin-1").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("STAID"))
    meta = {int(p[0]): p for p in (list(map(str.strip, line.split(","))) for line in lines[header + 1:]) if len(p) >= 6}
    rows = []
    for station, (label, staid) in STATIONS.items():
        series = precipitation(station)
        y = series.to_numpy()
        rows.append({"station": station, "label": label, "STAID": staid, "lat": dms(meta[staid][3]),
                     "lon": dms(meta[staid][4]), "elevation": int(meta[staid][5]),
                     "annual_mm": series.groupby(series.index.year).sum().mean(),
                     "wet_percent": 100 * (y > TAU).mean(), "sdii_mm": y[y > TAU].mean()})
    return pd.DataFrame(rows)


def info_table(info):
    lines = []
    for r in info.itertuples():
        lat = f"{abs(r.lat):.2f}$^\\circ$N"
        lon = f"{abs(r.lon):.2f}$^\\circ${'W' if r.lon < 0 else 'E'}"
        lines.append(f"{r.label} & {r.STAID} & {lat} & {lon} & {r.elevation} & {r.annual_mm:.0f} & "
                     f"{r.wet_percent:.1f}\\\\")
    return "\n".join(lines) + "\n"


def crps_table(crps):
    """Rows: model and scale; columns: stations and mean; bold: lowest CRPS of each station."""
    best = crps.min(axis=0)
    lines = []
    for i, model in enumerate(MAIN):
        if i:
            lines.append("\\midrule")
        for k, scale in enumerate(SCALES.values()):
            cells = [f"\\textbf{{{v:.3f}}}" if np.isclose(v, best[c]) else f"{v:.3f}"
                     for c, v in crps.loc[(model, scale)].items()]
            label = TEX[model] if model not in CITE else f"\\shortstack[l]{{{TEX[model]}\\\\{{\\scriptsize\\citep{{{CITE[model]}}}}}}}"
            name = f"\\multirow{{3}}{{*}}{{{label}}}" if k == 0 else ""
            lines.append(f"{name} & {scale} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines) + "\n"


def heatmap(relative, path):
    """Relative CRPS (%) to the best model and scale of each station."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    norm = mpl.colors.Normalize(vmin=0, vmax=15)
    image = ax.imshow(relative.to_numpy(), cmap="YlOrRd", norm=norm, aspect="auto")
    for (i, j), value in np.ndenumerate(relative.to_numpy()):
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if value > 11 else "black", fontweight="bold" if value < 0.05 else "normal")
    ax.set_xticks(range(relative.shape[1]), relative.columns, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(relative.shape[0]), [f"{LABEL[m]}, {s}" for m, s in relative.index], fontsize=7.5)
    for y in np.arange(2.5, relative.shape[0] - 1, 3):
        ax.axhline(y, color="white", linewidth=2)
    ax.axvline(relative.shape[1] - 1.5, color="white", linewidth=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, extend="max")
    bar.set_label("CRPS above the best model and scale of the station (%)", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / (path.stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = read_crps()
    crps = frame.pivot_table(index=["model", "scale"], columns="station", values="CRPS")
    crps = crps.reindex(index=pd.MultiIndex.from_product([MODELS, list(SCALES.values())]), columns=list(STATIONS))
    crps.columns = [STATIONS[s][0] for s in crps.columns]
    crps["Mean"] = crps.mean(axis=1)

    climatology = pd.Series({STATIONS[s][0]: climatology_crps(s) for s in STATIONS})
    climatology["Mean"] = climatology.mean()
    skill = 1 - crps.drop(columns="Mean").div(climatology.drop("Mean"), axis=1)
    skill["Mean"] = skill.mean(axis=1)
    relative = 100 * (crps.drop(columns="Mean") / crps.loc[MAIN].drop(columns="Mean").min(axis=0) - 1)
    relative["Mean"] = relative.mean(axis=1)

    # without the ECA&D station list, reuse the saved sheet
    info = station_info() if ECAD_STATIONS.exists() else pd.read_excel(OUT / "stations_crps.xlsx", sheet_name="stations")
    with pd.ExcelWriter(OUT / "stations_crps.xlsx") as writer:
        crps.to_excel(writer, sheet_name="CRPS")
        climatology.to_frame("CRPS").to_excel(writer, sheet_name="climatology")
        skill.to_excel(writer, sheet_name="CRPSS")
        relative.to_excel(writer, sheet_name="relative_to_best")
        info.to_excel(writer, sheet_name="stations", index=False)
    (OUT / "table_stations_info.tex").write_text(info_table(info))
    (OUT / "table_stations_crps.tex").write_text(crps_table(crps))
    heatmap(relative.loc[MAIN], OUT / "stations_crps.pdf")

    pd.set_option("display.width", 200)
    print(info.round(2).to_string(index=False))
    print("\nCRPS\n", crps.round(3).to_string())
    print("\nmonthly climatology CRPS\n", climatology.round(3).to_string())
    print("\nCRPSS against the monthly climatology\n", skill.round(3).to_string())
    print("\nCRPS above the best model and scale of the station (%)\n", relative.round(1).to_string())
    best = crps.drop(columns="Mean").idxmin(axis=0)
    print("\nbest model and scale per station:", best.to_dict())
    best_scale = crps.drop(columns="Mean").groupby(level=0).min().reindex(MODELS)
    print("\nCRPS of each model at its best scale\n", best_scale.round(3).to_string())
    print("\nmean over stations of the best-scale CRPS\n", best_scale.mean(axis=1).round(3).to_string())
    # Section 4.1: one training scale per model, the scale of its lowest mean CRPS (not the best scale of each station)
    scale = crps["Mean"].groupby(level=0).idxmin().reindex(MODELS)
    print("\nCRPSS averaged over the stations, each model at the scale of its lowest mean CRPS",
          {m: s[1] for m, s in scale.items()}, "\n",
          pd.Series({m: skill.loc[s, "Mean"] for m, s in scale.items()}).round(3).to_string())


if __name__ == "__main__":
    main()
