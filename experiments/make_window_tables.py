"""LaTeX tables of the training-window sensitivity of DSSM-BG (four versions, Annual scale), and window_metrics_long.xlsx
for sensitivity_tables.py."""
import argparse, math, os

import pandas as pd

import paths

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--w512-filter", nargs="+", default=[str(r) for r in paths.DSSM_RUNS.values()],
                    help="runs of run_dssm_parallel.py (window 512, DSSM_GRU and DSSM_BiGRU)")
parser.add_argument("--w512-smoother", nargs="+", default=[str(paths.SMOOTHER_RUN)],
                    help="runs of run_dssm_smoother_parallel.py (window 512, the smoother versions)")
parser.add_argument("--sweep", default=str(paths.WINDOW_RUN), help="folder with window128, window256, window1024, window_full")
parser.add_argument("--out", default=str(paths.RESULTS / "latex_tables_window_sensitivity"))
ARGS = parser.parse_args()

STATIONS = ["bordeaux_merignac", "brest_guipavas", "clermont_ferrand", "lille_lesquin", "montelimar", "perpignan",
            "strasbourg_entzheim", "tarbes_ossun", "tours"]
ARCHS = [("DSSM_GRU", "DSSM (GRU)"), ("DSSM_GRU_SmoothEnc", "DSSM (GRU, smoother)"),
         ("DSSM_BiGRU", "DSSM (BiGRU)"), ("DSSM_BiGRU_SmoothEnc", "DSSM (BiGRU, smoother)")]
WINDOWS = [(128, "128"), (256, "256"), (512, "512"), (1024, "1024"), ("full", "none")]


def folders(window, arch):
    if window == 512:
        return ARGS.w512_filter if arch in ("DSSM_GRU", "DSSM_BiGRU") else ARGS.w512_smoother
    return [os.path.join(ARGS.sweep, f"window{window}" if window != "full" else "window_full")]


rows = []
for window, _ in WINDOWS:
    for arch, _ in ARCHS:
        for root in folders(window, arch):
            for station in STATIONS:
                base = os.path.join(root, station, arch)
                if not os.path.exists(os.path.join(base, "point_metrics_overall.xlsx")):
                    continue
                scores = pd.read_excel(os.path.join(base, "scores.xlsx"))
                s = scores[(scores.variant == "DSSM_Annual") & (scores.period.astype(str) == "Annual")]
                points = pd.read_excel(os.path.join(base, "point_metrics_overall.xlsx"))
                p = points[(points.variant == "DSSM_Annual") & (points.group == "Overall")]
                chain = pd.read_excel(os.path.join(base, "chain_log.xlsx"))
                c = chain[chain.variant == "DSSM_Annual"]
                if s.empty or p.empty or c.empty:
                    continue
                s, p, c = s.iloc[0], p.iloc[0], c.iloc[0]
                windows_per_epoch = 1 if window == "full" else math.ceil(c.train_days / window)
                rows.append({"window": str(window), "arch": arch, "station": station, "CRPS": s.CRPS, "Brier": s.Brier, "AUC": s.AUC,
                             "Spearman": s.Spearman, "NLL": p.NLL, "MAE": p.MAE, "RMSE": p.RMSE, "P99_abs_bias": abs(s.P99wet_bias),
                             "validation_CRPS": c.validation_crps, "selected_epoch": c.selected_epoch, "epochs_run": c.epochs_run,
                             "updates_to_best": c.selected_epoch * windows_per_epoch, "minutes": c.minutes})
data = pd.DataFrame(rows)
assert not data.duplicated(["window", "arch", "station"]).any()
os.makedirs(ARGS.out, exist_ok=True)
data.to_excel(os.path.join(ARGS.out, "window_metrics_long.xlsx"), index=False)

# (column, header, direction, decimals)
METRICS = [("CRPS", r"CRPS$\downarrow$", "min", 3), ("Brier", r"Brier$\downarrow$", "min", 4), ("AUC", r"AUC$\uparrow$", "max", 3),
           ("Spearman", r"$\rho_s\uparrow$", "max", 3), ("NLL", r"NLL$\downarrow$", "min", 3), ("MAE", r"MAE$\downarrow$", "min", 3),
           ("RMSE", r"RMSE$\downarrow$", "min", 3), ("P99_abs_bias", r"$|$P99 bias$|\downarrow$", "min", 2),
           ("validation_CRPS", r"CRPS$_\mathrm{val}\downarrow$", "min", 3)]
COST = [("selected_epoch", "best epoch", 0), ("epochs_run", "epochs", 0), ("updates_to_best", "updates", 0), ("minutes", "min", 0)]


def fmt(value, decimals):
    return "--" if pd.isna(value) else f"{value:,.{decimals}f}".replace(",", r"\,")


def marks(values, direction, decimals):
    printed = values.round(decimals)
    key = printed if direction == "min" else -printed
    valid = sorted(set(key.dropna()))
    return {i: ("best" if v == valid[0] else "second" if len(valid) > 1 and v == valid[1] else None) for i, v in key.items() if pd.notna(v)}


def cell(value, decimals, mark):
    text = fmt(value, decimals)
    return rf"\textbf{{{text}}}" if mark == "best" else rf"\underline{{{text}}}" if mark == "second" else text


def wrap(lines, caption, label):
    return "\n".join([r"\begin{table}[p]", r"\centering", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%", *lines, "}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", r"\clearpage", ""])


blocks = []
# Table 1: mean over the stations, every metric and the training cost
mean = data.groupby(["arch", "window"]).agg({**{m: "mean" for m, *_ in METRICS + COST}, "station": "count"})
lines = [r"\begin{tabular}{llr" + "r" * len(METRICS) + "r" * len(COST) + "}", r"\toprule",
         rf" & & & \multicolumn{{{len(METRICS)}}}{{c}}{{Scores (test 2018--2025; CRPS$_\mathrm{{val}}$: 2013--2017)}} & \multicolumn{{{len(COST)}}}{{c}}{{Training (mean per chain)}} \\",
         rf"\cmidrule(lr){{4-{3 + len(METRICS)}}}\cmidrule(lr){{{4 + len(METRICS)}-{3 + len(METRICS) + len(COST)}}}",
         " & ".join(["Model", "Window", "n"] + [h for _, h, *_ in METRICS] + [h for _, h, _ in COST]) + r" \\", r"\midrule"]
for k, (arch, label) in enumerate(ARCHS):
    if k:
        lines.append(r"\midrule")
    block = mean.loc[arch] if arch in mean.index.get_level_values(0) else pd.DataFrame()
    block = block.reindex([str(w) for w, _ in WINDOWS])
    complete = block[block["station"] == len(STATIONS)] if len(block) else block
    m = {metric: marks(complete[metric], direction, decimals) if len(complete) else {} for metric, _, direction, decimals in METRICS}
    for j, (window, wlabel) in enumerate(WINDOWS):
        r = block.loc[str(window)] if len(block) else pd.Series(dtype=float)
        n = int(r["station"]) if len(r) and pd.notna(r.get("station")) else 0
        cells = [rf"\multirow{{{len(WINDOWS)}}}{{*}}{{{label}}}" if j == 0 else "", wlabel, str(n)]
        cells += [cell(r.get(metric), decimals, m[metric].get(str(window))) if n else "--" for metric, _, direction, decimals in METRICS]
        cells += [fmt(r.get(metric), decimals) if n else "--" for metric, _, decimals in COST]
        lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
blocks.append(wrap(lines, r"Training-window sensitivity of the DSSM (Annual variant): mean over the stations (n) of the scores on the test period "
                          r"2018--2025, the validation CRPS (2013--2017) and the training cost per chain. Window: length of the truncated "
                          r"backpropagation windows (days); none: the whole 1980--2012 chain (12\,054 days) is one window. "
                          r"Updates to best: best epoch $\times$ windows per epoch. Minutes depend on the machine load of each run. "
                          r"Per model, \textbf{bold}: best window, \underline{underlined}: second (complete rows only, $n = 9$).", "tab:window_mean"))

# Table 2: test CRPS per station
names = [s.replace("_", r"\_") for s in STATIONS]
lines = [r"\begin{tabular}{ll" + "r" * (len(STATIONS) + 1) + "}", r"\toprule",
         " & ".join(["Model", "Window"] + [n.replace(r"\_", r"\_\allowbreak ") for n in names] + ["Mean"]) + r" \\", r"\midrule"]
crps = data.pivot_table(index=["arch", "window"], columns="station", values="CRPS")
for k, (arch, label) in enumerate(ARCHS):
    if k:
        lines.append(r"\midrule")
    block = crps.loc[arch].reindex([str(w) for w, _ in WINDOWS]).reindex(columns=STATIONS) if arch in crps.index.get_level_values(0) \
        else pd.DataFrame(index=[str(w) for w, _ in WINDOWS], columns=STATIONS, dtype=float)
    block["Mean"] = block[STATIONS].mean(axis=1).where(block[STATIONS].notna().all(axis=1))
    m = {col: marks(block[col], "min", 3) for col in STATIONS + ["Mean"]}
    for j, (window, wlabel) in enumerate(WINDOWS):
        cells = [rf"\multirow{{{len(WINDOWS)}}}{{*}}{{{label}}}" if j == 0 else "", wlabel]
        cells += [cell(block.loc[str(window), col], 3, m[col].get(str(window))) for col in STATIONS + ["Mean"]]
        lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
blocks.append(wrap(lines, r"Test fair CRPS (mm/day, 2018--2025, Annual) per station and training window. Mean: only when all stations are available. "
                          r"Per model and station, \textbf{bold}: best window, \underline{underlined}: second.", "tab:window_crps_stations"))

# Table 3: best epoch and minutes per station
lines = [r"\begin{tabular}{ll" + "r" * len(STATIONS) + "}", r"\toprule",
         " & ".join(["Model", "Window"] + [n.replace(r"\_", r"\_\allowbreak ") for n in names]) + r" \\", r"\midrule"]
epochs = data.pivot_table(index=["arch", "window"], columns="station", values="selected_epoch")
minutes = data.pivot_table(index=["arch", "window"], columns="station", values="minutes")
for k, (arch, label) in enumerate(ARCHS):
    if k:
        lines.append(r"\midrule")
    for j, (window, wlabel) in enumerate(WINDOWS):
        cells = [rf"\multirow{{{len(WINDOWS)}}}{{*}}{{{label}}}" if j == 0 else "", wlabel]
        for station in STATIONS:
            try:
                e, t = epochs.loc[(arch, str(window)), station], minutes.loc[(arch, str(window)), station]
                cells.append("--" if pd.isna(e) else f"{int(e)} / {t:.0f}")
            except KeyError:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
blocks.append(wrap(lines, r"Best epoch / training minutes per station and training window (Annual). Minutes depend on the machine load of each run.",
                   "tab:window_cost_stations"))

with open(os.path.join(ARGS.out, "tables.tex"), "w") as f:
    f.write("\n".join(blocks))
status = "; ".join(f"{label}: {int(data[data.window == str(w)].station.count())}/{len(STATIONS) * len(ARCHS)} chains"
                   for w, label in WINDOWS)
with open(os.path.join(ARGS.out, "main.tex"), "w") as f:
    f.write("\n".join([
        r"\documentclass[10pt]{article}", r"\usepackage[a4paper,landscape,margin=1.5cm]{geometry}",
        r"\usepackage{booktabs}", r"\usepackage{graphicx}", r"\usepackage{multirow}", r"\begin{document}",
        r"\section*{DSSM: sensitivity to the training window}",
        r"\noindent The four DSSM versions combine a deterministic path over the predictors (causal GRU or BiGRU) with an inference network "
        r"(filter: causal GRU over $x_{1:t}, y_{1:t}$; smoother: BiGRU over $x_{1:T}, y_{1:T}$ of the training data). They are trained "
        r"with truncated backpropagation through time: each epoch walks through the 1980--2012 chain in consecutive windows of the given length, "
        r"one parameter update per window, the particles and recurrent states being carried from one window to the next with the gradient cut. "
        r"Window none: the whole chain is a single window (one update per epoch). Everything else is unchanged "
        r"(Annual variant, 8 predictors, ERA5 aggregated 06--06 UTC, seed 456, 40 particles, Adam $5\times10^{-4}$, validation every 5 epochs "
        r"on the free-run CRPS of 2013--2017, patience 6, tolerance $10^{-3}$, CPU with one thread per chain). Forecasts are made over the whole "
        r"test period at once: the encoder filters the history 1980--2017 and the prior runs freely over 2018--2025, so the window only "
        r"affects training. Window 512 is the setting of the main tables. "
        + rf"Runs available: {status}.",
        r"\clearpage", r"\input{tables}", r"\end{document}", ""]))
print(f"{len(data)} station x model x window rows -> {ARGS.out} ({status})")
