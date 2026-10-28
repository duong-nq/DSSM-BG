"""Computing times of the four models (Section 4.1) from the task logs of the runs; nothing is retrained. A task time
is the wall time of one task on one thread, so the sum over the tasks is the CPU time."""

import pandas as pd

import paths

STATIONS = ["brest_guipavas", "lille_lesquin", "tours", "strasbourg_entzheim", "clermont_ferrand",
            "bordeaux_merignac", "montelimar", "tarbes_ossun", "perpignan"]
SCALES = {"Annual": "Annual", "SeasonalConcat": "Season", "MonthConcat": "Month"}
OUT = paths.RESULTS / "computing_times"


def scale_of(name):
    return next(label for key, label in SCALES.items() if name.endswith(key))


def neural_tasks():
    rows = []
    for runs, model, column, keep in [(paths.DSSM_RUNS, "DSSM-BG", "variant", lambda d: d["arch"] == "bigru"),
                                      (paths.MLP_RUNS, "MLP-BG", "model", lambda d: d["model"].str.startswith(("MLP_BG", "ASSEMBLE")))]:
        log = pd.concat([pd.read_excel(run / "scheduler_log.xlsx") for run in runs.values()], ignore_index=True)
        log = log[log["station"].isin(STATIONS) & keep(log)]
        for _, r in log.iterrows():
            base = {"model": model, "station": r["station"]}
            if r["kind"] != "chain":
                rows.append({**base, "scale": "all", "part": "evaluation", "minutes": r["minutes"]})
                continue
            rows.append({**base, "scale": scale_of(r[column]), "part": "training", "minutes": r["minutes"]})
    return rows


def r_tasks():
    rows = []
    for runs, model in [(paths.GLM_RUNS.values(), "GLM-BG"), ([paths.RF_RUN], "RF-BG")]:
        log = pd.concat([pd.read_excel(run / "task_log.xlsx") for run in runs], ignore_index=True)
        log = log[log["station"].isin(STATIONS) & log["model"].astype(str).str.startswith(model.split("-")[0] + "_")]
        part = {"glm": "training", "rf_chunk": "training", "rf_predict": "inference", "model_eval": "evaluation"}
        for _, r in log.iterrows():
            rows.append({"model": model, "station": r["station"], "scale": scale_of(r["model"]),
                         "part": part[r["type"]], "minutes": r["seconds"] / 60})
    return rows


def main():
    tasks = pd.DataFrame(neural_tasks() + r_tasks())
    OUT.mkdir(parents=True, exist_ok=True)

    order = ["DSSM-BG", "MLP-BG", "GLM-BG", "RF-BG"]
    by_scale = tasks[tasks["part"] != "evaluation"].pivot_table(index="model", columns=["scale", "part"], values="minutes",
                                                                aggfunc="sum").reindex(order)
    longest = tasks[tasks["part"] == "training"].groupby(["model", "scale"])["minutes"].max().unstack().reindex(order)
    total = tasks.pivot_table(index="model", columns="part", values="minutes", aggfunc="sum").reindex(order).fillna(0)
    total["total"] = total.sum(axis=1)
    total["total_CPU_h"] = total["total"] / 60
    total["per_station_scale_min"] = total[["training", "inference"]].sum(axis=1) / (len(STATIONS) * len(SCALES))

    model_tasks = tasks[tasks["part"] != "evaluation"]
    station = model_tasks.pivot_table(index="station", columns=["model", "part"], values="minutes", aggfunc="sum")
    for model in order:
        station[(model, "total")] = station[model].sum(axis=1)
    station = station.reindex(columns=order, level=0)
    station_annual = model_tasks[model_tasks["scale"] == "Annual"].pivot_table(index="station", columns="model",
                                                                             values="minutes", aggfunc="sum")[order]

    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print("CPU minutes summed over the nine stations, by training scale (training and inference tasks):")
    print(by_scale.to_string(), "\n")
    print("Longest single training task (minutes, wall time on one thread):")
    print(longest.to_string(), "\n")
    print("CPU minutes summed over the nine stations and the three scales:")
    print(total.to_string(), "\n")
    print("CPU minutes of training and inference per station, three scales:")
    print(station.to_string())
    print(station.describe().loc[["mean", "min", "max"]].to_string(), "\n")
    print("CPU minutes of training and inference per station, Annual scale:")
    print(station_annual.to_string())
    print(station_annual.describe().loc[["mean", "min", "max"]].to_string())

    with pd.ExcelWriter(OUT / "computing_times.xlsx") as writer:
        tasks.to_excel(writer, sheet_name="tasks", index=False)
        by_scale.to_excel(writer, sheet_name="by_scale")
        longest.to_excel(writer, sheet_name="longest_task")
        total.to_excel(writer, sheet_name="total")
        station.to_excel(writer, sheet_name="per_station")
        station_annual.to_excel(writer, sheet_name="per_station_annual")


if __name__ == "__main__":
    main()
