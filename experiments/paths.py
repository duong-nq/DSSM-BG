"""Folders of the experiments: those set by run_all.sh (DSSM_DATA, DSSM_RUNS, DSSM_OUTPUT), else the runs of the paper."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
if os.environ.get("DSSM_DATA"):
    DATA = Path(os.environ["DSSM_DATA"]).resolve()
else:
    DATA = next((folder / "data" / "Data1Cell_06UTC" for folder in (HERE, *HERE.parents)
                 if (folder / "data" / "Data1Cell_06UTC").is_dir()), HERE.parent / "data" / "Data1Cell_06UTC")
ROOT = DATA.parent.parent
ECAD_STATIONS = DATA.parent / "ecad_stations.txt"

if os.environ.get("DSSM_RUNS"):
    RUNS = Path(os.environ["DSSM_RUNS"]).resolve()
    GLM_RUNS = {"main": RUNS / "glm_rf"}
    RF_RUN = RUNS / "glm_rf"
    MLP_RUNS = {"main": RUNS / "mlp_gru"}
    DSSM_RUNS = {"main": RUNS / "dssm"}
    SMOOTHER_RUN = RUNS / "dssm_smoother"
    BIGRU_RUN = RUNS / "bigru_bg"
    WINDOW_RUN = RUNS / "dssm_window"
    LATE_STATIONS = ()
    _OUTPUT = RUNS / "analysis"
else:
    # runs of the paper; Montelimar and Perpignan were run later
    PAPER = ROOT / "results"
    GLM_RUNS = {"main": PAPER / "glm_rf_allstations_8vars" / "GLM_RF20261006_204253",
                "late": PAPER / "glm_rf_allstations_8vars" / "GLM_RF20261007_164729"}
    RF_RUN = PAPER / "glm_rf_allstations_8vars" / "GLM_RF_tuned_20261009_013535"
    MLP_RUNS = {"main": PAPER / "MLP_GRU_parallel_8vars" / "20261007_015136",
                "late": PAPER / "MLP_GRU_parallel_8vars" / "20261007_164729"}
    DSSM_RUNS = {"main": PAPER / "DSSM_parallel_8vars" / "20261006_212505",
                 "late": PAPER / "DSSM_parallel_8vars" / "20261007_164729"}
    SMOOTHER_RUN = PAPER / "DSSM_smoother_8vars" / "20261009_190419"
    BIGRU_RUN = PAPER / "BiGRU_BG_8vars" / "20261014_222246_patience6"
    WINDOW_RUN = PAPER / "DSSM_window_8vars" / "20261010_163318"
    LATE_STATIONS = ("montelimar", "perpignan")
    _OUTPUT = PAPER / "reproduced"

RESULTS = Path(os.environ.get("DSSM_OUTPUT") or _OUTPUT).resolve()
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def run_of(runs, station):
    return runs["late" if station in LATE_STATIONS else "main"]
