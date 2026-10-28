#!/bin/bash
# Every experiment of the manuscript, then its tables and figures: bash run_all.sh [steps]
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
ALL_STEPS="data rf_tuning glm_rf mlp_gru bigru dssm smoother window analysis timing"
STEPS=${*:-$ALL_STEPS}
for step in $STEPS; do
    case " $ALL_STEPS " in *" $step "*) ;; *) echo "unknown step: $step (steps: $ALL_STEPS)"; exit 1 ;; esac
done

PYTHON=${PYTHON:-$([ -x ~/anaconda3/envs/nqd_phd/bin/python ] && echo ~/anaconda3/envs/nqd_phd/bin/python || echo python3)}
RSCRIPT=${RSCRIPT:-$([ -x ~/anaconda3/envs/nqd_R/bin/Rscript ] && echo ~/anaconda3/envs/nqd_R/bin/Rscript || echo Rscript)}
if [ -z "$DATA_DIR" ]; then
    folder=$HERE
    until [ -d "$folder/data/Data1Cell_06UTC" ] || [ "$folder" = / ]; do folder=$(dirname "$folder"); done
    DATA_DIR=$folder/data/Data1Cell_06UTC
fi
[ -d "$DATA_DIR" ] || { echo "No input folder: set DATA_DIR to the folder of era5_daily_cell_<station>.npz"; exit 1; }
DATA_DIR=$(cd "$DATA_DIR" && pwd)
MODE=$([ "$QUICK" = 1 ] && echo quick || echo full)
OUT=${OUT:-$(dirname "$HERE")/results/$([ "$MODE" = quick ] && echo quick_)$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
if [ -f "$OUT/logs/mode" ] && [ "$(cat "$OUT/logs/mode")" != "$MODE" ]; then
    echo "$OUT holds a $(cat "$OUT/logs/mode") run: choose another OUT"; exit 1
fi
echo "$MODE" > "$OUT/logs/mode"

WORKERS=${WORKERS:-$(getconf _NPROCESSORS_ONLN)}
GPU_WORKERS=${GPU_WORKERS:-12}
if [ -r /proc/meminfo ]; then MEM_GB=$(awk '/^MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
else MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 )); fi
if [ -z "$NOWINDOW_WORKERS" ]; then
    NOWINDOW_WORKERS=$(( WORKERS / 2 < MEM_GB / 5 ? WORKERS / 2 : MEM_GB / 5 ))
    [ "$NOWINDOW_WORKERS" -ge 1 ] || NOWINDOW_WORKERS=1
fi
OTHER_WINDOW_WORKERS=$(( WORKERS > NOWINDOW_WORKERS ? WORKERS - NOWINDOW_WORKERS : 1 ))

STATIONS=brest_guipavas,lille_lesquin,tours,strasbourg_entzheim,clermont_ferrand,bordeaux_merignac,montelimar,tarbes_ossun,perpignan
TUNING_STATIONS=$STATIONS
RF_PAPER="--rf-mtry 6 --rf-minbucket 20 --rf-minsplit 60 --rf-maxdepth 15"   # hyperparameters selected in the paper
if [ "$MODE" = quick ]; then    # fewer epochs, trees and configurations; 1000 members as in the full run
    TUNE_OPTIONS="--ntree 4 --mtry 2,6 --minbucket 10,20 --maxdepth2 15"
    GLM_RF_OPTIONS="--ntree 4"
    NN_OPTIONS="--max-epochs 2"
    DSSM_OPTIONS="--max-epochs 1"
fi
R_OPTIONS=${R_PYTHON:+--python $R_PYTHON}
ANALYSIS="make_window_tables stations_summary stations_indices stations_value_figures stations_annual_cycle
          downscaling_checks stations_appendix_tables sensitivity_tables ablation_tables tours_figures computing_times"


step_data() {
    "$PYTHON" - "$DATA_DIR" "$HERE/Data1Cell_06UTC.md5" <<'EOF'
import hashlib, sys
from pathlib import Path
data, listing = Path(sys.argv[1]), Path(sys.argv[2])
bad = [name for digest, name in (line.split() for line in listing.read_text().splitlines() if line.strip())
       if not (data / name).is_file() or hashlib.md5((data / name).read_bytes()).hexdigest() != digest]
print(f"checksums of {data}:", "OK" if not bad else "FAILED " + " ".join(bad))
sys.exit(bool(bad))
EOF
}

step_rf_tuning() {
    "$RSCRIPT" tune_rf_parallel.R --data-dir "$DATA_DIR" --stations "$TUNING_STATIONS" --workers "$WORKERS" --resume yes \
        $TUNE_OPTIONS $R_OPTIONS --out "$OUT/rf_tuning"
}

step_glm_rf() {
    # RF-BG with the hyperparameters selected by the rf_tuning step of this run, else with those of the paper
    local rf=$RF_PAPER selected=$OUT/rf_tuning/selected_config.json
    if [ -f "$selected" ]; then
        rf=$("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["selected"]
print(" ".join(f"--rf-{key} {s[key]}" for key in ("mtry", "minbucket", "minsplit", "maxdepth")))' "$selected") || return 1
        echo "RF-BG hyperparameters selected by the rf_tuning step: $rf"
    else
        echo "RF-BG hyperparameters of the paper (no rf_tuning step in $OUT): $rf"
    fi
    # the fitted models are saved with zstd, as in the paper, or with gzip by an R built without zstd
    local compress
    compress=$("$RSCRIPT" -e 'cat(tryCatch({saveRDS(1, tempfile(), compress = "zstd"); "zstd"}, error = function(e) "gzip"))') || return 1
    "$RSCRIPT" run_glm_rf_parallel.R --data-dir "$DATA_DIR" --stations "$STATIONS" --workers "$WORKERS" \
        $rf --save-models "$compress" $GLM_RF_OPTIONS $R_OPTIONS --out "$OUT/glm_rf"
}

step_mlp_gru() {
    "$PYTHON" run_mlp_gru_parallel.py --data-dir "$DATA_DIR" --stations "$STATIONS" --cpu-workers "$WORKERS" \
        --gpu-workers "$GPU_WORKERS" $NN_OPTIONS --out "$OUT/mlp_gru"
}

step_bigru() {
    # patience of six validations, as for the DSSM
    "$PYTHON" run_mlp_gru_parallel.py --arch bigru --patience 6 --data-dir "$DATA_DIR" --stations "$STATIONS" \
        --cpu-workers "$WORKERS" --gpu-workers "$GPU_WORKERS" $NN_OPTIONS --out "$OUT/bigru_bg"
}

step_dssm() {
    "$PYTHON" run_dssm_parallel.py --data-dir "$DATA_DIR" --stations "$STATIONS" --cpu-workers "$WORKERS" --resume \
        $DSSM_OPTIONS --out "$OUT/dssm"
}

step_smoother() {
    "$PYTHON" run_dssm_smoother_parallel.py --arch gru_smooth bigru_smooth --data-dir "$DATA_DIR" --stations "$STATIONS" \
        --cpu-workers "$WORKERS" --resume $DSSM_OPTIONS --out "$OUT/dssm_smoother"
}

step_window() {
    # The run without windows (one window of 100000 days) needs about 4.3 GB per chain: it runs with NOWINDOW_WORKERS
    # processes, while the windows of 128, 256 and 1024 days run one after another with the other processes.
    local common=(--variants DSSM_Annual --arch gru bigru gru_smooth bigru_smooth --data-dir "$DATA_DIR" --stations "$STATIONS" --resume)
    local failed=0 full window
    echo "$(date '+%F %T') no window, $NOWINDOW_WORKERS processes (log: $OUT/logs/window_full.log)"
    "$PYTHON" run_dssm_smoother_parallel.py "${common[@]}" $DSSM_OPTIONS --window 100000 --cpu-workers "$NOWINDOW_WORKERS" \
        --out "$OUT/dssm_window/window_full" > "$OUT/logs/window_full.log" 2>&1 &
    full=$!
    for window in 128 256 1024; do
        echo "$(date '+%F %T') window of $window days, $OTHER_WINDOW_WORKERS processes (log: $OUT/logs/window$window.log)"
        "$PYTHON" run_dssm_smoother_parallel.py "${common[@]}" $DSSM_OPTIONS --window $window --cpu-workers "$OTHER_WINDOW_WORKERS" \
            --out "$OUT/dssm_window/window$window" > "$OUT/logs/window$window.log" 2>&1 || failed=1
    done
    wait $full || failed=1
    echo "$(date '+%F %T') windows done"
    return $failed
}

step_analysis() {
    export DSSM_DATA=$DATA_DIR DSSM_RUNS=$OUT DSSM_OUTPUT=$OUT/analysis
    mkdir -p "$OUT/analysis/logs"
    local script
    for script in $ANALYSIS; do
        echo "$(date '+%F %T') $script.py"
        "$PYTHON" $script.py > "$OUT/analysis/logs/$script.log" 2>&1 || { echo "failed: see $OUT/analysis/logs/$script.log"; return 1; }
    done
    if "$PYTHON" -c "import cartopy, netCDF4, shapely, xarray" 2> /dev/null; then
        echo "$(date '+%F %T') plot_context_maps.py"
        "$PYTHON" plot_context_maps.py --all-on-coarse > "$OUT/analysis/logs/plot_context_maps.log" 2>&1 \
            || { echo "failed: see $OUT/analysis/logs/plot_context_maps.log"; return 1; }
    else
        echo "plot_context_maps.py skipped (Figure 3, maps of the stations): it needs cartopy, netCDF4, shapely and xarray"
    fi
}

step_timing() {
    # one process at a time, on one thread
    export DSSM_DATA=$DATA_DIR DSSM_RUNS=$OUT
    local r_python=$R_PYTHON
    [ -n "$r_python" ] || r_python=$(dirname "$(command -v "$RSCRIPT")")/python
    [ -x "$r_python" ] && export RETICULATE_PYTHON=$r_python
    "$PYTHON" bench_inference_neural.py ${STATIONS//,/ } &&
        "$RSCRIPT" bench_inference_baselines.R glm &&
        "$RSCRIPT" bench_inference_baselines.R rf
}


log() { echo "$(date '+%F %T') $*"; }

run_step() {
    local step=$1 start=$SECONDS
    if [ -f "$OUT/logs/$step.done" ]; then log "$step: finished in an earlier call, skipped"; return; fi
    log "$step: started (log: $OUT/logs/$step.log)"
    if "step_$step" >> "$OUT/logs/$step.log" 2>&1; then
        date '+%F %T' > "$OUT/logs/$step.done"
        log "$step: done in $(( (SECONDS - start) / 60 )) min"
    else
        log "$step: FAILED, see $OUT/logs/$step.log"
        exit 1
    fi
}

# Ctrl-C or kill: stop the worker processes too
trap 'trap - INT TERM; log "interrupted: stopping the workers"; kill 0' INT TERM

{
    echo "$(date '+%F %T') bash run_all.sh $STEPS on $(hostname)"
    echo "  PYTHON=$PYTHON ($("$PYTHON" --version 2>&1)), RSCRIPT=$RSCRIPT, R_PYTHON=${R_PYTHON:-default}"
    echo "  DATA_DIR=$DATA_DIR"
    echo "  WORKERS=$WORKERS GPU_WORKERS=$GPU_WORKERS NOWINDOW_WORKERS=$NOWINDOW_WORKERS RAM=${MEM_GB}GB MODE=$MODE"
} | tee -a "$OUT/logs/settings.log"
log "run folder: $OUT"
for step in $STEPS; do run_step "$step"; done
log "finished: $STEPS"
case " $STEPS " in *" analysis "*)
    log "tables and numbers in $OUT/analysis/model_comparison_stations and model_comparison_tours,"
    log "figures in $OUT/analysis/figures, logs of the analysis in $OUT/analysis/logs" ;;
esac
