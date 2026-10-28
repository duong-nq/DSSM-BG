"""Checks of the downscaling of DSSM-BG (Appendix B) with the saved checkpoints: reproduction of the test ensemble,
initial latent state z_T0 = 0, and context of the test period computed on windows as in training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import paths

OUT = paths.RESULTS / "downscaling_checks"
STATIONS = ["brest_guipavas", "lille_lesquin", "tours", "strasbourg_entzheim", "clermont_ferrand",
            "bordeaux_merignac", "montelimar", "tarbes_ossun", "perpignan"]


def worker(station):
    sys.argv = ["run_dssm_parallel.py", "--worker", "bigru", "DSSM_Annual", "ALL", "--station", station,
                "--data-dir", str(paths.DATA), "--device", "cpu", "--threads", "1", "--out", str(OUT / "unused")]
    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    import torch
    import run_dssm_parallel as r

    folder = paths.run_of(paths.DSSM_RUNS, station) / station / "DSSM_BiGRU" / "DSSM_Annual" / "seed456"
    log = json.loads((folder / "ALL_chain_log.json").read_text())[0]
    checkpoint = torch.load(folder / "ALL.pt", map_location="cpu", weights_only=False)
    model = r.DSSM(checkpoint["input_dim"], r.CFG.latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    assert np.allclose(checkpoint["metadata"]["scaler_mean"], r.x_mean)

    x = torch.tensor(r.X_standardized, dtype=torch.float32)
    past, test = r.history_rows, r.test_rows
    seed = log["chain_seed"] + 30000
    observed = r.y_all[test]
    margin = r.CFG.context_margin
    x_left = x[past][-margin:]

    # 1. reproduction of the saved ensemble
    filtered = r.predict(model, x[past], r.y_tensor[past], x[test], r.CFG.test_members, seed)
    crps_filtered = float(r.crps_ensemble_fair(observed, filtered["rainfall"].astype(np.float64)).mean())
    with np.load(folder / "test_ensemble.npz") as saved:
        crps_saved = float(r.crps_ensemble_fair(observed, saved["rainfall"].astype(np.float64)).mean())

    # 2. no filtering: z_T0 = 0, same context and same random numbers
    zero = torch.zeros(r.CFG.test_members, r.CFG.latent_dim)
    unfiltered = r.propagate_prior(model, x[test], zero, None, seed + 2, seed + 3, x_left=x_left)
    crps_zero = float(r.crps_ensemble_fair(observed, unfiltered["rainfall"].astype(np.float64)).mean())

    # 3. context on one sequence against windows of 512 days with margins, as in training
    sequence = torch.cat([x_left, x[test]])
    whole, _ = model.features(sequence, None, (margin, 0))
    pieces = []
    n = len(test)
    for start in range(0, n, r.CFG.window):
        stop = min(start + r.CFG.window, n)
        a, b = start, min(len(sequence), stop + 2 * margin)      # sequence rows: window start + margin ... stop + margin
        context, _ = model.features(sequence[a:b], None, (start + margin - a, b - stop - margin))
        pieces.append(context)
    windowed = torch.cat(pieces)
    difference = (whole - windowed).abs()
    # same filtered state and random numbers, windowed context
    boundary, _ = r.filter_history(model, x[past], r.y_tensor[past], r.CFG.test_members, seed + 1)
    features = model.features
    model.features = lambda *args, **kwargs: (windowed, None)
    on_windows = r.propagate_prior(model, x[test], boundary, None, seed + 2, seed + 3, x_left=x_left)
    model.features = features
    crps_windows = float(r.crps_ensemble_fair(observed, on_windows["rainfall"].astype(np.float64)).mean())
    interior = np.ones(n, bool)
    interior[-margin:] = False        # the last window has no right margin in both computations

    return {"station": station, "crps_saved": crps_saved, "crps_reproduced": crps_filtered, "crps_z0": crps_zero,
            "crps_windowed_context": crps_windows, "context_max_abs_diff": float(difference.max()),
            "context_mean_abs_diff": float(difference.mean()),
            "context_max_abs_diff_interior": float(difference[interior].max()),
            "context_abs_max": float(whole.abs().max())}


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    jobs = {s: subprocess.Popen([sys.executable, __file__, s], stdout=subprocess.PIPE, env=env, text=True) for s in STATIONS}
    rows = []
    for station, job in jobs.items():
        output, _ = job.communicate()
        assert job.returncode == 0, station
        rows.append(json.loads(output.strip().splitlines()[-1]))
    frame = pd.DataFrame(rows)
    frame["z0_change_percent"] = 100 * (frame.crps_z0 / frame.crps_reproduced - 1)
    frame["window_change_percent"] = 100 * (frame.crps_windowed_context / frame.crps_reproduced - 1)
    frame.to_excel(OUT / "downscaling_checks.xlsx", index=False)
    pd.set_option("display.width", 200)
    print(frame.round(5).to_string())
    gap = 100 * (frame.crps_reproduced / frame.crps_saved - 1).abs()
    print(f"reproduced against saved ensemble: max difference {gap.max():.3f}% ({frame.station[gap.idxmax()]}); "
          "the changes below are relative to the reproduced ensemble, computed on the same machine")
    print(f"z_T0 = 0: CRPS change {frame.z0_change_percent.min():+.2f} to {frame.z0_change_percent.max():+.2f}%, "
          f"mean CRPS {frame.crps_z0.mean():.4f} against {frame.crps_reproduced.mean():.4f}")
    print(f"context, one sequence vs windows: max abs difference {frame.context_max_abs_diff.max():.3f}, "
          f"mean {frame.context_mean_abs_diff.max():.1e}; CRPS change {frame.window_change_percent.min():+.3f} to "
          f"{frame.window_change_percent.max():+.3f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(worker(sys.argv[1])))
    else:
        main()
