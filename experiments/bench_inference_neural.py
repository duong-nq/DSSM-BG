"""Inference time of DSSM-BG and MLP-BG (Annual scale, one CPU thread) with the saved checkpoints: the 1000-member test
ensemble is regenerated, timed and compared with the saved one. Each station runs in its own process.

    python bench_inference_neural.py [stations]     # default: tours brest_guipavas
"""

import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import paths


def stub_scipy():
    """Empty scipy modules: only the scores use scipy, and they are not timed."""
    special, stats = types.ModuleType("scipy.special"), types.ModuleType("scipy.stats")
    special.gammainc = special.gammaln = special.logsumexp = stats.rankdata = None
    sys.modules.update({"scipy": types.ModuleType("scipy"), "scipy.special": special, "scipy.stats": stats})


def worker(model_name, station, out):
    stub_scipy()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(__file__).parent))
    common = ["--station", station, "--data-dir", str(paths.DATA), "--device", "cpu", "--threads", "1", "--out", out]
    if model_name == "DSSM-BG":
        sys.argv = ["run_dssm_parallel.py", "--worker", "bigru", "DSSM_Annual", "ALL", *common]
        import run_dssm_parallel as r
        folder = paths.run_of(paths.DSSM_RUNS, station) / station / "DSSM_BiGRU" / "DSSM_Annual" / "seed456"
        seed = json.loads((folder / "ALL_chain_log.json").read_text())[0]["chain_seed"] + 30000
        checkpoint = torch.load(folder / "ALL.pt", map_location="cpu", weights_only=False)
        model = r.DSSM(checkpoint["input_dim"], r.CFG.latent_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.requires_grad_(False)
        x = torch.tensor(r.X_standardized, dtype=torch.float32)
        started = time.perf_counter()
        rainfall = r.predict(model, x[r.history_rows], r.y_tensor[r.history_rows], x[r.test_rows],
                             r.CFG.test_members, seed)["rainfall"]
        seconds = time.perf_counter() - started
    else:
        sys.argv = ["run_mlp_gru_parallel.py", "--worker", "mlp", "Annual", "ALL", *common]
        import run_mlp_gru_parallel as r
        folder = paths.run_of(paths.MLP_RUNS, station) / station / "MLP_BG_Annual" / "seed42"
        model, _ = r.load_chain("MLP_BG", folder / "ALL.pt")
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            parameters = r.predict_parameters("MLP_BG", model, r.history_rows, r.test_rows)
            rainfall = r.simulate_bernoulli_gamma(parameters, r.CFG.ensemble_members, np.random.default_rng(r.CFG.seed))
            timings.append(time.perf_counter() - started)
        seconds = float(np.median(timings))
    with np.load(folder / "test_ensemble.npz") as saved:
        difference = float(np.abs(saved["rainfall"].astype(np.float64) - rainfall).max())
    print(json.dumps({"model": model_name, "station": station, "seconds": seconds, "max_abs_difference_mm": difference}))


def main():
    stations = sys.argv[1:] or ["tours", "brest_guipavas"]
    failed = []
    with tempfile.TemporaryDirectory() as out:
        for model_name in ["MLP-BG", "DSSM-BG"]:
            for station in stations:
                result = subprocess.run([sys.executable, __file__, "--worker", model_name, station, out],
                                        capture_output=True, text=True)
                lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
                print(lines[-1] if lines else result.stderr[-2000:], flush=True)
                if result.returncode != 0:
                    failed.append(f"{model_name} {station}")
    if failed:
        sys.exit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(*sys.argv[2:5])
    else:
        main()
