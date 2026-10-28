"""MLP-BG, GRU-BG and BiGRU-BG at every station of --data-dir. The chains of the three training scales are trained as
parallel worker processes; when the chains of a station are done, one ASSEMBLE process draws its ensembles in a fixed
order and scores them, so the results do not depend on the number of workers."""


import argparse, os, subprocess, sys, time

ARCHS = {"mlp": "MLP_BG", "gru": "GRU_BG", "bigru": "BiGRU_BG"}
TIME_SCALES = ("Annual", "SeasonalConcat", "MonthConcat")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--arch", nargs="+", default=["mlp", "gru"], choices=list(ARCHS))
parser.add_argument("--variants", nargs="+", default=list(TIME_SCALES), choices=list(TIME_SCALES))
parser.add_argument("--predictors", default="8", choices=["8", "all"], help="8: the eight reference daily means (default); all: all 19 ERA5 predictors")
parser.add_argument("--stations", default="all", help="comma-separated station slugs, or all (every era5_daily_cell_<station>.npz of --data-dir)")
parser.add_argument("--station", default=None, help=argparse.SUPPRESS)
parser.add_argument("--data-dir", default="Data1CellNew", help="folder of the station files")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu-archs", nargs="*", default=["gru", "bigru"], choices=list(ARCHS), help="architectures trained on the GPU (default: gru bigru); the others on the CPU")
parser.add_argument("--gpu-workers", type=int, default=12, help="chain processes sharing the GPU at once")
parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1, help="chain processes on the CPU at once")
parser.add_argument("--no-mps", action="store_true", help="do not start CUDA MPS")
parser.add_argument("--threads", type=int, default=1, help="intra-op threads per worker")
parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help=argparse.SUPPRESS)
parser.add_argument("--out", default=None, help="run folder (default: Results/MLP_GRU_parallel_<8|19>vars/<timestamp>)")
parser.add_argument("--patience", type=int, default=None, help="early-stopping patience in validation checks [10; the DSSM uses 6]")
parser.add_argument("--max-epochs", type=int, default=None, help="for quick tests only")
parser.add_argument("--worker", nargs=3, metavar=("ARCH", "VARIANT", "GROUP"), help=argparse.SUPPRESS)
ARGS = parser.parse_args()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
os.environ["MKL_NUM_THREADS"] = str(ARGS.threads)
ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
IS_SCHEDULER = ARGS.worker is None
if IS_SCHEDULER:
    ARGS.device = "cpu"                         # the scheduler never needs CUDA
if ARGS.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # no CUDA context for CPU processes
PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
if ARGS.out is None:
    from datetime import datetime as _dt
    _set = "8vars" if ARGS.predictors == "8" else "19vars"
    ARGS.out = os.path.join("Results", f"MLP_GRU_parallel_{_set}", _dt.now().strftime("%Y%m%d_%H%M%S"))


def list_stations():
    slugs = sorted(name[len("era5_daily_cell_"):-len(".npz")] for name in os.listdir(ARGS.data_dir)
                   if name.startswith("era5_daily_cell_") and name.endswith(".npz"))
    slugs = [s for s in slugs if os.path.exists(os.path.join(ARGS.data_dir, f"ecad_rr_{s}_1980_2025.npy"))]
    if ARGS.stations != "all":
        wanted = [s.strip() for s in ARGS.stations.split(",") if s.strip()]
        missing = sorted(set(wanted) - set(slugs))
        if missing:
            sys.exit(f"stations not found in {ARGS.data_dir}: {', '.join(missing)}")
        slugs = wanted
    if not slugs:
        sys.exit(f"no station files in {ARGS.data_dir}")
    return slugs


if ARGS.station is None:
    ARGS.station = list_stations()[0]   # the scheduler loads one station to list the chains


# ---- Libraries and configuration -----------------------------------------------------------------
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from precipitation_metrics import MONTH_PERIODS, PERIOD_MONTHS, crps_bernoulli_gamma_analytic, crps_ensemble_fair, evaluate_annual_seasons_months, metric_metadata

torch.set_num_threads(ARGS.threads)

PREDICTORS_8 = ("d2m_mean", "t2m_mean", "msl_mean", "tcwv_mean", "u850_mean", "v850_mean", "z850_mean", "q850_mean")
PREDICTORS_19 = ("d2m_mean", "d2m_max", "t2m_mean", "t2m_min", "t2m_max", "msl_mean", "msl_min", "tcwv_mean", "tcwv_max", "cape_max", "u850_mean", "v850_mean", "z850_mean", "q850_mean", "q850_max", "omega_up850_mean", "omega_up850_max", "r850_mean", "r850_max")


class Config:
    def __init__(self):
        self.tau = 1.0
        self.hidden_dim = 32
        self.head_hidden_dim = 32
        self.epsilon = 1e-6
        self.mlp_layers = 2
        self.gru_layers = 1
        self.context_margin = 64        # BiGRU_BG: predictor days added on each side of a window

        self.batch_size = 256
        self.window = 512
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.gradient_clip = 5.0
        self.max_epochs = 1000
        self.validation_every = 5
        self.patience = 10
        self.improvement_tolerance = 1e-3
        self.ensemble_members = 1000
        self.seed = 42

        self.architectures = ("MLP_BG", "GRU_BG")
        self.time_scales = TIME_SCALES
        self.seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
        self.predictors = PREDICTORS_8

        self.train_start = "1980-01-01"
        self.train_end = "2012-12-31"
        self.validation_start = "2013-01-01"
        self.validation_end = "2017-12-31"
        self.test_start = "2018-01-01"
        self.test_end = "2025-12-31"
        self.extreme_reference_start = "1980-01-01"
        self.extreme_reference_end = "2017-12-31"
        self.extreme_quantiles = {"P95": 0.95, "P98": 0.98, "P99": 0.99}

        self.device = torch.device(ARGS.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.era5_path = Path(ARGS.data_dir) / f"era5_daily_cell_{ARGS.station}.npz"
        self.precipitation_path = Path(ARGS.data_dir) / f"ecad_rr_{ARGS.station}_1980_2025.npy"
        self.output_dir = Path(ARGS.out) / ARGS.station

    def as_dict(self):
        return {"tau": self.tau, "hidden_dim": self.hidden_dim, "head_hidden_dim": self.head_hidden_dim, "epsilon": self.epsilon, "mlp_layers": self.mlp_layers, "gru_layers": self.gru_layers, "context_margin": self.context_margin,
                "batch_size": self.batch_size, "window": self.window, "learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "gradient_clip": self.gradient_clip,
                "max_epochs": self.max_epochs, "validation_every": self.validation_every, "patience": self.patience, "improvement_tolerance": self.improvement_tolerance,
                "ensemble_members": self.ensemble_members, "seed": self.seed, "architectures": list(self.architectures), "time_scales": list(self.time_scales),
                "seasons": {name: list(group_months) for name, group_months in self.seasons.items()}, "predictors": list(self.predictors),
                "train_period": [self.train_start, self.train_end], "validation_period": [self.validation_start, self.validation_end], "test_period": [self.test_start, self.test_end],
                "extreme_reference_period": [self.extreme_reference_start, self.extreme_reference_end], "extreme_quantiles": self.extreme_quantiles,
                "device": str(self.device), "era5_path": str(self.era5_path), "precipitation_path": str(self.precipitation_path), "output_dir": str(self.output_dir),
                "threads_per_worker": ARGS.threads}


CFG = Config()
CFG.seed = ARGS.seed
CFG.architectures = tuple(ARCHS[a] for a in ARCHS if a in ARGS.arch)      # the ensembles are drawn in this order
CFG.time_scales = tuple(v for v in TIME_SCALES if v in ARGS.variants)
if ARGS.predictors == "all":
    CFG.predictors = PREDICTORS_19
if ARGS.max_epochs is not None:
    CFG.max_epochs = ARGS.max_epochs
if ARGS.patience is not None:
    CFG.patience = ARGS.patience
if not IS_SCHEDULER:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
torch.use_deterministic_algorithms(True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- Data: split and standardisation -------------------------------------------------------------
with np.load(CFG.era5_path) as data:
    variable_names = data["variables"].astype(str).tolist()
    predictor_columns = [variable_names.index(name) for name in CFG.predictors]
    X_raw = data["data"][:, predictor_columns].astype(np.float64)
    dates = pd.DatetimeIndex(data["dates"].astype("datetime64[ns]"))

y_raw = np.load(CFG.precipitation_path).astype(np.float64)
y_all = np.where(y_raw <= CFG.tau, 0.0, y_raw)
months = dates.month.to_numpy()

train_rows = np.flatnonzero((dates >= CFG.train_start) & (dates <= CFG.train_end))
validation_rows = np.flatnonzero((dates >= CFG.validation_start) & (dates <= CFG.validation_end))
test_rows = np.flatnonzero((dates >= CFG.test_start) & (dates <= CFG.test_end))
history_rows = np.concatenate([train_rows, validation_rows])

x_mean = X_raw[train_rows].mean(axis=0)
x_std = X_raw[train_rows].std(axis=0, ddof=0)
x_std = np.where(x_std == 0.0, 1.0, x_std)
X_standardized = ((X_raw - x_mean) / x_std).astype(np.float32)
X_tensor = torch.tensor(X_standardized, dtype=torch.float32, device=CFG.device)
y_tensor = torch.tensor(y_all, dtype=torch.float32, device=CFG.device)
split_rows = {"train": train_rows, "validation": validation_rows, "test": test_rows}

extreme_reference_rows = np.flatnonzero((dates >= CFG.extreme_reference_start) & (dates <= CFG.extreme_reference_end))
reference_rainfall = y_all[extreme_reference_rows]
reference_wet_rainfall = reference_rainfall[reference_rainfall > CFG.tau]
threshold_by_level = {level: float(np.quantile(reference_wet_rainfall, quantile)) for level, quantile in CFG.extreme_quantiles.items()}


def save_data_tables():
    """Station files: configuration, scaler, splits, extreme thresholds and chains."""
    (CFG.output_dir / "configs.json").write_text(json.dumps(CFG.as_dict(), indent=2))
    np.savez(CFG.output_dir / "scaler.npz", mean=x_mean, std=x_std)
    split_table = pd.DataFrame([{"split": name, "first": dates[rows[0]], "last": dates[rows[-1]], "days": len(rows)} for name, rows in split_rows.items()])
    split_table.to_excel(CFG.output_dir / "splits.xlsx", index=False)

    extreme_thresholds = pd.DataFrame({"quantile": list(CFG.extreme_quantiles.values()), "threshold_mm_day": list(threshold_by_level.values())})
    extreme_thresholds["reference_first"] = dates[extreme_reference_rows[0]]
    extreme_thresholds["reference_last"] = dates[extreme_reference_rows[-1]]
    extreme_thresholds["reference_wet_days"] = len(reference_wet_rainfall)
    extreme_thresholds.to_excel(CFG.output_dir / "extreme_thresholds.xlsx", index=False)

    extreme_groups = [("Annual", "ALL", tuple(range(1, 13)))] + [("Seasonal", name, group_months) for name, group_months in CFG.seasons.items()] + [("Monthly", f"{month:02d}", (month,)) for month in range(1, 13)]
    extreme_count_rows = []
    for configuration, group, group_months in extreme_groups:
        row = {"configuration": configuration, "group": group}
        for split, rows in {"reference": extreme_reference_rows, "test": test_rows}.items():
            rainfall = y_all[rows[np.isin(months[rows], group_months)]]
            row[f"{split}_days"] = len(rainfall)
            row[f"{split}_wet_days"] = int((rainfall > CFG.tau).sum())
            for level, threshold in threshold_by_level.items():
                row[f"{split}_{level}_exceedance_days"] = int((rainfall > threshold).sum())
        extreme_count_rows.append(row)
    pd.DataFrame(extreme_count_rows).to_excel(CFG.output_dir / "extreme_counts_by_group.xlsx", index=False)

    chain_rows = []
    for time_scale in CFG.time_scales:
        for split, rows in split_rows.items():
            for group, indices in build_chains(rows, time_scale).items():
                chain_rows.append({"time_scale": time_scale, "split": split, "group": group, "days": len(indices)})
    pd.DataFrame(chain_rows).to_excel(CFG.output_dir / "temporal_chains.xlsx", index=False)


# ---- Chains of the three training scales ---------------------------------------------------------
def build_chains(rows, time_scale):
    if time_scale == "Annual":
        return {"ALL": rows}
    if time_scale == "SeasonalConcat":
        groups = CFG.seasons
    else:
        groups = {f"{month:02d}": (month,) for month in range(1, 13)}
    return {group: rows[np.isin(months[rows], group_months)] for group, group_months in groups.items()}


# ---- Bernoulli-Gamma models ----------------------------------------------------------------------
class BernoulliGammaHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, CFG.head_hidden_dim), nn.Tanh(), nn.Linear(CFG.head_hidden_dim, 3))

    def forward(self, context):
        wet_logit, raw_mu, raw_alpha = self.network(context).unbind(dim=-1)
        mu = CFG.epsilon + F.softplus(raw_mu)
        alpha = CFG.epsilon + F.softplus(raw_alpha)
        beta = alpha / mu
        wet_probability = torch.sigmoid(wet_logit)
        return {"wet_logit": wet_logit, "wet_probability": wet_probability, "mu": mu, "alpha": alpha, "beta": beta, "predicted_mean": wet_probability * (CFG.tau + mu)}


class MLPBG(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(CFG.mlp_layers):
            layers.extend([nn.Linear(current_dim, CFG.hidden_dim), nn.ReLU()])
            current_dim = CFG.hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = BernoulliGammaHead(CFG.hidden_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


class GRUBG(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, CFG.hidden_dim, num_layers=CFG.gru_layers, batch_first=True)
        self.head = BernoulliGammaHead(CFG.hidden_dim)

    def forward(self, x, hidden=None):
        context, hidden = self.encoder(x.unsqueeze(0), hidden)
        return self.head(context.squeeze(0)), hidden


class BiGRUBG(nn.Module):
    """Bidirectional GRU (hidden_dim // 2 units per direction) with the same Bernoulli-Gamma head."""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, CFG.hidden_dim // 2, num_layers=CFG.gru_layers, batch_first=True, bidirectional=True)
        self.head = BernoulliGammaHead(CFG.hidden_dim)

    def forward(self, x, pad=(0, 0)):
        """pad = (left, right) margin rows of x that are cropped from the output."""
        context, _ = self.encoder(x.unsqueeze(0))
        return self.head(context[0, pad[0]:len(x) - pad[1]])


def build_model(architecture):
    if architecture == "MLP_BG":
        return MLPBG(X_tensor.shape[1]).to(CFG.device)
    if architecture == "BiGRU_BG":
        return BiGRUBG(X_tensor.shape[1]).to(CFG.device)
    return GRUBG(X_tensor.shape[1]).to(CFG.device)


# ---- Bernoulli-Gamma likelihood ------------------------------------------------------------------
def bernoulli_gamma_log_prob(observed, wet_logit, alpha, beta):
    wet = observed > CFG.tau
    excess = torch.where(wet, observed - CFG.tau, torch.ones_like(observed))
    gamma_log_prob = alpha * torch.log(beta) - torch.lgamma(alpha) + (alpha - 1.0) * torch.log(excess) - beta * excess
    return torch.where(wet, F.logsigmoid(wet_logit) + gamma_log_prob, F.logsigmoid(-wet_logit))


def negative_log_likelihood(parameters, observed):
    return -bernoulli_gamma_log_prob(observed, parameters["wet_logit"], parameters["alpha"], parameters["beta"]).mean()


# ---- Training and validation ---------------------------------------------------------------------
def tensor_rows(tensor, rows):
    return tensor[torch.as_tensor(rows, dtype=torch.long, device=tensor.device)]


def train_mlp_epoch(model, optimizer, rows, generator):
    model.train()
    order = torch.randperm(len(rows), generator=generator).numpy()
    total_loss = 0.0
    for start in range(0, len(rows), CFG.batch_size):
        batch_rows = rows[order[start:start + CFG.batch_size]]
        parameters = model(tensor_rows(X_tensor, batch_rows))
        loss = negative_log_likelihood(parameters, tensor_rows(y_tensor, batch_rows))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch_rows)
    return total_loss / len(rows)


def train_gru_epoch(model, optimizer, rows):
    model.train()
    hidden = None
    total_loss = 0.0
    for start in range(0, len(rows), CFG.window):
        chunk_rows = rows[start:start + CFG.window]
        parameters, hidden = model(tensor_rows(X_tensor, chunk_rows), hidden)
        loss = negative_log_likelihood(parameters, tensor_rows(y_tensor, chunk_rows))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
        optimizer.step()
        hidden = hidden.detach()
        total_loss += float(loss.detach()) * len(chunk_rows)
    return total_loss / len(rows)


def train_bigru_epoch(model, optimizer, rows):
    model.train()
    total_loss = 0.0
    for start in range(0, len(rows), CFG.window):
        stop = min(start + CFG.window, len(rows))
        path_start, path_stop = max(0, start - CFG.context_margin), min(len(rows), stop + CFG.context_margin)
        parameters = model(tensor_rows(X_tensor, rows[path_start:path_stop]), (start - path_start, path_stop - stop))
        loss = negative_log_likelihood(parameters, tensor_rows(y_tensor, rows[start:stop]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * (stop - start)
    return total_loss / len(rows)


@torch.inference_mode()
def predict_parameters(architecture, model, filter_rows, target_rows):
    model.eval()
    if architecture == "MLP_BG":
        parameters = model(tensor_rows(X_tensor, target_rows))
    elif architecture == "BiGRU_BG":
        left = filter_rows[-CFG.context_margin:]
        parameters = model(tensor_rows(X_tensor, np.concatenate([left, target_rows])), (len(left), 0))
    else:
        hidden = None
        for start in range(0, len(filter_rows), CFG.window):
            _, hidden = model(tensor_rows(X_tensor, filter_rows[start:start + CFG.window]), hidden)
        parameters, _ = model(tensor_rows(X_tensor, target_rows), hidden)
    return {name: values.detach().cpu().numpy().astype(np.float64) for name, values in parameters.items() if name != "wet_logit"}


def validation_crps(architecture, model, training_chain, validation_chain):
    parameters = predict_parameters(architecture, model, training_chain, validation_chain)
    return float(crps_bernoulli_gamma_analytic(y_all[validation_chain], parameters["wet_probability"], parameters["alpha"], parameters["beta"], CFG.tau).mean())


def train_chain(architecture, training_chain, validation_chain, seed, checkpoint_path, metadata):
    set_seed(seed)
    model = build_model(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_crps = float("inf")
    bad_checks = 0
    history = []

    for epoch in range(1, CFG.max_epochs + 1):
        if architecture == "MLP_BG":
            train_nll = train_mlp_epoch(model, optimizer, training_chain, generator)
        elif architecture == "BiGRU_BG":
            train_nll = train_bigru_epoch(model, optimizer, training_chain)
        else:
            train_nll = train_gru_epoch(model, optimizer, training_chain)
        current_crps = np.nan

        if epoch % CFG.validation_every == 0 or epoch == CFG.max_epochs:
            current_crps = validation_crps(architecture, model, training_chain, validation_chain)
            if current_crps < best_crps - CFG.improvement_tolerance:
                best_crps = current_crps
                bad_checks = 0
                checkpoint = {"state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, "epoch": epoch, "validation_crps": current_crps, "seed": seed, "settings": CFG.as_dict(), "metadata": metadata}
                torch.save(checkpoint, checkpoint_path)
            else:
                bad_checks += 1
            print(f"{metadata['model']}/{metadata['group']} | epoch {epoch} | NLL {train_nll:.4f} | validation CRPS {current_crps:.4f}", flush=True)

        history.append({"epoch": epoch, "train_nll": train_nll, "validation_crps": current_crps})
        if bad_checks >= CFG.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=CFG.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, pd.DataFrame(history), checkpoint


def load_chain(architecture, checkpoint_path):
    """Rebuild a trained chain from its checkpoint, exactly as train_chain() returns it."""
    model = build_model(architecture)
    checkpoint = torch.load(checkpoint_path, map_location=CFG.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


# ---- Ensembles -----------------------------------------------------------------------------------
def simulate_bernoulli_gamma(parameters, members, rng):
    wet_probability = parameters["wet_probability"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    wet = rng.random((len(wet_probability), members)) < wet_probability[:, None]
    excess = rng.gamma(shape=alpha[:, None], scale=1.0 / beta[:, None], size=(len(wet_probability), members))
    return np.where(wet, CFG.tau + excess, 0.0)


def period_mean(values, evaluation_dates, period):
    evaluation_months = pd.DatetimeIndex(evaluation_dates).month.to_numpy()
    period_months = {**PERIOD_MONTHS, **MONTH_PERIODS}[period]
    selected = np.ones(len(values), dtype=bool) if period == "Annual" else np.isin(evaluation_months, period_months)
    return float(np.mean(values[selected]))


# ---- Parallel execution --------------------------------------------------------------------------
def model_folder(architecture, time_scale):
    folder = CFG.output_dir / f"{architecture}_{time_scale}" / f"seed{CFG.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_one_chain(architecture, time_scale, group):
    """Worker: train one chain and save its checkpoint and training log."""
    model_name = f"{architecture}_{time_scale}"
    folder = model_folder(architecture, time_scale)
    training_chain = build_chains(train_rows, time_scale)[group]
    validation_chain = build_chains(validation_rows, time_scale)[group]
    metadata = {"model": model_name, "architecture": architecture, "time_scale": time_scale, "group": group}
    started = time.perf_counter()
    _, training_log, checkpoint = train_chain(architecture, training_chain, validation_chain, CFG.seed, folder / f"{group}.pt", metadata)
    training_log.insert(0, "group", group)
    training_log.insert(0, "model", model_name)
    training_log.to_excel(folder / f"{group}_training.xlsx", index=False)
    (folder / f"{group}_device.json").write_text(json.dumps({"device": str(CFG.device), "threads": ARGS.threads if CFG.device.type == "cpu" else None}))
    print(f"DONE {ARGS.station} {model_name}/{group} on {CFG.device} | epoch {checkpoint['epoch']} | validation CRPS {checkpoint['validation_crps']:.4f} | {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def assemble():
    """Reload the checkpoints of a station, draw its ensembles in a fixed order and score them."""
    from precipitation_metrics import evaluate_point_metrics
    save_data_tables()
    evaluation_metadata = metric_metadata()
    evaluation_metadata = evaluation_metadata[evaluation_metadata["metric"] != "CRPS"].copy()
    evaluation_metadata = pd.concat([evaluation_metadata, pd.DataFrame([
        {"metric": "CRPS_analytic", "category": "Distribution", "unit": "mm/day", "preferred": "lower", "definition": "Analytical CRPS of the Bernoulli–Gamma predictive distribution"},
        {"metric": "CRPS_ensemble_fair", "category": "Distribution", "unit": "mm/day", "preferred": "lower", "definition": "Fair CRPS calculated from the simulated ensemble"},
    ])], ignore_index=True)
    evaluation_metadata.to_excel(CFG.output_dir / "evaluation_metric_metadata.xlsx", index=False)

    summary_tables, daily_tables, member_tables, training_tables, chain_log_rows = [], [], [], [], []
    ensemble_rng = np.random.default_rng(CFG.seed)
    for architecture in CFG.architectures:
        for time_scale in CFG.time_scales:
            model_name = f"{architecture}_{time_scale}"
            folder = model_folder(architecture, time_scale)
            training_chains = build_chains(train_rows, time_scale)
            validation_chains = build_chains(validation_rows, time_scale)
            history_chains = build_chains(history_rows, time_scale)
            test_chains = build_chains(test_rows, time_scale)
            combined = {}

            for group in training_chains:
                training_chain = training_chains[group]
                validation_chain = validation_chains[group]
                history_chain = history_chains[group]
                test_chain = test_chains[group]
                model, checkpoint = load_chain(architecture, folder / f"{group}.pt")
                training_tables.append(pd.read_excel(folder / f"{group}_training.xlsx"))

                parameters = predict_parameters(architecture, model, history_chain, test_chain)
                rainfall = simulate_bernoulli_gamma(parameters, CFG.ensemble_members, ensemble_rng)
                calendar_positions = np.searchsorted(test_rows, test_chain)
                for name, values in parameters.items():
                    if name not in combined:
                        combined[name] = np.empty(len(test_rows), dtype=np.float64)
                    combined[name][calendar_positions] = values
                if "rainfall" not in combined:
                    combined["rainfall"] = np.empty((len(test_rows), CFG.ensemble_members), dtype=np.float64)
                combined["rainfall"][calendar_positions] = rainfall

                trained_on = json.loads((folder / f"{group}_device.json").read_text())
                chain_log_rows.append({"station": ARGS.station, "trained_on": trained_on["device"], "train_threads": trained_on["threads"], "model": model_name, "architecture": architecture, "time_scale": time_scale, "group": group, "seed": CFG.seed, "train_days": len(training_chain), "validation_days": len(validation_chain), "history_days": len(history_chain), "test_days": len(test_chain), "selected_epoch": checkpoint["epoch"], "validation_crps": checkpoint["validation_crps"]})
                del model

            np.savez_compressed(folder / "test_ensemble.npz", dates=dates[test_rows].to_numpy(), **combined)
            observed_test = y_all[test_rows]
            test_dates = dates[test_rows].to_numpy()
            analytic_crps = crps_bernoulli_gamma_analytic(observed_test, combined["wet_probability"], combined["alpha"], combined["beta"], CFG.tau)
            ensemble_crps = crps_ensemble_fair(observed_test, combined["rainfall"], CFG.tau)
            evaluation = evaluate_annual_seasons_months(observed=observed_test, dates=test_dates, threshold=CFG.tau, wet_probability=combined["wet_probability"], alpha=combined["alpha"], beta=combined["beta"], ensemble=combined["rainfall"], predicted_mean=combined["predicted_mean"], crps_daily=analytic_crps)

            summary = evaluation["summary"].rename(columns={"CRPS": "CRPS_analytic"})
            summary["CRPS_method"] = "analytic Bernoulli-Gamma"
            summary["CRPS_ensemble_fair"] = [period_mean(ensemble_crps, test_dates, period) for period in summary["period"]]
            summary.insert(0, "seed", CFG.seed)
            summary.insert(0, "time_scale", time_scale)
            summary.insert(0, "architecture", architecture)
            summary.insert(0, "model", model_name)
            summary.to_excel(folder / "evaluation_summary.xlsx", index=False)
            summary_tables.append(summary)

            daily = evaluation["daily"].rename(columns={"CRPS": "CRPS_analytic"})
            daily["CRPS_method"] = "analytic Bernoulli-Gamma"
            daily["CRPS_ensemble_fair"] = ensemble_crps
            daily.insert(0, "model", model_name)
            daily.to_excel(folder / "daily_test_scores.xlsx", index=False)
            daily_tables.append(daily)

            members = evaluation["members"]
            members.insert(0, "model", model_name)
            members.to_excel(folder / "member_metrics.xlsx", index=False)
            member_tables.append(members)
            print(f"== {model_name} | CRPS analytic {analytic_crps.mean():.4f} | CRPS ensemble {ensemble_crps.mean():.4f}", flush=True)

    scores = pd.concat(summary_tables, ignore_index=True)
    member_metrics = pd.concat(member_tables, ignore_index=True)
    scores.to_excel(CFG.output_dir / "scores.xlsx", index=False)
    pd.concat(daily_tables, ignore_index=True).to_excel(CFG.output_dir / "daily_test_scores.xlsx", index=False)
    member_metrics.to_excel(CFG.output_dir / "member_metrics.xlsx", index=False)
    pd.concat(training_tables, ignore_index=True).to_excel(CFG.output_dir / "training_history.xlsx", index=False)
    pd.DataFrame(chain_log_rows).to_excel(CFG.output_dir / "chain_log.xlsx", index=False)

    headline_columns = ["model", "AUC", "Brier", "SDII_observed", "SDII_model", "Spearman", "SD_observed", "SD_model", "P95wet_observed", "P95wet_model", "P99wet_observed", "P99wet_model", "CRPS_analytic", "CRPS_ensemble_fair", "WW_observed", "WW_model", "CDD_observed", "CDD_model", "CWD_observed", "CWD_model"]
    scores[scores["period"] == "Annual"][headline_columns].reset_index(drop=True).to_excel(CFG.output_dir / "annual_comparison.xlsx", index=False)
    scores[["model", "period", "AUC", "Brier", "Spearman", "CRPS_analytic", "CRPS_ensemble_fair"]].to_excel(CFG.output_dir / "annual_seasonal_monthly_scores.xlsx", index=False)
    trajectory_columns = [metric for metric in evaluation_metadata["metric"] if metric in member_metrics.columns]
    member_metrics.groupby(["model", "period"])[trajectory_columns].agg(["mean", "std"]).to_excel(CFG.output_dir / "member_metric_summary.xlsx")

    # point metrics from the saved ensembles
    observed_test = y_all[test_rows]
    test_dates = dates[test_rows].to_numpy().astype("datetime64[D]")
    point_overall, point_seasonal, point_monthly = [], [], []
    for architecture in CFG.architectures:
        for time_scale in CFG.time_scales:
            model_name = f"{architecture}_{time_scale}"
            with np.load(model_folder(architecture, time_scale) / "test_ensemble.npz", allow_pickle=False) as saved:
                saved_dates = saved["dates"].astype("datetime64[D]")
                ensemble = saved["rainfall"]
                wet_probability, alpha, beta = saved["wet_probability"], saved["alpha"], saved["beta"]
            assert len(saved_dates) == len(test_dates) == len(observed_test) == len(ensemble), model_name
            assert np.array_equal(saved_dates, test_dates), f"{model_name}: forecast dates differ"
            tables = evaluate_point_metrics(observed=observed_test, dates=saved_dates, ensemble=ensemble, threshold=CFG.tau, wet_probability=wet_probability, alpha=alpha, beta=beta, model_name=model_name)
            for frame, store in ((tables["overall"], point_overall), (tables["seasonal"], point_seasonal), (tables["monthly"], point_monthly)):
                frame.insert(1, "architecture", architecture)
                frame.insert(2, "time_scale", time_scale)
                store.append(frame)
    for frame, name in ((point_overall, "overall"), (point_seasonal, "seasonal"), (point_monthly, "monthly")):
        pd.concat(frame, ignore_index=True).to_excel(CFG.output_dir / f"point_metrics_{name}.xlsx", index=False)

    import platform
    session_info = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "device": str(CFG.device), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "threads_per_worker": ARGS.threads, "assemble_device": str(CFG.device)}
    (CFG.output_dir / "session_info.json").write_text(json.dumps(session_info, indent=2))
    print("Results:", CFG.output_dir, flush=True)


def gpu_environment():
    """Environment of the GPU processes: the scheduler itself hides the GPU, its children must not."""
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if ORIGINAL_CUDA_VISIBLE_DEVICES is not None:
        env["CUDA_VISIBLE_DEVICES"] = ORIGINAL_CUDA_VISIBLE_DEVICES
    return env


def start_mps():
    """Start the CUDA MPS daemon unless it already runs. Returns True when this script started it."""
    if ARGS.no_mps or subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", "nvidia-cuda-mps-control"], capture_output=True).returncode == 0:
        return False
    env = gpu_environment()
    env["CUDA_MPS_LOG_DIRECTORY"] = os.path.abspath(os.path.join(ARGS.out, "logs", "mps"))
    os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    started = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, env=env).returncode == 0
    print("CUDA MPS daemon started" if started else "CUDA MPS could not be started; GPU processes are time-sliced", flush=True)
    return started


def stop_mps():
    # "quit -t": the server can report a stale worker thread after its clients exit and would then never stop
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit -t 10\n", text=True, capture_output=True, env=gpu_environment(), timeout=60)
    except subprocess.TimeoutExpired:
        print("CUDA MPS did not stop; stop it with: echo 'quit -t 10' | nvidia-cuda-mps-control", flush=True)
        return
    print("CUDA MPS daemon stopped", flush=True)


def main():
    stations = list_stations()
    gpu_available = subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    gpu_archs = {ARCHS[a] for a in ARGS.gpu_archs} if gpu_available else set()
    if ARGS.gpu_archs and not gpu_available:
        print("No GPU found: every chain runs on the CPU", flush=True)
    architectures = CFG.architectures
    # Chains in priority order: longest variants first, then station, then architecture.
    chains = [(station, architecture, time_scale, group)
              for time_scale in CFG.time_scales for station in stations for architecture in architectures
              for group in build_chains(train_rows, time_scale)]
    device_of = {c: "cuda" if c[1] in gpu_archs else "cpu" for c in chains}
    queues = {device: [c for c in chains if device_of[c] == device] for device in ("cuda", "cpu")}
    remaining = {station: sum(1 for c in chains if c[0] == station) for station in stations}
    slots = {"cuda": ARGS.gpu_workers if queues["cuda"] else 0, "cpu": max(ARGS.cpu_workers, 1)}
    assemble_queue, failed = [], set()

    log_dir = os.path.join(ARGS.out, "logs"); os.makedirs(log_dir, exist_ok=True)
    print(f"{len(stations)} station(s), {len(chains)} chains: {len(queues['cuda'])} on GPU ({slots['cuda']} at once), "
          f"{len(queues['cpu'])} on CPU ({slots['cpu']} at once x {ARGS.threads} thread(s)) -> {ARGS.out}", flush=True)
    mps_started = start_mps() if slots["cuda"] else False
    arch_keys = {name: key for key, name in ARCHS.items()}

    def launch(kind, unit, device):
        common = ["--seed", str(ARGS.seed), "--threads", str(ARGS.threads), "--out", ARGS.out, "--predictors", ARGS.predictors,
                  "--data-dir", ARGS.data_dir, "--device", device, "--arch", *ARGS.arch, "--variants", *ARGS.variants]
        if ARGS.max_epochs is not None:
            common += ["--max-epochs", str(ARGS.max_epochs)]
        if ARGS.patience is not None:
            common += ["--patience", str(ARGS.patience)]
        if kind == "chain":
            station, architecture, time_scale, group = unit
            command = [sys.executable, os.path.abspath(__file__), "--worker", arch_keys[architecture], time_scale, group, "--station", station, *common]
            log_name = f"{station}_{architecture}_{time_scale}_{group}.log"
        else:
            command = [sys.executable, os.path.abspath(__file__), "--worker", "ASSEMBLE", "-", "-", "--station", unit, *common]
            log_name = f"{unit}_ASSEMBLE.log"
        log = open(os.path.join(log_dir, log_name), "w")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=gpu_environment())
        return {"kind": kind, "unit": unit, "device": device, "process": process, "log": log, "started": time.time()}

    running, records, run_started = [], [], time.time()
    n_total = len(chains) + len(stations)
    try:
        while True:
            # CPU slots: evaluations first (they unlock finished stations), then CPU chains.
            while sum(r["device"] == "cpu" for r in running) < slots["cpu"] and (assemble_queue or queues["cpu"]):
                running.append(launch("assemble", assemble_queue.pop(0), "cpu") if assemble_queue else launch("chain", queues["cpu"].pop(0), "cpu"))
            while sum(r["device"] == "cuda" for r in running) < slots["cuda"] and queues["cuda"]:
                running.append(launch("chain", queues["cuda"].pop(0), "cuda"))
            if not running:
                break
            time.sleep(0.5)
            for r in [r for r in running if r["process"].poll() is not None]:
                running.remove(r); r["log"].close()
                code, seconds, unit = r["process"].returncode, time.time() - r["started"], r["unit"]
                is_chain = r["kind"] == "chain"
                records.append({"kind": r["kind"], "station": unit[0] if is_chain else unit, "model": f"{unit[1]}_{unit[2]}" if is_chain else "ASSEMBLE",
                                "group": unit[3] if is_chain else "", "device": r["device"], "return_code": code,
                                "minutes": seconds / 60, "finished_after_min": (time.time() - run_started) / 60})
                label = "/".join(unit) if is_chain else f"{unit} ASSEMBLE"
                print(f"[{len(records)}/{n_total} | {(time.time() - run_started) / 60:5.1f} min] {'finished' if code == 0 else 'FAILED'} {label} on {r['device']} ({seconds:.0f} s)", flush=True)
                station = unit[0] if is_chain else unit
                if code != 0:
                    failed.add(station)
                if is_chain:
                    remaining[station] -= 1
                    if remaining[station] == 0 and station not in failed:
                        assemble_queue.append(station)
    finally:
        for r in running:
            r["process"].kill()
        if mps_started:
            stop_mps()

    pd.DataFrame(records).to_excel(os.path.join(ARGS.out, "scheduler_log.xlsx"), index=False)
    for name in ("scores", "annual_comparison", "chain_log", "point_metrics_overall", "point_metrics_seasonal", "point_metrics_monthly"):
        frames = []
        for station in stations:
            path = os.path.join(ARGS.out, station, f"{name}.xlsx")
            if os.path.exists(path):
                frame = pd.read_excel(path)
                if "station" not in frame:
                    frame.insert(0, "station", station)
                frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(os.path.join(ARGS.out, f"all_stations_{name}.xlsx"), index=False)
    print(f"Done in {(time.time() - run_started) / 60:.1f} min -> {ARGS.out}", flush=True)
    if failed:
        sys.exit(f"failed stations: {', '.join(sorted(failed))}; see {log_dir}")


if __name__ == "__main__":
    if ARGS.worker is None:
        main()
    elif ARGS.worker[0] == "ASSEMBLE":
        assemble()
    else:
        run_one_chain(ARCHS[ARGS.worker[0]], ARGS.worker[1], ARGS.worker[2])
