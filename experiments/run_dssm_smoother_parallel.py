"""DSSM with a bidirectional inference network (smoother encoder): copy of run_dssm_parallel.py
with two more architectures, so that the deterministic path and the encoder can be compared 2 x 2:

    --arch          deterministic path d_t (prior, emission)   encoder q(z_t | z_{t-1}, .)
    gru             causal GRU over x_{1:t}                     causal GRU (filter):  x_{1:t}, y_{1:t}
    bigru           BiGRU over x_{1:T}                          causal GRU (filter):  x_{1:t}, y_{1:t}
    gru_smooth      causal GRU over x_{1:t}                     BiGRU (smoother):     x_{1:T}, y_{1:T}
    bigru_smooth    BiGRU over x_{1:T}                          BiGRU (smoother):     x_{1:T}, y_{1:T}
"""


import argparse, os, subprocess, sys, time

VARIANT_ORDER = ("DSSM_Annual", "DSSM_SeasonalConcat", "DSSM_MonthConcat")
ARCHS = {"bigru": "DSSM_BiGRU", "gru": "DSSM_GRU", "gru_smooth": "DSSM_GRU_SmoothEnc", "bigru_smooth": "DSSM_BiGRU_SmoothEnc"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--arch", nargs="+", default=["gru_smooth", "bigru_smooth"], choices=list(ARCHS))
parser.add_argument("--variants", nargs="+", default=list(VARIANT_ORDER), choices=list(VARIANT_ORDER))
parser.add_argument("--predictors", default="8", choices=["8", "all"], help="8: the eight reference daily means (default); all: all 19 ERA5 predictors")
parser.add_argument("--stations", default="all", help="comma-separated station slugs, or all (every era5_daily_cell_<station>.npz of --data-dir)")
parser.add_argument("--station", default=None, help=argparse.SUPPRESS)
parser.add_argument("--data-dir", default="Data1CellNew", help="folder of the station files")
parser.add_argument("--seed", type=int, default=456)
parser.add_argument("--gpu-workers", type=int, default=16, help="chain processes sharing the GPU at once (only with --gpu-variants)")
parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1, help="chain processes on the CPU at once")
parser.add_argument("--gpu-variants", nargs="*", default=[], choices=list(VARIANT_ORDER), help="variants trained on the GPU (default: none; the CPU is faster for this model)")
parser.add_argument("--no-mps", action="store_true", help="do not start CUDA MPS")
parser.add_argument("--resume", action="store_true", help="continue the run in --out: skip the chains and evaluations already finished")
parser.add_argument("--threads", type=int, default=1, help="intra-op threads per CPU worker")
parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help=argparse.SUPPRESS)
parser.add_argument("--out", default=None, help="run folder (default: Results/DSSM_smoother_<8|19>vars/<timestamp>)")
parser.add_argument("--window", type=int, default=512, help="training window (truncated backpropagation length, days) [512]")
parser.add_argument("--max-epochs", type=int, default=None, help="for quick tests only")
parser.add_argument("--test-members", type=int, default=None, help="for quick tests only")
parser.add_argument("--worker", nargs=3, metavar=("ARCH", "VARIANT", "GROUP"), help=argparse.SUPPRESS)
ARGS = parser.parse_args()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
os.environ["MKL_NUM_THREADS"] = str(ARGS.threads)
ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
if ARGS.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # no CUDA context for CPU workers
PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
if ARGS.out is None:
    from datetime import datetime as _dt
    _set = "8vars" if ARGS.predictors == "8" else "19vars"
    ARGS.out = os.path.join("Results", f"DSSM_smoother_{_set}", _dt.now().strftime("%Y%m%d_%H%M%S"))


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


IS_SCHEDULER = ARGS.worker is None
if IS_SCHEDULER:
    ARGS.device = "cpu"                         # the scheduler never needs CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
IS_CHAIN = ARGS.worker is not None and ARGS.worker[1] != "ASSEMBLE"
if ARGS.station is None:
    ARGS.station = list_stations()[0]   # the scheduler loads one station to list the chains
ARCH = ARGS.worker[0] if ARGS.worker else ARGS.arch[0]


# ---- Libraries and configuration -----------------------------------------------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from precipitation_metrics import crps_ensemble_fair, evaluate_annual_seasons_months


class Config:
    def __init__(self):
        self.tau = 1.0
        self.hidden_dim = 32
        self.context_dim = 32
        self.epsilon_alpha = 1e-6
        self.epsilon_mu = 1e-6
        self.epsilon_q = 0.01

        self.latent_dim = 8

        self.bidirectional = ARCH in ("bigru", "bigru_smooth")
        self.context_margin = 64        # days of context on each side of a training window and before a forecast
        self.smoother = ARCH.endswith("_smooth")

        self.window = 512
        self.learning_rate = 5e-4
        self.gradient_clip = 5.0
        self.train_members = 40
        self.validation_members = 100
        self.test_members = 1000
        self.max_epochs = 10000
        self.validation_every = 5
        self.patience = 6
        self.improvement_tolerance = 1e-3
        self.seeds = (456,)

        self.predictors = ("d2m_mean", "t2m_mean", "msl_mean", "tcwv_mean", "u850_mean", "v850_mean", "z850_mean", "q850_mean")
        if ARGS.predictors == "all":
            self.predictors = ("d2m_mean", "d2m_max", "t2m_mean", "t2m_min", "t2m_max",
                               "msl_mean", "msl_min", "tcwv_mean", "tcwv_max", "cape_max",
                               "u850_mean", "v850_mean", "z850_mean", "q850_mean", "q850_max",
                               "omega_up850_mean", "omega_up850_max", "r850_mean", "r850_max")
        self.variants = ("DSSM_Annual", "DSSM_SeasonalConcat", "DSSM_MonthConcat")
        self.seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),}

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
        self.output_dir = Path(ARGS.out) / ARGS.station / ARCHS[ARCH]

    def as_dict(self):
        return {
            "tau": self.tau, "hidden_dim": self.hidden_dim, "context_dim": self.context_dim,
            "epsilon_alpha": self.epsilon_alpha, "epsilon_mu": self.epsilon_mu,
            "epsilon_q": self.epsilon_q,
            "latent_dim": self.latent_dim,
            "bidirectional": self.bidirectional, "context_margin": self.context_margin,
            "smoother": self.smoother,
            "window": self.window, "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip, "train_members": self.train_members,
            "validation_members": self.validation_members, "test_members": self.test_members,
            "max_epochs": self.max_epochs, "validation_every": self.validation_every,
            "patience": self.patience, "improvement_tolerance": self.improvement_tolerance,
            "seeds": list(self.seeds), "predictors": list(self.predictors),
            "variants": list(self.variants),
            "seasons": {name: list(months) for name, months in self.seasons.items()},
            "train_period": [self.train_start, self.train_end],
            "validation_period": [self.validation_start, self.validation_end],
            "test_period": [self.test_start, self.test_end],
            "extreme_reference_period": [self.extreme_reference_start, self.extreme_reference_end],
            "extreme_quantiles": self.extreme_quantiles,
            "device": str(self.device), "era5_path": str(self.era5_path),
            "precipitation_path": str(self.precipitation_path),
            "output_dir": str(self.output_dir),
        }


CFG = Config()
CFG.seeds = (ARGS.seed,)
CFG.window = ARGS.window
CFG.variants = VARIANT_ORDER            # fixes variant_id for the chain seeds
if ARGS.max_epochs is not None:
    CFG.max_epochs = ARGS.max_epochs
if ARGS.test_members is not None:
    CFG.test_members = ARGS.test_members
torch.set_num_threads(ARGS.threads)
torch.use_deterministic_algorithms(True)
# the station files are written by the ASSEMBLE process only
WRITE_STATION_FILES = not IS_SCHEDULER and not IS_CHAIN
if WRITE_STATION_FILES:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    (CFG.output_dir / "configs.json").write_text(json.dumps(CFG.as_dict(), indent=2))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def make_generator(model, seed):
    # Independent generator per sampler, so each chain is reproducible on its own.
    return torch.Generator(device=next(model.parameters()).device).manual_seed(seed)


# ---- Data: split and standardisation -------------------------------------------------------------
with np.load(CFG.era5_path) as data:
    variable_names = data["variables"].astype(str).tolist()
    columns = [variable_names.index(name) for name in CFG.predictors]
    X_raw = data["data"][:, columns].astype(np.float64)
    dates = pd.DatetimeIndex(data["dates"].astype("datetime64[ns]"))

y_raw = np.load(CFG.precipitation_path).astype(np.float64)
y_all = np.where(y_raw <= CFG.tau, 0.0, y_raw)
y_tensor = torch.tensor(y_all, dtype=torch.float32, device=CFG.device)

months = dates.month.to_numpy()
train_rows = np.flatnonzero((dates >= CFG.train_start) & (dates <= CFG.train_end))
validation_rows = np.flatnonzero((dates >= CFG.validation_start) & (dates <= CFG.validation_end))
test_rows = np.flatnonzero((dates >= CFG.test_start) & (dates <= CFG.test_end))
history_rows = np.concatenate([train_rows, validation_rows])

x_mean = X_raw[train_rows].mean(axis=0)
x_std = X_raw[train_rows].std(axis=0, ddof=0)
x_std = np.where(x_std == 0, 1.0, x_std)
X_standardized = (X_raw - x_mean) / x_std
if WRITE_STATION_FILES:
    np.savez(CFG.output_dir / "scaler.npz", mean=x_mean, std=x_std)

split_rows = {"train": train_rows, "validation": validation_rows, "test": test_rows}
split_table = pd.DataFrame([{"split": name, "first": dates[rows[0]], "last": dates[rows[-1]], "days": len(rows)} for name, rows in split_rows.items()])
if WRITE_STATION_FILES:
    split_table.to_excel(CFG.output_dir / "splits.xlsx", index=False)


# P95, P98 and P99 are estimated once, from the wet days of 1980-2017 only.
extreme_reference_rows = np.flatnonzero((dates >= CFG.extreme_reference_start) & (dates <= CFG.extreme_reference_end))
reference_rainfall = y_all[extreme_reference_rows]
reference_wet_rainfall = reference_rainfall[reference_rainfall > CFG.tau]
threshold_by_level = {level: float(np.quantile(reference_wet_rainfall, quantile)) for level, quantile in CFG.extreme_quantiles.items()}
extreme_thresholds = pd.DataFrame({"quantile": list(CFG.extreme_quantiles.values()), "threshold_mm_day": list(threshold_by_level.values())})
extreme_thresholds["reference_first"] = dates[extreme_reference_rows[0]]
extreme_thresholds["reference_last"] = dates[extreme_reference_rows[-1]]
extreme_thresholds["reference_wet_days"] = len(reference_wet_rainfall)
if WRITE_STATION_FILES:
    extreme_thresholds.to_excel(CFG.output_dir / "extreme_thresholds.xlsx", index=False)


extreme_groups = [("Annual", "ALL", tuple(range(1, 13)))]
extreme_groups += [("Seasonal", name, group_months) for name, group_months in CFG.seasons.items()]
extreme_groups += [("Monthly", f"{month:02d}", (month,)) for month in range(1, 13)]
extreme_periods = {"reference": extreme_reference_rows, "test": test_rows,}
extreme_count_rows = []
for configuration, group, group_months in extreme_groups:
    row = {"configuration": configuration, "group": group}
    for period, period_rows in extreme_periods.items():
        selected_rows = period_rows[np.isin(months[period_rows], group_months)]
        rainfall = y_all[selected_rows]
        row[f"{period}_days"] = len(rainfall)
        row[f"{period}_wet_days"] = int((rainfall > CFG.tau).sum())
        for level, threshold in threshold_by_level.items():
            row[f"{period}_{level}_exceedance_days"] = int((rainfall > threshold).sum())
    extreme_count_rows.append(row)
extreme_counts_by_group = pd.DataFrame(extreme_count_rows)
if WRITE_STATION_FILES:
    extreme_counts_by_group.to_excel(CFG.output_dir / "extreme_counts_by_group.xlsx", index=False)


# ---- Chains of the three training scales ---------------------------------------------------------
def build_chains(rows, variant):
    if variant == "DSSM_Annual":
        return {"ALL": rows}

    if variant == "DSSM_SeasonalConcat":
        groups = CFG.seasons
    else:
        groups = {f"{month:02d}": (month,) for month in range(1, 13)}

    chains = {}
    for group, group_months in groups.items():
        selected = np.isin(months[rows], group_months)
        chains[group] = rows[selected]
    return chains


chain_description = []
for variant in CFG.variants:
    for split, rows in split_rows.items():
        for group, indices in build_chains(rows, variant).items():
            chain_description.append({"variant": variant, "split": split, "group": group, "days": len(indices),})
chain_description = pd.DataFrame(chain_description)
if WRITE_STATION_FILES:
    chain_description.to_excel(CFG.output_dir / "temporal_chains.xlsx", index=False)


# ---- Model, objective, prediction and training ---------------------------------------------------
class DSSM(nn.Module):
    """Deep state-space model: (Bi)GRU context h_t, latent transition, Bernoulli-Gamma emission
    and a causal encoder q(z_t | z_{t-1}, x_{1:t}, y_{1:t})."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # context h_t of size hidden_dim (hidden_dim // 2 units per direction for the BiGRU)
        if CFG.bidirectional:
            self.path = nn.GRU(input_dim, CFG.hidden_dim // 2, batch_first=True, bidirectional=True)
        else:
            self.path = nn.GRU(input_dim, CFG.hidden_dim, batch_first=True)
        self.emission_head = nn.Sequential(nn.Linear(latent_dim + CFG.hidden_dim, CFG.hidden_dim), nn.Tanh(), nn.Linear(CFG.hidden_dim, 3))

        self.transition = nn.Sequential(nn.Linear(latent_dim + CFG.hidden_dim, CFG.hidden_dim), nn.Tanh(), nn.Linear(CFG.hidden_dim, 2 * latent_dim))

        # Encoder on a_t = [x_t, d_t, log(1 + y_t)]; bidirectional for the smoother encoder
        if CFG.smoother:
            self.encoder = nn.GRU(input_dim + 2, CFG.context_dim // 2, batch_first=True, bidirectional=True)
        else:
            self.encoder = nn.GRU(input_dim + 2, CFG.context_dim, batch_first=True)
        self.posterior_out = nn.Linear(latent_dim + CFG.context_dim, 2 * latent_dim)

        self.register_buffer("anchor", torch.zeros(1))
        with torch.no_grad():
            # Start from a clearly stochastic transition
            self.transition[-1].bias[latent_dim:] = math.log(math.expm1(0.5))

    def initial(self, n_members):
        z = self.anchor.new_zeros(n_members, self.latent_dim)
        hidden = self.anchor.new_zeros(1, 1, CFG.context_dim)
        return z, hidden

    def features(self, x, carry=None, pad=(0, 0)):
        """Context h_t. For the BiGRU, pad = (left, right) margin rows that are cropped away."""
        if CFG.bidirectional:
            context, _ = self.path(x.unsqueeze(0))
            return context[0, pad[0]:len(x) - pad[1]], None
        context, carry = self.path(x.unsqueeze(0), carry)
        return context[0], carry

    def prior(self, z_previous, feature_t):
        inputs = torch.cat([z_previous, feature_t.expand(len(z_previous), -1)], dim=-1)
        mean, rho = self.transition(inputs).chunk(2, dim=-1)
        return mean, CFG.epsilon_q + F.softplus(rho)

    def encode(self, x, y, hidden, pad=(0, 0)):
        """Encoder context. For the smoother, x and y contain pad = (left, right) margin rows that
        are cropped away, and no hidden state is carried (returns None)."""
        wet = (y > CFG.tau).to(x.dtype)
        gru_input = torch.cat([x, wet[:, None], torch.log1p(y)[:, None]], dim=-1)
        if CFG.smoother:
            context, _ = self.encoder(gru_input.unsqueeze(0))
            return context[0, pad[0]:len(x) - pad[1]], None
        context, hidden = self.encoder(gru_input.unsqueeze(0), hidden)
        return context[0], hidden

    def posterior(self, z_previous, context_t):
        inputs = torch.cat([z_previous, context_t.expand(len(z_previous), -1)], dim=-1)
        mean, rho = self.posterior_out(inputs).chunk(2, dim=-1)
        return mean, CFG.epsilon_q + F.softplus(rho)

    def emission(self, z, feature_t):
        raw = self.emission_head(torch.cat([z, feature_t.expand(len(z), -1)], dim=-1))
        u, v, wet_logit = raw.unbind(dim=-1)
        alpha = CFG.epsilon_alpha + F.softplus(u)
        mu = CFG.epsilon_mu + F.softplus(v)
        return {"alpha": alpha, "mu": mu, "beta": alpha / mu, "wet_logit": wet_logit, "wet_probability": torch.sigmoid(wet_logit)}


def bg_log_prob(y, wet_logit, alpha, beta):
    wet = y > CFG.tau
    excess = torch.where(wet, y - CFG.tau, torch.ones_like(y))
    gamma_log_prob = (alpha * torch.log(beta) - torch.lgamma(alpha) + (alpha - 1) * torch.log(excess) - beta * excess)
    log_prob = torch.where(wet, F.logsigmoid(wet_logit) + gamma_log_prob, F.logsigmoid(-wet_logit))
    return torch.where((y == 0) | wet, log_prob, torch.full_like(log_prob, -torch.inf))


def gaussian_log_prob(value, mean, std):
    """Diagonal Gaussian log density, summed over the latent dimension."""
    return (-0.5 * math.log(2.0 * math.pi) - torch.log(std) - 0.5 * ((value - mean) / std) ** 2).sum(dim=-1)


def train_window(model, x, y, z_previous, hidden, carry, rng, x_path=None, pad=(0, 0), x_encoder=None, y_encoder=None, encoder_pad=(0, 0)):
    """One truncated-backpropagation window of the stepwise importance-weighted objective: nll gives the gradients of
    the generative parameters, the surrogate the DReG gradients of the encoder."""
    feature, carry = model.features(x if x_path is None else x_path, carry, pad)
    if x_encoder is None:
        context, hidden = model.encode(x, y, hidden)
    else:
        context, hidden = model.encode(x_encoder, y_encoder, hidden, encoder_pad)

    n_members = len(z_previous)
    log_weights, frozen_weights = [], []

    for t in range(len(y)):
        p_mean, p_std = model.prior(z_previous, feature[t])
        noise = torch.randn(p_std.shape, dtype=x.dtype, device=x.device, generator=rng)
        q_mean, q_std = model.posterior(z_previous, context[t])
        z = q_mean + q_std * noise

        eta = model.emission(z, feature[t])
        log_emission = bg_log_prob(y[t], eta["wet_logit"], eta["alpha"], eta["beta"])
        transition = gaussian_log_prob(z, p_mean, p_std)
        log_weights.append(log_emission + transition - gaussian_log_prob(z, q_mean, q_std))
        # Copy with the q parameters stopped, used only in the DReG surrogate.
        frozen_weights.append(log_emission + transition - gaussian_log_prob(z, q_mean.detach(), q_std.detach()))
        z_previous = z

    stacked = torch.stack(log_weights)
    bound = torch.logsumexp(stacked, dim=1) - math.log(n_members)
    nll = -bound.mean()
    normalised = torch.softmax(stacked, dim=1)
    ess = (1.0 / normalised.square().sum(dim=1)).mean()

    frozen = torch.stack(frozen_weights)
    weights = torch.softmax(frozen, dim=1).detach()
    surrogate = -(weights.square() * frozen).sum(dim=1).mean()
    return nll, surrogate, ess, z_previous, hidden, carry


@torch.no_grad()
def filter_history(model, x_history, y_history, n_members, seed):
    """Run the encoder over the observed history and sample the boundary state z_T0."""
    z, hidden = model.initial(n_members)
    rng = make_generator(model, seed)
    # For the causal GRU, the final hidden state (carry) continues into the forecast period.
    _, carry = model.features(x_history)
    context, hidden = model.encode(x_history, y_history, hidden)
    for t in range(len(y_history)):
        mean, std = model.posterior(z, context[t])
        z = mean + std * torch.randn(std.shape, dtype=z.dtype, device=z.device, generator=rng)
    return z, carry


@torch.no_grad()
def propagate_prior(model, x_future, boundary, carry, transition_seed, emission_seed, noise_scale=1.0, x_left=None):
    """Prior propagation from the boundary state. x_left: predictor rows placed before x_future
    as left context of the BiGRU (cropped)."""
    z = boundary.clone()
    transition_rng = make_generator(model, transition_seed)
    if CFG.bidirectional and x_left is not None:
        feature, _ = model.features(torch.cat([x_left, x_future]), None, (len(x_left), 0))
    else:
        feature, _ = model.features(x_future, carry)
    fields = ["rainfall", "alpha", "mu", "beta", "wet_logit", "wet_probability"]
    output = {name: [] for name in fields}

    tau_tensor = torch.tensor(CFG.tau, dtype=torch.float64, device=z.device)
    minimum_wet = torch.nextafter(tau_tensor, torch.full_like(tau_tensor, torch.inf))
    cuda_devices = [z.device.index] if z.is_cuda else []

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(emission_seed)
        for t in range(len(x_future)):
            mean, std = model.prior(z, feature[t])
            noise = torch.randn(std.shape, dtype=z.dtype, device=z.device, generator=transition_rng)
            z = mean + noise_scale * std * noise
            eta = model.emission(z, feature[t])
            gamma = torch.distributions.Gamma(eta["alpha"].double(), eta["beta"].double()).sample()
            wet = torch.rand_like(gamma) < eta["wet_probability"]
            amount = torch.maximum(CFG.tau + gamma, minimum_wet)
            values = {"rainfall": torch.where(wet, amount, torch.zeros_like(amount)), **eta}
            for name in fields:
                output[name].append(values[name].cpu())
    return {name: torch.stack(values).numpy() for name, values in output.items()}


def predict(model, x_history, y_history, x_future, n_members, seed, noise_scale=1.0):
    """Filter the history, then propagate the prior over x_future."""
    boundary, carry = filter_history(model, x_history, y_history, n_members, seed + 1)
    x_left = x_history[-CFG.context_margin:] if CFG.bidirectional else None
    return propagate_prior(model, x_future, boundary, carry, transition_seed=seed + 2, emission_seed=seed + 3, noise_scale=noise_scale, x_left=x_left)


def train_chain(x_train, y_train, x_validation, y_validation, seed, checkpoint_path, metadata):
    """Train one chain with early stopping on the validation fair CRPS."""
    set_seed(seed)
    model = DSSM(x_train.shape[1], CFG.latent_dim).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    rng = make_generator(model, seed)

    encoder_names = ("encoder", "posterior_out")
    encoder = [p for name, p in model.named_parameters() if name.startswith(encoder_names)]
    generative = [p for name, p in model.named_parameters() if not name.startswith(encoder_names)]

    best_crps, patience_crps, bad_checks, history = float("inf"), float("inf"), 0, []
    observed_validation = y_validation.cpu().numpy()

    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        z, hidden = model.initial(CFG.train_members)
        carry = None
        totals = np.zeros(3)

        for start in range(0, len(y_train), CFG.window):
            stop = min(start + CFG.window, len(y_train))
            optimizer.zero_grad(set_to_none=True)
            path_start = max(0, start - CFG.context_margin)
            path_stop = min(len(y_train), stop + CFG.context_margin)
            margins = (start - path_start, path_stop - stop)
            if CFG.bidirectional:
                x_path, pad = x_train[path_start:path_stop], margins
            else:
                x_path, pad = None, (0, 0)
            if CFG.smoother:
                x_encoder, y_encoder, encoder_pad = x_train[path_start:path_stop], y_train[path_start:path_stop], margins
            else:
                x_encoder, y_encoder, encoder_pad = None, None, (0, 0)
            nll, surrogate, ess, z, hidden, carry = train_window(model, x_train[start:stop], y_train[start:stop], z, hidden, carry, rng, x_path=x_path, pad=pad, x_encoder=x_encoder, y_encoder=y_encoder, encoder_pad=encoder_pad)

            # Generative parameters: gradient of the objective. Encoder: DReG surrogate.
            generative_grads = torch.autograd.grad(nll, generative, retain_graph=True, allow_unused=True)
            encoder_grads = torch.autograd.grad(surrogate, encoder, allow_unused=True)
            for parameter, gradient in zip(generative + encoder, list(generative_grads) + list(encoder_grads)):
                parameter.grad = torch.zeros_like(parameter) if gradient is None else gradient

            nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
            optimizer.step()
            z = z.detach()
            hidden = hidden if hidden is None else hidden.detach()
            carry = carry if carry is None else carry.detach()
            totals += np.array([float(nll.detach()), float(nll.detach()), float(ess.detach())]) * (stop - start)

        log = dict(zip(("loss", "nll", "ess"), totals / len(y_train)))
        log["epoch"] = epoch
        log["validation_crps"] = np.nan

        if epoch % CFG.validation_every == 0 or epoch == CFG.max_epochs:
            model.eval()
            validation_prediction = predict(model, x_train, y_train, x_validation, CFG.validation_members, seed + 20000)
            crps = float(crps_ensemble_fair(observed_validation, validation_prediction["rainfall"], CFG.tau).mean())
            log["validation_crps"] = crps

            if crps < best_crps:
                best_crps = crps
                checkpoint = {
                    "state_dict": {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()},
                    "input_dim": x_train.shape[1], "epoch": epoch,
                    "validation_crps": crps, "seed": seed,
                    "settings": CFG.as_dict(), "metadata": metadata,
                }
                torch.save(checkpoint, checkpoint_path)

            if crps < patience_crps - CFG.improvement_tolerance:
                patience_crps, bad_checks = crps, 0
            else:
                bad_checks += 1

            print(f"{metadata['variant']}/{metadata['group']} | epoch {epoch} | " f"loss {log['loss']:.4f} | ess {log['ess']:.1f} | " f"validation CRPS {crps:.4f}", flush=True)

        history.append(log)
        if bad_checks >= CFG.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=x_train.device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, pd.DataFrame(history), checkpoint


# ---- Parallel execution --------------------------------------------------------------------------
def chain_folder(variant, seed):
    folder = CFG.output_dir / variant / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_one_chain(variant, group):
    """Train one chain and save its checkpoint, training log and test predictions."""
    seed = CFG.seeds[0]
    variant_id = VARIANT_ORDER.index(variant)
    x = torch.tensor(X_standardized, dtype=torch.float32, device=CFG.device)
    train_chains = build_chains(train_rows, variant)
    group_id = list(train_chains).index(group)
    tr = train_chains[group]
    va = build_chains(validation_rows, variant)[group]
    past = build_chains(history_rows, variant)[group]
    te = build_chains(test_rows, variant)[group]
    chain_seed = int(np.random.SeedSequence([seed, variant_id, group_id]).generate_state(1)[0])
    folder = chain_folder(variant, seed)
    metadata = {"variant": variant, "group": group, "scaler_mean": x_mean.tolist(), "scaler_std": x_std.tolist(), "train_first": str(dates[tr[0]].date()), "train_last": str(dates[tr[-1]].date())}
    started = time.perf_counter()
    model, training_log, checkpoint = train_chain(x[tr], y_tensor[tr], x[va], y_tensor[va], chain_seed, folder / f"{group}.pt", metadata)
    training_log.to_excel(folder / f"{group}_training.xlsx", index=False)
    prediction = predict(model, x[past], y_tensor[past], x[te], CFG.test_members, chain_seed + 30000)
    quiet = predict(model, x[past], y_tensor[past], x[te], CFG.test_members, chain_seed + 30000, noise_scale=0.0)
    np.savez_compressed(folder / f"{group}_test.npz", rows=te, rainfall_noise_off=quiet["rainfall"].astype(np.float32), **prediction)
    log = {"station": ARGS.station, "variant": variant, "seed": seed, "group": group, "chain_seed": chain_seed,
           "device": str(CFG.device), "threads": ARGS.threads if CFG.device.type == "cpu" else None,
           "train_days": len(tr), "validation_days": len(va), "history_days": len(past),
           "test_days": len(te), "selected_epoch": checkpoint["epoch"],
           "epochs_run": len(training_log), "validation_crps": checkpoint["validation_crps"],
           "final_ess": float(training_log["ess"].iloc[-1]),
           "minutes": (time.perf_counter() - started) / 60}
    pd.DataFrame([log]).to_json(folder / f"{group}_chain_log.json", orient="records")
    print(f"DONE {ARGS.station} {ARCH} {variant}/{group} on {CFG.device} | epoch {log['selected_epoch']} | " f"validation CRPS {log['validation_crps']:.4f} | {log['minutes']:.1f} min", flush=True)


def assemble_variant(variant):
    """Rebuild the calendar-ordered test ensemble of one variant and score it."""
    from precipitation_metrics import evaluate_point_metrics
    seed = CFG.seeds[0]
    folder = chain_folder(variant, seed)
    groups = list(build_chains(train_rows, variant))
    combined, chain_logs = {}, []
    quiet_rainfall = np.empty((len(test_rows), CFG.test_members), dtype=np.float32)
    for group in groups:
        with np.load(folder / f"{group}_test.npz") as saved:
            positions = np.searchsorted(test_rows, saved["rows"])
            for name in ["rainfall", "alpha", "mu", "beta", "wet_logit", "wet_probability"]:
                values = saved[name]
                if name not in combined:
                    combined[name] = np.empty((len(test_rows), *values.shape[1:]), dtype=values.dtype)
                combined[name][positions] = values
            quiet_rainfall[positions] = saved["rainfall_noise_off"]
        chain_logs.append(pd.read_json(folder / f"{group}_chain_log.json", orient="records"))
    np.savez_compressed(folder / "test_ensemble.npz", dates=dates[test_rows].to_numpy(), **combined)

    predictive_mean = (combined["wet_probability"] * (CFG.tau + combined["mu"])).mean(axis=1)
    evaluation = evaluate_annual_seasons_months(
        observed=y_all[test_rows], dates=dates[test_rows].to_numpy(), threshold=CFG.tau,
        wet_probability=combined["wet_probability"], alpha=combined["alpha"],
        beta=combined["beta"], ensemble=combined["rainfall"], predicted_mean=predictive_mean)
    crps_on = float(crps_ensemble_fair(y_all[test_rows], combined["rainfall"].astype(np.float64), CFG.tau).mean())
    crps_off = float(crps_ensemble_fair(y_all[test_rows], quiet_rainfall.astype(np.float64), CFG.tau).mean())
    chain_log = pd.concat(chain_logs, ignore_index=True)
    chain_log["CRPS_noise_on"], chain_log["CRPS_noise_off"] = crps_on, crps_off

    summary = evaluation["summary"]; summary.insert(0, "seed", seed); summary.insert(0, "variant", variant)
    daily = evaluation["daily"]; daily.insert(0, "seed", seed); daily.insert(0, "variant", variant)
    members = evaluation["members"]; members.insert(0, "seed", seed); members.insert(0, "variant", variant)
    daily.to_excel(folder / "daily_test_scores.xlsx", index=False)
    summary.to_excel(folder / "evaluation_summary.xlsx", index=False)
    members.to_excel(folder / "member_metrics.xlsx", index=False)

    points = evaluate_point_metrics(
        observed=y_all[test_rows], dates=dates[test_rows].to_numpy().astype("datetime64[D]"),
        ensemble=combined["rainfall"], threshold=CFG.tau, wet_probability=combined["wet_probability"],
        alpha=combined["alpha"], beta=combined["beta"], model_name=f"{variant}_seed{seed}")
    print(f"== {ARGS.station} {ARCH} {variant} seed {seed} | CRPS {crps_on:.4f} | noise off {crps_off:.4f}", flush=True)
    return summary, members, chain_log, points


def gpu_environment():
    """Environment of the GPU processes: the scheduler itself hides the GPU, its children must not."""
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if ORIGINAL_CUDA_VISIBLE_DEVICES is not None:
        env["CUDA_VISIBLE_DEVICES"] = ORIGINAL_CUDA_VISIBLE_DEVICES
    return env


def start_mps():
    """Start the CUDA MPS daemon unless it already runs. Returns True when this script started it."""
    if ARGS.no_mps or subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "nvidia-cuda-mps-control"], capture_output=True).returncode == 0:
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
    if not gpu_available and ARGS.gpu_variants:
        print("No GPU found: every chain runs on the CPU", flush=True)
    gpu_variants = set(ARGS.gpu_variants) if gpu_available else set()
    gpu_slots = ARGS.gpu_workers if gpu_variants else 0
    cpu_slots = ARGS.cpu_workers
    if gpu_variants != set(ARGS.variants) and cpu_slots < 1:
        sys.exit("--cpu-workers must be >= 1 when some variants run on the CPU")

    # Chains in priority order: longest variants first, then station, then architecture.
    chains = [(station, arch, variant, group)
              for variant in VARIANT_ORDER if variant in ARGS.variants
              for station in stations for arch in ARGS.arch
              for group in build_chains(train_rows, variant)]
    device_of = {chain: "cuda" if chain[2] in gpu_variants else "cpu" for chain in chains}
    if ARGS.resume:
        # A chain is finished once its <group>_chain_log.json exists (written last by run_one_chain);
        # an evaluation once point_metrics_monthly.xlsx exists (written last by assemble_all).
        def chain_done(c):
            return os.path.exists(os.path.join(ARGS.out, c[0], ARCHS[c[1]], c[2], f"seed{ARGS.seed}", f"{c[3]}_chain_log.json"))
        def unit_done(station, arch):
            return os.path.exists(os.path.join(ARGS.out, station, ARCHS[arch], "point_metrics_monthly.xlsx"))
        n_before = len(chains)
        chains = [c for c in chains if not chain_done(c)]
        print(f"Resume: {n_before - len(chains)} chains already finished, {len(chains)} to run", flush=True)
    remaining = {(station, arch): sum(1 for c in chains if c[:2] == (station, arch)) for station in stations for arch in ARGS.arch}
    queues = {"cuda": [c for c in chains if device_of[c] == "cuda"], "cpu": [c for c in chains if device_of[c] == "cpu"]}
    assemble_queue, failed_units = [], set()
    if ARGS.resume:
        assemble_queue = [key for key, n in remaining.items() if n == 0 and not unit_done(*key)]

    log_dir = os.path.join(ARGS.out, "logs"); os.makedirs(log_dir, exist_ok=True)
    print(f"{len(stations)} station(s), {len(chains)} chains: {len(queues['cuda'])} on GPU ({gpu_slots} at once), "
          f"{len(queues['cpu'])} on CPU ({cpu_slots} at once x {ARGS.threads} thread(s)) -> {ARGS.out}", flush=True)
    mps_started = start_mps() if gpu_slots else False

    def common_options(station, device):
        options = ["--seed", str(ARGS.seed), "--window", str(ARGS.window), "--threads", str(ARGS.threads), "--out", ARGS.out, "--predictors", ARGS.predictors,
                   "--station", station, "--data-dir", ARGS.data_dir, "--device", device, "--variants", *ARGS.variants]
        if ARGS.max_epochs is not None:
            options += ["--max-epochs", str(ARGS.max_epochs)]
        if ARGS.test_members is not None:
            options += ["--test-members", str(ARGS.test_members)]
        return options

    def launch(kind, unit, device):
        if kind == "chain":
            station, arch, variant, group = unit
            command = [sys.executable, os.path.abspath(__file__), "--worker", arch, variant, group, *common_options(station, device)]
            log_name = f"{station}_{arch}_{variant}_{group}.log"
        else:
            station, arch = unit
            command = [sys.executable, os.path.abspath(__file__), "--worker", arch, "ASSEMBLE", "-", *common_options(station, "cpu")]
            log_name = f"{station}_{arch}_ASSEMBLE.log"
        env = gpu_environment()
        log = open(os.path.join(log_dir, log_name), "w")
        return {"kind": kind, "unit": unit, "device": device, "process": subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env),
                "log": log, "started": time.time()}

    running, records, run_started = [], [], time.time()
    n_total = len(chains) + sum(1 for key, n in remaining.items() if n > 0) + len(assemble_queue)
    try:
        while True:
            # CPU slots: evaluations first (they unlock finished stations), then CPU chains.
            while sum(r["device"] == "cpu" for r in running) < max(cpu_slots, 1 if assemble_queue else 0) and (assemble_queue or queues["cpu"]):
                running.append(launch("assemble", assemble_queue.pop(0), "cpu") if assemble_queue else launch("chain", queues["cpu"].pop(0), "cpu"))
            while sum(r["device"] == "cuda" for r in running) < gpu_slots and queues["cuda"]:
                running.append(launch("chain", queues["cuda"].pop(0), "cuda"))
            if not running:
                break
            time.sleep(1.0)
            for r in [r for r in running if r["process"].poll() is not None]:
                running.remove(r); r["log"].close()
                code, seconds = r["process"].returncode, time.time() - r["started"]
                unit = r["unit"]
                records.append({"kind": r["kind"], "station": unit[0], "arch": unit[1], "variant": unit[2] if r["kind"] == "chain" else "ASSEMBLE",
                                "group": unit[3] if r["kind"] == "chain" else "", "device": r["device"], "return_code": code,
                                "minutes": seconds / 60, "finished_after_min": (time.time() - run_started) / 60})
                status = "finished" if code == 0 else "FAILED"
                print(f"[{len(records)}/{n_total} | {(time.time() - run_started) / 60:6.1f} min] {status} {r['kind']} {'/'.join(unit)} on {r['device']} ({seconds / 60:.1f} min)", flush=True)
                if r["kind"] == "chain":
                    key = unit[:2]
                    if code != 0:
                        failed_units.add(key)
                    remaining[key] -= 1
                    if remaining[key] == 0 and key not in failed_units:
                        assemble_queue.append(key)
                elif code != 0:
                    failed_units.add(unit)
    finally:
        for r in running:
            r["process"].kill()
        if mps_started:
            stop_mps()

    pd.DataFrame(records).to_excel(os.path.join(ARGS.out, "scheduler_log.xlsx"), index=False)
    for name in ("scores", "chain_log", "point_metrics_overall", "point_metrics_seasonal", "point_metrics_monthly"):
        frames = []
        for station in stations:
            for arch in ARGS.arch:
                path = os.path.join(ARGS.out, station, ARCHS[arch], f"{name}.xlsx")
                if os.path.exists(path):
                    frame = pd.read_excel(path)
                    frame.insert(0, "architecture", ARCHS[arch])
                    if "station" not in frame:
                        frame.insert(0, "station", station)
                    frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(os.path.join(ARGS.out, f"all_stations_{name}.xlsx"), index=False)
    print(f"Done in {(time.time() - run_started) / 60:.1f} min -> {ARGS.out}", flush=True)
    if failed_units:
        sys.exit(f"failed: {', '.join('/'.join(u) for u in sorted(failed_units))}; see {log_dir}")


def assemble_all():
    summaries, members, logs, overall, seasonal, monthly = [], [], [], [], [], []
    for variant in ARGS.variants:
        summary, member, chain_log, points = assemble_variant(variant)
        summaries.append(summary); members.append(member); logs.append(chain_log)
        for frame, store in ((points["overall"], overall), (points["seasonal"], seasonal), (points["monthly"], monthly)):
            frame.insert(1, "variant", variant); store.append(frame)
    pd.concat(summaries, ignore_index=True).to_excel(CFG.output_dir / "scores.xlsx", index=False)
    pd.concat(members, ignore_index=True).to_excel(CFG.output_dir / "member_metrics.xlsx", index=False)
    pd.concat(logs, ignore_index=True).to_excel(CFG.output_dir / "chain_log.xlsx", index=False)
    for name, frames in (("overall", overall), ("seasonal", seasonal), ("monthly", monthly)):
        pd.concat(frames, ignore_index=True).to_excel(CFG.output_dir / f"point_metrics_{name}.xlsx", index=False)


if __name__ == "__main__":
    if ARGS.worker is None:
        main()
    elif ARGS.worker[1] == "ASSEMBLE":
        assemble_all()
    else:
        run_one_chain(ARGS.worker[1], ARGS.worker[2])
