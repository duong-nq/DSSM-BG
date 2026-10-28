"""Evaluation metrics for daily precipitation downscaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammainc, gammaln, logsumexp
from scipy.stats import rankdata


PERIOD_MONTHS = {
    "Annual": tuple(range(1, 13)),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

MONTH_PERIODS = {f"{month:02d}": (month,) for month in range(1, 13)}

METRIC_UNITS = {
    "R01": "fraction",
    "AUC": "unitless",
    "Brier": "unitless",
    "SDII": "mm/day",
    "Spearman": "unitless",
    "SD": "mm/day",
    "P95wet": "mm/day",
    "P98wet": "mm/day",
    "P99wet": "mm/day",
    "CRPS": "mm/day",
    "WW": "probability/fraction",
    "CDD": "days",
    "CWD": "days",
}

TRAJECTORY_METRICS = ("R01", "SDII", "SD", "P95wet", "P98wet", "P99wet", "WW", "CDD", "CWD")


def metric_metadata():
    """Definitions, units and preferred direction for every metric."""
    rows = [
        ("R01", "Occurrence", "fraction", "diagnostic", "Fraction of days with precipitation > threshold"),
        ("AUC", "Occurrence", "unitless", "higher", "ROC AUC for the wet-day target"),
        ("Brier", "Occurrence", "unitless", "lower", "Mean squared error of wet-day probability"),
        ("SDII", "Intensity", "mm/day", "diagnostic", "Mean precipitation on wet days"),
        ("Spearman", "Intensity", "unitless", "higher", "Rank correlation of observed precipitation and predictive mean"),
        ("SD", "Distribution", "mm/day", "diagnostic", "Sample standard deviation of daily precipitation"),
        ("P95wet", "Distribution", "mm/day", "diagnostic", "95th percentile of wet-day precipitation"),
        ("P98wet", "Distribution", "mm/day", "diagnostic", "98th percentile of wet-day precipitation"),
        ("P99wet", "Distribution", "mm/day", "diagnostic", "99th percentile of wet-day precipitation"),
        ("CRPS", "Distribution", "mm/day", "lower", "Mean continuous ranked probability score"),
        ("WW", "Temporal", "probability/fraction", "diagnostic", "P(wet today | wet yesterday)"),
        ("CDD", "Temporal", "days", "diagnostic", "Maximum consecutive dry-day spell"),
        ("CWD", "Temporal", "days", "diagnostic", "Maximum consecutive wet-day spell"),
    ]
    return pd.DataFrame(rows, columns=["metric", "category", "unit", "preferred", "definition"])


def threshold_precipitation(values, threshold=1.0):
    """Set precipitation at or below the wet-day threshold to zero."""
    values = np.asarray(values, dtype=float)
    return np.where(values > threshold, values, 0.0)


def wet_day(values, threshold=1.0):
    """Wet indicator using the strict definition precipitation > threshold."""
    return np.asarray(values, dtype=float) > threshold


def r01(values, threshold=1.0):
    """Wet-day frequency as a fraction in [0, 1]."""
    wet = wet_day(values, threshold)
    return float(np.mean(wet)) if wet.size else np.nan


def sdii(values, threshold=1.0):
    """Mean precipitation on wet days (mm/day)."""
    values = np.asarray(values, dtype=float)
    wet_values = values[wet_day(values, threshold)]
    return float(np.mean(wet_values)) if wet_values.size else np.nan


def precipitation_sd(values, threshold=1.0):
    """Sample SD of thresholded daily precipitation (mm/day)."""
    values = threshold_precipitation(values, threshold)
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def wet_day_percentile(values, percentile, threshold=1.0):
    """Percentile conditional on a wet day (mm/day)."""
    values = np.asarray(values, dtype=float)
    wet_values = values[wet_day(values, threshold)]
    return float(np.percentile(wet_values, percentile)) if wet_values.size else np.nan


def _consecutive_day_pairs(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    if dates.size < 2:
        return np.zeros(0, dtype=bool)
    return np.diff(dates).astype("timedelta64[D]").astype(int) == 1


def ww_probability(values, dates, threshold=1.0):
    """P(wet today | wet yesterday), only for consecutive calendar days."""
    wet = wet_day(values, threshold)
    consecutive = _consecutive_day_pairs(dates)
    previous_wet = wet[:-1] & consecutive
    denominator = np.sum(previous_wet)
    if denominator == 0:
        return np.nan
    return float(np.sum(previous_wet & wet[1:]) / denominator)


def _longest_spell(values, dates, target_wet, threshold=1.0):
    wet = wet_day(values, threshold)
    dates = np.asarray(dates, dtype="datetime64[D]")
    target = wet if target_wet else ~wet
    longest = 0
    current = 0

    for index, matches in enumerate(target):
        if index > 0 and (dates[index] - dates[index - 1]).astype(int) != 1:
            current = 0
        current = current + 1 if matches else 0
        longest = max(longest, current)

    return int(longest)


def cdd(values, dates, threshold=1.0):
    """Maximum number of consecutive dry days."""
    return _longest_spell(values, dates, target_wet=False, threshold=threshold)


def cwd(values, dates, threshold=1.0):
    """Maximum number of consecutive wet days."""
    return _longest_spell(values, dates, target_wet=True, threshold=threshold)


def brier_score(observed, wet_probability, threshold=1.0):
    """Brier score for the event precipitation > threshold."""
    target = wet_day(observed, threshold).astype(float)
    probability = np.clip(np.asarray(wet_probability, dtype=float), 0.0, 1.0)
    return float(np.mean((probability - target) ** 2))


def auc_score(observed, wet_probability, threshold=1.0):
    """ROC AUC for wet occurrence, computed from ranks."""
    target = wet_day(observed, threshold).astype(int)
    probability = np.asarray(wet_probability, dtype=float)
    n_wet = int(np.sum(target == 1))
    n_dry = int(np.sum(target == 0))
    if n_wet == 0 or n_dry == 0:
        return np.nan
    ranks = rankdata(probability, method="average")
    rank_sum_wet = np.sum(ranks[target == 1])
    return float((rank_sum_wet - n_wet * (n_wet + 1) / 2) / (n_wet * n_dry))


def spearman_correlation(observed, predicted_mean, threshold=1.0):
    """Spearman correlation of observed rain and predictive mean."""
    observed = threshold_precipitation(observed, threshold)
    predicted_mean = threshold_precipitation(predicted_mean, threshold)
    valid = np.isfinite(observed) & np.isfinite(predicted_mean)
    if np.sum(valid) < 2:
        return np.nan
    rank_observed = rankdata(observed[valid], method="average")
    rank_predicted = rankdata(predicted_mean[valid], method="average")
    if np.std(rank_observed) == 0 or np.std(rank_predicted) == 0:
        return np.nan
    return float(np.corrcoef(rank_observed, rank_predicted)[0, 1])


def crps_bernoulli_gamma_analytic(observed, wet_probability, alpha, beta, threshold=1.0):
    """Daily analytic CRPS for 0 or threshold + Gamma(alpha, rate=beta)."""
    observed = threshold_precipitation(observed, threshold)
    wet_probability = np.clip(np.asarray(wet_probability, dtype=float), 0.0, 1.0)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    shifted_observed = np.maximum(observed - threshold, 0.0)
    gamma_mean = alpha / beta
    gamma_cdf = gammainc(alpha, beta * shifted_observed)
    truncated_first_moment = gamma_mean * gammainc(alpha + 1.0, beta * shifted_observed)
    expected_absolute_error = shifted_observed * (2.0 * gamma_cdf - 1.0) + gamma_mean - 2.0 * truncated_first_moment
    mean_absolute_gamma_difference = 2.0 * np.exp(gammaln(alpha + 0.5) - gammaln(alpha)) / (beta * np.sqrt(np.pi))
    half_expected_pair_distance = wet_probability * (1.0 - wet_probability) * (threshold + gamma_mean) + 0.5 * wet_probability**2 * mean_absolute_gamma_difference

    dry = observed == 0.0
    crps_dry = wet_probability * (threshold + gamma_mean) - half_expected_pair_distance
    crps_wet = (1.0 - wet_probability) * observed + wet_probability * expected_absolute_error - half_expected_pair_distance
    return np.where(dry, crps_dry, crps_wet)


def crps_ensemble_fair(observed, ensemble, threshold=1.0):
    """Daily fair ensemble CRPS; ensemble shape is (time, member)."""
    observed = threshold_precipitation(observed, threshold)
    ensemble = threshold_precipitation(ensemble, threshold)
    n_members = ensemble.shape[1]
    if n_members < 2:
        return np.full(observed.shape, np.nan, dtype=float)

    first_term = np.mean(np.abs(ensemble - observed[:, None]), axis=1)
    ordered = np.sort(ensemble, axis=1)
    coefficients = 2.0 * np.arange(1, n_members + 1) - n_members - 1.0
    pairwise_sum = np.sum(ordered * coefficients[None, :], axis=1)
    correction = pairwise_sum / (n_members * (n_members - 1.0))
    return first_term - correction


def twcrps_ensemble_fair(observed, ensemble, level, threshold=1.0):
    """Threshold-weighted fair CRPS with weight 1{z >= level}: the CRPS of max(X, level) and max(y, level)."""
    observed = threshold_precipitation(observed, threshold)
    ensemble = threshold_precipitation(ensemble, threshold)
    return crps_ensemble_fair(np.maximum(observed, level),
                              np.maximum(np.asarray(ensemble, dtype=float), level),
                              threshold=0.0)


def trajectory_metrics(values, dates, threshold=1.0):
    """Metrics for one complete observed or simulated trajectory."""
    values = threshold_precipitation(values, threshold)
    return {
        "R01": r01(values, threshold),
        "SDII": sdii(values, threshold),
        "SD": precipitation_sd(values, threshold),
        "P95wet": wet_day_percentile(values, 95, threshold),
        "P98wet": wet_day_percentile(values, 98, threshold),
        "P99wet": wet_day_percentile(values, 99, threshold),
        "WW": ww_probability(values, dates, threshold),
        "CDD": cdd(values, dates, threshold),
        "CWD": cwd(values, dates, threshold),
    }


def ensemble_trajectory_metrics(ensemble, dates, threshold=1.0):
    """Trajectory metrics separately for every ensemble member."""
    ensemble = np.asarray(ensemble, dtype=float)
    rows = []
    for member in range(ensemble.shape[1]):
        row = {"member": member}
        row.update(trajectory_metrics(ensemble[:, member], dates, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_wet_probability(wet_probability):
    wet_probability = np.asarray(wet_probability, dtype=float)
    if wet_probability.ndim == 1:
        return wet_probability
    return np.mean(wet_probability, axis=1)


def _predictive_mean(wet_probability, alpha, beta, ensemble, threshold):
    if wet_probability is not None and alpha is not None and beta is not None:
        component_mean = np.asarray(wet_probability) * (threshold + np.asarray(alpha) / np.asarray(beta))
        return component_mean if component_mean.ndim == 1 else np.mean(component_mean, axis=1)
    if ensemble is not None:
        return np.mean(threshold_precipitation(ensemble, threshold), axis=1)
    return None


def daily_predictive_scores(
    observed,
    threshold=1.0,
    wet_probability=None,
    alpha=None,
    beta=None,
    ensemble=None,
    predicted_mean=None,
    crps_daily=None,
):
    """Daily scores. The CRPS is crps_daily if given, else the fair ensemble CRPS, else the analytic one."""
    observed = threshold_precipitation(observed, threshold)
    ensemble_array = None if ensemble is None else np.asarray(ensemble, dtype=float)

    if wet_probability is None and ensemble_array is not None:
        probability = np.mean(wet_day(ensemble_array, threshold), axis=1)
    elif wet_probability is not None:
        probability = _mean_wet_probability(wet_probability)
    else:
        probability = np.full(observed.shape, np.nan)
    probability = np.clip(probability, 0.0, 1.0)

    if predicted_mean is None:
        predicted_mean = _predictive_mean(wet_probability, alpha, beta, ensemble_array, threshold)
    if predicted_mean is None:
        predicted_mean = np.full(observed.shape, np.nan)

    has_parameters = wet_probability is not None and alpha is not None and beta is not None
    parameters_are_daily = has_parameters and all(np.asarray(value).ndim == 1 for value in (wet_probability, alpha, beta))

    if crps_daily is not None:
        crps = np.asarray(crps_daily, dtype=float)
        crps_method = "provided"
    elif ensemble_array is not None:
        crps = crps_ensemble_fair(observed, ensemble_array, threshold)
        crps_method = "fair ensemble"
    elif parameters_are_daily:
        crps = crps_bernoulli_gamma_analytic(observed, wet_probability, alpha, beta, threshold)
        crps_method = "analytic Bernoulli-Gamma"
    else:
        crps = np.full(observed.shape, np.nan)
        crps_method = "unavailable"

    wet_observed = wet_day(observed, threshold)
    return pd.DataFrame(
        {
            "observed": observed,
            "wet_observed": wet_observed.astype(int),
            "wet_probability": probability,
            "predicted_mean": np.asarray(predicted_mean, dtype=float),
            "Brier": (probability - wet_observed) ** 2,
            "CRPS": crps,
            "CRPS_method": crps_method,
        }
    )


def evaluate_period(
    observed,
    dates,
    period="Annual",
    threshold=1.0,
    wet_probability=None,
    alpha=None,
    beta=None,
    ensemble=None,
    predicted_mean=None,
    crps_daily=None,
):
    """Evaluate one held-out period; return summary, members and daily rows."""
    observed = threshold_precipitation(observed, threshold)
    dates = np.asarray(dates, dtype="datetime64[D]")
    daily = daily_predictive_scores(
        observed=observed,
        threshold=threshold,
        wet_probability=wet_probability,
        alpha=alpha,
        beta=beta,
        ensemble=ensemble,
        predicted_mean=predicted_mean,
        crps_daily=crps_daily,
    )

    observed_metrics = trajectory_metrics(observed, dates, threshold)
    if ensemble is not None:
        member_metrics = ensemble_trajectory_metrics(ensemble, dates, threshold)
    else:
        member_metrics = ensemble_trajectory_metrics(daily["predicted_mean"].to_numpy()[:, None], dates, threshold)

    row = {
        "period": period,
        "n_days": len(observed),
        "n_wet_observed": int(np.sum(wet_day(observed, threshold))),
        "wet_threshold_mm_day": float(threshold),
        "AUC": auc_score(observed, daily["wet_probability"], threshold),
        "Brier": float(np.mean(daily["Brier"])),
        "Spearman": spearman_correlation(observed, daily["predicted_mean"], threshold),
        "CRPS": float(np.mean(daily["CRPS"])),
        "CRPS_method": daily["CRPS_method"].iloc[0],
    }

    for metric in TRAJECTORY_METRICS:
        model_value = float(member_metrics[metric].mean())
        row[f"{metric}_observed"] = observed_metrics[metric]
        row[f"{metric}_model"] = model_value
        row[f"{metric}_bias"] = model_value - observed_metrics[metric]

    member_metrics.insert(0, "period", period)
    return row, member_metrics, daily


def _slice_time(values, mask):
    if values is None:
        return None
    return np.asarray(values)[mask]


def evaluate_annual_seasons_months(
    observed,
    dates,
    threshold=1.0,
    wet_probability=None,
    alpha=None,
    beta=None,
    ensemble=None,
    predicted_mean=None,
    crps_daily=None,
    model_name=None,
):
    """Scores of the whole period, the seasons and the 12 months; ensemble has shape (time, member)."""
    observed = np.asarray(observed, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[D]")
    months = pd.DatetimeIndex(dates).month.to_numpy()

    summary_rows = []
    member_tables = []
    annual_daily = None
    periods = PERIOD_MONTHS.copy()
    periods.update(MONTH_PERIODS)

    for period, period_months in periods.items():
        if period == "Annual":
            mask = np.ones(len(dates), dtype=bool)
        else:
            mask = np.isin(months, period_months)
        row, members, daily = evaluate_period(
            observed=observed[mask],
            dates=dates[mask],
            period=period,
            threshold=threshold,
            wet_probability=_slice_time(wet_probability, mask),
            alpha=_slice_time(alpha, mask),
            beta=_slice_time(beta, mask),
            ensemble=_slice_time(ensemble, mask),
            predicted_mean=_slice_time(predicted_mean, mask),
            crps_daily=_slice_time(crps_daily, mask),
        )
        summary_rows.append(row)
        member_tables.append(members)
        if period == "Annual":
            annual_daily = daily.copy()

    summary = pd.DataFrame(summary_rows)
    members = pd.concat(member_tables, ignore_index=True)
    annual_daily.insert(0, "date", dates)
    season_masks = [np.isin(months, PERIOD_MONTHS[name]) for name in ("DJF", "MAM", "JJA")]
    season = np.select(season_masks, ["DJF", "MAM", "JJA"], default="SON")
    annual_daily.insert(1, "season", season)
    annual_daily.insert(2, "month", months)

    if model_name is not None:
        summary.insert(0, "model", model_name)
        members.insert(0, "model", model_name)
        annual_daily.insert(0, "model", model_name)

    summary.attrs["units"] = METRIC_UNITS.copy()
    summary.attrs["wet_day_definition"] = f"precipitation > {threshold} mm/day"
    return {"summary": summary, "daily": annual_daily, "members": members, "metadata": metric_metadata()}


SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

POINT_METRICS = ("Delta_mu", "Delta_sigma", "MAE", "RMSE", "COR", "NLL")


def bernoulli_gamma_log_likelihood(observed, wet_probability, alpha, beta, threshold=1.0):
    """Daily log predictive density of 0 or threshold + Gamma(alpha, rate=beta). With parameters of shape
    (time, member), the predictive is the equal-weight mixture over the members."""
    observed = threshold_precipitation(observed, threshold)
    wet_probability = np.clip(np.asarray(wet_probability, dtype=float), 1e-12, 1.0 - 1e-12)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    mixture = wet_probability.ndim == 2
    wet = observed > 0.0
    excess = np.maximum(observed - threshold, 1e-12)
    if mixture:
        wet, excess = wet[:, None], excess[:, None]

    log_wet = (np.log(wet_probability) + alpha * np.log(beta) - gammaln(alpha)
               + (alpha - 1.0) * np.log(excess) - beta * excess)
    log_density = np.where(wet, log_wet, np.log1p(-wet_probability))
    if not mixture:
        return log_density
    return logsumexp(log_density, axis=1) - np.log(log_density.shape[1])


def absolute_mean_difference(observed, predicted):
    """|mean(prediction) - mean(observation)|. This is a bias in the group mean, not MAE."""
    return float(abs(np.mean(np.asarray(predicted, dtype=float))
                     - np.mean(np.asarray(observed, dtype=float))))


def absolute_sd_difference(observed, predicted):
    """|sd(prediction) - sd(observation)| using the population sd, denominator |T_g|."""
    return float(abs(np.std(np.asarray(predicted, dtype=float), ddof=0)
                     - np.std(np.asarray(observed, dtype=float), ddof=0)))


def mean_absolute_error(observed, predicted):
    """mean(|prediction - observation|)."""
    return float(np.mean(np.abs(np.asarray(predicted, dtype=float)
                                - np.asarray(observed, dtype=float))))


def root_mean_squared_error(observed, predicted):
    """sqrt(mean((prediction - observation)^2))."""
    difference = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return float(np.sqrt(np.mean(np.square(difference))))


def pearson_correlation(observed, predicted):
    """Pearson correlation over finite pairs; NaN when it is undefined rather than an error."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 2:
        return np.nan
    observed, predicted = observed[valid], predicted[valid]
    if np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return np.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def point_metrics(observed, predicted, log_likelihood=None):
    """Point metrics of the daily ensemble mean, and NLL from the daily log predictive densities."""
    row = {
        "n_days": int(np.size(observed)),
        "Delta_mu": absolute_mean_difference(observed, predicted),
        "Delta_sigma": absolute_sd_difference(observed, predicted),
        "MAE": mean_absolute_error(observed, predicted),
        "RMSE": root_mean_squared_error(observed, predicted),
        "COR": pearson_correlation(observed, predicted),
    }
    row["NLL"] = (np.nan if log_likelihood is None
                  else float(-np.mean(np.asarray(log_likelihood, dtype=float))))
    return row


def evaluate_point_metrics(
    observed,
    dates,
    ensemble,
    threshold=1.0,
    wet_probability=None,
    alpha=None,
    beta=None,
    log_likelihood=None,
    model_name=None,
):
    """Point metrics overall, by season and by month; the seasonal and monthly tables end with their unweighted mean."""
    observed = threshold_precipitation(observed, threshold)
    predicted = np.mean(threshold_precipitation(ensemble, threshold), axis=1)
    dates = np.asarray(dates, dtype="datetime64[D]")
    if not (len(observed) == len(predicted) == len(dates)):
        raise ValueError(f"length mismatch: observed {len(observed)}, "
                         f"predicted {len(predicted)}, dates {len(dates)}")

    if log_likelihood is None and wet_probability is not None:
        log_likelihood = bernoulli_gamma_log_likelihood(
            observed, wet_probability, alpha, beta, threshold)
    log_likelihood = None if log_likelihood is None else np.asarray(log_likelihood, dtype=float)

    months = pd.DatetimeIndex(dates).month.to_numpy()

    def rows_for(masks_by_label, mean_label):
        collected = []
        for label, mask in masks_by_label:
            entry = point_metrics(observed[mask], predicted[mask],
                                  None if log_likelihood is None else log_likelihood[mask])
            collected.append({"group": label, **entry})
        frame = pd.DataFrame(collected)
        average = {"group": mean_label, "n_days": int(frame["n_days"].sum())}
        average.update({name: float(frame[name].mean()) for name in POINT_METRICS})
        return pd.concat([frame, pd.DataFrame([average])], ignore_index=True)

    overall = pd.DataFrame([{"group": "Overall",
                             **point_metrics(observed, predicted, log_likelihood)}])
    seasonal = rows_for([(name, np.isin(months, PERIOD_MONTHS[name])) for name in SEASON_ORDER],
                        "Seasonal_mean")
    monthly = rows_for([(MONTH_LABELS[month - 1], months == month) for month in range(1, 13)],
                       "Monthly_mean")

    tables = {"overall": overall, "seasonal": seasonal, "monthly": monthly}
    if model_name is not None:
        for frame in tables.values():
            frame.insert(0, "model", model_name)
    return tables


__all__ = [
    "METRIC_UNITS",
    "MONTH_LABELS",
    "MONTH_PERIODS",
    "PERIOD_MONTHS",
    "POINT_METRICS",
    "SEASON_ORDER",
    "absolute_mean_difference",
    "absolute_sd_difference",
    "auc_score",
    "bernoulli_gamma_log_likelihood",
    "brier_score",
    "cdd",
    "crps_bernoulli_gamma_analytic",
    "crps_ensemble_fair",
    "cwd",
    "daily_predictive_scores",
    "ensemble_trajectory_metrics",
    "evaluate_annual_seasons_months",
    "evaluate_period",
    "evaluate_point_metrics",
    "mean_absolute_error",
    "metric_metadata",
    "pearson_correlation",
    "point_metrics",
    "precipitation_sd",
    "r01",
    "root_mean_squared_error",
    "sdii",
    "spearman_correlation",
    "threshold_precipitation",
    "trajectory_metrics",
    "twcrps_ensemble_fair",
    "wet_day",
    "wet_day_percentile",
    "ww_probability",
]
