#!/usr/bin/env Rscript
# GLM-BG and RF-BG at every station of --data-dir, run as a task graph on a pool of single-threaded R workers.

Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1", NUMEXPR_NUM_THREADS = "1")

suppressPackageStartupMessages({
  library(parallel)
  library(reticulate)
  library(jsonlite)
})

# ---- Command line -------------------------------------------------------------------------------
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
PROJECT <- if (length(script_file)) normalizePath(dirname(script_file[1])) else getwd()
setwd(PROJECT)

OPT <- list(data_dir = "Data1CellNew", stations = "all", predictors = "8", workers = parallel::detectCores(), seed = 42L,
            ntree = 500L, rf_mtry = 2L, rf_minbucket = 10L, rf_minsplit = 30L, rf_maxdepth = 10L, members = 1000L, chunk_seconds = 30, save_models = "zstd", out = NA_character_,
            python = normalizePath(file.path(R.home(), "..", "..", "bin", "python"), mustWork = FALSE))
cli <- commandArgs(trailingOnly = TRUE)
if (length(cli) %% 2L != 0L) stop("Options are given as pairs: --name value")
for (i in seq_len(length(cli) %/% 2L) * 2L - 1L) {
  key <- gsub("-", "_", sub("^--", "", cli[i]))
  if (!key %in% names(OPT)) stop("Unknown option: ", cli[i])
  OPT[[key]] <- if (is.numeric(OPT[[key]])) as(cli[i + 1L], class(OPT[[key]])) else cli[i + 1L]
}
stopifnot(OPT$predictors %in% c("8", "all"), OPT$save_models %in% c("zstd", "gzip", "xz", "none", "no"), OPT$workers >= 1L)

predictor_set <- if (OPT$predictors == "8") "8vars" else "19vars"
if (is.na(OPT$out)) OPT$out <- file.path("Results", paste0("glm_rf_allstations_", predictor_set), paste0("GLM_RF", format(Sys.time(), "%Y%m%d_%H%M%S")))
dir.create(OPT$out, recursive = TRUE, showWarnings = FALSE)
RUN_LOG <- file.path(OPT$out, "run.log")
log_line <- function(..., console = TRUE) {
  line <- paste0(format(Sys.time(), "%H:%M:%S"), "  ", sprintf(...))
  cat(line, "\n", sep = "", file = RUN_LOG, append = TRUE)
  if (console) cat(line, "\n", sep = "")
}

# ---- Configuration -------------------------------------------------------------------------------
CFG <- list(
  tau = 1.0,
  seed = OPT$seed,
  ensemble_members = OPT$members,
  predictors = c("d2m_mean", "t2m_mean", "msl_mean", "tcwv_mean", "u850_mean", "v850_mean", "z850_mean", "q850_mean"),
  families = c("GLM", "RF"),
  time_scales = c("Annual", "SeasonalConcat", "MonthConcat"),
  seasons = list(DJF = c(12L, 1L, 2L), MAM = 3L:5L, JJA = 6L:8L, SON = 9L:11L),
  train_start = as.Date("1980-01-01"),
  train_end = as.Date("2017-12-31"),
  test_start = as.Date("2018-01-01"),
  test_end = as.Date("2025-12-31"),
  extreme_quantiles = c(P95 = 0.95, P98 = 0.98, P99 = 0.99),
  glm_maxit = 500L,
  glm_stepsize = 0.5,
  rf_ntree = OPT$ntree,
  rf_mtry = OPT$rf_mtry,
  rf_minbucket = OPT$rf_minbucket,
  rf_minsplit = OPT$rf_minsplit,
  rf_maxdepth = OPT$rf_maxdepth,
  rf_predict_chunk = 128L,
  python_bin = OPT$python,
  model_compress = OPT$save_models
)
# seconds per tree ~ RF_COST * n_train^2: only cuts the forests into blocks and orders the tasks
RF_COST <- 2.2e-8
month_groups <- setNames(lapply(1:12, as.integer), sprintf("%02d", 1:12))

# ---- Shared helpers ------------------------------------------------------------------------------
py_env <- new.env()
get_py <- function() {
  if (is.null(py_env$np)) {
    use_python(CFG$python_bin, required = TRUE)
    py_env$np <- import("numpy", convert = FALSE)
    py_env$pm <- import_from_path("precipitation_metrics", path = PROJECT)
  }
  py_env
}
write_xlsx <- function(data, path) { get_py(); r_to_py(as.data.frame(data), convert = FALSE)$to_excel(path, index = FALSE); invisible(path) }

seed_of <- function(...) {
  h <- 0
  for (code in utf8ToInt(paste(CFG$seed, ..., sep = "|"))) h <- (h * 31 + code) %% 2147483629
  as.integer(h)
}

model_folder <- function(s, model_name) file.path(s$out_dir, model_name, paste0("seed", CFG$seed))
group_model_file <- function(s, model_name, group) file.path(model_folder(s, model_name), "_groups", paste0(group, ".rds"))

build_groups <- function(s, rows, time_scale) {
  month_sets <- if (time_scale == "Annual") list(ALL = 1L:12L) else if (time_scale == "SeasonalConcat") CFG$seasons else month_groups
  lapply(month_sets, function(group_months) rows[s$months[rows] %in% group_months])
}

sanitize_parameters <- function(parameters) {
  parameters <- as.matrix(parameters[, c("p", "alpha", "beta"), drop = FALSE])
  storage.mode(parameters) <- "double"
  parameters[, "p"] <- pmin(1, pmax(0, parameters[, "p"]))
  parameters[, "alpha"] <- pmax(1e-8, parameters[, "alpha"])
  parameters[, "beta"] <- pmax(1e-8, parameters[, "beta"])
  parameters
}

fit_glm_group <- function(s, rows) {
  occurrence_formula <- reformulate(CFG$predictors, response = "wet")
  intensity_formula <- reformulate(CFG$predictors, response = "excess")
  frame <- data.frame(wet = s$wet_all[rows], excess = s$excess_all[rows], s$X_standardized[rows, , drop = FALSE])
  wet_frame <- frame[frame$wet == 1L, , drop = FALSE]
  occurrence <- vglm(occurrence_formula, binomialff(link = "logitlink"), data = frame)
  intensity <- vglm(intensity_formula, gamma2(lmu = "loglink", lshape = "loglink", zero = NULL, parallel = FALSE), data = wet_frame, control = vglm.control(stepsize = CFG$glm_stepsize, maxit = CFG$glm_maxit))
  list(occurrence = occurrence, intensity = intensity, train_days = nrow(frame), train_wet_days = nrow(wet_frame))
}

predict_glm_parameters <- function(s, model, rows) {
  newdata <- data.frame(s$X_standardized[rows, , drop = FALSE])
  p <- plogis(drop(predict(model$occurrence, newdata = newdata, type = "link")))
  eta <- as.matrix(predict(model$intensity, newdata = newdata, type = "link"))
  mu <- exp(eta[, 1])
  alpha <- exp(eta[, 2])
  sanitize_parameters(cbind(p = p, alpha = alpha, beta = alpha / mu))
}

# One tree per randomForestTrain call, each with its own seed, so a forest does not depend on how it is cut into blocks.
fit_rf_trees <- function(s, rows, trees, seed_key) {
  x <- unname(s$X_standardized[rows, , drop = FALSE])
  y <- s$excess_all[rows]
  multiresponse <- NA
  forest <- lapply(trees, function(tree_index) {
    tree_seed <- seed_of(seed_key, "tree", tree_index)
    set.seed(tree_seed)
    .C("c_srand", tree_seed)  # the rpart fork draws the mtry variables with C rand(), which set.seed does not reach
    one <- randomForestTrain(x = x, y = y, ntree = 1L, mtry = CFG$rf_mtry, minbucket = CFG$rf_minbucket, minsplit = CFG$rf_minsplit, maxdepth = CFG$rf_maxdepth, method = "bernoulliGammaLLMME", resample = TRUE, replace = TRUE, parallel.plan = NA, keep.y = TRUE, progress.bar = FALSE)
    multiresponse <<- attr(one, "multiresponse")
    one[[1]]
  })
  list(trees = forest, multiresponse = multiresponse)
}

predict_rf_parameters <- function(s, model, rows) {
  estimate_bg_leaf <- getFromNamespace("estimateBernoulliGammabc3", "RandomForestDist")
  predictors <- unname(s$X_standardized[rows, , drop = FALSE])
  chunks <- split(seq_len(nrow(predictors)), ceiling(seq_len(nrow(predictors)) / CFG$rf_predict_chunk))
  parameters <- do.call(rbind, lapply(chunks, function(index) {
    leaves <- randomForestPredict(model$forest, newdata = predictors[index, , drop = FALSE], method = "leaves", parallel.plan = NA)
    t(vapply(leaves, function(values) estimate_bg_leaf(as.numeric(values), non_informative_beta = 1), numeric(3)))
  }))
  colnames(parameters) <- c("p", "alpha", "beta")
  sanitize_parameters(parameters)
}

simulate_bg <- function(parameters, members = CFG$ensemble_members) {
  parameters <- sanitize_parameters(parameters)
  n <- nrow(parameters)
  wet <- matrix(runif(n * members), nrow = n) < parameters[, "p"]
  excess <- matrix(rgamma(n * members, shape = rep(parameters[, "alpha"], members), rate = rep(parameters[, "beta"], members)), nrow = n)
  ifelse(wet, CFG$tau + excess, 0.0)
}

# ---- Tasks (run on the workers) -----------------------------------------------------------------
task_glm <- function(task) {
  s <- STATIONS[[task$station]]
  train <- build_groups(s, s$train_rows, task$time_scale)[[task$group]]
  test <- build_groups(s, s$test_rows, task$time_scale)[[task$group]]
  model <- fit_glm_group(s, train)
  parameters <- predict_glm_parameters(s, model, test)
  if (CFG$model_compress != "no") {
    dir.create(dirname(group_model_file(s, task$model, task$group)), recursive = TRUE, showWarnings = FALSE)
    saveRDS(model, group_model_file(s, task$model, task$group), compress = FALSE)
  }
  list(group = task$group, parameters = parameters, train_days = model$train_days, train_wet_days = model$train_wet_days, test_days = length(test))
}

task_rf_chunk <- function(task) {
  s <- STATIONS[[task$station]]
  train <- build_groups(s, s$train_rows, task$time_scale)[[task$group]]
  block <- fit_rf_trees(s, train, task$trees, paste(task$station, task$model, task$group, sep = "|"))
  dir.create(dirname(task$chunk_file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(block, task$chunk_file, compress = FALSE)
  list(trees = length(task$trees))
}

task_rf_predict <- function(task) {
  s <- STATIONS[[task$station]]
  train <- build_groups(s, s$train_rows, task$time_scale)[[task$group]]
  test <- build_groups(s, s$test_rows, task$time_scale)[[task$group]]
  blocks <- lapply(task$chunk_files, readRDS)
  forest <- structure(do.call(c, lapply(blocks, `[[`, "trees")), multiresponse = blocks[[1]]$multiresponse, class = "RandomForestDist")
  rm(blocks)
  model <- list(forest = forest, train_days = length(train), train_wet_days = sum(s$wet_all[train] == 1L))
  parameters <- predict_rf_parameters(s, model, test)
  if (CFG$model_compress != "no") {
    dir.create(dirname(group_model_file(s, task$model, task$group)), recursive = TRUE, showWarnings = FALSE)
    saveRDS(model, group_model_file(s, task$model, task$group), compress = FALSE)
  }
  unlink(task$chunk_files)
  rm(model, forest); invisible(gc())
  list(group = task$group, parameters = parameters, train_days = length(train), train_wet_days = sum(s$wet_all[train] == 1L), test_days = length(test))
}

task_model_eval <- function(task) {
  s <- STATIONS[[task$station]]
  py <- get_py(); np <- py$np; precipitation_metrics <- py$pm
  model_name <- task$model
  folder <- model_folder(s, model_name)
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)

  if (CFG$model_compress != "no") {
    groups <- names(build_groups(s, s$train_rows, task$time_scale))
    fitted_groups <- setNames(lapply(groups, function(group) readRDS(group_model_file(s, model_name, group))), groups)
    saveRDS(fitted_groups, file.path(folder, "fitted_groups.rds"), compress = if (CFG$model_compress == "none") FALSE else CFG$model_compress)
    rm(fitted_groups); invisible(gc())
    unlink(file.path(folder, "_groups"), recursive = TRUE)
  }

  unlink(file.path(folder, "_chunks"), recursive = TRUE)

  # ensemble and test_ensemble.npz
  test_dates_python <- np$array(format(s$dates[s$test_rows], "%Y-%m-%d"), dtype = "datetime64[D]")
  parameters <- sanitize_parameters(task$parameters)
  set.seed(seed_of(task$station, model_name, "ensemble"))
  ensemble <- simulate_bg(parameters)
  ensemble_float32 <- np$asarray(ensemble, dtype = "float32")
  ensemble <- py_to_r(ensemble_float32$astype("float64"))
  predicted_mean <- parameters[, "p"] * (CFG$tau + parameters[, "alpha"] / parameters[, "beta"])
  np$savez_compressed(file.path(folder, "test_ensemble.npz"), dates = test_dates_python, rainfall = ensemble_float32, wet_probability = parameters[, "p"], alpha = parameters[, "alpha"], beta = parameters[, "beta"], predicted_mean = predicted_mean)

  # probabilistic scores
  period_months <- c(list(Annual = 1L:12L), CFG$seasons, month_groups)
  test_months <- s$months[s$test_rows]
  period_mean <- function(values, period) mean(values[test_months %in% period_months[[period]]])
  y_test <- s$y_all[s$test_rows]
  analytic_crps <- as.numeric(precipitation_metrics$crps_bernoulli_gamma_analytic(y_test, parameters[, "p"], parameters[, "alpha"], parameters[, "beta"], CFG$tau))
  ensemble_crps <- as.numeric(precipitation_metrics$crps_ensemble_fair(y_test, ensemble, CFG$tau))
  evaluation <- precipitation_metrics$evaluate_annual_seasons_months(observed = y_test, dates = test_dates_python, threshold = CFG$tau, wet_probability = parameters[, "p"], alpha = parameters[, "alpha"], beta = parameters[, "beta"], ensemble = ensemble, predicted_mean = predicted_mean, crps_daily = analytic_crps)

  summary <- evaluation$summary
  names(summary)[names(summary) == "CRPS"] <- "CRPS_analytic"
  summary$CRPS_method <- "analytic Bernoulli-Gamma"
  summary$CRPS_ensemble_fair <- vapply(summary$period, function(period) period_mean(ensemble_crps, period), numeric(1))
  parts <- strsplit(model_name, "_", fixed = TRUE)[[1]]
  summary <- cbind(model = model_name, family = parts[1], time_scale = paste(parts[-1], collapse = "_"), seed = CFG$seed, summary)

  daily <- evaluation$daily
  names(daily)[names(daily) == "CRPS"] <- "CRPS_analytic"
  daily$CRPS_method <- "analytic Bernoulli-Gamma"
  daily$CRPS_ensemble_fair <- ensemble_crps
  daily <- cbind(model = model_name, seed = CFG$seed, daily)
  members <- cbind(model = model_name, seed = CFG$seed, evaluation$members)
  write_xlsx(summary, file.path(folder, "evaluation_summary.xlsx"))
  write_xlsx(daily, file.path(folder, "daily_test_scores.xlsx"))
  write_xlsx(members, file.path(folder, "member_metrics.xlsx"))

  # point metrics
  forecast_dates <- as.character(py_to_r(np$datetime_as_string(test_dates_python, unit = "D")))
  stopifnot(identical(forecast_dates, format(s$dates[s$test_rows], "%Y-%m-%d")), length(y_test) == length(forecast_dates))
  observed_censored <- as.numeric(precipitation_metrics$threshold_precipitation(y_test, CFG$tau))
  sanity <- data.frame(model = model_name, n_test_days = nrow(ensemble), first_date = forecast_dates[1], last_date = forecast_dates[length(forecast_dates)], mean_observed = mean(observed_censored), mean_prediction = mean(rowMeans(ensemble)))
  tables <- precipitation_metrics$evaluate_point_metrics(observed = y_test, dates = test_dates_python, ensemble = ensemble, threshold = CFG$tau, wet_probability = parameters[, "p"], alpha = parameters[, "alpha"], beta = parameters[, "beta"], model_name = model_name)
  label <- function(frame) cbind(family = parts[1], time_scale = paste(parts[-1], collapse = "_"), frame)

  list(model = model_name, fit_log = task$fit_log, summary = summary, daily = daily, members = members, sanity = sanity,
       point_overall = label(tables$overall), point_seasonal = label(tables$seasonal), point_monthly = label(tables$monthly))
}

task_station <- function(task) {
  s <- STATIONS[[task$station]]
  get_py()
  results <- task$model_results
  stack <- function(field) { frame <- do.call(rbind, lapply(results, `[[`, field)); rownames(frame) <- NULL; frame }
  fit_log <- stack("fit_log")
  scores <- stack("summary")
  daily_scores <- stack("daily")
  member_metrics <- stack("members")
  write_xlsx(fit_log, file.path(s$out_dir, "fit_log.xlsx"))
  write_xlsx(scores, file.path(s$out_dir, "scores.xlsx"))
  write_xlsx(daily_scores, file.path(s$out_dir, "daily_test_scores.xlsx"))
  write_xlsx(member_metrics, file.path(s$out_dir, "member_metrics.xlsx"))

  metric_metadata <- py_env$pm$metric_metadata()
  metric_metadata <- metric_metadata[metric_metadata$metric != "CRPS", ]
  metric_metadata <- rbind(metric_metadata, data.frame(metric = c("CRPS_analytic", "CRPS_ensemble_fair"), category = "Distribution", unit = "mm/day", preferred = "lower", definition = c("Analytical CRPS of the Bernoulli-Gamma predictive distribution", "Fair CRPS calculated from the simulated ensemble")))
  write_xlsx(metric_metadata, file.path(s$out_dir, "evaluation_metric_metadata.xlsx"))

  headline_columns <- c("model", "AUC", "Brier", "SDII_observed", "SDII_model", "Spearman", "SD_observed", "SD_model", "P95wet_observed", "P95wet_model", "P98wet_observed", "P98wet_model", "P99wet_observed", "P99wet_model", "CRPS_analytic", "CRPS_ensemble_fair", "WW_observed", "WW_model", "CDD_observed", "CDD_model", "CWD_observed", "CWD_model")
  annual_scores <- scores[scores$period == "Annual", headline_columns]
  period_scores <- scores[, c("model", "period", "AUC", "Brier", "Spearman", "CRPS_analytic", "CRPS_ensemble_fair")]
  trajectory_columns <- metric_metadata$metric[metric_metadata$metric %in% names(member_metrics)]
  member_metric_summary <- aggregate(member_metrics[, trajectory_columns, drop = FALSE], by = list(model = member_metrics$model, period = member_metrics$period), FUN = mean, na.rm = TRUE)
  write_xlsx(annual_scores, file.path(s$out_dir, "annual_comparison.xlsx"))
  write_xlsx(period_scores, file.path(s$out_dir, "annual_seasonal_monthly_scores.xlsx"))
  write_xlsx(member_metric_summary, file.path(s$out_dir, "member_metric_summary.xlsx"))

  point <- list(overall = stack("point_overall"), seasonal = stack("point_seasonal"), monthly = stack("point_monthly"))
  for (level in names(point)) write_xlsx(point[[level]], file.path(s$out_dir, paste0("point_metrics_", level, ".xlsx")))
  write_xlsx(stack("sanity"), file.path(s$out_dir, "sanity_check.xlsx"))

  writeLines(capture.output(sessionInfo()), file.path(s$out_dir, "session_info.txt"))
  station_column <- function(frame) cbind(station = task$station, frame)
  list(scores = station_column(scores), annual = station_column(annual_scores), point_overall = station_column(point$overall),
       point_seasonal = station_column(point$seasonal), point_monthly = station_column(point$monthly), fit_log = station_column(fit_log))
}

TASK_RUNNERS <- list(glm = task_glm, rf_chunk = task_rf_chunk, rf_predict = task_rf_predict, model_eval = task_model_eval, station = task_station)

run_task <- function(task) {
  started <- proc.time()[["elapsed"]]
  warnings_seen <- character()
  value <- tryCatch(withCallingHandlers(TASK_RUNNERS[[task$type]](task), warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  }), error = function(e) structure(list(message = conditionMessage(e)), class = "task_error"))
  list(value = value, seconds = proc.time()[["elapsed"]] - started, warnings = unique(warnings_seen), pid = Sys.getpid())
}

# ---- Stations: data, split, standardisation, groups ----------------------------------------------
invisible(get_py())
np <- py_env$np
station_slugs <- sub("^era5_daily_cell_(.+)\\.npz$", "\\1", list.files(OPT$data_dir, "^era5_daily_cell_.+\\.npz$"))
station_slugs <- station_slugs[file.exists(file.path(OPT$data_dir, paste0("ecad_rr_", station_slugs, "_1980_2025.npy")))]
if (OPT$stations != "all") {
  wanted <- trimws(strsplit(OPT$stations, ",")[[1]])
  if (length(setdiff(wanted, station_slugs))) stop("Stations not found in ", OPT$data_dir, ": ", paste(setdiff(wanted, station_slugs), collapse = ", "))
  station_slugs <- wanted
}
if (!length(station_slugs)) stop("No station files in ", OPT$data_dir)

STATIONS <- list()
for (station in station_slugs) {
  era5_path <- file.path(OPT$data_dir, paste0("era5_daily_cell_", station, ".npz"))
  precipitation_path <- file.path(OPT$data_dir, paste0("ecad_rr_", station, "_1980_2025.npy"))
  era5 <- np$load(era5_path, allow_pickle = FALSE)
  X_all <- py_to_r(era5[["data"]])
  dates <- as.Date(as.character(py_to_r(np$datetime_as_string(era5[["dates"]], unit = "D"))))
  variable_names <- as.character(py_to_r(era5[["variables"]]))
  invisible(era5$close())
  if (OPT$predictors == "all" && !length(STATIONS)) CFG$predictors <- variable_names
  if (anyNA(match(CFG$predictors, variable_names))) stop(station, ": missing predictors ", paste(setdiff(CFG$predictors, variable_names), collapse = ", "))

  X_raw <- X_all[, match(CFG$predictors, variable_names), drop = FALSE]
  colnames(X_raw) <- CFG$predictors
  storage.mode(X_raw) <- "double"
  y_raw <- as.numeric(py_to_r(np$load(precipitation_path, allow_pickle = FALSE)))
  stopifnot(length(y_raw) == nrow(X_raw), !anyNA(y_raw), !anyNA(X_raw))
  y_all <- ifelse(y_raw > CFG$tau, y_raw, 0.0)
  wet_all <- as.integer(y_all > CFG$tau)
  excess_all <- ifelse(wet_all == 1L, y_all - CFG$tau, 0.0)
  months <- as.integer(format(dates, "%m"))
  train_rows <- which(dates >= CFG$train_start & dates <= CFG$train_end)
  test_rows <- which(dates >= CFG$test_start & dates <= CFG$test_end)
  x_mean <- colMeans(X_raw[train_rows, , drop = FALSE])
  x_sd <- apply(X_raw[train_rows, , drop = FALSE], 2, sd)
  x_sd[!is.finite(x_sd) | x_sd == 0] <- 1
  X_standardized <- sweep(sweep(X_raw, 2, x_mean, "-"), 2, x_sd, "/")
  colnames(X_standardized) <- CFG$predictors

  out_dir <- file.path(OPT$out, station)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  s <- list(name = station, dates = dates, months = months, X_standardized = X_standardized, y_all = y_all, wet_all = wet_all,
            excess_all = excess_all, train_rows = train_rows, test_rows = test_rows, out_dir = out_dir)
  STATIONS[[station]] <- s

  write_json(c(CFG, list(station = station, era5_path = era5_path, precipitation_path = precipitation_path, output_dir = out_dir)), file.path(out_dir, "config.json"), pretty = TRUE, auto_unbox = TRUE)
  split_table <- data.frame(split = c("train", "test"), first = c(dates[train_rows[1]], dates[test_rows[1]]), last = c(dates[tail(train_rows, 1)], dates[tail(test_rows, 1)]), days = c(length(train_rows), length(test_rows)))
  write_xlsx(split_table, file.path(out_dir, "splits.xlsx"))

  reference_wet_rainfall <- y_all[train_rows][y_all[train_rows] > CFG$tau]
  threshold_by_level <- quantile(reference_wet_rainfall, CFG$extreme_quantiles, names = TRUE, type = 7)
  extreme_thresholds <- data.frame(quantile = unname(CFG$extreme_quantiles), threshold_mm_day = unname(threshold_by_level), reference_first = dates[train_rows[1]], reference_last = dates[tail(train_rows, 1)], reference_wet_days = length(reference_wet_rainfall))
  write_xlsx(extreme_thresholds, file.path(out_dir, "extreme_thresholds.xlsx"))

  chain_rows <- list()
  for (time_scale in CFG$time_scales) for (split in c("train", "test")) {
    groups <- build_groups(s, if (split == "train") train_rows else test_rows, time_scale)
    for (group in names(groups)) chain_rows[[length(chain_rows) + 1L]] <- data.frame(time_scale = time_scale, split = split, group = group, days = length(groups[[group]]))
  }
  write_xlsx(do.call(rbind, chain_rows), file.path(out_dir, "temporal_groups.xlsx"))

  extreme_groups <- c(list(Annual = list(ALL = 1L:12L)), list(Seasonal = CFG$seasons), list(Monthly = month_groups))
  extreme_count_rows <- list()
  for (configuration in names(extreme_groups)) for (group in names(extreme_groups[[configuration]])) {
    group_months <- extreme_groups[[configuration]][[group]]
    row <- data.frame(configuration = configuration, group = group)
    for (split in c("reference", "test")) {
      rows <- if (split == "reference") train_rows else test_rows
      rainfall <- y_all[rows[months[rows] %in% group_months]]
      row[[paste0(split, "_days")]] <- length(rainfall)
      row[[paste0(split, "_wet_days")]] <- sum(rainfall > CFG$tau)
      for (level in names(threshold_by_level)) row[[paste0(split, "_", level, "_exceedance_days")]] <- sum(rainfall > threshold_by_level[[level]])
    }
    extreme_count_rows[[length(extreme_count_rows) + 1L]] <- row
  }
  write_xlsx(do.call(rbind, extreme_count_rows), file.path(out_dir, "extreme_counts_by_group.xlsx"))
}

# ---- Task graph ---------------------------------------------------------------------------------
STAGE <- c(glm = 1L, rf_chunk = 1L, rf_predict = 2L, model_eval = 3L, station = 4L)
tasks <- list()
add_task <- function(task) { task$index <- length(tasks) + 1L; tasks[[task$index]] <<- task; task$index }
for (station_index in seq_along(STATIONS)) {
  s <- STATIONS[[station_index]]
  model_tasks <- integer()
  for (family in CFG$families) for (time_scale in CFG$time_scales) {
    model_name <- paste(family, time_scale, sep = "_")
    train_groups <- build_groups(s, s$train_rows, time_scale)
    test_groups <- build_groups(s, s$test_rows, time_scale)
    group_tasks <- integer()
    for (group in names(train_groups)) {
      n_train <- length(train_groups[[group]])
      base <- list(station = s$name, station_index = station_index, model = model_name, family = family, time_scale = time_scale, group = group)
      if (family == "GLM") {
        group_tasks <- c(group_tasks, add_task(c(base, list(type = "glm", deps = integer(), cost = 1e-9 * n_train^2 + 0.5))))
      } else {
        tree_cost <- RF_COST * n_train^2
        n_blocks <- min(CFG$rf_ntree, max(1L, ceiling(CFG$rf_ntree * tree_cost / OPT$chunk_seconds)))
        blocks <- unname(split(seq_len(CFG$rf_ntree), ceiling(seq_len(CFG$rf_ntree) * n_blocks / CFG$rf_ntree)))
        chunk_files <- file.path(model_folder(s, model_name), "_chunks", sprintf("%s_%03d.rds", group, seq_along(blocks)))
        chunk_tasks <- vapply(seq_along(blocks), function(b) add_task(c(base, list(type = "rf_chunk", deps = integer(), trees = blocks[[b]], chunk_file = chunk_files[b], cost = tree_cost * length(blocks[[b]])))), integer(1))
        group_tasks <- c(group_tasks, add_task(c(base, list(type = "rf_predict", deps = chunk_tasks, chunk_files = chunk_files, cost = 0.035 * CFG$rf_ntree * length(test_groups[[group]]) / 2922 + 2))))
      }
    }
    model_tasks <- c(model_tasks, add_task(list(type = "model_eval", station = s$name, station_index = station_index, model = model_name, family = family, time_scale = time_scale, group = "", deps = group_tasks, cost = 20)))
  }
  add_task(list(type = "station", station = s$name, station_index = station_index, model = "", family = "", time_scale = "", group = "", deps = model_tasks, cost = 20))
}

# Inputs a task needs from the tasks it depends on (assembled on the master, which holds the results).
task_inputs <- function(task, results) {
  if (task$type == "model_eval") {
    s <- STATIONS[[task$station]]
    test_groups <- build_groups(s, s$test_rows, task$time_scale)
    parameters <- matrix(NA_real_, nrow = length(s$test_rows), ncol = 3L, dimnames = list(NULL, c("p", "alpha", "beta")))
    fit_rows <- list()
    for (dep in task$deps) {
      v <- results[[dep]]$value
      parameters[match(test_groups[[v$group]], s$test_rows), ] <- v$parameters
      fit_rows[[length(fit_rows) + 1L]] <- data.frame(model = task$model, family = task$family, time_scale = task$time_scale, group = v$group, seed = CFG$seed, train_days = v$train_days, train_wet_days = v$train_wet_days, test_days = v$test_days, seconds = results[[dep]]$cpu_seconds)
    }
    task$parameters <- parameters
    task$fit_log <- do.call(rbind, fit_rows)
  } else if (task$type == "station") {
    task$model_results <- lapply(task$deps, function(dep) results[[dep]]$value)
  }
  task
}

# ---- Scheduler ----------------------------------------------------------------------------------
n_tasks <- length(tasks)
type_of <- vapply(tasks, `[[`, "", "type")
stage_of <- STAGE[type_of]
station_of <- vapply(tasks, `[[`, 0L, "station_index")
cost_of <- vapply(tasks, `[[`, 0, "cost")
deps_left <- vapply(tasks, function(t) length(t$deps), 0L)
dependents <- split(rep(seq_len(n_tasks), lengths(lapply(tasks, `[[`, "deps"))), unlist(lapply(tasks, `[[`, "deps")))
state <- rep("waiting", n_tasks)
results <- vector("list", n_tasks)
task_log <- vector("list", n_tasks)

workers <- min(OPT$workers, n_tasks)
log_line("Stations (%d): %s", length(STATIONS), paste(names(STATIONS), collapse = ", "))
log_line("Predictors (%d): %s", length(CFG$predictors), paste(CFG$predictors, collapse = ", "))
log_line("Tasks: %s | workers: %d | output: %s", paste(names(table(type_of)), table(type_of), sep = "=", collapse = " "), workers, OPT$out)
log_line("Estimated work: %.0f CPU-min (single-core timings)", sum(cost_of) / 60)

# srand() for the C rand() used by the rpart fork, compiled once for this run
SRAND_LIB <- local({
  build_dir <- tempfile("c_srand_")
  dir.create(build_dir)
  writeLines(c("#include <stdlib.h>", "void c_srand(int *seed) { srand((unsigned int) *seed); }"), file.path(build_dir, "c_srand.c"))
  Sys.setenv(PATH = paste(normalizePath(file.path(R.home(), "..", "..", "bin")), Sys.getenv("PATH"), sep = .Platform$path.sep))
  status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "-o", shQuote(file.path(build_dir, paste0("c_srand", .Platform$dynlib.ext))), shQuote(file.path(build_dir, "c_srand.c"))), stdout = FALSE, stderr = FALSE)
  if (status != 0) stop("Could not compile c_srand.c (R CMD SHLIB)")
  file.path(build_dir, paste0("c_srand", .Platform$dynlib.ext))
})

cl <- makePSOCKcluster(workers)
on.exit(stopCluster(cl), add = TRUE)
clusterExport(cl, c("SRAND_LIB", "PROJECT", "CFG", "STATIONS", "month_groups", "get_py", "write_xlsx", "seed_of", "model_folder", "group_model_file", "build_groups", "sanitize_parameters", "fit_glm_group", "predict_glm_parameters", "fit_rf_trees", "predict_rf_parameters", "simulate_bg", "task_glm", "task_rf_chunk", "task_rf_predict", "task_model_eval", "task_station", "TASK_RUNNERS", "run_task"))
invisible(clusterEvalQ(cl, {
  suppressPackageStartupMessages({ library(reticulate); library(VGAM); library(RandomForestDist); library(jsonlite) })
  setwd(PROJECT)
  dyn.load(SRAND_LIB)
  py_env <- new.env()
  NULL
}))

started <- Sys.time()
idle <- seq_len(workers)
running <- 0L
done_cost <- 0
n_done <- 0L
n_failed <- 0L
failed_messages <- character()
mark_failed <- function(i, why) {
  state[i] <<- "failed"
  n_failed <<- n_failed + 1L
  for (d in dependents[[as.character(i)]]) if (state[d] == "waiting") mark_failed(d, paste("input failed:", tasks[[i]]$type, tasks[[i]]$station, tasks[[i]]$model, tasks[[i]]$group))
}
label_of <- function(t) trimws(paste(t$type, t$station, t$model, t$group, if (!is.null(t$trees)) sprintf("trees %d-%d", min(t$trees), max(t$trees)) else ""))

repeat {
  while (length(idle) && any(ready <- state == "waiting" & deps_left == 0L)) {
    candidates <- which(ready)
    i <- candidates[order(-stage_of[candidates], station_of[candidates], -cost_of[candidates], candidates)[1]]
    node <- idle[1]; idle <- idle[-1]
    parallel:::sendCall(cl[[node]], run_task, list(task_inputs(tasks[[i]], results)), tag = c(i, node))
    state[i] <- "running"; running <- running + 1L
    if (type_of[i] %in% c("model_eval", "station")) for (dep in tasks[[i]]$deps) results[[dep]]$value <- NULL
  }
  if (running == 0L) break
  received <- parallel:::recvOneResult(cl)
  i <- received$tag[1]; node <- received$tag[2]
  idle <- c(idle, node); running <- running - 1L
  r <- received$value
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  if (inherits(r, "try-error") || inherits(r$value, "task_error")) {
    message_text <- if (inherits(r, "try-error")) as.character(r) else r$value$message
    failed_messages <- c(failed_messages, sprintf("%s: %s", label_of(tasks[[i]]), message_text))
    log_line("FAILED %s: %s", label_of(tasks[[i]]), message_text)
    mark_failed(i, message_text)
    next
  }
  state[i] <- "done"; n_done <- n_done + 1L
  r$cpu_seconds <- r$seconds + sum(vapply(tasks[[i]]$deps[type_of[tasks[[i]]$deps] == "rf_chunk"], function(dep) task_log[[dep]]$seconds, 0))
  results[[i]] <- r
  task_log[[i]] <- data.frame(task = i, type = type_of[i], station = tasks[[i]]$station, model = tasks[[i]]$model, group = tasks[[i]]$group,
                              trees = if (is.null(tasks[[i]]$trees)) NA_character_ else sprintf("%d-%d", min(tasks[[i]]$trees), max(tasks[[i]]$trees)),
                              seconds = r$seconds, finished_after_s = elapsed, worker_pid = r$pid, warnings = paste(r$warnings, collapse = " | "))
  for (d in dependents[[as.character(i)]]) deps_left[d] <- deps_left[d] - 1L
  done_cost <- done_cost + cost_of[i]
  eta <- elapsed * (sum(cost_of[state != "failed"]) - done_cost) / max(done_cost, 1e-9)
  important <- type_of[i] %in% c("model_eval", "station") || n_done %% 25L == 0L
  log_line("[%4d/%d %5.1f%% | %s elapsed | ~%s left] %s (%.1f s)%s", n_done + n_failed, n_tasks, 100 * done_cost / sum(cost_of),
           format(.POSIXct(elapsed, tz = "UTC"), "%H:%M:%S"), format(.POSIXct(max(0, eta), tz = "UTC"), "%H:%M:%S"), label_of(tasks[[i]]), r$seconds,
           if (length(r$warnings)) sprintf(" [%d warning(s)]", length(r$warnings)) else "", console = important)
  if (type_of[i] == "station") log_line("Station done: %s -> %s", tasks[[i]]$station, STATIONS[[tasks[[i]]$station]]$out_dir)
}

# ---- Run summary --------------------------------------------------------------------------------
task_log_table <- do.call(rbind, task_log[!vapply(task_log, is.null, TRUE)])
write_xlsx(task_log_table, file.path(OPT$out, "task_log.xlsx"))
station_results <- lapply(which(type_of == "station" & state == "done"), function(i) results[[i]]$value)
if (length(station_results)) {
  for (field in c("scores", "annual", "point_overall", "point_seasonal", "point_monthly", "fit_log")) {
    frame <- do.call(rbind, lapply(station_results, `[[`, field)); rownames(frame) <- NULL
    write_xlsx(frame, file.path(OPT$out, paste0("all_stations_", field, ".xlsx")))
  }
}
write_json(c(OPT, list(predictors_used = CFG$predictors, stations = names(STATIONS), workers_used = workers)), file.path(OPT$out, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)
total <- as.numeric(difftime(Sys.time(), started, units = "secs"))
log_line("Finished in %s: %d tasks done, %d failed, %.0f CPU-min of task time (x%.1f speed-up)", format(.POSIXct(total, tz = "UTC"), "%H:%M:%S"), n_done, n_failed, sum(task_log_table$seconds) / 60, sum(task_log_table$seconds) / total)
if (n_failed) {
  log_line("Failures:\n%s", paste(failed_messages, collapse = "\n"))
  quit(status = 1)
}
log_line("Results: %s", OPT$out)
