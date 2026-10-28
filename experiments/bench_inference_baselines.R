# Inference time of GLM-BG and RF-BG at the Annual scale on one thread: Bernoulli-Gamma parameters of the 2922 test
# days and their 1000-member ensemble. GLM-BG is refitted; the forests of RF-BG are read from fitted_groups.rds.
#   Rscript bench_inference_baselines.R glm|rf [stations] [--parts]

suppressMessages(library(reticulate))
args <- grep("^--", commandArgs(trailingOnly = TRUE), value = TRUE, invert = TRUE)
family <- args[1]
stations <- if (length(args) >= 2) args[-1] else c("brest_guipavas", "lille_lesquin", "tours", "strasbourg_entzheim",
  "clermont_ferrand", "bordeaux_merignac", "montelimar", "tarbes_ossun", "perpignan")
suppressMessages(if (family == "glm") library(VGAM) else library(RandomForestDist))
CFG <- list(tau = 1.0, members = 1000L, glm_stepsize = 0.5, glm_maxit = 500L, rf_predict_chunk = 128L,
            predictors = c("d2m_mean", "t2m_mean", "msl_mean", "tcwv_mean", "u850_mean", "v850_mean", "z850_mean", "q850_mean"))
# folders of run_all.sh, else those of the paper runs
data_dir <- Sys.getenv("DSSM_DATA", Find(dir.exists, file.path(c("..", file.path("..", "..")), "data", "Data1Cell_06UTC")))
rf_run <- if (nzchar(Sys.getenv("DSSM_RUNS"))) file.path(Sys.getenv("DSSM_RUNS"), "glm_rf") else
  file.path(dirname(dirname(data_dir)), "results", "glm_rf_allstations_8vars", "GLM_RF_tuned_20261009_013535")
np <- import("numpy", convert = FALSE)

# ---- functions of run_glm_rf_parallel.R (station data passed as s) ----------------------------------------------
sanitize_parameters <- function(parameters) {
  parameters <- as.matrix(parameters[, c("p", "alpha", "beta"), drop = FALSE])
  storage.mode(parameters) <- "double"
  parameters[, "p"] <- pmin(1, pmax(0, parameters[, "p"]))
  parameters[, "alpha"] <- pmax(1e-8, parameters[, "alpha"])
  parameters[, "beta"] <- pmax(1e-8, parameters[, "beta"])
  parameters
}
fit_glm_group <- function(s, rows) {
  frame <- data.frame(wet = s$wet_all[rows], excess = s$excess_all[rows], s$X_standardized[rows, , drop = FALSE])
  wet_frame <- frame[frame$wet == 1L, , drop = FALSE]
  occurrence <- vglm(reformulate(CFG$predictors, response = "wet"), binomialff(link = "logitlink"), data = frame)
  intensity <- vglm(reformulate(CFG$predictors, response = "excess"), gamma2(lmu = "loglink", lshape = "loglink", zero = NULL, parallel = FALSE), data = wet_frame, control = vglm.control(stepsize = CFG$glm_stepsize, maxit = CFG$glm_maxit))
  list(occurrence = occurrence, intensity = intensity)
}
predict_glm_parameters <- function(s, model, rows) {
  newdata <- data.frame(s$X_standardized[rows, , drop = FALSE])
  p <- plogis(drop(predict(model$occurrence, newdata = newdata, type = "link")))
  eta <- as.matrix(predict(model$intensity, newdata = newdata, type = "link"))
  mu <- exp(eta[, 1])
  alpha <- exp(eta[, 2])
  sanitize_parameters(cbind(p = p, alpha = alpha, beta = alpha / mu))
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
simulate_bg <- function(parameters, members = CFG$members) {
  parameters <- sanitize_parameters(parameters)
  n <- nrow(parameters)
  wet <- matrix(runif(n * members), nrow = n) < parameters[, "p"]
  excess <- matrix(rgamma(n * members, shape = rep(parameters[, "alpha"], members), rate = rep(parameters[, "beta"], members)), nrow = n)
  ifelse(wet, CFG$tau + excess, 0.0)
}

load_station <- function(station) {
  era5 <- np$load(file.path(data_dir, sprintf("era5_daily_cell_%s.npz", station)), allow_pickle = FALSE)
  X_all <- py_to_r(era5[["data"]])
  dates <- as.Date(as.character(py_to_r(np$datetime_as_string(era5[["dates"]], unit = "D"))))
  X_raw <- X_all[, match(CFG$predictors, as.character(py_to_r(era5[["variables"]]))), drop = FALSE]
  storage.mode(X_raw) <- "double"
  y_raw <- as.numeric(py_to_r(np$load(file.path(data_dir, sprintf("ecad_rr_%s_1980_2025.npy", station)), allow_pickle = FALSE)))
  y_all <- ifelse(y_raw > CFG$tau, y_raw, 0.0)
  train_rows <- which(dates >= as.Date("1980-01-01") & dates <= as.Date("2017-12-31"))
  X_standardized <- scale(X_raw, center = colMeans(X_raw[train_rows, ]), scale = apply(X_raw[train_rows, ], 2, sd))
  colnames(X_standardized) <- CFG$predictors
  list(X_standardized = X_standardized, wet_all = as.integer(y_all > CFG$tau), excess_all = ifelse(y_all > CFG$tau, y_all - CFG$tau, 0.0),
       train_rows = train_rows, test_rows = which(dates >= as.Date("2018-01-01") & dates <= as.Date("2025-12-31")))
}

for (station in stations) {
  s <- load_station(station)
  set.seed(42)
  if (family == "glm") {
    fit_seconds <- system.time(model <- fit_glm_group(s, s$train_rows))[["elapsed"]]
    seconds <- system.time(ensemble <- simulate_bg(predict_glm_parameters(s, model, s$test_rows)))[["elapsed"]]
    cat(sprintf("GLM-BG %-20s fit %5.2f s | inference %5.2f s\n", station, fit_seconds, seconds))
  } else {
    model <- readRDS(file.path(rf_run, station, "RF_Annual", "seed42", "fitted_groups.rds"))[["ALL"]]
    seconds <- system.time({
      parameters <- predict_rf_parameters(s, model, s$test_rows)
      ensemble <- simulate_bg(parameters)
    })[["elapsed"]]
    saved <- np$load(file.path(rf_run, station, "RF_Annual", "seed42", "test_ensemble.npz"))
    difference <- max(abs(parameters[, "p"] - py_to_r(saved[["wet_probability"]])), abs(parameters[, "alpha"] - py_to_r(saved[["alpha"]])))
    cat(sprintf("RF-BG  %-20s %d trees | inference %5.1f s | max |difference| of p, alpha from the paper run %.1e\n",
                station, length(model$forest), seconds, difference))
    if ("--parts" %in% commandArgs()) {   # the three steps of predict_rf_parameters and simulate_bg, timed apart
      estimate_bg_leaf <- getFromNamespace("estimateBernoulliGammabc3", "RandomForestDist")
      predictors <- unname(s$X_standardized[s$test_rows, , drop = FALSE])
      leaf_seconds <- system.time(leaves <- unlist(lapply(split(seq_len(nrow(predictors)), ceiling(seq_len(nrow(predictors)) / CFG$rf_predict_chunk)),
        function(index) randomForestPredict(model$forest, newdata = predictors[index, , drop = FALSE], method = "leaves", parallel.plan = NA)), recursive = FALSE))[["elapsed"]]
      estimate_seconds <- system.time(vapply(leaves, function(values) estimate_bg_leaf(as.numeric(values), non_informative_beta = 1), numeric(3)))[["elapsed"]]
      simulate_seconds <- system.time(simulate_bg(parameters))[["elapsed"]]
      cat(sprintf("       leaves of the 500 trees %5.1f s | Bernoulli-Gamma estimate per day %5.1f s | 1000 members %4.1f s | median pooled sample per day %d values\n",
                  leaf_seconds, estimate_seconds, simulate_seconds, as.integer(median(lengths(leaves)))))
    }
    rm(model); invisible(gc())
  }
}
