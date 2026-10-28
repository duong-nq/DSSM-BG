#!/usr/bin/env Rscript
# Hyperparameters of RF-BG: forests trained on 1980-2012 and scored by the fair CRPS of 1000 members on 2013-2017.
# Stage 1 searches mtry x minbucket, stage 2 maxdepth; minsplit = 3 x minbucket. The configuration with the lowest CRPS
# averaged over the stations replaces the reference one if it lowers that CRPS by more than --threshold.

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

OPT <- list(data_dir = "Data1CellNew", stations = "all", workers = parallel::detectCores(), seed = 42L, ntree = 200L, members = 1000L,
            mtry = "2,3,4,6,8", minbucket = "5,10,20,50", maxdepth = 10L, maxdepth2 = "8,15,30", reference = "2,10,10", threshold = 0.005,
            train = "1980-01-01,2012-12-31", validation = "2013-01-01,2017-12-31", resume = "no", out = NA_character_,
            python = normalizePath(file.path(R.home(), "..", "..", "bin", "python"), mustWork = FALSE))
cli <- commandArgs(trailingOnly = TRUE)
if (length(cli) %% 2L != 0L) stop("Options are given as pairs: --name value")
for (i in seq_len(length(cli) %/% 2L) * 2L - 1L) {
  key <- gsub("-", "_", sub("^--", "", cli[i]))
  if (!key %in% names(OPT)) stop("Unknown option: ", cli[i])
  OPT[[key]] <- if (is.numeric(OPT[[key]])) as(cli[i + 1L], class(OPT[[key]])) else cli[i + 1L]
}
as_integers <- function(text) as.integer(trimws(strsplit(text, ",")[[1]]))
as_dates <- function(text) as.Date(trimws(strsplit(text, ",")[[1]]))
MTRY <- as_integers(OPT$mtry)
MINBUCKET <- as_integers(OPT$minbucket)
MAXDEPTH2 <- if (OPT$maxdepth2 == "none") integer() else as_integers(OPT$maxdepth2)
REFERENCE <- setNames(as_integers(OPT$reference), c("mtry", "minbucket", "maxdepth"))
TRAIN <- as_dates(OPT$train)
VALIDATION <- as_dates(OPT$validation)
stopifnot(length(REFERENCE) == 3L, length(TRAIN) == 2L, length(VALIDATION) == 2L, TRAIN[2] < VALIDATION[1],
          VALIDATION[2] < as.Date("2018-01-01"), OPT$resume %in% c("yes", "no"), OPT$workers >= 1L)

if (is.na(OPT$out)) OPT$out <- file.path("Results", "rf_tuning_8vars", format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(OPT$out, recursive = TRUE, showWarnings = FALSE)
RUN_LOG <- file.path(OPT$out, "run.log")
SCORES_CSV <- file.path(OPT$out, "tuning_scores.csv")
log_line <- function(...) {
  line <- paste0(format(Sys.time(), "%H:%M:%S"), "  ", sprintf(...))
  cat(line, "\n", sep = "", file = RUN_LOG, append = TRUE)
  cat(line, "\n", sep = "")
}

# ---- Configuration -------------------------------------------------------------------------------
CFG <- list(
  tau = 1.0,
  seed = OPT$seed,
  ensemble_members = OPT$members,
  ntree = OPT$ntree,
  predictors = c("d2m_mean", "t2m_mean", "msl_mean", "tcwv_mean", "u850_mean", "v850_mean", "z850_mean", "q850_mean"),
  rf_predict_chunk = 128L,
  python_bin = OPT$python
)
SECONDS_PER_TREE <- c(`2` = 6.4, `3` = 5.5, `4` = 6.5, `6` = 9.7, `8` = 12.2)   # measured on Tours (12054 rows); only orders the tasks

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
config_label <- function(mtry, minbucket, maxdepth) sprintf("mtry%d_minbucket%d_maxdepth%d", mtry, minbucket, maxdepth)

sanitize_parameters <- function(parameters) {
  parameters <- as.matrix(parameters[, c("p", "alpha", "beta"), drop = FALSE])
  storage.mode(parameters) <- "double"
  parameters[, "p"] <- pmin(1, pmax(0, parameters[, "p"]))
  parameters[, "alpha"] <- pmax(1e-8, parameters[, "alpha"])
  parameters[, "beta"] <- pmax(1e-8, parameters[, "beta"])
  parameters
}

simulate_bg <- function(parameters, members = CFG$ensemble_members) {
  parameters <- sanitize_parameters(parameters)
  n <- nrow(parameters)
  wet <- matrix(runif(n * members), nrow = n) < parameters[, "p"]
  excess <- matrix(rgamma(n * members, shape = rep(parameters[, "alpha"], members), rate = rep(parameters[, "beta"], members)), nrow = n)
  ifelse(wet, CFG$tau + excess, 0.0)
}

# One task: fit one forest (one tree per call, own seed), predict the validation days, score them.
task_forest <- function(task) {
  s <- STATIONS[[task$station]]
  label <- config_label(task$mtry, task$minbucket, task$maxdepth)
  x <- unname(s$X[s$train_rows, , drop = FALSE])
  y <- s$excess_all[s$train_rows]
  multiresponse <- NA
  trees <- lapply(seq_len(CFG$ntree), function(tree_index) {
    tree_seed <- seed_of("tune", task$station, label, "tree", tree_index)
    set.seed(tree_seed)
    .C("c_srand", tree_seed)  # the rpart fork draws the mtry variables with C rand(), which set.seed does not reach
    one <- randomForestTrain(x = x, y = y, ntree = 1L, mtry = task$mtry, minbucket = task$minbucket, minsplit = 3L * task$minbucket,
                             maxdepth = task$maxdepth, method = "bernoulliGammaLLMME", resample = TRUE, replace = TRUE, parallel.plan = NA,
                             keep.y = TRUE, progress.bar = FALSE)
    multiresponse <<- attr(one, "multiresponse")
    one[[1]]
  })
  forest <- structure(trees, multiresponse = multiresponse, class = "RandomForestDist")
  rm(trees)

  estimate_bg_leaf <- getFromNamespace("estimateBernoulliGammabc3", "RandomForestDist")
  predictors <- unname(s$X[s$validation_rows, , drop = FALSE])
  chunks <- split(seq_len(nrow(predictors)), ceiling(seq_len(nrow(predictors)) / CFG$rf_predict_chunk))
  parameters <- do.call(rbind, lapply(chunks, function(index) {
    leaves <- randomForestPredict(forest, newdata = predictors[index, , drop = FALSE], method = "leaves", parallel.plan = NA)
    t(vapply(leaves, function(values) estimate_bg_leaf(as.numeric(values), non_informative_beta = 1), numeric(3)))
  }))
  colnames(parameters) <- c("p", "alpha", "beta")
  parameters <- sanitize_parameters(parameters)
  rm(forest); invisible(gc())

  set.seed(seed_of("tune", task$station, label, "ensemble"))
  ensemble <- simulate_bg(parameters)
  pm <- get_py()$pm
  y_validation <- s$y_all[s$validation_rows]
  p <- parameters[, "p"]; alpha <- parameters[, "alpha"]; beta <- parameters[, "beta"]
  data.frame(
    station = task$station, stage = task$stage, config = label, mtry = task$mtry, minbucket = task$minbucket, minsplit = 3L * task$minbucket,
    maxdepth = task$maxdepth, ntree = CFG$ntree, members = CFG$ensemble_members,
    CRPS_ensemble_fair = mean(as.numeric(pm$crps_ensemble_fair(y_validation, ensemble, CFG$tau))),
    CRPS_analytic = mean(as.numeric(pm$crps_bernoulli_gamma_analytic(y_validation, p, alpha, beta, CFG$tau))),
    Brier = as.numeric(pm$brier_score(y_validation, p, CFG$tau)),
    AUC = as.numeric(pm$auc_score(y_validation, p, CFG$tau)),
    NLL = -mean(as.numeric(pm$bernoulli_gamma_log_likelihood(y_validation, p, alpha, beta, CFG$tau))),
    train_days = length(s$train_rows), validation_days = length(s$validation_rows),
    train_first = format(s$dates[s$train_rows[1]]), train_last = format(s$dates[tail(s$train_rows, 1)]),
    validation_first = format(s$dates[s$validation_rows[1]]), validation_last = format(s$dates[tail(s$validation_rows, 1)]),
    stringsAsFactors = FALSE)
}

run_task <- function(task) {
  started <- proc.time()[["elapsed"]]
  value <- tryCatch(suppressWarnings(task_forest(task)), error = function(e) structure(list(message = conditionMessage(e)), class = "task_error"))
  list(value = value, seconds = proc.time()[["elapsed"]] - started)
}

# ---- Stations: data, train/validation split, standardisation on the training days ---------------
np <- get_py()$np
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
  era5 <- np$load(file.path(OPT$data_dir, paste0("era5_daily_cell_", station, ".npz")), allow_pickle = FALSE)
  X_all <- py_to_r(era5[["data"]])
  dates <- as.Date(as.character(py_to_r(np$datetime_as_string(era5[["dates"]], unit = "D"))))
  variable_names <- as.character(py_to_r(era5[["variables"]]))
  invisible(era5$close())
  if (anyNA(match(CFG$predictors, variable_names))) stop(station, ": missing predictors")
  X_raw <- X_all[, match(CFG$predictors, variable_names), drop = FALSE]
  storage.mode(X_raw) <- "double"
  y_raw <- as.numeric(py_to_r(np$load(file.path(OPT$data_dir, paste0("ecad_rr_", station, "_1980_2025.npy")), allow_pickle = FALSE)))
  stopifnot(length(y_raw) == nrow(X_raw), !anyNA(y_raw), !anyNA(X_raw))
  y_all <- ifelse(y_raw > CFG$tau, y_raw, 0.0)
  excess_all <- ifelse(y_all > CFG$tau, y_all - CFG$tau, 0.0)
  train_rows <- which(dates >= TRAIN[1] & dates <= TRAIN[2])
  validation_rows <- which(dates >= VALIDATION[1] & dates <= VALIDATION[2])
  x_mean <- colMeans(X_raw[train_rows, , drop = FALSE])
  x_sd <- apply(X_raw[train_rows, , drop = FALSE], 2, sd)
  x_sd[!is.finite(x_sd) | x_sd == 0] <- 1
  X <- sweep(sweep(X_raw, 2, x_mean, "-"), 2, x_sd, "/")
  rows <- c(train_rows, validation_rows)
  # Only the training and validation days are kept on the workers.
  STATIONS[[station]] <- list(name = station, dates = dates, X = X, y_all = y_all, excess_all = excess_all,
                              train_rows = train_rows, validation_rows = validation_rows)
  STATIONS[[station]]$X[-rows, ] <- NA
  STATIONS[[station]]$y_all[-rows] <- NA
}

# ---- Worker pool ---------------------------------------------------------------------------------
SRAND_LIB <- local({
  build_dir <- tempfile("c_srand_")
  dir.create(build_dir)
  writeLines(c("#include <stdlib.h>", "void c_srand(int *seed) { srand((unsigned int) *seed); }"), file.path(build_dir, "c_srand.c"))
  Sys.setenv(PATH = paste(normalizePath(file.path(R.home(), "..", "..", "bin")), Sys.getenv("PATH"), sep = .Platform$path.sep))
  status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "-o", shQuote(file.path(build_dir, paste0("c_srand", .Platform$dynlib.ext))), shQuote(file.path(build_dir, "c_srand.c"))), stdout = FALSE, stderr = FALSE)
  if (status != 0) stop("Could not compile c_srand.c (R CMD SHLIB)")
  file.path(build_dir, paste0("c_srand", .Platform$dynlib.ext))
})

if (OPT$resume == "no" && file.exists(SCORES_CSV)) stop(SCORES_CSV, " exists: use --resume yes or another --out")

CLUSTER <- NULL
run_stage <- function(stage, configs) {
  # Tasks already scored (by an earlier stage or an interrupted run) are not repeated.
  done_scores <- if (file.exists(SCORES_CSV)) read.csv(SCORES_CSV, stringsAsFactors = FALSE) else NULL
  tasks <- list()
  for (station in names(STATIONS)) for (k in seq_len(nrow(configs))) {
    task <- list(station = station, stage = stage, mtry = configs$mtry[k], minbucket = configs$minbucket[k], maxdepth = configs$maxdepth[k])
    label <- config_label(task$mtry, task$minbucket, task$maxdepth)
    if (!is.null(done_scores) && any(done_scores$station == station & done_scores$config == label & done_scores$ntree == CFG$ntree & done_scores$members == CFG$ensemble_members)) next
    task$cost <- SECONDS_PER_TREE[[as.character(min(8L, task$mtry))]] * CFG$ntree
    tasks[[length(tasks) + 1L]] <- task
  }
  log_line("Stage %d: %d configurations x %d stations, %d tasks to run (%d already in tuning_scores.csv), ~%.0f CPU-h",
           stage, nrow(configs), length(STATIONS), length(tasks), nrow(configs) * length(STATIONS) - length(tasks),
           sum(vapply(tasks, `[[`, 0, "cost")) / 3600)
  if (!length(tasks)) return(invisible())
  if (is.null(CLUSTER)) {
    CLUSTER <<- makePSOCKcluster(min(OPT$workers, length(tasks)))
    clusterExport(CLUSTER, c("SRAND_LIB", "PROJECT", "CFG", "STATIONS", "get_py", "seed_of", "config_label", "sanitize_parameters", "simulate_bg", "task_forest", "run_task"))
    invisible(clusterEvalQ(CLUSTER, {
      suppressPackageStartupMessages({ library(reticulate); library(RandomForestDist) })
      setwd(PROJECT)
      dyn.load(SRAND_LIB)
      py_env <- new.env()
      NULL
    }))
  }
  tasks <- tasks[order(-vapply(tasks, `[[`, 0, "cost"))]            # longest first
  workers <- length(CLUSTER)
  idle <- seq_len(workers); running <- 0L; next_task <- 1L; n_done <- 0L; failed <- character()
  started <- Sys.time()
  repeat {
    while (length(idle) && next_task <= length(tasks)) {
      node <- idle[1]; idle <- idle[-1]
      parallel:::sendCall(CLUSTER[[node]], run_task, list(tasks[[next_task]]), tag = c(next_task, node))
      next_task <- next_task + 1L; running <- running + 1L
    }
    if (running == 0L) break
    received <- parallel:::recvOneResult(CLUSTER)
    i <- received$tag[1]; idle <- c(idle, received$tag[2]); running <- running - 1L
    r <- received$value; t <- tasks[[i]]
    label <- sprintf("%s %s", t$station, config_label(t$mtry, t$minbucket, t$maxdepth))
    if (inherits(r, "try-error") || inherits(r$value, "task_error")) {
      message_text <- if (inherits(r, "try-error")) as.character(r) else r$value$message
      failed <- c(failed, sprintf("%s: %s", label, message_text))
      log_line("FAILED %s: %s", label, message_text)
      next
    }
    row <- r$value; row$seconds <- r$seconds
    write.table(row, SCORES_CSV, sep = ",", row.names = FALSE, col.names = !file.exists(SCORES_CSV), append = file.exists(SCORES_CSV))
    n_done <- n_done + 1L
    elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
    log_line("[stage %d %3d/%d | %5.1f min] %s CRPS %.4f (%.1f min)", stage, n_done + length(failed), length(tasks), elapsed, label, row$CRPS_ensemble_fair, r$seconds / 60)
  }
  if (length(failed)) {
    stopCluster(CLUSTER)
    stop(sprintf("%d task(s) failed in stage %d; rerun with --resume yes --out %s\n%s", length(failed), stage, OPT$out, paste(failed, collapse = "\n")))
  }
}

# Mean over stations of one stage's configurations, and the choice with the threshold rule.
summarise_stage <- function(stage, configs, reference_label) {
  scores <- read.csv(SCORES_CSV, stringsAsFactors = FALSE)
  labels <- config_label(configs$mtry, configs$minbucket, configs$maxdepth)
  scores <- scores[scores$config %in% labels & scores$station %in% names(STATIONS) & scores$ntree == CFG$ntree & scores$members == CFG$ensemble_members, ]
  scores <- scores[!duplicated(scores[, c("station", "config")]), ]
  complete <- table(scores$config) == length(STATIONS)
  stopifnot(all(labels %in% names(complete)), all(complete[labels]))
  metrics <- c("CRPS_ensemble_fair", "CRPS_analytic", "Brier", "AUC", "NLL")
  summary <- aggregate(scores[, metrics], by = list(config = scores$config), FUN = mean)
  summary <- merge(unique(scores[, c("config", "mtry", "minbucket", "minsplit", "maxdepth")]), summary, by = "config")
  wins <- vapply(summary$config, function(cfg) sum(vapply(split(scores, scores$station), function(f) f$config[which.min(f$CRPS_ensemble_fair)] == cfg, TRUE)), 0L)
  summary$stations_best <- wins
  reference_crps <- summary$CRPS_ensemble_fair[summary$config == reference_label]
  summary$gain_vs_reference <- (reference_crps - summary$CRPS_ensemble_fair) / reference_crps
  summary <- summary[order(summary$CRPS_ensemble_fair), ]
  summary$rank <- seq_len(nrow(summary))
  summary$stage <- stage
  best <- summary[1, ]
  chosen <- if (best$gain_vs_reference > OPT$threshold) best else summary[summary$config == reference_label, ]
  log_line("Stage %d: best %s (mean CRPS %.4f, %+.2f%% vs reference %s); chosen %s",
           stage, best$config, best$CRPS_ensemble_fair, 100 * best$gain_vs_reference, reference_label, chosen$config)
  list(summary = summary, chosen = chosen, best = best, reference_label = reference_label)
}

# ---- Run -----------------------------------------------------------------------------------------
log_line("Stations (%d): %s", length(STATIONS), paste(names(STATIONS), collapse = ", "))
log_line("Train %s..%s, validation %s..%s; %d trees, %d members; output %s", TRAIN[1], TRAIN[2], VALIDATION[1], VALIDATION[2], CFG$ntree, CFG$ensemble_members, OPT$out)
write_json(c(OPT, list(predictors = CFG$predictors, stations_used = names(STATIONS))), file.path(OPT$out, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)
run_started <- Sys.time()

stage1 <- unique(rbind(expand.grid(mtry = MTRY, minbucket = MINBUCKET, maxdepth = OPT$maxdepth),
                       data.frame(mtry = REFERENCE[["mtry"]], minbucket = REFERENCE[["minbucket"]], maxdepth = REFERENCE[["maxdepth"]])))
run_stage(1L, stage1)
result1 <- summarise_stage(1L, stage1, config_label(REFERENCE[["mtry"]], REFERENCE[["minbucket"]], REFERENCE[["maxdepth"]]))
summaries <- list(result1$summary)
final <- result1

if (length(MAXDEPTH2)) {
  chosen1 <- result1$chosen
  stage2 <- unique(data.frame(mtry = chosen1$mtry, minbucket = chosen1$minbucket, maxdepth = c(chosen1$maxdepth, MAXDEPTH2)))
  run_stage(2L, stage2)
  final <- summarise_stage(2L, stage2, chosen1$config)
  summaries <- c(summaries, list(final$summary))
}
if (!is.null(CLUSTER)) stopCluster(CLUSTER)

scores <- read.csv(SCORES_CSV, stringsAsFactors = FALSE)
write_xlsx(scores, file.path(OPT$out, "tuning_scores.xlsx"))
write_xlsx(do.call(rbind, summaries), file.path(OPT$out, "tuning_summary.xlsx"))
chosen <- final$chosen
refit <- sprintf("Rscript run_glm_rf_parallel.R --data-dir %s --rf-mtry %d --rf-minbucket %d --rf-minsplit %d --rf-maxdepth %d --out Results/glm_rf_allstations_8vars/GLM_RF_tuned_<timestamp>",
                 OPT$data_dir, chosen$mtry, chosen$minbucket, chosen$minsplit, chosen$maxdepth)
write_json(list(selected = list(mtry = chosen$mtry, minbucket = chosen$minbucket, minsplit = chosen$minsplit, maxdepth = chosen$maxdepth),
                selected_config = chosen$config, mean_validation_CRPS = chosen$CRPS_ensemble_fair,
                stage1_best = result1$best$config, stage1_reference = result1$reference_label, stage1_chosen = result1$chosen$config,
                final_best = final$best$config, threshold = OPT$threshold, stations = names(STATIONS),
                train = format(TRAIN), validation = format(VALIDATION), ntree = CFG$ntree, members = CFG$ensemble_members, seed = CFG$seed,
                refit_command = refit),
           file.path(OPT$out, "selected_config.json"), pretty = TRUE, auto_unbox = TRUE)
log_line("Selected: %s (mean validation CRPS %.4f). Done in %.1f min.", chosen$config, chosen$CRPS_ensemble_fair, as.numeric(difftime(Sys.time(), run_started, units = "mins")))
log_line("Refit: %s", refit)
