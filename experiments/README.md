# Experiments

Scripts of all experiments of the paper at the nine stations: training of the four models and of the ablation variants,
then the tables, figures and numbers quoted in the manuscript. `run_all.sh` runs them with one command.

## Usage

```bash
bash run_all.sh                                                  # every step
setsid nohup bash run_all.sh > run_all.log 2>&1 < /dev/null &    # the same, detached from the terminal
bash run_all.sh dssm analysis                                    # selected steps
OUT=../results/<run> bash run_all.sh                              # resume a run; finished steps are skipped
QUICK=1 bash run_all.sh                                          # smoke test: one or two epochs, four trees
```

## Files

**Pipeline and shared modules**

| File | Function |
|---|---|
| `run_all.sh` | runs the steps below in order; each training step uses all cores |
| `Data1Cell_06UTC.md5` | MD5 checksums of the 18 input files (step `data`) |
| `paths.py` | input, run and output folders of the analysis scripts |
| `precipitation_metrics.py` | scores: fair and analytic CRPS, threshold-weighted CRPS, Brier score, AUC, Bernoulli–Gamma log-likelihood, point metrics, temporal indices |
| `make_tables_8vars.py` | reads the scores stored in the run folders |

**Training.** Each model is trained at three scales (on all days, by season, by month) and simulates 1000-member
ensembles over the test period 2018–2025. The neural networks are trained on 1980–2012 and selected on 2013–2017.

| File | Models | Step |
|---|---|---|
| `tune_rf_parallel.R` | hyperparameters of RF–BG: forests fitted on 1980–2012, fair CRPS on 2013–2017 | `rf_tuning` |
| `run_glm_rf_parallel.R` | GLM–BG and RF–BG, fitted on 1980–2017 | `glm_rf` |
| `run_mlp_gru_parallel.py` | MLP–BG and GRU–BG | `mlp_gru` |
| | BiGRU–BG (ablation, `--arch bigru`) | `bigru` |
| `run_dssm_parallel.py` | DSSM–BG and its causal-context variant, with the filtering inference network | `dssm` |
| `run_dssm_smoother_parallel.py` | the same two models with the smoothing inference network (ablation) | `smoother` |
| | the four versions trained on windows of 128, 256 and 1024 days and without windows (`--window`) | `window` |

**Analysis** (step `analysis`)

| Script | Manuscript |
|---|---|
| `stations_summary.py` | Table 1 (stations), Table 4 (CRPS at the three training scales), CRPS skill scores |
| `stations_appendix_tables.py` | Table 5 (NLL, Brier score, AUC, COR, RMSE), Table 6 (seasonal CRPS) |
| `stations_indices.py` | Tables 10–12 (predictive distributions, VALUE indices, KS statistic); heavy precipitation (Section 4.3) |
| `stations_value_figures.py` | Figures 4–8: VALUE indices at the nine stations |
| `stations_annual_cycle.py` | Figure 9: annual cycle of mean precipitation |
| `ablation_tables.py` | Tables 7 and 8: ablation of DSSM–BG |
| `make_window_tables.py` | scores of the window runs, read by `sensitivity_tables.py` |
| `sensitivity_tables.py` | Table 9: length of the training windows |
| `downscaling_checks.py` | Appendix B: initial latent state z<sub>T0</sub> = 0, context of the test period |
| `tours_figures.py` | Figures 10–13: results at Tours (Appendix F) |
| `plot_context_maps.py` | Figure 3: maps of the stations (needs cartopy, netCDF4, shapely and xarray) |

**Computing time** (step `timing`)

| Script | Function |
|---|---|
| `computing_times.py` | training and inference CPU time of the four models, from the task logs |
| `bench_inference_neural.py` | inference time of DSSM–BG and MLP–BG on one CPU thread |
| `bench_inference_baselines.R` | inference time of GLM–BG and RF–BG on one CPU thread |