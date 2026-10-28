# A deep state-space Bernoulli–Gamma model for probabilistic daily precipitation downscaling

Code and data of the paper. DSSM–BG downscales daily precipitation at a station from large-scale predictors.

## Structure

```
Code_DSSM/
├── data/          inputs of the nine stations, 1980–2025
├── notebooks/     each model implemented step by step
└── experiments/   every experiment of the paper
```

- `data/Data1Cell_06UTC/`: for each station, 8 daily [ERA5](https://doi.org/10.1002/qj.3803) predictors of its grid
  cell, aggregated over the ECA&D day (06–06 UTC), and the daily precipitation of
  [ECA&D](https://doi.org/10.1002/joc.773).
- `notebooks/`: `DSSM_BG.ipynb` (DSSM–BG), `MLP_GRU.ipynb` (MLP–BG and GRU–BG) and `GLM_RF.ipynb` (GLM–BG and RF–BG,
  R kernel).
- `experiments/`: training of all models at the nine stations, then the tables and figures of the paper:
  `cd experiments && bash run_all.sh` (see [experiments/README.md](experiments/README.md)).

## Contributions

- **DSSM–BG**, a deep state-space model for daily precipitation with a Bernoulli–Gamma distribution. Its parameters
  combine a deterministic context, computed by a recurrent network from the predictors of the current, preceding and
  following days, and a stochastic latent state estimated by a filtering inference network. The model is trained with
  an importance-weighted objective and doubly reparameterized gradients.
- **A comparison with three baselines that use the same distribution**: a generalized linear model (GLM–BG), an
  a-posteriori random forest (RF–BG) and a multilayer perceptron (MLP–BG), each trained on all days, by season and by
  month, at nine stations with different climatologies of France. The models are evaluated with probabilistic scores,
  the VALUE indices and diagnostics of heavy precipitation.
- **Results**: a single DSSM–BG trained on all days has the lowest CRPS at all nine stations, also against the seasonal
  and monthly baselines, and the lowest negative log-likelihood and Brier score at every station. Averaged over the
  four seasons, its simulated series have the smallest biases of most VALUE indices.

## Citation

```bibtex
@article{nguyen2026dssmbg,
  title   = {A deep state-space {B}ernoulli--{G}amma model for probabilistic daily precipitation downscaling},
  author  = {Nguyen, Quoc-Duong and Zhu, Xujia and El Korso, Mohammed Nabil and Vaittinada Ayar, Pradeebane and
             Thao, Soulivanh},
  journal = {Environmental Data Science},
  year    = {2026},
  note    = {Submitted}
}
```
