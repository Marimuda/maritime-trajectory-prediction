# REPORT
---

# 1) Gap → Module Mapping (what’s missing)

| Gap               | Missing Capability                      | Add This Module (new)   | Purpose                                                                  |
| ----------------- | --------------------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| Statistical rigor | CI/Significance tests; model comparison | `evalx/stats/`          | Bootstrap CIs, paired tests, effect sizes, multiple-comparison control   |
| Baseline coverage | Physics + classical ML                  | `baselines/`            | Kalman/IMM, CV models (SVR/RF), ARIMA/SSM, particle filter               |
| Error analysis    | Failure mining, per-condition slicing   | `evalx/error_analysis/` | Horizon-wise error curves; by vessel type, weather, density; case mining |
| Domain validation | CPA/TCPA, COLREGS, operational metrics  | `maritime/validators/`  | CPA/TCPA computation; rule checks; warning time & false alert rate       |
| Multi-region      | Ingestion, splits, configs              | `datax/regions/`        | Standardized loaders; Group/Time splits; region registry                 |
| Uncertainty       | Conformal + Bayesian                    | `evalx/uncertainty/`    | Conformal prediction (APS/ICP), MC-dropout/ensembles                     |
| Interpretability  | Feature/attention attribution           | `evalx/explain/`        | SHAP/LIME (tabular), attention rollout (seq/graph)                       |
| Reporting         | Auto reports for papers                 | `reporting/`            | Jinja2 + Matplotlib/Altair; PDF/HTML; tables + figs reproducible         |
| Reproducibility   | Seeds, versioning, configs              | `infra/repro/`          | Hydra configs, seed control, data version stamps, run manifests          |

---

# 2) Repository Structure (additions only)

```
repo/
  baselines/
    kalman.py              # CV/CT, tuned IMM
    arima.py               # seasonal ARIMA, AIC model select
    svr_rf.py              # SVR, RF with time-aware CV
    particle_filter.py
  datax/
    regions/
      faroe.py
      baltic.py
      med.py
    splits.py              # TimeSeriesSplit, GroupKFold by MMSI
    schema.py              # Typed feature schemas (13D, 16D, graph)
  maritime/
    geometry.py            # Haversine, rhumb, great-circle ops
    cpa_tcpa.py            # CPA/TCPA core
    colregs.py             # Rule 13-17 checks (param.)
    ops_metrics.py         # Warning time, FAR, coverage
  evalx/
    metrics.py             # ADE/FDE (geodesic), circ-RMSE
    stats/
      bootstrap.py         # BCa CIs
      tests.py             # paired t-test, Wilcoxon, Cliff’s d
      mcnemar.py           # for classifiers
    error_analysis/
      slicers.py           # weather/vessel/density stratifiers
      mining.py            # worst-k cases, cluster failures
    uncertainty/
      conformal.py         # split/online CP for seq. regression
      bayes.py             # MC-dropout, deep ensembles
    explain/
      shap_tabular.py
      attn_rollout.py      # transformer, graph attention maps
  reporting/
    templates/             # Jinja2: methods.tex, results.tex, figs.html
    build.py               # compiles HTML/PDF, export figs/tables
  infra/
    hydra/                 # region/model/task configs
    repro/
      seeding.py
      versions.py
      manifest.py          # run hash: code+data+config
```

---

# 3) Core APIs (precise signatures)

### Data & Splits

```python
# datax/schema.py
from typing import TypedDict, Literal
class Traj13D(TypedDict):  # lat, lon in degrees; cog in degrees
    lat: float; lon: float; sog: float; cog: float; heading: float; turn: float
    temporal_hour: int; temporal_dow: int; temporal_month: int
    movement_speed_change: float; movement_course_change: float
    movement_distance: float; spatial_distance_from_center: float

FeatureSet = Literal["traj13", "anomaly16", "graph"]
```

```python
# datax/splits.py
def time_series_split(df, n_splits: int, min_gap: int) -> list[tuple]:
    """Return chronological folds with leakage-safe gaps."""
def group_kfold_by_mmsi(df, n_splits: int) -> list[tuple]:
    """Disjoint MMSI across folds (generalization to unseen vessels)."""
def region_kfold(datasets: dict[str, "Dataset"]) -> list[tuple[str, ...]]:
    """Leave-one-region-out protocol."""
```

### Metrics (geodesic + circular)

```python
# evalx/metrics.py
def ade_km(y_pred_latlon, y_true_latlon) -> float: ...
def fde_km(y_pred_latlon, y_true_latlon) -> float: ...
def rmse_course_circ(y_pred_deg, y_true_deg) -> float: ...
```

### Maritime validators

```python
# maritime/cpa_tcpa.py
def cpa_tcpa(p1, v1, p2, v2) -> tuple[float, float]:
    """Return (CPA_meters, TCPA_minutes)."""
# maritime/colregs.py
def colregs_violations(track_pair, params) -> dict:
    """Flags potential Rule 13–17 issues (parametric thresholds)."""
```

### Statistical rigor

```python
# evalx/stats/bootstrap.py
def bca_ci(values: list[float], alpha=0.05, B=2000) -> tuple[float,float]: ...
# evalx/stats/tests.py
def paired_t(a, b) -> dict
def wilcoxon(a, b) -> dict
def cliffs_delta(a, b) -> dict
```

### Uncertainty (conformal)

```python
# evalx/uncertainty/conformal.py
def split_conformal_regression(residuals_cal, alpha=0.1) -> float:
    """Return quantile radius for prediction bands."""
def sequence_conformal(y_hat_seq, q_radius) -> list[tuple[lo, hi]]:
    """Per-step bands for ADE/FDE-relevant targets."""
```

### Reporting

```python
# reporting/build.py
def build_run_report(run_dir: str, artifacts: dict) -> str:
    """Renders HTML/PDF; returns path. Includes tables+figs reproducibly."""
```

---

# 4) Statistical Validation Package (how to use it)

**Protocol:**

* Choose **split type**: `GroupKFold by MMSI` (unseen vessels) and **TimeSeriesSplit** (temporal generalization).
* For each fold: compute metric vectors (ADE/FDE per track).
* **Compare models** with paired tests on per-track metrics; report **BCa 95% CIs** and **Cliff’s δ**.

**Deliverables (auto-saved):**

* `results/table_model_comparison.csv` with mean±CI, p-values, δ.
* `figs/horizon_curves.png` (ADE/FDE vs horizon with CIs).
* `figs/violin_per_vessel_type.png`.

---

# 5) Baseline Suite (publication-critical)

* **Kalman CV/CT + IMM**: tuned process/measurement noise via CV grid; latent heading as circular state.
* **ARIMA/State-space**: seasonal models for SOG; coupled with dead-reckoning for position.
* **SVR/RF**: regress Δlat, Δlon, Δcog; **time-aware CV** (no leakage).
* **Particle Filter**: non-Gaussian maneuvers; resampling on acceleration bursts.

All baselines must implement:

```python
class TrajBaseline(Protocol):
    def fit(self, seqs: np.ndarray, meta: dict) -> None: ...
    def predict(self, seq: np.ndarray, horizon: int) -> np.ndarray: ...
```

---

# 6) Error Analysis & Failure Mining

* **Slicers**: by `vessel_type`, `wind_speed bins`, `traffic_density quantiles`, `distance_to_port`.
* **Horizon curves**: ADE/FDE per step (1..H).
* **Failure clusters**: k-means on feature snapshots of top-k worst tracks; label with dominant context (e.g., “high winds + fishing vessels near port”).
* **Case cards**: per failure→ plot past/future tracks, winds, nearby vessels, CPA/TCPA trajectory.

---

# 7) Maritime Domain Validation (operators care about this)

* **CPA/TCPA quality**: error distributions vs. ground truth; **early warning time** distribution (minutes before threshold breach).
* **COLREGS checks**: count/proportion of predicted trajectories that **reduce** potential violations vs baselines.
* **Operational metrics**:

  * **Warning Time** (median, 10th percentile)
  * **False Alert Rate** at operational thresholds
  * **Coverage** (% of traffic with reliable outputs)
  * **Throughput** (Hz per GPU/CPU)

---

# 8) Uncertainty Quantification (publishable & useful)

* **Conformal prediction** on ADE/FDE:

  * Calibrate on **held-out temporal slice** to avoid leakage.
  * Report **coverage vs. nominal** (e.g., 90%).
* **MC-Dropout / Ensembles**:

  * Correlate **predictive variance** with **error** (calibration curve).
  * Use variance-aware **risk gating** for alerts.

---

# 9) Interpretability

* **SHAP** on anomaly 16D features → ranked drivers per class/region.
* **Attention rollout** for TrAISformer/Motion-Transformer: produce **saliency over timesteps**; store as PNG + CSV.
* **Graph attention** (if used): edge importance heatmaps; map to vessel pairs.

---

# 10) Multi-Region Readiness (scaffold now)

* Region registry with **unit normalization** and CRS handling.
* **Leave-one-region-out** experiments defined in Hydra: `+exp=traj_loro`.
* Per-region **data statements**: traffic density, weather regimes, message noise.

---

# 11) Reporting: “Reviewer-ready” Artifacts

Auto-generated per run:

* **Methods (LaTeX)**: model configs, splits, metrics, statistical protocols.
* **Results (LaTeX + CSV)**: main tables with mean±95% CI, bold best; p-values with Holm-Bonferroni.
* **Figures**:

  * ADE/FDE horizon with CIs
  * Calibration curves (uncertainty)
  * ROC/PR for anomaly detection (with CIs)
  * CPA/TCPA scatter + density
* **Appendix**: ablations (feature drop, SSL weight β), sensitivity to proximity threshold (~5km).

---

# 12) Concrete Additions Per Experiment

### Trajectory Prediction

* **Metrics**: switch to great-circle ADE/FDE; ensure **circular RMSE** for COG.
* **Splits**: MMSI GroupKFold + temporal splits.
* **Baselines**: IMM-Kalman, ARIMA+DR, SVR/RF.
* **Uncertainty**: conformal bands per step; coverage report.
* **Deliverables**: per-horizon table; failure cards; variance-aware warnings.

### Anomaly Detection

* **Labels**: weak supervision using rule-based screens (speed>30kn, jumps>10km) + expert spot-checks.
* **Calibration**: threshold via ROC-optimal F1 on validation; report **FAR at target TPR**.
* **Explainability**: SHAP per anomaly type; top features per vessel class/region.

### Vessel Interaction (Graph)

* **Edge policy**: add **relative-motion aware edges** (CPA/TCPA-driven), not only radius.
* **Targets**: collision label robustness; include near-misses.
* **Metrics**: Collision F1, **CPA/TCPA RMSE**, lead-time histogram.
* **Ablations**: radius vs CPA-aware edges; static vs dynamic graphs.

### Multi-Modal Fusion

* **Ablation grid**: AIS only / AIS+weather / AIS+graph / AIS+static.
* **SSL check**: correlation of weather-trajectory embeddings; causal sanity (do weather shuffles drop performance?).
* **Generalization**: stress-test under extreme weather bins.

---

# 13) Reproducibility & Run Manifests

* Fix seeds across PyTorch, NumPy, CUDA.
* Save `manifest.json` with:

  * git commit, data hash, config hash, env (`pip freeze`), hardware.
* Every figure/table references a **manifest id**.

---

# 14) Milestones & Acceptance Criteria

**M1 (2 weeks): Stats + Baselines**

* `evalx/stats` finished (BCa CI, paired tests, δ).
* IMM-Kalman + SVR baselines in CI.
* Report with **mean±CI** and **p-values** vs. LSTM.

**M2 (4 weeks): Error + Domain**

* Failure mining; horizon curves; CPA/TCPA validators.
* Ops metrics (warning time, FAR).
* Report includes domain metrics and case cards.

**M3 (6–8 weeks): Uncertainty + Multi-Region**

* Conformal coverage ≥ nominal±2%.
* Leave-one-region-out scaffold + one external region ingested.

**M4 (10–12 weeks): Publication Pack**

* Auto LaTeX `methods.tex`, `results.tex`, and `figs/`.
* Complete ablations and interpretability panels.

---

## Short Conclusion

You have the models and data flow. To be **publication-strong** and **domain-compelling**, add:

1. **Stats rigor + baselines** (prove it),
2. **Domain validators** (make it operational),
3. **Uncertainty + interpretability** (make it trustworthy),
4. **Reporting automation** (make it paper-ready).

If you want, I can draft the initial stubs for `evalx/stats`, `maritime/cpa_tcpa`, and `reporting/build.py` exactly as above so you can drop them in and start wiring.
