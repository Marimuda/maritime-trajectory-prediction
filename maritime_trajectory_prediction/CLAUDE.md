# Project Refactor Blueprint

### *Internal Technical Specification for the Maritime‑Trajectory‑Prediction codebase*

> **Document codename:** **team CLAUDE**
> **Audience:** Senior & mid‑level Python / ML engineers
> **Maintainer:** Platform Engineering Guild

---

\## 1 Executive Summary
The **maritime‑trajectory‑prediction** repository accumulated endpoint sprawl, duplicated logic, and configuration drift. This blueprint prescribes a **single‑entry, Hydra‑driven architecture** that unifies preprocessing, training, hyper‑parameter sweeps, inference, and evaluation while improving testability and developer experience.
*All timeline estimates have been removed; work is assumed to be executed by autonomous agents unless otherwise noted.*

---

\## 2 Objectives

|  #  | Goal                                                                  | Success Metric                              |
| --- | --------------------------------------------------------------------- | ------------------------------------------- |
|  1  | Eliminate duplicated scripts; converge to **one CLI (`main.py`)**     | 100 % existing use‑cases runnable via CLI   |
|  2  | Centralise configs under **Hydra composition**                        | Zero hard‑coded paths/flags in source       |
|  3  | Extract reusable modules (data, models, trainers, inference)          | >80 % code covered by unit tests            |
|  4  | Provide deterministic CI pipeline (lint, type‑check, tests, coverage) | All automated checks green on every PR      |
|  5  | Remove deprecated scripts listed in §5                                | Deprecated scripts deleted from main branch |

---

\## 3 Current State Analysis
\### 3.1 Repositories & Key Scripts

```
maritime_trajectory_prediction/
    configs/           # Hydra‑style but not wired into all scripts
    src/
        data/          # rich but partially duplicated
        experiments/   # standalone train/eval logic
        models/        # baseline + SOTA + duplicates
        train_lightning.py
predict_trajectory.py               # ad‑hoc inference
inference_transformer_models.py     # richer inference but overlapping
train_simple_model.py               # quick validation script
process_ais_catcher.py              # ingestion pipeline (stand‑alone)
```

*Multiple entrypoints duplicate data loading, model factory logic, and logging.*

\### 3.2 Pain Points

* **Config drift:** YAMLs not honoured by every script.
* **Environment coupling:** paths & device flags sprinkled across code.
* **Testing gaps:** Large monolithic scripts untestable without real data.
* **On‑boarding friction:** New devs need tribal knowledge to pick the right script.

---

\## 4 Target Architecture

```
maritime_trajectory_prediction/
├── configs/                 # Hydra groups (data/, model/, trainer/, …)
├── src/
│   ├── config/             # pydantic/omegaconf dataclasses & ConfigStore
│   ├── data/
│   │   ├── preprocess.py   # CLI‑called runner
│   │   ├── datamodule.py   # Lightning‑ready, reused everywhere
│   ├── models/
│   │   └── factory.py      # single entry to build any model described in configs
│   ├── training/
│   │   └── trainer.py      # wraps PL Trainer + baseline fallback
│   ├── inference/
│   │   └── predictor.py    # predict / batch / realtime
│   ├── experiments/
│   │   ├── train.py        # thin wrapper around training.trainer
│   │   └── evaluation.py   # metric report generator
│   └── utils/
└── main.py                  # Hydra @main – dispatch by cfg.mode
```

*Hydra groups*: `mode/`, `data/`, `model/`, `trainer/`, `logger/`, `callbacks/`, `experiment/`.

---

\## 5 Scripts Slated for Deletion or Absorption
The following top‑level scripts will be **retired** once their logic is subsumed by consolidated modules. Tests referencing them must be updated or removed.

| Path                              | Reason for removal                                              | New home                                           |
| --------------------------------- | --------------------------------------------------------------- | -------------------------------------------------- |
| `predict_trajectory.py`           | Ad‑hoc trajectory inference, single‑task                        | `src/inference/predictor.py`                       |
| `inference_transformer_models.py` | Expanded inference but duplicates model loading & preprocessing | `src/inference/predictor.py`                       |
| `train_simple_model.py`           | Validation hack, shadows Hydra configs                          | `src/experiments/train.py` test mode               |
| `train_lightning.py`              | Stand‑alone trainer wrapper                                     | refactored into `src/training/trainer.py`          |
| `evaluate_transformer_models.py`  | Monolithic evaluation/benchmark                                 | `src/experiments/evaluation.py`                    |
| `process_ais_catcher.py`          | Stand‑alone ingest pipeline                                     | logic moved to `src/data/preprocess.py`            |
| `test_benchmark_models.py`        | Duplicates unit/perf tests, uses old factory                    | folded into `tests/performance/test_benchmarks.py` |

*Note:* unit & integration test files remain; they will be updated to call the unified CLI.

---

\## 6 Hydra Configuration Strategy

```
# configs/config.yaml
defaults:
  - mode: train       # train | preprocess | predict | evaluate
  - data: ais_processed
  - model: traisformer
  - trainer: gpu
  - logger: tensorboard
  - callbacks: default
  - experiment: base
```

\### Example Overrides

```bash
# Sweep
python main.py experiment=traisformer_sweep hydra.sweeper.n_trials=50

# Real‑time inference
python main.py mode=predict data.file=ais.parquet model.path=outputs/best.pt predict.output=preds.csv
```

---

\## 7 Coding Conventions & Tooling

| Area       | Tool/Standard                                                  | Action                                                        |
| ---------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| Code style | **ruff + black**                                               | Enforce via pre‑commit                                        |
| Typing     | **mypy (strict)**                                              | Enable gradual typing, fail build on new violations           |
| Logging    | **loguru**                                                     | Replace bare `logging.*` usages                               |
| Config     | **Hydra + pydantic**                                           | Dataclass validation & defaults                               |
| Testing    | **pytest** with markers `unit` / `integration` / `performance` | 90 % critical‑path coverage                                   |
| CI         | GitHub Actions                                                 | Steps: lint → type‑check → tests → coverage → artifact upload |

---

\## 8 Testing & CI

1. **Fixtures** updated to launch the unified CLI via Hydra for end‑to‑end tests.
2. **Contract test**: sample AIS log flows through preprocess → train (dry‑run epoch) → predict without error.
3. **Performance benchmarks** invoke `main.py mode=benchmark` (to be added under `src/experiments/benchmark.py`).
4. Coverage gate: `pytest --cov=src -m "not slow"` on every push.

---

\## 9 Appendix
\### A. `main.py` skeleton

```python
@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    from hydra.utils import call

    registry = {
        "preprocess": "src.data.preprocess.run_preprocess",
        "train": "src.experiments.train.run_training",
        "evaluate": "src.experiments.evaluation.run_evaluation",
        "predict": "src.inference.predictor.run_prediction",
    }
    call(registry[cfg.mode], cfg)
```

\### B. ConfigStore registration snippet

```python
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=RootConfig)
cs.store(group="mode", name="train", node={"mode": "train"})
# repeat for each group …
```

---

**End of document – team CLAUDE**
