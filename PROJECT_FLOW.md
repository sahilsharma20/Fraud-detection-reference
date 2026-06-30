# 🔄 How This Project Works — The Flow (read this to understand it)

This file explains **how the project flows** — what happens, in what order, and **which file does
what**. No need to build anything here; this is just to *understand* the project you have.

---

## 1. The big picture (one screen)

```
                          ┌─────────────────────────────────────────────┐
   A claim comes in  ──▶  │  STAGE 1: Fraud model   → P(fraud) = 0..1     │
   (web form / API)       └─────────────────────────────────────────────┘
                                         │
                          is P(fraud) ≥ threshold (0.26)?
                                         │
                ┌────────────── YES ─────┴───── NO ──────────────┐
                ▼                                                 ▼
        🚩 NEEDS REVIEW                                 ┌──────────────────────┐
     (sent to a human,                                  │ STAGE 2: Severity     │
      NO money estimate)                                │ model → ₹ amount      │
                                                        └──────────────────────┘
                                                                  ▼
                                                        ✅ GENUINE + ₹ estimate
                                                           + "why" reasons (SHAP)
```

Two models, **chained**: the second one only runs if the first says "genuine".
Everything is served by **one Flask app**, and both the website and the API call **one** prediction
function (`src/inference_pipeline.py`) — so there's never two versions of the logic.

---

## 2. There are TWO flows: **training** and **serving**

A machine-learning app has two completely separate "times":
- **Training time** (done once, on your machine): learn from data, produce model files.
- **Serving time** (every time a user checks a claim): load those model files, answer instantly.

Understanding the project = understanding these two flows.

---

## 3. 🏋️ Training flow — what happens when you run `python -m src.train`

`src/train.py` is the conductor. It calls the other files **in this order**:

| # | Step | File that does it | What it produces |
|---|------|-------------------|------------------|
| 1 | **Load + check + split data** | `src/data_ingestion.py` | Reads `data/raw/insurance_claims.csv`, validates the columns (pandera), splits into train/test → `data/processed/train.csv`, `test.csv` |
| 2 | **Explore the data (EDA)** | `src/eda.py` | Charts (class balance, skew, correlations) → `reports/plots/`, and `eda_summary.json` |
| 3 | **Snapshot data for drift** | `monitoring/drift_monitor.py` | `models/reference_profile.json` (what "normal" data looks like) |
| 4 | **Train Stage 1 (fraud)** | `src/fraud_model.py` | Trains LogReg + RandomForest + XGBoost, picks best by **PR-AUC**, logs to **MLflow**, registers it, saves `models/fraud_model.joblib` |
| 5 | **Choose the threshold** | `src/threshold_optimizer.py` | Tries every cut-off, picks the cheapest using the cost matrix → saves `0.26` to `models/inference_config.json` |
| 6 | **Train Stage 2 (severity)** | `src/severity_model.py` | Trains on **genuine claims only** (leakage guard), log-transformed target, saves `models/severity_model.joblib` + the error band (MAE) |
| 7 | **Make explanations** | `src/explainability.py` | SHAP charts for both models → `reports/plots/` |
| 8 | **Save a summary** | `src/train.py` | `reports/metrics/training_summary.json` |

**Helpers used throughout:** `src/config.py` (reads all settings from `config.yaml`),
`src/logger.py` (prints nice logs), `src/feature_engineering.py` (the cleaning pipeline both models share).

➡️ **After training, the important outputs are the 2 model files + `inference_config.json`** in
`models/`. That's all the serving app needs.

---

## 4. ⚡ Serving flow — what happens when a user clicks "Check Claim"

```
[Browser]                [Flask]                    [The "brain"]                 [The models]
index.html  ──POST──▶  flask_app.py  ──calls──▶  inference_pipeline.py  ──uses──▶  *.joblib
   form data            /predict                  .predict(claim)                  files
      ▲                                                  │
      └──────────────── JSON result ◀────────────────────┘
   app.js shows
   the result card
```

Step by step:

1. **You fill the form** and click *Check Claim* — `static/app.js` collects the fields and sends them
   as JSON to `POST /predict`.
2. **`flask_app.py`** receives it, **validates** the input with `src/schemas.py` (Pydantic). Bad input
   → a clean error, never a crash.
3. It calls **`get_service().predict(claim)`** in `src/inference_pipeline.py` — the one and only
   prediction function. Inside it:
   - builds a 1-row table from the claim,
   - **Stage 1:** `fraud_model.joblib` → probability of fraud,
   - compares to the **threshold** (read from `inference_config.json`),
   - asks **SHAP** (`explainability.py`) for the top human-readable reasons,
   - **if flagged** → returns `NEEDS_REVIEW`, **no** severity,
   - **if genuine** → runs **Stage 2** `severity_model.joblib` → ₹ amount + a ± error band, and its reasons.
4. **`flask_app.py`** sends the result back as JSON.
5. **`app.js`** draws the result card (verdict, probability bar, ₹ amount, reasons).

➡️ **Notice:** the models are loaded **once** when the app starts (cached), so each request is fast.

---

## 5. 🗺️ File-by-file map (what each file is for)

```
config.yaml                  ← ALL settings (paths, model params, costs, threshold)
src/
  config.py                  ← reads config.yaml safely
  logger.py                  ← logging (instead of print)
  exception.py               ← clear error messages
  data_ingestion.py          ← load + validate + split data
  eda.py                     ← charts + data summary
  feature_engineering.py     ← the cleaning pipeline (shared by both models)
  fraud_model.py             ← Stage 1: train fraud models (+ MLflow)
  threshold_optimizer.py     ← pick the cost-best cut-off
  severity_model.py          ← Stage 2: train severity model (genuine only)
  explainability.py          ← SHAP: charts + per-claim "reasons"
  inference_pipeline.py      ← THE BRAIN: chained predict (one source of truth)
  schemas.py                 ← input validation rules for the API
  tracking.py                ← MLflow setup helper
  train.py                   ← runs the whole training in order
monitoring/
  drift_monitor.py           ← detects if new data drifts from training data
flask_app.py                 ← the web server (UI + /predict API)
frontend/index.html          ← the web page
static/styles.css, app.js    ← the page's look + behaviour
scripts/
  generate_synthetic_data.py ← makes practice data (Kaggle-shaped)
  build_report.py            ← builds the shareable reports/report.html
  make_notebook.py           ← builds the Jupyter walkthrough
tests/                       ← automated tests (pytest)
models/                      ← the trained model files (the app loads these)
reports/                     ← plots + the HTML report
Dockerfile, render.yaml      ← packaging + hosting
.github/workflows/ci.yml     ← runs tests + builds image on every push
```

---

## 6. 🧳 Follow ONE claim through the whole system (a story)

> A claim arrives: *Major Damage, no police report, ₹95,000, 3 AM.*

1. `app.js` packages it and POSTs to `/predict`.
2. `flask_app.py` checks the fields are valid (Pydantic) → OK.
3. `inference_pipeline.py` cleans it with the **same pipeline** used in training
   (`feature_engineering.py`), then the **fraud model** says **P(fraud) = 0.65**.
4. 0.65 ≥ 0.26 (the cost-optimal threshold) → **flagged**.
5. SHAP lists the reasons: *"Incident severity is 'Major Damage' — increases risk", "No police report",
   "High claim amount"*.
6. Because it's flagged, the **severity model is skipped** (we don't estimate money for suspicious
   claims).
7. Result returned: `NEEDS_REVIEW`, 65%, reasons. `app.js` shows the red card. 🚩

> A second claim: *Minor Damage, police report YES, ₹32,000, 2 PM.*
> → P(fraud) = 0.10 < 0.26 → **genuine** → severity model runs → **₹39,279 ± ₹12,592** → green card ✅.

---

## 7. 🔑 The 3 ideas that make this "senior", and where they live in the flow

1. **One cleaning pipeline, used in training AND serving** (`feature_engineering.py`) → predictions
   can't break from data being cleaned differently. (Saved inside each `*.joblib`.)
2. **Two leakage guards** (`severity_model.py` + `config.yaml`): fraud rows are dropped from severity
   training, and the claim-part columns are excluded from severity features.
3. **Cost-based threshold** (`threshold_optimizer.py`): the cut-off is chosen by money, not the
   default 0.5.

---

👉 Want to **build it yourself**? See **[`COMPLETE_PROJECT_GUIDE.md`](COMPLETE_PROJECT_GUIDE.md)**
(step-by-step, beginner-friendly, with the commands and code).
👉 Want to **just run it**? See **[`HOW_TO_RUN.md`](HOW_TO_RUN.md)**.
