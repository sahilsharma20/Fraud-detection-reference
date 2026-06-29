# 🧭 Build Guide — Insurance Claim Fraud Detection + Severity Prediction

> A complete, beginner-friendly, **step-by-step** guide to building this two-stage ML project from
> scratch using **Python · Jupyter · scikit-learn / XGBoost · SHAP · Flask**.
> Follow it top to bottom and you'll end with a trained ML system, a Flask web app, tests, Docker,
> and a deployable demo. Each step says **what** to do, **why**, and gives the **code**.

**What you're building:** a system that (1) predicts whether an insurance claim is **fraud**, and
(2) for genuine claims only, predicts the **claim amount (severity)**. The two models are *chained*:
flagged claims stop and go to a human; genuine claims get a money estimate. Every decision is
explained in plain English with SHAP.

---

## Table of contents
- [Prerequisites](#prerequisites)
- [The big picture](#the-big-picture)
- [Phase 0 — Project setup](#phase-0--project-setup)
- [Phase 1 — Get the data](#phase-1--get-the-data)
- [Phase 2 — Explore the data in Jupyter](#phase-2--explore-the-data-in-jupyter)
- [Phase 3 — Feature engineering (the pipeline)](#phase-3--feature-engineering-the-pipeline)
- [Phase 4 — Stage 1: fraud model](#phase-4--stage-1-fraud-model)
- [Phase 5 — Cost-sensitive threshold](#phase-5--cost-sensitive-threshold)
- [Phase 6 — Data leakage + Stage 2: severity model](#phase-6--data-leakage--stage-2-severity-model)
- [Phase 7 — Explainability with SHAP](#phase-7--explainability-with-shap)
- [Phase 8 — Chained inference (one source of truth)](#phase-8--chained-inference-one-source-of-truth)
- [Phase 9 — The Flask web app](#phase-9--the-flask-web-app)
- [Phase 10 — Testing](#phase-10--testing)
- [Phase 11 — Docker](#phase-11--docker)
- [Phase 12 — CI/CD with GitHub Actions](#phase-12--cicd-with-github-actions)
- [Phase 13 — Deploy (free hosting)](#phase-13--deploy-free-hosting)
- [Phase 14 — The shareable report](#phase-14--the-shareable-report)
- [Common mistakes to avoid](#common-mistakes-to-avoid)

---

## Prerequisites
- **Python 3.12** installed (`python --version`)
- **pip** and **git**
- A code editor (VS Code recommended)
- Basic Python + a little ML familiarity (what a model/feature/train-test split is)

You do **not** need a GPU. Everything runs on a laptop in minutes.

---

## The big picture

```
A claim comes in
      │
      ▼
┌──────────────┐   P(fraud) ≥ threshold?
│  STAGE 1     │ ───────── YES ─────────►  🚩 NEEDS REVIEW  (human investigates, no money estimate)
│ Fraud model  │
└──────────────┘ ───────── NO ──────────►  ┌──────────────┐
                                            │  STAGE 2     │ ─► ✅ GENUINE + ₹ severity estimate
                                            │ Severity reg.│
                                            └──────────────┘
```

We use **Jupyter** to explore and prototype the ML, then move the logic into clean **Python
modules** in `src/`, and serve it with a **Flask** web app.

---

## Phase 0 — Project setup

**Why first?** Good structure and config up front is what separates a real project from a messy
notebook. We never hardcode paths or "magic numbers" — they live in `config.yaml`.

### 0.1 Create the folder structure
```
fraud-detection/
├── src/                 # python package: all the ML logic
├── notebooks/           # Jupyter exploration
├── tests/               # pytest tests
├── frontend/            # the web UI (HTML)
├── static/              # CSS + JS for the UI
├── data/raw/            # the dataset goes here
├── models/              # trained models get saved here
├── reports/             # plots + the final report
├── config.yaml          # ALL settings in one place
├── flask_app.py         # the Flask web server
└── requirements.txt
```

### 0.2 Create a virtual environment & install packages
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install pandas numpy scikit-learn xgboost shap mlflow flask flask-cors \
            matplotlib seaborn pyyaml joblib pydantic pandera pytest \
            notebook ipykernel
pip freeze > requirements.txt   # pin versions so the build is reproducible
```

### 0.3 The config file — `config.yaml`
Put every path, hyperparameter, and threshold here so code stays clean:
```yaml
project: { name: fraud-severity, random_state: 42 }
paths:
  raw_data: data/raw/insurance_claims.csv
  fraud_model: models/fraud_model.joblib
  severity_model: models/severity_model.joblib
data:
  target_fraud: fraud_reported
  target_severity: total_claim_amount
  test_size: 0.2
threshold:
  cost_false_negative: 30000    # cost of MISSING a fraud (it gets paid out)
  cost_false_positive: 5000     # cost of a FALSE alarm (investigation + customer trust)
business:
  currency_symbol: "₹"
  annual_claim_volume: 100000
  fraud_rate: 0.10
  avg_fraud_payout: 30000
```

### 0.4 Logging & error handling (no `print()`!)
**Why:** `print()` has no levels/timestamps and can't be turned off in production. Configure logging
once and reuse it everywhere.

`src/logger.py`:
```python
import logging, sys
def get_logger(name):
    logging.basicConfig(level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
    return logging.getLogger(name)
```

`src/exception.py` (errors that tell you *where* they happened):
```python
import sys, traceback
class FraudDetectionError(Exception):
    def __init__(self, error):
        _, _, tb = sys.exc_info()
        if tb:
            last = traceback.extract_tb(tb)[-1]
            error = f"{type(error).__name__} in {last.filename}:{last.lineno} -> {error}"
        super().__init__(str(error))
```

> 📁 The full, production versions are in `src/logger.py`, `src/exception.py`, `src/config.py`.

---

## Phase 1 — Get the data

**Dataset:** *Auto Insurance Claims Fraud Detection* (Kaggle). It has both a `fraud_reported` label
**and** a `total_claim_amount` field — we need both, one per stage.
👉 https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
Download it and save as **`data/raw/insurance_claims.csv`**.

**No Kaggle account?** This repo includes `scripts/generate_synthetic_data.py`, which creates a
dataset with the **exact same 40 columns** and a realistic fraud signal, so the whole project runs
without any download:
```bash
python -m scripts.generate_synthetic_data   # writes data/raw/insurance_claims.csv
```
The key idea of the generator is to keep the real arithmetic identity:
`total_claim_amount = injury_claim + property_claim + vehicle_claim` (this matters in Phase 6).

---

## Phase 2 — Explore the data in Jupyter

**Why Jupyter here?** EDA is interactive and visual — perfect for a notebook. Open it:
```bash
jupyter notebook        # then open notebooks/fraud_severity_project.ipynb
```
> 📓 This repo ships a ready, fully-executed notebook: **`notebooks/fraud_severity_project.ipynb`**.
> It mirrors this whole guide interactively. The cells below are its essence.

```python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = pd.read_csv("data/raw/insurance_claims.csv")

# THE key insight: the fraud label is imbalanced
fraud_rate = (df.fraud_reported == "Y").mean()
print(f"Fraud rate: {fraud_rate:.1%}")
print(f"'Always genuine' accuracy: {1-fraud_rate:.1%}  <-- but catches ZERO fraud!")
```

**⚠️ The #1 beginner trap:** using **accuracy** for fraud. With ~25% fraud, a model that always says
"genuine" is ~75% accurate and useless. So we will measure **Precision, Recall, F1, and PR-AUC**
instead. Also check the severity target — it's **right-skewed**, so we'll log-transform it later:
```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df.fraud_reported.value_counts().plot.bar(ax=ax[0], color=["#2e7d32", "#c62828"])
sns.histplot(df.total_claim_amount, bins=40, kde=True, ax=ax[1])
plt.show()
```

---

## Phase 3 — Feature engineering (the pipeline)

**Why a `Pipeline`?** Training and serving **must** transform data identically. If you clean data in
the notebook one way and in the API another way, predictions silently break. The fix: put *all*
transforms in **one** scikit-learn `Pipeline`, fit it on the training data, save it, and reuse the
exact same object at prediction time.

`src/feature_engineering.py` (core idea):
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd, numpy as np

class RawFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives 'customer_tenure_days' from dates, drops IDs, cleans '?' -> NaN."""
    def __init__(self, bind_col, incident_col, drop_cols):
        self.bind_col, self.incident_col, self.drop_cols = bind_col, incident_col, drop_cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        df["customer_tenure_days"] = (pd.to_datetime(df[self.incident_col], errors="coerce")
            - pd.to_datetime(df[self.bind_col], errors="coerce")).dt.days.clip(lower=0).fillna(0)
        df = df.drop(columns=[c for c in self.drop_cols + [self.bind_col, self.incident_col]
                              if c in df.columns])
        return df.replace(["?", ""], np.nan)

def build_preprocessor(numeric_cols, categorical_cols):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            # handle_unknown='ignore' => unseen categories don't crash inference
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, numeric_cols),
                              ("cat", categorical, categorical_cols)])
```

**Two pitfalls avoided:**
1. **Never fit on test data** — fit the pipeline on the training fold only.
2. `OneHotEncoder(handle_unknown="ignore")` means a category never seen in training won't crash a
   live request.

> 📁 Full version handles per-stage feature lists and the severity leakage columns (Phase 6).

---

## Phase 4 — Stage 1: fraud model

**Why three models?** Start with a simple baseline (Logistic Regression). If a fancy model can't beat
it, your features (not the model) are the problem. Then try Random Forest and XGBoost.

`src/fraud_model.py` (core training loop):
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score
import mlflow, joblib

X = df.drop(columns=["fraud_reported"])
y = (df.fraud_reported == "Y").astype(int)
# stratify => keep the same fraud % in train and test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    "xgb": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, eval_metric="aucpr"),
}
best, best_score = None, -1
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pr_auc = average_precision_score(yte, proba)   # <-- the metric we select on
    with mlflow.start_run(run_name=name):          # MLflow tracks every experiment
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
    print(name, "PR-AUC", round(pr_auc, 3))
    if pr_auc > best_score:
        best, best_score = pipe, pr_auc
joblib.dump(best, "models/fraud_model.joblib")     # save the winner for serving
```

**MLflow** records params/metrics/models for every run so you can compare them later (run
`mlflow ui` to see a dashboard). The repo also **registers** the best model in the MLflow Model
Registry (a versioned, promotable artifact — what production teams deploy from).

---

## Phase 5 — Cost-sensitive threshold

**Why?** The model outputs a probability; *you* choose the cut-off. The default 0.5 assumes a false
alarm and a missed fraud cost the same — they don't. A **missed fraud** costs the payout (₹30,000); a
**false alarm** costs an investigation (₹5,000). So we sweep the threshold to minimise total cost:

`src/threshold_optimizer.py` (core):
```python
import numpy as np
COST_FN, COST_FP = 30000, 5000          # from config.yaml
def total_cost(y_true, proba, t):
    pred = (proba >= t).astype(int)
    fn = ((y_true == 1) & (pred == 0)).sum()   # missed fraud
    fp = ((y_true == 0) & (pred == 1)).sum()   # false alarm
    return fn * COST_FN + fp * COST_FP

grid = np.linspace(0.01, 0.99, 99)
costs = [total_cost(yte, proba, t) for t in grid]
chosen = grid[int(np.argmin(costs))]    # e.g. ~0.26, well below 0.5
print("Cost-optimal threshold:", round(chosen, 2))
```
Because a missed fraud is 6× costlier, the best threshold drops **below 0.5** — we accept more false
alarms (which go to human review, not auto-denial) to stop expensive fraud. **Save this number**; the
inference step uses it.

---

## Phase 6 — Data leakage + Stage 2: severity model

This is the most important ML-judgement part of the project. There are **two** leakage traps:

**Leakage trap #1 — exclude fraud claims from severity training.**
A fraudulent claim's amount is *fabricated* (what the crook tried to extract). If you train the
"how much to reserve" model on those numbers, it learns inflated amounts and ruins genuine
predictions. Also, in production you only ever predict severity for genuine claims — so train on
genuine claims only:
```python
genuine = df[df.fraud_reported == "N"]      # <-- LEAKAGE GUARD #1
```

**Leakage trap #2 — drop the claim-component columns.**
`total_claim_amount = injury_claim + property_claim + vehicle_claim` exactly. If you feed those three
to the model, it just adds them up (R²≈1.0) and learns nothing. **Exclude them** from the severity
features (but keep them for the *fraud* model, where they're fine).

**Train the severity model on a log-transformed target** (because it's skewed), inverting back to
rupees automatically:
```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

Xs = genuine.drop(columns=["total_claim_amount", "fraud_reported",
                           "injury_claim", "property_claim", "vehicle_claim"])  # guard #2
ys = genuine["total_claim_amount"].astype(float)
Xstr, Xste, ystr, yste = train_test_split(Xs, ys, test_size=0.2, random_state=42)

reg = TransformedTargetRegressor(           # train on log1p(y), predict back in ₹
    regressor=RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42),
    func=np.log1p, inverse_func=np.expm1)
sev_pipe = Pipeline([("prep", severity_preprocessor), ("model", reg)])
sev_pipe.fit(Xstr, ystr)
pred = sev_pipe.predict(Xste)
print("MAE ₹", round(mean_absolute_error(yste, pred)), "| R²", round(r2_score(yste, pred), 2))
joblib.dump(sev_pipe, "models/severity_model.joblib")
```
Then translate the error into business language: *"on average the reserve estimate is off by ₹X per
claim."* A claims manager understands that; "RMSE=17889" they don't.

---

## Phase 7 — Explainability with SHAP

**Why?** Nobody trusts a black box that says "fraud" with no reason. SHAP tells you *which features*
pushed the decision, per claim. We turn raw SHAP values into plain-English reasons for the UI.
```python
import shap
prep, model = sev_pipe.named_steps["prep"], best.named_steps["model"]
X_trans = best.named_steps["prep"].transform(Xte[:200])
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_trans)
shap.summary_plot(shap_values[..., 1], X_trans)   # global importance for the fraud model
```
> 📁 `src/explainability.py` also produces per-request reasons like *"Incident severity is 'Major
> Damage' — increases fraud risk"*, shown on the result card.

---

## Phase 8 — Chained inference (one source of truth)

**Why this matters:** the web app and any API must use the **exact same** prediction logic, or they'll
disagree. So write the chaining **once** and have everything call it.

`src/inference_pipeline.py` (the heart of the system):
```python
import joblib, time, numpy as np, pandas as pd

class InferenceService:
    def __init__(self):
        self.fraud = joblib.load("models/fraud_model.joblib")
        self.severity = joblib.load("models/severity_model.joblib")
        self.threshold = 0.26          # the cost-optimal value from Phase 5

    def predict(self, claim: dict) -> dict:
        row = pd.DataFrame([claim])
        p = float(self.fraud.predict_proba(row)[:, 1][0])
        if p >= self.threshold:                       # ── the chaining rule ──
            return {"verdict": "NEEDS_REVIEW", "fraud_probability": round(p, 3),
                    "severity": None, "note": "flagged for manual review"}
        amount = float(self.severity.predict(row)[0])  # only genuine claims reach here
        return {"verdict": "GENUINE", "fraud_probability": round(p, 3),
                "predicted_severity": round(max(amount, 0))}

service = InferenceService()   # load once, reuse for every request
```

---

## Phase 9 — The Flask web app

**Why Flask?** It's a simple, popular Python web framework — perfect for serving a model behind a web
page. Our Flask app is a thin shell over the inference service from Phase 8 (no duplicated logic) and
serves a clean HTML form.

`flask_app.py`:
```python
from flask import Flask, jsonify, request, send_from_directory
from src.inference_pipeline import service   # the ONE prediction implementation

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.get("/")
def home():
    return send_from_directory("frontend", "index.html")

@app.post("/predict")
def predict():
    claim = request.get_json(force=True)
    return jsonify(service.predict(claim))

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```
Run it:
```bash
python flask_app.py        # open http://localhost:5000
```

The **frontend** (`frontend/index.html` + `static/styles.css` + `static/app.js`) is a simple form that
collects claim fields and `fetch()`es `/predict`, then shows a result card (verdict, fraud
probability, severity, reasons). Minimal JS:
```javascript
const res = await fetch("/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(collectFormValues()),
});
const result = await res.json();
renderResultCard(result);   // verdict badge + probability bar + severity + reasons
```
> 📁 The polished UI (status colours, probability bar, reason chips) is in `frontend/` and `static/`.
> **Tip:** validate the request with Pydantic (`src/schemas.py`) so bad input returns a clean error
> instead of crashing the model.

---

## Phase 10 — Testing

**Why?** Tests catch regressions before your users do, and let CI verify every change automatically.
```python
# tests/test_threshold.py
import numpy as np
from src.threshold_optimizer import total_cost
def test_cost_counts_mistakes():
    y = np.array([1, 0, 1, 0]); proba = np.array([0.9, 0.1, 0.2, 0.8])
    cost, fn, fp = ... # at 0.5 -> 1 missed fraud + 1 false alarm
    assert fn == 1 and fp == 1

# tests/test_flask.py  (integration — hits the real endpoint via Flask's test client)
def test_genuine_gets_severity(client):     # client = flask_app.app.test_client()
    r = client.post("/predict", json=sample_genuine_claim())
    body = r.get_json()
    assert body["fraud"]["verdict"] == "GENUINE"
    assert body["severity"]["predicted_amount"] is not None
```
Run: `pytest -q`.

---

## Phase 11 — Docker

**Why?** Docker packages your app + its exact dependencies so it runs identically anywhere (your
laptop, CI, the cloud). Use a **multi-stage** build (small final image) and a **non-root** user.

`Dockerfile`:
```dockerfile
FROM python:3.12-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && useradd --create-home appuser
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src/ ./src/
COPY flask_app.py ./
COPY frontend/ ./frontend/
COPY static/ ./static/
COPY models/ ./models/
USER appuser
CMD ["sh", "-c", "gunicorn flask_app:app -b 0.0.0.0:${PORT:-5000}"]
```
Build & run: `docker build -t fraud-app . && docker run -p 5000:5000 fraud-app`.

---

## Phase 12 — CI/CD with GitHub Actions

**Why?** Every push should automatically run tests and build the image, so a broken change can't reach
production unnoticed.

`.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: pytest -q
  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t fraud-app .
```

---

## Phase 13 — Deploy (free hosting)

Push to GitHub, then deploy on **Render** (free tier):
1. Create a GitHub repo and push:
   ```bash
   git init && git add . && git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. On https://dashboard.render.com → **New +** → **Web Service** (or **Blueprint** if you include a
   `render.yaml`) → connect the repo.
3. Settings: **Environment = Docker** (it auto-detects the Dockerfile), Health check path `/health`.
4. Click **Create** — Render builds and gives you a public `https://…onrender.com` URL.
5. Put that URL in your README.

> 💡 Commit the trained `models/*.joblib` (~10 MB) so the app deploys with **no training step** —
> the most reliable path on free tiers.

---

## Phase 14 — The shareable report

A single, self-contained `reports/report.html` (images embedded as base64, openable in any browser,
printable to PDF) that explains the project to non-technical stakeholders: the architecture diagram,
metrics tables, the key plots, the **business impact** (with assumptions), screenshots, the leakage
decision, and limitations.
> 📁 `scripts/build_report.py` generates it from the training artifacts. Run `python -m
> scripts.build_report`.

---

## Common mistakes to avoid

| Mistake (fresher) | What to do instead |
|---|---|
| Using **accuracy** for fraud | Use **PR-AUC / precision / recall** (imbalanced data) |
| Fitting the scaler/encoder on **all** data | Fit on **train only** (inside a `Pipeline`) |
| Shipping the **0.5** threshold | Tune it with a **cost matrix** |
| Training severity on **fraud claims** | **Exclude** them (fabricated amounts) |
| Feeding `injury+property+vehicle` to severity | **Drop** them (they sum to the target) |
| Re-implementing prediction in the UI and API | **One** inference module both call |
| `print()` everywhere; hardcoded paths | **Logging** + a **`config.yaml`** |
| No tests, manual deploys | **pytest** + **CI** + **Docker** |

---

🎉 **That's the whole project.** For the complete, production-grade code of every step, read the
matching file in `src/` (this guide shows the essence; the repo has the full version with type hints,
docstrings, and error handling). Open `notebooks/fraud_severity_project.ipynb` to run the ML steps
interactively, and `reports/report.html` for the final write-up.
