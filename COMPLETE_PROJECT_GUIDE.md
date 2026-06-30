# 📘 Complete Project Guide (Beginner-Friendly, Step by Step)

This guide helps you build the **full, real project** in this repo — with **all the commands**, the
**necessary code**, and a **little "how & why"** for each step, in **easy language**.

> 🟢 **Two ways to use this guide:**
> 1. **Just understand + run it:** read the short bits and run the commands. The full code is already
>    in this repo — you don't have to retype it.
> 2. **Build your own from scratch:** copy each file's code (shown here, or from the repo file it
>    points to) into a fresh folder, in the order below.
>
> 🔰 **Totally new to this?** Start with **[`FROM_SCRATCH_GUIDE.md`](FROM_SCRATCH_GUIDE.md)** first
> (it builds a *simpler* version with full code), then come back here for the full version.
> To understand *how it all connects*, read **[`PROJECT_FLOW.md`](PROJECT_FLOW.md)**.

---

## 📑 Contents
1. [What you're building](#1-what-youre-building)
2. [Tools we use (plain English)](#2-tools-we-use-plain-english)
3. [Setup (commands)](#3-setup-commands)
4. [The folder layout](#4-the-folder-layout)
5. [Step A — Settings, logging, errors](#step-a--settings-logging-errors)
6. [Step B — Get the data](#step-b--get-the-data)
7. [Step C — Load + check + split](#step-c--load--check--split-data)
8. [Step D — Explore (EDA)](#step-d--explore-the-data-eda)
9. [Step E — The cleaning pipeline](#step-e--the-cleaning-pipeline-features)
10. [Step F — Stage 1: fraud model + MLflow](#step-f--stage-1-fraud-model--mlflow)
11. [Step G — Cost-based threshold](#step-g--cost-based-threshold)
12. [Step H — Stage 2: severity model](#step-h--stage-2-severity-model)
13. [Step I — Explanations (SHAP)](#step-i--explanations-shap)
14. [Step J — The brain: chained prediction](#step-j--the-brain-chained-prediction)
15. [Step K — The Flask web app](#step-k--the-flask-web-app)
16. [Step L — Train everything (one command)](#step-l--train-everything-one-command)
17. [Step M — Tests](#step-m--tests)
18. [Step N — Docker](#step-n--docker)
19. [Step O — CI/CD](#step-o--cicd-github-actions)
20. [Step P — Deploy (Render)](#step-p--deploy-free-on-render)
21. [All commands in one place](#-all-commands-in-one-place)
22. [Glossary](#-glossary-plain-english)

---

## 1. What you're building
A website where you enter an insurance claim and it tells you:
1. **Is it fraud?** ✅ Genuine or 🚩 Needs Review
2. **If genuine — how much money** the claim is worth.
3. **Why** (the top reasons).

Two models work together (chained): fraud check first; money estimate only if genuine.

## 2. Tools we use (plain English)
| Tool | Why we use it |
|---|---|
| **Python** | The language everything is written in. |
| **pandas / numpy** | Handle the data (tables and numbers). |
| **scikit-learn** | The main ML toolbox (builds & runs models). |
| **XGBoost** | A strong extra model we compare against. |
| **SHAP** | Explains *why* a model made a decision. |
| **MLflow** | A "logbook" of every training run (scores + the model). |
| **matplotlib / seaborn** | Draw charts (EDA + report). |
| **pandera / Pydantic** | Check data/input is valid so nothing crashes. |
| **Flask** | Turns the model into a web page + API. |
| **pytest** | Automated tests. |
| **Docker** | Packs the app so it runs the same anywhere. |
| **GitHub Actions** | Auto-runs tests + builds the image on every push. |

## 3. Setup (commands)
```bash
# 1. make a folder and a private environment for the packages
python -m venv .venv
.venv\Scripts\activate           # Windows   (Mac/Linux: source .venv/bin/activate)

# 2. install everything (the repo's requirements.txt lists exact versions)
pip install -r requirements.txt
```
> 💡 *Why a virtual environment?* It keeps this project's packages separate from other projects, so
> versions never clash.

## 4. The folder layout
See **[`PROJECT_FLOW.md` → file map](PROJECT_FLOW.md#5--file-by-file-map-what-each-file-is-for)** for
what every file does. In short: `src/` holds the ML code, `flask_app.py` serves it, `models/` holds the
trained files, and the rest is docs/tests/deploy.

---

## Step A — Settings, logging, errors

**Why first?** Pros never hardcode file paths or "magic numbers", never use bare `print()`, and make
errors easy to read. This is the difference between a toy and a real project.

**`config.yaml`** — every setting in ONE place (paths, model settings, the cost matrix). Example bits:
```yaml
data:
  target_fraud: fraud_reported
  target_severity: total_claim_amount
  test_size: 0.2
threshold:
  cost_false_negative: 30000   # cost of MISSING a fraud
  cost_false_positive: 5000    # cost of a FALSE alarm
```
👉 Full file: [`config.yaml`](config.yaml). Read by [`src/config.py`](src/config.py) (loads it once
and lets you do `cfg.get("threshold.cost_false_negative")`).

**`src/logger.py`** — logging instead of print (has timestamps, levels, and works in servers):
```python
import logging, sys
def get_logger(name):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    return logging.getLogger(name)
```
👉 Full version (with safe file logging) is in [`src/logger.py`](src/logger.py).

**`src/exception.py`** — errors that say *where* they happened:
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

---

## Step B — Get the data

**Dataset:** Kaggle *Auto Insurance Claims* — it has both `fraud_reported` and `total_claim_amount`
(one target per model). 👉 https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
Save it as **`data/raw/insurance_claims.csv`**.

**No Kaggle account?** This repo includes a generator that makes data with the *same columns*:
```bash
python -m scripts.generate_synthetic_data     # creates data/raw/insurance_claims.csv
```
> 💡 *Why a generator?* So anyone can run the whole project with **zero downloads**. To use the real
> data later, just drop the real CSV in `data/raw/` and re-run training.

---

## Step C — Load + check + split data

**Why:** Bad data is the #1 cause of ML bugs. We **validate** the data the moment we read it, then
**split** it into a part to learn from (train) and a part to check on (test).

Key idea from [`src/data_ingestion.py`](src/data_ingestion.py):
```python
from sklearn.model_selection import train_test_split

# 1) validate the columns/types with pandera (raises a clear error if data is wrong)
# 2) split, keeping the same fraud % in both halves (stratify)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["fraud_reported"])
```
> 💡 *Why "stratify"?* Fraud is rare. Without it, the test set might get very few frauds and your
> scores become unreliable.
> 💡 *Why a random split (not by date)?* We score each claim on its own as it arrives — it's not a
> "predict the future" time problem.

---

## Step D — Explore the data (EDA)

**Why:** Look before you model. Check how rare fraud is, whether amounts are skewed, and what relates
to fraud. Run:
```bash
python -m src.eda
```
This saves charts to `reports/plots/`. The key lessons (also why later choices make sense):
- Fraud is **rare** → **don't use accuracy**; use **PR-AUC**.
- Claim amount is **skewed** → **log-transform** it for the severity model.

👉 Code: [`src/eda.py`](src/eda.py).

---

## Step E — The cleaning pipeline (features)

**Why this is important:** training and serving **must** clean data the *same way*, or predictions
break. The trick: put all cleaning in **one** scikit-learn `Pipeline`, save it, and reuse it.

Core idea from [`src/feature_engineering.py`](src/feature_engineering.py):
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(stage):                 # stage = "fraud" or "severity"
    numeric = NUMERIC + (CLAIM_PARTS if stage == "fraud" else [])  # leakage guard #2
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])  # ignore unseen words
    return ColumnTransformer([("num", num, numeric), ("cat", cat, CATEGORICAL)])
```
> 💡 *Leakage guard #2:* the severity model does **not** get the claim-part columns
> (`injury_claim + property_claim + vehicle_claim = total`), or it would "cheat".
> 💡 `handle_unknown="ignore"` → if a category appears that wasn't in training, the app won't crash.

---

## Step F — Stage 1: fraud model + MLflow

**Why three models?** Start simple (Logistic Regression). If a fancy model can't beat it, your
*features* are the problem, not the model. Then try RandomForest and XGBoost. Pick the best by
**PR-AUC**. Track everything in **MLflow**.

Core idea from [`src/fraud_model.py`](src/fraud_model.py):
```python
import mlflow
from sklearn.metrics import average_precision_score

mlflow.set_experiment("fraud_classification")
for name, clf in candidates.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("preprocess", build_preprocessor("fraud")), ("model", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, proba)   # the score we choose on
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
# the best run is also registered in the MLflow Model Registry (a named, versioned model)
```
**See your experiments:** `mlflow ui` → open http://localhost:5000 (compare runs side by side).
> 💡 *Why not accuracy?* With ~25% fraud, "always genuine" is ~75% accurate but catches **zero** fraud.
> PR-AUC measures how well you find the *rare* class.

---

## Step G — Cost-based threshold

**Why:** the model gives a probability; *you* pick the cut-off. `0.5` assumes a false alarm and a
missed fraud cost the same — they don't. We pick the cut-off with the **lowest total cost**.

Core idea from [`src/threshold_optimizer.py`](src/threshold_optimizer.py):
```python
def total_cost(y_true, proba, t):
    pred = (proba >= t).astype(int)
    missed = ((y_true == 1) & (pred == 0)).sum()   # cost 30000 each
    alarms = ((y_true == 0) & (pred == 1)).sum()   # cost 5000 each
    return missed * 30000 + alarms * 5000

best_threshold = min(grid, key=lambda t: total_cost(y_test, proba, t))   # ≈ 0.26
```
> 💡 A missed fraud is 6× costlier here, so the best cut-off drops **below 0.5** — we accept more
> false alarms (sent to human review) to stop expensive fraud.

---

## Step H — Stage 2: severity model

**Why train on genuine claims only?** A fraud claim's amount is *fake*. Training the money model on
fake amounts ruins it. (This is **leakage guard #1**.) Also log-transform the skewed target.

Core idea from [`src/severity_model.py`](src/severity_model.py):
```python
from sklearn.compose import TransformedTargetRegressor
import numpy as np

genuine_train = train_df[train_df["fraud_reported"] == "N"]      # leakage guard #1
reg = TransformedTargetRegressor(                               # learn on log(amount)
    regressor=RandomForestRegressor(...), func=np.log1p, inverse_func=np.expm1)
severity = Pipeline([("preprocess", build_preprocessor("severity")), ("model", reg)])
severity.fit(genuine_train_X, genuine_train_y)
```
> 💡 We report the error as **MAE** (e.g. "off by ~₹12,592 per claim") because that's what a manager
> understands — not "RMSE=17889".

---

## Step I — Explanations (SHAP)

**Why:** nobody trusts a black box. SHAP says which fields pushed the decision, per claim — shown as
plain-English reasons in the app.

Core idea from [`src/explainability.py`](src/explainability.py):
```python
import shap
explainer = shap.TreeExplainer(model)          # works for RandomForest / XGBoost
shap_values = explainer(X_one_row)             # contributions for this claim
# then turn the biggest contributions into text like
#   "Incident severity is 'Major Damage' — increases fraud risk"
```

---

## Step J — The brain: chained prediction

**Why:** the website and the API must use the **same** logic. So we write it **once** here, and
everyone calls it.

Core idea from [`src/inference_pipeline.py`](src/inference_pipeline.py):
```python
class InferenceService:
    def predict(self, claim: dict) -> dict:
        row = pd.DataFrame([claim])
        p = float(self.fraud_model.predict_proba(row)[:, 1][0])   # Stage 1
        if p >= self.threshold:                                   # the chaining rule
            return {"verdict": "NEEDS_REVIEW", "fraud_probability": p, "severity": None}
        amount = float(self.severity_model.predict(row)[0])       # Stage 2 (genuine only)
        return {"verdict": "GENUINE", "fraud_probability": p, "predicted_severity": amount}
```
> 💡 The models are loaded **once** when the app starts (cached), so each request is fast.

---

## Step K — The Flask web app

**Why Flask?** A simple, popular way to put a model behind a web page. It just calls the brain above.

Core idea from [`flask_app.py`](flask_app.py):
```python
from flask import Flask, request, jsonify, send_from_directory
from src.inference_pipeline import get_service
from src.schemas import ClaimRequest          # Pydantic input validation

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.get("/")
def home():
    return send_from_directory("frontend", "index.html")

@app.post("/predict")
def predict():
    claim = ClaimRequest(**request.get_json(force=True))   # validate, then predict
    return jsonify(get_service().predict(claim.to_claim_dict()))
```
The web page is [`frontend/index.html`](frontend/index.html) + [`static/`](static/) (form → `fetch('/predict')`
→ show the result card).

**Run the app locally:**
```bash
python flask_app.py        # open http://localhost:5000
```

---

## Step L — Train everything (one command)

`src/train.py` runs all the steps above in order (ingest → EDA → fraud → threshold → severity → SHAP).
```bash
python -m scripts.generate_synthetic_data   # make data (skip if you have the real CSV)
python -m src.train                         # trains + saves everything to models/
```

---

## Step M — Tests

**Why:** tests catch mistakes automatically before users do. Run:
```bash
pytest -q
```
Example test (the API gives a money estimate for a genuine claim) — see [`tests/`](tests/):
```python
def test_genuine_gets_severity(client):
    r = client.post("/predict", json=sample_genuine_claim())
    assert r.get_json()["fraud"]["verdict"] == "GENUINE"
    assert r.get_json()["severity"]["predicted_amount"] is not None
```

---

## Step N — Docker

**Why:** Docker packs the app + its exact packages so it runs identically anywhere (your laptop, a
server). 👉 Full file: [`Dockerfile`](Dockerfile).
```bash
docker build -t fraud-app .
docker run -p 8000:8000 fraud-app          # open http://localhost:8000
```

---

## Step O — CI/CD (GitHub Actions)

**Why:** every time you push to GitHub, it should auto-run tests and build the image, so broken code
can't slip through. 👉 Full file: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). You don't
run anything — GitHub runs it for you and shows a green ✓ or red ✗.

---

## Step P — Deploy (free on Render)

**Why Render (not Netlify)?** Render runs a real Python server (our app needs one to load the models).
Netlify only hosts static pages, so it **can't** run this.
1. Push your code to GitHub.
2. https://dashboard.render.com → **New +** → **Web Service** → pick your repo → it detects the
   `Dockerfile` → choose the **Free** plan → **Create**.
3. Wait ~5–8 min → you get a public link like `https://your-app.onrender.com`. 🎉

(Detailed clicks + screenshots-style help are in [`HOW_TO_RUN.md`](HOW_TO_RUN.md).)

---

## 🧾 All commands in one place
```bash
# setup
python -m venv .venv
.venv\Scripts\activate                       # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# data + train
python -m scripts.generate_synthetic_data    # make practice data
python -m src.eda                            # (optional) charts -> reports/plots/
python -m src.train                          # train both models + threshold + SHAP
mlflow ui                                     # (optional) view experiments -> http://localhost:5000

# build the shareable report
python -m scripts.build_report               # -> reports/report.html

# run the app
python flask_app.py                          # -> http://localhost:5000

# quality
pytest -q                                     # run tests
ruff check src tests flask_app.py scripts monitoring   # lint

# docker
docker build -t fraud-app .
docker run -p 8000:8000 fraud-app            # -> http://localhost:8000
```

---

## 📖 Glossary (plain English)
| Term | Meaning |
|---|---|
| **Feature** | An input column the model learns from. |
| **Target** | What you predict (fraud Y/N, or the money amount). |
| **Train / test split** | Learn on one part, check on a part it never saw. |
| **Pipeline** | "Clean data + run model" as one object, identical in training and serving. |
| **Imbalanced data** | One class is rare (few frauds) → accuracy lies. |
| **PR-AUC** | A score focused on finding the rare class (fraud). We choose models by this. |
| **Threshold** | The "call it fraud" cut-off — chosen by cost, not 0.5. |
| **Data leakage** | Accidentally giving the model the answer. Avoided twice in this project. |
| **Log-transform** | Train on `log(amount)` when numbers are very skewed; convert back after. |
| **Chaining** | Use model 1's output to decide whether to run model 2. |
| **MLflow** | A logbook that records each training run (settings, scores, model). |
| **SHAP** | Tells you which features drove a single prediction. |
| **Drift** | When new data starts looking different from training data (time to retrain). |
| **CI/CD** | Automation that tests and builds your app on every code change. |

---

🎉 That's the whole project. For the *complete* code of any step, open the file it points to — it's all
in this repo. To understand how the pieces connect, read **[`PROJECT_FLOW.md`](PROJECT_FLOW.md)**.
Happy building! 🚀
