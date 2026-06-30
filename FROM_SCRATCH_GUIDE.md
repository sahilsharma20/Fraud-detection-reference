# 👶➡️👨‍💻 Build Your Own Fraud + Severity Project (Fresher Edition)

This guide teaches you to build a **simpler version** of this project **from a blank folder**, with
**all the code you need**. It's written for a **complete beginner** — every step says *what to type*,
*what it does*, and *why*.

By the end you'll have a working web app where you type in an insurance claim and it tells you:
1. **Is it fraud?** (Genuine ✅ or Needs Review 🚩)
2. **If genuine — how much money** the claim is worth.

> 🧠 We use only **3 main tools**: **Python**, **scikit-learn** (machine learning), and **Flask**
> (the web part). That's it — easy to install, hard to break.
>
> When you're ready for the "pro" version (XGBoost, SHAP explanations, MLflow, Docker, tests), the
> full code is in this same repo — see the **[Level up](#-level-up-to-the-full-project)** section at the end.

---

## 📑 What we'll build (the plan)

```
A claim  →  [Model 1: Fraud check]  →  is it risky?
                                         ├─ YES → "Needs Review" (stop, no money guess)
                                         └─ NO  → [Model 2: Money estimate] → show ₹ amount
```

Our tiny project will have just **7 files**:
```
my-fraud-project/
├── requirements.txt      # the packages we need
├── config.py             # all our settings in one place
├── generate_data.py      # makes practice data (a CSV file)
├── train.py              # trains both models and saves them
├── predict.py            # the "brain": takes a claim, returns the answer
├── app.py                # the Flask website
├── templates/
│   └── index.html        # the web page (form + result)
```

---

## ✅ Step 0 — Install Python (one time)

Download **Python 3.12** from https://www.python.org/downloads/ and **tick "Add Python to PATH"**
during install. Check it worked:
```bash
python --version
```
You should see `Python 3.12.x`.

---

## ✅ Step 1 — Make the project folder and a "virtual environment"

A **virtual environment** is just a private box for this project's packages, so they don't mess with
other projects.

```bash
mkdir my-fraud-project
cd my-fraud-project
python -m venv .venv

# turn it on:
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
```
When it's on, your terminal line starts with `(.venv)`.

---

## ✅ Step 2 — List and install the packages

Create a file **`requirements.txt`**:
```text
pandas
numpy
scikit-learn
flask
joblib
matplotlib
seaborn
mlflow
```
Install them:
```bash
pip install -r requirements.txt
```
*(First time takes a couple of minutes — it's downloading. That's normal.)*

**What each one is (simple):**
| Package | What it does |
|---|---|
| **pandas / numpy** | Work with data — tables and numbers. |
| **scikit-learn** | The machine-learning toolbox (builds & runs the models). |
| **matplotlib / seaborn** | Draw charts so you can *see* your data (used in EDA). |
| **mlflow** | A "logbook" that records every training run (scores, settings, the model). |
| **flask** | Turns your model into a web page with a button. |
| **joblib** | Saves a trained model to a file so you can reuse it. |

---

## ✅ Step 3 — Put all settings in `config.py`

**Why:** Never scatter numbers and file paths around your code. Keep them in **one** place so you can
change them easily. Create **`config.py`**:

```python
# config.py — every setting lives here (so code stays clean)

RANDOM_STATE = 42          # a fixed number => same results every run (reproducible)

# where files live
DATA_FILE          = "data/claims.csv"
FRAUD_MODEL_FILE   = "models/fraud_model.joblib"
SEVERITY_MODEL_FILE = "models/severity_model.joblib"
THRESHOLD_FILE     = "models/threshold.txt"

# the two things we predict
TARGET_FRAUD    = "fraud_reported"      # Y / N
TARGET_SEVERITY = "total_claim_amount"  # the money amount

# the input columns the models learn from
NUMERIC = ["age", "months_as_customer", "policy_annual_premium",
           "incident_hour", "number_of_vehicles", "witnesses"]

CATEGORICAL = ["incident_type", "incident_severity",
               "authorities_contacted", "police_report_available"]

# These are fine for the FRAUD model, but they are CHEATING for the severity model
# (because injury + property + vehicle = total, and total IS what severity predicts).
# So we only give them to the fraud model. (This is "data leakage" — explained later.)
FRAUD_ONLY_NUMERIC = ["injury_claim", "property_claim", "vehicle_claim", "total_claim_amount"]

# Cost matrix: a MISSED fraud costs us more than a FALSE alarm.
COST_FALSE_NEGATIVE = 30000   # we paid out a fraud we should have caught
COST_FALSE_POSITIVE = 5000    # we investigated a genuine claim for nothing
```

---

## ✅ Step 4 — Make practice data with `generate_data.py`

Normally you'd download a dataset (e.g. the Kaggle *Auto Insurance Claims* dataset). To keep this
guide self-contained, we'll **generate** a realistic practice dataset. Create **`generate_data.py`**:

```python
# generate_data.py — creates a practice dataset (data/claims.csv)
import os
import numpy as np
import pandas as pd
import config as C


def main():
    rng = np.random.default_rng(C.RANDOM_STATE)   # random generator with a fixed seed
    n = 1500                                       # number of claims to make

    # ----- random claim details -----
    severity   = rng.choice(["Trivial", "Minor", "Major", "Total Loss"], n, p=[0.2, 0.4, 0.25, 0.15])
    police     = rng.choice(["YES", "NO", "UNKNOWN"], n, p=[0.4, 0.35, 0.25])
    authorities = rng.choice(["Police", "Fire", "Ambulance", "None"], n, p=[0.4, 0.2, 0.2, 0.2])
    itype      = rng.choice(["Single Vehicle", "Multi Vehicle", "Theft", "Parked Car"], n)
    hour       = rng.integers(0, 24, n)

    # bigger damage => bigger claim. The 3 parts ALWAYS add up to the total.
    mult = np.select([severity == "Trivial", severity == "Minor",
                      severity == "Major", severity == "Total Loss"],
                     [0.4, 0.8, 1.4, 1.8], default=1.0)
    vehicle  = np.round(rng.gamma(4, 9000, n) * mult, -1)
    injury   = np.round(rng.gamma(2, 3000, n) * mult, -1)
    property_ = np.round(rng.gamma(2, 3000, n) * mult, -1)
    total = (vehicle + injury + property_).astype(int)

    # ----- decide which claims are fraud (with a learnable pattern + some randomness) -----
    # Higher "score" => more likely fraud. The model's job is to learn this pattern.
    score = (-3.0
             + 1.4 * np.isin(severity, ["Major", "Total Loss"])
             + 1.1 * (police == "NO")
             + 1.0 * (authorities == "None")
             + 0.8 * (total > 60000)
             + 0.7 * ((hour < 5) | (hour > 22))
             + rng.normal(0, 0.4, n))                # noise => not too easy (realistic)
    prob = 1 / (1 + np.exp(-score))                  # turn score into a 0–1 probability
    fraud = np.where(rng.binomial(1, prob) == 1, "Y", "N")

    df = pd.DataFrame({
        "age": rng.integers(19, 70, n),
        "months_as_customer": rng.integers(0, 400, n),
        "policy_annual_premium": np.round(rng.normal(1250, 250, n), 2),
        "incident_type": itype,
        "incident_severity": severity,
        "authorities_contacted": authorities,
        "police_report_available": police,
        "incident_hour": hour,
        "number_of_vehicles": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.25, 0.15, 0.1]),
        "witnesses": rng.choice([0, 1, 2, 3], n),
        "injury_claim": injury.astype(int),
        "property_claim": property_.astype(int),
        "vehicle_claim": vehicle.astype(int),
        "total_claim_amount": total,
        "fraud_reported": fraud,
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv(C.DATA_FILE, index=False)
    print(f"Saved {C.DATA_FILE}  shape={df.shape}  fraud rate={(fraud == 'Y').mean():.1%}")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python generate_data.py
```
You now have `data/claims.csv`. Open it in Excel to see what claims look like.

---

## ✅ Step 5 — Explore your data first (EDA)

**EDA = Exploratory Data Analysis** = *look at your data before modelling*. You're answering: How
much fraud is there? Are amounts skewed? Any missing values? Which fields seem related to fraud? This
tells you what to do next (and catches surprises early). Create **`eda.py`**:

```python
# eda.py — look at the data BEFORE training, and save charts to the plots/ folder.
import os
import matplotlib
matplotlib.use("Agg")          # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import config as C


def main():
    df = pd.read_csv(C.DATA_FILE)
    print("Rows, columns:", df.shape)
    print(df.head(), "\n")

    # 1) How rare is fraud?  (this is WHY accuracy is a bad score here)
    fraud_rate = (df[C.TARGET_FRAUD] == "Y").mean()
    print(f"Fraud rate: {fraud_rate:.1%}")
    print(f"A lazy 'always genuine' model is {1 - fraud_rate:.1%} accurate but catches 0 fraud!\n")

    # 2) Missing values per column (here there are none, but always check)
    print("Missing values per column:\n", df.isna().sum(), "\n")

    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1 — fraud vs genuine counts (shows the imbalance)
    plt.figure(figsize=(5, 4))
    df[C.TARGET_FRAUD].value_counts().plot.bar(color=["#2e7d32", "#c62828"])
    plt.title(f"Fraud class balance (fraud = {fraud_rate:.1%})")
    plt.ylabel("number of claims")
    plt.tight_layout(); plt.savefig("plots/class_balance.png"); plt.close()

    # Chart 2 — claim amount is skewed; log() makes it look normal (why we log-transform)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[C.TARGET_SEVERITY], bins=40, kde=True, ax=ax[0], color="#1565c0")
    ax[0].set_title(f"total_claim_amount (skew = {df[C.TARGET_SEVERITY].skew():.2f})")
    sns.histplot(np.log1p(df[C.TARGET_SEVERITY]), bins=40, kde=True, ax=ax[1], color="#6a1b9a")
    ax[1].set_title("log(total_claim_amount) — much more balanced")
    plt.tight_layout(); plt.savefig("plots/claim_amount.png"); plt.close()

    # Chart 3 — which numbers relate most to fraud?
    fraud_as_number = (df[C.TARGET_FRAUD] == "Y").astype(int)
    corr = (df[C.NUMERIC + C.FRAUD_ONLY_NUMERIC]
            .corrwith(fraud_as_number).abs().sort_values())
    plt.figure(figsize=(6, 4))
    corr.plot.barh(color="#0d9488")
    plt.title("How strongly each number relates to fraud")
    plt.tight_layout(); plt.savefig("plots/fraud_correlation.png"); plt.close()

    print("Saved 3 charts in the plots/ folder — open them and have a look!")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python eda.py
```
Open the **`plots/`** folder and look at the 3 images. **What you should notice (and why it matters):**
- **`class_balance.png`** — fraud is the small bar. Imbalance ⇒ *don't trust accuracy*; we'll score
  with **PR-AUC** instead.
- **`claim_amount.png`** — the left chart is lopsided (skewed), the right (log) is balanced ⇒ we'll
  **log-transform** the amount for the severity model.
- **`fraud_correlation.png`** — the longest bars are the fields most linked to fraud (severity, claim
  size…). Good — the model has real signal to learn from.

> 💡 In the full project, EDA lives in [`src/eda.py`](src/eda.py) and the charts are embedded into the
> shareable `reports/report.html`.

---

## ✅ Step 6 — Train both models with `train.py` (with MLflow tracking)

This is the heart of the project. We also add **MLflow** — think of it as a *logbook*: every time you
train, it records the settings, the scores, and the model itself, so you can compare experiments
later in a nice web dashboard. Read the comments — they explain each ML idea simply.
Create **`train.py`**:

```python
# train.py — trains the fraud model + the severity model, and saves them.
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                             mean_absolute_error, r2_score)
import mlflow                 # the experiment "logbook"
import config as C


def build_preprocessor(stage: str) -> ColumnTransformer:
    """Prepares raw data for a model. The SAME steps are reused at predict time
    (this prevents bugs where training and predicting clean data differently)."""
    numeric = C.NUMERIC + (C.FRAUD_ONLY_NUMERIC if stage == "fraud" else [])
    #   ^ severity model does NOT get the claim-part columns (that would be cheating)

    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),   # fill blanks with the middle value
        ("scale", StandardScaler()),                          # put numbers on the same scale
    ])
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        # turn words into 0/1 columns; ignore words we never saw in training
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # ColumnTransformer applies the right steps to the right columns.
    # Columns we don't list are dropped automatically.
    return ColumnTransformer([
        ("numbers", numeric_steps, numeric),
        ("words", categorical_steps, C.CATEGORICAL),
    ])


def main():
    df = pd.read_csv(C.DATA_FILE)

    # ---------- a quick look at the data (EDA) ----------
    fraud_rate = (df[C.TARGET_FRAUD] == "Y").mean()
    print(f"Fraud rate: {fraud_rate:.1%}")
    print(f"If we just said 'always genuine' we'd be {1 - fraud_rate:.1%} accurate "
          f"-- but catch ZERO fraud. That's why we DON'T use accuracy as our score.")
    print(f"Claim amount is skewed (skew={df[C.TARGET_SEVERITY].skew():.2f}); "
          f"we'll log-transform it for the severity model.\n")

    # ---------- split into train (learn) and test (check) ----------
    # stratify=... keeps the same fraud % in both halves.
    train, test = train_test_split(df, test_size=0.2, random_state=C.RANDOM_STATE,
                                   stratify=df[C.TARGET_FRAUD])

    # =================== MODEL 1: FRAUD ===================
    X_train = train.drop(columns=[C.TARGET_FRAUD])
    y_train = (train[C.TARGET_FRAUD] == "Y").astype(int)   # Y/N -> 1/0
    X_test = test.drop(columns=[C.TARGET_FRAUD])
    y_test = (test[C.TARGET_FRAUD] == "Y").astype(int)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                min_samples_leaf=5, random_state=C.RANDOM_STATE),
    }
    mlflow.set_experiment("fraud_detection")      # MLflow groups all these runs together
    best_model, best_name, best_score, best_proba = None, None, -1, None

    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name):     # each model = one tracked "run"
            # a "pipeline" = clean the data + run the model, as one object
            pipe = Pipeline([("prep", build_preprocessor("fraud")), ("model", clf)])
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]        # probability of fraud (0..1)

            # PR-AUC is the RIGHT score for rare events like fraud (not accuracy!)
            pr_auc = average_precision_score(y_test, proba)
            roc = roc_auc_score(y_test, proba)
            acc = accuracy_score(y_test, (proba >= 0.5).astype(int))

            # ---- write everything to the MLflow logbook ----
            mlflow.log_param("model_type", name)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("roc_auc", roc)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(pipe, artifact_path="model")   # save the model too

            print(f"[{name}] PR-AUC={pr_auc:.3f}  ROC-AUC={roc:.3f}  accuracy={acc:.3f} (<- misleading)")
            if pr_auc > best_score:
                best_model, best_name, best_score, best_proba = pipe, name, pr_auc, proba

    # (optional) register the WINNING model in MLflow's Model Registry — a named,
    # versioned model you could later mark as "production".
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.sklearn.log_model(best_model, artifact_path="model",
                                 registered_model_name="fraud_classifier")

    joblib.dump(best_model, C.FRAUD_MODEL_FILE)   # also save a plain file for the website
    print(f"Best fraud model: {best_name}. Saved + tracked in MLflow.\n")

    # =================== CHOOSE A SMART THRESHOLD ===================
    # The model gives a probability. WE choose the cut-off. 0.5 is rarely best,
    # because a missed fraud costs more than a false alarm. We try every cut-off
    # and pick the one with the LOWEST total cost.
    grid = np.linspace(0.01, 0.99, 99)

    def total_cost(t):
        pred = (best_proba >= t).astype(int)
        missed_fraud = ((y_test == 1) & (pred == 0)).sum()   # false negatives
        false_alarms = ((y_test == 0) & (pred == 1)).sum()   # false positives
        return missed_fraud * C.COST_FALSE_NEGATIVE + false_alarms * C.COST_FALSE_POSITIVE

    costs = [total_cost(t) for t in grid]
    threshold = float(grid[int(np.argmin(costs))])
    print(f"Best threshold = {threshold:.2f}  (cost {total_cost(threshold):,.0f}) "
          f"vs default 0.50 (cost {total_cost(0.5):,.0f})\n")
    with open(C.THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))

    # =================== MODEL 2: SEVERITY (genuine claims only) ===================
    # LEAKAGE GUARD #1: train ONLY on genuine claims. Fraud amounts are fake numbers
    # and would teach the model wrong values.
    g_train = train[train[C.TARGET_FRAUD] == "N"]
    g_test = test[test[C.TARGET_FRAUD] == "N"]
    Xs_train = g_train.drop(columns=[C.TARGET_FRAUD, C.TARGET_SEVERITY])
    ys_train = g_train[C.TARGET_SEVERITY].astype(float)
    Xs_test = g_test.drop(columns=[C.TARGET_FRAUD, C.TARGET_SEVERITY])
    ys_test = g_test[C.TARGET_SEVERITY].astype(float)

    # The target is skewed, so we train on log(amount) and convert back automatically.
    regressor = TransformedTargetRegressor(
        regressor=RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=5,
                                        random_state=C.RANDOM_STATE),
        func=np.log1p, inverse_func=np.expm1)
    # LEAKAGE GUARD #2: build_preprocessor("severity") does NOT include the claim parts.
    severity_model = Pipeline([("prep", build_preprocessor("severity")), ("model", regressor)])
    severity_model.fit(Xs_train, ys_train)

    pred = severity_model.predict(Xs_test)
    mae = mean_absolute_error(ys_test, pred)
    print(f"Severity model: MAE={mae:,.0f}  R2={r2_score(ys_test, pred):.2f}")
    print(f"  In plain words: on average the money estimate is off by about {mae:,.0f}.")
    joblib.dump(severity_model, C.SEVERITY_MODEL_FILE)
    print("Saved severity model. Done!")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
```

Run it:
```bash
python train.py
```
You'll see the scores printed, 3 files appear in `models/`, **and a new `mlruns/` folder appears** —
that's MLflow's logbook of your training runs.

**Now open the MLflow dashboard to compare your experiments:**
```bash
mlflow ui --port 5001
```
Open **http://localhost:5001** in your browser. You'll see each model as a row with its PR-AUC /
ROC-AUC side by side, the settings you used, and the saved model — so you can prove *which* model is
best and why. (We use port **5001** so it doesn't clash with the Flask website on 5000. Press
**Ctrl + C** to stop the dashboard.)

> 💡 You can track the **severity** model the same way — wrap its training in
> `with mlflow.start_run(run_name="severity"):` and `mlflow.log_metric("mae", mae)`. The full project
> does exactly this in [`src/fraud_model.py`](src/fraud_model.py) and [`src/severity_model.py`](src/severity_model.py).

---

## ✅ Step 7 — The "brain": `predict.py`

This loads the saved models and does the **chaining** (fraud first, then money only if genuine).
Create **`predict.py`**:

```python
# predict.py — takes one claim (a dictionary) and returns the answer.
import joblib
import numpy as np
import pandas as pd
import config as C

# load the saved models once (when this file is first imported)
fraud_model = joblib.load(C.FRAUD_MODEL_FILE)
severity_model = joblib.load(C.SEVERITY_MODEL_FILE)
with open(C.THRESHOLD_FILE) as f:
    THRESHOLD = float(f.read())

# every column the models expect to exist on the input
NEEDED = C.NUMERIC + C.FRAUD_ONLY_NUMERIC + C.CATEGORICAL


def predict_claim(claim: dict) -> dict:
    """The two-stage decision."""
    row = pd.DataFrame([claim])
    for col in NEEDED:                 # if a field is missing, add it as blank
        if col not in row.columns:
            row[col] = np.nan

    # ----- Stage 1: fraud check -----
    fraud_probability = float(fraud_model.predict_proba(row)[:, 1][0])

    if fraud_probability >= THRESHOLD:
        # flagged -> STOP. Do NOT predict money.
        return {
            "verdict": "NEEDS_REVIEW",
            "fraud_probability": round(fraud_probability, 3),
            "threshold": round(THRESHOLD, 3),
            "predicted_severity": None,
            "note": "Flagged for manual review — no money estimate given.",
        }

    # ----- Stage 2: money estimate (only for genuine claims) -----
    amount = float(severity_model.predict(row)[0])
    return {
        "verdict": "GENUINE",
        "fraud_probability": round(fraud_probability, 3),
        "threshold": round(THRESHOLD, 3),
        "predicted_severity": round(max(amount, 0)),
    }


# quick self-test: run `python predict.py`
if __name__ == "__main__":
    example = {
        "age": 40, "months_as_customer": 120, "policy_annual_premium": 1200,
        "incident_type": "Multi Vehicle", "incident_severity": "Minor",
        "authorities_contacted": "Police", "police_report_available": "YES",
        "incident_hour": 14, "number_of_vehicles": 2, "witnesses": 2,
        "injury_claim": 6000, "property_claim": 6000, "vehicle_claim": 20000,
        "total_claim_amount": 32000,
    }
    print(predict_claim(example))
```

Test it:
```bash
python predict.py
```

---

## ✅ Step 8 — The website: `app.py` + `templates/index.html`

Create **`app.py`**:
```python
# app.py — the Flask website.
from flask import Flask, request, jsonify, render_template
from predict import predict_claim

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")   # show the web page


@app.post("/predict")
def predict():
    claim = request.get_json(force=True)   # read the form data sent by the page
    return jsonify(predict_claim(claim))   # return the answer as JSON


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

Create a folder **`templates/`** and inside it **`index.html`** (this is the page; CSS and JavaScript
are included right here to keep it to one file):
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Claim Checker</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 30px auto; padding: 0 16px; }
    h1 { font-size: 22px; }
    label { display: block; margin: 8px 0 2px; font-size: 13px; color: #555; }
    input, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
    button { margin-top: 16px; width: 100%; padding: 12px; border: 0; border-radius: 8px;
             background: #2563eb; color: #fff; font-size: 16px; cursor: pointer; }
    #result { margin-top: 20px; padding: 16px; border-radius: 10px; display: none; }
    .genuine { background: #e9f9ef; border: 1px solid #16a34a; }
    .review  { background: #fdecec; border: 1px solid #dc2626; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  </style>
</head>
<body>
  <h1>🛡️ Insurance Claim Checker</h1>
  <p>Fill the claim and click Check.</p>

  <div class="grid">
    <div><label>Age</label><input id="age" type="number" value="40"></div>
    <div><label>Months as customer</label><input id="months_as_customer" type="number" value="120"></div>
    <div><label>Annual premium</label><input id="policy_annual_premium" type="number" value="1200"></div>
    <div><label>Incident hour (0-23)</label><input id="incident_hour" type="number" value="14"></div>
    <div><label>Incident type</label>
      <select id="incident_type"><option>Single Vehicle</option><option selected>Multi Vehicle</option><option>Theft</option><option>Parked Car</option></select></div>
    <div><label>Incident severity</label>
      <select id="incident_severity"><option>Trivial</option><option selected>Minor</option><option>Major</option><option>Total Loss</option></select></div>
    <div><label>Authorities contacted</label>
      <select id="authorities_contacted"><option selected>Police</option><option>Fire</option><option>Ambulance</option><option>None</option></select></div>
    <div><label>Police report</label>
      <select id="police_report_available"><option selected>YES</option><option>NO</option><option>UNKNOWN</option></select></div>
    <div><label>Vehicles involved</label><input id="number_of_vehicles" type="number" value="2"></div>
    <div><label>Witnesses</label><input id="witnesses" type="number" value="2"></div>
    <div><label>Injury claim</label><input id="injury_claim" type="number" value="6000"></div>
    <div><label>Property claim</label><input id="property_claim" type="number" value="6000"></div>
    <div><label>Vehicle claim</label><input id="vehicle_claim" type="number" value="20000"></div>
    <div><label>Total claim amount</label><input id="total_claim_amount" type="number" value="32000"></div>
  </div>

  <button onclick="check()">Check Claim</button>
  <div id="result"></div>

  <script>
    const NUMERIC = ["age","months_as_customer","policy_annual_premium","incident_hour",
                     "number_of_vehicles","witnesses","injury_claim","property_claim",
                     "vehicle_claim","total_claim_amount"];
    const TEXT = ["incident_type","incident_severity","authorities_contacted","police_report_available"];

    async function check() {
      const claim = {};
      NUMERIC.forEach(id => claim[id] = Number(document.getElementById(id).value));
      TEXT.forEach(id => claim[id] = document.getElementById(id).value);

      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(claim),
      });
      const r = await res.json();

      const box = document.getElementById("result");
      box.style.display = "block";
      if (r.verdict === "GENUINE") {
        box.className = "genuine";
        box.innerHTML = `<b>✅ GENUINE</b> (fraud probability ${(r.fraud_probability*100).toFixed(1)}%)
                         <br>Estimated claim amount: <b>₹${r.predicted_severity.toLocaleString()}</b>`;
      } else {
        box.className = "review";
        box.innerHTML = `<b>🚩 NEEDS REVIEW</b> (fraud probability ${(r.fraud_probability*100).toFixed(1)}%)
                         <br>${r.note}`;
      }
    }
  </script>
</body>
</html>
```

Run the website:
```bash
python app.py
```
Open **http://localhost:5000** in your browser. Change the dropdowns to *Major / NO / None* and a big
claim amount → you'll see **🚩 NEEDS REVIEW**. Use mild values → **✅ GENUINE** with a money estimate. 🎉

---

## 🎓 The big ideas you just used (glossary in plain English)

| Term | Simple meaning |
|---|---|
| **Feature** | An input column the model learns from (age, severity, …). |
| **Target** | What you're predicting (fraud Y/N, or the money amount). |
| **Train / test split** | Learn on one part of the data, *check* on a part it never saw — so you know it really learned. |
| **Pipeline** | "Clean the data + run the model" bundled as one object, used the same way in training and predicting. |
| **Imbalanced data** | One class is rare (few frauds). Accuracy lies here — a "do nothing" model looks great but catches nothing. |
| **PR-AUC** | A score that focuses on how well you find the *rare* class (fraud). We pick the model with the best PR-AUC. |
| **Threshold** | The cut-off for "call it fraud". We pick it by *cost*, not the default 0.5. |
| **Data leakage** | Accidentally giving the model an answer key. We avoid it twice: (1) no fraud rows in severity training, (2) no claim-part columns in severity features. |
| **Log-transform** | When numbers are very skewed, train on `log(number)` and convert back — the model learns better. |
| **Chaining** | Use model 1's output to decide whether to run model 2. |

---

## 🧩 Run order (cheat sheet)

```bash
.venv\Scripts\activate          # turn on the environment
pip install -r requirements.txt # once
python generate_data.py         # 1. make data
python eda.py                   # 2. explore data -> charts in plots/
python train.py                 # 3. train + save models (+ MLflow logbook)
mlflow ui --port 5001           # 4. (optional) view experiments -> http://localhost:5001
python app.py                   # 5. start the website -> http://localhost:5000
```

---

## 🚀 Level up to the full project

Once your simple version works, study how this repo turns it into a **production** project. Each idea
maps to a file you can open and read:

| You learned (simple) | The "pro" version in this repo |
|---|---|
| `config.py` constants | [`config.yaml`](config.yaml) + a typed loader ([`src/config.py`](src/config.py)) |
| `print()` for messages | Real logging ([`src/logger.py`](src/logger.py)) |
| `build_preprocessor()` | The same idea, reusable + a custom transformer ([`src/feature_engineering.py`](src/feature_engineering.py)) |
| EDA charts (`eda.py`) | Same idea, charts embedded into the HTML report ([`src/eda.py`](src/eda.py)) |
| LR + RF + **MLflow** tracking | Adds **XGBoost** (a 3rd model) + MLflow **Model Registry** promotion ([`src/fraud_model.py`](src/fraud_model.py)) |
| cost threshold | [`src/threshold_optimizer.py`](src/threshold_optimizer.py) (saves a cost-vs-threshold plot) |
| severity model | [`src/severity_model.py`](src/severity_model.py) |
| (no explanations) | **SHAP** "why" reasons ([`src/explainability.py`](src/explainability.py)) |
| `predict.py` | [`src/inference_pipeline.py`](src/inference_pipeline.py) (one source of truth) |
| `app.py` | [`flask_app.py`](flask_app.py) (+ input validation) |
| run by hand | **Tests** ([`tests/`](tests/)), **Docker**, **GitHub Actions CI**, deploy on **Render** |
| — | A drift monitor + a shareable HTML report |

Open **[`BUILD_GUIDE.md`](BUILD_GUIDE.md)** for the production walkthrough, and run
**[`notebooks/fraud_severity_project.ipynb`](notebooks/fraud_severity_project.ipynb)** to play with the
ML step by step.

---

### 💡 Final tips for a fresher
- **Type the code yourself** instead of only copy-pasting — you'll learn 10× faster.
- If something errors, **read the last line** of the error first — it usually says exactly what's wrong.
- Change one thing at a time and re-run. Small steps beat big leaps.
- When this works, put it on GitHub and write a short README — recruiters love seeing a *working* project.

You've got this. 🚀
