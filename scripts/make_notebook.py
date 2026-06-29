"""Generate notebooks/fraud_severity_project.ipynb programmatically.

We build the notebook with ``nbformat`` (rather than hand-editing JSON) so it is
reproducible and version-controllable. After building, execute it with:

    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=300 notebooks/fraud_severity_project.ipynb

The notebook is an educational, runnable ML walkthrough of the two-stage system.
It reuses the project's shared preprocessing (``build_preprocessor``) so the
Jupyter experiments stay consistent with the production pipeline.
"""

from __future__ import annotations

import nbformat as nbf

from src.config import load_config


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code(src: str):
    return nbf.v4.new_code_cell(src)


def build() -> None:
    cfg = load_config()
    cells = []

    cells.append(_md(
        "# 🛡️ Insurance Claim Fraud Detection + Severity Prediction\n"
        "### A two-stage ML walkthrough (Jupyter · Python · scikit-learn · XGBoost · SHAP)\n\n"
        "This notebook builds and explains the full pipeline interactively:\n\n"
        "1. **Stage 1 — Fraud classification**: is the claim fraudulent?\n"
        "2. **Stage 2 — Severity prediction**: for *genuine* claims only, how much to reserve?\n\n"
        "The two models are **chained**: flagged claims stop at Stage 1 (`NEEDS REVIEW`); "
        "genuine claims flow to Stage 2 for a severity estimate.\n\n"
        "> This notebook reuses the project's shared preprocessing pipeline so experiments match "
        "production. The full modular code lives in `src/`, served via `flask_app.py` / `app.py`."
    ))

    cells.append(_md("## 0 · Setup\nImport the scientific stack and the project's helpers."))
    cells.append(_code(
        "%matplotlib inline\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "# make the project root importable whether this notebook is launched from\n"
        "# the repo root or from notebooks/ (so `import src...` always works)\n"
        "import sys, pathlib\n"
        "ROOT = pathlib.Path.cwd()\n"
        "if not (ROOT / 'src').exists():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.linear_model import LogisticRegression, LinearRegression\n"
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        "from sklearn.compose import TransformedTargetRegressor\n"
        "from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,\n"
        "    recall_score, f1_score, accuracy_score, precision_recall_curve,\n"
        "    mean_squared_error, mean_absolute_error, r2_score)\n"
        "from xgboost import XGBClassifier, XGBRegressor\n"
        "import shap\n\n"
        "# project helpers (single source of truth for transforms)\n"
        "from src.config import load_config\n"
        "from scripts.generate_synthetic_data import generate\n"
        "from src.feature_engineering import build_preprocessor, get_feature_names\n\n"
        "cfg = load_config(); RANDOM_STATE = cfg.get('project.random_state')\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('setup OK')"
    ))

    cells.append(_md(
        "## 1 · Get the data\nWe generate a synthetic dataset that matches the Kaggle "
        "*Auto Insurance Claims* schema exactly (40 columns, incl. `fraud_reported` and "
        "`total_claim_amount`). To use the **real** dataset instead, download it from Kaggle, "
        "save it as `data/raw/insurance_claims.csv`, and load that file here."
    ))
    cells.append(_code(
        "df = generate(n_rows=2000, seed=RANDOM_STATE)\n"
        "print('shape:', df.shape)\n"
        "df[['age','incident_severity','police_report_available','total_claim_amount',\n"
        "    'injury_claim','property_claim','vehicle_claim','fraud_reported']].head()"
    ))
    cells.append(_md(
        "**Note the arithmetic identity** — the three claim components sum to the target. "
        "This is why they are *leakage* for the severity model (more in Step 5)."
    ))
    cells.append(_code(
        "ok = (df.injury_claim + df.property_claim + df.vehicle_claim == df.total_claim_amount).all()\n"
        "print('total_claim_amount == injury + property + vehicle :', bool(ok))"
    ))

    cells.append(_md(
        "## 2 · EDA — why accuracy is the WRONG metric\n"
        "The fraud label is **imbalanced**. A model that always predicts *genuine* would score "
        "high accuracy while catching **zero** fraud. That's why we select on **PR-AUC** and report "
        "precision/recall — not accuracy."
    ))
    cells.append(_code(
        "fraud_rate = (df.fraud_reported == 'Y').mean()\n"
        "print(f'Fraud rate: {fraud_rate:.1%}')\n"
        "print(f'\"Always genuine\" accuracy: {1-fraud_rate:.1%}  <-- catches 0 fraud!')\n\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df.fraud_reported.value_counts().plot.bar(ax=ax[0], color=['#2e7d32','#c62828'])\n"
        "ax[0].set_title(f'Class balance (fraud={fraud_rate:.1%})'); ax[0].set_ylabel('claims')\n"
        "sns.histplot(df.total_claim_amount, bins=40, kde=True, ax=ax[1], color='#1565c0')\n"
        "ax[1].set_title('total_claim_amount is right-skewed -> log-transform for Stage 2')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(_md(
        "## 3 · Train/test split (stratified)\n"
        "Random **stratified** split (not time-based): claims are scored independently as they "
        "arrive — it's not a forecasting problem — and stratifying keeps the fraud rate identical "
        "in both folds."
    ))
    cells.append(_code(
        "train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE,\n"
        "                                     stratify=df.fraud_reported)\n"
        "print('train:', train_df.shape, '| test:', test_df.shape)\n"
        "print('fraud rate  train=%.1f%%  test=%.1f%%' % (\n"
        "    (train_df.fraud_reported=='Y').mean()*100, (test_df.fraud_reported=='Y').mean()*100))"
    ))

    cells.append(_md(
        "## 4 · Stage 1 — Fraud classification\n"
        "Each model is a **full `Pipeline`** (shared preprocessor + estimator), fit on train only. "
        "Using `build_preprocessor('fraud')` guarantees the notebook uses the *same* transforms as "
        "production. We compare a Logistic Regression baseline, Random Forest and XGBoost."
    ))
    cells.append(_code(
        "ycol = 'fraud_reported'\n"
        "X_train, y_train = train_df.drop(columns=[ycol]), (train_df[ycol]=='Y').astype(int)\n"
        "X_test,  y_test  = test_df.drop(columns=[ycol]),  (test_df[ycol]=='Y').astype(int)\n\n"
        "candidates = {\n"
        "    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),\n"
        "    'random_forest': RandomForestClassifier(n_estimators=300, max_depth=12,\n"
        "                        min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),\n"
        "    'xgboost': XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,\n"
        "                        subsample=0.9, colsample_bytree=0.9, eval_metric='aucpr',\n"
        "                        random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'),\n"
        "}\n"
        "rows, fitted = [], {}\n"
        "for name, clf in candidates.items():\n"
        "    pipe = Pipeline([('preprocess', build_preprocessor('fraud', cfg)), ('model', clf)])\n"
        "    pipe.fit(X_train, y_train)\n"
        "    proba = pipe.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)\n"
        "    fitted[name] = (pipe, proba)\n"
        "    rows.append({'model': name,\n"
        "        'PR_AUC': average_precision_score(y_test, proba),\n"
        "        'ROC_AUC': roc_auc_score(y_test, proba),\n"
        "        'precision': precision_score(y_test, pred, zero_division=0),\n"
        "        'recall': recall_score(y_test, pred, zero_division=0),\n"
        "        'F1': f1_score(y_test, pred, zero_division=0),\n"
        "        'accuracy': accuracy_score(y_test, pred)})\n"
        "results = pd.DataFrame(rows).set_index('model').round(3)\n"
        "results"
    ))
    cells.append(_md(
        "Notice how **accuracy barely separates the models** while PR-AUC does — exactly why "
        "accuracy is misleading here. We pick the best model by PR-AUC."
    ))
    cells.append(_code(
        "best_name = results['PR_AUC'].idxmax()\n"
        "best_pipe, best_proba = fitted[best_name]\n"
        "print('Best fraud model:', best_name)\n\n"
        "prec, rec, _ = precision_recall_curve(y_test, best_proba)\n"
        "plt.figure(figsize=(6,5))\n"
        "plt.plot(rec, prec, color='#c62828', lw=2,\n"
        "         label=f'{best_name} (AP={average_precision_score(y_test, best_proba):.3f})')\n"
        "plt.axhline(y_test.mean(), ls='--', color='gray', label=f'random ({y_test.mean():.2f})')\n"
        "plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall curve')\n"
        "plt.legend(); plt.show()"
    ))

    cells.append(_md(
        "## 5 · Cost-sensitive threshold (move off 0.5)\n"
        "0.5 implicitly assumes a false positive and a false negative cost the same. They don't: a "
        "**missed fraud** (FN) costs the payout; a **false alarm** (FP) costs an investigation. We "
        "sweep thresholds and pick the one that minimises total expected cost."
    ))
    cells.append(_code(
        "COST_FN = cfg.get('threshold.cost_false_negative')   # missed fraud (e.g. 30000)\n"
        "COST_FP = cfg.get('threshold.cost_false_positive')   # false alarm (e.g. 5000)\n"
        "grid = np.linspace(0.01, 0.99, 99)\n"
        "def total_cost(t):\n"
        "    pred = (best_proba >= t).astype(int)\n"
        "    fn = ((y_test==1) & (pred==0)).sum(); fp = ((y_test==0) & (pred==1)).sum()\n"
        "    return fn*COST_FN + fp*COST_FP\n"
        "costs = np.array([total_cost(t) for t in grid])\n"
        "best_t = grid[costs.argmin()]\n"
        "print(f'Cost-optimal threshold = {best_t:.2f}  (vs default 0.50)')\n"
        "print(f'cost @ {best_t:.2f} = {costs.min():,.0f}   |   cost @ 0.50 = {total_cost(0.5):,.0f}')\n\n"
        "plt.figure(figsize=(7,4))\n"
        "plt.plot(grid, costs, color='#1565c0', lw=2)\n"
        "plt.axvline(best_t, color='#2e7d32', ls='--', label=f'chosen={best_t:.2f}')\n"
        "plt.axvline(0.5, color='gray', ls=':', label='default=0.50')\n"
        "plt.xlabel('threshold'); plt.ylabel('total expected cost'); plt.legend()\n"
        "plt.title('Cost vs threshold — minimum is left of 0.5'); plt.show()"
    ))

    cells.append(_md(
        "## 6 · Data leakage + Stage 2 severity\n"
        "**Two leakage guards:**\n\n"
        "1. **Exclude fraud claims from severity training** — fraudulent amounts are fabricated and "
        "would poison the reserving model. We also only predict severity for genuine claims in "
        "production, so train/serve populations must match.\n"
        "2. **Drop the claim-component columns** — they sum to the target. `build_preprocessor('severity')` "
        "already excludes them.\n\n"
        "We train on the **log-transformed** target and invert predictions back to rupees."
    ))
    cells.append(_code(
        "scol = 'total_claim_amount'\n"
        "# leakage guard #1: genuine claims only\n"
        "g_train = train_df[train_df.fraud_reported != 'Y']\n"
        "g_test  = test_df[test_df.fraud_reported != 'Y']\n"
        "print(f'Severity training rows: {len(g_train)} (dropped {len(train_df)-len(g_train)} fraud claims)')\n\n"
        "Xs_tr, ys_tr = g_train.drop(columns=[scol, 'fraud_reported']), g_train[scol].astype(float)\n"
        "Xs_te, ys_te = g_test.drop(columns=[scol, 'fraud_reported']),  g_test[scol].astype(float)\n\n"
        "reg = TransformedTargetRegressor(  # log1p target, expm1 on predict\n"
        "    regressor=RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=5,\n"
        "                                    random_state=RANDOM_STATE, n_jobs=-1),\n"
        "    func=np.log1p, inverse_func=np.expm1)\n"
        "sev_pipe = Pipeline([('preprocess', build_preprocessor('severity', cfg)), ('model', reg)])\n"
        "sev_pipe.fit(Xs_tr, ys_tr)\n"
        "sp = sev_pipe.predict(Xs_te)\n"
        "print('RMSE = %.0f | MAE = %.0f | R2 = %.3f' % (\n"
        "    np.sqrt(mean_squared_error(ys_te, sp)), mean_absolute_error(ys_te, sp), r2_score(ys_te, sp)))"
    ))
    cells.append(_code(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax[0].scatter(ys_te, sp, alpha=.4, s=14, color='#1565c0')\n"
        "lo, hi = ys_te.min(), ys_te.max(); ax[0].plot([lo,hi],[lo,hi],'r--')\n"
        "ax[0].set_xlabel('actual'); ax[0].set_ylabel('predicted'); ax[0].set_title('Severity: predicted vs actual')\n"
        "ax[1].hist(sp - ys_te, bins=40, color='#6a1b9a'); ax[1].axvline(0, color='red', ls='--')\n"
        "ax[1].set_title('Error distribution'); plt.tight_layout(); plt.show()"
    ))

    cells.append(_md(
        "## 7 · Explainability with SHAP\n"
        "Global feature importance for the fraud model — turning the black box into reasons a claims "
        "handler can act on. In the app these become plain-English factors per claim."
    ))
    cells.append(_code(
        "pre = best_pipe.named_steps['preprocess']\n"
        "model = best_pipe.named_steps['model']\n"
        "X_te_trans = pre.transform(X_test)\n"
        "feat_names = get_feature_names(pre)\n"
        "explainer = shap.TreeExplainer(model)\n"
        "sv = explainer(X_te_trans[:200], check_additivity=False)\n"
        "vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values\n"
        "shap.summary_plot(vals, X_te_trans[:200], feature_names=feat_names, max_display=12, show=True)"
    ))

    cells.append(_md(
        "## 8 · Chained inference — putting it together\n"
        "A claim comes in → fraud check → if genuine, predict severity; if flagged, return "
        "`NEEDS REVIEW` with **no** severity. This is exactly what `src/inference_pipeline.py` does "
        "in production and what both the Flask and FastAPI apps serve."
    ))
    cells.append(_code(
        "def screen_claim(row: pd.DataFrame):\n"
        "    p = float(best_pipe.predict_proba(row)[:, 1][0])\n"
        "    if p >= best_t:\n"
        "        return {'verdict': 'NEEDS_REVIEW', 'fraud_probability': round(p,3),\n"
        "                'severity': None, 'note': 'flagged for manual review'}\n"
        "    amount = float(sev_pipe.predict(row)[0])\n"
        "    return {'verdict': 'GENUINE', 'fraud_probability': round(p,3),\n"
        "            'predicted_severity': round(max(amount,0))}\n\n"
        "# a clearly-genuine claim vs a clearly-suspicious one\n"
        "genuine = test_df[test_df.fraud_reported=='N'].drop(columns=['fraud_reported']).iloc[[0]]\n"
        "suspicious = test_df[test_df.fraud_reported=='Y'].drop(columns=['fraud_reported']).iloc[[0]]\n"
        "print('Genuine    ->', screen_claim(genuine))\n"
        "print('Suspicious ->', screen_claim(suspicious))"
    ))

    cells.append(_md(
        "## ✅ Summary\n"
        "We built a chained two-stage system: a **fraud classifier** (selected by PR-AUC, with a "
        "**cost-optimised threshold**) and a **severity regressor** (trained on genuine claims only, "
        "with two leakage guards and a log-transformed target), made explainable with **SHAP**.\n\n"
        "**Next:** the production code is in `src/` (config-driven, logged, tested), served by "
        "`flask_app.py` (Flask) or `app.py` (FastAPI), with MLflow tracking, drift monitoring, Docker "
        "and CI. See `BUILD_GUIDE.md` to build it all from scratch, and `reports/report.html` for the "
        "full write-up."
    ))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = cfg.root / "notebooks" / "fraud_severity_project.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Notebook written: {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    build()
