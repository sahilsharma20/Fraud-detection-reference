"""Build the self-contained, shareable HTML report (Deliverable C).

Produces a single ``reports/report.html`` with EVERY image embedded as base64, so
it opens in any browser and prints to PDF with no external files. It reads the
artifacts produced by `make train` (metrics JSONs, plots) plus the UI
screenshots, computes the business-impact figures from the held-out predictions,
and renders an inline-SVG architecture diagram.

Run after training:  python -m scripts.build_report   (or `make report`)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────── helpers ───────────────────────────
def _b64(path: Path) -> str:
    """Return an <img> data-URI for a PNG, or '' if the file is missing."""
    if not path.exists():
        log.warning("Report asset missing: %s", path)
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _img(path: Path, alt: str, caption: str = "") -> str:
    uri = _b64(path)
    if not uri:
        return f'<div class="missing">[missing: {alt}]</div>'
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return f'<figure><img src="{uri}" alt="{alt}"/>{cap}</figure>'


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt_money(sym: str, n: float) -> str:
    return f"{sym}{round(n):,}"


# ─────────────────────────── architecture diagram (inline SVG) ───────────────────────────
def _architecture_svg(threshold: float, sym: str) -> str:
    thr = f"{threshold:.2f}"
    return f"""
<svg viewBox="0 0 960 360" xmlns="http://www.w3.org/2000/svg" class="arch">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#475569"/>
    </marker>
    <style>
      .box {{ fill:#fff; stroke:#cbd5e1; stroke-width:1.5; rx:12; }}
      .lbl {{ font:600 14px Inter,sans-serif; fill:#0f172a; }}
      .sub {{ font:400 11px Inter,sans-serif; fill:#64748b; }}
      .edge {{ stroke:#475569; stroke-width:1.6; fill:none; marker-end:url(#arr); }}
      .ok {{ fill:#16a34a; }} .bad {{ fill:#dc2626; }} .stage {{ fill:#eef2ff; stroke:#6366f1; }}
    </style>
  </defs>

  <rect class="box" x="20" y="150" width="120" height="56" rx="12"/>
  <text class="lbl" x="80" y="174" text-anchor="middle">Claim</text>
  <text class="sub" x="80" y="192" text-anchor="middle">web form / API</text>

  <rect class="box stage" x="190" y="140" width="160" height="76" rx="12"/>
  <text class="lbl" x="270" y="168" text-anchor="middle">Stage 1</text>
  <text class="sub" x="270" y="186" text-anchor="middle">Fraud classifier (RF)</text>
  <text class="sub" x="270" y="202" text-anchor="middle">→ P(fraud)</text>

  <polygon class="box" points="430,178 480,150 530,178 480,206" fill="#fff"/>
  <text class="sub" x="480" y="176" text-anchor="middle">P ≥ {thr}?</text>
  <text class="sub" x="480" y="190" text-anchor="middle">threshold</text>

  <rect class="box" x="600" y="40" width="200" height="76" rx="12"/>
  <text class="lbl bad" x="700" y="68" text-anchor="middle">🚩 NEEDS REVIEW</text>
  <text class="sub" x="700" y="86" text-anchor="middle">routed to investigator</text>
  <text class="sub" x="700" y="102" text-anchor="middle">NO severity predicted</text>

  <rect class="box stage" x="600" y="150" width="160" height="76" rx="12"/>
  <text class="lbl" x="680" y="178" text-anchor="middle">Stage 2</text>
  <text class="sub" x="680" y="196" text-anchor="middle">Severity reg. (RF)</text>
  <text class="sub" x="680" y="212" text-anchor="middle">genuine claims only</text>

  <rect class="box" x="600" y="262" width="240" height="74" rx="12"/>
  <text class="lbl ok" x="720" y="290" text-anchor="middle">✅ GENUINE</text>
  <text class="sub" x="720" y="308" text-anchor="middle">{sym} predicted severity ± MAE band</text>
  <text class="sub" x="720" y="324" text-anchor="middle">+ SHAP top factors</text>

  <path class="edge" d="M140,178 L186,178"/>
  <path class="edge" d="M350,178 L426,178"/>
  <path class="edge" d="M505,162 C560,120 560,90 596,82"/>
  <text class="sub bad" x="545" y="120">YES (fraud)</text>
  <path class="edge" d="M505,194 C560,210 560,200 596,190"/>
  <text class="sub ok" x="545" y="232">NO (genuine)</text>
  <path class="edge" d="M680,226 L680,258"/>
</svg>
"""


# ─────────────────────────── report build ───────────────────────────
def build_report() -> Path:
    """Assemble and write reports/report.html. Returns its path."""
    cfg = load_config()
    plots = cfg.path("paths.plots_dir")
    metrics = cfg.path("paths.metrics_dir")
    shots = cfg.root / "reports" / "screenshots"
    sym = cfg.get("business.currency_symbol")

    summary = _load_json(metrics / "training_summary.json")
    fraud_m = _load_json(metrics / "fraud_metrics.json")
    sev_m = _load_json(metrics / "severity_metrics.json")
    thr_m = _load_json(metrics / "threshold.json")
    eda = _load_json(metrics / "eda_summary.json")
    inf = _load_json(cfg.path("paths.inference_config"))

    threshold = float(inf.get("fraud_threshold", 0.5))
    best_fraud = fraud_m.get("best_model", "random_forest")
    best_sev = sev_m.get("best_model", "random_forest")

    # ── operating-point metrics (precision/recall AT the chosen threshold) ──
    preds = pd.read_csv(metrics / "fraud_test_predictions.csv")
    y, p = preds["y_true"].to_numpy(), preds["y_proba"].to_numpy()
    pred = (p >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    recall_op = tp / max(tp + fn, 1)
    precision_op = tp / max(tp + fp, 1)
    fp_rate = fp / max(fp + tn, 1)

    # ── severity: naive baseline (predict the mean) vs the model, on genuine test ──
    test_df = pd.read_csv(cfg.path("paths.test_data"))
    genuine_test = test_df[test_df[cfg.get("data.target_fraud")] != cfg.get("data.fraud_positive_label")]
    sev_true = genuine_test[cfg.get("data.target_severity")].to_numpy(dtype=float)
    naive_mae = float(np.mean(np.abs(sev_true - sev_true.mean())))
    model_mae = float(sev_m["per_model"][best_sev]["mae"])
    mae_improve = (naive_mae - model_mae) / naive_mae * 100

    # ── business impact (every number traces to config + the operating point) ──
    vol = int(cfg.get("business.annual_claim_volume"))
    fraud_rate = float(cfg.get("business.fraud_rate"))
    payout = float(cfg.get("business.avg_fraud_payout"))
    invest = float(cfg.get("threshold.cost_false_positive"))
    annual_fraud = vol * fraud_rate
    exposure = annual_fraud * payout
    fraud_prevented = exposure * recall_op
    genuine_vol = vol * (1 - fraud_rate)
    annual_false_flags = genuine_vol * fp_rate
    annual_caught = annual_fraud * recall_op
    review_cost = (annual_false_flags + annual_caught) * invest
    net_benefit = fraud_prevented - review_cost

    # ── tables ──
    fraud_rows = "".join(
        f"<tr class='{'win' if name == best_fraud else ''}'><td>{name}</td>"
        f"<td>{m['pr_auc']:.3f}</td><td>{m['roc_auc']:.3f}</td><td>{m['precision']:.3f}</td>"
        f"<td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td><td>{m['accuracy']:.3f}</td></tr>"
        for name, m in summary.get("fraud", {}).get("metrics", fraud_m.get("per_model", {})).items()
    )
    sev_rows = "".join(
        f"<tr class='{'win' if name == best_sev else ''}'><td>{name}</td>"
        f"<td>{m['rmse']:,.0f}</td><td>{m['mae']:,.0f}</td><td>{m['r2']:.3f}</td>"
        f"<td>{m['mean_bias']:+,.0f}</td></tr>"
        for name, m in sev_m.get("per_model", {}).items()
    )

    css = _REPORT_CSS
    arch = _architecture_svg(threshold, sym)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ClaimGuard — Technical Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>{css}</style></head><body>
<div class="report">

  <header class="cover">
    <span class="badge">Two-stage ML system · Technical Report</span>
    <h1>Insurance Claim Fraud Detection<br/>+ Severity Prediction</h1>
    <p class="lede">A chained pipeline that flags fraudulent auto-insurance claims, and — for genuine
    claims only — estimates the claim severity (reserve amount), with cost-optimised decisions and
    SHAP explanations for every verdict.</p>
    <div class="kpis">
      <div class="kpi"><b>{summary.get('fraud', {}).get('metrics', {}).get(best_fraud, {}).get('pr_auc', 0):.3f}</b><span>Fraud PR-AUC</span></div>
      <div class="kpi"><b>{recall_op*100:.0f}%</b><span>Fraud recall @ operating point</span></div>
      <div class="kpi"><b>{_fmt_money(sym, model_mae)}</b><span>Severity MAE</span></div>
      <div class="kpi"><b>{threshold:.2f}</b><span>Cost-optimal threshold</span></div>
    </div>
  </header>

  <section>
    <h2>1 · What this system is — and why</h2>
    <p>An insurer receives a claim and must answer two questions: <em>is it fraudulent?</em> and, if
    not, <em>how much should we reserve for it?</em> We answer them in two stages, chained so the
    second only runs when the first says "genuine":</p>
    <ul>
      <li><b>Stage 1 — Fraud classification.</b> A model scores the probability the claim is fraud.
      If the probability clears a <em>cost-optimised</em> threshold, the claim is flagged
      <span class="tag bad">NEEDS REVIEW</span> and routed to a human investigator — it is not
      auto-denied, and no severity number is produced.</li>
      <li><b>Stage 2 — Severity regression.</b> For claims judged <span class="tag ok">GENUINE</span>,
      a second model predicts <code>total_claim_amount</code> so the insurer can set an accurate
      reserve, shown with a plain-English error band.</li>
    </ul>
    {arch}
    <p class="muted">Both stages run behind one API; the web app and the API share a single inference
    implementation (no duplicated logic).</p>
  </section>

  <section>
    <h2>2 · The data & why accuracy is the wrong metric</h2>
    <p>The dataset is the Kaggle "Auto Insurance Claims" schema (40 fields incl. <code>fraud_reported</code>
    and <code>total_claim_amount</code>). The fraud label is imbalanced — only
    <b>{eda.get('fraud_rate', 0)*100:.1f}%</b> of claims are fraud. A model that predicts "never fraud"
    therefore scores <b>{eda.get('majority_class_accuracy', 0)*100:.1f}%</b> accuracy while catching
    <b>zero</b> fraud. That is why we select on <b>PR-AUC</b> (precision–recall over the fraud class),
    and report precision/recall/F1 — never accuracy alone.</p>
    <div class="grid2">
      {_img(plots / "class_balance.png", "class balance", "Fig 1. Class imbalance — the accuracy trap.")}
      {_img(plots / "target_distribution.png", "target distribution", "Fig 2. Severity target is right-skewed → we train on log1p(y).")}
    </div>
  </section>

  <section>
    <h2>3 · Stage 1 — Fraud model selection</h2>
    <p>Three models of increasing capacity were trained as full sklearn pipelines and tracked in
    MLflow; the best by PR-AUC was registered in the MLflow Model Registry. Note how <b>accuracy
    barely separates the models</b> while PR-AUC does — exactly the point above.</p>
    <table><thead><tr><th>model</th><th>PR-AUC</th><th>ROC-AUC</th><th>precision</th><th>recall</th><th>F1</th><th>accuracy</th></tr></thead>
    <tbody>{fraud_rows}</tbody></table>
    <p class="muted">Winner: <b>{best_fraud}</b> (highlighted). Imbalance is handled by the decision
    threshold (next section), not by class reweighting — so the probabilities stay calibrated for the
    cost optimisation.</p>
    {_img(plots / "pr_curve.png", "PR curve", "Fig 3. Precision–Recall curve vs the random baseline (= prevalence).")}
  </section>

  <section>
    <h2>4 · Cost-sensitive threshold (moving off 0.5)</h2>
    <p>The default 0.5 cut-off implicitly assumes a false positive and a false negative cost the same.
    They don't. We use a cost matrix:</p>
    <table class="cost"><thead><tr><th></th><th>Predicted genuine</th><th>Predicted fraud</th></tr></thead>
      <tbody>
        <tr><th>Actually genuine</th><td class="ok">correct (₹0)</td><td>False Positive — investigation + churn (<b>{_fmt_money(sym, invest)}</b>)</td></tr>
        <tr><th>Actually fraud</th><td class="bad">False Negative — fraud paid out (<b>{_fmt_money(sym, payout)}</b>)</td><td class="ok">correct (₹0)</td></tr>
      </tbody></table>
    <p>A missed fraud costs <b>{payout/invest:.0f}×</b> a false alarm, so sweeping the threshold to
    minimise total expected cost lands at <b>{threshold:.2f}</b>, well below 0.5. On the held-out set
    this cuts cost from {_fmt_money(sym, thr_m.get('default_0.5_cost', 0))} (at 0.5) to
    {_fmt_money(sym, thr_m.get('chosen_cost', 0))} — a saving of
    <b>{_fmt_money(sym, thr_m.get('cost_saving_vs_default', 0))}</b>. The business consciously accepts
    more false alarms (sent to human review) because each missed fraud is far costlier.</p>
    {_img(plots / "cost_vs_threshold.png", "cost vs threshold", "Fig 4. Total expected cost vs threshold — minimum well left of 0.5.")}
    <p class="muted">At the chosen threshold (on test): recall <b>{recall_op*100:.0f}%</b>,
    precision <b>{precision_op*100:.0f}%</b>, false-positive rate <b>{fp_rate*100:.0f}%</b>.</p>
  </section>

  <section>
    <h2>5 · Data-leakage decisions (two of them)</h2>
    <div class="callout">
      <b>Leakage guard #1 — exclude fraud claims from severity training.</b>
      A fraudulent claim's amount is <em>fabricated</em> — it's what the fraudster tried to extract, not
      the real cost of an incident. Training the reserving model on those amounts would teach it that a
      minor incident is worth a fortune, poisoning every genuine prediction. So all
      <code>fraud_reported == 'Y'</code> rows are dropped before severity training. This also matches
      production: severity is only ever predicted for claims judged genuine, so we must train on genuine
      claims only (train/serve populations must match).
    </div>
    <div class="callout">
      <b>Leakage guard #2 — drop the claim-component columns.</b>
      <code>total_claim_amount = injury_claim + property_claim + vehicle_claim</code> exactly. Feeding
      those three components to the severity model lets it reconstruct the target (R²≈1.0) and learn
      nothing generalisable. They're excluded from the severity feature set (but kept for the fraud
      model, where they're legitimate signal).
    </div>
  </section>

  <section>
    <h2>6 · Stage 2 — Severity model</h2>
    <p>Trained on genuine claims only, on a log-transformed target (predictions inverted back to rupees).
    Errors are reported in rupee space and translated into a business statement.</p>
    <table><thead><tr><th>model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>mean bias</th></tr></thead>
    <tbody>{sev_rows}</tbody></table>
    <p class="biz">{sev_m.get('business_statement', '')}</p>
    {_img(plots / "severity_errors.png", "severity errors", "Fig 5. Predicted-vs-actual and error distribution for the severity model.")}
  </section>

  <section>
    <h2>7 · Explainability (SHAP)</h2>
    <p>Every prediction surfaces its top drivers in plain English (e.g. <em>"Incident severity is
    'Major Damage' — increases fraud risk"</em>). Global SHAP importance for both models:</p>
    <div class="grid2">
      {_img(plots / "shap_summary_fraud.png", "shap fraud", "Fig 6. Global SHAP — fraud model.")}
      {_img(plots / "shap_summary_severity.png", "shap severity", "Fig 7. Global SHAP — severity model.")}
    </div>
  </section>

  <section>
    <h2>8 · Business impact</h2>
    <p>All assumptions are stated; every figure is computed from them and the model's measured
    operating point — nothing is invented.</p>
    <table class="assume"><thead><tr><th>Assumption</th><th>Value</th></tr></thead><tbody>
      <tr><td>Annual claim volume</td><td>{vol:,}</td></tr>
      <tr><td>Production fraud rate (assumed lower than the dataset's training rate)</td><td>{fraud_rate*100:.0f}%</td></tr>
      <tr><td>Average fraudulent payout (FN cost)</td><td>{_fmt_money(sym, payout)}</td></tr>
      <tr><td>Investigation cost per flagged claim (FP cost)</td><td>{_fmt_money(sym, invest)}</td></tr>
      <tr><td>Model fraud recall @ operating threshold</td><td>{recall_op*100:.0f}%</td></tr>
      <tr><td>Model false-positive rate @ operating threshold</td><td>{fp_rate*100:.0f}%</td></tr>
    </tbody></table>
    <div class="impact">
      <div class="impact-card"><span>Annual fraud exposure</span><b>{_fmt_money(sym, exposure)}</b><small>{annual_fraud:,.0f} fraud × {_fmt_money(sym, payout)}</small></div>
      <div class="impact-card good"><span>Fraud loss prevented / yr</span><b>{_fmt_money(sym, fraud_prevented)}</b><small>{recall_op*100:.0f}% of exposure caught</small></div>
      <div class="impact-card warn"><span>Review cost / yr</span><b>{_fmt_money(sym, review_cost)}</b><small>{annual_false_flags+annual_caught:,.0f} claims reviewed × {_fmt_money(sym, invest)}</small></div>
      <div class="impact-card net"><span>Net benefit / yr</span><b>{_fmt_money(sym, net_benefit)}</b><small>prevented − review cost</small></div>
    </div>
    <p><b>Reserving accuracy.</b> Predicting the mean reserve gives an MAE of {_fmt_money(sym, naive_mae)};
    the model achieves {_fmt_money(sym, model_mae)} — a <b>{mae_improve:.0f}%</b> reduction in average
    reserving error per genuine claim, tightening capital reserves across {genuine_vol:,.0f} genuine
    claims a year.</p>
  </section>

  <section>
    <h2>9 · The running web app</h2>
    <p>A non-technical user fills a form (or loads a sample) and clicks <b>Check Claim</b>. The result
    card shows the verdict, fraud probability vs the threshold, the severity estimate with its error
    band (genuine only), and the top human-readable factors.</p>
    <div class="grid2">
      {_img(shots / "ui_genuine.png", "genuine UI", "Fig 8. Genuine claim → severity estimate + factors.")}
      {_img(shots / "ui_fraud.png", "fraud UI", "Fig 9. Suspicious claim → flagged, no severity, factors shown.")}
    </div>
  </section>

  <section>
    <h2>10 · Monitoring & retraining</h2>
    <p>A drift monitor computes the Population Stability Index (PSI) of incoming claims vs the training
    distribution per feature (PSI &gt; 0.25 ⇒ significant drift). Retraining triggers: a monthly
    scheduled refresh on matured labels, a drift-based trigger from this monitor, and a
    performance-based trigger once labels mature. New models are promoted in the registry only if they
    beat the current champion on a frozen holdout (champion/challenger), then canaried before full
    rollout.</p>
  </section>

  <section>
    <h2>11 · Limitations</h2>
    <ul>
      <li><b>Synthetic data.</b> This demo trains on a synthetic dataset matching the Kaggle schema
      (the real file is small and redistribution-restricted). Absolute metrics are illustrative; the
      <em>engineering</em> — pipeline parity, leakage handling, cost optimisation, explainability,
      monitoring — is the real deliverable. Drop the real CSV in and rerun to retrain.</li>
      <li><b>Dataset size.</b> Public auto-fraud datasets are ~1–2k rows; production models would train
      on far more and likely include richer behavioural/network features.</li>
      <li><b>Label maturity & drift.</b> Fraud labels mature months after the incident; the live model
      will drift as tactics evolve — hence the monitoring above.</li>
      <li><b>Threshold is a business choice.</b> The cost matrix values are assumptions; they should be
      set with the claims/finance teams and revisited as costs change.</li>
    </ul>
  </section>

  <footer class="foot">
    <p>Generated by <code>scripts/build_report.py</code> · models tracked in MLflow · explanations via SHAP.
    Self-contained — print to PDF to share.</p>
  </footer>
</div></body></html>"""

    out = cfg.path("paths.report_html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Report written -> %s (%.0f KB)", out, out.stat().st_size / 1024)
    return out


_REPORT_CSS = """
* { box-sizing: border-box; }
body { margin:0; background:#eef2f8; font-family:Inter,system-ui,sans-serif; color:#1e293b; line-height:1.6; }
.report { max-width:900px; margin:0 auto; background:#fff; box-shadow:0 8px 40px rgba(2,6,23,.10); }
.cover { background:radial-gradient(900px 300px at 20% -30%, #2548a0 0%, #0f1729 70%); color:#fff; padding:54px 48px; }
.badge { display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); padding:6px 12px; border-radius:999px; font-size:12px; letter-spacing:.4px; }
.cover h1 { font-size:34px; line-height:1.15; margin:18px 0 10px; letter-spacing:-.6px; }
.lede { color:#c7d2e8; font-size:15px; max-width:680px; }
.kpis { display:flex; gap:16px; margin-top:26px; flex-wrap:wrap; }
.kpi { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); border-radius:14px; padding:14px 18px; min-width:130px; }
.kpi b { display:block; font-size:24px; }
.kpi span { font-size:11.5px; color:#aebbd6; }
section { padding:30px 48px; border-top:1px solid #eef1f6; }
h2 { font-size:20px; letter-spacing:-.3px; margin:0 0 12px; }
p, li { font-size:14.5px; }
code { background:#f1f5f9; padding:1px 5px; border-radius:5px; font-size:13px; }
.muted { color:#64748b; font-size:13px; }
.biz { background:#ecfdf5; border-left:4px solid #16a34a; padding:10px 14px; border-radius:8px; font-weight:600; }
.tag { padding:1px 8px; border-radius:6px; font-size:12px; font-weight:700; }
.tag.ok { background:#e9f9ef; color:#16a34a; } .tag.bad { background:#fdecec; color:#dc2626; }
table { width:100%; border-collapse:collapse; margin:14px 0; font-size:13.5px; }
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid #eef1f6; }
thead th { background:#f8fafc; font-size:12px; text-transform:uppercase; letter-spacing:.4px; color:#64748b; }
tr.win { background:#ecfdf5; font-weight:600; }
table.cost td, table.cost th { border:1px solid #e2e8f0; }
td.ok { color:#16a34a; } td.bad { color:#dc2626; }
figure { margin:16px 0; text-align:center; }
figure img { max-width:100%; border:1px solid #e6eaf0; border-radius:10px; }
figcaption { font-size:12px; color:#64748b; margin-top:6px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.grid2 figure { margin:6px 0; }
.callout { background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:14px 16px; margin:12px 0; font-size:14px; }
.callout b { color:#92400e; }
.impact { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }
.impact-card { border:1px solid #e6eaf0; border-radius:12px; padding:14px; }
.impact-card span { font-size:11.5px; color:#64748b; display:block; }
.impact-card b { font-size:19px; display:block; margin:4px 0; letter-spacing:-.4px; }
.impact-card small { font-size:11px; color:#94a3b8; }
.impact-card.good { background:#ecfdf5; border-color:#bbf7d0; }
.impact-card.warn { background:#fff7ed; border-color:#fed7aa; }
.impact-card.net { background:#eff6ff; border-color:#bfdbfe; }
.arch { width:100%; height:auto; background:#fbfcfe; border:1px solid #eef1f6; border-radius:14px; margin:14px 0; }
.missing { color:#b91c1c; font-size:12px; padding:10px; background:#fef2f2; border-radius:8px; }
.foot { padding:24px 48px; color:#64748b; font-size:12.5px; background:#f8fafc; }
@media print {
  body { background:#fff; } .report { box-shadow:none; max-width:none; }
  section { page-break-inside:avoid; } .cover { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
@media (max-width:680px) { .grid2, .impact { grid-template-columns:1fr; } section, .cover, .foot { padding-left:24px; padding-right:24px; } }
"""


if __name__ == "__main__":
    build_report()
