// ClaimGuard front-end. Talks ONLY to the API (/predict, /sample) — no inference
// logic lives here; the server's chained pipeline is the single source of truth.

const form = document.getElementById("claimForm");
const resultBody = document.getElementById("resultBody");
const emptyState = document.getElementById("emptyState");

/** Collect form fields into a JSON claim payload (numbers coerced). */
function readForm() {
  const data = {};
  new FormData(form).forEach((value, key) => {
    const el = form.elements[key];
    data[key] = el && el.type === "number" ? Number(value) : value;
  });
  return data;
}

/** Populate the form from a sample claim object returned by /sample. */
function fillForm(sample) {
  Object.entries(sample).forEach(([key, value]) => {
    const el = form.elements[key];
    if (el) el.value = value;
  });
}

async function loadSample(type) {
  const res = await fetch(`/sample?type=${type}`);
  fillForm(await res.json());
}

document.getElementById("loadGenuine").addEventListener("click", () => loadSample("genuine"));
document.getElementById("loadFraud").addEventListener("click", () => loadSample("fraud"));

const fmtCurrency = (sym, n) =>
  `${sym}${Math.round(n).toLocaleString("en-IN")}`;

function renderReasons(reasons) {
  const ul = document.getElementById("reasonsList");
  ul.innerHTML = "";
  (reasons || []).forEach((r) => {
    const up = r.direction.includes("increase") || r.direction.includes("raise");
    const li = document.createElement("li");
    li.className = "reason-item";
    li.innerHTML = `
      <span class="reason-arrow ${up ? "up" : "down"}">${up ? "▲" : "▼"}</span>
      <span class="reason-text">${r.reason}
        <small>${r.value !== "" && r.value != null ? "value: " + r.value + " · " : ""}${r.direction}</small>
      </span>`;
    ul.appendChild(li);
  });
}

function render(result) {
  const f = result.fraud;
  const s = result.severity;
  const sym = result.meta.currency || "₹";
  const flagged = f.verdict === "NEEDS_REVIEW";

  // verdict badge
  const badge = document.getElementById("verdictBadge");
  badge.textContent = flagged ? "🚩 NEEDS REVIEW" : "✅ GENUINE";
  badge.className = "verdict-badge " + (flagged ? "review" : "genuine");

  // probability bar
  const pct = (f.fraud_probability * 100).toFixed(1);
  document.getElementById("probValue").textContent = pct + "%";
  const fill = document.getElementById("probFill");
  fill.style.width = Math.min(f.fraud_probability * 100, 100) + "%";
  fill.style.background = flagged ? "var(--red)" : "var(--green)";
  document.getElementById("probThreshold").style.left = f.threshold * 100 + "%";
  document.getElementById("thresholdNote").textContent =
    `Decision threshold = ${(f.threshold * 100).toFixed(1)}% (cost-optimised, not the default 50%). Flagged when probability ≥ threshold.`;

  // severity
  const box = document.getElementById("severityBox");
  const note = document.getElementById("severityNote");
  if (flagged || s.predicted_amount == null) {
    document.getElementById("severityAmount").textContent = "—";
    document.getElementById("severityBand").textContent = "";
    note.textContent = s.note || "Severity is not predicted for flagged claims.";
    note.classList.remove("hidden");
  } else {
    document.getElementById("severityAmount").textContent = fmtCurrency(sym, s.predicted_amount);
    document.getElementById("severityBand").textContent =
      s.error_band
        ? `Expected range ${fmtCurrency(sym, s.lower_estimate)} – ${fmtCurrency(sym, s.upper_estimate)} (± model MAE of ${fmtCurrency(sym, s.error_band)})`
        : "";
    note.classList.add("hidden");
  }

  renderReasons(f.reasons);

  document.getElementById("footerMeta").innerHTML = `
    <span>Fraud model: <b>${result.meta.fraud_model}</b></span>
    <span>Severity model: <b>${result.meta.severity_model}</b></span>
    <span>Latency: <b>${result.meta.latency_ms} ms</b></span>`;

  emptyState.classList.add("hidden");
  resultBody.classList.remove("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("checkBtn");
  btn.textContent = "Checking…";
  form.classList.add("loading");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    if (!res.ok) throw new Error("Server error " + res.status);
    render(await res.json());
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    btn.textContent = "Check Claim";
    form.classList.remove("loading");
  }
});
