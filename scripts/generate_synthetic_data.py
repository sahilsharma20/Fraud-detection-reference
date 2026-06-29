"""Generate a synthetic dataset that is *schema-identical* to the Kaggle
"Auto Insurance Claims Fraud Detection" file (``insurance_claims.csv``).

WHY THIS EXISTS (a deliberate engineering decision, not a shortcut):
    The real Kaggle dataset is only ~1,000 rows and is redistribution-restricted,
    so it cannot be committed to a public repo or baked into a public Docker
    image. To keep the project *fully reproducible and deployable by anyone* —
    `git clone && make data && make train && make serve` works with zero manual
    downloads — we generate a synthetic file with:
      * the exact 40-column schema (same names, dtypes, category levels, even the
        junk trailing ``_c39`` column the Kaggle export ships),
      * the real arithmetic identity ``total_claim_amount = injury + property +
        vehicle`` preserved (so the severity leakage lesson is demonstrable),
      * an injected, *learnable but noisy* fraud signal so model metrics are
        meaningful rather than random.

    DROP-IN REAL DATA: download the Kaggle CSV, drop it at
    ``data/raw/insurance_claims.csv`` (overwriting this synthetic one) and rerun
    `make train`. Nothing else changes — the schema matches by construction.

Kaggle source (confirmed to have BOTH `fraud_reported` and `total_claim_amount`):
    https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
    Exact filename to place in data/raw/:  insurance_claims.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)

# ─── Category levels mirrored from the real Kaggle dataset ──────────────────────
POLICY_STATES = ["OH", "IN", "IL"]
CSL = ["100/300", "250/500", "500/1000"]
SEX = ["MALE", "FEMALE"]
EDU = ["JD", "High School", "Associate", "MD", "Masters", "PhD", "College"]
OCCUPATION = [
    "craft-repair", "machine-op-inspct", "sales", "armed-forces", "tech-support",
    "prof-specialty", "other-service", "priv-house-serv", "exec-managerial",
    "protective-serv", "transport-moving", "handlers-cleaners", "adm-clerical",
    "farming-fishing",
]
HOBBIES = [
    "sleeping", "reading", "board-games", "bungie-jumping", "base-jumping", "golf",
    "camping", "dancing", "skydiving", "movies", "hiking", "yachting", "paintball",
    "chess", "kayaking", "polo", "basketball", "video-games", "cross-fit", "exercise",
]
RELATIONSHIP = ["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family"]
INCIDENT_TYPE = ["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"]
COLLISION = ["Side Collision", "Rear Collision", "Front Collision"]
SEVERITY = ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"]
AUTHORITIES = ["Police", "Fire", "Other", "Ambulance", "None"]
INCIDENT_STATE = ["NY", "SC", "WV", "VA", "NC", "PA", "OH"]
INCIDENT_CITY = ["Springfield", "Arlington", "Columbus", "Northbend", "Hillsdale", "Riverwood", "Northbrook"]
YESNO_Q = ["YES", "NO", "?"]  # '?' is how the raw file encodes "unknown"
AUTO_MAKE = [
    "Saab", "Mercedes", "Dodge", "Chevrolet", "Accura", "Nissan", "Audi", "Toyota",
    "Ford", "Suburu", "BMW", "Jeep", "Honda", "Volkswagen",
]

# Hobbies that are oddly fraud-predictive in the real dataset — we reproduce that
# quirk so SHAP surfaces something interesting (and a touch of realism).
HIGH_RISK_HOBBIES = {"chess", "cross-fit"}

N_ROWS = 2000
SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build the synthetic claims DataFrame with an injected fraud signal.

    Args:
        n_rows: Number of claim rows to synthesise.
        seed: RNG seed for full reproducibility.

    Returns:
        A DataFrame with the exact 40-column Kaggle schema.
    """
    rng = np.random.default_rng(seed)

    months_as_customer = rng.integers(0, 480, n_rows)
    age = np.clip((months_as_customer / 12) + rng.integers(19, 45, n_rows), 19, 80).astype(int)

    # Policies bound 0-15 years ago; incidents occur in a recent 1-year window.
    bind_offset = rng.integers(30, 15 * 365, n_rows)
    policy_bind_date = (pd.Timestamp("2015-01-01") - pd.to_timedelta(bind_offset, unit="D"))
    incident_offset = rng.integers(1, 365, n_rows)
    incident_date = pd.Timestamp("2015-01-01") + pd.to_timedelta(incident_offset, unit="D")

    incident_type = rng.choice(INCIDENT_TYPE, n_rows, p=[0.4, 0.15, 0.35, 0.10])
    severity = rng.choice(SEVERITY, n_rows, p=[0.25, 0.35, 0.15, 0.25])
    hobbies = rng.choice(HOBBIES, n_rows)
    police_report = rng.choice(YESNO_Q, n_rows, p=[0.35, 0.35, 0.30])
    authorities = rng.choice(AUTHORITIES, n_rows, p=[0.35, 0.15, 0.15, 0.20, 0.15])

    # Claim components. Vehicle claim dominates; severity scales the magnitude.
    sev_mult = np.select(
        [severity == "Trivial Damage", severity == "Minor Damage",
         severity == "Major Damage", severity == "Total Loss"],
        [0.4, 0.8, 1.4, 1.8], default=1.0,
    )
    vehicle_claim = np.round(rng.gamma(shape=4.0, scale=9000, size=n_rows) * sev_mult, -1)
    injury_claim = np.round(rng.gamma(shape=2.0, scale=3000, size=n_rows) * sev_mult, -1)
    property_claim = np.round(rng.gamma(shape=2.0, scale=3000, size=n_rows) * sev_mult, -1)
    # ── arithmetic identity preserved: this is what makes the components leakage
    #    for the severity target (they sum to it exactly). ──
    total_claim_amount = (vehicle_claim + injury_claim + property_claim).astype(int)

    # ─── Injected fraud signal: a logit over several drivers, INCLUDING
    #     non-linear interactions that a linear model cannot capture but tree
    #     ensembles can — so XGBoost/RandomForest legitimately beat the LR
    #     baseline (a realistic outcome, not a rigged one). ───
    incident_hour = rng.integers(0, 24, n_rows)  # reused for the stored column below
    major_or_total = np.isin(severity, ["Major Damage", "Total Loss"]).astype(float)
    no_police = (police_report == "NO").astype(float)
    late_night = (incident_hour < 5) | (incident_hour > 22)
    logit = (
        -3.5  # intercept tuned so overall fraud prevalence ≈ 25% (matches real data)
        + 1.5 * major_or_total
        + 1.2 * no_police
        + 1.1 * (authorities == "None").astype(float)
        + 1.4 * np.isin(hobbies, list(HIGH_RISK_HOBBIES)).astype(float)
        + 1.0 * (total_claim_amount > 60000).astype(float)
        + 1.7 * (major_or_total * no_police)          # interaction -> favours trees
        + 0.9 * late_night.astype(float)              # time-of-day non-linearity
        + rng.normal(0, 0.30, n_rows)                 # noise kept so metrics stay realistic (no fake 0.99)
    )
    fraud_prob = _sigmoid(logit)
    fraud_flag = rng.binomial(1, fraud_prob)
    fraud_reported = np.where(fraud_flag == 1, "Y", "N")

    df = pd.DataFrame(
        {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_number": rng.integers(100000, 999999, n_rows),
            "policy_bind_date": policy_bind_date.strftime("%Y-%m-%d"),
            "policy_state": rng.choice(POLICY_STATES, n_rows),
            "policy_csl": rng.choice(CSL, n_rows),
            "policy_deductable": rng.choice([500, 1000, 2000], n_rows),
            "policy_annual_premium": np.round(rng.normal(1250, 250, n_rows), 2),
            "umbrella_limit": rng.choice([0, 0, 0, 4000000, 6000000, 8000000], n_rows),
            "insured_zip": rng.integers(430000, 620000, n_rows),
            "insured_sex": rng.choice(SEX, n_rows),
            "insured_education_level": rng.choice(EDU, n_rows),
            "insured_occupation": rng.choice(OCCUPATION, n_rows),
            "insured_hobbies": hobbies,
            "insured_relationship": rng.choice(RELATIONSHIP, n_rows),
            "capital-gains": rng.choice([0], n_rows) + (rng.random(n_rows) < 0.4) * rng.integers(0, 100000, n_rows),
            "capital-loss": -((rng.random(n_rows) < 0.3) * rng.integers(0, 110000, n_rows)),
            "incident_date": incident_date.strftime("%Y-%m-%d"),
            "incident_type": incident_type,
            # collision_type is '?' for theft/parked incidents (mirrors raw quirk)
            "collision_type": np.where(
                np.isin(incident_type, ["Vehicle Theft", "Parked Car"]),
                "?", rng.choice(COLLISION, n_rows),
            ),
            "incident_severity": severity,
            "authorities_contacted": authorities,
            "incident_state": rng.choice(INCIDENT_STATE, n_rows),
            "incident_city": rng.choice(INCIDENT_CITY, n_rows),
            "incident_location": [f"{rng.integers(100, 9999)} {city} Rd" for city in rng.choice(INCIDENT_CITY, n_rows)],
            "incident_hour_of_the_day": incident_hour,  # links to the late_night fraud signal
            "number_of_vehicles_involved": rng.choice([1, 2, 3, 4], n_rows, p=[0.5, 0.25, 0.15, 0.10]),
            "property_damage": rng.choice(YESNO_Q, n_rows, p=[0.3, 0.3, 0.4]),
            "bodily_injuries": rng.choice([0, 1, 2], n_rows),
            "witnesses": rng.choice([0, 1, 2, 3], n_rows),
            "police_report_available": police_report,
            "total_claim_amount": total_claim_amount,
            "injury_claim": injury_claim.astype(int),
            "property_claim": property_claim.astype(int),
            "vehicle_claim": vehicle_claim.astype(int),
            "auto_make": rng.choice(AUTO_MAKE, n_rows),
            "auto_model": [f"M{m}" for m in rng.integers(100, 999, n_rows)],
            "auto_year": rng.integers(1995, 2016, n_rows),
            "fraud_reported": fraud_reported,
            "_c39": np.nan,  # the empty junk column the Kaggle CSV ships with
        }
    )
    return df


def main() -> None:
    """Generate the dataset and write it to the configured raw-data path."""
    cfg = load_config()
    out_path = cfg.path("paths.raw_data")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(out_path, index=False)

    fraud_rate = (df["fraud_reported"] == "Y").mean()
    log.info("Synthetic dataset written: %s", out_path)
    log.info("Rows=%d  Cols=%d  Fraud rate=%.1f%%", len(df), df.shape[1], fraud_rate * 100)
    # Prove the arithmetic identity holds (the basis of the severity leakage rule)
    identity_ok = bool(
        (df["injury_claim"] + df["property_claim"] + df["vehicle_claim"] == df["total_claim_amount"]).all()
    )
    log.info("total = injury+property+vehicle identity holds: %s", identity_ok)


if __name__ == "__main__":
    main()
