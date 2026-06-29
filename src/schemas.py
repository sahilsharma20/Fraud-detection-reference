"""Pydantic request/response contracts for the API boundary (Module 9).

WHY Pydantic here (vs pandera at the data boundary): this validates UNTRUSTED
input arriving over HTTP. A bad type, an out-of-range age, a negative claim
amount — caught here with a clean 422 and a precise message, never reaching the
model. Defaults mirror a typical claim so a partially-filled form still scores.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimRequest(BaseModel):
    """A single auto-insurance claim submitted for scoring.

    Field names match the dataset columns. ``capital-gains`` / ``capital-loss``
    carry hyphens in the raw data, so they're exposed via aliases and dumped with
    ``by_alias=True`` before inference.
    """

    model_config = ConfigDict(populate_by_name=True)

    # ── policy / customer ──
    months_as_customer: int = Field(120, ge=0, le=600)
    age: int = Field(42, ge=16, le=100)
    policy_state: Literal["OH", "IN", "IL"] = "OH"
    policy_csl: Literal["100/300", "250/500", "500/1000"] = "250/500"
    policy_deductable: int = Field(1000, ge=0)
    policy_annual_premium: float = Field(1280.5, gt=0)
    umbrella_limit: int = Field(0, ge=0)
    policy_bind_date: str = "2010-05-12"

    # ── insured ──
    insured_sex: Literal["MALE", "FEMALE"] = "MALE"
    insured_education_level: str = "Masters"
    insured_occupation: str = "exec-managerial"
    insured_hobbies: str = "reading"
    insured_relationship: str = "husband"
    capital_gains: int = Field(0, alias="capital-gains")
    capital_loss: int = Field(0, alias="capital-loss")

    # ── incident ──
    incident_date: str = "2015-02-20"
    incident_type: str = "Multi-vehicle Collision"
    collision_type: str = "Rear Collision"
    incident_severity: str = "Minor Damage"
    authorities_contacted: str = "Police"
    incident_state: str = "NY"
    incident_city: str = "Columbus"
    incident_hour_of_the_day: int = Field(14, ge=0, le=23)
    number_of_vehicles_involved: int = Field(2, ge=1, le=10)
    property_damage: str = "NO"
    bodily_injuries: int = Field(0, ge=0, le=10)
    witnesses: int = Field(2, ge=0, le=20)
    police_report_available: str = "YES"

    # ── claim amounts (valid features for fraud; leakage for severity) ──
    total_claim_amount: int = Field(32000, ge=0)
    injury_claim: int = Field(6000, ge=0)
    property_claim: int = Field(6000, ge=0)
    vehicle_claim: int = Field(20000, ge=0)

    # ── vehicle ──
    auto_make: str = "Toyota"
    auto_year: int = Field(2012, ge=1950, le=2025)

    def to_claim_dict(self) -> dict:
        """Return a dict with original column names (hyphens restored)."""
        return self.model_dump(by_alias=True)


class Reason(BaseModel):
    """A single human-readable SHAP-derived factor."""

    reason: str
    value: str
    impact: float
    direction: str


class FraudResult(BaseModel):
    """Stage-1 fraud verdict + probability + explanation."""

    verdict: Literal["GENUINE", "NEEDS_REVIEW"]
    is_fraud_suspected: bool
    fraud_probability: float
    threshold: float
    reasons: list[Reason]


class SeverityResult(BaseModel):
    """Stage-2 severity estimate (null fields when the claim was flagged)."""

    predicted_amount: float | None = None
    currency: str | None = None
    error_band: float | None = None
    lower_estimate: float | None = None
    upper_estimate: float | None = None
    reasons: list[Reason] | None = None
    note: str | None = None


class PredictionResponse(BaseModel):
    """The chained two-stage result returned to the UI/API caller."""

    fraud: FraudResult
    severity: SeverityResult
    meta: dict
