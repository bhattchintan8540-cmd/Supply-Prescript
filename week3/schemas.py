from __future__ import annotations

import datetime as dt
from pydantic import BaseModel, Field


class ShipmentFeatures(BaseModel):
    """Everything the delay model needs to make a prediction. Mirrors the
    columns on the Shipment table minus the label (and shipment_date,
    which is training-only for temporal splits)."""

    sku: str
    supplier: str
    origin_region: str
    distance_km: float = Field(gt=0)
    historical_avg_lead_time_days: float = Field(gt=0)
    order_quantity: int = Field(gt=0)
    unit_cost_usd: float = Field(gt=0)
    is_peak_season: bool = False


class DelayPrediction(BaseModel):
    predicted_delay_days: float
    predicted_delay_probability: float  # P(delay > 3 days), see week1/delay_model.py


class PrescribeRequest(BaseModel):
    shipment: ShipmentFeatures
    budget_cap_usd: float | None = None  # falls back to config default if omitted
    max_acceptable_delay_days: int | None = None
    # When omitted, server uses PARTIAL_FULFILLMENT_USEFUL from config.
    partial_fulfillment_useful: bool | None = None
    min_on_time_fraction: float | None = None


class Option(BaseModel):
    label: str
    description: str
    cost_usd: float
    resulting_delay_days: float
    within_budget: bool
    within_sla: bool | None = None
    solver_status: str | None = None
    allocation_units: dict[str, float] | None = None
    delay_constraint_mode: str | None = None


class PrescribeResponse(BaseModel):
    prediction: DelayPrediction
    options: list[Option]
    shipment_sku: str
    budget_cap_usd: float
    delay_constraint_mode: str | None = None
    no_action_cost_usd: float | None = None


class DecisionCreate(BaseModel):
    shipment_sku: str
    predicted_delay_days: float
    predicted_delay_probability: float
    options: list[Option]
    chosen_option_label: str
    budget_cap_usd: float
    # Optional feature snapshot for closed-loop retraining.
    shipment_features: ShipmentFeatures | None = None
    no_action_cost_usd: float | None = None


class DecisionOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    shipment_sku: str
    predicted_delay_days: float
    predicted_delay_probability: float
    chosen_option_label: str
    predicted_cost_usd: float
    no_action_cost_usd: float | None = None
    budget_cap_usd: float
    actual_cost_usd: float | None
    actual_delay_days: float | None
    created_at: dt.datetime
    resolved_at: dt.datetime | None
    is_resolved: bool


class OutcomeUpdate(BaseModel):
    actual_cost_usd: float = Field(gt=0)
    actual_delay_days: float = Field(ge=0)


class CostAccuracySummary(BaseModel):
    """Cost forecast accuracy and budget adherence — NOT ROI.

    These metrics answer: "How well did we predict the cost of the chosen
    option, and did outcomes stay within budget?" They do not measure
    value created versus doing nothing.
    """

    total_decisions: int
    resolved_decisions: int
    avg_predicted_cost_usd: float | None
    avg_actual_cost_usd: float | None
    avg_cost_error_pct: float | None
    decisions_within_budget_pct: float | None


# Backward-compatible alias used by older clients / docs during transition.
RoiSummary = CostAccuracySummary


class InterventionRoiSummary(BaseModel):
    """True intervention ROI versus the no-action (Delay Launch) baseline.

    Avoided loss = no_action_cost - actual_cost_after_intervention.
    ROI% = avoided_loss / no_action_cost.

    Requires no_action_cost_usd to have been stored at decision time.
    """

    total_decisions: int
    resolved_decisions: int
    decisions_with_counterfactual: int
    avg_no_action_cost_usd: float | None
    avg_actual_cost_usd: float | None
    avg_avoided_loss_usd: float | None
    avg_roi_pct: float | None
    interventions_beating_no_action_pct: float | None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelInfo(BaseModel):
    """Training metrics for demos /docs — loaded from data/metrics.json when present.

    Metrics are recovered from *synthetic* data with programmed relationships.
    They demonstrate that the model recovers those relationships under
    temporal validation — not real-world supply-chain accuracy.
    """

    model_loaded: bool
    model_path: str
    mae_days: float | None = None
    auc: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    n_val: int | None = None
    validation_strategy: str | None = None
    validation_used_for_tuning: bool | None = None
    decision_threshold: float | None = None
    baseline_mae_days: float | None = None
    baseline_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    brier_score: float | None = None
    data_is_synthetic: bool = True
    top_features: list[FeatureImportance] = []
