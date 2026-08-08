from __future__ import annotations

import datetime as dt
from pydantic import BaseModel, Field


class ShipmentFeatures(BaseModel):
    """Everything the delay model needs to make a prediction. Mirrors the
    columns on the Shipment table minus the label."""

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


class Option(BaseModel):
    label: str
    description: str
    cost_usd: float
    resulting_delay_days: float
    within_budget: bool


class PrescribeResponse(BaseModel):
    prediction: DelayPrediction
    options: list[Option]
    shipment_sku: str
    budget_cap_usd: float


class DecisionCreate(BaseModel):
    shipment_sku: str
    predicted_delay_days: float
    predicted_delay_probability: float
    options: list[Option]
    chosen_option_label: str
    budget_cap_usd: float


class DecisionOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    shipment_sku: str
    predicted_delay_days: float
    predicted_delay_probability: float
    chosen_option_label: str
    predicted_cost_usd: float
    budget_cap_usd: float
    actual_cost_usd: float | None
    actual_delay_days: float | None
    created_at: dt.datetime
    resolved_at: dt.datetime | None
    is_resolved: bool


class OutcomeUpdate(BaseModel):
    actual_cost_usd: float = Field(gt=0)
    actual_delay_days: float = Field(ge=0)


class RoiSummary(BaseModel):
    total_decisions: int
    resolved_decisions: int
    avg_predicted_cost_usd: float | None
    avg_actual_cost_usd: float | None
    avg_cost_error_pct: float | None
    decisions_within_budget_pct: float | None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelInfo(BaseModel):
    """Training metrics for demos /docs — loaded from data/metrics.json when present."""

    model_loaded: bool
    model_path: str
    mae_days: float | None = None
    auc: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    top_features: list[FeatureImportance] = []
    dataset_rows: int | None = None
    dataset_message: str | None = None


class SupplierStat(BaseModel):
    supplier: str
    n: int
    mean_delay_days: float


class DatasetSource(BaseModel):
    key: str
    label: str
    n_rows: int


class DatasetFile(BaseModel):
    key: str
    label: str
    path: str
    n_rows: int


class DatasetSummary(BaseModel):
    available: bool
    message: str
    n_rows: int
    n_suppliers: int
    n_skus: int
    n_regions: int
    late_rate_pct: float | None = None
    mean_delay_days: float | None = None
    median_delay_days: float | None = None
    regions: list[str] = []
    top_suppliers: list[SupplierStat] = []
    sources: list[DatasetSource] = []
    files: list[DatasetFile] = []


class DemoScenario(BaseModel):
    id: str
    label: str
    blurb: str
    values: dict


class SampleShipment(BaseModel):
    sku: str
    supplier: str
    origin_region: str
    distance_km: float
    historical_avg_lead_time_days: float
    order_quantity: int
    unit_cost_usd: float
    is_peak_season: bool
    actual_delay_days: float
