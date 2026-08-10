"""
Week 3 - wires the trained model + solver up to a small FastAPI service
with the write-back path the project brief calls for: a manager hits
/prescribe, picks an option, POSTs it to /decisions (that's the INSERT
into the operational table), and later PATCHes the outcome once the
real cost/delay is known.

Measurement endpoints (intentionally separated):
  GET /decisions/cost-accuracy  — forecast error + budget adherence
  GET /decisions/roi            — true ROI vs Delay Launch (no-action)

The old "ROI" label for cost-error metrics was misleading; both routes
exist so demos and tests can show the distinction clearly.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

# main.py lives in week3/ but needs week1 (data layer + model) and week2
# (solver) as siblings - put the project root on sys.path so `week1.` /
# `week2.` imports resolve regardless of how this gets launched
# (uvicorn week3.main:app from the repo root, or a direct python run).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from week1 import models
from week1.config import (
    DEFAULT_BUDGET_USD,
    DEFAULT_MAX_DELAY_DAYS,
    DEFAULT_MIN_ON_TIME_FRACTION,
    METRICS_PATH,
    MODEL_PATH,
    PARTIAL_FULFILLMENT_USEFUL,
    ROOT_DIR,
)
from week1.database import get_session, init_db
from week1.delay_model import DelayModel
from week2.solver import pure_options, solve_optimal_allocation

from . import schemas

FRONTEND_DIR = ROOT_DIR / "week2" / "frontend"
DELAY_LAUNCH_LABEL = "Delay Launch"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SupplyPrescript API", version="0.5.0", lifespan=lifespan)
# Allow local dashboard opened as a file (null origin) or via a simple
# static server on common localhost ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> RedirectResponse:
    """Send browsers to the dashboard; API docs stay at /docs."""
    return RedirectResponse(url="/ui/")


_model: DelayModel | None = None


def get_model() -> DelayModel:
    """Lazy-load so `pytest` doesn't need a trained artifact on disk just
    to import this module - only routes that actually predict pay for it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Delay model not found at {MODEL_PATH} - run week1/train_model.py first",
            )
        _model = DelayModel.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=schemas.ModelInfo)
def model_info(model: DelayModel = Depends(get_model)) -> schemas.ModelInfo:
    """Return held-out training metrics so presenters can quote MAE/AUC live.

    Metrics come from synthetic data with programmed relationships —
    useful for demos, not claims about real-world delay prediction.
    """
    payload: dict = {
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "mae_days": None,
        "auc": None,
        "n_train": None,
        "n_test": None,
        "validation_strategy": None,
        "baseline_mae_days": None,
        "baseline_auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "brier_score": None,
        "data_is_synthetic": True,
        "top_features": [],
    }
    if METRICS_PATH.exists():
        saved = json.loads(METRICS_PATH.read_text())
        for key in (
            "mae_days",
            "auc",
            "n_train",
            "n_test",
            "validation_strategy",
            "baseline_mae_days",
            "baseline_auc",
            "precision",
            "recall",
            "f1",
            "brier_score",
            "data_is_synthetic",
        ):
            if key in saved:
                payload[key] = saved[key]
        payload["top_features"] = saved.get("top_features") or []
    else:
        # Fallback: compute top features from the loaded model artifact.
        payload["top_features"] = model.feature_importance(top_n=5)
    return schemas.ModelInfo(**payload)


@app.post("/predict", response_model=schemas.DelayPrediction)
def predict(shipment: schemas.ShipmentFeatures, model: DelayModel = Depends(get_model)):
    days, prob = model.predict_one(shipment.model_dump())
    return schemas.DelayPrediction(predicted_delay_days=round(days, 1), predicted_delay_probability=round(prob, 3))


@app.post("/prescribe", response_model=schemas.PrescribeResponse)
def prescribe(request: schemas.PrescribeRequest, model: DelayModel = Depends(get_model)):
    days, prob = model.predict_one(request.shipment.model_dump())
    days = round(days, 1)
    prob = round(prob, 3)
    budget_cap = request.budget_cap_usd or DEFAULT_BUDGET_USD
    max_delay = request.max_acceptable_delay_days or DEFAULT_MAX_DELAY_DAYS
    partial = (
        PARTIAL_FULFILLMENT_USEFUL
        if request.partial_fulfillment_useful is None
        else request.partial_fulfillment_useful
    )
    min_on_time = (
        DEFAULT_MIN_ON_TIME_FRACTION
        if request.min_on_time_fraction is None
        else request.min_on_time_fraction
    )

    options = pure_options(
        unit_cost_usd=request.shipment.unit_cost_usd,
        order_quantity=request.shipment.order_quantity,
        predicted_delay_days=days,
        budget_cap_usd=budget_cap,
        predicted_delay_probability=prob,
    )
    no_action = next(o for o in options if o["label"] == DELAY_LAUNCH_LABEL)

    # bolt the solver's blended recommendation on as a fourth card - it's
    # usually the cheapest way to satisfy the delay constraint, which a
    # manager comparing three "all or nothing" options wouldn't see.
    blend = solve_optimal_allocation(
        unit_cost_usd=request.shipment.unit_cost_usd,
        order_quantity=request.shipment.order_quantity,
        predicted_delay_days=days,
        budget_cap_usd=budget_cap,
        max_acceptable_delay_days=max_delay,
        predicted_delay_probability=prob,
        partial_fulfillment_useful=partial,
        min_on_time_fraction=min_on_time,
    )
    allocation_desc = ", ".join(
        f"{qty:.0f} units via {label.replace('_', ' ')}"
        for label, qty in blend["allocation_units"].items()
        if qty > 0
    )
    options.append(
        {
            "label": "Optimizer Recommended Split",
            "description": (
                f"MILP allocation ({blend['delay_constraint_mode']}): {allocation_desc}. "
                f"Fixed fees ${blend['fixed_fees_usd']:,.0f} included in budget check."
                + (
                    " (over budget cap - shown anyway since the budget-constrained problem was infeasible)"
                    if blend["budget_relaxed"]
                    else ""
                )
            ),
            "cost_usd": blend["total_cost_usd"],
            "resulting_delay_days": blend["resulting_delay_days"],
            "within_budget": blend["within_budget"],
        }
    )

    return schemas.PrescribeResponse(
        prediction=schemas.DelayPrediction(predicted_delay_days=days, predicted_delay_probability=prob),
        options=[schemas.Option(**opt) for opt in options],
        shipment_sku=request.shipment.sku,
        budget_cap_usd=budget_cap,
        delay_constraint_mode=blend["delay_constraint_mode"],
        no_action_cost_usd=no_action["cost_usd"],
    )


@app.post("/decisions", response_model=schemas.DecisionOut, status_code=201)
def create_decision(payload: schemas.DecisionCreate, session: Session = Depends(get_session)):
    """The write-back step: persist which option the manager actually picked."""
    chosen = next((o for o in payload.options if o.label == payload.chosen_option_label), None)
    if chosen is None:
        raise HTTPException(status_code=422, detail=f"'{payload.chosen_option_label}' isn't one of the options that were offered")

    # Prefer explicit no-action cost; otherwise recover Delay Launch from options.
    no_action_cost = payload.no_action_cost_usd
    if no_action_cost is None:
        delay_opt = next((o for o in payload.options if o.label == DELAY_LAUNCH_LABEL), None)
        no_action_cost = delay_opt.cost_usd if delay_opt is not None else None

    features_json = None
    if payload.shipment_features is not None:
        features_json = json.dumps(payload.shipment_features.model_dump())

    decision = models.Decision(
        shipment_sku=payload.shipment_sku,
        predicted_delay_days=payload.predicted_delay_days,
        predicted_delay_probability=payload.predicted_delay_probability,
        options_json=json.dumps([o.model_dump() for o in payload.options]),
        shipment_features_json=features_json,
        chosen_option_label=payload.chosen_option_label,
        predicted_cost_usd=chosen.cost_usd,
        no_action_cost_usd=no_action_cost,
        budget_cap_usd=payload.budget_cap_usd,
    )
    session.add(decision)
    session.commit()
    session.refresh(decision)
    return decision


@app.get("/decisions", response_model=list[schemas.DecisionOut])
def list_decisions(session: Session = Depends(get_session)):
    return session.query(models.Decision).order_by(models.Decision.created_at.desc()).all()


@app.patch("/decisions/{decision_id}/outcome", response_model=schemas.DecisionOut)
def record_outcome(decision_id: int, outcome: schemas.OutcomeUpdate, session: Session = Depends(get_session)):
    """Closes the loop: three weeks later, someone finds out air freight
    actually cost $18k instead of the predicted $15k, and logs it here."""
    decision = session.get(models.Decision, decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="No decision with that id")
    if decision.is_resolved:
        raise HTTPException(status_code=409, detail="Outcome already recorded for this decision")

    decision.actual_cost_usd = outcome.actual_cost_usd
    decision.actual_delay_days = outcome.actual_delay_days
    decision.resolved_at = dt.datetime.now(dt.UTC)
    session.commit()
    session.refresh(decision)
    return decision


def _cost_accuracy_summary(session: Session) -> schemas.CostAccuracySummary:
    all_decisions = session.query(models.Decision).all()
    resolved = [d for d in all_decisions if d.is_resolved]

    if not resolved:
        return schemas.CostAccuracySummary(
            total_decisions=len(all_decisions),
            resolved_decisions=0,
            avg_predicted_cost_usd=None,
            avg_actual_cost_usd=None,
            avg_cost_error_pct=None,
            decisions_within_budget_pct=None,
        )

    avg_predicted = sum(d.predicted_cost_usd for d in resolved) / len(resolved)
    avg_actual = sum(d.actual_cost_usd for d in resolved) / len(resolved)
    errors_pct = [
        abs(d.actual_cost_usd - d.predicted_cost_usd) / d.predicted_cost_usd
        for d in resolved
        if d.predicted_cost_usd
    ]
    within_budget = [d for d in resolved if d.actual_cost_usd <= d.budget_cap_usd]

    return schemas.CostAccuracySummary(
        total_decisions=len(all_decisions),
        resolved_decisions=len(resolved),
        avg_predicted_cost_usd=round(avg_predicted, 2),
        avg_actual_cost_usd=round(avg_actual, 2),
        avg_cost_error_pct=round(sum(errors_pct) / len(errors_pct) * 100, 1) if errors_pct else None,
        decisions_within_budget_pct=round(len(within_budget) / len(resolved) * 100, 1),
    )


@app.get("/decisions/cost-accuracy", response_model=schemas.CostAccuracySummary)
def decisions_cost_accuracy(session: Session = Depends(get_session)):
    """Cost forecast accuracy + budget adherence (not ROI)."""
    return _cost_accuracy_summary(session)


@app.get("/decisions/roi", response_model=schemas.InterventionRoiSummary)
def decisions_roi(session: Session = Depends(get_session)):
    """True intervention ROI versus the stored no-action counterfactual.

    Avoided loss = no_action_cost - actual_cost.
    Falls back to an empty counterfactual summary when no decisions have
    a Delay Launch baseline stored.
    """
    all_decisions = session.query(models.Decision).all()
    resolved = [d for d in all_decisions if d.is_resolved]
    with_cf = [d for d in resolved if d.no_action_cost_usd is not None and d.no_action_cost_usd > 0]

    if not with_cf:
        return schemas.InterventionRoiSummary(
            total_decisions=len(all_decisions),
            resolved_decisions=len(resolved),
            decisions_with_counterfactual=0,
            avg_no_action_cost_usd=None,
            avg_actual_cost_usd=None,
            avg_avoided_loss_usd=None,
            avg_roi_pct=None,
            interventions_beating_no_action_pct=None,
        )

    avoided = [d.no_action_cost_usd - d.actual_cost_usd for d in with_cf]
    roi_pcts = [(d.no_action_cost_usd - d.actual_cost_usd) / d.no_action_cost_usd * 100 for d in with_cf]
    beats = [a for a in avoided if a > 0]

    return schemas.InterventionRoiSummary(
        total_decisions=len(all_decisions),
        resolved_decisions=len(resolved),
        decisions_with_counterfactual=len(with_cf),
        avg_no_action_cost_usd=round(sum(d.no_action_cost_usd for d in with_cf) / len(with_cf), 2),
        avg_actual_cost_usd=round(sum(d.actual_cost_usd for d in with_cf) / len(with_cf), 2),
        avg_avoided_loss_usd=round(sum(avoided) / len(avoided), 2),
        avg_roi_pct=round(sum(roi_pcts) / len(roi_pcts), 1),
        interventions_beating_no_action_pct=round(len(beats) / len(with_cf) * 100, 1),
    )


# Mount last so /ui never shadows API routes above.
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
