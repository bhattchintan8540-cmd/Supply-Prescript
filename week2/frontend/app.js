// Resolve the FastAPI origin so the dashboard works in a real browser:
// same-origin at /ui/ (uvicorn), file://, VS Code Live Server, Cursor preview.
// Override with window.SP_API_BASE if the API is not on 127.0.0.1:8000.
const DEFAULT_API_ORIGIN = "http://127.0.0.1:8000";

function resolveApiBase() {
  if (typeof window.SP_API_BASE === "string") return window.SP_API_BASE;
  const loc = window.location;
  if (!String(loc.protocol).startsWith("http")) return DEFAULT_API_ORIGIN;
  const path = loc.pathname || "";
  const onFastApiDashboard = path === "/" || path === "/ui" || path.startsWith("/ui/");
  const port = loc.port || (loc.protocol === "https:" ? "443" : "80");
  const onFastApiPort = port === "8000";
  if (onFastApiDashboard || onFastApiPort) return "";
  return DEFAULT_API_ORIGIN;
}

const API_BASE = resolveApiBase();

const form = document.getElementById("shipment-form");
const predictionSection = document.getElementById("prediction-section");
const predictionSummary = document.getElementById("prediction-summary");
const optionsSection = document.getElementById("options-section");
const optionsCards = document.getElementById("options-cards");
const roiSummaryEl = document.getElementById("roi-summary");
const costAccuracyEl = document.getElementById("cost-accuracy-summary");
const decisionsTbody = document.querySelector("#decisions-table tbody");
const scenarioButtons = document.getElementById("scenario-buttons");

let lastPrescription = null; // stashed so "Execute Decision" has everything it needs to POST

// One-click stories for live demos — fill the form, then auto-prescribe.
const DEMO_SCENARIOS = [
  {
    id: "safe",
    label: "Demo A · Reliable / off-peak",
    blurb: "Low risk baseline",
    values: {
      sku: "SENSOR-IR",
      supplier: "Meridian Fasteners",
      origin_region: "North America",
      distance_km: 2400,
      historical_avg_lead_time_days: 9,
      order_quantity: 4000,
      unit_cost_usd: 8.5,
      is_peak_season: false,
      budget_cap_usd: 45000,
      max_acceptable_delay_days: 5,
    },
  },
  {
    id: "peak",
    label: "Demo B · Same supplier / peak",
    blurb: "Only seasonality changes",
    values: {
      sku: "SENSOR-IR",
      supplier: "Meridian Fasteners",
      origin_region: "North America",
      distance_km: 2400,
      historical_avg_lead_time_days: 9,
      order_quantity: 4000,
      unit_cost_usd: 8.5,
      is_peak_season: true,
      budget_cap_usd: 45000,
      max_acceptable_delay_days: 5,
    },
  },
  {
    id: "risky",
    label: "Demo C · Risky supplier / peak",
    blurb: "Highest delay risk",
    values: {
      sku: "MICROCHIP-A2",
      supplier: "Delta Cove Electronics",
      origin_region: "Asia Pacific",
      distance_km: 9500,
      historical_avg_lead_time_days: 18,
      order_quantity: 6000,
      unit_cost_usd: 14.2,
      is_peak_season: true,
      budget_cap_usd: 95000,
      max_acceptable_delay_days: 5,
    },
  },
];

function fillForm(values) {
  for (const [key, value] of Object.entries(values)) {
    const el = form.elements[key];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value;
  }
}

async function runPrescription() {
  const data = new FormData(form);
  const shipment = {
    sku: data.get("sku"),
    supplier: data.get("supplier"),
    origin_region: data.get("origin_region"),
    distance_km: Number(data.get("distance_km")),
    historical_avg_lead_time_days: Number(data.get("historical_avg_lead_time_days")),
    order_quantity: Number(data.get("order_quantity")),
    unit_cost_usd: Number(data.get("unit_cost_usd")),
    is_peak_season: form.elements["is_peak_season"].checked,
  };
  const budgetCap = Number(data.get("budget_cap_usd")) || undefined;
  const maxDelay = Number(data.get("max_acceptable_delay_days")) || undefined;

  const resp = await fetch(`${API_BASE}/prescribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ shipment, budget_cap_usd: budgetCap, max_acceptable_delay_days: maxDelay }),
  });

  if (!resp.ok) {
    alert(`Prescription request failed: ${resp.status}. Is the API running?`);
    return;
  }
  const body = await resp.json();
  lastPrescription = body;
  renderPrediction(body);
  renderOptions(body);
}

function renderScenarios() {
  scenarioButtons.innerHTML = "";
  DEMO_SCENARIOS.forEach((scenario) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "scenario-btn";
    btn.innerHTML = `<strong>${scenario.label}</strong><span>${scenario.blurb}</span>`;
    btn.addEventListener("click", async () => {
      fillForm(scenario.values);
      await runPrescription();
      predictionSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    scenarioButtons.appendChild(btn);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrescription();
});

function renderPrediction(body) {
  const { predicted_delay_days, predicted_delay_probability } = body.prediction;
  const pct = Math.round(predicted_delay_probability * 100);
  predictionSummary.innerHTML = `
    <p><strong>${body.shipment_sku}</strong> — expected delay
      <span class="big-num">${predicted_delay_days}</span> day(s)</p>
    <p>Chance of a meaningful delay (&gt; 3 days):
      <span class="big-num">${pct}%</span></p>
    <div class="risk-bar" aria-hidden="true">
      <div class="risk-fill" style="width:${pct}%"></div>
    </div>
    <p class="muted">Budget cap $${body.budget_cap_usd.toLocaleString()}</p>
  `;
  predictionSection.hidden = false;
}

function renderOptions(body) {
  optionsCards.innerHTML = "";
  body.options.forEach((opt) => {
    const card = document.createElement("div");
    const overBudget = !opt.within_budget;
    const overSla = opt.within_sla === false;
    card.className = "option-card" + (overBudget || overSla ? " over-budget" : "");
    const budgetTag = `<span class="tag ${opt.within_budget ? "ok" : "over"}">${
      opt.within_budget ? "within budget" : "over budget"
    }</span>`;
    const slaTag =
      opt.within_sla == null
        ? ""
        : `<span class="tag ${opt.within_sla ? "ok" : "over"}">${
            opt.within_sla ? "within SLA" : "over SLA"
          }</span>`;
    const statusNote =
      opt.solver_status && opt.solver_status !== "Optimal"
        ? `<p class="metric muted">Solver: ${opt.solver_status}</p>`
        : "";
    card.innerHTML = `
      <h3>${opt.label}</h3>
      ${budgetTag}
      ${slaTag}
      <p class="desc">${opt.description}</p>
      <p class="metric"><strong>$${Number(opt.cost_usd).toLocaleString()}</strong> total cost</p>
      <p class="metric">${opt.resulting_delay_days} day(s) resulting delay</p>
      ${statusNote}
      <button data-label="${opt.label}" ${overBudget && opt.cost_usd === 0 ? "disabled" : ""}>Execute decision</button>
    `;
    const btn = card.querySelector("button");
    if (!btn.disabled) {
      btn.addEventListener("click", () => executeDecision(opt.label));
    }
    optionsCards.appendChild(card);
  });
  optionsSection.hidden = false;
}

async function executeDecision(label) {
  if (!lastPrescription) return;
  const formData = new FormData(form);
  const shipmentFeatures = {
    sku: formData.get("sku"),
    supplier: formData.get("supplier"),
    origin_region: formData.get("origin_region"),
    distance_km: Number(formData.get("distance_km")),
    historical_avg_lead_time_days: Number(formData.get("historical_avg_lead_time_days")),
    order_quantity: Number(formData.get("order_quantity")),
    unit_cost_usd: Number(formData.get("unit_cost_usd")),
    is_peak_season: form.elements["is_peak_season"].checked,
  };
  const payload = {
    shipment_sku: lastPrescription.shipment_sku,
    predicted_delay_days: lastPrescription.prediction.predicted_delay_days,
    predicted_delay_probability: lastPrescription.prediction.predicted_delay_probability,
    options: lastPrescription.options,
    chosen_option_label: label,
    budget_cap_usd: lastPrescription.budget_cap_usd,
    shipment_features: shipmentFeatures,
    no_action_cost_usd: lastPrescription.no_action_cost_usd,
  };
  const resp = await fetch(`${API_BASE}/decisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    alert(`Could not save the decision: ${resp.status}`);
    return;
  }
  await loadDecisions();
  document.getElementById("roi-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadDecisions() {
  const [roiResp, accuracyResp, decisionsResp] = await Promise.all([
    fetch(`${API_BASE}/decisions/roi`),
    fetch(`${API_BASE}/decisions/cost-accuracy`),
    fetch(`${API_BASE}/decisions`),
  ]);
  if (roiResp.ok) renderRoi(await roiResp.json());
  if (accuracyResp.ok) renderCostAccuracy(await accuracyResp.json());
  if (decisionsResp.ok) renderDecisions(await decisionsResp.json());
}

function renderRoi(roi) {
  roiSummaryEl.innerHTML = `
    <div><strong>${roi.decisions_with_counterfactual ?? 0}</strong>with no-action baseline</div>
    <div><strong>${roi.avg_avoided_loss_usd != null ? "$" + Number(roi.avg_avoided_loss_usd).toLocaleString() : "—"}</strong>avg avoided loss</div>
    <div><strong>${roi.avg_roi_pct ?? "—"}${roi.avg_roi_pct != null ? "%" : ""}</strong>avg ROI vs no action</div>
    <div><strong>${roi.interventions_beating_no_action_pct ?? "—"}${roi.interventions_beating_no_action_pct != null ? "%" : ""}</strong>beat doing nothing</div>
  `;
}

function renderCostAccuracy(summary) {
  if (!costAccuracyEl) return;
  costAccuracyEl.innerHTML = `
    <div><strong>${summary.total_decisions}</strong>decisions logged</div>
    <div><strong>${summary.resolved_decisions}</strong>outcomes recorded</div>
    <div><strong>${summary.avg_cost_error_pct ?? "—"}${summary.avg_cost_error_pct != null ? "%" : ""}</strong>avg cost prediction error</div>
    <div><strong>${summary.decisions_within_budget_pct ?? "—"}${summary.decisions_within_budget_pct != null ? "%" : ""}</strong>ended up within budget</div>
  `;
}

function renderDecisions(decisions) {
  decisionsTbody.innerHTML = "";
  decisions.forEach((d) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${d.id}</td>
      <td>${d.shipment_sku}</td>
      <td>${d.chosen_option_label}</td>
      <td>$${d.predicted_cost_usd.toLocaleString()}</td>
      <td>${d.no_action_cost_usd != null ? "$" + Number(d.no_action_cost_usd).toLocaleString() : "—"}</td>
      <td>${d.actual_cost_usd != null ? "$" + d.actual_cost_usd.toLocaleString() : "—"}</td>
      <td>${d.is_resolved ? "resolved" : "pending"}</td>
      <td></td>
    `;
    if (!d.is_resolved) {
      const cell = row.lastElementChild;
      const btn = document.createElement("button");
      btn.textContent = "Log outcome";
      btn.addEventListener("click", () => logOutcome(d.id, d.predicted_cost_usd));
      cell.appendChild(btn);
    }
    decisionsTbody.appendChild(row);
  });
}

async function logOutcome(decisionId, predictedCost) {
  const suggested = Math.round(predictedCost * 1.08);
  const actualCost = prompt(
    `Actual cost (USD)?\nTip for demos: try ${suggested} (~8% above predicted ${Math.round(predictedCost)})`,
    String(suggested),
  );
  if (actualCost === null) return;
  const actualDelay = prompt("Actual delay (days)?", "2");
  if (actualDelay === null) return;

  const resp = await fetch(`${API_BASE}/decisions/${decisionId}/outcome`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actual_cost_usd: Number(actualCost), actual_delay_days: Number(actualDelay) }),
  });
  if (!resp.ok) {
    alert(`Could not log the outcome: ${resp.status}`);
    return;
  }
  await loadDecisions();
}

function money(value) {
  if (value == null) return "—";
  return "$" + Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

async function loadPhase2() {
  const summary = document.getElementById("phase2-summary");
  const tbody = document.querySelector("#phase2-grid tbody");
  if (!summary || !tbody) return;
  try {
    const recommend = await fetch(`${API_BASE}/phase2/recommend`);
    if (!recommend.ok) throw new Error("recommend " + recommend.status);
    const body = await recommend.json();
    const draftResp = await fetch(`${API_BASE}/phase2/draft-decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const draft = draftResp.ok ? await draftResp.json() : null;
    const winner = body.winner_label || "No feasible pure option";
    summary.innerHTML = `
      <div>Demo point <span class="big-num">${money(body.budget_cap_usd)}</span> and <span class="big-num">${body.max_acceptable_delay_days}d</span></div>
      <p>Cheapest feasible pure option: <strong>${winner}</strong> (${money(body.winner_cost_usd)}). Same winner on ${body.same_winner_cells} of ${body.grid_cells} cells. MILP ${body.milp_feasible ? "feasible" : "infeasible"}.</p>
      <p class="muted">Draft preview for ${draft ? draft.shipment_sku : body.sku}: persisted = ${draft ? draft.persisted : "unknown"}. No decision row is inserted.</p>
    `;
    tbody.innerHTML = "";
    body.grid.forEach((cell) => {
      const row = document.createElement("tr");
      const isDemo = cell.budget_cap_usd === body.budget_cap_usd && cell.max_acceptable_delay_days === body.max_acceptable_delay_days;
      if (isDemo) row.className = "demo-cell";
      row.innerHTML = `
        <td>${money(cell.budget_cap_usd)}</td>
        <td>${cell.max_acceptable_delay_days}d</td>
        <td>${cell.winner_label || "No feasible pure option"}</td>
        <td>${money(cell.winner_cost_usd)}</td>
        <td>${cell.milp_feasible ? (cell.winner_label ? "feasible" : "budget relaxed") : "infeasible"}</td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    summary.textContent = "Phase 2 could not load. Start the API from the repo root, then refresh this page. " + err.message;
  }
}

function pct(value) {
  if (value == null) return "—";
  return (Number(value) * 100).toFixed(1) + "%";
}

function setLoopFeedback(text) {
  const note = document.getElementById("loop-feedback");
  if (note) note.textContent = text;
}

async function loadPhase3() {
  const box = document.getElementById("phase3-status");
  const retrainBtn = document.getElementById("phase3-retrain");
  if (!box) return;
  try {
    const resp = await fetch(`${API_BASE}/phase3/status`);
    if (!resp.ok) throw new Error("status " + resp.status);
    const body = await resp.json();
    const limits = body.thresholds;
    if (retrainBtn) retrainBtn.disabled = false;
    const driftLine = body.resolved_decisions === 0
      ? "No resolved outcome yet, so drift cannot be high. Execute a decision, then log its actual cost and delay."
      : (body.should_retrain
        ? "A drift signal is over its threshold. Retrain is available."
        : "Drift is under the limits, so the model stays as it is.");
    box.innerHTML = `
      <div><span class="big-num">${body.resolved_decisions}</span> resolved of ${body.total_decisions} decisions</div>
      <p>Cost error ${pct(body.cost_mape)} (retrain at ${pct(limits.cost_mape)}). Delay MAE ${body.delay_mae == null ? "—" : Number(body.delay_mae).toFixed(2) + "d"} (limit ${limits.delay_mae_days}d). Hard-miss ${pct(body.hard_miss_rate)} (limit ${pct(limits.hard_miss_rate)}). Brier ${body.outcome_brier == null ? "—" : Number(body.outcome_brier).toFixed(3)} (limit ${limits.outcome_brier}).</p>
      <p class="muted">${driftLine} ${body.triggers.length ? "Triggers: " + body.triggers.join(", ") + "." : ""} Outcomes without a feature snapshot can raise drift but cannot become training rows.</p>
    `;
    return body;
  } catch (err) {
    box.textContent = "Closed-loop status could not load. " + err.message;
    return null;
  }
}

async function retrainIfDrift() {
  const box = document.getElementById("phase3-status");
  const button = document.getElementById("phase3-retrain");
  if (button) button.disabled = true;
  setLoopFeedback("Checking drift and refitting only if a signal is over its limit…");
  try {
    const resp = await fetch(`${API_BASE}/phase3/retrain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: false }),
    });
    if (!resp.ok) throw new Error("retrain " + resp.status);
    const body = await resp.json();
    await loadPhase3();
    setLoopFeedback(body.retrained
      ? "Model refit and reloaded from disk."
      : "No refit. " + (body.reason || "Drift is not over the limit."));
  } catch (err) {
    if (box) box.textContent = "Retrain check failed. " + err.message;
    setLoopFeedback("Retrain check failed. " + err.message);
  } finally {
    if (button) button.disabled = false;
  }
}

document.getElementById("refresh-roi").addEventListener("click", async () => {
  setLoopFeedback("Refreshing ROI, cost accuracy, and drift…");
  await Promise.all([loadDecisions(), loadPhase3()]);
  const stamp = new Date().toLocaleTimeString();
  setLoopFeedback("Refreshed at " + stamp + ". Numbers change after you log an outcome.");
});
document.getElementById("phase3-retrain").addEventListener("click", retrainIfDrift);
renderScenarios();
loadDecisions();
loadPhase2();
loadPhase3();
