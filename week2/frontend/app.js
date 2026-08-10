// Same-origin when the dashboard is served by FastAPI at `/`.
// Falls back to localhost:8000 when the HTML is opened as a local file.
const API_BASE = window.SP_API_BASE || (window.location.protocol.startsWith("http")
  ? ""
  : "http://localhost:8000");

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
    card.className = "option-card" + (opt.within_budget ? "" : " over-budget");
    card.innerHTML = `
      <h3>${opt.label}</h3>
      <span class="tag ${opt.within_budget ? "ok" : "over"}">
        ${opt.within_budget ? "within budget" : "over budget"}
      </span>
      <p class="desc">${opt.description}</p>
      <p class="metric"><strong>$${opt.cost_usd.toLocaleString()}</strong> total cost</p>
      <p class="metric">${opt.resulting_delay_days} day(s) resulting delay</p>
      <button data-label="${opt.label}">Execute decision</button>
    `;
    card.querySelector("button").addEventListener("click", () => executeDecision(opt.label));
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

document.getElementById("refresh-roi").addEventListener("click", loadDecisions);
renderScenarios();
loadDecisions();
