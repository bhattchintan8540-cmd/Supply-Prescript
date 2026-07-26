// Talks straight to the FastAPI service - swap this if you deploy the
// backend somewhere other than localhost:8000.
const API_BASE = window.SP_API_BASE || "http://localhost:8000";

const form = document.getElementById("shipment-form");
const predictionSection = document.getElementById("prediction-section");
const predictionSummary = document.getElementById("prediction-summary");
const optionsSection = document.getElementById("options-section");
const optionsCards = document.getElementById("options-cards");
const roiSummaryEl = document.getElementById("roi-summary");
const decisionsTbody = document.querySelector("#decisions-table tbody");

let lastPrescription = null; // stashed so "Execute Decision" has everything it needs to POST

form.addEventListener("submit", async (event) => {
  event.preventDefault();
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
    alert(`Prescription request failed: ${resp.status}`);
    return;
  }
  const body = await resp.json();
  lastPrescription = body;
  renderPrediction(body);
  renderOptions(body);
});

function renderPrediction(body) {
  const { predicted_delay_days, predicted_delay_probability } = body.prediction;
  predictionSummary.textContent =
    `${body.shipment_sku}: expected delay of ${predicted_delay_days} day(s), ` +
    `${Math.round(predicted_delay_probability * 100)}% chance it's a meaningful delay ` +
    `(budget cap $${body.budget_cap_usd.toLocaleString()}).`;
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
  const payload = {
    shipment_sku: lastPrescription.shipment_sku,
    predicted_delay_days: lastPrescription.prediction.predicted_delay_days,
    predicted_delay_probability: lastPrescription.prediction.predicted_delay_probability,
    options: lastPrescription.options,
    chosen_option_label: label,
    budget_cap_usd: lastPrescription.budget_cap_usd,
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
}

async function loadDecisions() {
  const [roiResp, decisionsResp] = await Promise.all([
    fetch(`${API_BASE}/decisions/roi`),
    fetch(`${API_BASE}/decisions`),
  ]);
  if (roiResp.ok) renderRoi(await roiResp.json());
  if (decisionsResp.ok) renderDecisions(await decisionsResp.json());
}

function renderRoi(roi) {
  roiSummaryEl.innerHTML = `
    <div><strong>${roi.total_decisions}</strong>decisions logged</div>
    <div><strong>${roi.resolved_decisions}</strong>outcomes recorded</div>
    <div><strong>${roi.avg_cost_error_pct ?? "—"}${roi.avg_cost_error_pct != null ? "%" : ""}</strong>avg cost prediction error</div>
    <div><strong>${roi.decisions_within_budget_pct ?? "—"}${roi.decisions_within_budget_pct != null ? "%" : ""}</strong>ended up within budget</div>
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
      <td>${d.actual_cost_usd != null ? "$" + d.actual_cost_usd.toLocaleString() : "—"}</td>
      <td>${d.is_resolved ? "resolved" : "pending"}</td>
      <td></td>
    `;
    if (!d.is_resolved) {
      const cell = row.lastElementChild;
      const btn = document.createElement("button");
      btn.textContent = "Log outcome";
      btn.addEventListener("click", () => logOutcome(d.id));
      cell.appendChild(btn);
    }
    decisionsTbody.appendChild(row);
  });
}

async function logOutcome(decisionId) {
  const actualCost = prompt("Actual cost (USD)?");
  const actualDelay = prompt("Actual delay (days)?");
  if (actualCost === null || actualDelay === null) return;

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
loadDecisions();
