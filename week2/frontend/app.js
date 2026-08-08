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
const decisionsTbody = document.querySelector("#decisions-table tbody");
const scenarioButtons = document.getElementById("scenario-buttons");
const datasetSummaryEl = document.getElementById("dataset-summary");
const modelMetricsEl = document.getElementById("model-metrics");
const samplesTbody = document.querySelector("#samples-table tbody");
const headerBadge = document.getElementById("header-dataset-badge");

let lastPrescription = null;

// Fallback demos if /dataset/demos is empty (offline / no extract yet).
const FALLBACK_SCENARIOS = [
  {
    id: "safe",
    label: "Demo A · Trinity Biotech",
    blurb: "Low historical delay (Europe)",
    values: {
      sku: "HRDT-UNI-GOLD-HIV-1-2",
      supplier: "Trinity Biotech, Plc",
      origin_region: "Europe",
      distance_km: 6200,
      historical_avg_lead_time_days: 78,
      order_quantity: 400,
      unit_cost_usd: 1.6,
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
      sku: "HRDT-UNI-GOLD-HIV-1-2",
      supplier: "Trinity Biotech, Plc",
      origin_region: "Europe",
      distance_km: 6200,
      historical_avg_lead_time_days: 78,
      order_quantity: 400,
      unit_cost_usd: 1.6,
      is_peak_season: true,
      budget_cap_usd: 45000,
      max_acceptable_delay_days: 5,
    },
  },
  {
    id: "risky",
    label: "Demo C · CIPLA / Asia peak",
    blurb: "Higher delay-risk corridor",
    values: {
      sku: "ARV-GENERIC-TENOFOVIR-DISOPROXIL-FUMARAT",
      supplier: "CIPLA LIMITED",
      origin_region: "Asia Pacific",
      distance_km: 8450,
      historical_avg_lead_time_days: 128,
      order_quantity: 20000,
      unit_cost_usd: 0.12,
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

function statCard(label, value, detail = "") {
  return `
    <div class="stat-card">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
      ${detail ? `<div class="detail">${detail}</div>` : ""}
    </div>
  `;
}

async function loadDatasetPanel() {
  const [summaryResp, modelResp, demosResp, samplesResp] = await Promise.all([
    fetch(`${API_BASE}/dataset/summary`),
    fetch(`${API_BASE}/model/info`),
    fetch(`${API_BASE}/dataset/demos`),
    fetch(`${API_BASE}/dataset/samples?limit=12`),
  ]);

  if (summaryResp.ok) {
    const summary = await summaryResp.json();
    renderDatasetSummary(summary);
  } else {
    datasetSummaryEl.innerHTML = `<div class="sources-list">${statCard("Dataset", "Unavailable", "Is the API running?")}</div>`;
    headerBadge.textContent = "Dataset unavailable";
  }

  if (modelResp.ok) {
    renderModelMetrics(await modelResp.json());
  }

  let scenarios = FALLBACK_SCENARIOS;
  if (demosResp.ok) {
    const live = await demosResp.json();
    if (Array.isArray(live) && live.length) scenarios = live;
  }
  renderScenarios(scenarios);

  if (samplesResp.ok) {
    renderSamples(await samplesResp.json());
  }
}

function renderDatasetSummary(summary) {
  if (!summary.available) {
    headerBadge.textContent = "No dataset loaded — run ingest_real_data.py";
    datasetSummaryEl.innerHTML = `
      <div class="sources-list">
        <strong>${summary.message}</strong>
        <p class="hint" style="margin:0.4rem 0 0">Then refresh this page.</p>
      </div>`;
    return;
  }

  const sourceNames = (summary.sources || []).map((s) => s.label).join(" · ") || "Training extract";
  headerBadge.textContent = `Live on ${sourceNames} · ${summary.n_rows.toLocaleString()} rows`;

    datasetSummaryEl.innerHTML = [
    statCard("Shipments", summary.n_rows.toLocaleString(), summary.message),
    statCard("Suppliers", summary.n_suppliers.toLocaleString()),
    statCard("SKUs", summary.n_skus.toLocaleString()),
    statCard("Regions", summary.n_regions.toLocaleString(), (summary.regions || []).join(", ")),
    statCard("Late > 3d", summary.late_rate_pct != null ? `${summary.late_rate_pct}%` : "—"),
    statCard("Mean delay", summary.mean_delay_days != null ? `${summary.mean_delay_days}d` : "—",
      summary.median_delay_days != null ? `median ${summary.median_delay_days}d` : ""),
    `<div class="sources-list"><strong>Loaded source</strong><ul>${
      (summary.sources || [])
        .map((s) => `<li>${s.label} (${Number(s.n_rows).toLocaleString()} rows)</li>`)
        .join("")
    }</ul><strong style="display:block;margin-top:0.55rem">Dataset packages in repo</strong><ul>${
      (summary.files || [])
        .map((f) => `<li>${f.label}: <code>${f.path || f.key}</code> (${Number(f.n_rows).toLocaleString()} rows)</li>`)
        .join("") || "<li>Run ingest to populate datasets/</li>"
    }</ul></div>`,
  ].join("");
}

function renderModelMetrics(info) {
  if (!info || !info.model_loaded) {
    modelMetricsEl.innerHTML = statCard("Model", "Not loaded", "Run week1/train_model.py");
    return;
  }
  modelMetricsEl.innerHTML = [
    statCard("Model", "Ready", info.dataset_message || "Trained on real extract"),
    statCard("Train rows", info.n_train != null ? Number(info.n_train).toLocaleString() : "—",
      info.n_test != null ? `held out ${Number(info.n_test).toLocaleString()}` : ""),
    statCard("MAE", info.mae_days != null ? `${info.mae_days}d` : "—"),
    statCard("AUC", info.auc != null ? info.auc : "—"),
    info.top_features && info.top_features.length
      ? `<div class="sources-list"><strong>Top features</strong><ul>${
          info.top_features.slice(0, 5).map((f) => `<li>${f.feature} <span class="detail">(${f.importance})</span></li>`).join("")
        }</ul></div>`
      : "",
  ].join("");
}

function renderSamples(rows) {
  samplesTbody.innerHTML = "";
  if (!rows.length) {
    samplesTbody.innerHTML = `<tr><td colspan="7">No sample rows — ingest the real dataset first.</td></tr>`;
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.supplier}</td>
      <td>${row.sku}</td>
      <td>${row.origin_region}</td>
      <td>${Number(row.order_quantity).toLocaleString()}</td>
      <td>$${Number(row.unit_cost_usd).toFixed(2)}</td>
      <td>${row.actual_delay_days}d</td>
      <td></td>
    `;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "secondary";
    btn.textContent = "Run model";
    btn.addEventListener("click", async () => {
      fillForm({
        ...row,
        budget_cap_usd: form.elements.budget_cap_usd.value || 95000,
        max_acceptable_delay_days: form.elements.max_acceptable_delay_days.value || 5,
      });
      await runPrescription();
      predictionSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    tr.lastElementChild.appendChild(btn);
    samplesTbody.appendChild(tr);
  });
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
    alert(`Prescription request failed: ${resp.status}. Is the API running with a trained model?`);
    return;
  }
  const body = await resp.json();
  lastPrescription = body;
  renderPrediction(body);
  renderOptions(body);
}

function renderScenarios(scenarios) {
  scenarioButtons.innerHTML = "";
  scenarios.forEach((scenario) => {
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
    <p class="muted">Budget cap $${body.budget_cap_usd.toLocaleString()} · model trained on the real open extract above</p>
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
  document.getElementById("roi-section").scrollIntoView({ behavior: "smooth", block: "start" });
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
loadDatasetPanel();
loadDecisions();
