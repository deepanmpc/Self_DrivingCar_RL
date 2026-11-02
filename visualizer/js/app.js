import { NetworkRenderer } from "./renderer.js";
import { checkValidation, formatValue, infer } from "./inference.js";
import {
  actionLabels,
  loadCatalog,
  loadHealth,
  loadLiveModel,
  loadModelFile,
  loadModelFromUpload,
} from "./model.js";
import { LiveClient } from "./live.js";

const $ = (id) => document.getElementById(id);

const PRESETS = [
  {
    id: "cruise",
    label: "Centered cruise",
    values: {
      speed: 0.35,
      lane_offset: 0,
      heading_error: 0,
      left_boundary_distance: 0.7,
      right_boundary_distance: 0.7,
      front_obstacle_distance: 1,
      road_curvature: 0,
      previous_steering: 0,
    },
  },
  {
    id: "drift-right",
    label: "Drift right",
    values: {
      speed: 0.3,
      lane_offset: 0.65,
      heading_error: -0.35,
      left_boundary_distance: 0.9,
      right_boundary_distance: 0.3,
      front_obstacle_distance: 1,
      road_curvature: 0,
      previous_steering: -0.1,
    },
  },
  {
    id: "drift-left",
    label: "Drift left",
    values: {
      speed: 0.3,
      lane_offset: -0.65,
      heading_error: 0.35,
      left_boundary_distance: 0.3,
      right_boundary_distance: 0.9,
      front_obstacle_distance: 1,
      road_curvature: 0,
      previous_steering: 0.1,
    },
  },
  {
    id: "curve",
    label: "Left curve",
    values: {
      speed: 0.4,
      lane_offset: 0.1,
      heading_error: 0.05,
      left_boundary_distance: 0.6,
      right_boundary_distance: 0.6,
      front_obstacle_distance: 1,
      road_curvature: 0.7,
      previous_steering: 0,
    },
  },
  {
    id: "obstacle",
    label: "Obstacle ahead",
    values: {
      speed: 0.5,
      lane_offset: 0,
      heading_error: 0,
      left_boundary_distance: 0.7,
      right_boundary_distance: 0.7,
      front_obstacle_distance: 0.2,
      road_curvature: 0,
      previous_steering: 0,
    },
  },
  {
    id: "edge",
    label: "Road edge",
    values: {
      speed: 0.4,
      lane_offset: 0.9,
      heading_error: -0.4,
      left_boundary_distance: 0.85,
      right_boundary_distance: 0.15,
      front_obstacle_distance: 1,
      road_curvature: 0,
      previous_steering: -0.3,
    },
  },
];

const NETWORK_LABELS = {
  actor: "Actor",
  critic: "Critic",
};

const state = {
  mode: "explore",
  liveAvailable: false,
  health: null,
  liveStatus: "idle",
  liveVersion: null,
  liveState: null,
  paused: false,
  driveMode: "baseline",
  catalog: [],
  catalogIndex: -1,
  uploadName: null,
  model: null,
  exploreModel: null,
  networkName: "actor",
  inputs: [],
  actorResult: null,
  criticResult: null,
  result: null,
  selection: null,
  validation: null,
  inputEls: [],
  actionEls: [],
  liveClient: null,
  manual: { left: false, right: false, forward: false, back: false },
  manualTimer: null,
  toastTimer: null,
  lastLiveError: null,
  lastLengthWarning: null,
  lastServerError: null,
  loadToken: 0,
};

const renderer = new NetworkRenderer($("networkCanvas"), {
  onHover: showTooltip,
  onSelect: handleSelect,
  onEdgeCount: (visible, total) => {
    $("edgeCount").textContent = `${visible.toLocaleString()} of ${total.toLocaleString()} edges`;
  },
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return entities[character];
  });
}

function prettify(label) {
  return String(label ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function signed(value, digits = 3) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  const text = formatValue(Math.abs(value), digits);
  return value >= 0 ? `+${text}` : `-${text}`;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function showToast(message, kind = "info") {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast ${kind === "error" ? "error" : ""}`;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => {
    toast.classList.add("hidden");
  }, kind === "error" ? 6500 : 3500);
}

function defaultInputs(model) {
  const features = model.observation.features;
  const preset = PRESETS[0].values;
  return features.map((feature, index) => {
    const low = Number(feature.range?.[0] ?? -1);
    const high = Number(feature.range?.[1] ?? 1);
    const value = preset[feature.key];
    if (Number.isFinite(value)) {
      return clamp(value, low, high);
    }
    return (low + high) / 2;
  });
}

function physicalText(feature, normalized) {
  const normalizedRange = feature.range ?? [-1, 1];
  const physicalRange = feature.physicalRange ?? normalizedRange;
  const span = normalizedRange[1] - normalizedRange[0] || 1;
  const ratio = (normalized - normalizedRange[0]) / span;
  const physical = physicalRange[0] + ratio * (physicalRange[1] - physicalRange[0]);
  const unit = feature.unit ? ` ${feature.unit}` : "";

  if (feature.key === "front_obstacle_distance" && normalized >= 0.999) {
    return "clear";
  }
  if (feature.key === "heading_error") {
    return `${formatValue(physical, 2)} rad (${formatValue((physical * 180) / Math.PI, 1)}°)`;
  }
  if (feature.key === "road_curvature") {
    return `${formatValue(physical, 3)} 1/m`;
  }
  if (feature.key === "previous_steering") {
    return formatValue(physical, 2);
  }
  return `${formatValue(physical, 1)}${unit}`;
}

function updateInputDisplay(index) {
  const entry = state.inputEls[index];
  if (!entry) {
    return;
  }
  const normalized = state.inputs[index];
  entry.value.textContent = formatValue(normalized, 2);
  entry.physical.textContent = physicalText(entry.feature, normalized);
  if (entry.slider.value !== String(normalized)) {
    entry.slider.value = String(normalized);
  }
}

function setupInputs() {
  if (!state.model) {
    return;
  }
  const container = $("inputControls");
  container.innerHTML = "";
  state.inputEls = [];

  state.model.observation.features.forEach((feature, index) => {
    const row = document.createElement("div");
    row.className = "input-row";
    const low = Number(feature.range?.[0] ?? -1);
    const high = Number(feature.range?.[1] ?? 1);
    const value = Number(state.inputs[index] ?? (low + high) / 2);

    const head = document.createElement("div");
    head.className = "input-head";
    const label = document.createElement("span");
    label.className = "input-label";
    label.textContent = feature.label;
    label.title = feature.description ?? feature.key;
    const valueWrap = document.createElement("span");
    valueWrap.className = "input-value";
    const numeric = document.createElement("span");
    const physical = document.createElement("span");
    physical.className = "input-physical";
    valueWrap.append(numeric, document.createTextNode(" · "), physical);
    head.append(label, valueWrap);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(low);
    slider.max = String(high);
    slider.step = "0.01";
    slider.value = String(value);
    slider.setAttribute("aria-label", feature.label);
    slider.addEventListener("input", () => {
      state.inputs[index] = Number(slider.value);
      updateInputDisplay(index);
      runInference();
    });

    row.append(head, slider);
    container.append(row);
    state.inputEls.push({ slider, value: numeric, physical, feature, row });
    updateInputDisplay(index);
  });

  renderPresets();
  updateInputEditability();
}

function renderPresets() {
  const container = $("presets");
  container.innerHTML = "";
  const keys = new Set(state.model?.observation.features.map((feature) => feature.key) ?? []);
  for (const preset of PRESETS) {
    if (!Object.keys(preset.values).some((key) => keys.has(key))) {
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset";
    button.textContent = preset.label;
    button.disabled = state.mode === "live";
    button.addEventListener("click", () => applyPreset(preset));
    container.append(button);
  }
}

function applyPreset(preset) {
  if (state.mode === "live" || !state.model) {
    return;
  }
  state.inputs = state.model.observation.features.map((feature, index) => {
    const low = Number(feature.range?.[0] ?? -1);
    const high = Number(feature.range?.[1] ?? 1);
    const presetValue = preset.values[feature.key];
    if (Number.isFinite(presetValue)) {
      return clamp(presetValue, low, high);
    }
    return Number(state.inputs[index] ?? (low + high) / 2);
  });
  state.inputEls.forEach((_, index) => updateInputDisplay(index));
  runInference();
}

function updateInputEditability() {
  const live = state.mode === "live";
  $("inputMode").textContent = live ? "Live stream" : "Editable";
  state.inputEls.forEach((entry) => {
    entry.slider.disabled = live;
    entry.row.classList.toggle("disabled", live);
  });
  document.querySelectorAll(".preset").forEach((button) => {
    button.disabled = live;
  });
  ["modelSelect", "modelFile"].forEach((id) => {
    $(id).disabled = live;
  });
}

function setupNetworkTabs() {
  const container = $("networkTabs");
  container.innerHTML = "";
  const names = Object.keys(state.model?.networks ?? {});
  if (!names.includes(state.networkName)) {
    state.networkName = names.includes("actor") ? "actor" : names[0];
  }
  names.forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `network-tab ${name === state.networkName ? "active" : ""}`;
    button.textContent = NETWORK_LABELS[name] ?? prettify(name);
    button.addEventListener("click", () => switchNetwork(name));
    container.append(button);
  });
}

function switchNetwork(name) {
  if (!state.model?.networks[name] || state.networkName === name) {
    return;
  }
  state.networkName = name;
  state.selection = null;
  document.querySelectorAll(".network-tab").forEach((button) => {
    button.classList.toggle("active", button.textContent === (NETWORK_LABELS[name] ?? prettify(name)));
  });
  renderer.setNetwork(name);
  runInference();
  renderer.fit();
  renderInspector(null);
  hideTooltip();
}

function runInference() {
  if (!state.model || !state.inputs.length) {
    return;
  }
  try {
    const names = Object.keys(state.model.networks);
    state.actorResult = state.model.networks.actor
      ? infer(state.model, "actor", state.inputs)
      : null;
    state.criticResult = state.model.networks.critic
      ? infer(state.model, "critic", state.inputs)
      : null;
    if (state.networkName === "critic" && state.criticResult) {
      state.result = state.criticResult;
    } else if (state.networkName === "actor" && state.actorResult) {
      state.result = state.actorResult;
    } else {
      state.result =
        state.actorResult ?? state.criticResult ?? infer(state.model, names[0], state.inputs);
      state.networkName = state.result.networkName;
    }
    renderer.setResult(state.result);
    if (state.selection) {
      const info = infoFromSelection(state.selection);
      renderer.selected = info;
      if (info) {
        renderInspector(info);
      } else {
        state.selection = null;
        renderInspector(null);
      }
      renderer.render();
    }
    updateActionBar();
  } catch (error) {
    showToast(`Inference failed: ${error.message}`, "error");
  }
}

function selectionFromInfo(info) {
  if (!info) {
    return null;
  }
  if (info.type === "node") {
    return { type: "node", columnIndex: info.columnIndex, nodeIndex: info.nodeIndex };
  }
  return {
    type: "edge",
    blockIndex: info.edge.blockIndex,
    source: info.edge.source,
    target: info.edge.target,
  };
}

function infoFromSelection(selection) {
  if (!selection || !renderer.columns.length) {
    return null;
  }
  if (selection.type === "node") {
    const column = renderer.columns[selection.columnIndex];
    if (!column || selection.nodeIndex >= column.values.length) {
      return null;
    }
    return {
      type: "node",
      networkName: state.networkName,
      columnIndex: selection.columnIndex,
      nodeIndex: selection.nodeIndex,
      column,
      block: selection.columnIndex > 0 ? renderer.blocks[selection.columnIndex - 1] : null,
      value: column.values[selection.nodeIndex],
    };
  }
  const edge = renderer.allEdges.find(
    (candidate) =>
      candidate.blockIndex === selection.blockIndex &&
      candidate.source === selection.source &&
      candidate.target === selection.target,
  );
  if (!edge) {
    return null;
  }
  return {
    type: "edge",
    networkName: state.networkName,
    edge,
    sourceColumn: renderer.columns[edge.sourceColumn],
    targetColumn: renderer.columns[edge.targetColumn],
  };
}

function handleSelect(info) {
  state.selection = selectionFromInfo(info);
  $("selectionTag").textContent = info
    ? info.type === "node"
      ? "Neuron"
      : "Connection"
    : "Network";
  renderInspector(info);
}

function columnName(index) {
  if (index === 0) {
    return "Input";
  }
  if (index === renderer.columns.length - 1) {
    return state.model?.networks[state.networkName]?.role === "actor" ? "Action" : "Value";
  }
  return `Dense ${renderer.blocks[index - 1]?.linear.out ?? ""}`.trim();
}

function nodeName(columnIndex, nodeIndex) {
  if (columnIndex === 0) {
    return state.model?.observation.features[nodeIndex]?.label ?? `Input ${nodeIndex + 1}`;
  }
  if (columnIndex === renderer.columns.length - 1) {
    const labels = state.model?.networks[state.networkName]?.output?.labels ?? [];
    return labels[nodeIndex] ? prettify(labels[nodeIndex]) : `Output ${nodeIndex + 1}`;
  }
  return `Neuron ${nodeIndex + 1}`;
}

function miniTable(title, headers, rows) {
  if (!rows.length) {
    return "";
  }
  return `
    <div class="section-heading" style="margin-top:12px;margin-bottom:4px">
      <h2>${escapeHtml(title)}</h2>
    </div>
    <table class="value-table">
      <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

function renderNodeInspector(info) {
  const { column, columnIndex, nodeIndex } = info;
  const block = info.block;
  const feature = columnIndex === 0 ? state.model.observation.features[nodeIndex] : null;
  const title = feature
    ? `${feature.label}`
    : `${column.label} · ${nodeName(columnIndex, nodeIndex)}`;
  const subtitle = feature
    ? `Input ${nodeIndex + 1} · ${physicalText(feature, column.values[nodeIndex])}`
    : `Index ${nodeIndex + 1} of ${column.values.length}`;

  const rows = [];
  if (block) {
    rows.push(
      `<tr><td>Linear z</td><td>${signed(block.pre[nodeIndex], 4)}</td></tr>`,
    );
    if (block.normed) {
      rows.push(
        `<tr><td>LayerNorm</td><td>${signed(block.normed[nodeIndex], 4)}</td></tr>`,
      );
    }
    if (block.activation) {
      rows.push(
        `<tr><td>${block.activation.toUpperCase()}</td><td>${formatValue(block.values[nodeIndex], 4)}</td></tr>`,
      );
    }
    if (block.linear.bias) {
      rows.push(
        `<tr><td>Bias</td><td>${signed(block.linear.bias[nodeIndex], 4)}</td></tr>`,
      );
    }
  } else {
    rows.push(
      `<tr><td>Activation</td><td>${formatValue(column.values[nodeIndex], 4)}</td></tr>`,
    );
  }

  let incoming = [];
  if (block) {
    const sourceValues = block.source;
    const row = block.linear.weight[nodeIndex];
    const incomingItems = row.map((weight, source) => ({
      source,
      weight,
      contribution: weight * sourceValues[source],
    }));
    incomingItems.sort(
      (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution),
    );
    incoming = incomingItems.slice(0, 6).map((item) => {
      const sourceLabel =
        columnIndex === 1
          ? state.model.observation.features[item.source]?.label ?? `In ${item.source + 1}`
          : `${columnName(columnIndex - 1)} ${item.source + 1}`;
      return `<tr><td>${escapeHtml(sourceLabel)}</td><td>${signed(item.weight, 4)}</td><td>${signed(item.contribution, 4)}</td></tr>`;
    });
  }

  let outgoing = [];
  if (columnIndex < renderer.blocks.length) {
    const nextBlock = renderer.blocks[columnIndex];
    const nextColumnIndex = columnIndex + 1;
    const items = nextBlock.linear.weight.map((row, target) => ({
      target,
      weight: row[nodeIndex],
      contribution: row[nodeIndex] * column.values[nodeIndex],
    }));
    items.sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution));
    outgoing = items.slice(0, 6).map((item) => {
      const targetLabel =
        nextColumnIndex === renderer.columns.length - 1
          ? nodeName(nextColumnIndex, item.target)
          : `${columnName(nextColumnIndex)} ${item.target + 1}`;
      return `<tr><td>${escapeHtml(targetLabel)}</td><td>${signed(item.weight, 4)}</td><td>${signed(item.contribution, 4)}</td></tr>`;
    });
  }

  $("inspector").innerHTML = `
    <div class="inspector-title">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(subtitle)}</span>
    </div>
    <table class="value-table">
      <thead><tr><th>Stage</th><th>Value</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
    ${miniTable("Top incoming", ["Source", "Weight", "Contribution"], incoming)}
    ${miniTable("Top outgoing", ["Target", "Weight", "Contribution"], outgoing)}
  `;
}

function renderEdgeInspector(info) {
  const { edge, sourceColumn, targetColumn } = info;
  const sourceActivation = sourceColumn.values[edge.source];
  const bias = edge.bias ?? 0;
  const rows = [
    ["Weight", signed(edge.weight, 4)],
    ["Source activation", signed(sourceActivation, 4)],
    ["Contribution", signed(edge.contribution, 4)],
    ["Target bias", signed(bias, 4)],
  ];
  $("inspector").innerHTML = `
    <div class="inspector-title">
      <strong>${escapeHtml(columnName(info.edge.sourceColumn))} ${edge.source + 1} → ${escapeHtml(columnName(info.edge.targetColumn))} ${edge.target + 1}</strong>
      <span>Connection</span>
    </div>
    <table class="value-table">
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>${rows.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join("")}</tbody>
    </table>
    <p style="margin:10px 0 0;line-height:1.5">Line brightness shows the signed product of weight and source activation. Click a neuron to inspect all of its strongest connections.</p>
  `;
}

function renderModelSummary() {
  if (!state.model) {
    $("inspector").innerHTML =
      '<div class="inspector-placeholder"><p>No model loaded. Export one and reload.</p></div>';
    return;
  }
  const metadata = state.model.metadata ?? {};
  const validation = state.validation;
  const network = state.model.networks[state.networkName];
  const architecture = network
    ? network.layers
        .filter((layer) => layer.kind === "linear")
        .map((layer, index, array) => (index === 0 ? layer.in : layer.out))
        .join(" → ")
    : "";
  const validationText = validation
    ? validation.checked
      ? validation.ok
        ? "Passed"
        : "Failed"
      : "No samples"
    : "Not checked";
  const validationClass = validation?.checked
    ? validation.ok
      ? "ok"
      : "warn"
    : "";

  $("inspector").innerHTML = `
    <div class="inspector-placeholder">
      <p>Hover a neuron or edge for details. Click to pin values and inspect the strongest incoming and outgoing weights.</p>
      <div class="stat-grid">
        <div class="stat-card"><span class="stat-label">Parameters</span><span class="stat-value">${Number(metadata.parameters ?? 0).toLocaleString()}</span></div>
        <div class="stat-card"><span class="stat-label">Networks</span><span class="stat-value">${Object.keys(state.model.networks).map((name) => NETWORK_LABELS[name] ?? prettify(name)).join(" + ")}</span></div>
        <div class="stat-card"><span class="stat-label">Validation</span><span class="stat-value ${validationClass}">${validationText}</span></div>
        <div class="stat-card"><span class="stat-label">Format</span><span class="stat-value">v${state.model.formatVersion ?? "?"}</span></div>
      </div>
      <div class="meta-grid" style="margin-top:12px">
        <div class="meta-row"><span>Current network</span><span>${escapeHtml(NETWORK_LABELS[state.networkName] ?? prettify(state.networkName))}</span></div>
        <div class="meta-row"><span>Architecture</span><span>${escapeHtml(architecture)}</span></div>
        <div class="meta-row"><span>Source</span><span>${escapeHtml(metadata.checkpoint ?? (metadata.seed !== null && metadata.seed !== undefined ? `untrained · seed ${metadata.seed}` : "configured network"))}</span></div>
        <div class="meta-row"><span>Created</span><span>${escapeHtml(metadata.createdAt ?? "n/a")}</span></div>
      </div>
      ${
        validation?.checked
          ? `<div class="meta-row" style="margin-top:4px"><span>Max JS error</span><span>${formatValue(validation.maxError, 6)}</span></div>`
          : ""
      }
    </div>
  `;
}

function renderInspector(info) {
  if (!info) {
    $("selectionTag").textContent = "Network";
    renderModelSummary();
    return;
  }
  if (info.type === "node") {
    renderNodeInspector(info);
  } else if (info.type === "edge") {
    renderEdgeInspector(info);
  }
}

function showTooltip(info, event) {
  const tooltip = $("tooltip");
  if (!info || !event) {
    hideTooltip();
    return;
  }
  if (info.type === "node") {
    const title =
      info.columnIndex === 0
        ? state.model.observation.features[info.nodeIndex]?.label
        : `${info.column.label} · ${nodeName(info.columnIndex, info.nodeIndex)}`;
    const block = info.block;
    const extra =
      block && block.activation
        ? `<div>${block.activation.toUpperCase()} <span class="mono">${formatValue(info.value, 4)}</span></div>`
        : `<div>Value <span class="mono">${formatValue(info.value, 4)}</span></div>`;
    tooltip.innerHTML = `<strong>${escapeHtml(title)}</strong>${extra}<div style="color:var(--faint)">Click to inspect</div>`;
  } else {
    const { edge, sourceColumn, targetColumn } = info;
    tooltip.innerHTML = `<strong>${escapeHtml(columnName(edge.sourceColumn))} ${edge.source + 1} → ${escapeHtml(columnName(edge.targetColumn))} ${edge.target + 1}</strong><div>Weight <span class="mono">${signed(edge.weight, 4)}</span></div><div>Contribution <span class="mono">${signed(edge.contribution, 4)}</span></div>`;
  }
  tooltip.classList.remove("hidden");
  const width = 270;
  const left = Math.min(event.x + 14, window.innerWidth - width);
  const top = Math.min(event.y + 14, window.innerHeight - 120);
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip() {
  $("tooltip").classList.add("hidden");
}

function buildActionBar() {
  if (!state.model) {
    $("actionBar").innerHTML = "";
    state.actionEls = [];
    return;
  }
  const actor = state.model.networks.actor;
  const critic = state.model.networks.critic;
  const blocks = [];
  state.actionEls = [];

  if (actor) {
    const labels = actionLabels(state.model);
    labels.forEach((label, index) => {
      blocks.push(`
        <div class="action-block" data-action-index="${index}">
          <div class="action-head">
            <span class="action-name">${escapeHtml(label)}</span>
            <span class="action-number" data-role="number">--</span>
          </div>
          <div class="action-track"><div class="action-fill" data-role="fill"></div></div>
          <div class="action-sub"><span data-role="raw">raw --</span><span data-role="std">σ --</span></div>
        </div>
      `);
      state.actionEls.push({ type: "actor", index });
    });
  }

  if (critic) {
    blocks.push(`
      <div class="action-block critic">
        <div class="action-head">
          <span class="action-name">Critic V(s)</span>
          <span class="action-number" data-role="number">--</span>
        </div>
        <div class="action-track"><div class="action-fill" data-role="fill"></div></div>
        <div class="action-sub"><span data-role="raw">state value</span><span data-role="std">untrained</span></div>
      </div>
    `);
    state.actionEls.push({ type: "critic", index: 0 });
  }

  $("actionBar").innerHTML = blocks.join("");
  const elements = Array.from($("actionBar").querySelectorAll(".action-block"));
  elements.forEach((element, index) => {
    const entry = state.actionEls[index];
    entry.element = element;
    entry.number = element.querySelector('[data-role="number"]');
    entry.fill = element.querySelector('[data-role="fill"]');
    entry.raw = element.querySelector('[data-role="raw"]');
    entry.std = element.querySelector('[data-role="std"]');
  });
}

function setActionBar(fill, value, maximum) {
  const normalized = clamp(value / maximum, -1, 1);
  const width = Math.abs(normalized) * 50;
  fill.style.left = normalized >= 0 ? "50%" : `${50 - width}%`;
  fill.style.width = `${width}%`;
  fill.classList.toggle("negative", normalized < 0);
}

function updateActionBar() {
  if (!state.actionEls.length) {
    return;
  }
  const applied = state.mode === "live" ? state.liveState?.appliedAction : null;
  for (const entry of state.actionEls) {
    if (entry.type === "actor" && state.actorResult) {
      const value = state.actorResult.actor.actions[entry.index];
      const mean = state.actorResult.actor.mean[entry.index];
      const std = state.actorResult.actor.std[entry.index];
      entry.number.textContent = signed(value, 3);
      setActionBar(entry.fill, value, 1);
      const appliedSuffix =
        Array.isArray(applied) && Number.isFinite(applied[entry.index])
          ? ` · last ${signed(applied[entry.index], 3)}`
          : "";
      entry.raw.textContent = `raw ${signed(mean, 3)}${appliedSuffix}`;
      entry.std.textContent = `σ ${formatValue(std, 3)}`;
    } else if (entry.type === "critic" && state.criticResult) {
      const value = state.criticResult.critic.value;
      entry.number.textContent = signed(value, 3);
      setActionBar(entry.fill, value, 5);
      const liveValue = state.mode === "live" ? state.liveState?.criticValue : null;
      entry.raw.textContent = "state value";
      entry.std.textContent = Number.isFinite(liveValue)
        ? `live ${signed(liveValue, 3)}`
        : "untrained";
    }
  }
}

function updateModelMeta() {
  if (!state.model) {
    $("modelMeta").innerHTML = "";
    $("modelStatus").textContent = "--";
    return;
  }
  const metadata = state.model.metadata ?? {};
  const rows = [
    ["Label", metadata.label ?? "unnamed"],
    ["Source", state.mode === "live" ? `Live server${state.liveVersion ? ` · v${state.liveVersion}` : ""}` : "Exported snapshot"],
    ["Parameters", Number(metadata.parameters ?? 0).toLocaleString()],
    ["Networks", Object.keys(state.model.networks).join(", ")],
    [
      "Checkpoint",
      metadata.checkpoint ??
        (metadata.seed !== null && metadata.seed !== undefined ? `random · seed ${metadata.seed}` : "n/a"),
    ],
    ["Created", metadata.createdAt ?? "n/a"],
  ];
  if (state.validation) {
    rows.push([
      "Validation",
      state.validation.checked
        ? `${state.validation.ok ? "passed" : "failed"} · max ${formatValue(state.validation.maxError, 2)}`
        : "not available",
    ]);
  }
  $("modelMeta").innerHTML = rows
    .map(
      ([label, value]) =>
        `<div class="meta-row"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>`,
    )
    .join("");
  $("modelStatus").textContent = state.mode === "live" ? `live v${state.liveVersion ?? "?"}` : "snapshot";
  $("modelStatus").className = `section-note ${state.mode === "live" ? "ok" : ""}`;
}

function populateModelSelect() {
  const select = $("modelSelect");
  select.innerHTML = "";
  if (!state.catalog.length) {
    const option = document.createElement("option");
    option.value = "-1";
    option.textContent = "No exported models";
    select.append(option);
  } else {
    state.catalog.forEach((entry, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${entry.label}${entry.createdAt ? ` · ${entry.createdAt.slice(0, 10)}` : ""}`;
      select.append(option);
    });
  }
  if (state.uploadName) {
    const option = document.createElement("option");
    option.value = "__upload__";
    option.textContent = `Uploaded · ${state.uploadName}`;
    select.append(option);
  }
  if (state.uploadName) {
    select.value = "__upload__";
  } else if (state.catalogIndex >= 0 && state.catalog[state.catalogIndex]) {
    select.value = String(state.catalogIndex);
  } else if (state.catalog.length) {
    select.value = "0";
  }
  select.disabled = state.mode === "live";
}

async function loadCatalogEntry(index, options = {}) {
  const entry = state.catalog[index];
  if (!entry) {
    return false;
  }
  const token = ++state.loadToken;
  try {
    const model = await loadModelFile(entry.url);
    if (token !== state.loadToken) {
      return false;
    }
    state.catalogIndex = index;
    state.uploadName = null;
    applyModel(model, { live: false, source: "snapshot" });
    populateModelSelect();
    if (!options.silent) {
      showToast(`Loaded ${entry.label}`);
    }
    return true;
  } catch (error) {
    if (token === state.loadToken) {
      showToast(`Could not load ${entry.file ?? entry.url}: ${error.message}`, "error");
    }
    return false;
  }
}

function applyModel(model, { live = false, preserveInputs = true } = {}) {
  const previousInputs = state.inputs;
  const previousNetwork = state.networkName;
  state.model = model;
  state.lastLengthWarning = null;
  if (!live) {
    state.exploreModel = model;
  }
  state.validation = checkValidation(model);

  const names = Object.keys(model.networks);
  const desiredNetwork =
    names.includes(previousNetwork) ? previousNetwork : names.includes("actor") ? "actor" : names[0];
  state.networkName = desiredNetwork;

  state.inputs =
    preserveInputs && previousInputs.length === model.observation.features.length
      ? previousInputs.slice()
      : defaultInputs(model);
  state.selection = null;
  renderer.selected = null;

  setupNetworkTabs();
  setupInputs();
  buildActionBar();
  updateModelMeta();
  renderModelSummary();
  updateInputEditability();
  updateModeUI();

  renderer.setModel(model, state.networkName);
  runInference();
  renderer.fit();
  hideTooltip();
}

function showErrorState(title, message) {
  $("emptyOverlay").querySelector("h2").textContent = title;
  $("emptyOverlay").querySelector("p").textContent = message;
  $("emptyOverlay").classList.remove("hidden");
}

function hideErrorState() {
  $("emptyOverlay").classList.add("hidden");
}

function updateModeUI() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.mode);
  });
  $("liveTab").disabled = !state.liveAvailable;
  $("liveBar").classList.toggle("hidden", state.mode !== "live");
  $("telemetrySection").classList.toggle("hidden", state.mode !== "live");
  $("inputMode").textContent = state.mode === "live" ? "Live stream" : "Editable";
  $("modelSelect").disabled = state.mode === "live";
  $("modelFile").disabled = state.mode === "live";
  document.querySelectorAll(".preset").forEach((button) => {
    button.disabled = state.mode === "live";
  });
  state.inputEls.forEach((entry) => {
    entry.slider.disabled = state.mode === "live";
    entry.row.classList.toggle("disabled", state.mode === "live");
  });
  $("manualHint").classList.toggle(
    "hidden",
    !(state.mode === "live" && state.driveMode === "manual"),
  );
}

function updateConnectionStatus(status, detail = {}) {
  const pill = $("connectionStatus");
  state.liveStatus = status;
  if (status === "live") {
    pill.className = "status-pill status-live";
    pill.textContent = `Live · ${state.health?.checkpoint ? "checkpoint" : "untrained"}`;
    state.lastLiveError = null;
  } else if (status === "reconnecting") {
    pill.className = "status-pill";
    pill.textContent = "Live · reconnecting";
  } else if (status === "error") {
    pill.className = "status-pill status-error";
    pill.textContent = "Live · error";
    const message = detail.error ?? "Live stream unavailable";
    if (message !== state.lastLiveError) {
      state.lastLiveError = message;
      showToast(message, "error");
    }
  } else if (status === "unavailable") {
    pill.className = "status-pill status-idle";
    pill.textContent = state.model ? "Static model" : "No live server";
  } else {
    pill.className = "status-pill status-idle";
    pill.textContent = state.model ? "Static model" : "Loading";
  }
}

function handleLiveStatus(status, detail) {
  if (status === "live" && detail?.modelVersion) {
    state.liveVersion = detail.modelVersion;
  }
  updateConnectionStatus(status, detail);
  if (status === "live") {
    state.liveAvailable = true;
    $("liveTab").disabled = false;
  } else if (status === "unavailable") {
    state.liveAvailable = false;
    $("liveTab").disabled = true;
  }
}

function handleLiveModel(model, detail = {}) {
  state.liveVersion = detail?.modelVersion ?? state.liveVersion;
  state.health = {
    ...(state.health ?? {}),
    checkpoint: model.metadata?.checkpoint ?? null,
    label: model.metadata?.label ?? state.health?.label ?? null,
  };
  applyModel(model, { live: true, preserveInputs: true });
  updateConnectionStatus("live");
  hideErrorState();
}

function handleLiveState(snapshot) {
  const previousSeq = state.liveState?.seq;
  state.liveState = snapshot;
  state.paused = Boolean(snapshot.paused);
  if (snapshot.error) {
    if (snapshot.error !== state.lastServerError) {
      state.lastServerError = snapshot.error;
      showToast(snapshot.error, "error");
    }
  } else {
    state.lastServerError = null;
  }
  if (snapshot.modelVersion && snapshot.modelVersion !== state.liveVersion) {
    state.liveVersion = snapshot.modelVersion;
    updateModelMeta();
  }
  if (snapshot.seq !== previousSeq) {
    if (Array.isArray(snapshot.observation)) {
      if (snapshot.observation.length === state.inputs.length) {
        state.inputs = snapshot.observation.map(Number);
        state.inputEls.forEach((_, index) => updateInputDisplay(index));
        runInference();
      } else if (state.model) {
        const message = `Live observation has ${snapshot.observation.length} features, model expects ${state.inputs.length}.`;
        if (message !== state.lastLengthWarning) {
          state.lastLengthWarning = message;
          showToast(message, "error");
        }
      }
    }
    renderTelemetry(snapshot);
  }
  updateActionBar();
  updateLiveMeta(snapshot);
  updateConnectionStatus("live");
}

function updateLiveMeta(snapshot) {
  const fps = Number(snapshot.fps ?? 0);
  const termination = snapshot.terminated || snapshot.truncated ? " · episode ended" : "";
  $("liveMeta").textContent = `ep ${snapshot.episode ?? 0} · step ${snapshot.step ?? 0} · ${fps.toFixed(1)} Hz${termination} · v${state.liveVersion ?? "?"}`;
  $("pauseButton").textContent = state.paused ? "Resume" : "Pause";
  $("stepButton").disabled = !state.paused;
  if (snapshot.mode && state.driveMode !== snapshot.mode && Date.now() - (state.modeChangedAt ?? 0) > 1200) {
    state.driveMode = snapshot.mode;
    $("driveMode").value = snapshot.mode;
    $("manualHint").classList.toggle("hidden", snapshot.mode !== "manual");
    updateManualLoop();
  }
}

function renderTelemetry(snapshot) {
  const components = snapshot.rewardComponents ?? {};
  const entries = Object.entries(components).filter(([key]) => key !== "total_reward");
  const maxComponent =
    entries.reduce((maximum, [, value]) => Math.max(maximum, Math.abs(Number(value))), 0) || 1;
  const rows = entries
    .map(([key, value]) => {
      const numeric = Number(value);
      const signedValue = key.endsWith("_penalty") ? -numeric : numeric;
      const width = Math.min(100, (Math.abs(numeric) / maxComponent) * 100);
      return `
        <div class="reward-row">
          <span>${escapeHtml(prettify(key))}</span>
          <div class="reward-bar"><div class="reward-fill ${signedValue < 0 ? "negative" : ""}" style="width:${width}%"></div></div>
          <span>${signed(signedValue, 3)}</span>
        </div>
      `;
    })
    .join("");

  $("episodeTag").textContent = `ep ${snapshot.episode ?? 0}`;
  $("telemetry").innerHTML = `
    <div class="telemetry-grid">
      <div class="telemetry-card"><span>Reward</span><strong>${signed(Number(snapshot.reward ?? 0), 3)}</strong></div>
      <div class="telemetry-card"><span>Step</span><strong>${snapshot.step ?? 0}</strong></div>
      <div class="telemetry-card"><span>Critic value</span><strong>${signed(Number(snapshot.criticValue ?? 0), 3)}</strong></div>
      <div class="telemetry-card"><span>Sim rate</span><strong>${Number(snapshot.fps ?? 0).toFixed(1)} Hz</strong></div>
    </div>
    <div class="reward-list">${rows || '<span style="color:var(--faint)">No reward components reported.</span>'}</div>
  `;
}

async function switchMode(mode) {
  if (mode === state.mode) {
    return;
  }
  if (mode === "live" && !state.liveAvailable) {
    return;
  }
  stopManualLoop();
  resetManualKeys();
  state.mode = mode;
  state.modeChangedAt = Date.now();
  hideTooltip();
  if (mode === "live") {
    updateModeUI();
    renderer.setTopK(Number($("topKInput").value));
    const client = ensureLiveClient();
    const ok = await client.start();
    if (!ok) {
      state.mode = "explore";
      updateModeUI();
      showToast("Live server is not available. Start it with --live.", "error");
      return;
    }
    updateConnectionStatus("live");
    const health = state.health;
    if (health?.mode) {
      state.driveMode = health.mode;
      $("driveMode").value = health.mode;
    }
    updateModeUI();
    updateManualLoop();
  } else {
    state.liveClient?.stop();
    state.liveState = null;
    state.paused = false;
    state.driveMode = "baseline";
    $("driveMode").value = "baseline";
    updateModeUI();
    updateConnectionStatus("unavailable");
    if (state.exploreModel) {
      applyModel(state.exploreModel, { live: false, preserveInputs: true });
    } else if (state.catalog.length) {
      await loadCatalogEntry(state.catalogIndex >= 0 ? state.catalogIndex : 0);
    } else {
      const model = await loadLiveModel().catch(() => null);
      if (model) {
        applyModel(model, { live: false, preserveInputs: true });
      } else {
        showErrorState(
          "No model exported yet",
          "Generate an untrained snapshot, then reload this page.",
        );
      }
    }
  }
}

function ensureLiveClient() {
  if (!state.liveClient) {
    state.liveClient = new LiveClient({
      onState: handleLiveState,
      onModel: handleLiveModel,
      onStatus: handleLiveStatus,
      onHealth: (health) => {
        state.health = health;
        state.liveAvailable = Boolean(health.live);
        if (health.modelVersion) {
          state.liveVersion = health.modelVersion;
        }
        $("liveTab").disabled = !state.liveAvailable;
        if (health.live) {
          updateConnectionStatus("live");
        }
      },
    });
  }
  return state.liveClient;
}

function manualAction() {
  const steering = (state.manual.left ? 1 : 0) - (state.manual.right ? 1 : 0);
  const throttle = (state.manual.forward ? 1 : 0) - (state.manual.back ? 1 : 0);
  return { type: "drive", steering, throttle };
}

function sendManualAction() {
  if (state.mode !== "live" || state.driveMode !== "manual" || !state.liveClient) {
    return;
  }
  state.liveClient.send(manualAction());
}

function startManualLoop() {
  if (state.manualTimer) {
    return;
  }
  state.manualTimer = setInterval(sendManualAction, 50);
}

function stopManualLoop() {
  if (state.manualTimer) {
    clearInterval(state.manualTimer);
    state.manualTimer = null;
  }
}

function updateManualLoop() {
  if (state.mode === "live" && state.driveMode === "manual") {
    startManualLoop();
  } else {
    resetManualKeys();
    stopManualLoop();
  }
  $("manualHint").classList.toggle(
    "hidden",
    !(state.mode === "live" && state.driveMode === "manual"),
  );
}

function resetManualKeys() {
  state.manual = { left: false, right: false, forward: false, back: false };
}

function clearManualKeys() {
  resetManualKeys();
  sendManualAction();
}

function handleKeyDown(event) {
  if (state.mode !== "live" || state.driveMode !== "manual") {
    return;
  }
  if ($("helpDialog").open) {
    return;
  }
  if (event.target instanceof Element && event.target.closest("input, textarea, select")) {
    return;
  }
  const mapping = {
    KeyA: "left",
    ArrowLeft: "left",
    KeyD: "right",
    ArrowRight: "right",
    KeyW: "forward",
    ArrowUp: "forward",
    KeyS: "back",
    ArrowDown: "back",
  };
  const direction = mapping[event.code];
  if (!direction) {
    return;
  }
  event.preventDefault();
  state.manual[direction] = true;
  sendManualAction();
}

function handleKeyUp(event) {
  const mapping = {
    KeyA: "left",
    ArrowLeft: "left",
    KeyD: "right",
    ArrowRight: "right",
    KeyW: "forward",
    ArrowUp: "forward",
    KeyS: "back",
    ArrowDown: "back",
  };
  const direction = mapping[event.code];
  if (!direction || state.mode !== "live" || state.driveMode !== "manual") {
    return;
  }
  event.preventDefault();
  state.manual[direction] = false;
  sendManualAction();
}

function bindEvents() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => switchMode(button.dataset.mode));
  });
  $("modelSelect").addEventListener("change", (event) => {
    if (event.target.value === "__upload__" || event.target.value === "-1") {
      return;
    }
    loadCatalogEntry(Number(event.target.value));
  });
  $("modelFile").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const model = await loadModelFromUpload(file);
      state.uploadName = file.name;
      applyModel(model, { live: false, preserveInputs: true });
      populateModelSelect();
      showToast(`Loaded ${file.name}`);
    } catch (error) {
      showToast(`Could not load model: ${error.message}`, "error");
    } finally {
      event.target.value = "";
    }
  });
  $("validateButton").addEventListener("click", () => {
    if (!state.model) {
      return;
    }
    state.validation = checkValidation(state.model);
    updateModelMeta();
    renderInspector(state.selection ? infoFromSelection(state.selection) : null);
    showToast(
      state.validation.checked
        ? state.validation.ok
          ? `Validation passed · max error ${formatValue(state.validation.maxError, 2)}`
          : `Validation failed · max error ${formatValue(state.validation.maxError, 2)}`
        : "Model has no validation samples.",
      state.validation.checked && !state.validation.ok ? "error" : "info",
    );
  });
  $("fitButton").addEventListener("click", () => renderer.fit());
  $("helpButton").addEventListener("click", () => $("helpDialog").showModal());
  $("topKInput").addEventListener("input", (event) => {
    $("topKOutput").textContent = event.target.value;
    renderer.setTopK(Number(event.target.value));
  });
  $("thresholdInput").addEventListener("input", (event) => {
    $("thresholdOutput").textContent = `${event.target.value}%`;
    renderer.setThreshold(Number(event.target.value));
  });
  $("driveMode").addEventListener("change", (event) => {
    state.driveMode = event.target.value;
    state.modeChangedAt = Date.now();
    state.liveClient?.send({ type: "mode", mode: state.driveMode });
    clearManualKeys();
    updateManualLoop();
    event.target.blur();
  });
  $("pauseButton").addEventListener("click", () => {
    state.paused = !state.paused;
    $("pauseButton").textContent = state.paused ? "Resume" : "Pause";
    $("stepButton").disabled = !state.paused;
    state.liveClient?.send({ type: "pause", paused: state.paused });
  });
  $("stepButton").addEventListener("click", () => {
    state.liveClient?.send({ type: "step" });
  });
  $("resetButton").addEventListener("click", () => {
    state.liveClient?.send({ type: "reset" });
  });
  window.addEventListener("keydown", handleKeyDown);
  window.addEventListener("keyup", handleKeyUp);
  window.addEventListener("blur", clearManualKeys);
}

async function init() {
  bindEvents();
  updateConnectionStatus("idle");
  const health = await loadHealth();
  state.health = health;
  state.liveAvailable = Boolean(health.live);
  $("liveTab").disabled = !state.liveAvailable;
  updateConnectionStatus(health.live ? "live" : "unavailable");

  state.catalog = await loadCatalog();
  populateModelSelect();

  let loaded = false;
  if (state.catalog.length) {
    loaded = await loadCatalogEntry(0, { silent: true });
  }
  if (!loaded && health.live) {
    try {
      const model = await loadLiveModel();
      applyModel(model, { live: false, preserveInputs: false });
      loaded = true;
    } catch (error) {
      state.lastLiveError = error.message;
    }
  }
  if (!loaded) {
    showErrorState(
      "No model exported yet",
      "Generate an untrained snapshot, then reload this page.",
    );
    updateModeUI();
    renderer.render();
  } else {
    hideErrorState();
  }

  if (state.liveAvailable) {
    await switchMode("live");
  } else {
    state.mode = "explore";
    updateModeUI();
    populateModelSelect();
  }
}

init().catch((error) => {
  showToast(`Startup failed: ${error.message}`, "error");
  showErrorState("Visualizer failed to start", error.message);
});
