async function fetchJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }
  return response.json();
}

export async function loadHealth() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const health = await fetchJson("/api/health", { signal: controller.signal });
    clearTimeout(timeout);
    return { live: Boolean(health.live), mode: health.mode ?? null, ...health };
  } catch (error) {
    return { live: false, mode: null };
  }
}

export async function loadCatalog() {
  try {
    const entries = await fetchJson("models/manifest.json");
    if (!Array.isArray(entries)) {
      throw new Error("Model manifest must be an array.");
    }
    return entries.map((entry) => ({
      ...entry,
      url: `models/${entry.file}`,
    }));
  } catch (error) {
    try {
      const model = await loadModelFile("models/latest.json");
      return [
        {
          id: "latest",
          label: model.metadata?.label ?? "latest",
          file: "latest.json",
          url: "models/latest.json",
          createdAt: model.metadata?.createdAt ?? null,
          parameters: model.metadata?.parameters ?? null,
          networks: model.metadata?.networks ?? Object.keys(model.networks ?? {}),
        },
      ];
    } catch (fallbackError) {
      return [];
    }
  }
}

export async function loadModelFile(url) {
  const model = await fetchJson(url);
  validateModel(model);
  return model;
}

export async function loadLiveModel() {
  const model = await fetchJson(`/api/model?t=${Date.now()}`);
  validateModel(model);
  return model;
}

export function loadModelFromUpload(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the selected file."));
    reader.onload = () => {
      try {
        const model = JSON.parse(reader.result);
        validateModel(model);
        resolve(model);
      } catch (error) {
        reject(error);
      }
    };
    reader.readAsText(file);
  });
}

export function validateModel(model) {
  if (!model || typeof model !== "object") {
    throw new Error("Model file is not a JSON object.");
  }
  if (!model.networks || typeof model.networks !== "object") {
    throw new Error("Model is missing a networks mapping.");
  }
  const names = Object.keys(model.networks);
  if (names.length === 0) {
    throw new Error("Model contains no networks.");
  }
  const features = model.observation?.features;
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error("Model observation metadata is missing.");
  }
  for (const name of names) {
    const network = model.networks[name];
    if (!Array.isArray(network.layers) || network.layers.length === 0) {
      throw new Error(`Network ${name} has no layers.`);
    }
    const linear = network.layers.find((layer) => layer.kind === "linear");
    if (!linear) {
      throw new Error(`Network ${name} has no linear layers.`);
    }
    if (linear.in !== features.length) {
      throw new Error(
        `Network ${name} expects ${linear.in} inputs but metadata lists ${features.length}.`,
      );
    }
    for (const layer of network.layers) {
      if (layer.kind === "linear") {
        if (
          !Array.isArray(layer.weight) ||
          layer.weight.length !== layer.out ||
          layer.weight.some((row) => !Array.isArray(row) || row.length !== layer.in)
        ) {
          throw new Error(`Network ${name} has a malformed weight matrix.`);
        }
      }
    }
  }
  return true;
}

export function actionLabels(model) {
  const actor = model.networks.actor;
  const labels = actor?.output?.labels ?? ["steering", "throttle_brake"];
  return labels.map((label) =>
    String(label).replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase()),
  );
}
