export function activate(name, value) {
  switch (name) {
    case "relu":
      return value > 0 ? value : 0;
    case "tanh":
      return Math.tanh(value);
    case "silu":
      return value / (1 + Math.exp(-value));
    case "gelu":
      return 0.5 * value * (1 + erf(value / Math.SQRT2));
    default:
      throw new Error(`Unsupported activation: ${name}`);
  }
}

function erf(value) {
  const sign = value < 0 ? -1 : 1;
  const magnitude = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * magnitude);
  const polynomial =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  return sign * (1 - polynomial * Math.exp(-magnitude * magnitude));
}

export function layerNorm(values, eps, weight, bias) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const denominator = Math.sqrt(variance + (eps ?? 1e-5));
  return values.map((value, index) => {
    const normalized = (value - mean) / denominator;
    const scale = weight ? weight[index] : 1;
    const offset = bias ? bias[index] : 0;
    return normalized * scale + offset;
  });
}

export function forwardNetwork(network, input) {
  const layers = network.layers;
  const firstLinear = layers.find((layer) => layer.kind === "linear");
  if (!firstLinear) {
    throw new Error("Network has no linear layers.");
  }
  if (firstLinear.in !== input.length) {
    throw new Error(
      `Model expects ${firstLinear.in} inputs but received ${input.length}.`,
    );
  }

  const blocks = [];
  let values = input.slice();
  let current = null;

  for (const layer of layers) {
    if (layer.kind === "linear") {
      const pre = new Array(layer.out);
      for (let target = 0; target < layer.out; target += 1) {
        const row = layer.weight[target];
        let total = layer.bias ? layer.bias[target] : 0;
        for (let source = 0; source < layer.in; source += 1) {
          total += row[source] * values[source];
        }
        pre[target] = total;
      }
      current = {
        index: blocks.length,
        linear: layer,
        source: values.slice(),
        pre,
        normed: null,
        activation: null,
        values: pre,
      };
      blocks.push(current);
      values = pre;
    } else if (layer.kind === "layernorm") {
      if (!current) {
        throw new Error("LayerNorm must follow a linear layer.");
      }
      const normalized = layerNorm(
        values,
        layer.eps,
        layer.weight,
        layer.bias,
      );
      current.normed = normalized;
      current.values = normalized;
      values = normalized;
    } else if (layer.kind === "activation") {
      if (!current) {
        throw new Error("Activation must follow a linear layer.");
      }
      const activated = values.map((value) => activate(layer.name, value));
      current.activation = layer.name;
      current.values = activated;
      values = activated;
    } else {
      throw new Error(`Unsupported layer kind: ${layer.kind}`);
    }
  }

  return {
    input: input.slice(),
    blocks,
    output: values,
  };
}

export function clampLogStd(network, logStd) {
  const minimum = network.output.minLogStd ?? -Infinity;
  const maximum = network.output.maxLogStd ?? Infinity;
  return logStd.map((value) => Math.min(maximum, Math.max(minimum, value)));
}

export function actorFromOutput(network, mean) {
  const logStd = clampLogStd(network, network.output.logStd);
  return {
    mean,
    actions: mean.map((value) => Math.tanh(value)),
    logStd,
    std: logStd.map((value) => Math.exp(value)),
  };
}

export function infer(model, networkName, input) {
  const network = model.networks[networkName];
  if (!network) {
    throw new Error(`Model does not contain a ${networkName} network.`);
  }
  const inference = forwardNetwork(network, input);
  if (network.role === "actor") {
    return {
      networkName,
      network,
      inference,
      actor: actorFromOutput(network, inference.output),
    };
  }
  return {
    networkName,
    network,
    inference,
    critic: { value: inference.output[0] },
  };
}

export function checkValidation(model) {
  const samples = model.validation?.samples ?? [];
  const tolerance = model.validation?.tolerance ?? 1e-3;
  let maxError = 0;
  let checked = 0;

  for (const sample of samples) {
    for (const [networkName, expected] of Object.entries(sample.outputs)) {
      const result = infer(model, networkName, sample.input);
      if (result.actor) {
        for (let index = 0; index < expected.mean.length; index += 1) {
          maxError = Math.max(
            maxError,
            Math.abs(result.actor.mean[index] - expected.mean[index]),
          );
        }
        for (let index = 0; index < expected.actions.length; index += 1) {
          maxError = Math.max(
            maxError,
            Math.abs(result.actor.actions[index] - expected.actions[index]),
          );
        }
      } else if (result.critic) {
        maxError = Math.max(
          maxError,
          Math.abs(result.critic.value - expected.value),
        );
      }
      checked += 1;
    }
  }

  return { ok: checked > 0 && maxError <= tolerance, maxError, checked, tolerance };
}

export function formatValue(value, digits = 3) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  if (value !== 0 && Math.abs(value) < 10 ** -digits) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
}
