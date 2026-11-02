const POSITIVE = [56, 189, 248];
const NEGATIVE = [251, 146, 60];
const OUTPUT_COLOR = [244, 114, 182];

function rgba(color, alpha) {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function pointSegmentDistance(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) {
    return Math.hypot(px - x1, py - y1);
  }
  const t = clamp(((px - x1) * dx + (py - y1) * dy) / lengthSquared, 0, 1);
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

export class NetworkRenderer {
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.callbacks = callbacks;
    this.model = null;
    this.networkName = null;
    this.result = null;
    this.columns = [];
    this.blocks = [];
    this.allEdges = [];
    this.edges = [];
    this.topK = 3;
    this.threshold = 0;
    this.hover = null;
    this.selected = null;
    this.view = { x: 0, y: 0, scale: 1 };
    this.cssWidth = 0;
    this.cssHeight = 0;
    this.dpr = window.devicePixelRatio || 1;
    this.autoFit = true;
    this.pointer = null;
    this.drag = null;

    this.boundPointerMove = (event) => this.handlePointerMove(event);
    this.boundPointerUp = (event) => this.handlePointerUp(event);
    this.canvas.addEventListener("wheel", (event) => this.handleWheel(event), {
      passive: false,
    });
    this.canvas.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    this.canvas.addEventListener("pointermove", this.boundPointerMove);
    this.canvas.addEventListener("pointerup", this.boundPointerUp);
    this.canvas.addEventListener("pointercancel", this.boundPointerUp);
    this.canvas.addEventListener("pointerleave", () => this.handlePointerLeave());
    this.canvas.addEventListener("dblclick", () => this.fit());

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }

  setModel(model, networkName) {
    this.model = model;
    this.networkName = networkName;
    this.result = null;
    this.selected = null;
    this.hover = null;
    this.autoFit = true;
    this.rebuildColumns();
    this.fit();
  }

  setNetwork(networkName) {
    if (!this.model || !this.model.networks[networkName]) {
      return;
    }
    this.networkName = networkName;
    this.result = null;
    this.selected = null;
    this.hover = null;
    this.rebuildColumns();
    this.fit();
  }

  setResult(result) {
    this.result = result;
    this.rebuildColumns();
    this.recomputeEdges();
    this.render();
  }

  setTopK(value) {
    this.topK = clamp(Math.round(value), 1, 12);
    this.recomputeEdges();
    this.render();
  }

  setThreshold(value) {
    this.threshold = clamp(Number(value), 0, 100);
    this.recomputeEdges();
    this.render();
  }

  rebuildColumns() {
    if (!this.model || !this.networkName || !this.result) {
      this.columns = [];
      this.blocks = [];
      this.layout = null;
      return;
    }
    const network = this.model.networks[this.networkName];
    const features = this.model.observation.features;
    const blocks = this.result.inference.blocks;
    this.blocks = blocks;

    const columns = [
      {
        kind: "input",
        label: "Input",
        sublabel: `${features.length} features`,
        values: this.result.inference.input,
      },
      ...blocks.map((block, index) => {
        const isLast = index === blocks.length - 1;
        const activation = block.activation ? block.activation.toUpperCase() : "";
        const hasNorm = Boolean(block.normed);
        let label = isLast ? (network.role === "actor" ? "Action means" : "State value") : `Dense ${block.linear.out}`;
        if (isLast && network.role === "critic") {
          label = "Value";
        }
        const parts = [];
        if (!isLast) {
          parts.push(`${block.linear.out} units`);
        } else {
          parts.push(`${block.linear.out} output${block.linear.out === 1 ? "" : "s"}`);
        }
        if (activation) {
          parts.push(activation);
        }
        if (hasNorm) {
          parts.push("LayerNorm");
        }
        return {
          kind: isLast ? "output" : "hidden",
          label,
          sublabel: parts.join(" · "),
          values: block.values,
          block,
        };
      }),
    ];

    this.columns = columns;
    this.computeLayout(network);
  }

  computeLayout(network) {
    const sizes = this.columns.map((column) => column.values.length);
    const maxNodes = Math.max(...sizes);
    const spacing =
      maxNodes > 192 ? 7 : maxNodes > 96 ? 9 : maxNodes > 32 ? 14 : maxNodes > 12 ? 20 : 32;
    const margins = 130;
    const topPadding = 70;
    const bottomPadding = 46;
    const width = Math.max(980, (this.columns.length - 1) * 210 + margins * 2);
    const height = Math.max(680, topPadding + maxNodes * spacing + bottomPadding);
    const usableWidth = width - margins * 2;
    const usableHeight = height - topPadding - bottomPadding;

    this.columns.forEach((column, columnIndex) => {
      const size = column.values.length;
      const x =
        this.columns.length === 1
          ? width / 2
          : margins + (usableWidth * columnIndex) / (this.columns.length - 1);
      const totalHeight = (size - 1) * spacing;
      const startY = topPadding + (usableHeight - totalHeight) / 2;
      const radius =
        column.kind === "output"
          ? Math.min(9, Math.max(4, spacing * 0.4))
          : Math.min(7, Math.max(1.6, spacing * 0.34));
      column.x = x;
      column.size = size;
      column.radius = radius;
      column.positions = Array.from({ length: size }, (_, index) => ({
        x,
        y: startY + index * spacing,
      }));
      column.top = startY - radius;
      column.bottom = startY + totalHeight + radius;
    });

    const outputLabels = network.output?.labels ?? [];
    this.layout = {
      width,
      height,
      spacing,
      sizes,
      outputLabels,
      margins,
      topPadding,
    };
  }

  recomputeEdges() {
    this.allEdges = [];
    this.edges = [];
    if (!this.columns.length || !this.blocks.length) {
      return;
    }

    const topK = this.topK;
    this.blocks.forEach((block, blockIndex) => {
      const sourceValues = block.source;
      const weights = block.linear.weight;
      const bias = block.linear.bias;
      for (let target = 0; target < block.linear.out; target += 1) {
        const row = weights[target];
        const top = [];
        for (let source = 0; source < block.linear.in; source += 1) {
          const contribution = row[source] * sourceValues[source];
          const magnitude = Math.abs(contribution);
          if (magnitude === 0) {
            continue;
          }
          if (top.length < topK || magnitude > top[top.length - 1].magnitude) {
            let position = top.length;
            while (position > 0 && top[position - 1].magnitude < magnitude) {
              position -= 1;
            }
            top.splice(position, 0, {
              source,
              weight: row[source],
              contribution,
              magnitude,
            });
            if (top.length > topK) {
              top.pop();
            }
          }
        }
        for (const item of top) {
          this.allEdges.push({
            blockIndex,
            sourceColumn: blockIndex,
            targetColumn: blockIndex + 1,
            source: item.source,
            target,
            weight: item.weight,
            contribution: item.contribution,
            magnitude: item.magnitude,
            bias: bias ? bias[target] : 0,
          });
        }
      }
    });

    this.maxContribution =
      this.allEdges.reduce((maximum, edge) => Math.max(maximum, edge.magnitude), 0) || 1;
    const cutoff = (this.threshold / 100) * this.maxContribution;
    this.edges = this.allEdges.filter((edge) => edge.magnitude >= cutoff);
    if (this.callbacks.onEdgeCount) {
      this.callbacks.onEdgeCount(this.edges.length, this.allEdges.length);
    }
  }

  resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) {
      return;
    }
    this.dpr = window.devicePixelRatio || 1;
    this.cssWidth = width;
    this.cssHeight = height;
    this.canvas.width = Math.round(width * this.dpr);
    this.canvas.height = Math.round(height * this.dpr);
    if (this.autoFit) {
      this.fit();
    } else {
      this.render();
    }
  }

  fit() {
    if (!this.layout || !this.cssWidth || !this.cssHeight) {
      this.render();
      return;
    }
    const scale = Math.min(
      (this.cssWidth - 40) / this.layout.width,
      (this.cssHeight - 32) / this.layout.height,
    );
    this.view.scale = clamp(scale, 0.08, 1.4);
    this.view.x = (this.cssWidth - this.layout.width * this.view.scale) / 2;
    this.view.y = (this.cssHeight - this.layout.height * this.view.scale) / 2;
    this.autoFit = true;
    this.render();
  }

  zoomAt(screenX, screenY, factor) {
    if (!this.layout) {
      return;
    }
    const before = this.screenToVirtual(screenX, screenY);
    this.view.scale = clamp(this.view.scale * factor, 0.08, 3);
    this.view.x = screenX - before.x * this.view.scale;
    this.view.y = screenY - before.y * this.view.scale;
    this.autoFit = false;
    this.render();
  }

  screenToVirtual(x, y) {
    return {
      x: (x - this.view.x) / this.view.scale,
      y: (y - this.view.y) / this.view.scale,
    };
  }

  toScreen(point) {
    return {
      x: point.x * this.view.scale + this.view.x,
      y: point.y * this.view.scale + this.view.y,
    };
  }

  nodeAt(screenX, screenY) {
    if (!this.columns.length) {
      return null;
    }
    const point = this.screenToVirtual(screenX, screenY);
    let best = null;
    let bestDistance = Infinity;
    for (let columnIndex = 0; columnIndex < this.columns.length; columnIndex += 1) {
      const column = this.columns[columnIndex];
      const threshold = Math.max(column.radius * 1.9, 6.5 / this.view.scale);
      for (let nodeIndex = 0; nodeIndex < column.positions.length; nodeIndex += 1) {
        const position = column.positions[nodeIndex];
        const distance = Math.hypot(point.x - position.x, point.y - position.y);
        if (distance < threshold && distance < bestDistance) {
          bestDistance = distance;
          best = { columnIndex, nodeIndex };
        }
      }
    }
    if (!best) {
      return null;
    }
    const column = this.columns[best.columnIndex];
    return {
      type: "node",
      networkName: this.networkName,
      columnIndex: best.columnIndex,
      nodeIndex: best.nodeIndex,
      column,
      block: best.columnIndex > 0 ? this.blocks[best.columnIndex - 1] : null,
      value: column.values[best.nodeIndex],
    };
  }

  edgeAt(screenX, screenY) {
    if (!this.edges.length || this.view.scale < 0.25) {
      return null;
    }
    const point = this.screenToVirtual(screenX, screenY);
    const threshold = 5 / this.view.scale;
    let best = null;
    let bestDistance = threshold;
    for (const edge of this.edges) {
      const source = this.columns[edge.sourceColumn].positions[edge.source];
      const target = this.columns[edge.targetColumn].positions[edge.target];
      const distance = pointSegmentDistance(
        point.x,
        point.y,
        source.x,
        source.y,
        target.x,
        target.y,
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        best = edge;
      }
    }
    if (!best) {
      return null;
    }
    return {
      type: "edge",
      networkName: this.networkName,
      edge: best,
      sourceColumn: this.columns[best.sourceColumn],
      targetColumn: this.columns[best.targetColumn],
    };
  }

  handleWheel(event) {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const factor = Math.exp(-event.deltaY * 0.0015);
    this.zoomAt(event.clientX - rect.left, event.clientY - rect.top, factor);
  }

  handlePointerDown(event) {
    if (event.button !== 0) {
      return;
    }
    this.drag = {
      startX: event.clientX,
      startY: event.clientY,
      x: this.view.x,
      y: this.view.y,
      moved: false,
    };
    this.canvas.setPointerCapture(event.pointerId);
  }

  handlePointerMove(event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (this.drag) {
      const dx = event.clientX - this.drag.startX;
      const dy = event.clientY - this.drag.startY;
      if (Math.hypot(dx, dy) > 3) {
        this.drag.moved = true;
        this.autoFit = false;
        this.view.x = this.drag.x + dx;
        this.view.y = this.drag.y + dy;
        this.canvas.style.cursor = "grabbing";
        this.render();
      }
      return;
    }

    const info = this.nodeAt(x, y) ?? this.edgeAt(x, y);
    this.hover = info;
    this.canvas.style.cursor = info ? "pointer" : "grab";
    if (this.callbacks.onHover) {
      this.callbacks.onHover(info, { x: event.clientX, y: event.clientY });
    }
    this.render();
  }

  handlePointerUp(event) {
    if (!this.drag) {
      return;
    }
    const moved = this.drag.moved;
    this.drag = null;
    this.canvas.style.cursor = "grab";
    if (event.type === "pointercancel" || moved) {
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const info = this.nodeAt(x, y) ?? this.edgeAt(x, y);
    this.selected = info;
    if (this.callbacks.onSelect) {
      this.callbacks.onSelect(this.selected);
    }
    this.render();
  }

  handlePointerLeave() {
    this.hover = null;
    if (this.callbacks.onHover) {
      this.callbacks.onHover(null);
    }
    this.render();
  }

  render() {
    const ctx = this.ctx;
    if (!ctx || !this.cssWidth || !this.cssHeight) {
      return;
    }
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);
    this.drawBackground();

    if (!this.layout || !this.columns.length) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
      ctx.font = "500 13px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No network loaded", this.cssWidth / 2, this.cssHeight / 2);
      return;
    }

    const scale = this.view.scale;
    if (scale < 0.18) {
      this.drawBlockOverview();
    } else {
      if (scale >= 0.27) {
        this.drawEdges();
      } else {
        this.drawHighlightedEdges();
      }
      this.drawNodes();
    }

    this.drawHeaders();
    this.drawNodeLabels();
  }

  drawBackground() {
    const ctx = this.ctx;
    ctx.fillStyle = "#0a101d";
    ctx.fillRect(0, 0, this.cssWidth, this.cssHeight);

    const gradient = ctx.createRadialGradient(
      this.cssWidth * 0.5,
      this.cssHeight * 0.35,
      40,
      this.cssWidth * 0.5,
      this.cssHeight * 0.35,
      Math.max(this.cssWidth, this.cssHeight) * 0.75,
    );
    gradient.addColorStop(0, "rgba(30, 64, 105, 0.18)");
    gradient.addColorStop(1, "rgba(10, 16, 29, 0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, this.cssWidth, this.cssHeight);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.055)";
    ctx.lineWidth = 1;
    const step = 32;
    const offsetX = this.view.x % step;
    const offsetY = this.view.y % step;
    ctx.beginPath();
    for (let x = offsetX; x < this.cssWidth; x += step) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.cssHeight);
    }
    for (let y = offsetY; y < this.cssHeight; y += step) {
      ctx.moveTo(0, y);
      ctx.lineTo(this.cssWidth, y);
    }
    ctx.stroke();
  }

  drawBlockOverview() {
    const ctx = this.ctx;
    const scale = this.view.scale;
    for (const column of this.columns) {
      const top = this.toScreen(column.positions[0]).y - 8;
      const bottom = this.toScreen(column.positions[column.positions.length - 1]).y + 8;
      const x = this.toScreen({ x: column.x, y: 0 }).x;
      const width = Math.max(3, column.radius * 2 * scale);
      const max = Math.max(...column.values.map((value) => Math.abs(value)), 1e-6);
      const mean =
        column.values.reduce((sum, value) => sum + Math.abs(value), 0) / column.values.length;
      ctx.fillStyle = rgba(POSITIVE, 0.08 + 0.45 * (mean / max));
      ctx.beginPath();
      ctx.roundRect(x - width / 2, top, width, bottom - top, width / 2);
      ctx.fill();
    }
  }

  edgeStrength(edge) {
    return edge.magnitude / this.maxContribution;
  }

  drawEdges() {
    const ctx = this.ctx;
    const scale = this.view.scale;
    const cutoff = (this.threshold / 100) * this.maxContribution;
    const levels = 5;
    const batches = new Map();

    for (const edge of this.allEdges) {
      if (edge.magnitude < cutoff) {
        continue;
      }
      const source = this.toScreen(
        this.columns[edge.sourceColumn].positions[edge.source],
      );
      const target = this.toScreen(
        this.columns[edge.targetColumn].positions[edge.target],
      );
      if (
        Math.max(source.x, target.x) < -20 ||
        Math.min(source.x, target.x) > this.cssWidth + 20 ||
        Math.max(source.y, target.y) < -20 ||
        Math.min(source.y, target.y) > this.cssHeight + 20
      ) {
        continue;
      }
      const strength = this.edgeStrength(edge);
      const level = Math.min(levels - 1, Math.floor(strength * levels));
      const key = `${edge.contribution >= 0 ? "pos" : "neg"}:${level}`;
      if (!batches.has(key)) {
        batches.set(key, []);
      }
      batches.get(key).push({ edge, source, target });
    }

    for (const [key, items] of batches) {
      const [sign, levelText] = key.split(":");
      const level = Number(levelText);
      const color = sign === "pos" ? POSITIVE : NEGATIVE;
      const strength = (level + 1) / levels;
      ctx.beginPath();
      for (const item of items) {
        ctx.moveTo(item.source.x, item.source.y);
        ctx.lineTo(item.target.x, item.target.y);
      }
      ctx.strokeStyle = rgba(color, 0.045 + 0.34 * strength);
      ctx.lineWidth = Math.max(0.45, (0.35 + 2.2 * strength) * Math.min(1, scale * 1.8));
      ctx.lineCap = "round";
      ctx.stroke();
    }

    this.drawHighlightedEdges();
  }

  drawHighlightedEdges() {
    const focus = this.hover?.type === "node" ? this.hover : this.selected?.type === "node" ? this.selected : null;
    if (!focus) {
      return;
    }
    const ctx = this.ctx;
    const scale = this.view.scale;
    for (const edge of this.allEdges) {
      const connected =
        (edge.sourceColumn === focus.columnIndex && edge.source === focus.nodeIndex) ||
        (edge.targetColumn === focus.columnIndex && edge.target === focus.nodeIndex);
      if (!connected) {
        continue;
      }
      const source = this.toScreen(this.columns[edge.sourceColumn].positions[edge.source]);
      const target = this.toScreen(this.columns[edge.targetColumn].positions[edge.target]);
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = rgba(
        edge.contribution >= 0 ? POSITIVE : NEGATIVE,
        0.55 + 0.35 * this.edgeStrength(edge),
      );
      ctx.lineWidth = Math.max(1.2, 2.6 * this.edgeStrength(edge) * Math.min(1, scale * 2));
      ctx.lineCap = "round";
      ctx.stroke();
    }
  }

  drawNodes() {
    const ctx = this.ctx;
    const scale = this.view.scale;
    const hover = this.hover?.type === "node" ? this.hover : null;
    const selected = this.selected?.type === "node" ? this.selected : null;
    const focusKey = (info) => (info ? `${info.columnIndex}:${info.nodeIndex}` : null);
    const hoverKey = focusKey(hover);
    const selectedKey = focusKey(selected);

    this.columns.forEach((column, columnIndex) => {
      const maxAbs = Math.max(
        ...column.values.map((value) => Math.abs(value)),
        1e-6,
      );
      column.positions.forEach((position, nodeIndex) => {
        const screen = this.toScreen(position);
        if (
          screen.x < -20 ||
          screen.x > this.cssWidth + 20 ||
          screen.y < -20 ||
          screen.y > this.cssHeight + 20
        ) {
          return;
        }
        const value = column.values[nodeIndex];
        const magnitude = Math.min(1, Math.abs(value) / maxAbs);
        const isOutput = column.kind === "output";
        const color = isOutput ? OUTPUT_COLOR : value >= 0 ? POSITIVE : NEGATIVE;
        const radius = Math.max(column.radius * scale, 0.9);
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = rgba(color, 0.12 + 0.78 * magnitude);
        ctx.fill();
        if (radius > 1.8) {
          ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
        const key = `${columnIndex}:${nodeIndex}`;
        if (key === selectedKey || key === hoverKey) {
          ctx.beginPath();
          ctx.arc(screen.x, screen.y, Math.max(radius + 3, 4), 0, Math.PI * 2);
          ctx.strokeStyle = key === selectedKey ? "#f8fafc" : "rgba(248, 250, 252, 0.7)";
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
      });
    });
  }

  drawHeaders() {
    const ctx = this.ctx;
    const scale = this.view.scale;
    const showSublabels = scale >= 0.3;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    this.columns.forEach((column, index) => {
      const screenX = this.toScreen({ x: column.x, y: 0 }).x;
      if (screenX < -80 || screenX > this.cssWidth + 80) {
        return;
      }
      const offset = index % 2 === 0 ? 0 : 13;
      ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
      ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(column.label, screenX, 10 + offset);
      if (showSublabels) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
        ctx.font = "500 9px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(column.sublabel, screenX, 24 + offset);
      }
    });
  }

  drawNodeLabels() {
    const ctx = this.ctx;
    const scale = this.view.scale;
    if (!this.model) {
      return;
    }
    const features = this.model.observation.features;
    const inputColumn = this.columns[0];
    const outputColumn = this.columns[this.columns.length - 1];
    ctx.textBaseline = "middle";
    ctx.font = "500 10px ui-sans-serif, system-ui, sans-serif";

    if (scale >= 0.42 && inputColumn.size <= 16) {
      ctx.textAlign = "right";
      inputColumn.positions.forEach((position, index) => {
        const screen = this.toScreen(position);
        if (screen.y < 30 || screen.y > this.cssHeight - 10) {
          return;
        }
        ctx.fillStyle = "rgba(203, 213, 225, 0.82)";
        ctx.fillText(features[index]?.label ?? `Input ${index + 1}`, screen.x - 10, screen.y);
      });
    }

    const labels =
      this.model.networks[this.networkName]?.output?.labels ?? this.layout.outputLabels;
    if (scale >= 0.42 && outputColumn.size <= 8) {
      ctx.textAlign = "left";
      outputColumn.positions.forEach((position, index) => {
        const screen = this.toScreen(position);
        if (screen.y < 30 || screen.y > this.cssHeight - 10) {
          return;
        }
        const label = String(labels[index] ?? `Output ${index + 1}`)
          .replaceAll("_", " ")
          .replace(/\b\w/g, (char) => char.toUpperCase());
        ctx.fillStyle = "rgba(251, 207, 232, 0.88)";
        ctx.fillText(label, screen.x + 12, screen.y);
      });
    }
  }
}
