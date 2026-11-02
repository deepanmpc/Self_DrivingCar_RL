import { loadHealth, loadLiveModel } from "./model.js";

export class LiveClient {
  constructor({ onState, onModel, onStatus, onHealth } = {}) {
    this.onState = onState;
    this.onModel = onModel;
    this.onStatus = onStatus;
    this.onHealth = onHealth;
    this.active = false;
    this.timer = null;
    this.version = null;
    this.pollInterval = 80;
    this.token = 0;
    this.failures = 0;
  }

  async start() {
    this.stop();
    const health = await loadHealth();
    if (!health.live) {
      this.emitStatus("unavailable", health);
      return false;
    }
    this.active = true;
    this.token += 1;
    this.version = health.modelVersion ?? null;
    this.pollInterval = health.pollIntervalMs ?? 80;
    this.onHealth?.(health);
    try {
      const model = await loadLiveModel();
      this.onModel?.(model, health);
      this.emitStatus("live", health);
    } catch (error) {
      this.version = null;
      this.emitStatus("error", { ...health, error: error.message });
    }
    this.schedulePoll();
    return true;
  }

  stop() {
    this.active = false;
    this.token += 1;
    this.failures = 0;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  schedulePoll(delay = this.pollInterval) {
    if (!this.active) {
      return;
    }
    const token = this.token;
    this.timer = setTimeout(() => this.poll(token), delay);
  }

  async poll(token) {
    if (!this.active || token !== this.token) {
      return;
    }
    try {
      const response = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`State request failed (${response.status})`);
      }
      const snapshot = await response.json();
      if (!this.active || token !== this.token) {
        return;
      }
      this.failures = 0;
      this.onState?.(snapshot);
      this.emitStatus("live", snapshot);
      if (snapshot.modelVersion && snapshot.modelVersion !== this.version) {
        await this.refreshModel(snapshot.modelVersion, token);
      }
      this.schedulePoll();
    } catch (error) {
      this.failures += 1;
      this.emitStatus(this.failures > 2 ? "error" : "reconnecting", {
        error: error.message,
      });
      this.schedulePoll(Math.min(2500, 500 * this.failures));
    }
  }

  async refreshModel(version, token) {
    try {
      const model = await loadLiveModel();
      if (!this.active || token !== this.token) {
        return;
      }
      this.version = version;
      this.onModel?.(model, { modelVersion: version });
    } catch (error) {
      this.version = null;
      this.emitStatus("error", { error: `Model refresh failed: ${error.message}` });
    }
  }

  async send(command) {
    try {
      const response = await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(command),
      });
      if (!response.ok) {
        throw new Error(`Control request failed (${response.status})`);
      }
      return await response.json();
    } catch (error) {
      this.emitStatus("error", { error: error.message });
      return null;
    }
  }

  emitStatus(status, detail = {}) {
    this.onStatus?.(status, detail);
  }
}
