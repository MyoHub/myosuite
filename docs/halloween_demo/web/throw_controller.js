// MyoHalloween throw generation shared by manual and automatic modes.

export const DEFAULT_THROW_CONFIG = {
  gravity: [0, 0, -9.81],
  autoMinInterval: 1.5,
  autoMaxInterval: 4.0,
  speedMin: 5.0,
  speedMax: 12.0,
  targetHeightMin: 0.45,
  targetHeightMax: 1.8,
  lateralAim: 0.45,
};

export function manualThrowFromDrag({spawn, drag, gain = 0.035, verticalBias = 1.5}) {
  return {
    position: [...spawn],
    velocity: [drag.x * gain, drag.y * gain, -drag.screenY * gain + verticalBias],
    source: "manual",
  };
}

export function automaticThrow({rng = Math.random, spawn, target, cfg = DEFAULT_THROW_CONFIG}) {
  const speed = lerp(cfg.speedMin, cfg.speedMax, rng());
  const desired = [
    target[0] + lerp(-cfg.lateralAim, cfg.lateralAim, rng()),
    target[1] + lerp(-cfg.lateralAim, cfg.lateralAim, rng()),
    lerp(cfg.targetHeightMin, cfg.targetHeightMax, rng()),
  ];
  const d = sub(desired, spawn);
  const n = norm(d) || 1;
  return {
    position: [...spawn],
    velocity: d.map((v) => (v / n) * speed),
    source: "auto",
  };
}

export function nextAutoDelayMs(rng = Math.random, cfg = DEFAULT_THROW_CONFIG) {
  return 1000 * lerp(cfg.autoMinInterval, cfg.autoMaxInterval, rng());
}

function lerp(a, b, t) { return a + (b - a) * t; }
function sub(a, b) { return a.map((v, i) => v - b[i]); }
function norm(v) { return Math.hypot(...v); }
