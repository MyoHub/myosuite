// Predict pumpkin/body closest approach and convert it into a compact dodge command.

export function predictThreat({pumpkinPos, pumpkinVel, bodyTargets, gravity = [0, 0, -9.81], horizon = 1.5, dt = 0.02, hitRadius = 0.18}) {
  let best = null;
  for (let t = 0; t <= horizon; t += dt) {
    const p = ballistic(pumpkinPos, pumpkinVel, gravity, t);
    for (const target of bodyTargets) {
      const d = distance(p, target.position);
      const radius = hitRadius + (target.radius ?? 0.12);
      if (!best || d < best.distance) {
        best = {distance: d, ttc: t, point: p, target};
      }
      if (d <= radius) {
        return makeCommand({distance: d, ttc: t, point: p, target, threatening: true});
      }
    }
  }
  return best ? makeCommand({...best, threatening: false}) : noThreat();
}

function makeCommand({distance, ttc, point, target, threatening}) {
  if (!threatening) return noThreat();
  const side = Math.sign(point[1] - target.position[1]) || 1;
  const vertical = point[2];
  const duck = vertical > 1.15 ? clamp(1.0 - ttc / 1.0, 0, 1) : 0;
  const urgency = clamp(1.0 - ttc / 1.2, 0, 1);
  return {
    threat: true,
    bodyPart: target.name,
    distance,
    ttc,
    command: [
      0.0,                 // escape_x
      -side,                // escape_y
      duck,
      0.0,                  // torso_lean_x
      -0.45 * side,         // torso_lean_y
      urgency,
      ttc,
      vertical,
    ],
  };
}

export function noThreat() {
  return {threat: false, bodyPart: null, distance: Infinity, ttc: Infinity, command: [0,0,0,0,0,0,1.5,0]};
}

function ballistic(p, v, g, t) {
  return p.map((x, i) => x + v[i] * t + 0.5 * g[i] * t * t);
}
function distance(a, b) { return Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]); }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
