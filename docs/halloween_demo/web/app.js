import {automaticThrow, manualThrowFromDrag, nextAutoDelayMs} from './throw_controller.js';
import {predictThreat} from './threat_predictor.js';

const canvas = document.querySelector('#view');
const ctx = canvas.getContext('2d');
const manualBtn = document.querySelector('#manual');
const autoBtn = document.querySelector('#auto');
const status = document.querySelector('#status');

let mode = 'manual';
let dragStart = null;
let pumpkin = null;
let nextAutoAt = performance.now() + 1200;
let lastT = performance.now();

const bodyTargets = [
  {name:'head',   position:[0,0,1.65], radius:0.13},
  {name:'thorax', position:[0,0,1.25], radius:0.22},
  {name:'pelvis', position:[0,0,0.90], radius:0.18},
  {name:'left_thigh', position:[0,0.16,0.58], radius:0.13},
  {name:'right_thigh',position:[0,-0.16,0.58], radius:0.13},
];

manualBtn.onclick = () => setMode('manual');
autoBtn.onclick = () => setMode('auto');

function setMode(next) {
  mode = next;
  manualBtn.classList.toggle('active', mode === 'manual');
  autoBtn.classList.toggle('active', mode === 'auto');
  status.textContent = mode === 'manual' ? 'Drag and release to throw a pumpkin.' : 'Automatic pumpkin throws enabled.';
  if (mode === 'auto') nextAutoAt = performance.now() + 500;
}

canvas.addEventListener('pointerdown', (e) => {
  if (mode !== 'manual') return;
  dragStart = {x:e.clientX, y:e.clientY};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
  if (!dragStart || mode !== 'manual') return;
  const drag = {x:(dragStart.x - e.clientX), y:0, screenY:(dragStart.y - e.clientY)};
  const spec = manualThrowFromDrag({spawn:[-3.2,0,1.1], drag, gain:0.055, verticalBias:1.2});
  pumpkin = makePumpkin(spec);
  dragStart = null;
});

function makePumpkin(spec) {
  return {pos:[...spec.position], vel:[...spec.velocity], source:spec.source, age:0};
}

function spawnAuto() {
  const side = Math.random() < 0.5 ? -1 : 1;
  const spawn = [-3.2, side * 0.6, 1.1 + Math.random() * 0.5];
  const target = [0,0,1.15];
  pumpkin = makePumpkin(automaticThrow({spawn, target}));
  nextAutoAt = performance.now() + nextAutoDelayMs();
}

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  canvas.style.width = innerWidth+'px'; canvas.style.height = innerHeight+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
addEventListener('resize', resize); resize();

function step(now) {
  const dt = Math.min((now-lastT)/1000, 0.033); lastT = now;
  if (mode === 'auto' && now >= nextAutoAt) spawnAuto();
  if (pumpkin) {
    pumpkin.age += dt;
    pumpkin.vel[2] -= 9.81 * dt;
    for (let i=0;i<3;i++) pumpkin.pos[i] += pumpkin.vel[i] * dt;
    if (pumpkin.pos[2] < 0 || pumpkin.age > 4) pumpkin = null;
  }
  render();
  requestAnimationFrame(step);
}
requestAnimationFrame(step);

function render() {
  ctx.clearRect(0,0,innerWidth,innerHeight);
  ctx.fillStyle = '#090b12'; ctx.fillRect(0,0,innerWidth,innerHeight);
  const sx = (x) => innerWidth*0.55 + x*150;
  const sy = (z) => innerHeight*0.83 - z*250;
  ctx.strokeStyle = '#293241'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0,sy(0)); ctx.lineTo(innerWidth,sy(0)); ctx.stroke();

  // Diagnostic stand-in for MyoFullBody; this is replaced by the mjswan/MuJoCo scene.
  ctx.strokeStyle = '#eee9dc'; ctx.lineWidth = 7; ctx.lineCap='round';
  ctx.beginPath(); ctx.arc(sx(0),sy(1.68),28,0,Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx(0),sy(1.54)); ctx.lineTo(sx(0),sy(.88)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx(0),sy(1.32)); ctx.lineTo(sx(-.34),sy(.98)); ctx.moveTo(sx(0),sy(1.32)); ctx.lineTo(sx(.34),sy(.98)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx(0),sy(.88)); ctx.lineTo(sx(-.22),sy(.18)); ctx.moveTo(sx(0),sy(.88)); ctx.lineTo(sx(.22),sy(.18)); ctx.stroke();

  let threat = null;
  if (pumpkin) {
    threat = predictThreat({pumpkinPos:pumpkin.pos, pumpkinVel:pumpkin.vel, bodyTargets});
    ctx.font='36px serif'; ctx.fillText('🎃', sx(pumpkin.pos[0])-18, sy(pumpkin.pos[2])+12);
  }
  if (threat?.threat) {
    status.textContent = `${pumpkin.source.toUpperCase()} · threat ${threat.bodyPart} · TTC ${threat.ttc.toFixed(2)} s · dodge ${threat.command[1] > 0 ? 'left' : 'right'}`;
  } else if (pumpkin) {
    status.textContent = `${pumpkin.source.toUpperCase()} · no predicted hit`;
  } else {
    status.textContent = mode === 'manual' ? 'Drag and release to throw a pumpkin.' : 'Automatic pumpkin throws enabled.';
  }
}
