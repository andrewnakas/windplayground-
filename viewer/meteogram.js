// Point meteogram: wind speed and direction over the forecast at one spot,
// hand-rolled SVG. No chart library -- the needs are one line plot, one
// uncertainty band, and a row of direction arrows, and owning the ~200 lines
// keeps the panel styleable and dependency-free.

const NS = 'http://www.w3.org/2000/svg';

function el(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

// series: [{ label, color, width, dash?, points: [{t: Date, speed, dir} | null] }]
//   -- all series share the `times` array; a null point is a lead the source
//      does not carry. The FIRST series is primary: it gets the direction
//      arrows and the uncertainty band (if `sigma` is given).
// sigma: array aligned to times (m/s std) or null.
// convert: m/s -> display unit; unitLabel for the axis.
// onPickTime(i): clicking a time column hands the index back (drives the map).
export function renderMeteogram(container, { times, series, sigma, convert,
                                             unitLabel, activeIdx, onPickTime }) {
  container.innerHTML = '';
  const W = 640, H = 300, l = 44, r = 10, t = 14, bArrows = 46, bAxis = 30;
  const plotB = H - bArrows - bAxis;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'meteo-svg' }, container);

  const t0 = times[0].getTime(), t1 = times[times.length - 1].getTime();
  const x = d => l + (d.getTime() - t0) / Math.max(1, t1 - t0) * (W - l - r);

  let ymax = 1;
  const prim = series[0];
  for (const s of series)
    for (const p of s.points)
      if (p && p.speed > ymax) ymax = p.speed;
  for (let i = 0; i < times.length; i++)
    if (sigma && sigma[i] != null && prim.points[i])
      ymax = Math.max(ymax, prim.points[i].speed + sigma[i]);
  ymax *= 1.08;
  const y = v => plotB - v / ymax * (plotB - t);

  // grid + axes: a line per day boundary, y ticks at round display values
  for (let i = 0; i < times.length; i++) {
    const d = times[i];
    if (d.getUTCHours() !== 0 && i !== 0) continue;
    el('line', { x1: x(d), x2: x(d), y1: t, y2: plotB, class: 'meteo-grid' }, svg);
    const lab = el('text', { x: x(d) + 3, y: plotB + 16, class: 'meteo-lab' }, svg);
    lab.textContent = `${DAYS[d.getUTCDay()]} ${d.getUTCDate()}`;
  }
  for (const frac of [0.25, 0.5, 0.75, 1]) {
    const vDisp = Math.round(convert(ymax) * frac * 10) / 10;
    const vMs = ymax * frac;
    el('line', { x1: l, x2: W - r, y1: y(vMs), y2: y(vMs), class: 'meteo-grid' }, svg);
    const lab = el('text', { x: l - 5, y: y(vMs) + 3, 'text-anchor': 'end',
                             class: 'meteo-lab' }, svg);
    lab.textContent = String(vDisp);
  }
  const unit = el('text', { x: l - 5, y: t + 2, 'text-anchor': 'end',
                            class: 'meteo-lab' }, svg);
  unit.textContent = unitLabel;

  // uncertainty band around the primary line
  if (sigma) {
    let up = '', down = '';
    for (let i = 0; i < times.length; i++) {
      const p = prim.points[i];
      if (!p || sigma[i] == null) continue;
      up += `${up ? 'L' : 'M'}${x(times[i]).toFixed(1)},${y(p.speed + sigma[i]).toFixed(1)}`;
      down = `L${x(times[i]).toFixed(1)},${y(Math.max(0, p.speed - sigma[i])).toFixed(1)}` + down;
    }
    if (up) el('path', { d: up + down + 'Z', class: 'meteo-band' }, svg);
  }

  for (const s of [...series].reverse()) {          // primary drawn last, on top
    let d = '', pen = false;
    for (let i = 0; i < times.length; i++) {
      const p = s.points[i];
      if (!p) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${x(times[i]).toFixed(1)},${y(p.speed).toFixed(1)}`;
      pen = true;
    }
    el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width,
                 ...(s.dash ? { 'stroke-dasharray': s.dash } : {}) }, svg);
  }

  // direction strip: arrows point WHERE THE WIND BLOWS, like the particles.
  // (Barbs re-encode speed, which the line above already shows better.)
  const ay = H - bArrows / 2 - 4;
  for (let i = 0; i < times.length; i++) {
    const p = prim.points[i];
    if (!p) continue;
    el('path', {
      d: 'M0,-7 L4,3 L0,1 L-4,3 Z',
      transform: `translate(${x(times[i]).toFixed(1)},${ay}) rotate(${p.dir.toFixed(0)})`,
      fill: prim.color, opacity: 0.9,
    }, svg);
  }
  const dlab = el('text', { x: l, y: H - 4, class: 'meteo-lab' }, svg);
  dlab.textContent = 'arrows point where the wind blows';

  // hover crosshair + click-to-scrub
  const cross = el('line', { y1: t, y2: plotB, class: 'meteo-cross', visibility: 'hidden' }, svg);
  const tip = el('text', { y: t + 10, class: 'meteo-tip', visibility: 'hidden' }, svg);
  const active = el('line', { y1: t, y2: plotB, class: 'meteo-active' }, svg);
  if (activeIdx != null && times[activeIdx]) {
    active.setAttribute('x1', x(times[activeIdx]));
    active.setAttribute('x2', x(times[activeIdx]));
  } else active.setAttribute('visibility', 'hidden');

  const nearest = evx => {
    let best = 0, bd = Infinity;
    for (let i = 0; i < times.length; i++) {
      const d = Math.abs(x(times[i]) - evx);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };
  const toSvgX = ev => {
    const rect = svg.getBoundingClientRect();
    return (ev.clientX - rect.left) / rect.width * W;
  };
  svg.addEventListener('pointermove', ev => {
    const i = nearest(toSvgX(ev));
    const p = prim.points[i];
    cross.setAttribute('visibility', 'visible');
    cross.setAttribute('x1', x(times[i])); cross.setAttribute('x2', x(times[i]));
    tip.setAttribute('visibility', 'visible');
    const left = x(times[i]) < W / 2;
    tip.setAttribute('x', x(times[i]) + (left ? 6 : -6));
    tip.setAttribute('text-anchor', left ? 'start' : 'end');
    const d = times[i];
    tip.textContent = p
      ? `${DAYS[d.getUTCDay()]} ${String(d.getUTCHours()).padStart(2, '0')}Z · `
        + `${(Math.round(convert(p.speed) * 10) / 10)} ${unitLabel}`
      : 'no data at this time';
  });
  svg.addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden');
    tip.setAttribute('visibility', 'hidden');
  });
  svg.addEventListener('click', ev => onPickTime?.(nearest(toSvgX(ev))));

  // tiny legend under the chart
  const legend = document.createElement('div');
  legend.className = 'meteo-legend';
  for (const s of series) {
    const item = document.createElement('span');
    item.innerHTML = `<i style="background:${s.color};${s.dash ? 'opacity:.6;' : ''}"></i>${s.label}`;
    legend.appendChild(item);
  }
  if (sigma) {
    const item = document.createElement('span');
    item.innerHTML = `<i class="band"></i>±1σ across members`;
    legend.appendChild(item);
  }
  container.appendChild(legend);
}
