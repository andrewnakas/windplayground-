import { RasterLayer, gridFromRecord, sampleGrid } from './raster.js';
import { renderMeteogram } from './meteogram.js';
import { UNITS, currentUnit, setUnit, convert } from './units.js';

(async function () {
  // ONE definition of the wind color scale; the particle layer, the speed
  // raster, and the legend all read it, so they cannot drift apart.
  const WIND_STOPS = ['#3288bd', '#66c2a5', '#abdda4', '#e6f598', '#fee08b',
                      '#fdae61', '#f46d43', '#d53e4f'];
  const MAX_WIND = 25;                      // m/s, top of every wind scale
  // diverging RdBu for signed error, viridis for magnitudes without a sign
  const DIV_STOPS = ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b'];
  const SEQ_STOPS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];
  const BIAS_DOMAIN = [-5, 5], VEC_DOMAIN = [0, 10], SPREAD_DOMAIN = [0, 5];
  const VENDOR_UNIT = { ms: 'm/s', kn: 'kt', kmh: 'k/h', mph: 'mph' };
  const BLEND_SOURCE = 'live_blend';

  const statusEl = document.getElementById('status');
  const say = (msg, isErr) => { statusEl.textContent = msg; statusEl.className = isErr ? 'err' : ''; };

  // Land is drawn TWICE, in two panes, and that is the whole point.
  //
  // leaflet-velocity puts its particle canvas in `overlayPane` (its default),
  // and L.geoJSON renders through Leaflet's SVG renderer into `overlayPane`
  // too. Two layers in one pane means stacking falls to DOM insertion order,
  // and an opaque land fill landing on top of the canvas is exactly what "no
  // wind over land" looks like -- which is the bug this had. Brightening the
  // fill, as I tried first, made it worse.
  //
  // So state the z-order explicitly instead of hoping insertion order works
  // out: filled land BELOW the particles, coastline strokes ABOVE them. Thin
  // lines do not hide a flow field, so you get legible geography with wind
  // running across it.
  // Land fill stays DARK on purpose. Wind over land is mostly slow, and slow
  // particles are dark teal on this colour scale, so a light landmass washes
  // them out completely -- which is what a first attempt at a "brighter, more
  // visible" land did. Legibility comes from the coastline stroke instead;
  // the fill only needs to be distinguishable from the #0b1020 ocean.
  const LAND_FILL = {
    bright: { stroke: false, fillColor: '#161f38', fillOpacity: 1 },
    dim:    { stroke: false, fillColor: '#0e1426', fillOpacity: 1 },
  };
  const COAST_LINE = {
    bright: { color: '#9db4e8', weight: 1.0, opacity: 0.9, fill: false },
    dim:    { color: '#46567f', weight: 0.7, opacity: 0.7, fill: false },
  };
  let landFill = null, coastLine = null;
  let landBright = true;

  const map = L.map('map', {
    center: [20, 0], zoom: 2, minZoom: 1, maxZoom: 6,
    worldCopyJump: true, attributionControl: false,
  });

  // Leaflet's default overlayPane is z-index 400, which is where the particle
  // canvas goes; 350 puts the fill under it and 450 puts the coast over it.
  // deliberate debug/automation hook: verification steps drive the view and
  // assert pane order through this handle from the browser console
  window.__map = map;

  map.createPane('landPane').style.zIndex = 350;
  map.createPane('coastPane').style.zIndex = 450;
  // the coastline sits above the map surface, so it must not eat drag events
  map.getPane('coastPane').style.pointerEvents = 'none';

  // Self-contained basemap: Natural Earth land polygons, no external tiles.
  try {
    const land = await (await fetch('vendor/land.geo.json')).json();
    landFill = L.geoJSON(land, {
      pane: 'landPane', style: LAND_FILL.bright, interactive: false,
    }).addTo(map);
    coastLine = L.geoJSON(land, {
      pane: 'coastPane', style: COAST_LINE.bright, interactive: false,
    }).addTo(map);
  } catch (e) { say('Could not load coastlines: ' + e.message, true); }

  L.control.scale({ imperial: false, position: 'bottomright' }).addTo(map);

  let manifest;
  try {
    manifest = await (await fetch('data/manifest.json')).json();
  } catch (e) {
    say('No exported data found — run scripts/export_wind.py first.', true);
    return;
  }

  // Scores come from build_site.py's export of artifacts/results/ -- the
  // viewer never hard-codes a number. Optional: the map works without it.
  let metrics = null;
  try { metrics = await (await fetch('data/metrics.json')).json(); }
  catch (e) { document.getElementById('skill').style.display = 'none'; }

  const sourceSel = document.getElementById('source');
  const initSel = document.getElementById('init');
  const leadSlider = document.getElementById('lead');
  const leadOut = document.getElementById('leadOut');

  const GROUPS = {
    truth: 'Reanalysis',
    ckpt: 'Our models (trained here, 5.625°)',
    competitor: 'Published frontier models (0.25°, regridded)',
    live: 'Live operational runs (dynamical.org)',
    persistence: 'Baseline',
  };
  for (const [kind, groupLabel] of Object.entries(GROUPS)) {
    const inGroup = manifest.sources.filter(
      s => s.kind === kind && s.level !== '100m');
    if (!inGroup.length) continue;
    const og = document.createElement('optgroup');
    og.label = groupLabel;
    for (const s of inGroup) {
      const o = document.createElement('option');
      // show the score next to the name so picking a model and judging it
      // happen in the same place
      const score = s.rmse72 != null ? `  -  ${s.rmse72} m/s @72h` : '';
      o.value = s.id; o.textContent = s.label + score;
      og.appendChild(o);
    }
    sourceSel.appendChild(og);
  }
  sourceSel.value = 'era5';

  document.getElementById('landBtn').addEventListener('click', (e) => {
    landBright = !landBright;
    e.target.classList.toggle('on', landBright);
    const k = landBright ? 'bright' : 'dim';
    // only ever changes shading -- neither layer can occlude the wind now
    if (landFill) landFill.setStyle(LAND_FILL[k]);
    if (coastLine) coastLine.setStyle(COAST_LINE[k]);
  });

  const byId = Object.fromEntries(manifest.sources.map(s => [s.id, s]));
  let leads = [];

  // ---- 10 m / 100 m ------------------------------------------------------
  // 100 m sources are the same fleet under ids like aifs100_live; the picker
  // shows only 10 m and this toggle swaps the id render() actually loads.
  const level100 = {};
  for (const s of manifest.sources)
    if (s.level === '100m') level100[s.id.replace('100', '')] = s.id;
  let levelMode = '10m';
  const levelCtl = document.getElementById('levelCtl');

  function effId(id = sourceSel.value) {
    return levelMode === '100m' && level100[id] ? level100[id] : id;
  }

  function updateLevelCtl() {
    if (!Object.keys(level100).length) return;      // no 100 m data exported yet
    levelCtl.style.display = '';
    const has100 = Boolean(level100[sourceSel.value]);
    if (!has100) levelMode = '10m';
    levelCtl.innerHTML = '';
    for (const [mode, label] of [['10m', '10 m'], ['100m', '100 m']]) {
      const b = document.createElement('button');
      b.textContent = label;
      b.classList.toggle('on', mode === levelMode);
      if (mode === '100m' && !has100) {
        b.disabled = true;
        b.title = '100 m wind is exported for the live sources only';
      }
      b.addEventListener('click', () => {
        levelMode = mode;
        updateLevelCtl(); updateViewCtl(); render();
      });
      levelCtl.appendChild(b);
    }
  }

  // Sources cover different periods (a live run is from today, the research
  // forecasts from the 2020 test year), so the pickers follow the source.
  function syncPickers() {
    const src = byId[sourceSel.value];
    const inits = src.inits || manifest.inits;
    const wanted = initSel.value;
    initSel.innerHTML = '';
    for (const t of inits) {
      const o = document.createElement('option');
      // A live source stores its run under the fixed key "latest" so refreshes
      // overwrite instead of piling up; the real init time rides in the
      // manifest. Show that, plus its age, so "live" is checkable and not
      // just a label.
      if (t === 'latest' && src.init_time) {
        const d = new Date(src.init_time);
        const hrs = (Date.now() - d.getTime()) / 3.6e6;
        const age = hrs < 1 ? 'just now'
          : hrs < 48 ? `${Math.round(hrs)} h ago`
          : `${Math.round(hrs / 24)} d ago`;
        o.value = t;
        o.textContent = `${String(d.getUTCHours()).padStart(2, '0')}Z `
          + `${d.toISOString().slice(0, 10)}  (${age})`;
      } else {
        o.value = t; o.textContent = t.replace('T', '  ') + ':00 UTC';
      }
      initSel.appendChild(o);
    }
    if (inits.includes(wanted)) initSel.value = wanted;

    leads = src.leads || manifest.leads;
    const prevLead = leads[Number(leadSlider.value)] ?? leads[0];
    leadSlider.max = String(leads.length - 1);
    const idx = leads.indexOf(prevLead);
    leadSlider.value = String(idx >= 0 ? idx : 0);

    updateLevelCtl();
    updateViewCtl();
    updateAgeChip(src);
    renderSkill();
    buildTicks(src);
  }

  // ---- timeline: real valid times, not bare lead offsets ------------------
  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const validTimeEl = document.getElementById('validTime');
  const ticksEl = document.getElementById('tlTicks');

  function initDate(src, init) {
    if (init === 'latest' && src.init_time) return new Date(src.init_time);
    return new Date(init + ':00:00Z');
  }

  function validAt(src, init, lead) {
    return new Date(initDate(src, init).getTime() + lead * 3.6e6);
  }

  function buildTicks(src) {
    const init = initSel.value;
    ticksEl.innerHTML = '';
    for (const h of leads) {
      const d = validAt(src, init, h);
      const el = document.createElement('span');
      if (d.getUTCHours() === 0) {
        el.className = 'day';
        el.textContent = `${DAYS[d.getUTCDay()]} ${d.getUTCDate()}`;
      } else if (leads.length <= 9 || d.getUTCHours() === 12) {
        el.textContent = `${String(d.getUTCHours()).padStart(2, '0')}Z`;
      }
      ticksEl.appendChild(el);
    }
  }

  function updateValidTime(src, init, lead) {
    const d = validAt(src, init, lead);
    validTimeEl.textContent =
      `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ` +
      `${String(d.getUTCHours()).padStart(2, '0')}Z`;
  }

  // ---- skill panel: sparkline + leaderboard from metrics.json ----
  const skillBody = document.getElementById('skillBody');
  const I72 = () => metrics.lead_hours.indexOf(72);

  function sparklineSVG(selId) {
    const L = metrics.lead_hours;
    const curves = [
      { id: 'persistence', cls: 'ref', label: 'persistence' },
      { id: 'avg5', cls: 'ref2', label: 'best blend' },
      { id: selId, cls: 'cur', label: '' },
    ].filter(c => metrics.models[c.id]?.wind_speed);
    if (!curves.some(c => c.id === selId)) return '';
    const W = 300, H = 104, l = 30, r = 6, t = 8, b = 16;
    let ymax = 0;
    for (const c of curves)
      for (const v of metrics.models[c.id].wind_speed.rmse)
        if (v != null && v > ymax) ymax = v;
    ymax = Math.ceil(ymax * 2) / 2;
    const x = h => l + (h - L[0]) / (L[L.length - 1] - L[0]) * (W - l - r);
    const y = v => H - b - v / ymax * (H - t - b);
    let svg = `<svg class="spark" viewBox="0 0 ${W} ${H}" aria-label="RMSE vs lead time">`;
    svg += `<line class="ax" x1="${l}" y1="${H - b}" x2="${W - r}" y2="${H - b}"/>`;
    svg += `<line class="ax" x1="${l}" y1="${t}" x2="${l}" y2="${H - b}"/>`;
    for (const h of [24, 72, 120])
      svg += `<text x="${x(h)}" y="${H - 4}" text-anchor="middle">+${h}h</text>`;
    for (const v of [ymax, ymax / 2])
      svg += `<text x="${l - 3}" y="${y(v) + 3}" text-anchor="end">${(Math.round(convert(v) * 10) / 10)}</text>`;
    for (const c of curves) {
      const rm = metrics.models[c.id].wind_speed.rmse;
      let d = '', pen = false;
      for (let i = 0; i < L.length; i++) {
        if (rm[i] == null) { pen = false; continue; }
        d += `${pen ? 'L' : 'M'}${x(L[i]).toFixed(1)},${y(rm[i]).toFixed(1)}`;
        pen = true;
      }
      svg += `<path class="${c.cls}" d="${d}"/>`;
      if (c.label) {
        const last = rm.map((v, i) => v == null ? null : i).filter(i => i != null).pop();
        if (last != null)
          svg += `<text x="${W - r}" y="${y(rm[last]) - 3}" text-anchor="end">${c.label}</text>`;
      }
    }
    return svg + '</svg>';
  }

  function leaderboardHTML(selId) {
    const i72 = I72();
    const rows = Object.entries(metrics.models)
      .filter(([id]) => byId[id])
      .map(([id, m]) => ({ id, v: m.wind_speed?.rmse[i72] }))
      .filter(r => r.v != null)
      .sort((a, b) => a.v - b.v);
    if (!rows.length) return '';
    const worst = rows[rows.length - 1].v;
    return '<div class="lb">' + rows.map(r =>
      `<div class="lb-row${r.id === selId ? ' cur' : ''}" data-id="${r.id}" title="wind speed RMSE at +72 h">`
      + `<span class="nm">${byId[r.id].label}</span>`
      + `<span class="bar" style="width:${(r.v / worst * 100).toFixed(0)}%"></span>`
      + `<span class="v">${convert(r.v).toFixed(2)}</span></div>`).join('') + '</div>';
  }

  function renderSkill() {
    if (!metrics) return;
    const sid = sourceSel.value;
    const src = byId[sid];
    let out = '';
    if (src.kind === 'live') {
      // the honesty rule: hindcast skill is a 2020 number; a live run has no
      // verifying truth yet, and pretending otherwise is how trust dies
      out += '<div class="skill-badge">2020 hindcast skill unavailable — this is a live run; no verifying truth exists yet.</div>';
    } else if (src.kind === 'truth') {
      out += '<div class="skill-note">ERA5 is the truth every score below is measured against.</div>';
    } else if (metrics.models[sid]) {
      out += sparklineSVG(sid);
      out += `<div class="skill-note">10 m wind speed RMSE (${UNITS[currentUnit()].label}) vs lead — lower is better.</div>`;
      const sh = metrics.sharpness[sid]?.['120'];
      if (sh) out += `<div class="skill-note" title="ratio of forecast to ERA5 small-scale spectral power at +120 h">`
        + `sharpness: retains ${Math.round(sh.ws_spec_ratio * 100)}% of small-scale power @120h — blurrier ≠ worse RMSE</div>`;
    }
    out += leaderboardHTML(sid);
    out += `<div class="skill-note">${metrics.provenance}</div>`;
    skillBody.innerHTML = out;
    for (const row of skillBody.querySelectorAll('.lb-row'))
      row.addEventListener('click', () => {
        sourceSel.value = row.dataset.id;
        sourceSel.dispatchEvent(new Event('change'));
      });
  }

  // Freshness is this site's weak point (the refresh job is scheduled, not
  // guaranteed), so surface data age instead of letting "live" go unchecked.
  const ageEl = document.getElementById('age');
  function updateAgeChip(src) {
    if (src.kind !== 'live' || !src.init_time) { ageEl.className = ''; return; }
    const hrs = (Date.now() - new Date(src.init_time).getTime()) / 3.6e6;
    const h = Math.round(hrs);
    if (hrs < 8) { ageEl.className = 'fresh'; ageEl.textContent = `● live — run is ${h} h old`; }
    else if (hrs <= 18) { ageEl.className = 'aging'; ageEl.textContent = `● run is ${h} h old — next refresh due`; }
    else { ageEl.className = 'stale'; ageEl.textContent = `● refresh may be stalled — data is ${h} h old`; }
  }

  let velocityLayer = null;
  const cache = new Map();

  // Speed underlay: particles show direction and flow, but magnitude at a
  // glance needs a filled field -- hover-only readouts don't survive contact
  // with a real question like "how windy is the North Atlantic today".
  const speedLayer = new RasterLayer({
    colorStops: WIND_STOPS, domain: [0, MAX_WIND], opacity: 0.55,
  }).addTo(map);
  let shadeOn = true;
  const shadeBtn = document.getElementById('shadeBtn');
  shadeBtn.addEventListener('click', () => {
    shadeOn = !shadeOn;
    shadeBtn.classList.toggle('on', shadeOn);
    render();          // the raster is mode-owned; only render knows its role
  });
  let currentSpeedGrid = null;

  function speedGridOf(data) {
    const u = data[0].data, v = data[1].data;
    const sp = new Float64Array(u.length);
    for (let i = 0; i < u.length; i++) sp[i] = Math.hypot(u[i], v[i]);
    return gridFromRecord(data[0], sp);
  }

  // ---- legend + units: generated from the same stops/domain as the map ----
  const legendBar = document.getElementById('legendBar');
  const legendTicks = document.getElementById('legendTicks');
  function setLegend(stops, domainMs, plus = true) {
    legendBar.style.background = `linear-gradient(90deg, ${stops.join(',')})`;
    const u = currentUnit();
    const n = 5, parts = [];
    for (let i = 0; i < n; i++) {
      const ms = domainMs[0] + (domainMs[1] - domainMs[0]) * i / (n - 1);
      const val = Math.round(convert(ms, u) * 10) / 10;
      const txt = (i === n - 1)
        ? `${val}${plus ? '+' : ''} ${UNITS[u].label}` : String(val);
      parts.push(`<span>${txt}</span>`);
    }
    legendTicks.innerHTML = parts.join('');
  }

  const unitsCtl = document.getElementById('unitsCtl');
  for (const key of Object.keys(UNITS)) {
    const b = document.createElement('button');
    b.textContent = UNITS[key].label;
    b.dataset.u = key;
    b.classList.toggle('on', key === currentUnit());
    b.addEventListener('click', () => {
      setUnit(key);
      for (const x of unitsCtl.children) x.classList.toggle('on', x.dataset.u === key);
      renderSkill();
      render();          // rebuilds the legend and readout in the new unit
    });
    unitsCtl.appendChild(b);
  }
  setLegend(WIND_STOPS, [0, MAX_WIND]);

  // Bounded LRU: a full live source is ~21 fields x ~150 KB parsed; sixty
  // entries covers three sources warm without letting an afternoon of
  // browsing hold every field ever seen.
  const CACHE_MAX = 60;

  async function loadField(sourceId, init, lead) {
    const key = `${sourceId}_${init}_${String(lead).padStart(3, '0')}`;
    if (cache.has(key)) {
      const v = cache.get(key);
      cache.delete(key); cache.set(key, v);     // refresh recency
      return v;
    }
    const res = await fetch(`data/${key}.json`);
    if (!res.ok) throw new Error(`no field for ${sourceId} at +${lead} h`);
    const data = await res.json();
    cache.set(key, data);
    if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
    return data;
  }

  // Warm the active source's other leads while the map idles, so scrubbing
  // and playback never fetch-stall. The token cancels a sweep the moment the
  // selection moves on.
  let prefetchToken = 0;
  function prefetchLeads() {
    const token = ++prefetchToken;
    const sourceId = effId(), init = initSel.value, want = [...leads];
    const run = async () => {
      for (const h of want) {
        if (token !== prefetchToken) return;
        await loadField(sourceId, init, h).catch(() => {});
      }
    };
    (window.requestIdleCallback ?? (f => setTimeout(f, 600)))(() => run());
  }

  function removeVelocity() {
    if (velocityLayer) { map.removeLayer(velocityLayer); velocityLayer = null; }
  }

  let velocityConfig = null;
  function showVelocity(data) {
    // setData keeps the particle canvas alive between frames; a rebuild is
    // only needed when the readout control must change (units, level)
    const config = currentUnit() + levelMode;
    if (velocityLayer && velocityConfig === config) {
      velocityLayer.setData(data);
      return;
    }
    removeVelocity();
    velocityConfig = config;
    velocityLayer = L.velocityLayer({
      displayValues: true,
      displayOptions: {
        velocityType: (levelMode === '100m' ? '100' : '10') + ' m wind',
        position: 'bottomleft',
        emptyString: 'hover the map for a wind readout',
        angleConvention: 'bearingCW',
        speedUnit: VENDOR_UNIT[currentUnit()],
      },
      data,
      maxVelocity: MAX_WIND,
      velocityScale: 0.012,
      particleAge: 70,
      lineWidth: 1.4,
      particleMultiplier: 1 / 260,
      colorScale: WIND_STOPS,
    });
    velocityLayer.addTo(map);
  }

  function setSub(data, src) {
    const nx = data[0].header.nx, ny = data[0].header.ny;
    const degrees = (360 / nx).toFixed(2).replace(/\.?0+$/, '');
    const height = src.level === '100m' ? '100' : '10';
    document.getElementById('sub').innerHTML =
      `${height}&nbsp;m wind &middot; ${nx}&times;${ny} grid (${degrees}&deg;) &middot; ` +
      (src.kind === 'live' ? 'live operational run' : '2020 test year');
  }

  async function render() {
    const sourceId = effId();
    const init = initSel.value;
    const lead = leads[Number(leadSlider.value)];
    leadOut.textContent = `+${lead} h`;
    const src = byId[sourceId];
    updateValidTime(src, init, lead);
    try {
      if (viewMode === 'error') {
        const ref = refFor(src);
        const [a, b] = await Promise.all([
          loadField(sourceId, init, lead), loadField(ref.id, init, lead)]);
        removeVelocity();                 // the error field IS the content
        const bias = errKind === 'bias';
        speedLayer.setStyle({ colorStops: bias ? DIV_STOPS : SEQ_STOPS,
                              domain: bias ? BIAS_DOMAIN : VEC_DOMAIN, opacity: 0.8 });
        speedLayer.setGrid(diffGrid(a, b, errKind));
        setLegend(bias ? DIV_STOPS : SEQ_STOPS, bias ? BIAS_DOMAIN : VEC_DOMAIN, !bias);
        setSub(a, src);
        say(src.kind === 'live'
          ? `departure from the multi-model mean at +${lead} h — a consistency check, not verification`
          : `${bias ? 'speed bias' : 'vector error'} vs ERA5 truth at +${lead} h`);
        return;
      }
      if (viewMode === 'spread') {
        const spreadFile = src.level === '100m' ? 'live_spread100' : 'live_spread';
        const [rec, data] = await Promise.all([
          loadField(spreadFile, init, lead), loadField(sourceId, init, lead)]);
        showVelocity(data);
        speedLayer.setStyle({ colorStops: SEQ_STOPS, domain: SPREAD_DOMAIN, opacity: 0.7 });
        speedLayer.setGrid(gridFromRecord(rec[0]));
        setLegend(SEQ_STOPS, SPREAD_DOMAIN);
        setSub(data, src);
        const n = (rec[0].header.members || []).length;
        say(`spread of ${n} models (std of speed)` +
            (n === 2 ? ' — with 2 members this is half the gap between them' : ''));
        return;
      }
      // 'wind', or 'ref' (same drawing, the reference's field instead)
      const shownId = viewMode === 'ref' ? refFor(src).id : sourceId;
      const data = await loadField(shownId, init, lead);
      showVelocity(data);
      currentSpeedGrid = speedGridOf(data);
      speedLayer.setStyle({ colorStops: WIND_STOPS, domain: [0, MAX_WIND], opacity: 0.55 });
      speedLayer.setGrid(shadeOn ? currentSpeedGrid : null);
      setLegend(WIND_STOPS, [0, MAX_WIND]);
      setSub(data, src);
      const shown = byId[shownId];
      if (viewMode === 'ref') {
        say(`${shown.label} — the reference for ${src.label}, +${lead} h from ${init}`);
      } else {
        say(lead === 0 ? `${src.label} — analysis at ${init}`
                       : `${src.label} — +${lead} h from ${init}`);
      }
      prefetchLeads();
    } catch (e) {
      say(e.message, true);
    }
    // every control path funnels through render(), so an open meteogram
    // follows units, level, source, init and the active-lead marker for free
    drawMeteogram();
  }

  sourceSel.addEventListener('change', () => { syncPickers(); render(); });
  initSel.addEventListener('change', render);
  leadSlider.addEventListener('input', render);

  // ---- view modes: Wind | reference | Error | Spread --------------------
  // "Compare vs ERA5" used to flip the source select; a mode never mutates
  // the pickers, it only changes what render() draws for the same selection.
  let viewMode = 'wind';
  let errKind = 'bias';
  const viewCtl = document.getElementById('viewCtl');
  const errCtl = document.getElementById('errCtl');

  // The reference a source can honestly be compared against: hindcasts have
  // ERA5 truth at the same init; a live member only has the blend -- a
  // consistency check, not verification. The blend itself has its spread.
  function refFor(src) {
    if (src.kind === 'truth') return null;
    if (src.kind === 'live') {
      if (src.id.startsWith(BLEND_SOURCE)) return null;    // the blends themselves
      return byId[src.level === '100m' ? BLEND_SOURCE + '100' : BLEND_SOURCE] ?? null;
    }
    return byId['era5'] ?? null;
  }

  function modesFor(src) {
    const modes = [{ id: 'wind', label: 'Wind' }];
    const ref = refFor(src);
    if (ref) {
      const live = src.kind === 'live';
      modes.push({ id: 'ref', label: live ? 'Blend' : 'ERA5' });
      modes.push({ id: 'error', label: live ? 'Δ vs blend' : 'Error' });
    }
    if (src.id.startsWith(BLEND_SOURCE) && (src.spread_leads || []).length)
      modes.push({ id: 'spread', label: 'Spread' });
    return modes;
  }

  function updateViewCtl() {
    const modes = modesFor(byId[effId()]);
    if (!modes.some(m => m.id === viewMode)) viewMode = 'wind';
    viewCtl.style.display = modes.length > 1 ? '' : 'none';
    viewCtl.innerHTML = '';
    for (const m of modes) {
      const b = document.createElement('button');
      b.textContent = m.label;
      b.classList.toggle('on', m.id === viewMode);
      b.addEventListener('click', () => { viewMode = m.id; updateViewCtl(); render(); });
      viewCtl.appendChild(b);
    }
    errCtl.style.display = viewMode === 'error' ? '' : 'none';
  }

  for (const [id, label] of [['bias', 'speed bias'], ['vec', 'vector error']]) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.k = id;
    b.classList.toggle('on', id === errKind);
    b.addEventListener('click', () => {
      errKind = id;
      for (const x of errCtl.children) x.classList.toggle('on', x.dataset.k === id);
      render();
    });
    errCtl.appendChild(b);
  }

  // Both error scalars from two fields on the SAME grid (guaranteed: all
  // hindcasts share the 64x32 export grid, all live sources the 180x90 one).
  // bias = |V|a - |V|b (the sign people argue about); vec = |Va - Vb| (misses
  // nothing, but blurs direction error into one magnitude).
  function diffGrid(a, b, kind) {
    const ua = a[0].data, va = a[1].data, ub = b[0].data, vb = b[1].data;
    const out = new Float64Array(ua.length);
    for (let i = 0; i < out.length; i++)
      out[i] = kind === 'bias'
        ? Math.hypot(ua[i], va[i]) - Math.hypot(ub[i], vb[i])
        : Math.hypot(ua[i] - ub[i], va[i] - vb[i]);
    return gridFromRecord(a[0], out);
  }

  const playBtn = document.getElementById('playBtn');
  const speedBtn = document.getElementById('speedBtn');
  const SPEEDS = [0.5, 1, 2];
  let speedIdx = 1;
  let timer = null;

  function stepLead(delta) {
    const next = (Number(leadSlider.value) + delta + leads.length) % leads.length;
    leadSlider.value = String(next);
    render();
  }

  function stopPlay() {
    if (timer) clearInterval(timer);
    timer = null;
    playBtn.classList.remove('on');
    playBtn.textContent = '▶';
  }

  function startPlay() {
    playBtn.classList.add('on');
    playBtn.textContent = '❚❚';
    timer = setInterval(() => stepLead(1), 1600 / SPEEDS[speedIdx]);
  }

  playBtn.addEventListener('click', () => (timer ? stopPlay() : startPlay()));
  speedBtn.addEventListener('click', () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    speedBtn.textContent = `${SPEEDS[speedIdx]}×`;
    if (timer) { clearInterval(timer); timer = setInterval(() => stepLead(1), 1600 / SPEEDS[speedIdx]); }
  });

  // ---- help + keyboard ----------------------------------------------------
  const help = document.getElementById('help');
  document.getElementById('helpBtn').addEventListener('click', () => { help.hidden = false; });
  help.addEventListener('click', () => { help.hidden = true; });

  function cycleModel(delta) {
    const opts = [...sourceSel.options];
    const i = opts.findIndex(o => o.value === sourceSel.value);
    sourceSel.value = opts[(i + delta + opts.length) % opts.length].value;
    sourceSel.dispatchEvent(new Event('change'));
  }

  document.addEventListener('keydown', (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = ev.target.tagName;
    if ((tag === 'INPUT' && ev.target.id !== 'lead') || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (!help.hidden) { help.hidden = true; return; }
    switch (ev.key) {
      case ' ': ev.preventDefault(); timer ? stopPlay() : startPlay(); break;
      case 'ArrowLeft': ev.preventDefault(); stepLead(-1); break;
      case 'ArrowRight': ev.preventDefault(); stepLead(1); break;
      case 'ArrowUp': ev.preventDefault(); cycleModel(-1); break;
      case 'ArrowDown': ev.preventDefault(); cycleModel(1); break;
      case 'u': {
        const keys = Object.keys(UNITS);
        const next = keys[(keys.indexOf(currentUnit()) + 1) % keys.length];
        [...unitsCtl.children].find(b => b.dataset.u === next)?.click();
        break;
      }
      case 'e':
        if (modesFor(byId[sourceSel.value]).some(m => m.id === 'error')) {
          viewMode = viewMode === 'error' ? 'wind' : 'error';
          updateViewCtl(); render();
        }
        break;
      case 's': shadeBtn.click(); break;
      case '?': help.hidden = !help.hidden; break;
    }
  });

  // ---- mobile: collapse the panel behind a toggle -------------------------
  const panel = document.getElementById('panel');
  document.getElementById('panelToggle').addEventListener('click', () => {
    panel.classList.toggle('hidden');
  });
  // phones start with the map, not a sheet of controls over it
  if (window.matchMedia('(max-width: 620px)').matches) panel.classList.add('hidden');

    // ---- point meteogram: click the map, get the forecast at that spot ------
  const meteo = document.getElementById('meteo');
  const meteoTitle = document.getElementById('meteoTitle');
  const meteoSub = document.getElementById('meteoSub');
  const meteoChart = document.getElementById('meteoChart');
  const meteoNote = document.getElementById('meteoNote');
  const CARDS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  let meteoPoint = null;
  let meteoMarker = null;
  let meteoToken = 0;

  function shortLabel(src) {
    return src.label.split(/[(—]/)[0].trim();
  }

  async function pointAt(sourceId, init, lead, latlng) {
    const data = await loadField(sourceId, init, lead);
    const u = sampleGrid(gridFromRecord(data[0]), latlng.lat, latlng.lng);
    const v = sampleGrid(gridFromRecord(data[1]), latlng.lat, latlng.lng);
    // bearing the wind blows TOWARD, matching the particles on the map
    return { speed: Math.hypot(u, v), dir: Math.atan2(u, v) * 180 / Math.PI };
  }

  async function drawMeteogram() {
    if (!meteoPoint || meteo.hidden) return;
    const token = ++meteoToken;
    const src = byId[effId()];
    const init = initSel.value;
    const live = src.kind === 'live';
    const blendId = src.level === '100m' ? BLEND_SOURCE + '100' : BLEND_SOURCE;
    const spreadFile = src.level === '100m' ? 'live_spread100' : 'live_spread';
    const blend = byId[blendId];

    // Which lines: live context reads the blend against its members; a
    // hindcast reads the model against ERA5 truth at the same valid times.
    let list;
    if (live && blend) {
      const members = (blend.members || []).filter(id => byId[id]);
      list = [{ id: blendId, label: 'multi-model mean', color: '#58a6ff', width: 2.5 },
              ...members.map(id => ({ id, label: shortLabel(byId[id]),
                                      color: '#7d8bb8', width: 1.2 }))];
    } else if (src.kind === 'truth') {
      list = [{ id: src.id, label: 'ERA5 (truth)', color: '#58a6ff', width: 2.5 }];
    } else {
      list = [{ id: src.id, label: shortLabel(src), color: '#58a6ff', width: 2.5 },
              { id: 'era5', label: 'ERA5 truth', color: '#e8ecf7', width: 1.4, dash: '4 3' }];
    }
    list = list.filter(sr => byId[sr.id]);

    const primSrc = byId[list[0].id];
    const primLeads = leads.filter(h => (primSrc.leads || leads).includes(h));
    const times = primLeads.map(h => validAt(src, init, h));

    meteoTitle.textContent =
      `${Math.abs(meteoPoint.lat).toFixed(1)}°${meteoPoint.lat >= 0 ? 'N' : 'S'}, ` +
      `${Math.abs(meteoPoint.lng).toFixed(1)}°${meteoPoint.lng >= 0 ? 'E' : 'W'}`;
    meteoSub.textContent = 'loading forecast at this point…';

    const series = [];
    for (const sr of list) {
      const own = byId[sr.id].leads || leads;
      const points = [];
      for (const h of primLeads) {
        if (!own.includes(h)) { points.push(null); continue; }
        try { points.push(await pointAt(sr.id, init, h, meteoPoint)); }
        catch (e) { points.push(null); }
      }
      series.push({ ...sr, points });
      if (token !== meteoToken) return;
    }

    let sigma = null;
    if (live && blend && (blend.spread_leads || []).length) {
      sigma = [];
      for (const h of primLeads) {
        if (!blend.spread_leads.includes(h)) { sigma.push(null); continue; }
        try {
          const rec = await loadField(spreadFile, 'latest', h);
          sigma.push(sampleGrid(gridFromRecord(rec[0]), meteoPoint.lat, meteoPoint.lng));
        } catch (e) { sigma.push(null); }
      }
      if (token !== meteoToken) return;
    }

    const lead = leads[Number(leadSlider.value)];
    const activeIdx = primLeads.indexOf(lead);
    const now = series[0].points[activeIdx >= 0 ? activeIdx : 0];
    if (now) {
      const from = CARDS[Math.round(((now.dir + 180) % 360) / 22.5) % 16];
      meteoSub.textContent =
        `${(Math.round(convert(now.speed) * 10) / 10)} ${UNITS[currentUnit()].label}` +
        ` · from ${from} · ${series[0].label}`;
    } else meteoSub.textContent = series[0].label;

    renderMeteogram(meteoChart, {
      times, series, sigma,
      convert: ms => convert(ms),
      unitLabel: UNITS[currentUnit()].label,
      activeIdx: activeIdx >= 0 ? activeIdx : null,
      onPickTime: i => {
        const idx = leads.indexOf(primLeads[i]);
        if (idx >= 0) { leadSlider.value = String(idx); render(); }
      },
    });
    meteoNote.textContent = live
      ? 'Live forecasts, unverified. The band is the std of the current on-cycle members.'
      : 'Model vs the ERA5 reanalysis it is scored against (2020 test year).';
  }

  map.on('click', ev => {
    meteoPoint = ev.latlng;
    if (meteoMarker) meteoMarker.remove();
    meteoMarker = L.circleMarker(ev.latlng, {
      radius: 5, color: '#58a6ff', weight: 2, fillColor: '#0b1020', fillOpacity: 0.7,
    }).addTo(map);
    meteo.hidden = false;
    drawMeteogram();
  });
  document.getElementById('meteoClose').addEventListener('click', () => {
    meteo.hidden = true;
    if (meteoMarker) { meteoMarker.remove(); meteoMarker = null; }
    meteoPoint = null;
  });

  syncPickers();
  await render();
})();
