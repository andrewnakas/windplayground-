(async function () {
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
    const inGroup = manifest.sources.filter(s => s.kind === kind);
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

    updateTruthBtn();
    updateAgeChip(src);
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

  async function loadField(sourceId, init, lead) {
    const key = `${sourceId}_${init}_${String(lead).padStart(3, '0')}`;
    if (cache.has(key)) return cache.get(key);
    const res = await fetch(`data/${key}.json`);
    if (!res.ok) throw new Error(`no field for ${sourceId} at +${lead} h`);
    const data = await res.json();
    cache.set(key, data);
    return data;
  }

  async function render() {
    const sourceId = sourceSel.value;
    const init = initSel.value;
    const lead = leads[Number(leadSlider.value)];
    leadOut.textContent = `+${lead} h`;
    try {
      const data = await loadField(sourceId, init, lead);
      if (velocityLayer) map.removeLayer(velocityLayer);
      velocityLayer = L.velocityLayer({
        displayValues: true,
        displayOptions: {
          velocityType: '10 m wind',
          position: 'bottomleft',
          emptyString: 'hover the map for a wind readout',
          angleConvention: 'bearingCW',
          speedUnit: 'm/s',
        },
        data,
        maxVelocity: 25,
        velocityScale: 0.012,
        particleAge: 70,
        lineWidth: 1.4,
        particleMultiplier: 1 / 260,
        colorScale: ['#3288bd','#66c2a5','#abdda4','#e6f598','#fee08b','#fdae61','#f46d43','#d53e4f'],
      });
      velocityLayer.addTo(map);
      const src = byId[sourceId];
      const nx = data[0].header.nx, ny = data[0].header.ny;
      const degrees = (360 / nx).toFixed(2).replace(/\.?0+$/, '');
      document.getElementById('sub').innerHTML =
        `10&nbsp;m wind &middot; ${nx}&times;${ny} grid (${degrees}&deg;) &middot; ` +
        (src.kind === 'live' ? 'live operational run' : '2020 test year');
      say(lead === 0 ? `${src.label} — analysis at ${init}`
                     : `${src.label} — +${lead} h from ${init}`);
    } catch (e) {
      say(e.message, true);
    }
  }

  sourceSel.addEventListener('change', () => { resetTruthFlip(); syncPickers(); render(); });
  initSel.addEventListener('change', render);
  leadSlider.addEventListener('input', render);

  // Flip to ERA5 and back, so forecast error is visible by eye.
  const truthBtn = document.getElementById('truthBtn');
  let previous = null;

  // A flip only makes sense where truth exists: the source and ERA5 must share
  // an init. Live runs are from today and ERA5's archive is 2020, so for them
  // the button is disabled rather than left to error out on a missing file.
  function updateTruthBtn() {
    if (previous !== null) { truthBtn.disabled = false; return; }
    const era5 = byId['era5'];
    const src = byId[sourceSel.value];
    const shared = era5 && src && src.id !== 'era5'
      && (src.inits || manifest.inits).some(t => (era5.inits || manifest.inits).includes(t));
    truthBtn.disabled = !shared;
    truthBtn.title = shared ? '' : 'no verifying truth exists yet for a live run';
  }

  function resetTruthFlip() {
    previous = null;
    truthBtn.classList.remove('on');
    truthBtn.textContent = 'Compare vs ERA5';
  }

  truthBtn.addEventListener('click', () => {
    if (previous === null) {
      if (sourceSel.value === 'era5') { say('Already showing ERA5 — pick a model to compare.'); return; }
      previous = sourceSel.value;
      sourceSel.value = 'era5';
      truthBtn.classList.add('on');
      truthBtn.textContent = '↩ Back to model';
    } else {
      sourceSel.value = previous;
      resetTruthFlip();
    }
    // the pickers follow the source, so a flip must re-sync them too --
    // otherwise a live source's "latest" init leaks into the ERA5 fetch
    syncPickers();
    render();
  });

  const playBtn = document.getElementById('playBtn');
  let timer = null;
  playBtn.addEventListener('click', () => {
    if (timer) {
      clearInterval(timer); timer = null;
      playBtn.classList.remove('on'); playBtn.textContent = '▶ Animate leads';
      return;
    }
    playBtn.classList.add('on'); playBtn.textContent = '❚❚ Stop';
    timer = setInterval(() => {
      const next = (Number(leadSlider.value) + 1) % leads.length;
      leadSlider.value = String(next);
      render();
    }, 1600);
  });

  syncPickers();
  await render();
})();
