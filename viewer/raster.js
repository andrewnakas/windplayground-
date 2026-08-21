/* global L */
// Scalar raster shading under the particle layer: speed underlay, error and
// spread views all render through this one class. The canvas lives in its own
// pane at z 360 -- above the land fill (350), below the particles (400) --
// and is drawn at HALF device resolution, CSS-upscaled: the browser's bilinear
// smoothing hides the downsampling and keeps a full redraw under ~5 ms for a
// 180x90 grid.

// A grid is {nx, ny, la1, lo1, dx, dy, values} -- row-major from the NW
// corner, longitudes 0..360 east, exactly the exported field layout.
export function gridFromRecord(rec, values) {
  const h = rec.header;
  return { nx: h.nx, ny: h.ny, la1: h.la1, lo1: h.lo1, dx: h.dx, dy: h.dy,
           values: values ?? Float64Array.from(rec.data) };
}

// Bilinear sample; the same function feeds the raster, the click readout,
// and the meteogram so they can never disagree. Global grids wrap in
// longitude; a regional grid (HRRR's CONUS box) must NOT wrap -- its east
// edge is not its west edge -- and anywhere outside it samples as NaN,
// which every consumer renders as "no data".
export function sampleGrid(g, lat, lon) {
  const cyclic = Math.abs(g.nx * g.dx - 360) < g.dx * 1.5;
  let iy = (g.la1 - lat) / g.dy;
  if (iy < 0 || iy > g.ny - 1) {
    if (!cyclic) return NaN;
    iy = iy < 0 ? 0 : g.ny - 1;          // clamp at the poles on global grids
  }
  const iy0 = Math.floor(iy), iy1 = Math.min(iy0 + 1, g.ny - 1), fy = iy - iy0;
  const ix = (((lon - g.lo1) % 360) + 360) % 360 / g.dx;
  if (!cyclic && ix > g.nx - 1) return NaN;
  const ix0 = Math.floor(ix) % g.nx;
  const ix1 = cyclic ? (ix0 + 1) % g.nx : Math.min(ix0 + 1, g.nx - 1);
  const fx = ix - Math.floor(ix);
  const v = g.values;
  const a = v[iy0 * g.nx + ix0], b = v[iy0 * g.nx + ix1];
  const c = v[iy1 * g.nx + ix0], d = v[iy1 * g.nx + ix1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// 256-entry RGBA lookup table from a list of hex color stops.
export function makeLUT(stops) {
  const rgb = stops.map(hexToRgb);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (rgb.length - 1);
    const k = Math.min(Math.floor(t), rgb.length - 2), f = t - k;
    for (let c = 0; c < 3; c++)
      lut[i * 3 + c] = rgb[k][c] * (1 - f) + rgb[k + 1][c] * f;
  }
  return lut;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export class RasterLayer extends L.Layer {
  // opts: { colorStops: [hex...], domain: [min, max], opacity }
  constructor(opts) {
    super();
    this._grid = null;
    this.setStyle(opts);
  }

  setStyle({ colorStops, domain, opacity }) {
    if (colorStops) this._lut = makeLUT(colorStops);
    if (domain) this._domain = domain;
    if (opacity != null) this._opacity = opacity;
    if (this._map) this._redraw();
    return this;
  }

  setGrid(grid) {
    this._grid = grid;
    if (this._map) this._redraw();
    return this;
  }

  onAdd(map) {
    this._map = map;
    let pane = map.getPane('rasterPane');
    if (!pane) {
      pane = map.createPane('rasterPane');
      pane.style.zIndex = 360;
      pane.style.pointerEvents = 'none';
    }
    this._canvas = document.createElement('canvas');
    this._canvas.style.imageRendering = 'auto';
    pane.appendChild(this._canvas);
    map.on('moveend zoomend resize', this._redraw, this);
    // Leaflet CSS-scales panes during the zoom animation; a screen-space
    // canvas would smear, so hide it and come back on the redraw after.
    map.on('zoomanim', this._hide, this);
    this._redraw();
    return this;
  }

  onRemove(map) {
    map.off('moveend zoomend resize', this._redraw, this);
    map.off('zoomanim', this._hide, this);
    this._canvas.remove();
    this._map = null;
    return this;
  }

  _hide() { if (this._canvas) this._canvas.style.opacity = '0'; }

  _redraw() {
    const map = this._map, canvas = this._canvas, g = this._grid;
    if (!map || !canvas) return;
    const size = map.getSize();
    // pin the canvas to the current viewport inside the moving pane
    L.DomUtil.setPosition(canvas, map.containerPointToLayerPoint([0, 0]));
    canvas.style.width = size.x + 'px';
    canvas.style.height = size.y + 'px';
    const w = Math.max(1, Math.ceil(size.x / 2));
    const h = Math.max(1, Math.ceil(size.y / 2));
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    canvas.style.opacity = '1';
    if (!g) return;

    // Web Mercator is separable: one lat per canvas row, one lon per column,
    // computed once each -- the inner loop is pure array math.
    const lats = new Float64Array(h), lons = new Float64Array(w);
    for (let cy = 0; cy < h; cy++)
      lats[cy] = map.containerPointToLatLng([0, cy * 2 + 1]).lat;
    for (let cx = 0; cx < w; cx++)
      lons[cx] = map.containerPointToLatLng([cx * 2 + 1, 0]).lng;

    const [d0, d1] = this._domain, lut = this._lut;
    const alpha = Math.round(this._opacity * 255);
    const img = ctx.createImageData(w, h);
    const px = img.data;
    let p = 0;
    for (let cy = 0; cy < h; cy++) {
      const lat = lats[cy];
      const clipped = lat > g.la1 || lat < g.la1 - (g.ny - 1) * g.dy;
      for (let cx = 0; cx < w; cx++, p += 4) {
        if (clipped) continue;                     // poleward of the grid
        const v = sampleGrid(g, lat, lons[cx]);
        if (Number.isNaN(v)) continue;
        let t = (v - d0) / (d1 - d0);
        if (t < 0) t = 0; else if (t > 1) t = 1;
        const i = (t * 255 | 0) * 3;
        px[p] = lut[i]; px[p + 1] = lut[i + 1]; px[p + 2] = lut[i + 2];
        px[p + 3] = alpha;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
}
