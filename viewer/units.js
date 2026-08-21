// Wind speed display units. Data is m/s everywhere internally; conversion
// happens only at the last moment, when a number is put in front of the user.

export const UNITS = {
  ms:  { label: 'm/s',  factor: 1,        decimals: 1 },
  kn:  { label: 'kn',   factor: 1.943844, decimals: 0 },
  kmh: { label: 'km/h', factor: 3.6,      decimals: 0 },
  mph: { label: 'mph',  factor: 2.236936, decimals: 0 },
};

const KEY = 'wind.units';

export function currentUnit() {
  const u = localStorage.getItem(KEY);
  return u in UNITS ? u : 'ms';
}

export function setUnit(u) {
  if (u in UNITS) localStorage.setItem(KEY, u);
}

export function convert(ms, u = currentUnit()) {
  return ms * UNITS[u].factor;
}

export function format(ms, u = currentUnit()) {
  const spec = UNITS[u];
  return `${(ms * spec.factor).toFixed(spec.decimals)} ${spec.label}`;
}

// A tick value that is round in the DISPLAY unit, for legends and axes.
export function niceTicks(maxMs, count, u = currentUnit()) {
  const max = convert(maxMs, u);
  const step = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step)));
  const nice = [1, 2, 2.5, 5, 10].map(m => m * mag)
    .reduce((a, b) => Math.abs(b - step) < Math.abs(a - step) ? b : a);
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += nice) ticks.push(v);
  return ticks;
}
