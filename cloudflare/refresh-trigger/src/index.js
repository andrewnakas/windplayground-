// Event-driven refresh trigger for the wind playground.
//
// GitHub's own cron is best-effort -- it queued nothing for two weeks once,
// and even now fires up to an hour late on a fixed 4x/day grid. This Worker
// turns refresh into an event: every 20 minutes it asks "has any model
// actually published a cycle the deployed site does not have?", and only
// then dispatches the live-wind workflow. Models publish staggered (AIFS
// ~4-5 h after init, IFS ~7 h, HRRR ~1 h), so the site now follows each
// publication instead of a clock.

const UA = { "User-Agent": "windplayground-refresh-worker" };

function fmtCycle(d) {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  const h = String(d.getUTCHours()).padStart(2, "0");
  return { ymd: `${y}${m}${day}`, hh: h, iso: `${y}-${m}-${day}T${h}:00:00Z` };
}

// last N six-hourly cycle datetimes, newest first
function recentCycles(n, stepH) {
  const out = [];
  const now = new Date();
  const floored = new Date(Math.floor(now.getTime() / (stepH * 3600e3)) * stepH * 3600e3);
  for (let i = 0; i < n; i++)
    out.push(new Date(floored.getTime() - i * stepH * 3600e3));
  return out;
}

async function head(url) {
  try {
    const r = await fetch(url, { method: "HEAD", headers: UA });
    return r.ok;
  } catch (e) {
    return false;
  }
}

// newest published ECMWF open-data cycle for a model ('' if none of the last 4)
async function newestEcmwf(model) {
  for (const d of recentCycles(4, 6)) {
    const c = fmtCycle(d);
    const url = `https://data.ecmwf.int/forecasts/${c.ymd}/${c.hh}z/${model}/0p25/oper/` +
                `${c.ymd}${c.hh}0000-0h-oper-fc.index`;
    if (await head(url)) return c.iso;
  }
  return "";
}

// newest COMPLETE HRRR cycle (the +48 h pressure file is the completion marker)
async function newestHrrr() {
  for (const d of recentCycles(8, 1)) {
    const c = fmtCycle(d);
    const url = `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.${c.ymd}/conus/` +
                `hrrr.t${c.hh}z.wrfprsf48.grib2.idx`;
    if (await head(url)) return c.iso;
  }
  return "";
}

async function deployedInits(env) {
  const r = await fetch(`${env.SITE}/data/manifest.json`, {
    headers: UA, cf: { cacheTtl: 0 } });
  if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
  const man = await r.json();
  const by = Object.fromEntries(man.sources.map(s => [s.id, s]));
  const globals = ["aifs_live", "ifs_live", "gfs_live"]
    .map(id => by[id]?.init_time || "").filter(Boolean);
  return {
    global: globals.length ? globals.sort().at(-1) : "",
    hrrr: by.hrrr_live?.init_time || "",
  };
}

async function lastRunAt(env) {
  const r = await fetch(
    `https://api.github.com/repos/${env.REPO}/actions/workflows/${env.WORKFLOW}/runs?per_page=1`,
    { headers: { ...UA, Authorization: `Bearer ${env.GH_TOKEN}`,
                 Accept: "application/vnd.github+json" } });
  if (!r.ok) return null;
  const j = await r.json();
  return j.workflow_runs?.[0]?.created_at ?? null;
}

async function dispatch(env) {
  const r = await fetch(
    `https://api.github.com/repos/${env.REPO}/actions/workflows/${env.WORKFLOW}/dispatches`,
    { method: "POST",
      headers: { ...UA, Authorization: `Bearer ${env.GH_TOKEN}`,
                 Accept: "application/vnd.github+json" },
      body: JSON.stringify({ ref: "main" }) });
  return r.status; // 204 on success
}

async function check(env, dryRun = false) {
  const hoursOld = iso => (Date.now() - Date.parse(iso)) / 3600e3;
  const deployed = await deployedInits(env);
  const [aifs, ifs, hrrr] = await Promise.all(
    [newestEcmwf("aifs-single"), newestEcmwf("ifs"), newestHrrr()]);
  const upstreamGlobal = [aifs, ifs].filter(Boolean).sort().at(-1) || "";

  const reasons = [];
  if (upstreamGlobal && deployed.global && upstreamGlobal > deployed.global)
    reasons.push(`global upstream ${upstreamGlobal} > deployed ${deployed.global}`);
  if (hrrr && deployed.hrrr &&
      Date.parse(hrrr) - Date.parse(deployed.hrrr) >= Number(env.HRRR_STALE_H) * 3600e3)
    reasons.push(`hrrr upstream ${hrrr} vs deployed ${deployed.hrrr}`);
  if (deployed.global && hoursOld(deployed.global) > Number(env.MAX_AGE_H))
    reasons.push(`deployed global init ${hoursOld(deployed.global).toFixed(1)} h old`);

  const status = { deployed, upstream: { aifs, ifs, hrrr }, reasons };
  if (!reasons.length) return { ...status, action: "fresh enough" };
  if (dryRun) return { ...status, action: "would dispatch (dry run)" };

  const last = await lastRunAt(env);
  if (last && (Date.now() - Date.parse(last)) < Number(env.MIN_INTERVAL_MIN) * 60e3)
    return { ...status, action: `holding: last run ${last} within ${env.MIN_INTERVAL_MIN} min` };

  const code = await dispatch(env);
  return { ...status, action: code === 204 ? "dispatched" : `dispatch failed ${code}` };
}

export default {
  async scheduled(event, env, ctx) {
    const result = await check(env);
    console.log(JSON.stringify(result));
  },
  // GET the worker for a dry status view (no token needed to look, only to act)
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname !== "/") return new Response("not found", { status: 404 });
    // the public URL is a window, not a lever: it never dispatches
    const result = await check(env, true);
    return new Response(JSON.stringify(result, null, 2), {
      headers: { "content-type": "application/json" } });
  },
};
