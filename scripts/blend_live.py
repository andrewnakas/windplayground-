"""Average the live operational forecasts into one multi-model ensemble.

Everything else in this repo is a hindcast: the blend that beats GraphCast is an
average of 2020 forecast *archives*. This does the same thing to today's
forecasts, so the viewer can show a current multi-model mean rather than only
replayed history.

Why a plain mean of AIFS and GFS is the right pairing rather than an arbitrary
one: the sharpness work in REPORT.md found that ML models are systematically
blurred at long leads and the physics model is not (at 120 h GraphCast retains
78% of its high-wavenumber power against HRES's 96%). AIFS is the operational
ML system and GFS is operational physics, so this averages one of each family --
which is where a plain mean has the most to gain, and it is the same zero-fitted-
parameter recipe that gave -12% z500 and -7% wind speed on the 2020 archives.

**No spectral recalibration here, deliberately.** Those amplification factors
were fitted against ERA5 truth for 2020 models. There is no truth available for
AIFS or GFS at today's dates -- WeatherBench-2's ERA5 ends in 2023 -- so
transferring factors across both model and year would be an unverified step, and
the live product would stop being something anyone can check. The live blend is
the plain mean.

Equally, the hindcast gain came from four members; two will give less, and none
of it is verified for these models at these dates. This is a live ensemble, not
a live result.

    python scripts/blend_live.py
    python scripts/blend_live.py --members aifs_live gfs_live
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path("docs/data")
BLEND_ID = "live_blend"
SPREAD_ID = "live_spread"
# the grid fields that must agree before two forecasts can be averaged
GRID_KEYS = ("nx", "ny", "la1", "lo1", "la2", "lo2", "dx", "dy", "forecastTime")


def load(source_id: str, lead: int) -> list | None:
    p = OUT_DIR / f"{source_id}_latest_{lead:03d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def member_init(source_id: str, leads: list[int]) -> str | None:
    """The one init a member carries across all its leads, or None if absent.

    A member whose own files disagree is broken rather than stale -- a fetch
    that died halfway leaves lead 24 on the new cycle and lead 96 on the old --
    so that stays fatal. Disagreement *between* members is the ordinary case
    handled by align_members().
    """
    refs = set()
    for lead in leads:
        rec = load(source_id, lead)
        if rec is None:
            continue
        refs |= {c["header"]["refTime"] for c in rec}
    if not refs:
        return None
    if len(refs) > 1:
        raise SystemExit(
            f"{source_id} carries more than one init across its own leads: "
            f"{sorted(refs)}. That is a half-finished fetch, not a stale "
            f"member -- re-run the fetch for {source_id} before blending.")
    return refs.pop()


def align_members(members: list[str], leads: list[int]) -> list[str]:
    """Keep only the members on the newest init, and say which were dropped.

    Averaging two different cycles is not an ensemble, which is why this file
    refuses to do it. It used to refuse by aborting -- but the members do not
    all refresh on the same clock: AIFS and GFS come from dynamical.org and the
    6-hourly workflow re-fetches them, while WeatherNext needs a credentialed
    BigQuery export that only runs from a developer machine. Aborting meant one
    unavoidably-lagging member froze the whole live blend, including the two
    that had fresh data.

    So the guard's intent is kept exactly -- nothing off-cycle is ever averaged
    -- and only the response changes: drop the laggard, name it, carry on.
    """
    inits = {m: member_init(m, leads) for m in members}
    present = {m: r for m, r in inits.items() if r is not None}
    for m, r in inits.items():
        if r is None:
            print(f"windml {m}: no files on disk, not a member of this blend")
    if not present:
        raise SystemExit("no live member has any data on disk")

    newest = max(present.values())
    aligned = [m for m in members if present.get(m) == newest]
    for m in members:
        if m in present and m not in aligned:
            print(f"windml EXCLUDED {m}: init {present[m]} is behind the "
                  f"newest live init {newest}. It is not averaged in -- "
                  f"re-fetch it to bring it back into the blend.")
    if len(aligned) < 2:
        raise SystemExit(
            f"only {len(aligned)} member(s) on the newest init {newest}: "
            f"{aligned}. A 'multi-model mean' of one model is not one; "
            f"re-fetch the others onto this cycle.")
    return aligned


def check_compatible(recs: list[tuple[str, list]]) -> str:
    """All members must share a grid AND an init time. Returns that init time.

    Off-cycle members are already removed by align_members(), so reaching the
    init branch here means something inconsistent survived that filter. It is
    kept as a real check rather than an assertion because the cost of being
    wrong is silent: filenames are stable (`aifs_live_latest_072.json` is
    overwritten in place), so a stale file looks perfectly valid, and averaging
    a 12Z forecast with yesterday's 00Z would simply be quietly wrong.
    """
    ref0 = None
    grid0 = None
    for name, rec in recs:
        for comp in rec:
            h = comp["header"]
            grid = tuple(h[k] for k in GRID_KEYS)
            if grid0 is None:
                grid0, ref0, first = grid, h["refTime"], name
            if grid != grid0:
                raise SystemExit(
                    f"grid mismatch: {name} {grid} != {first} {grid0}; "
                    f"regridding is out of scope, fix the export instead")
            if h["refTime"] != ref0:
                raise SystemExit(
                    f"init time mismatch: {name} is {h['refTime']} but {first} "
                    f"is {ref0}. One of the fetches is stale -- averaging "
                    f"different cycles would not be an ensemble.")
    return ref0


def blend_lead(members: list[str], lead: int,
               blend_id: str = BLEND_ID, spread_id: str = SPREAD_ID) -> str | None:
    recs = [(m, r) for m in members if (r := load(m, lead)) is not None]
    if len(recs) < 2:
        return None
    ref = check_compatible(recs)

    out = []
    for comp in range(len(recs[0][1])):                    # u record, then v
        stacks = [r[comp]["data"] for _, r in recs]
        n = len(stacks)
        # 2 decimals: the members themselves carry 1, so a third decimal on
        # their mean is file weight, not information
        mean = [round(sum(vals) / n, 2) for vals in zip(*stacks)]
        header = dict(recs[0][1][comp]["header"])
        out.append({"header": header, "data": mean})

    (OUT_DIR / f"{blend_id}_latest_{lead:03d}.json").write_text(
        json.dumps(out, separators=(",", ":")))

    # Ensemble spread, exported HERE because this is the one place where
    # "which members are on-cycle" is already resolved -- the viewer must
    # never re-derive membership. Per gridpoint: population std of the
    # members' wind SPEEDS (std of components would hide direction-only
    # disagreement in the quantity people actually read off the map).
    us = [r[0]["data"] for _, r in recs]
    vs = [r[1]["data"] for _, r in recs]
    n = len(recs)
    spread = []
    for i in range(len(us[0])):
        sp = [math.hypot(us[m][i], vs[m][i]) for m in range(n)]
        mu = sum(sp) / n
        spread.append(round(math.sqrt(sum((s - mu) ** 2 for s in sp) / n), 1))
    header = dict(recs[0][1][0]["header"])
    header["parameterNumberName"] = "wind_speed_stddev"
    header["parameterUnit"] = "m.s-1"
    header["members"] = [m for m, _ in recs]
    (OUT_DIR / f"{spread_id}_latest_{lead:03d}.json").write_text(
        json.dumps([{"header": header, "data": spread}], separators=(",", ":")))
    return ref


def age_hours(ref_time: str) -> float:
    """Hours between a member's init and now, both UTC."""
    t = datetime.strptime(ref_time.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
    return (datetime.now(timezone.utc).replace(tzinfo=None) - t).total_seconds() / 3600.0


def main() -> None:
    global OUT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=None,
                   help="source ids to average; default = every live source")
    p.add_argument("--out", default=str(OUT_DIR))
    # The failure this exists for: the 6-hourly refresh workflow stopped firing
    # and nobody noticed for twelve days. Every guard in this file compares
    # members to EACH OTHER, so a refresh that fetches nothing leaves a set of
    # mutually consistent, equally stale files and passes every check. Nothing
    # anywhere compared the data to the clock. 18 h is a little over two missed
    # 6-hourly cycles -- late enough not to trip on a single upstream hiccup,
    # early enough that a stopped schedule shows up the same day.
    p.add_argument("--max-age-h", type=float, default=18.0,
                   help="fail if the members' init is older than this; 0 disables")
    a = p.parse_args()
    OUT_DIR = Path(a.out)
    man_path = OUT_DIR / "manifest.json"
    man = json.loads(man_path.read_text())

    # 10 m and 100 m are separate products with separate members; averaging
    # across heights would be meaningless, so each level blends on its own.
    blended_any = False
    newest_ref = None
    for level, blend_id, spread_id in (("10m", BLEND_ID, SPREAD_ID),
                                       ("100m", BLEND_ID + "100", SPREAD_ID + "100")):
        live = [s for s in man["sources"]
                if s.get("kind") == "live"
                and s.get("level", "10m") == level
                and not s.get("domain")     # regional grids cannot join a
                                            # global mean (CONUS != the world)
                and s["id"] not in (BLEND_ID, BLEND_ID + "100")]
        members = a.members or [s["id"] for s in live]
        members = [m for m in members if m in {s["id"] for s in live}]
        if len(members) < 2:
            if level == "10m":
                raise SystemExit(f"need >=2 live sources to blend, found {members}")
            continue                     # no 100 m fleet yet: nothing to do
        leads = sorted({ld for s in live if s["id"] in members for ld in s["leads"]})
        print(f"windml level={level} members={members} candidate_leads={leads}")
        members = align_members(members, leads)
        print(f"windml level={level} blending members={members}")

        done, ref = [], None
        for lead in leads:
            r = blend_lead(members, lead, blend_id, spread_id)
            if r:
                done.append(lead)
                ref = r
        if not done:
            if level == "10m":
                raise SystemExit("no lead had >=2 members present; nothing blended")
            continue
        blended_any = True
        newest_ref = ref

        labels = [s["label"].split("(")[0].strip() for s in live if s["id"] in members]
        entry = {
            "id": blend_id,
            "base": BLEND_ID,
            "label": (f"Live multi-model mean ({' + '.join(labels)})" if level == "10m"
                      else f"Live multi-model mean @100 m ({' + '.join(labels)})"),
            "kind": "live",
            "level": level,
            "inits": ["latest"],
            "leads": done,
            "init_time": ref,
            "members": members,
            "spread_leads": done,
        }
        man["sources"] = [s for s in man["sources"] if s["id"] != blend_id] + [entry]
        print(f"windml blended level={level} leads={done} init={ref} -> {blend_id}")

    if not blended_any:
        raise SystemExit("nothing blended at any level")
    man_path.write_text(json.dumps(man, indent=2))
    ref = newest_ref

    if a.max_age_h > 0 and ref:
        age = age_hours(ref)
        print(f"windml init age = {age:.1f} h")
        if age > a.max_age_h:
            raise SystemExit(
                f"STALE: the newest init every live member carries is {ref}, "
                f"{age:.1f} h old (limit {a.max_age_h:.0f} h).\n"
                f"The blend itself is fine -- the members agree -- so this is "
                f"not a data error, it is the refresh not having happened. "
                f"Check that .github/workflows/live-wind.yml is still being "
                f"scheduled, and that dynamical.org's latest.zarr has moved.\n"
                f"The JSON written above is valid and the site will still "
                f"build; pass --max-age-h 0 to publish it deliberately.")


if __name__ == "__main__":
    main()
