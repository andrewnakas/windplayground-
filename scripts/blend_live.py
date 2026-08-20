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
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path("viewer/data")
BLEND_ID = "live_blend"
# the grid fields that must agree before two forecasts can be averaged
GRID_KEYS = ("nx", "ny", "la1", "lo1", "la2", "lo2", "dx", "dy", "forecastTime")


def load(source_id: str, lead: int) -> list | None:
    p = OUT_DIR / f"{source_id}_latest_{lead:03d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def check_compatible(recs: list[tuple[str, list]]) -> str:
    """All members must share a grid AND an init time. Returns that init time.

    The init check is the one that actually matters. Filenames are stable
    (`aifs_live_latest_072.json` is overwritten in place), so if one fetch fails
    the previous cycle's file stays on disk looking perfectly valid. Averaging a
    12Z forecast with yesterday's 00Z is not an ensemble, and nothing downstream
    would reveal it -- the blend would simply be quietly wrong.
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


def blend_lead(members: list[str], lead: int) -> str | None:
    recs = [(m, r) for m in members if (r := load(m, lead)) is not None]
    if len(recs) < 2:
        return None
    ref = check_compatible(recs)

    out = []
    for comp in range(len(recs[0][1])):                    # u record, then v
        stacks = [r[comp]["data"] for _, r in recs]
        n = len(stacks)
        mean = [round(sum(vals) / n, 3) for vals in zip(*stacks)]
        header = dict(recs[0][1][comp]["header"])
        out.append({"header": header, "data": mean})

    (OUT_DIR / f"{BLEND_ID}_latest_{lead:03d}.json").write_text(
        json.dumps(out, separators=(",", ":")))
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

    live = [s for s in man["sources"]
            if s.get("kind") == "live" and s["id"] != BLEND_ID]
    members = a.members or [s["id"] for s in live]
    if len(members) < 2:
        raise SystemExit(f"need >=2 live sources to blend, found {members}")
    leads = sorted({ld for s in live if s["id"] in members for ld in s["leads"]})
    print(f"windml members={members} candidate_leads={leads}")

    done, ref = [], None
    for lead in leads:
        r = blend_lead(members, lead)
        if r:
            done.append(lead)
            ref = r
    if not done:
        raise SystemExit("no lead had >=2 members present; nothing blended")

    labels = [s["label"].split("(")[0].strip() for s in live if s["id"] in members]
    entry = {
        "id": BLEND_ID,
        "label": f"Live multi-model mean ({' + '.join(labels)})",
        "kind": "live",
        "inits": ["latest"],
        "leads": done,
        "init_time": ref,
        "members": members,
    }
    man["sources"] = [s for s in man["sources"] if s["id"] != BLEND_ID] + [entry]
    man_path.write_text(json.dumps(man, indent=2))
    print(f"windml blended leads={done} init={ref} -> {BLEND_ID}")

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
