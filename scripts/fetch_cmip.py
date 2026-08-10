#!/usr/bin/env python3
"""Download the CMIP6 pretraining archive Rasp & Thuerey used.

MPI-ESM-HR historical, regridded, from the WeatherBench data repository. This
is the data behind the difference between their ERA5-only z500 of 314 and their
pretrained 268 -- the whole reason the anchor was out of reach on ERA5 alone.

Only the five pressure-level variables exist in that archive. There is no
2m_temperature and no precipitation at either resolution (probed directly; see
CMIP_AVAILABLE_VARIABLES in windml.config), which is why pretraining runs on a
narrower input stack than fine-tuning and the surface channels get grown in
afterwards.

Sizing, measured (a plain HEAD returns nothing useful, but a Range request's
Content-Range header carries the total, which is how these were obtained):

    5.625deg   geopotential        46.4 GB   (exact, Content-Range)
               temperature         41.8 GB
               specific_humidity   45.0 GB
               u_component_of_wind 12.8 GB   (exact, Content-Range)
               v_component_of_wind 12.0 GB
               TOTAL              ~158 GB

    2.8125deg  ~4x that, roughly 600 GB -- out of scope for now.

So `--vars` is the real disk lever. Dropping specific_humidity saves 45 GB and
still covers both scored pressure-level targets; z and T alone are 85 GB. Note
that z/T/q are ~3.6x the size of u/v, which most likely means they are archived
on more vertical levels -- worth confirming after the first extract, since it
affects how many of the paper's 7 levels are actually available per variable.

Expect ~13 s of server-side latency before the first byte of any request, and
highly variable throughput (6-28 MB/s observed). Downloads resume on
interruption via HTTP Range, which the server honours (verified: 206).

Usage:
    python scripts/fetch_cmip.py --dest /data/cmip --resolution 5.625deg
    python scripts/fetch_cmip.py --dest /data/cmip --vars geopotential,temperature
    python scripts/fetch_cmip.py --dest /data/cmip --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://dataserv.ub.tum.de/s/m1524895/download"

# Verified present at both 2.8125deg and 5.625deg. Order puts the two variables
# that matter most for the scored targets first, so an interrupted or
# disk-limited run still yields something usable.
CMIP_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

CHUNK = 1 << 20  # 1 MiB


# The /download?path=... endpoint 303-redirects to WebDAV. Measured: going
# straight to the WebDAV URL is ~2x faster (1.5 MB/s against 717 kB/s), because
# every ranged request otherwise pays a redirect round trip. That matters a lot
# when a 46 GB archive is fetched in 8 MB pieces.
DAV = "https://dataserv.ub.tum.de/public.php/dav/files/m1524895"


def url_for(var: str, resolution: str) -> str:
    return f"{DAV}/CMIP/MPI-ESM/{resolution}/{var}/{var}_{resolution}.zip"


def legacy_url_for(var: str, resolution: str) -> str:
    """The redirecting endpoint, kept because it is what the dataset page gives."""
    return (
        f"{BASE}?path=%2FCMIP%2FMPI-ESM%2F{resolution}%2F{var}"
        f"&files={var}_{resolution}.zip"
    )


def remote_size(var: str, resolution: str, timeout: int = 90) -> int | None:
    """Total bytes, read from Content-Range on a 2-byte Range request.

    HEAD returns no Content-Length for these downloads and a plain GET streams
    without one, but a ranged GET answers `Content-Range: bytes 0-1/<total>`.
    That is the only way to know the size up front, which is what makes a disk
    precheck and a real progress percentage possible.
    """
    req = urllib.request.Request(url_for(var, resolution))
    req.add_header("Range", "bytes=0-1")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rng = resp.headers.get("Content-Range", "")
    except Exception:  # noqa: BLE001 - size is a nicety, not a requirement
        return None
    if "/" not in rng:
        return None
    try:
        return int(rng.rsplit("/", 1)[1])
    except ValueError:
        return None


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def download(var: str, resolution: str, dest: Path, timeout: int = 120) -> Path:
    """Fetch one variable's zip, resuming a partial file if present.

    ownCloud honours Range on these file downloads (verified: a range request
    returns 206), so an interrupted multi-GB transfer picks up where it
    stopped instead of starting over.
    """
    out = dest / f"{var}_{resolution}.zip"
    part = out.with_suffix(".zip.part")
    if out.exists():
        print(f"windml cmip.{var}=already_complete size={human(out.stat().st_size)}")
        return out

    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url_for(var, resolution))
    if start:
        req.add_header("Range", f"bytes={start}-")
        print(f"windml cmip.{var}=resuming_at {human(start)}")

    t0 = time.time()
    got = start
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if start and resp.status != 206:
            # Server ignored the range; restarting is the only correct move,
            # since appending a full body to a partial file corrupts the zip.
            print(f"windml cmip.{var}=range_ignored restarting")
            start, got = 0, 0
            part.unlink(missing_ok=True)
        mode = "ab" if start else "wb"
        with open(part, mode) as fh:
            last = t0
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                got += len(block)
                now = time.time()
                if now - last > 30:
                    rate = (got - start) / max(now - t0, 1e-6) / 2**20
                    print(f"windml cmip.{var}=downloading got={human(got)} "
                          f"rate={rate:.1f}MB/s", flush=True)
                    last = now

    # Only promote to the final name once the transfer completed, so a killed
    # run never leaves a truncated file that looks finished.
    part.rename(out)
    mins = (time.time() - t0) / 60
    print(f"windml cmip.{var}=done size={human(out.stat().st_size)} minutes={mins:.1f}")
    return out


def verify(path: Path) -> bool:
    """A truncated zip is the most likely corruption, and it is cheap to catch."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        if bad:
            print(f"windml verify.{path.name}=corrupt member={bad}")
            return False
    except zipfile.BadZipFile as exc:
        print(f"windml verify.{path.name}=badzip {exc}")
        return False
    print(f"windml verify.{path.name}=ok")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", required=True, help="directory for the zips")
    p.add_argument("--resolution", default="5.625deg",
                   choices=["5.625deg", "2.8125deg"])
    p.add_argument("--vars", default=",".join(CMIP_VARS),
                   help="comma-separated subset; the disk lever")
    p.add_argument("--dry-run", action="store_true",
                   help="print URLs and exit without downloading")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    wanted = [v.strip() for v in args.vars.split(",") if v.strip()]
    unknown = [v for v in wanted if v not in CMIP_VARS]
    if unknown:
        print(f"RESULT FAIL unknown variables {unknown}; choose from {CMIP_VARS}")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for v in wanted:
            print(f"windml url.{v}={url_for(v, args.resolution)}")
        for v in wanted:
            size = remote_size(v, args.resolution)
            print(f"windml size.{v}={human(size) if size else 'unknown'}")
        print("windml note=a plain GET streams without Content-Length; sizes "
              "come from Content-Range on a ranged request. 5.625deg total "
              "measures ~154GB across all five variables.")
        print("RESULT OK")
        return 0

    free = shutil.disk_usage(dest).free
    print(f"windml dest={dest} free={human(free)} vars={len(wanted)} "
          f"resolution={args.resolution}")

    # Size the whole job before starting it. Running out of disk 100 GB into a
    # multi-hour download is the expensive failure this prevents.
    sizes = {v: remote_size(v, args.resolution) for v in wanted}
    known = [s for s in sizes.values() if s]
    for v, s in sizes.items():
        already = (dest / f"{v}_{args.resolution}.zip")
        done = human(already.stat().st_size) if already.exists() else "-"
        print(f"windml size.{v}={human(s) if s else 'unknown'} have={done}")
    if known:
        need = sum(known) - sum(
            (dest / f"{v}_{args.resolution}.zip").stat().st_size
            for v in wanted if (dest / f"{v}_{args.resolution}.zip").exists()
        )
        print(f"windml need={human(max(need, 0))} free={human(free)}")
        if need > free:
            print(f"RESULT FAIL need {human(need)} but only {human(free)} free; "
                  f"drop a variable with --vars (specific_humidity is the "
                  f"largest at ~45GB and the least central)")
            return 1

    for var in wanted:
        try:
            path = download(var, args.resolution, dest)
        except Exception as exc:  # noqa: BLE001 - report and let the caller retry
            print(f"RESULT FAIL download of {var} failed: {type(exc).__name__}: {exc}")
            return 1
        if not args.no_verify and not verify(path):
            print(f"RESULT FAIL {path.name} did not verify; delete it and re-run")
            return 1

    total = sum((dest / f"{v}_{args.resolution}.zip").stat().st_size for v in wanted)
    print(f"windml cmip_total={human(total)}")
    print("RESULT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
