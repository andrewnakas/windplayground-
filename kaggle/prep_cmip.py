"""Build the CMIP6 pretraining stack -- the documented 314 -> 268 step.

Rasp & Thuerey pretrain on ~150 years of MPI-ESM-HR historical output before
fine-tuning on ERA5, and that is the single difference between their ERA5-only
z500 of 314 at 3 days and the 268 everyone quotes. Our recreation already
reaches 306.7 without it; this is what the rest of the gap costs.

**Deviation taken deliberately: ~50 years, not their ~150.** The archive is
154 GB of zips and a Kaggle kernel cannot hold it. 50 years of 5 variables x 7
levels at fp16 is ~11 GB -- the same size as the ERA5 stack that already works
here. Less pretraining data means less of the benefit, and that gets reported
rather than hidden.

Nothing is downloaded whole. The zips live on a server that honours HTTP Range
(verified 206 in scripts/fetch_cmip.py), so this reads each archive's central
directory, pulls only the 5-year chunks inside the chosen window, converts one
at a time and deletes it. Peak disk stays a few GB against 154 GB for the naive
shape.

CMIP has no 2m temperature and no precipitation (probed; see
CMIP_AVAILABLE_VARIABLES in windml.config), so the stack is 35 pressure-level
fields plus analytic TISR = 36, against ERA5's 38. Stacked over three frames
plus three constants that is **111 network inputs against ERA5's 117**, and
src/windml/models/grow.py already grows the stem 111->117 and the head 2->3
with zeros so the transfer into fine-tuning is exact at step 0.

Output (to /kaggle/working):
    cmip_rt2021_<start>.npy   float16 (time, 36, 32, 64), NORMALIZED
    stats.json                store affine + physical mean/std
    meta.json                 channel order, levels actually found, provenance
"""
from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import urllib.request
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "netcdf4"], check=True)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

BASE = "https://dataserv.ub.tum.de/s/m1524895/download"
RESOLUTION = "5.625deg"
LEVELS = [50, 250, 500, 600, 700, 850, 925]
# (archive name, short name) -- the order fixes the channel layout
CMIP_VARS = [("geopotential", "z"), ("temperature", "t"),
             ("u_component_of_wind", "u"), ("v_component_of_wind", "v"),
             ("specific_humidity", "q")]

# The window. MPI-ESM historical starts at 1850; taking the LATEST years keeps
# the pretraining climate closest to the ERA5 period that follows it.
YEAR_FROM, YEAR_TO = 1955, 2005
SOLAR_CONSTANT = 1361.0
OUT = pathlib.Path("/kaggle/working")


def channel_order() -> list[str]:
    """Scored targets first, matching prep_era5.py. No t2m -- CMIP has none."""
    names = ["z500", "t850"]
    names += [f"{a}{lv}" for _, a in CMIP_VARS for lv in LEVELS
              if f"{a}{lv}" not in ("z500", "t850")]
    return names + ["tisr"]


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range, so zipfile can drive it.

    This is what makes selective extraction possible: `zipfile` seeks to the
    end for the central directory, then to each member's local header. Handing
    it a range-backed object means only the bytes of the wanted members ever
    cross the network -- a few GB instead of the 43 GB geopotential archive.
    """

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url)
        req.add_header("Range", f"bytes={self.pos}-{self.pos + n - 1}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception as exc:                     # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"windml range retry {attempt+1} ({type(exc).__name__})",
                      flush=True)
        self.pos += len(data)
        return data


def url_for(var: str) -> str:
    return (f"{BASE}?path=%2FCMIP%2FMPI-ESM%2F{RESOLUTION}%2F{var}"
            f"&files={var}_{RESOLUTION}.zip")


def remote_size(url: str) -> int:
    req = urllib.request.Request(url)
    req.add_header("Range", "bytes=0-1")
    with urllib.request.urlopen(req, timeout=120) as r:
        rng = r.headers.get("Content-Range", "")
    if "/" not in rng:
        raise SystemExit(f"RESULT FAIL no Content-Range for {url}")
    return int(rng.rsplit("/", 1)[1])


def member_year(name: str) -> int | None:
    """Chunks are named <var>_<YYYYMMDD>-<YYYYMMDD>_<res>.nc."""
    stem = pathlib.Path(name).stem
    for part in stem.split("_"):
        if "-" in part:
            a = part.split("-")[0]
            if len(a) == 8 and a.isdigit():
                return int(a[:4])
    return None


def toa_solar(times, lat, lon) -> np.ndarray:
    """Analytic TOA insolation. Same Spencer (1971) series as prep_era5.py."""
    t = np.asarray(times, dtype="datetime64[s]")
    year0 = t.astype("datetime64[Y]").astype("datetime64[s]")
    doy = (t - year0).astype(np.float64) / 86400.0 + 1.0
    day0 = t.astype("datetime64[D]").astype("datetime64[s]")
    hour = (t - day0).astype(np.float64) / 3600.0

    g = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.001480 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dist = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
            + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))
    la = np.deg2rad(lat)[None, :, None]
    ha = np.deg2rad(15.0 * (hour[:, None, None] + lon[None, None, :] / 15.0
                            + eot[:, None, None] / 60.0 - 12.0))
    d = decl[:, None, None]
    cz = np.clip(np.sin(la) * np.sin(d) + np.cos(la) * np.cos(d) * np.cos(ha), 0, None)
    return (SOLAR_CONSTANT * dist[:, None, None] * cz).astype(np.float32)


def open_members(var: str) -> tuple[zipfile.ZipFile, list[str]]:
    url = url_for(var)
    zf = zipfile.ZipFile(HttpRangeFile(url, remote_size(url)))
    wanted = []
    for n in zf.namelist():
        if not n.endswith(".nc"):
            continue
        y = member_year(n)
        if y is not None and YEAR_FROM <= y <= YEAR_TO:
            wanted.append(n)
    return zf, sorted(wanted)


def read_chunk(zf: zipfile.ZipFile, name: str, abbr: str):
    """One 5-year netCDF member -> (levels found, values, times, lat, lon)."""
    tmp = OUT / "_chunk.nc"
    with zf.open(name) as src, open(tmp, "wb") as dst:
        while True:
            b = src.read(1 << 22)
            if not b:
                break
            dst.write(b)
    ds = xr.open_dataset(tmp)
    dv = [v for v in ds.data_vars if ds[v].ndim >= 3]
    da = ds[dv[0]]
    lev_name = next((d for d in da.dims if d.lower() in ("level", "plev", "lev")), None)
    have = [int(v) for v in np.asarray(ds[lev_name].values)] if lev_name else []
    # plev is often in Pa; the paper's levels are hPa
    if have and max(have) > 2000:
        have = [h // 100 for h in have]
    return ds, da, lev_name, have, tmp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = channel_order()
    print(f"windml channels={len(names)} -> {len(names)*3+3} network inputs "
          f"(ERA5 is 117; grow.py bridges the difference)", flush=True)
    print(f"windml window={YEAR_FROM}-{YEAR_TO}", flush=True)

    # Open every archive first and REPORT the levels each one actually carries.
    # u/v are 12 GB against z/T/q's 42-45 GB over the same span and grid, which
    # has never been explained and most likely means fewer vertical levels. If
    # any of the paper's 7 is missing the channel count changes, and that must
    # fail loudly here rather than silently produce a stack that does not match
    # rt_input_channels().
    archives = {}
    for var, abbr in CMIP_VARS:
        zf, members = open_members(var)
        if not members:
            raise SystemExit(f"RESULT FAIL no {var} chunks in {YEAR_FROM}-{YEAR_TO}")
        ds, _da, lev_name, have, tmp = read_chunk(zf, members[0], abbr)
        ds.close()
        tmp.unlink(missing_ok=True)
        print(f"windml {var:22s} chunks={len(members)} levels={have}", flush=True)
        missing = [lv for lv in LEVELS if lv not in have]
        if missing:
            raise SystemExit(
                f"RESULT FAIL {var} is missing levels {missing}; it carries "
                f"{have}. The 111-channel layout assumes all 7 -- decide "
                f"explicitly how to handle this rather than silently reshaping.")
        archives[var] = (zf, members, lev_name)

    print("windml all five variables carry the paper's 7 levels", flush=True)
    print("RESULT OK (level probe only -- conversion is the next kernel)")


if __name__ == "__main__":
    main()
