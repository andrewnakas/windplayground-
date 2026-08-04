"""Build the local ERA5 cache, normalization stats, and climatology.

Usage:
    python scripts/build_cache.py                  # full 1979-2020 cache
    python scripts/build_cache.py --years 2018 2020   # subset (e.g. smoke tests)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from windml.config import ARTIFACTS, DataConfig
from windml.data.build_cache import build_cache, load_years
from windml.data.climatology import compute_climatology, save_climatology
from windml.data.dataset import year_range_times
from windml.data.normalization import compute_stats, save_stats

STATS_PATH = ARTIFACTS / "data" / "stats.json"
CLIM_PATH = ARTIFACTS / "climatology" / "clim_train.npy"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("START", "END"))
    p.add_argument("--skip-derived", action="store_true", help="cache only, no stats/clim")
    args = p.parse_args()

    cfg = DataConfig()
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    else:
        years = list(range(cfg.train_years[0], cfg.test_years[1] + 1))
    build_cache(cfg, years)

    if args.skip_derived:
        return

    train_span = (
        (max(cfg.train_years[0], args.years[0]), min(cfg.train_years[1], args.years[1]))
        if args.years
        else cfg.train_years
    )
    print(f"computing stats + climatology on {train_span} ...")
    train = load_years(cfg, list(range(train_span[0], train_span[1] + 1)))
    train = np.asarray(train)
    stats = compute_stats(train)
    save_stats(stats, STATS_PATH)
    print(f"stats -> {STATS_PATH}")

    times = year_range_times(train_span)
    clim = compute_climatology(train, times)
    save_climatology(clim, CLIM_PATH)
    print(f"climatology {clim.shape} -> {CLIM_PATH}")


if __name__ == "__main__":
    main()
