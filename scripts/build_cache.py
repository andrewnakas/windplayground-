"""Build the local ERA5 cache, normalization stats, and climatology.

Usage:
    python scripts/build_cache.py                  # full 1979-2020 cache
    python scripts/build_cache.py --years 2018 2020   # subset (e.g. smoke tests)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataclasses import replace

from windml.config import ARTIFACTS, DataConfig
from windml.data.build_cache import build_cache, load_years, year_path
from windml.data.climatology import compute_climatology, save_climatology
from windml.data.dataset import year_range_times
from windml.data.normalization import compute_stats_streaming, save_stats

CLIM_PATH = ARTIFACTS / "climatology" / "clim_train.npy"
# climatology is only used for the 8 scored channels, so the core set owns it


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("START", "END"))
    p.add_argument("--skip-derived", action="store_true", help="cache only, no stats/clim")
    p.add_argument("--config", default=None,
                   help="training config yaml whose data: section selects grid/url")
    p.add_argument("--variable-set", default=None,
                   help="core | levels | rt2021 | rt2021_cmip (overrides --config)")
    args = p.parse_args()

    if args.config:
        from windml.config import Config

        cfg = Config.from_yaml(args.config).data
    else:
        cfg = DataConfig()
    if args.variable_set:
        cfg = replace(cfg, variable_set=args.variable_set)
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
    train_years = list(range(train_span[0], train_span[1] + 1))
    # one year at a time: concatenating 39 years of the 20-channel set is
    # ~8.7 GB and gets the process OOM-killed alongside a training run
    stats = compute_stats_streaming(
        (np.load(year_path(cfg, y), mmap_mode="r") for y in train_years), cfg.channels
    )
    save_stats(stats, cfg.stats_path)
    print(f"stats -> {cfg.stats_path}")

    if cfg.variable_set != "core":
        print("skipping climatology: the core set already provides it")
        return
    times = year_range_times(train_span)
    clim = compute_climatology(train, times)
    save_climatology(clim, CLIM_PATH)
    print(f"climatology {clim.shape} -> {CLIM_PATH}")


if __name__ == "__main__":
    main()
