"""Fetch the published WB2 competitor forecasts at 64x32.

Usage: python scripts/fetch_competitors.py [--names graphcast_2020 hres_2020 ...]
"""
from __future__ import annotations

import argparse

from windml.config import DataConfig
from windml.data.competitors import COMPETITOR_URLS, fetch_competitor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", default=list(COMPETITOR_URLS))
    args = p.parse_args()
    cfg = DataConfig()
    for name in args.names:
        year = int(name.rsplit("_", 1)[-1])
        path = fetch_competitor(cfg, name, year)
        print(f"{name} -> {path}")


if __name__ == "__main__":
    main()
