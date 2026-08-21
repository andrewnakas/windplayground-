"""Export wind fields to leaflet-velocity JSON for the map viewer.

leaflet-velocity expects the grib2json shape: a two-element array holding the
u and v records, each with a header describing the grid and a flat `data`
array in row-major order starting at the NORTH-WEST corner.

Our ERA5 grid runs south -> north and 0 -> 354.375 degrees east, so latitudes
are flipped on export and la1 is the northern edge.

Usage:
    python scripts/export_wind.py                       # default init times
    python scripts/export_wind.py --inits 2020-01-15T00 2020-10-01T00
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from windml.config import ARTIFACTS, CHANNELS, DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.competitors import CompetitorForecaster
from windml.data.dataset import Era5Dataset, year_range_times
from windml.data.normalization import Normalizer, load_stats
from windml.eval.forecasters import ModelForecaster, PersistenceForecaster

U10, V10 = CHANNELS.index("u10"), CHANNELS.index("v10")
STATS_PATH = ARTIFACTS / "data" / "stats.json"
OUT_DIR = Path("viewer/data")

# Every trained checkpoint the picker offers, newest lineage last. A model only
# appears if its best.pt exists, so a partly-trained repo still exports cleanly.
TRAINED = [
    ("unet", "U-Net (baseline)"),
    ("unet_ft2", "U-Net + rollout K=2"),
    ("unet_ft4", "U-Net + rollout K=4"),
    ("unet_long", "U-Net (long schedule)"),
    ("unet_long_ft2", "U-Net long + rollout K=2"),
    ("unet_long_ft4", "U-Net long + rollout K=4"),
    ("vit", "Stormer-style ViT"),
    ("vit_ft2", "ViT + rollout K=2"),
    ("afno", "AFNO (FourCastNet)"),
    ("afno_ft2", "AFNO + rollout K=2"),
    ("graph", "GraphCast-mini GNN"),
    ("levels72", "vertical inputs, direct 72h"),
    ("levels120", "vertical inputs, direct 120h"),
    ("anchor72", "anchor attempt (6.6M, z500-weighted)"),
    ("resnet72", "full-res ResNet, direct 72h"),
    ("resnet72_big", "full-res ResNet 1.7M, direct 72h"),
]

# what the model picker offers: (id, label, kind, spec)
SOURCES = (
    [("era5", "ERA5 (truth)", "truth", None)]
    + [
        (rid, f"ours: {label}", "ckpt",
         ARTIFACTS / "checkpoints" / rid / "best.pt")
        for rid, label in TRAINED
        if (ARTIFACTS / "checkpoints" / rid / "best.pt").exists()
    ]
    + [
        ("graphcast", "GraphCast (DeepMind)", "competitor", "graphcast_2020"),
        ("gencast_mean", "GenCast ensemble mean (DeepMind)", "competitor",
         "gencast_mean_2020"),
        ("pangu", "Pangu-Weather (Huawei)", "competitor", "pangu_2020"),
        ("hres", "ECMWF HRES", "competitor", "hres_2020"),
        ("fuxi", "FuXi", "competitor", "fuxi_2020"),
        ("persistence", "persistence", "persistence", None),
    ]
)


def scored_rmse(model_id: str, variable: str = "wind_speed",
                lead_h: int = 72) -> float | None:
    """72h wind-speed RMSE for a model, or None if it was never scored.

    Shown beside each name in the picker so choosing a model and judging it
    happen in the same place. Read from the results CSVs rather than hardcoded,
    for the same reason the landing page does: the site cannot then disagree
    with the numbers it publishes.
    """
    path = ARTIFACTS / "results" / f"{model_id}_test.csv"
    if not path.exists():
        return None
    import csv as _csv

    with open(path) as fh:
        for row in _csv.DictReader(fh):
            if row["variable"] == variable and int(row["lead_h"]) == lead_h:
                try:
                    return float(row["rmse"])
                except (ValueError, TypeError):
                    return None
    return None


def velocity_records(u: np.ndarray, v: np.ndarray, lat: np.ndarray,
                     lon: np.ndarray, ref_time: str, forecast_hours: int) -> list:
    """(u, v) fields on our grid -> the two-record leaflet-velocity payload."""
    order = np.argsort(-lat)  # north first
    u_n, v_n = u[order], v[order]
    lat_sorted = lat[order]
    dy = float(abs(lat_sorted[0] - lat_sorted[1]))
    dx = float(abs(lon[1] - lon[0]))

    def header(param_number: int) -> dict:
        return {
            "parameterUnit": "m.s-1",
            "parameterNumber": param_number,   # 2 = U component, 3 = V
            "parameterNumberName": "U-component_of_wind" if param_number == 2
                                   else "V-component_of_wind",
            "parameterCategory": 2,            # momentum
            "nx": len(lon),
            "ny": len(lat),
            "lo1": float(lon[0]),
            "la1": float(lat_sorted[0]),
            "lo2": float(lon[-1]),
            "la2": float(lat_sorted[-1]),
            "dx": dx,
            "dy": dy,
            "refTime": ref_time,
            "forecastTime": forecast_hours,
        }

    return [
        {"header": header(2), "data": [round(float(x), 3) for x in u_n.ravel()]},
        {"header": header(3), "data": [round(float(x), 3) for x in v_n.ravel()]},
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inits", nargs="+",
                   default=["2020-01-15T00", "2020-07-15T00", "2020-10-28T00"],
                   help="init times inside the 2020 test year")
    p.add_argument("--leads", nargs="+", type=int, default=[0, 24, 48, 72, 96, 120])
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()

    torch.set_num_threads(1)  # stay out of the training run's way
    cfg = DataConfig()
    statics = load_statics(cfg)
    lat, lon = statics["latitude"], statics["longitude"]
    truth = np.asarray(load_years(cfg, [2020]), dtype=np.float32)
    times = year_range_times((2020, 2020))
    # each checkpoint now brings its own normalizer, built from the stats of
    # the variable set it was trained on -- a single shared one was what made
    # the multi-level models unloadable here

    init_idx = {}
    for s in args.inits:
        t = np.datetime64(s, "h")
        hits = np.where(times == t)[0]
        if not len(hits):
            raise SystemExit(f"init time {s} not on the 6-hourly 2020 grid")
        init_idx[s] = int(hits[0])

    max_lead = max(args.leads)
    K = max_lead // 6
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"inits": args.inits, "leads": args.leads, "sources": []}

    for src_id, label, kind, spec in SOURCES:
        forecaster = None
        if kind == "ckpt":
            if not Path(spec).exists():
                print(f"skip {src_id}: no checkpoint yet")
                continue
            payload = torch.load(spec, map_location="cpu", weights_only=False)
            from windml.eval.forecasters import DirectForecaster, scored_channel_map
            from windml.models import build_model

            # Each checkpoint carries the variable set it was trained on, and
            # they differ: 'core' is 25 inputs / 8 outputs, 'levels' is 49 / 20.
            # Building every model against the default config loaded a
            # 49-channel checkpoint into a 25-channel model and blew up on
            # levels72. Mirror what scripts/evaluate.py does -- dataset first,
            # because only it knows the input width.
            mcfg = DataConfig(variable_set=payload.get("variable_set", "core"))
            mnorm = Normalizer(load_stats(mcfg.stats_path))
            direct_h = payload.get("direct_lead_h")
            ds = Era5Dataset(
                mcfg, (2020, 2020), mnorm, rollout_steps=1,
                two_frame=payload.get("two_frame", True),
                direct_steps=(direct_h // 6) if direct_h else None,
                n_frames=payload.get("n_frames"),
            )
            model = build_model(payload["model_name"],
                                in_channels=ds.n_input_channels,
                                out_channels=len(mcfg.target_channels),
                                **payload.get("model_params", {}))
            model.load_state_dict(payload["state_dict"])
            cmap = scored_channel_map(mcfg.target_channels)
            if direct_h:
                # a one-shot model produces exactly one lead; the rest stay NaN
                # and the viewer simply has fewer frames for it
                forecaster = DirectForecaster(model, ds, src_id, channel_map=cmap)
            else:
                forecaster = ModelForecaster(model, ds, mnorm, src_id,
                                             channel_map=cmap)
        elif kind == "competitor":
            try:
                forecaster = CompetitorForecaster(cfg, spec, 2020, display_name=src_id)
            except FileNotFoundError:
                print(f"skip {src_id}: forecasts not cached")
                continue
        elif kind == "persistence":
            forecaster = PersistenceForecaster(truth)

        wrote_inits, wrote_leads = [], set()
        for init_s, t0 in init_idx.items():
            fields = None if kind == "truth" else forecaster.forecast(t0, K)
            wrote_here = False
            for lead in args.leads:
                if lead == 0:
                    u, v = truth[t0, U10], truth[t0, V10]   # analysis at init
                elif kind == "truth":
                    u, v = truth[t0 + lead // 6, U10], truth[t0 + lead // 6, V10]
                else:
                    f = fields[lead // 6 - 1]
                    if not np.isfinite(f).all():
                        continue        # e.g. 12-hourly models at odd leads
                    u, v = f[U10], f[V10]
                payload = velocity_records(u, v, lat, lon, f"{init_s}:00:00Z", lead)
                (out_dir / f"{src_id}_{init_s}_{lead:03d}.json").write_text(
                    json.dumps(payload, separators=(",", ":")))
                wrote_leads.add(lead)
                wrote_here = True
            if wrote_here:
                wrote_inits.append(init_s)
        if wrote_inits:
            # per-source lists: the viewer filters its pickers by these, so a
            # live 2026 run and a 2020 research forecast can coexist
            entry = {
                "id": src_id, "label": label, "kind": kind,
                "inits": wrote_inits, "leads": sorted(wrote_leads),
            }
            # the CSV name matches the source id for our models; competitors
            # carry the _2020 suffix their result files use
            rmse = scored_rmse(src_id) or scored_rmse(f"{src_id}_2020")
            if rmse is not None:
                entry["rmse72"] = round(rmse, 3)
            manifest["sources"].append(entry)
            print(f"exported {src_id}")

    # keep live sources added by scripts/fetch_dynamical.py: they are fetched
    # separately (and expensively), so a re-export must not drop them
    man_path = out_dir / "manifest.json"
    if man_path.exists():
        old = json.loads(man_path.read_text())
        live = [s for s in old.get("sources", []) if s.get("kind") == "live"
                and all((out_dir / f"{s['id']}_{i}_{ld:03d}.json").exists()
                        for i in s.get("inits", []) for ld in s.get("leads", []))]
        if live:
            manifest["sources"] += live
            manifest["inits"] += [i for s in live for i in s.get("inits", [])
                                  if i not in manifest["inits"]]
            print(f"kept {len(live)} live source(s)")
    man_path.write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in out_dir.glob("*.json"))
    print(f"\n{len(manifest['sources'])} sources -> {out_dir} ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
