"""Grid geometry helpers: latitude weights and static field encodings."""
from __future__ import annotations

import numpy as np


def latitude_weights(lat_deg: np.ndarray) -> np.ndarray:
    """WeatherBench latitude weights: cos(lat) normalized to mean 1.

    Grid cells shrink toward the poles; weighting by cos(lat) makes area-fair
    global averages. Returns shape (n_lat,).
    """
    w = np.cos(np.deg2rad(lat_deg))
    w = np.clip(w, 0.0, None)
    return w / w.mean()


def spatial_encodings(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Per-pixel positional channels: sin(lat), cos(lat)*sin(lon), cos(lat)*cos(lon).

    These three embed the sphere in R^3, so models see continuous position
    without a dateline discontinuity. Returns (3, n_lat, n_lon).
    """
    lat = np.deg2rad(lat_deg)[:, None]
    lon = np.deg2rad(lon_deg)[None, :]
    ones = np.ones_like(lon)
    return np.stack(
        [
            np.sin(lat) * ones,
            np.cos(lat) * np.sin(lon),
            np.cos(lat) * np.cos(lon),
        ]
    ).astype(np.float32)


def time_encodings(hours_since_epoch: np.ndarray) -> np.ndarray:
    """Scalar time-of-day and day-of-year sin/cos features, shape (n_time, 4)."""
    hour = hours_since_epoch % 24
    doy = (hours_since_epoch / 24.0) % 365.25
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ],
        axis=-1,
    ).astype(np.float32)
