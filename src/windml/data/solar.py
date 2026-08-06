"""Top-of-atmosphere incident solar radiation, computed analytically.

Rasp & Thuerey (2021) feed `toa_incident_solar_radiation` to the network as one
of their 38 per-timestep fields. The WeatherBench 2 64x32 ERA5 zarr ships every
other field they use but not that one -- so we compute it instead of
downloading it. That is not a compromise: TISR is a pure function of latitude,
longitude and time (it is the solar geometry, with no atmosphere involved), so
the analytic value is exact and costs no storage or bandwidth. GraphCast and
several other models synthesize it the same way.

We return instantaneous irradiance in W/m^2 where ERA5 archives an accumulation
in J/m^2 over the preceding hour. The two differ by a constant factor for a
given accumulation window, and every input channel is standardized before it
reaches the network, so the distinction washes out. What matters -- the diurnal
cycle, the seasonal cycle, and the day/night terminator -- is identical.

Formulas are Spencer (1971) Fourier fits for declination, the equation of time,
and the Earth-Sun distance correction; they are accurate to ~0.01 degrees, far
finer than a 5.625 degree grid cell.
"""
from __future__ import annotations

import numpy as np

# Solar constant (W/m^2), the WMO/IPCC value used by ERA5.
SOLAR_CONSTANT = 1361.0


def _spencer_terms(day_of_year: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Declination (rad), equation of time (minutes), and distance factor.

    `day_of_year` is 1-based and may be fractional.
    """
    # Fractional year angle. Spencer's series is defined on a 365-day year;
    # using 365.25 here would shift the phase, so keep 365.
    g = 2.0 * np.pi * (day_of_year - 1.0) / 365.0

    declination = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.001480 * np.sin(3 * g)
    )
    # Equation of time: the offset between mean solar time and true solar time,
    # driven by orbital eccentricity and axial tilt. 229.18 converts radians to
    # minutes of hour angle.
    eot_minutes = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    # (mean distance / actual distance)^2 -- the ~3.4% annual swing in irradiance.
    distance_factor = (
        1.000110
        + 0.034221 * np.cos(g)
        + 0.001280 * np.sin(g)
        + 0.000719 * np.cos(2 * g)
        + 0.000077 * np.sin(2 * g)
    )
    return declination, eot_minutes, distance_factor


def toa_incident_solar_radiation(
    times: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray
) -> np.ndarray:
    """Instantaneous TOA irradiance in W/m^2, shape (n_time, n_lat, n_lon).

    `times` must be numpy datetime64. Night-side cells are exactly 0.
    """
    times = np.asarray(times, dtype="datetime64[s]")

    # Day of year (1-based, fractional) without any calendar library: the year
    # boundary comes from casting to datetime64[Y], which truncates.
    year_start = times.astype("datetime64[Y]").astype("datetime64[s]")
    seconds_into_year = (times - year_start).astype(np.float64)
    day_of_year = seconds_into_year / 86400.0 + 1.0

    # UTC hour of day.
    day_start = times.astype("datetime64[D]").astype("datetime64[s]")
    utc_hour = (times - day_start).astype(np.float64) / 3600.0

    declination, eot_minutes, distance_factor = _spencer_terms(day_of_year)

    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))[None, :, None]
    lon = np.asarray(lon_deg, dtype=np.float64)[None, None, :]

    # True solar time, in hours, at each longitude. Longitude contributes
    # 1 hour per 15 degrees east.
    solar_time = (
        utc_hour[:, None, None]
        + lon / 15.0
        + eot_minutes[:, None, None] / 60.0
    )
    # Hour angle: 0 at local solar noon, 15 degrees per hour.
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))

    decl = declination[:, None, None]
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(
        hour_angle
    )
    # The night side gets no sun; clipping (rather than taking |cos|) is what
    # creates the terminator the network needs to see.
    cos_zenith = np.clip(cos_zenith, 0.0, None)

    return SOLAR_CONSTANT * distance_factor[:, None, None] * cos_zenith
