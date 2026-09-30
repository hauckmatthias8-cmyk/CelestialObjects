"""
celestial_objects.py

Standalone Sun/Moon alignment search without Skyfield/Astropy.

Dependencies:
    pip install numpy scipy matplotlib

Optional for IANA timezone output on Windows:
    pip install tzdata

Purpose
-------
Given
    - an observer position: latitude [deg], longitude [deg], altitude above
      mean sea level [m] + observer vertical offset [m]
    - an object position: latitude [deg], longitude [deg], altitude above
      mean sea level [m] + object vertical offset [m]

find dates/times when the apparent centre of the Sun or Moon is at a requested
angular position relative to the object as seen from the observer.  The solver
uses direct/event-based candidate generation rather than a regular time grid:

                  +Y [deg]
                    above
                      ^
                      |
        -X [deg]  <-- OBJECT -->  +X [deg]
           left                      right
                      |
                      v
                    below
                  -Y [deg]

Important accuracy note
-----------------------
The astronomical model in this file is analytical and self-contained. It is
suitable for scenario generation and broad alignment searches, but it is NOT a
JPL ephemeris. In particular, the Moon model is an approximation. The final
optimizer can refine TIME to fractions of a second, but that numerical time
resolution must not be confused with absolute astronomical accuracy.

The default main() searches from datetime.now(UTC) to exactly 300 CALENDAR
YEARS later. Calendar-year arithmetic is used so leap years are included.
Sun and Moon are both searched. Positions below the local horizon are allowed.
Atmospheric refraction is disabled by default.
"""

from __future__ import annotations

import calendar
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from scipy.optimize import minimize_scalar


# =============================================================================
# Constants
# =============================================================================

# WGS84 semi-major axis / equatorial Earth radius [m]
WGS84_A_M = 6_378_137.0

# WGS84 flattening [-]
WGS84_F = 1.0 / 298.257223563

# WGS84 first eccentricity squared [-]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Earth radius used by the analytical lunar orbital model [km]
EARTH_RADIUS_KM = 6_378.14

# Astronomical Unit [km]
AU_KM = 149_597_870.7

# Seconds per mean solar day [s/day]
SECONDS_PER_DAY = 86_400.0


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class AlignmentGeometry:
    """Observer -> object viewing geometry."""

    # Unit vector from observer toward object in local ENU coordinates [-]
    forward_unit: np.ndarray

    # Unit vector pointing visually right in the object-centred camera plane [-]
    right_unit: np.ndarray

    # Unit vector pointing visually up in the object-centred camera plane [-]
    camera_up_unit: np.ndarray

    # Straight-line observer -> object distance [m]
    object_distance_m: float

    # Object azimuth as seen from observer [deg]
    object_azimuth_deg: float

    # Object altitude/elevation as seen from observer [deg]
    object_altitude_deg: float


# =============================================================================
# Time helpers
# =============================================================================


def get_timezone(name: str = "Europe/Berlin"):
    """
    Return an IANA timezone if available.

    Parameters
    ----------
    name : str
        IANA timezone key [-], e.g. "Europe/Berlin".

    Returns
    -------
    tzinfo
        Timezone object [-].

    Notes
    -----
    Windows Python installations may not include the IANA database. Installing
    `tzdata` fixes that. If it is unavailable, this function falls back to UTC
    rather than silently inventing future daylight-saving rules.
    """

    if name.upper() == "UTC":
        return timezone.utc

    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        print(
            f"WARNING: Timezone database does not contain '{name}'.\n"
            "         Falling back to UTC for displayed local_time.\n"
            "         Install optional package 'tzdata' for IANA timezone output."
        )
        return timezone.utc


def add_calendar_years(dt: datetime, years: int) -> datetime:
    """
    Add whole CALENDAR years to a datetime.

    Parameters
    ----------
    dt : datetime
        Input datetime [-].

    years : int
        Number of calendar years to add [years].

    Returns
    -------
    datetime
        Resulting datetime [-].

    Leap-year handling
    ------------------
    2028-02-29 + 1 year  -> 2029-02-28
    2028-02-29 + 4 years -> 2032-02-29
    """

    target_year = dt.year + years  # [calendar year]

    # Number of days in target month [days]
    target_month_days = calendar.monthrange(target_year, dt.month)[1]

    # Calendar day of month [day]
    target_day = min(dt.day, target_month_days)

    return dt.replace(year=target_year, day=target_day)


def _julian_day_utc(dt: datetime) -> float:
    """
    Convert timezone-aware datetime to Julian Day using UTC as UT1 approximation.

    Parameters
    ----------
    dt : datetime
        Time [-], must be timezone-aware.

    Returns
    -------
    float
        Julian Day [days].
    """

    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")

    dt_utc = dt.astimezone(timezone.utc)

    # Time since Unix epoch [s].  Using datetime arithmetic instead of
    # datetime.timestamp() avoids platform-specific range limits for dates
    # centuries in the future.
    unix_epoch_utc = datetime(1970, 1, 1, tzinfo=timezone.utc)
    unix_seconds = (dt_utc - unix_epoch_utc).total_seconds()  # [s]

    # Julian Day [days]
    return unix_seconds / SECONDS_PER_DAY + 2_440_587.5


def _decimal_year(dt: datetime) -> float:
    """
    Approximate decimal Gregorian year.

    Parameters
    ----------
    dt : datetime
        Time [-].

    Returns
    -------
    float
        Decimal year [years].
    """

    dt_utc = dt.astimezone(timezone.utc)
    year_start = datetime(dt_utc.year, 1, 1, tzinfo=timezone.utc)
    next_year = datetime(dt_utc.year + 1, 1, 1, tzinfo=timezone.utc)

    elapsed_seconds = (dt_utc - year_start).total_seconds()  # [s]
    year_seconds = (next_year - year_start).total_seconds()  # [s]

    return dt_utc.year + elapsed_seconds / year_seconds


def _delta_t_seconds(dt: datetime) -> float:
    """
    Approximate Delta-T = TT - UT1.

    Parameters
    ----------
    dt : datetime
        Observation time [-].

    Returns
    -------
    float
        Delta-T [s].

    Notes
    -----
    Future Earth rotation cannot be known exactly. For dates beyond modern
    observations this is necessarily a prediction/extrapolation. The search
    therefore returns an astronomical estimate, not a guarantee of future UTC
    clock time to sub-second accuracy.
    """

    y = _decimal_year(dt)  # [years]

    # Piecewise polynomial approximations commonly used for Delta-T.
    if 2005.0 <= y < 2050.0:
        t = y - 2000.0  # [years]
        return 62.92 + 0.32217 * t + 0.005589 * t * t  # [s]

    if 2050.0 <= y < 2150.0:
        u = (y - 1820.0) / 100.0  # [centuries]
        return -20.0 + 32.0 * u * u - 0.5628 * (2150.0 - y)  # [s]

    # Long-term extrapolation. This is intentionally approximate.
    u = (y - 1820.0) / 100.0  # [centuries]
    return -20.0 + 32.0 * u * u  # [s]


def _gmst_deg(jd_ut_days: float) -> float:
    """
    Greenwich Mean Sidereal Time.

    Parameters
    ----------
    jd_ut_days : float
        Julian Day on UT/UT1-like time scale [days].

    Returns
    -------
    float
        Greenwich Mean Sidereal Time [deg].
    """

    # Julian centuries since J2000.0 [centuries]
    t_centuries = (jd_ut_days - 2_451_545.0) / 36_525.0

    # GMST [deg]
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut_days - 2_451_545.0)
        + 0.000387933 * t_centuries * t_centuries
        - t_centuries * t_centuries * t_centuries / 38_710_000.0
    )

    return gmst_deg % 360.0


# =============================================================================
# Basic numeric helpers
# =============================================================================


def _norm_deg(angle_deg: float) -> float:
    """Normalize angle [deg] into [0 deg, 360 deg)."""
    return angle_deg % 360.0


def _unit(vector: np.ndarray) -> np.ndarray:
    """
    Normalize vector.

    Parameters
    ----------
    vector : np.ndarray
        Vector [any consistent unit].

    Returns
    -------
    np.ndarray
        Unit vector [-].
    """

    magnitude = float(np.linalg.norm(vector))  # [same unit as vector]
    if magnitude == 0.0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / magnitude


# =============================================================================
# WGS84 terrestrial geometry
# =============================================================================


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height_ellipsoid_m: float) -> np.ndarray:
    """
    Convert WGS84 geodetic coordinates to ECEF.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude [deg].

    lon_deg : float
        Geodetic longitude [deg].

    height_ellipsoid_m : float
        Height above WGS84 ellipsoid [m].

    Returns
    -------
    np.ndarray
        ECEF [X, Y, Z] [m].
    """

    lat_rad = math.radians(lat_deg)  # [rad]
    lon_rad = math.radians(lon_deg)  # [rad]

    sin_lat = math.sin(lat_rad)  # [-]
    cos_lat = math.cos(lat_rad)  # [-]

    # Prime vertical radius of curvature [m]
    n_m = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    # ECEF coordinates [m]
    x_m = (n_m + height_ellipsoid_m) * cos_lat * math.cos(lon_rad)
    y_m = (n_m + height_ellipsoid_m) * cos_lat * math.sin(lon_rad)
    z_m = (n_m * (1.0 - WGS84_E2) + height_ellipsoid_m) * sin_lat

    return np.array([x_m, y_m, z_m], dtype=float)


def _ecef_vector_to_enu(
    vector_ecef: np.ndarray,
    observer_lat_deg: float,
    observer_lon_deg: float,
) -> np.ndarray:
    """
    Convert an ECEF vector into local East/North/Up components.

    Parameters
    ----------
    vector_ecef : np.ndarray
        ECEF vector [any consistent length unit].

    observer_lat_deg : float
        Observer latitude [deg].

    observer_lon_deg : float
        Observer longitude [deg].

    Returns
    -------
    np.ndarray
        [East, North, Up] [same length unit as input].
    """

    lat_rad = math.radians(observer_lat_deg)  # [rad]
    lon_rad = math.radians(observer_lon_deg)  # [rad]

    x, y, z = vector_ecef  # [input length unit]

    east = -math.sin(lon_rad) * x + math.cos(lon_rad) * y

    north = (
        -math.sin(lat_rad) * math.cos(lon_rad) * x
        - math.sin(lat_rad) * math.sin(lon_rad) * y
        + math.cos(lat_rad) * z
    )

    up = (
        math.cos(lat_rad) * math.cos(lon_rad) * x
        + math.cos(lat_rad) * math.sin(lon_rad) * y
        + math.sin(lat_rad) * z
    )

    return np.array([east, north, up], dtype=float)


def _build_alignment_geometry(
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_height_ellipsoid_m: float,
    object_lat_deg: float,
    object_lon_deg: float,
    object_height_ellipsoid_m: float,
) -> AlignmentGeometry:
    """Build object-centred viewing axes and diagnostic object geometry."""

    # Observer ECEF position [m]
    observer_ecef_m = _geodetic_to_ecef(
        observer_lat_deg,
        observer_lon_deg,
        observer_height_ellipsoid_m,
    )

    # Object ECEF position [m]
    object_ecef_m = _geodetic_to_ecef(
        object_lat_deg,
        object_lon_deg,
        object_height_ellipsoid_m,
    )

    # Observer -> object ECEF vector [m]
    object_delta_ecef_m = object_ecef_m - observer_ecef_m

    # Straight-line distance [m]
    object_distance_m = float(np.linalg.norm(object_delta_ecef_m))
    if object_distance_m < 0.001:
        raise ValueError("Observer and object are effectively at the same position")

    # Observer -> object local ENU vector [m]
    object_enu_m = _ecef_vector_to_enu(
        object_delta_ecef_m,
        observer_lat_deg,
        observer_lon_deg,
    )

    # Forward/look direction [-]
    forward_unit = _unit(object_enu_m)

    # Local ENU up direction [-]
    local_up_unit = np.array([0.0, 0.0, 1.0], dtype=float)

    # Visual right direction [-]
    right_unit = np.cross(forward_unit, local_up_unit)
    if float(np.linalg.norm(right_unit)) < 1e-12:
        raise ValueError(
            "Object is almost exactly above/below observer; left/right is undefined"
        )
    right_unit = _unit(right_unit)

    # Visual up direction in object-centred image plane [-]
    camera_up_unit = _unit(np.cross(right_unit, forward_unit))

    # Object altitude/elevation [deg]
    object_altitude_deg = math.degrees(
        math.asin(float(np.clip(forward_unit[2], -1.0, 1.0)))
    )

    # Object azimuth [deg], 0=N, 90=E
    object_azimuth_deg = (
        math.degrees(math.atan2(float(forward_unit[0]), float(forward_unit[1])))
        + 360.0
    ) % 360.0

    return AlignmentGeometry(
        forward_unit=forward_unit,
        right_unit=right_unit,
        camera_up_unit=camera_up_unit,
        object_distance_m=object_distance_m,
        object_azimuth_deg=object_azimuth_deg,
        object_altitude_deg=object_altitude_deg,
    )


# =============================================================================
# Orbital helpers
# =============================================================================


def _solve_kepler(mean_anomaly_rad: float, eccentricity: float) -> float:
    """
    Solve E - e*sin(E) = M.

    Parameters
    ----------
    mean_anomaly_rad : float
        Mean anomaly M [rad].

    eccentricity : float
        Orbital eccentricity [-].

    Returns
    -------
    float
        Eccentric anomaly E [rad].
    """

    eccentric_anomaly_rad = mean_anomaly_rad  # [rad]

    for _ in range(12):
        correction_rad = (
            eccentric_anomaly_rad
            - eccentricity * math.sin(eccentric_anomaly_rad)
            - mean_anomaly_rad
        ) / (1.0 - eccentricity * math.cos(eccentric_anomaly_rad))

        eccentric_anomaly_rad -= correction_rad

        if abs(correction_rad) < 1e-14:
            break

    return eccentric_anomaly_rad


# =============================================================================
# Analytical Sun model
# =============================================================================


def _sun_ecliptic_xyz(jd_tt_days: float) -> np.ndarray:
    """
    Approximate geocentric apparent Sun vector in ecliptic-of-date coordinates.

    Parameters
    ----------
    jd_tt_days : float
        Julian Day on TT-like time scale [days].

    Returns
    -------
    np.ndarray
        Geocentric ecliptic XYZ [km].
    """

    # Julian centuries since J2000.0 [centuries]
    t = (jd_tt_days - 2_451_545.0) / 36_525.0

    # Geometric mean longitude of Sun [deg]
    mean_longitude_deg = _norm_deg(
        280.46646 + 36_000.76983 * t + 0.0003032 * t * t
    )

    # Mean anomaly [deg]
    mean_anomaly_deg = _norm_deg(
        357.52911 + 35_999.05029 * t - 0.0001537 * t * t + t**3 / 24_490_000.0
    )

    # Orbital eccentricity [-]
    eccentricity = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t

    mean_anomaly_rad = math.radians(mean_anomaly_deg)  # [rad]

    # Equation of centre [deg]
    equation_center_deg = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mean_anomaly_rad)
        + (0.019993 - 0.000101 * t) * math.sin(2.0 * mean_anomaly_rad)
        + 0.000289 * math.sin(3.0 * mean_anomaly_rad)
    )

    # True longitude [deg]
    true_longitude_deg = mean_longitude_deg + equation_center_deg

    # True anomaly [deg]
    true_anomaly_deg = mean_anomaly_deg + equation_center_deg
    true_anomaly_rad = math.radians(true_anomaly_deg)  # [rad]

    # Earth-Sun distance [AU]
    distance_au = (
        1.000001018 * (1.0 - eccentricity * eccentricity)
        / (1.0 + eccentricity * math.cos(true_anomaly_rad))
    )

    # Apparent longitude correction [deg]
    omega_deg = 125.04 - 1934.136 * t
    apparent_longitude_deg = (
        true_longitude_deg
        - 0.00569
        - 0.00478 * math.sin(math.radians(omega_deg))
    )

    apparent_longitude_rad = math.radians(apparent_longitude_deg)  # [rad]
    distance_km = distance_au * AU_KM  # [km]

    return np.array(
        [
            distance_km * math.cos(apparent_longitude_rad),
            distance_km * math.sin(apparent_longitude_rad),
            0.0,
        ],
        dtype=float,
    )


# =============================================================================
# Analytical Moon model
# =============================================================================


def _moon_ecliptic_xyz(jd_tt_days: float) -> np.ndarray:
    """
    Approximate geocentric Moon vector in ecliptic-of-date coordinates.

    Parameters
    ----------
    jd_tt_days : float
        Julian Day on TT-like time scale [days].

    Returns
    -------
    np.ndarray
        Geocentric ecliptic XYZ [km].

    Notes
    -----
    This is a compact analytical model with the major lunar perturbations. It
    is intentionally dependency-free, but it is not a JPL lunar ephemeris.
    """

    # Days since 2000 Jan 0.0 [days]
    d_days = jd_tt_days - 2_451_543.5

    # Longitude of ascending node [deg]
    node_deg = _norm_deg(125.1228 - 0.0529538083 * d_days)

    # Orbital inclination [deg]
    inclination_deg = 5.1454

    # Argument of periapsis [deg]
    periapsis_deg = _norm_deg(318.0634 + 0.1643573223 * d_days)

    # Semi-major axis [Earth radii]
    semi_major_axis_er = 60.2666

    # Orbital eccentricity [-]
    eccentricity = 0.054900

    # Mean anomaly [deg]
    mean_anomaly_deg = _norm_deg(115.3654 + 13.0649929509 * d_days)
    mean_anomaly_rad = math.radians(mean_anomaly_deg)  # [rad]

    # Eccentric anomaly [rad]
    eccentric_anomaly_rad = _solve_kepler(mean_anomaly_rad, eccentricity)

    # Orbital-plane coordinates [Earth radii]
    x_orbit_er = semi_major_axis_er * (
        math.cos(eccentric_anomaly_rad) - eccentricity
    )
    y_orbit_er = (
        semi_major_axis_er
        * math.sqrt(1.0 - eccentricity * eccentricity)
        * math.sin(eccentric_anomaly_rad)
    )

    # True anomaly [rad]
    true_anomaly_rad = math.atan2(y_orbit_er, x_orbit_er)

    # Earth-Moon distance [Earth radii]
    distance_er = math.hypot(x_orbit_er, y_orbit_er)

    node_rad = math.radians(node_deg)  # [rad]
    periapsis_rad = math.radians(periapsis_deg)  # [rad]
    inclination_rad = math.radians(inclination_deg)  # [rad]
    orbital_argument_rad = true_anomaly_rad + periapsis_rad  # [rad]

    # Unperturbed ecliptic XYZ [Earth radii]
    x_er = distance_er * (
        math.cos(node_rad) * math.cos(orbital_argument_rad)
        - math.sin(node_rad)
        * math.sin(orbital_argument_rad)
        * math.cos(inclination_rad)
    )
    y_er = distance_er * (
        math.sin(node_rad) * math.cos(orbital_argument_rad)
        + math.cos(node_rad)
        * math.sin(orbital_argument_rad)
        * math.cos(inclination_rad)
    )
    z_er = (
        distance_er
        * math.sin(orbital_argument_rad)
        * math.sin(inclination_rad)
    )

    # Ecliptic longitude [deg]
    longitude_deg = math.degrees(math.atan2(y_er, x_er))

    # Ecliptic latitude [deg]
    latitude_deg = math.degrees(
        math.atan2(z_er, math.hypot(x_er, y_er))
    )

    # Sun longitude of perihelion [deg]
    sun_perihelion_deg = 282.9404 + 4.70935e-5 * d_days

    # Sun mean anomaly [deg]
    sun_mean_anomaly_deg = _norm_deg(356.0470 + 0.9856002585 * d_days)

    # Sun mean longitude [deg]
    sun_mean_longitude_deg = _norm_deg(
        sun_mean_anomaly_deg + sun_perihelion_deg
    )

    # Moon mean longitude [deg]
    moon_mean_longitude_deg = _norm_deg(
        mean_anomaly_deg + periapsis_deg + node_deg
    )

    # Mean elongation of Moon [deg]
    elongation_deg = _norm_deg(
        moon_mean_longitude_deg - sun_mean_longitude_deg
    )

    # Moon argument of latitude [deg]
    argument_latitude_deg = _norm_deg(moon_mean_longitude_deg - node_deg)

    def sind(angle_deg: float) -> float:
        """sin(angle), input [deg], output [-]."""
        return math.sin(math.radians(angle_deg))

    def cosd(angle_deg: float) -> float:
        """cos(angle), input [deg], output [-]."""
        return math.cos(math.radians(angle_deg))

    # Longitude perturbations [deg]
    longitude_deg += (
        -1.274 * sind(mean_anomaly_deg - 2.0 * elongation_deg)
        + 0.658 * sind(2.0 * elongation_deg)
        - 0.186 * sind(sun_mean_anomaly_deg)
        - 0.059 * sind(2.0 * mean_anomaly_deg - 2.0 * elongation_deg)
        - 0.057
        * sind(mean_anomaly_deg - 2.0 * elongation_deg + sun_mean_anomaly_deg)
        + 0.053 * sind(mean_anomaly_deg + 2.0 * elongation_deg)
        + 0.046 * sind(2.0 * elongation_deg - sun_mean_anomaly_deg)
        + 0.041 * sind(mean_anomaly_deg - sun_mean_anomaly_deg)
        - 0.035 * sind(elongation_deg)
        - 0.031 * sind(mean_anomaly_deg + sun_mean_anomaly_deg)
        - 0.015 * sind(2.0 * argument_latitude_deg - 2.0 * elongation_deg)
        + 0.011 * sind(mean_anomaly_deg - 4.0 * elongation_deg)
    )

    # Latitude perturbations [deg]
    latitude_deg += (
        -0.173 * sind(argument_latitude_deg - 2.0 * elongation_deg)
        - 0.055
        * sind(mean_anomaly_deg - argument_latitude_deg - 2.0 * elongation_deg)
        - 0.046
        * sind(mean_anomaly_deg + argument_latitude_deg - 2.0 * elongation_deg)
        + 0.033 * sind(argument_latitude_deg + 2.0 * elongation_deg)
        + 0.017 * sind(2.0 * mean_anomaly_deg + argument_latitude_deg)
    )

    # Distance perturbations [Earth radii]
    distance_er += (
        -0.58 * cosd(mean_anomaly_deg - 2.0 * elongation_deg)
        - 0.46 * cosd(2.0 * elongation_deg)
    )

    longitude_rad = math.radians(longitude_deg)  # [rad]
    latitude_rad = math.radians(latitude_deg)  # [rad]

    # Earth-Moon distance [km]
    distance_km = distance_er * EARTH_RADIUS_KM

    cos_lat = math.cos(latitude_rad)  # [-]

    return np.array(
        [
            distance_km * cos_lat * math.cos(longitude_rad),
            distance_km * cos_lat * math.sin(longitude_rad),
            distance_km * math.sin(latitude_rad),
        ],
        dtype=float,
    )


# =============================================================================
# Ecliptic -> equatorial
# =============================================================================


def _mean_obliquity_deg(jd_tt_days: float) -> float:
    """
    Mean obliquity of the ecliptic.

    Parameters
    ----------
    jd_tt_days : float
        Julian Day on TT-like time scale [days].

    Returns
    -------
    float
        Mean obliquity [deg].
    """

    t = (jd_tt_days - 2_451_545.0) / 36_525.0  # [centuries]

    # Arcseconds correction from J2000 [arcsec]
    seconds = (
        21.448
        - 46.8150 * t
        - 0.00059 * t * t
        + 0.001813 * t * t * t
    )

    # Obliquity [deg]
    return 23.0 + 26.0 / 60.0 + seconds / 3600.0


def _ecliptic_to_equatorial(vector_ecliptic_km: np.ndarray, jd_tt_days: float) -> np.ndarray:
    """
    Convert ecliptic XYZ to equatorial XYZ.

    Parameters
    ----------
    vector_ecliptic_km : np.ndarray
        Ecliptic XYZ [km].

    jd_tt_days : float
        Julian Day on TT-like time scale [days].

    Returns
    -------
    np.ndarray
        Equatorial XYZ [km].
    """

    obliquity_rad = math.radians(_mean_obliquity_deg(jd_tt_days))  # [rad]

    x_km, y_km, z_km = vector_ecliptic_km  # [km]

    return np.array(
        [
            x_km,
            y_km * math.cos(obliquity_rad) - z_km * math.sin(obliquity_rad),
            y_km * math.sin(obliquity_rad) + z_km * math.cos(obliquity_rad),
        ],
        dtype=float,
    )


# =============================================================================
# Optional atmospheric refraction
# =============================================================================


def _apply_refraction(
    enu_unit_vector: np.ndarray,
    pressure_mbar: float = 1010.0,
    temperature_c: float = 10.0,
) -> np.ndarray:
    """
    Apply approximate Bennett-style atmospheric refraction.

    Parameters
    ----------
    enu_unit_vector : np.ndarray
        Local ENU direction [-].

    pressure_mbar : float
        Atmospheric pressure [mbar = hPa].

    temperature_c : float
        Air temperature [deg C].

    Returns
    -------
    np.ndarray
        Refracted local ENU unit direction [-].

    Notes
    -----
    For searches where below-horizon geometry matters, keep refraction=False.
    """

    enu_unit_vector = _unit(enu_unit_vector)
    east, north, up = enu_unit_vector  # [-]

    # Geometric altitude [deg]
    altitude_deg = math.degrees(math.asin(float(np.clip(up, -1.0, 1.0))))

    # Azimuth [rad]
    azimuth_rad = math.atan2(float(east), float(north))

    # Do not apply the horizon approximation far below the horizon.
    if altitude_deg <= -1.0:
        return enu_unit_vector

    # Refraction correction [arcmin]
    refraction_arcmin = 1.02 / math.tan(
        math.radians(altitude_deg + 10.3 / (altitude_deg + 5.11))
    )

    # Pressure/temperature scaling [-]
    refraction_arcmin *= (pressure_mbar / 1010.0) * (
        283.0 / (273.0 + temperature_c)
    )

    # Refracted altitude [deg]
    refracted_altitude_deg = altitude_deg + refraction_arcmin / 60.0
    refracted_altitude_rad = math.radians(refracted_altitude_deg)  # [rad]

    return np.array(
        [
            math.cos(refracted_altitude_rad) * math.sin(azimuth_rad),
            math.cos(refracted_altitude_rad) * math.cos(azimuth_rad),
            math.sin(refracted_altitude_rad),
        ],
        dtype=float,
    )


# =============================================================================
# Topocentric Sun/Moon direction
# =============================================================================


def _body_enu(
    dt: datetime,
    body: str,
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_height_ellipsoid_m: float,
    refraction: bool = False,
    pressure_mbar: float = 1010.0,
    temperature_c: float = 10.0,
) -> np.ndarray:
    """
    Calculate topocentric Sun/Moon direction in local ENU coordinates.

    Parameters
    ----------
    dt : datetime
        Observation time [-], timezone-aware.

    body : str
        "sun" or "moon" [-].

    observer_lat_deg : float
        Observer latitude [deg].

    observer_lon_deg : float
        Observer longitude [deg].

    observer_height_ellipsoid_m : float
        Observer WGS84 ellipsoid height [m].

    refraction : bool
        Apply approximate atmospheric refraction [-].

    pressure_mbar : float
        Atmospheric pressure [mbar = hPa].

    temperature_c : float
        Air temperature [deg C].

    Returns
    -------
    np.ndarray
        Local topocentric [East, North, Up] unit vector [-].
    """

    # UT-like Julian Day for Earth rotation [days]
    jd_ut_days = _julian_day_utc(dt)

    # Delta-T = TT - UT1 [s]
    delta_t_s = _delta_t_seconds(dt)

    # TT-like Julian Day for orbital model [days]
    jd_tt_days = jd_ut_days + delta_t_s / SECONDS_PER_DAY

    body_name = body.lower().strip()

    if body_name == "sun":
        body_ecliptic_km = _sun_ecliptic_xyz(jd_tt_days)  # [km]
    elif body_name == "moon":
        body_ecliptic_km = _moon_ecliptic_xyz(jd_tt_days)  # [km]
    else:
        raise ValueError("body must be 'sun' or 'moon'")

    # Geocentric equatorial celestial-body position [km]
    body_equatorial_km = _ecliptic_to_equatorial(
        body_ecliptic_km,
        jd_tt_days,
    )

    # Observer ECEF position [m]
    observer_ecef_m = _geodetic_to_ecef(
        observer_lat_deg,
        observer_lon_deg,
        observer_height_ellipsoid_m,
    )

    # Observer ECEF position [km]
    observer_ecef_km = observer_ecef_m / 1000.0

    # Earth rotation angle via GMST [deg]
    gmst_deg = _gmst_deg(jd_ut_days)
    gmst_rad = math.radians(gmst_deg)  # [rad]

    cos_gmst = math.cos(gmst_rad)  # [-]
    sin_gmst = math.sin(gmst_rad)  # [-]

    ox_km, oy_km, oz_km = observer_ecef_km  # [km]

    # Observer position in equatorial inertial frame [km]
    observer_equatorial_km = np.array(
        [
            cos_gmst * ox_km - sin_gmst * oy_km,
            sin_gmst * ox_km + cos_gmst * oy_km,
            oz_km,
        ],
        dtype=float,
    )

    # Observer -> body topocentric vector [km].
    # This explicitly includes parallax; it is especially important for Moon.
    topocentric_equatorial_km = body_equatorial_km - observer_equatorial_km

    tx_km, ty_km, tz_km = topocentric_equatorial_km  # [km]

    # Equatorial inertial -> Earth-fixed vector [km]
    topocentric_ecef_km = np.array(
        [
            cos_gmst * tx_km + sin_gmst * ty_km,
            -sin_gmst * tx_km + cos_gmst * ty_km,
            tz_km,
        ],
        dtype=float,
    )

    # Local ENU topocentric vector [km]
    topocentric_enu_km = _ecef_vector_to_enu(
        topocentric_ecef_km,
        observer_lat_deg,
        observer_lon_deg,
    )

    enu_unit = _unit(topocentric_enu_km)  # [-]

    if refraction:
        enu_unit = _apply_refraction(
            enu_unit,
            pressure_mbar=pressure_mbar,
            temperature_c=temperature_c,
        )

    return _unit(enu_unit)


# =============================================================================
# Alignment evaluator
# =============================================================================


def _make_alignment_evaluator(
    *,
    body: str,
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_height_ellipsoid_m: float,
    geometry: AlignmentGeometry,
    refraction: bool,
    pressure_mbar: float,
    temperature_c: float,
) -> Callable[[datetime], tuple[float, float, float, float]]:
    """Create a fast per-time evaluator closure."""

    def evaluate(dt_utc: datetime) -> tuple[float, float, float, float]:
        """
        Evaluate body relative to object.

        Returns
        -------
        x_deg : float
            Horizontal offset from object [deg], +right / -left.

        y_deg : float
            Vertical offset from object [deg], +above / -below.

        body_altitude_deg : float
            Body altitude relative to local horizon [deg].

        body_azimuth_deg : float
            Body azimuth [deg], 0=N, 90=E.
        """

        body_unit = _body_enu(
            dt=dt_utc,
            body=body,
            observer_lat_deg=observer_lat_deg,
            observer_lon_deg=observer_lon_deg,
            observer_height_ellipsoid_m=observer_height_ellipsoid_m,
            refraction=refraction,
            pressure_mbar=pressure_mbar,
            temperature_c=temperature_c,
        )

        # Object-centred camera projections [-]
        forward_component = float(np.dot(body_unit, geometry.forward_unit))
        right_component = float(np.dot(body_unit, geometry.right_unit))
        up_component = float(np.dot(body_unit, geometry.camera_up_unit))

        # Apparent horizontal image-plane offset [deg]
        x_deg = math.degrees(math.atan2(right_component, forward_component))

        # Apparent vertical image-plane offset [deg]
        y_deg = math.degrees(math.atan2(up_component, forward_component))

        # Body altitude [deg]
        body_altitude_deg = math.degrees(
            math.asin(float(np.clip(body_unit[2], -1.0, 1.0)))
        )

        # Body azimuth [deg], 0=N, 90=E
        body_azimuth_deg = (
            math.degrees(math.atan2(float(body_unit[0]), float(body_unit[1])))
            + 360.0
        ) % 360.0

        return x_deg, y_deg, body_altitude_deg, body_azimuth_deg

    return evaluate


# =============================================================================
# Hierarchical search helpers
# =============================================================================


def _sample_seconds(duration_seconds: float, step_seconds: float) -> np.ndarray:
    """
    Create inclusive sample coordinates.

    Parameters
    ----------
    duration_seconds : float
        Interval duration [s].

    step_seconds : float
        Sampling interval [s].

    Returns
    -------
    np.ndarray
        Sample offsets from interval start [s].
    """

    if duration_seconds < 0.0:
        raise ValueError("duration_seconds must be >= 0")
    if step_seconds <= 0.0:
        raise ValueError("step_seconds must be > 0")

    if duration_seconds == 0.0:
        return np.array([0.0], dtype=float)

    samples = np.arange(0.0, duration_seconds, step_seconds, dtype=float)

    if len(samples) == 0 or samples[-1] != duration_seconds:
        samples = np.append(samples, duration_seconds)

    return samples


def _local_minimum_indices(values: np.ndarray) -> list[int]:
    """
    Return indices of local minima, including eligible boundaries.

    Parameters
    ----------
    values : np.ndarray
        Objective values [-].

    Returns
    -------
    list[int]
        Array indices [-].
    """

    count = len(values)
    if count == 0:
        return []
    if count == 1:
        return [0]

    result: list[int] = []

    if values[0] <= values[1]:
        result.append(0)

    for index in range(1, count - 1):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            result.append(index)

    if values[-1] <= values[-2]:
        result.append(count - 1)

    return result


def _deduplicate_results(results: list[dict], duplicate_window_seconds: float = 2.0) -> list[dict]:
    """Deduplicate refined events that represent the same minimum."""

    if not results:
        return []

    sorted_results = sorted(results, key=lambda item: item["utc"])
    unique: list[dict] = []

    for item in sorted_results:
        if not unique:
            unique.append(item)
            continue

        delta_seconds = abs((item["utc"] - unique[-1]["utc"]).total_seconds())  # [s]

        if delta_seconds > duplicate_window_seconds:
            unique.append(item)
        elif item["error_2d_deg"] < unique[-1]["error_2d_deg"]:
            unique[-1] = item

    return unique


# =============================================================================
# Direct / event-based alignment solver
# =============================================================================


def _signed_angle_difference_deg(target_deg: float, value_deg: float) -> float:
    """Return target - value wrapped to [-180, +180) [deg]."""

    return (target_deg - value_deg + 180.0) % 360.0 - 180.0


def _target_enu_from_image_offsets(
    geometry: AlignmentGeometry,
    target_x_deg: float,
    target_y_deg: float,
) -> np.ndarray:
    """
    Convert object-centred image-plane offsets to one local ENU direction.

    Parameters
    ----------
    geometry : AlignmentGeometry
        Observer -> object camera frame [-].

    target_x_deg : float
        Desired horizontal offset [deg], +right / -left.

    target_y_deg : float
        Desired vertical offset [deg], +above / -below.

    Returns
    -------
    np.ndarray
        Local [East, North, Up] target unit vector [-].

    Notes
    -----
    The public evaluator defines

        x = atan2(right_component, forward_component)
        y = atan2(up_component,    forward_component)

    Therefore the inverse direction is exactly proportional to

        forward + tan(x) * right + tan(y) * up

    for target directions in the forward hemisphere.
    """

    if abs(target_x_deg) >= 89.0 or abs(target_y_deg) >= 89.0:
        raise ValueError(
            "Direct candidate solver requires abs(target_x_deg) and "
            "abs(target_y_deg) < 89 deg"
        )

    x_rad = math.radians(target_x_deg)  # [rad]
    y_rad = math.radians(target_y_deg)  # [rad]

    target_enu = (
        geometry.forward_unit
        + math.tan(x_rad) * geometry.right_unit
        + math.tan(y_rad) * geometry.camera_up_unit
    )

    return _unit(target_enu)


def _enu_to_declination_hour_angle(
    enu_unit: np.ndarray,
    observer_lat_deg: float,
) -> tuple[float, float, float, float]:
    """
    Convert one fixed local sky direction to equatorial geometry.

    Parameters
    ----------
    enu_unit : np.ndarray
        Local [East, North, Up] unit vector [-].

    observer_lat_deg : float
        Observer geodetic latitude [deg].

    Returns
    -------
    declination_deg : float
        Required celestial declination [deg].

    hour_angle_deg : float
        Required local hour angle [deg], positive westward.

    altitude_deg : float
        Target local altitude [deg].

    azimuth_deg : float
        Target local azimuth [deg], 0=N, 90=E.
    """

    east, north, up = map(float, _unit(enu_unit))  # [-]

    altitude_rad = math.asin(float(np.clip(up, -1.0, 1.0)))  # [rad]
    azimuth_rad = math.atan2(east, north)  # [rad], north through east
    latitude_rad = math.radians(observer_lat_deg)  # [rad]

    sin_alt = math.sin(altitude_rad)  # [-]
    cos_alt = math.cos(altitude_rad)  # [-]
    sin_lat = math.sin(latitude_rad)  # [-]
    cos_lat = math.cos(latitude_rad)  # [-]

    # Horizontal -> equatorial conversion [-].
    sin_declination = (
        sin_alt * sin_lat
        + cos_alt * cos_lat * math.cos(azimuth_rad)
    )
    declination_rad = math.asin(
        float(np.clip(sin_declination, -1.0, 1.0))
    )  # [rad]

    cos_declination = max(1.0e-15, math.cos(declination_rad))  # [-]

    # Hour angle H: positive westward [rad].
    sin_hour_angle = (
        -cos_alt * math.sin(azimuth_rad) / cos_declination
    )
    cos_hour_angle = (
        sin_alt * cos_lat
        - cos_alt * sin_lat * math.cos(azimuth_rad)
    ) / cos_declination

    hour_angle_rad = math.atan2(
        sin_hour_angle,
        cos_hour_angle,
    )  # [rad]

    declination_deg = math.degrees(declination_rad)  # [deg]
    hour_angle_deg = math.degrees(hour_angle_rad)  # [deg]
    altitude_deg = math.degrees(altitude_rad)  # [deg]
    azimuth_deg = (math.degrees(azimuth_rad) + 360.0) % 360.0  # [deg]

    return declination_deg, hour_angle_deg, altitude_deg, azimuth_deg


def _body_geocentric_ra_dec_distance(
    dt_utc: datetime,
    body: str,
) -> tuple[float, float, float]:
    """
    Return analytical geocentric right ascension, declination and distance.

    Returns
    -------
    ra_deg : float
        Right ascension [deg], 0..360.

    declination_deg : float
        Geocentric declination [deg].

    distance_km : float
        Geocentric body distance [km].
    """

    jd_ut_days = _julian_day_utc(dt_utc)  # [days]
    delta_t_s = _delta_t_seconds(dt_utc)  # [s]
    jd_tt_days = jd_ut_days + delta_t_s / SECONDS_PER_DAY  # [days]

    body_name = body.lower().strip()
    if body_name == "sun":
        ecliptic_km = _sun_ecliptic_xyz(jd_tt_days)  # [km]
    elif body_name == "moon":
        ecliptic_km = _moon_ecliptic_xyz(jd_tt_days)  # [km]
    else:
        raise ValueError("body must be 'sun' or 'moon'")

    equatorial_km = _ecliptic_to_equatorial(
        ecliptic_km,
        jd_tt_days,
    )  # [km]

    x_km, y_km, z_km = map(float, equatorial_km)  # [km]
    distance_km = math.sqrt(x_km * x_km + y_km * y_km + z_km * z_km)  # [km]

    ra_deg = (
        math.degrees(math.atan2(y_km, x_km)) + 360.0
    ) % 360.0  # [deg]

    declination_deg = math.degrees(
        math.asin(float(np.clip(z_km / distance_km, -1.0, 1.0)))
    )  # [deg]

    return ra_deg, declination_deg, distance_km


def _body_geocentric_hour_angle_deg(
    dt_utc: datetime,
    body: str,
    observer_lon_deg: float,
) -> float:
    """Return geocentric local hour angle [deg], positive westward."""

    jd_ut_days = _julian_day_utc(dt_utc)  # [days]
    ra_deg, _, _ = _body_geocentric_ra_dec_distance(dt_utc, body)  # [deg]

    local_sidereal_deg = (
        _gmst_deg(jd_ut_days) + observer_lon_deg
    ) % 360.0  # [deg]

    return (
        local_sidereal_deg - ra_deg + 180.0
    ) % 360.0 - 180.0  # [deg]


def _sun_ecliptic_longitude_deg(dt_utc: datetime) -> float:
    """Return analytical apparent geocentric solar ecliptic longitude [deg]."""

    jd_ut_days = _julian_day_utc(dt_utc)  # [days]
    delta_t_s = _delta_t_seconds(dt_utc)  # [s]
    jd_tt_days = jd_ut_days + delta_t_s / SECONDS_PER_DAY  # [days]

    sun_ecliptic_km = _sun_ecliptic_xyz(jd_tt_days)  # [km]
    x_km = float(sun_ecliptic_km[0])  # [km]
    y_km = float(sun_ecliptic_km[1])  # [km]

    return (
        math.degrees(math.atan2(y_km, x_km)) + 360.0
    ) % 360.0  # [deg]


def _predict_hour_angle_passage(
    event_dt_utc: datetime,
    *,
    body: str,
    observer_lon_deg: float,
    target_hour_angle_deg: float,
) -> datetime:
    """
    Predict the nearby time when body hour angle equals target hour angle.

    A local numerical derivative is used, so the different daily rates of the
    Sun and Moon are automatically included.
    """

    derivative_step_s = 600.0  # [s]

    h0_deg = _body_geocentric_hour_angle_deg(
        event_dt_utc,
        body,
        observer_lon_deg,
    )  # [deg]

    h1_deg = _body_geocentric_hour_angle_deg(
        event_dt_utc + timedelta(seconds=derivative_step_s),
        body,
        observer_lon_deg,
    )  # [deg]

    hour_angle_rate_deg_per_s = (
        _signed_angle_difference_deg(h1_deg, h0_deg) / derivative_step_s
    )  # [deg/s]

    # Fallback rates: about 15 deg/h for Sun, 14.5 deg/h for Moon [deg/s].
    if abs(hour_angle_rate_deg_per_s) < 1.0e-6:
        nominal_deg_per_hour = 15.0 if body == "sun" else 14.5  # [deg/h]
        hour_angle_rate_deg_per_s = nominal_deg_per_hour / 3600.0  # [deg/s]

    required_change_deg = _signed_angle_difference_deg(
        target_hour_angle_deg,
        h0_deg,
    )  # [deg]

    correction_s = required_change_deg / hour_angle_rate_deg_per_s  # [s]

    # The nearest passage must be within about half one apparent day.  Clamp
    # pathological analytical cases to a safe +/- 14 h interval [s].
    correction_s = float(np.clip(correction_s, -14.0 * 3600.0, 14.0 * 3600.0))

    return event_dt_utc + timedelta(seconds=correction_s)


def _refine_full_alignment_near_passage(
    passage_guess_utc: datetime,
    *,
    objective_at_datetime: Callable[[datetime], float],
    make_result: Callable[[datetime], dict],
    start_utc: datetime,
    end_utc: datetime,
    refinement_seconds: float,
    search_half_window_hours: float,
    sample_step_minutes: float = 20.0,
) -> dict | None:
    """
    Refine one predicted daily passage using the full topocentric objective.

    The coarse local samples protect scipy's bounded minimizer from being given
    a window containing more than one unrelated local minimum.
    """

    half_window_s = search_half_window_hours * 3600.0  # [s]
    sample_step_s = sample_step_minutes * 60.0  # [s]

    left_utc = max(
        start_utc,
        passage_guess_utc - timedelta(seconds=half_window_s),
    )
    right_utc = min(
        end_utc,
        passage_guess_utc + timedelta(seconds=half_window_s),
    )

    if right_utc <= left_utc:
        return None

    duration_s = (right_utc - left_utc).total_seconds()  # [s]
    sample_seconds = _sample_seconds(duration_s, sample_step_s)  # [s]

    sample_scores = np.empty(len(sample_seconds), dtype=float)  # [-]
    for index, offset_s in enumerate(sample_seconds):
        sample_scores[index] = objective_at_datetime(
            left_utc + timedelta(seconds=float(offset_s))
        )

    best_index = int(np.argmin(sample_scores))  # [-]
    left_index = max(0, best_index - 1)  # [-]
    right_index = min(len(sample_seconds) - 1, best_index + 1)  # [-]

    refine_left_s = float(sample_seconds[left_index])  # [s]
    refine_right_s = float(sample_seconds[right_index])  # [s]

    if refine_right_s <= refine_left_s:
        best_dt_utc = left_utc + timedelta(seconds=refine_left_s)
        return make_result(best_dt_utc)

    def objective_seconds(seconds_from_left: float) -> float:
        return objective_at_datetime(
            left_utc + timedelta(seconds=float(seconds_from_left))
        )

    optimization = minimize_scalar(
        objective_seconds,
        bounds=(refine_left_s, refine_right_s),
        method="bounded",
        options={
            "xatol": refinement_seconds,  # [s]
            "maxiter": 80,  # [-]
        },
    )

    best_dt_utc = left_utc + timedelta(seconds=float(optimization.x))
    return make_result(best_dt_utc)


def _sun_declination_event_seeds(
    *,
    start_utc: datetime,
    end_utc: datetime,
    target_declination_deg: float,
    progress_every_years: int = 25,
) -> list[datetime]:
    """
    Generate a few analytical solar declination-event seeds per calendar year.

    Instead of stepping through every hour, the required solar declination is
    converted to the corresponding ecliptic longitude(s).  Solar solstices are
    also seeded so a closest approach is still found if the requested
    declination lies outside the Sun's reachable declination range.
    """

    seeds: list[datetime] = []

    first_year = start_utc.year  # [calendar year]
    final_year = end_utc.year  # [calendar year]
    next_progress_year = first_year + max(1, progress_every_years)

    print(
        f"[SUN] candidate generation: calendar years "
        f"{first_year}..{final_year}"
    )

    for year in range(first_year, final_year + 1):
        year_start_utc = datetime(year, 1, 1, tzinfo=timezone.utc)
        if year == 9999:
            year_end_utc = datetime.max.replace(tzinfo=timezone.utc)
        else:
            year_end_utc = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

        interval_start_utc = max(start_utc, year_start_utc)
        interval_end_utc = min(end_utc, year_end_utc)
        if interval_end_utc <= interval_start_utc:
            continue

        midyear_utc = year_start_utc + (year_end_utc - year_start_utc) / 2
        jd_mid_ut_days = _julian_day_utc(midyear_utc)  # [days]
        jd_mid_tt_days = (
            jd_mid_ut_days
            + _delta_t_seconds(midyear_utc) / SECONDS_PER_DAY
        )  # [days]
        obliquity_deg = _mean_obliquity_deg(jd_mid_tt_days)  # [deg]

        target_longitudes_deg: list[float] = []
        denominator = math.sin(math.radians(obliquity_deg))  # [-]
        if abs(denominator) > 1.0e-12:
            ratio = (
                math.sin(math.radians(target_declination_deg)) / denominator
            )  # [-]

            if abs(ratio) <= 1.0:
                lambda_a_deg = math.degrees(math.asin(ratio)) % 360.0  # [deg]
                lambda_b_deg = (180.0 - lambda_a_deg) % 360.0  # [deg]
                target_longitudes_deg.extend([lambda_a_deg, lambda_b_deg])

        # Solstice longitudes [deg] guarantee a nearest-declination seed even
        # if the requested declination is outside the solar range.
        target_longitudes_deg.extend([90.0, 270.0])

        # Deduplicate nearly equal longitudes [deg].
        unique_longitudes_deg: list[float] = []
        for longitude_deg in target_longitudes_deg:
            if not any(
                abs(_signed_angle_difference_deg(longitude_deg, existing_deg))
                < 0.01
                for existing_deg in unique_longitudes_deg
            ):
                unique_longitudes_deg.append(longitude_deg)

        # Solar angular speed around the ecliptic [deg/day].
        mean_solar_rate_deg_per_day = 360.0 / 365.2422

        jan1_longitude_deg = _sun_ecliptic_longitude_deg(year_start_utc)  # [deg]

        for target_longitude_deg in unique_longitudes_deg:
            forward_angle_deg = (
                target_longitude_deg - jan1_longitude_deg
            ) % 360.0  # [deg]

            initial_days = forward_angle_deg / mean_solar_rate_deg_per_day  # [days]
            guess_utc = year_start_utc + timedelta(days=initial_days)

            # Refine only a small local interval around the analytical guess.
            local_left_utc = max(
                year_start_utc,
                guess_utc - timedelta(days=3.0),
            )
            local_right_utc = min(
                year_end_utc,
                guess_utc + timedelta(days=3.0),
            )

            local_duration_s = (
                local_right_utc - local_left_utc
            ).total_seconds()  # [s]

            if local_duration_s <= 0.0:
                continue

            def longitude_error_squared(seconds_from_left: float) -> float:
                dt_utc = local_left_utc + timedelta(
                    seconds=float(seconds_from_left)
                )
                difference_deg = _signed_angle_difference_deg(
                    target_longitude_deg,
                    _sun_ecliptic_longitude_deg(dt_utc),
                )
                return difference_deg * difference_deg  # [deg^2]

            optimization = minimize_scalar(
                longitude_error_squared,
                bounds=(0.0, local_duration_s),
                method="bounded",
                options={"xatol": 0.5, "maxiter": 60},  # [s], [-]
            )

            event_utc = local_left_utc + timedelta(
                seconds=float(optimization.x)
            )

            if interval_start_utc <= event_utc <= interval_end_utc:
                seeds.append(event_utc)

        if progress_every_years > 0 and year >= next_progress_year:
            print(
                f"[SUN] candidate scan reached year {year} "
                f"| seeds so far: {len(seeds)}"
            )
            next_progress_year += progress_every_years

    print(
        f"[SUN] candidate scan reached year {final_year} "
        f"| raw seeds: {len(seeds)}"
    )

    seeds.sort()

    # Longitudes can collapse to the same event near a solstice [s].
    unique_seeds: list[datetime] = []
    for seed_utc in seeds:
        if not unique_seeds or abs(
            (seed_utc - unique_seeds[-1]).total_seconds()
        ) > 6.0 * 3600.0:
            unique_seeds.append(seed_utc)

    return unique_seeds


def _moon_declination_event_seeds(
    *,
    start_utc: datetime,
    end_utc: datetime,
    target_declination_deg: float,
    step_hours: float,
    progress_every_years: int,
) -> list[datetime]:
    """
    Generate lunar declination-crossing/extremum seeds.

    Only the Moon's slowly changing GEOCENTRIC declination is sampled here,
    normally once per 24 h.  This is not the old sky-position brute-force
    search: daily Earth rotation is solved analytically later from hour angle.

    A 300-year search therefore needs only about 110,000 cheap declination
    evaluations instead of millions of full topocentric alignment evaluations.
    """

    if step_hours <= 0.0:
        raise ValueError("moon_declination_step_hours must be > 0")

    step = timedelta(hours=step_hours)
    seeds: list[datetime] = []

    # Keep three samples so local minima of |declination-target| are also
    # detected when the requested declination lies outside a monthly range.
    t_prevprev: datetime | None = None
    d_prevprev: float | None = None
    t_prev = start_utc
    _, dec_prev_deg, _ = _body_geocentric_ra_dec_distance(t_prev, "moon")
    d_prev = dec_prev_deg - target_declination_deg  # [deg]

    next_progress_year = start_utc.year + max(1, progress_every_years)

    t_current = min(start_utc + step, end_utc)
    while t_current <= end_utc:
        _, dec_current_deg, _ = _body_geocentric_ra_dec_distance(
            t_current,
            "moon",
        )
        d_current = dec_current_deg - target_declination_deg  # [deg]

        # Sign change -> target declination crossing. Linear interpolation is
        # already plenty accurate for seeding the later hour-angle solver.
        if d_prev == 0.0 or d_prev * d_current < 0.0:
            denominator = abs(d_prev) + abs(d_current)  # [deg]
            fraction = 0.0 if denominator == 0.0 else abs(d_prev) / denominator  # [-]
            crossing_utc = t_prev + (t_current - t_prev) * fraction
            seeds.append(crossing_utc)

        # Local minimum of absolute declination error -> monthly extremum seed.
        if (
            t_prevprev is not None
            and d_prevprev is not None
            and abs(d_prev) <= abs(d_prevprev)
            and abs(d_prev) <= abs(d_current)
        ):
            seeds.append(t_prev)

        if progress_every_years > 0 and t_current.year >= next_progress_year:
            print(
                f"[MOON] candidate scan reached year {t_current.year} "
                f"| seeds so far: {len(seeds)}"
            )
            next_progress_year += progress_every_years

        t_prevprev = t_prev
        d_prevprev = d_prev
        t_prev = t_current
        d_prev = d_current

        if t_current >= end_utc:
            break

        t_current = min(t_current + step, end_utc)

    seeds.sort()

    # Crossing and extremum detectors can report the same event [s].
    unique_seeds: list[datetime] = []
    for seed_utc in seeds:
        if not unique_seeds or abs(
            (seed_utc - unique_seeds[-1]).total_seconds()
        ) > 6.0 * 3600.0:
            unique_seeds.append(seed_utc)

    return unique_seeds


def find_sun_moon_alignment(
    *,
    # Observer latitude [deg]
    lat: float,
    # Observer longitude [deg]
    lon: float,
    # Observer altitude above mean sea level [m]
    alt: float,
    # Observer/camera vertical offset above alt [m]
    offset: float,
    # Object latitude [deg]
    latO: float,
    # Object longitude [deg]
    lonO: float,
    # Object altitude above mean sea level [m]
    altO: float,
    # Object target-point vertical offset above altO [m]
    offsetO: float,
    # Search start [-], timezone-aware datetime
    start: datetime,
    # Search end [-], timezone-aware datetime
    end: datetime,
    # Celestial body: "sun" or "moon" [-]
    body: str = "sun",
    # Desired horizontal angular offset [deg], +right / -left
    target_x_deg: float = 0.0,
    # Desired vertical angular offset [deg], +above / -below
    target_y_deg: float = 0.0,
    # Allowed horizontal error / maximum deviation [deg]
    tolerance_x_deg: float = 0.05,
    # Allowed vertical error / maximum deviation [deg]
    tolerance_y_deg: float = 0.05,
    # Final numerical time-refinement tolerance [s]
    refinement_seconds: float = 0.05,
    # Moon-only slow declination candidate interval [h]
    moon_declination_step_hours: float = 24.0,
    # Full topocentric local-refinement half-window [h]
    local_refinement_half_window_hours: float = 4.0,
    # Minimum allowed body altitude [deg]; -90 allows below horizon
    min_body_altitude_deg: float = -90.0,
    # Apply atmospheric refraction [-]
    refraction: bool = False,
    # Atmospheric pressure [mbar = hPa]
    pressure_mbar: float = 1010.0,
    # Atmospheric temperature [deg C]
    temperature_c: float = 10.0,
    # Observer geoid undulation N, ellipsoid height = MSL + N [m]
    observer_geoid_undulation_m: float = 0.0,
    # Object geoid undulation N, ellipsoid height = MSL + N [m]
    object_geoid_undulation_m: float = 0.0,
    # Timezone used only to display local_time [-]
    output_timezone: str = "UTC",
    # If there are no tolerance matches, return closest refined event [-]
    return_closest_if_none: bool = True,
    # Maximum returned matches; None means unlimited [-]
    max_results: int | None = None,
    # Progress output interval for long Moon candidate scans [years]
    progress_every_years: int = 25,
) -> list[dict]:
    """
    Find Sun/Moon alignments using direct/event-based candidate generation.

    This replaces the former multi-hour full-sky grid search.

    SUN
    ---
    The requested object-relative direction is converted to one fixed local
    altitude/azimuth, hence to a fixed declination and hour angle.  The solar
    declination condition gives one or two ecliptic longitudes per year.  Only
    those analytical annual events (plus solstices for closest-approach safety)
    are evaluated.

    MOON
    ----
    The same fixed declination/hour-angle target is used.  Because lunar motion
    contains several strong perturbation cycles, a single closed-form inverse
    timestamp is not reliable.  The code therefore detects only slow lunar
    declination events, normally once per 24 h, then calculates the daily hour
    angle passage directly and performs a small topocentric refinement.

    The returned event is the minimum error near each candidate passage.  A
    result is marked within_tolerance when BOTH horizontal and vertical limits
    are satisfied.
    """

    body_name = body.lower().strip()
    if body_name not in ("sun", "moon"):
        raise ValueError("body must be 'sun' or 'moon'")

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware datetimes")

    if tolerance_x_deg <= 0.0 or tolerance_y_deg <= 0.0:
        raise ValueError("tolerance_x_deg and tolerance_y_deg must be > 0")

    if refinement_seconds <= 0.0:
        raise ValueError("refinement_seconds must be > 0")

    if local_refinement_half_window_hours <= 0.0:
        raise ValueError("local_refinement_half_window_hours must be > 0")

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")

    output_tz = get_timezone(output_timezone)

    # Observer height above WGS84 ellipsoid [m]
    observer_height_ellipsoid_m = (
        alt + offset + observer_geoid_undulation_m
    )

    # Object target-point height above WGS84 ellipsoid [m]
    object_height_ellipsoid_m = (
        altO + offsetO + object_geoid_undulation_m
    )

    geometry = _build_alignment_geometry(
        observer_lat_deg=lat,
        observer_lon_deg=lon,
        observer_height_ellipsoid_m=observer_height_ellipsoid_m,
        object_lat_deg=latO,
        object_lon_deg=lonO,
        object_height_ellipsoid_m=object_height_ellipsoid_m,
    )

    # Exact local direction represented by target_x_deg / target_y_deg [-].
    target_enu_unit = _target_enu_from_image_offsets(
        geometry,
        target_x_deg,
        target_y_deg,
    )

    (
        target_declination_deg,
        target_hour_angle_deg,
        target_altitude_deg,
        target_azimuth_deg,
    ) = _enu_to_declination_hour_angle(
        target_enu_unit,
        lat,
    )

    evaluate = _make_alignment_evaluator(
        body=body_name,
        observer_lat_deg=lat,
        observer_lon_deg=lon,
        observer_height_ellipsoid_m=observer_height_ellipsoid_m,
        geometry=geometry,
        refraction=refraction,
        pressure_mbar=pressure_mbar,
        temperature_c=temperature_c,
    )

    def objective_at_datetime(dt_utc: datetime) -> float:
        """Normalized squared full topocentric alignment error [-]."""

        x_deg, y_deg, body_altitude_deg, _ = evaluate(dt_utc)

        if body_altitude_deg < min_body_altitude_deg:
            altitude_error_deg = min_body_altitude_deg - body_altitude_deg  # [deg]
            return 1.0e12 + altitude_error_deg * altitude_error_deg

        dx_norm = (x_deg - target_x_deg) / tolerance_x_deg  # [-]
        dy_norm = (y_deg - target_y_deg) / tolerance_y_deg  # [-]
        return dx_norm * dx_norm + dy_norm * dy_norm

    def make_result(dt_utc: datetime) -> dict:
        """Build one unit-explicit result dictionary."""

        x_deg, y_deg, body_altitude_deg, body_azimuth_deg = evaluate(dt_utc)

        error_x_deg = x_deg - target_x_deg  # [deg]
        error_y_deg = y_deg - target_y_deg  # [deg]
        error_2d_deg = math.hypot(error_x_deg, error_y_deg)  # [deg]

        within_tolerance = (
            abs(error_x_deg) <= tolerance_x_deg
            and abs(error_y_deg) <= tolerance_y_deg
            and body_altitude_deg >= min_body_altitude_deg
        )

        return {
            "body": body_name,  # [-]
            "utc": dt_utc,  # [UTC datetime]
            "local_time": dt_utc.astimezone(output_tz),  # [timezone-aware datetime]
            "x_deg": x_deg,  # [deg]
            "y_deg": y_deg,  # [deg]
            "target_x_deg": target_x_deg,  # [deg]
            "target_y_deg": target_y_deg,  # [deg]
            "error_x_deg": error_x_deg,  # [deg]
            "error_y_deg": error_y_deg,  # [deg]
            "error_2d_deg": error_2d_deg,  # [deg]
            "within_tolerance": within_tolerance,  # [-]
            "body_altitude_deg": body_altitude_deg,  # [deg]
            "body_azimuth_deg": body_azimuth_deg,  # [deg]
            "object_altitude_deg": geometry.object_altitude_deg,  # [deg]
            "object_azimuth_deg": geometry.object_azimuth_deg,  # [deg]
            "object_distance_m": geometry.object_distance_m,  # [m]
            "target_sky_altitude_deg": target_altitude_deg,  # [deg]
            "target_sky_azimuth_deg": target_azimuth_deg,  # [deg]
            "target_declination_deg": target_declination_deg,  # [deg]
            "target_hour_angle_deg": target_hour_angle_deg,  # [deg]
        }

    # ------------------------------------------------------------------
    # Generate only astronomical event candidates, not a full time grid.
    # ------------------------------------------------------------------

    candidate_wall_start_s = time.perf_counter()  # [s]

    if body_name == "sun":
        event_seeds_utc = _sun_declination_event_seeds(
            start_utc=start_utc,
            end_utc=end_utc,
            target_declination_deg=target_declination_deg,
            progress_every_years=progress_every_years,
        )
    else:
        event_seeds_utc = _moon_declination_event_seeds(
            start_utc=start_utc,
            end_utc=end_utc,
            target_declination_deg=target_declination_deg,
            step_hours=moon_declination_step_hours,
            progress_every_years=progress_every_years,
        )

    candidate_wall_elapsed_s = time.perf_counter() - candidate_wall_start_s  # [s]

    print(
        f"[{body_name.upper()}] astronomical seeds: {len(event_seeds_utc)} "
        f"| generation wall time: {candidate_wall_elapsed_s:.2f} s"
    )

    results: list[dict] = []
    closest_result: dict | None = None

    # ------------------------------------------------------------------
    # Each declination event predicts one nearby daily hour-angle passage.
    # Only that small passage is evaluated topocentrically.
    # ------------------------------------------------------------------

    refinement_wall_start_s = time.perf_counter()  # [s]

    for seed_index, event_seed_utc in enumerate(event_seeds_utc, start=1):
        passage_guess_utc = _predict_hour_angle_passage(
            event_seed_utc,
            body=body_name,
            observer_lon_deg=lon,
            target_hour_angle_deg=target_hour_angle_deg,
        )

        candidate_result = _refine_full_alignment_near_passage(
            passage_guess_utc,
            objective_at_datetime=objective_at_datetime,
            make_result=make_result,
            start_utc=start_utc,
            end_utc=end_utc,
            refinement_seconds=refinement_seconds,
            search_half_window_hours=local_refinement_half_window_hours,
        )

        if candidate_result is None:
            continue

        if candidate_result["within_tolerance"]:
            results.append(candidate_result)

        if (
            closest_result is None
            or candidate_result["error_2d_deg"] < closest_result["error_2d_deg"]
        ):
            closest_result = candidate_result

        # Explicit progress for both bodies [-].
        # The Sun is normally much faster, so use a smaller reporting interval.
        progress_seed_interval = 250 if body_name == "sun" else 1000  # [seeds]
        if (
            progress_every_years > 0
            and seed_index % progress_seed_interval == 0
        ):
            elapsed_s = time.perf_counter() - refinement_wall_start_s  # [s]
            print(
                f"[{body_name.upper()}] refined "
                f"{seed_index}/{len(event_seeds_utc)} seeds "
                f"| matches: {len(results)} | wall time: {elapsed_s:.1f} s"
            )

    refinement_wall_elapsed_s = time.perf_counter() - refinement_wall_start_s  # [s]
    print(
        f"[{body_name.upper()}] refinement complete | "
        f"processed seeds: {len(event_seeds_utc)} | "
        f"raw matches: {len(results)} | "
        f"wall time: {refinement_wall_elapsed_s:.1f} s"
    )

    # Different seed types can converge onto the same physical passage.
    unique_results = _deduplicate_results(
        results,
        duplicate_window_seconds=120.0,  # [s]
    )

    unique_results.sort(key=lambda item: item["utc"])

    if unique_results:
        if max_results is None:
            return unique_results
        return unique_results[:max_results]

    if return_closest_if_none and closest_result is not None:
        return [closest_result]

    return []


# =============================================================================
# Output helpers
# =============================================================================


def print_results(body: str, results: list[dict]) -> None:
    """Print one body's results with units in all numeric column headers."""

    print()
    print("=" * 144)
    print(f" {body.upper()}")
    print("=" * 144)

    if not results:
        print(f"No {body} alignment found.")
        return

    print(
        f"{'UTC DATE / TIME':<33}"
        f"{'LOCAL DATE / TIME':<33}"
        f"{'X [deg]':>11}"
        f"{'Y [deg]':>11}"
        f"{'ERR X [deg]':>13}"
        f"{'ERR Y [deg]':>13}"
        f"{'ERR 2D [deg]':>14}"
        f"{'ALT [deg]':>11}"
        f"{'AZ [deg]':>11}"
        f"{'RESULT':>12}"
    )

    print("-" * 144)

    for result in results:
        status = "MATCH" if result["within_tolerance"] else "CLOSEST"

        print(
            f"{str(result['utc']):<33}"
            f"{str(result['local_time']):<33}"
            f"{result['x_deg']:>+11.5f}"
            f"{result['y_deg']:>+11.5f}"
            f"{result['error_x_deg']:>+13.5f}"
            f"{result['error_y_deg']:>+13.5f}"
            f"{result['error_2d_deg']:>14.5f}"
            f"{result['body_altitude_deg']:>+11.5f}"
            f"{result['body_azimuth_deg']:>11.5f}"
            f"{status:>12}"
        )


def print_best_summary(body: str, results: list[dict]) -> None:
    """
    Print the single best/closest returned result for one celestial body.

    Units
    -----
    v_position_target_degree : [deg]
        Vertical position/elevation of the terrestrial target object.

    v_offset_target_degree : [deg]
        Vertical Sun/Moon offset relative to the target object.
        Positive = above, negative = below.

    h_position_target_degree : [deg]
        Horizontal position/azimuth of the terrestrial target object.

    h_offset_target_degree : [deg]
        Horizontal Sun/Moon offset relative to the target object.
        Positive = right, negative = left.
    """

    if not results:
        print(f"Name {body.title()}, no result found")
        return

    # Smallest combined angular mismatch [deg].
    best_result = min(
        results,
        key=lambda result: result["error_2d_deg"],
    )

    print(
        f"Name {body.title()}, "
        f"Datetime: {best_result['local_time']}, "
        f"v_position_target_degree: {best_result['object_altitude_deg']:.6f}, "
        f"v_offset_target_degree: {best_result['y_deg']:.6f}, "
        f"h_position_target_degree: {best_result['object_azimuth_deg']:.6f}, "
        f"h_offset_target_degree: {best_result['x_deg']:.6f}"
    )




# =============================================================================
# 3-D visualization helpers
# =============================================================================


def _enu_unit_from_alt_az(altitude_deg: float, azimuth_deg: float) -> np.ndarray:
    """
    Convert local altitude/azimuth to an ENU unit vector.

    Parameters
    ----------
    altitude_deg : float
        Altitude above local horizon [deg].
        Negative values are below the horizon.

    azimuth_deg : float
        Azimuth [deg], measured clockwise from North:
        0 deg = North, 90 deg = East.

    Returns
    -------
    np.ndarray
        Local ENU unit vector [East, North, Up] [-].
    """

    altitude_rad = math.radians(altitude_deg)  # [rad]
    azimuth_rad = math.radians(azimuth_deg)  # [rad]

    cos_altitude = math.cos(altitude_rad)  # [-]

    return np.array(
        [
            cos_altitude * math.sin(azimuth_rad),  # East [-]
            cos_altitude * math.cos(azimuth_rad),  # North [-]
            math.sin(altitude_rad),  # Up [-]
        ],
        dtype=float,
    )


def _set_3d_axes_equal(ax, radius_display_m: float) -> None:
    """
    Apply a symmetric, equal-scale viewing volume to a Matplotlib 3-D axis.

    Parameters
    ----------
    ax
        Matplotlib 3-D axis [-].

    radius_display_m : float
        Half-width of displayed ENU scene [display m].

    Notes
    -----
    The celestial-body range is deliberately schematic, so the axis unit is
    labelled "display m" rather than implying the true Sun/Moon distance.
    """

    radius_display_m = max(float(radius_display_m), 1.0)  # [display m]
    ax.set_xlim(-radius_display_m, radius_display_m)
    ax.set_ylim(-radius_display_m, radius_display_m)
    ax.set_zlim(-radius_display_m, radius_display_m)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def show_best_alignment_3d(
    body: str,
    results: list[dict],
    *,
    celestial_display_distance_factor: float = 1.8,
    minimum_celestial_display_distance_m: float = 100.0,
) -> None:
    """
    Show an interactive simplified 3-D view of the best alignment result.

    Parameters
    ----------
    body : str
        Celestial object name, e.g. "sun" or "moon" [-].

    results : list[dict]
        Results returned by find_sun_moon_alignment() [-].

    celestial_display_distance_factor : float
        Artificial celestial display distance relative to the real
        observer->object distance [-].  The astronomical distance is NOT
        drawn to scale because the terrestrial scene would become invisible.

    minimum_celestial_display_distance_m : float
        Minimum artificial celestial display radius [display m].

    Interaction
    -----------
    The Matplotlib window is interactive:
        left mouse drag  -> rotate
        right mouse drag -> zoom (backend dependent)
        toolbar          -> pan / zoom / save

    Coordinate system
    -----------------
        X = East  [display m]
        Y = North [display m]
        Z = Up    [display m]

    Important
    ---------
    Observer->object distance is represented using the real straight-line
    distance [m]. Sun/Moon distance is intentionally schematic; only its
    angular direction is geometrically meaningful in this view.
    """

    if not results:
        print(f"[{body.upper()}] 3-D view skipped: no result available.")
        return

    if celestial_display_distance_factor <= 0.0:
        raise ValueError("celestial_display_distance_factor must be > 0")

    if minimum_celestial_display_distance_m <= 0.0:
        raise ValueError("minimum_celestial_display_distance_m must be > 0")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            f"[{body.upper()}] 3-D view unavailable: matplotlib is not installed.\n"
            "Install it with: python -m pip install matplotlib"
        )
        return

    best_result = min(
        results,
        key=lambda result: result["error_2d_deg"],
    )

    # Real observer->object straight-line distance [m].
    object_distance_m = float(best_result["object_distance_m"])

    # Object direction in local ENU [-].
    object_direction_unit = _enu_unit_from_alt_az(
        best_result["object_altitude_deg"],
        best_result["object_azimuth_deg"],
    )

    # Real local object position relative to observer [m].
    object_position_m = object_direction_unit * object_distance_m

    # Actual best-match Sun/Moon direction [-].
    body_direction_unit = _enu_unit_from_alt_az(
        best_result["body_altitude_deg"],
        best_result["body_azimuth_deg"],
    )

    # Requested target celestial direction [-].
    target_direction_unit = _enu_unit_from_alt_az(
        best_result["target_sky_altitude_deg"],
        best_result["target_sky_azimuth_deg"],
    )

    # Artificial celestial display distance [display m].
    celestial_display_distance_m = max(
        object_distance_m * celestial_display_distance_factor,
        minimum_celestial_display_distance_m,
    )

    # Schematic celestial positions [display m].
    body_display_position_m = (
        body_direction_unit * celestial_display_distance_m
    )
    target_display_position_m = (
        target_direction_unit * celestial_display_distance_m
    )

    # Scene half-width [display m].
    scene_radius_m = max(
        object_distance_m * 1.15,
        celestial_display_distance_m * 1.15,
        10.0,
    )

    figure = plt.figure(
        num=f"Best {body.title()} alignment - 3D scene",
        figsize=(10.5, 8.0),
    )
    ax = figure.add_subplot(111, projection="3d")

    # ------------------------------------------------------------------
    # Local horizon plane: Z = 0 [display m]
    # ------------------------------------------------------------------

    horizon_half_size_m = scene_radius_m * 0.75  # [display m]
    grid_coordinates_m = np.linspace(
        -horizon_half_size_m,
        horizon_half_size_m,
        9,
    )
    horizon_east_m, horizon_north_m = np.meshgrid(
        grid_coordinates_m,
        grid_coordinates_m,
    )
    horizon_up_m = np.zeros_like(horizon_east_m)  # [display m]

    ax.plot_surface(
        horizon_east_m,
        horizon_north_m,
        horizon_up_m,
        alpha=0.08,
        linewidth=0.25,
        edgecolor="0.65",
        color="0.8",
    )

    # Observer [display m].
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        s=90,
        marker="o",
        color="black",
        label="Observer",
        depthshade=False,
    )

    # Real terrestrial object position [m].
    ax.scatter(
        [object_position_m[0]],
        [object_position_m[1]],
        [object_position_m[2]],
        s=110,
        marker="s",
        label=f"Object ({object_distance_m:.1f} m)",
        depthshade=False,
    )

    # Real observer -> object line of sight [m].
    ax.plot(
        [0.0, object_position_m[0]],
        [0.0, object_position_m[1]],
        [0.0, object_position_m[2]],
        linewidth=2.0,
        label="Object line of sight",
    )

    # Requested target celestial ray [display m].
    ax.plot(
        [0.0, target_display_position_m[0]],
        [0.0, target_display_position_m[1]],
        [0.0, target_display_position_m[2]],
        linestyle="--",
        linewidth=1.8,
        label="Requested celestial direction",
    )
    ax.scatter(
        [target_display_position_m[0]],
        [target_display_position_m[1]],
        [target_display_position_m[2]],
        s=80,
        marker="x",
        label="Requested direction point",
        depthshade=False,
    )

    # Actual Sun/Moon ray for the best result [display m].
    ax.plot(
        [0.0, body_display_position_m[0]],
        [0.0, body_display_position_m[1]],
        [0.0, body_display_position_m[2]],
        linewidth=2.4,
        label=f"Actual {body.title()} direction",
    )
    ax.scatter(
        [body_display_position_m[0]],
        [body_display_position_m[1]],
        [body_display_position_m[2]],
        s=180 if body.lower() == "sun" else 140,
        marker="o",
        label=body.title(),
        depthshade=False,
    )

    # Small segment between requested and achieved direction at equal
    # schematic radius. This visually exposes the residual angular error.
    ax.plot(
        [target_display_position_m[0], body_display_position_m[0]],
        [target_display_position_m[1], body_display_position_m[1]],
        [target_display_position_m[2], body_display_position_m[2]],
        linestyle=":",
        linewidth=1.5,
        label=f"Residual error ({best_result['error_2d_deg']:.4f} deg)",
    )

    # ------------------------------------------------------------------
    # ENU axis arrows [display m]
    # ------------------------------------------------------------------

    axis_arrow_length_m = scene_radius_m * 0.32  # [display m]

    ax.quiver(
        0.0, 0.0, 0.0,
        axis_arrow_length_m, 0.0, 0.0,
        arrow_length_ratio=0.08,
        linewidth=1.5,
    )
    ax.text(
        axis_arrow_length_m, 0.0, 0.0,
        " East",
    )

    ax.quiver(
        0.0, 0.0, 0.0,
        0.0, axis_arrow_length_m, 0.0,
        arrow_length_ratio=0.08,
        linewidth=1.5,
    )
    ax.text(
        0.0, axis_arrow_length_m, 0.0,
        " North",
    )

    ax.quiver(
        0.0, 0.0, 0.0,
        0.0, 0.0, axis_arrow_length_m,
        arrow_length_ratio=0.08,
        linewidth=1.5,
    )
    ax.text(
        0.0, 0.0, axis_arrow_length_m,
        " Up",
    )

    # Labels next to important points.
    label_offset_m = scene_radius_m * 0.025  # [display m]
    ax.text(
        object_position_m[0] + label_offset_m,
        object_position_m[1] + label_offset_m,
        object_position_m[2] + label_offset_m,
        "Object",
    )
    ax.text(
        body_display_position_m[0] + label_offset_m,
        body_display_position_m[1] + label_offset_m,
        body_display_position_m[2] + label_offset_m,
        body.title(),
    )

    _set_3d_axes_equal(ax, scene_radius_m)

    ax.set_xlabel("East [display m]")
    ax.set_ylabel("North [display m]")
    ax.set_zlabel("Up [display m]")

    status = "MATCH" if best_result["within_tolerance"] else "CLOSEST"
    ax.set_title(
        f"Best {body.title()} alignment ({status})\n"
        f"{best_result['local_time']}\n"
        f"offset: X={best_result['x_deg']:+.4f} deg, "
        f"Y={best_result['y_deg']:+.4f} deg | "
        f"error={best_result['error_2d_deg']:.4f} deg"
    )

    ax.legend(loc="upper left", fontsize=8)

    # Helpful default camera angle. The user can rotate freely afterwards.
    ax.view_init(elev=22.0, azim=-55.0)

    figure.tight_layout()


def show_best_scenes_3d(
    all_results: dict[str, list[dict]],
    *,
    block: bool = True,
) -> None:
    """
    Open one interactive 3-D Matplotlib figure for Sun and one for Moon.

    Parameters
    ----------
    all_results : dict[str, list[dict]]
        Result dictionary containing "sun" and "moon" result lists [-].

    block : bool
        Passed to matplotlib.pyplot.show() [-].
        True keeps the script alive until all figure windows are closed.
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "3-D visualization skipped: matplotlib is not installed.\n"
            "Install it with: python -m pip install matplotlib"
        )
        return

    show_best_alignment_3d("sun", all_results.get("sun", []))
    show_best_alignment_3d("moon", all_results.get("moon", []))

    # Display both figure windows together.
    if plt.get_fignums():
        plt.show(block=block)


# =============================================================================
# Example main()
# =============================================================================


def main() -> None:
    """
    Example direct/event-based search for BOTH Sun and Moon.

    Default search window:
        start = datetime.now(UTC)
        end   = exactly 300 calendar years later

    Leap years are included automatically by add_calendar_years().
    Sun and Moon are both searched. Positions below the local horizon are
    allowed. Atmospheric refraction is disabled by default.
    """

    # -------------------------------------------------------------------------
    # Observer / ego configuration
    # -------------------------------------------------------------------------

    # Observer latitude [deg]
    observer_lat_deg = 50.8270

    # Observer longitude [deg]
    observer_lon_deg = 12.9210

    # Observer altitude above mean sea level [m]
    observer_altitude_msl_m = 300.0

    # Camera/eye height above observer altitude [m]
    observer_vertical_offset_m = 1.50

    # Local geoid undulation N at observer [m].
    # WGS84 ellipsoid height = MSL height + N.
    # Keep 0.0 if no geoid model/value is available.
    observer_geoid_undulation_m = 0.0

    # -------------------------------------------------------------------------
    # Object configuration
    # -------------------------------------------------------------------------

    # Object latitude [deg]
    object_lat_deg = 50.8275

    # Object longitude [deg]
    object_lon_deg = 12.9220

    # Object base altitude above mean sea level [m]
    object_altitude_msl_m = 302.0

    # Target point height above object base altitude [m]
    object_vertical_offset_m = 2.50

    # Local geoid undulation N at object [m].
    # WGS84 ellipsoid height = MSL height + N.
    object_geoid_undulation_m = 0.0

    # -------------------------------------------------------------------------
    # Search interval
    # -------------------------------------------------------------------------

    # Search start [UTC datetime]
    start_time = datetime.now(timezone.utc)

    # Search end [UTC datetime], exactly 300 CALENDAR years later.
    # Calendar arithmetic includes every leap year naturally.
    end_time = add_calendar_years(start_time, 300)

    # -------------------------------------------------------------------------
    # Requested body position relative to object
    # -------------------------------------------------------------------------

    # Desired horizontal offset [deg].
    # Negative = left of object, positive = right of object.
    target_x_deg = -2.0

    # Desired vertical offset [deg].
    # Negative = below object, positive = above object.
    target_y_deg = +1.0

    # Maximum allowed horizontal mismatch from target [deg]
    tolerance_x_deg = 0.05

    # Maximum allowed vertical mismatch from target [deg]
    tolerance_y_deg = 0.05

    # -------------------------------------------------------------------------
    # Direct/event-based solver settings
    # -------------------------------------------------------------------------

    # Final numerical optimizer tolerance in TIME [s].
    # This is numerical resolution, NOT guaranteed astronomical accuracy.
    refinement_seconds = 0.05

    # Moon only: sample the slowly changing geocentric lunar declination [h].
    # 24 h is normally sufficient because lunar declination varies on the
    # ~27-day orbital timescale. Daily Earth rotation is NOT sampled here;
    # its hour-angle crossing is calculated directly.
    moon_declination_step_hours = 24.0

    # Half-width around each predicted daily passage for final full
    # topocentric refinement [h].
    local_refinement_half_window_hours = 4.0

    # -------------------------------------------------------------------------
    # Horizon / atmosphere
    # -------------------------------------------------------------------------

    # Minimum allowed Sun/Moon altitude [deg].
    # -90 deg allows the entire celestial sphere, including below horizon.
    min_body_altitude_deg = -90.0

    # Apply approximate atmospheric refraction [-].
    # False is recommended because below-horizon geometry is allowed.
    refraction = False

    # Atmospheric pressure [mbar = hPa]
    pressure_mbar = 1010.0

    # Atmospheric temperature [deg C]
    temperature_c = 10.0

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    # Display timezone [-]. Search itself is performed using UTC.
    output_timezone = "Europe/Berlin"

    # Return closest event if no event is within both tolerances [-]
    return_closest_if_none = True

    # Maximum number of matching events returned per body [-].
    # Use None to return every match found.
    max_results = None

    # Progress interval for long Moon candidate generation [years]
    progress_every_years = 25

    # Open one interactive simplified 3-D scene for the BEST Sun result and
    # one for the BEST Moon result after the textual output [-].
    # Requires optional package: matplotlib
    show_3d_best_scenes = True

    print("Search configuration")
    print("--------------------")
    print(f"Start UTC                        : {start_time}")
    print(f"End UTC                          : {end_time}")
    print("Search duration                   : 300 calendar years")
    print(f"Target X                          : {target_x_deg:+.6f} deg")
    print(f"Target Y                          : {target_y_deg:+.6f} deg")
    print(f"Tolerance X                       : {tolerance_x_deg:.6f} deg")
    print(f"Tolerance Y                       : {tolerance_y_deg:.6f} deg")
    print(f"Moon declination seed interval    : {moon_declination_step_hours:.3f} h")
    print(f"Local refinement half-window      : {local_refinement_half_window_hours:.3f} h")
    print(f"Final optimizer tolerance         : {refinement_seconds:.3f} s")
    print(f"Minimum body altitude             : {min_body_altitude_deg:.3f} deg")
    print()
    print(
        "NOTE: Candidate generation is direct/event-based, not a regular "
        "time-of-day grid search. The analytical astronomy model is still "
        "not JPL-grade, especially for the Moon over centuries."
    )

    all_results: dict[str, list[dict]] = {}

    for body_name in ("sun", "moon"):
        body_wall_start_s = time.perf_counter()  # [s]

        results = find_sun_moon_alignment(
            lat=observer_lat_deg,  # [deg]
            lon=observer_lon_deg,  # [deg]
            alt=observer_altitude_msl_m,  # [m MSL]
            offset=observer_vertical_offset_m,  # [m]
            latO=object_lat_deg,  # [deg]
            lonO=object_lon_deg,  # [deg]
            altO=object_altitude_msl_m,  # [m MSL]
            offsetO=object_vertical_offset_m,  # [m]
            start=start_time,  # [datetime]
            end=end_time,  # [datetime]
            body=body_name,  # [-]
            target_x_deg=target_x_deg,  # [deg]
            target_y_deg=target_y_deg,  # [deg]
            tolerance_x_deg=tolerance_x_deg,  # [deg]
            tolerance_y_deg=tolerance_y_deg,  # [deg]
            refinement_seconds=refinement_seconds,  # [s]
            moon_declination_step_hours=moon_declination_step_hours,  # [h]
            local_refinement_half_window_hours=local_refinement_half_window_hours,  # [h]
            min_body_altitude_deg=min_body_altitude_deg,  # [deg]
            refraction=refraction,  # [-]
            pressure_mbar=pressure_mbar,  # [mbar = hPa]
            temperature_c=temperature_c,  # [deg C]
            observer_geoid_undulation_m=observer_geoid_undulation_m,  # [m]
            object_geoid_undulation_m=object_geoid_undulation_m,  # [m]
            output_timezone=output_timezone,  # [-]
            return_closest_if_none=return_closest_if_none,  # [-]
            max_results=max_results,  # [-]
            progress_every_years=progress_every_years,  # [years]
        )

        all_results[body_name] = results

        body_wall_elapsed_s = time.perf_counter() - body_wall_start_s  # [s]
        print(
            f"{body_name.upper()} search complete | "
            f"returned results: {len(results)} | "
            f"wall time: {body_wall_elapsed_s:.1f} s"
        )

    print_results("sun", all_results["sun"])
    print_results("moon", all_results["moon"])

    print()
    print("=" * 144)
    print(" BEST RESULT PER CELESTIAL OBJECT")
    print("=" * 144)
    print_best_summary("sun", all_results["sun"])
    print_best_summary("moon", all_results["moon"])

    # Object geometry is identical for Sun and Moon, so print it once.
    first_result = None
    for body_name in ("sun", "moon"):
        if all_results[body_name]:
            first_result = all_results[body_name][0]
            break

    if first_result is not None:
        print()
        print("=" * 80)
        print(" OBJECT / TARGET SKY GEOMETRY")
        print("=" * 80)
        print(f"Object distance [m]          : {first_result['object_distance_m']:.3f}")
        print(f"Object azimuth [deg]         : {first_result['object_azimuth_deg']:.6f}")
        print(f"Object altitude [deg]        : {first_result['object_altitude_deg']:.6f}")
        print(f"Target sky azimuth [deg]     : {first_result['target_sky_azimuth_deg']:.6f}")
        print(f"Target sky altitude [deg]    : {first_result['target_sky_altitude_deg']:.6f}")
        print(f"Target declination [deg]     : {first_result['target_declination_deg']:.6f}")
        print(f"Target hour angle [deg]      : {first_result['target_hour_angle_deg']:.6f}")

    # -------------------------------------------------------------------------
    # Interactive simplified 3-D views of the best Sun and Moon results
    # -------------------------------------------------------------------------

    if show_3d_best_scenes:
        show_best_scenes_3d(
            all_results,
            block=True,
        )


if __name__ == "__main__":
    main()
