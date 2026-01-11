"""
Plot JWST position vs time using JPL Horizons vectors (1 day step).
Save as jwst_horizons_plot.py and run with python3 (requires requests, numpy, matplotlib, pandas).
"""

import datetime as dt
import io
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ========== USER SETTINGS ==========
# time span (modify if you want another year)
START_TIME = "2025-11-24"  # inclusive, UTC/TDB string accepted by Horizons
STOP_TIME = "2026-11-24"  # inclusive
STEP_SIZE = "1 d"  # user asked 1 day step
# JWST target for Horizons:
COMMAND = "-170"  # JWST Horizons identifier (works with name or -170)
CENTER = "@0"  # solar-system barycenter; for geocentric use "500@399" or "399"
OUTFILE = "jwst_vectors.txt"  # local cache
# Horizons API endpoint
HORIZONS_API = "https://ssd-api.jpl.nasa.gov/api/horizons.api"

# ========== Build the query URL ==========
# We request format=text and VECTORS (table style 3 = geometric cartesian states X/Y/Z Vx/Vy/Vz)
params = {
    "format": "text",
    "COMMAND": f"'{COMMAND}'",
    "EPHEM_TYPE": "VECTORS",
    "CENTER": f"'{CENTER}'",
    "START_TIME": f"'{START_TIME}'",
    "STOP_TIME": f"'{STOP_TIME}'",
    "STEP_SIZE": f"'{STEP_SIZE}'",
    "VEC_TABLE": "3",  # table format with X,Y,Z,VX,VY,VZ
    "OUT_UNITS": "KM-S",  # km and km/s
    "REF_SYSTEM": "ICRF",  # inertial frame
    "MAKE_EPHEM": "YES",
    "OBJ_DATA": "NO",
}


# Helper to build URL (for logging / reproducibility)
def build_url(base, params):
    q = "&".join(f"{k}={requests.utils.quote(v, safe='')}" for k, v in params.items())
    return base + "?" + q


def download_horizons_text(params, outfile=None, show_url=False):
    # url = build_url(HORIZONS_API, params)
    url = Path("src", "horizons_results.txt").read_text().strip()

    """if show_url:
        print("Horizons URL:\n", url)
    print("Requesting Horizons... this runs on your machine and queries JPL's API.")
    r = requests.get(url, timeout=60)
    r.raise_for_status()"""
    with open(url, "r", encoding="utf-8") as f:
        r = io.StringIO(f.read())
    text = r.read()
    if outfile:
        with open(outfile, mode="w", encoding="utf-8") as f:
            f.write(text)
    return text


def parse_vectors_from_text(text):
    """
    Extract data between $$SOE and $$EOE. Try to parse commonly returned vector formats.
    Expected typical line (VEC_TABLE=3): JDTDB, Calendar Date (TDB), X, Y, Z, VX, VY, VZ
    But we will be robust: we look for lines that can be split into float tokens of length >= 7.
    """
    lines = text.splitlines()
    # find start/end markers
    try:
        i0 = next(i for i, l in enumerate(lines) if l.strip().startswith("$$SOE"))
        i1 = next(i for i, l in enumerate(lines) if l.strip().startswith("$$EOE"))
    except StopIteration:
        raise ValueError(
            "Couldn't find $$SOE/$$EOE in Horizons output. Full output was saved for inspection."
        )
    data_lines = lines[i0 + 1 : i1]
    rows = []
    dates = []
    for L in data_lines:
        L = L.strip()
        if not L:
            continue
        # horizons often uses comma-separated values; try splitting by comma
        parts = [p.strip() for p in L.split(",")]
        # remove empty parts
        parts = [p for p in parts if p != ""]
        # Try to detect calendar date token among second token(s)
        # Heuristic: if parts[1] looks like "A.D. 2025-Nov-24 00:00:00.0000" or "2025-Nov-24 00:00:00.0000"
        date_token = None
        coord_tokens = None
        # Search for a token that contains a year pattern "202" etc
        for idx, p in enumerate(parts[:4]):  # date is early in the line
            if any(
                year_str in p
                for year_str in (
                    "2020",
                    "2021",
                    "2022",
                    "2023",
                    "2024",
                    "2025",
                    "2026",
                    "2027",
                    "2028",
                    "2029",
                )
            ):
                date_token = p
                # rest tokens after idx are numeric coordinates
                coord_tokens = parts[idx + 1 :]
                break
        if date_token is None:
            # fallback: some outputs use JDTDB then values; detect if first token is JD (a large number) followed by floats
            try:
                jdt = float(parts[0])
                # date might be second token or composed tokens; attempt parse later
                date_token = None
                coord_tokens = parts[2:] if len(parts) > 6 else parts[1:]
            except Exception:
                coord_tokens = parts[-6:]
        # Now attempt to parse coordinates as floats; we need at least 3 floats for X,Y,Z
        float_vals = []
        for tok in coord_tokens:
            # remove potential extraneous characters (units, parentheses)
            tok_clean = (
                tok.replace("(", "")
                .replace(")", "")
                .replace("d", "")
                .replace("A.D.", "")
                .strip()
            )
            try:
                v = float(tok_clean)
                float_vals.append(v)
            except:
                # skip non-float tokens
                pass
        if len(float_vals) >= 3:
            # Accept first three as X,Y,Z; if 6 floats available, interpret velocities too
            x, y, z = float_vals[0], float_vals[1], float_vals[2]
            vx = vy = vz = None
            if len(float_vals) >= 6:
                vx, vy, vz = float_vals[3], float_vals[4], float_vals[5]
            # Parse date more robustly: try to find calendar substring in the original line
            # Many horizons vector lines have 'A.D. YYYY-Mon-DD HH:MM:SS.SSSS' between commas \u2014 try to extract with a regex-like approach
            date_str = None
            for candidate in parts[:6]:
                if any(
                    year in candidate
                    for year in (
                        "2020",
                        "2021",
                        "2022",
                        "2023",
                        "2024",
                        "2025",
                        "2026",
                        "2027",
                    )
                ):
                    # remove "A.D." or trailing commas
                    date_str = candidate.replace("A.D.", "").strip()
                    break
            if date_str is None:
                # fallback: use JD from first column if available -> convert to datetime (TDB approx)
                try:
                    jd = float(parts[0])
                    # convert JD (TDB) to datetime UTC approx (not exact for leap seconds) using algorithm:
                    # We'll convert JD to calendar assuming JD is Julian Day (adds offset)
                    # This is approximate for plotting and matches the user request uses 1-day step.
                    J = int(jd + 0.5)
                    # crude conversion using astropy would be better; here we just store JD number
                    date_obj = jd
                except:
                    date_obj = None
            else:
                # try parse like '2025-Nov-24 00:00:00.0000' -> convert to datetime
                # replace possible commas and excess text
                ds = date_str
                # remove possible leading "A.D." or "AD" markers
                ds = ds.replace("A.D.", "").replace("AD", "").strip()
                # try multiple formats
                date_obj = None
                fmts = [
                    "%Y-%b-%d %H:%M:%S.%f",
                    "%Y-%b-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                ]
                for fmt in fmts:
                    try:
                        date_obj = dt.datetime.strptime(ds, fmt)
                        break
                    except:
                        pass
            rows.append((date_obj, x, y, z, vx, vy, vz))
    if not rows:
        raise ValueError(
            "No numeric vector rows parsed. Check Horizons output format (VEC_TABLE, OUT_UNITS)."
        )
    # Build pandas DataFrame
    df_rows = []
    for r in rows:
        date_obj = r[0]
        # If date_obj is a JD numeric, convert to datetime roughly:
        if isinstance(date_obj, float):
            # convert JD to datetime UTC approx
            jd = date_obj
            # Convert JD to datetime (UTC) \u2014 approximate using algorithm
            # JD->datetime: (this will be close enough for daily plotting)
            unix_days = jd - 2451545.0  # days since J2000
            # epoch J2000 = 2000-01-01 12:00:00 TT ~ 2000-01-01 11:59:27.816 UTC (approx). We'll use 2000-01-01 12:00
            epoch = dt.datetime(2000, 1, 1, 12, 0, 0)
            date_dt = epoch + dt.timedelta(days=unix_days)
        else:
            date_dt = date_obj
        df_rows.append(
            {
                "date": date_dt,
                "x": r[1],
                "y": r[2],
                "z": r[3],
                "vx": r[4],
                "vy": r[5],
                "vz": r[6],
            }
        )
    df = pd.DataFrame(df_rows)
    # drop rows with None dates (should be rare)
    df = df[df["date"].notna()].reset_index(drop=True)
    return df


def plot_jwst_timeseries(df, start_time=START_TIME, stop_time=STOP_TIME):
    # compute distance to barycenter or Earth: if center was '@0' -> barycentric; Earth distance requires an Earth ephemeris
    # For demonstration, we compute radial distance from origin (barycenter)
    df = df.copy()
    df["r_km"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    # plot x,y,z vs time and distance
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(df["date"], df["x"], label="x (km)")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(df["date"], df["y"], label="y (km)")
    axes[1].grid(True)
    axes[1].legend()
    axes[2].plot(df["date"], df["z"], label="z (km)")
    axes[2].grid(True)
    axes[2].legend()
    axes[3].plot(
        df["date"], df["r_km"] / 1e6, label="distance from barycenter (10^6 km)"
    )
    axes[3].grid(True)
    axes[3].legend()
    axes[-1].set_xlabel("date (UTC / TDB)")
    # Add vertical lines at station-keeping cadence every 21 days (approx)
    try:
        start_dt = df["date"].iloc[0]
        end_dt = df["date"].iloc[-1]
        # generate list of SK dates
        sk_dates = []
        cur = start_dt
        while cur <= end_dt:
            sk_dates.append(cur)
            cur = cur + dt.timedelta(days=21)
        for ax in axes:
            for sd in sk_dates:
                ax.axvline(sd, color="k", linestyle="--", alpha=0.25)
    except Exception:
        pass
    plt.suptitle(
        "JWST position components and barycentric distance (data from JPL Horizons)"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# =======================
# Main execution
# =======================
if __name__ == "__main__":
    # 1) download

    text = download_horizons_text(params, outfile=OUTFILE, show_url=True)
    """except Exception as e:
        print("Erreur lors du t�l�chargement depuis Horizons:", e)
        print(
            "Si vous �tes dans un environnement sans Internet, t�l�chargez manuellement via:"
        )
        print(build_url(HORIZONS_API, params))
        sys.exit(1)
    """

    # 2) parse
    try:
        df = parse_vectors_from_text(text)
    except Exception as e:
        print("Erreur au parsing des vecteurs Horizons:", e)
        # save text for inspection
        with open("horizons_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(
            "Le fichier complet Horizons a �t� sauvegard� sous 'horizons_raw_output.txt' pour inspection."
        )
        sys.exit(1)

    print("Parsed rows:", len(df))
    print(df.head())

    # 3) plot
    plot_jwst_timeseries(df)
