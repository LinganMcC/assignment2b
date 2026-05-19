"""
Parse the raw SCATS October 2006 spreadsheet into a tidy time-series CSV.

OWNER: Person A

Input:  data/raw/Scats_Data_October_2006.xls
Output: data/processed/flow_timeseries.csv  (per INTERFACES.md interface 1)
        data/processed/site_locations.csv   (one canonical lat/lon per site)

Usage:
    python -m src.data.parse_scats

Notes for the team
------------------
- The raw file has multiple rows per (site, date) — one per detector loop at the
  intersection (typically 4: north/south/east/west approaches). We sum these to
  get total intersection flow per 15-minute interval, which is what we want for
  travel-time estimation between intersections.
- Lat/lon also varies slightly per detector (each sits on a different approach).
  We use the centroid of all detectors as the canonical site location.
- The 'Date' column in the raw file always shows time = 00:15. This is a quirk
  of the original spreadsheet — ignore the time portion. The actual time of each
  reading comes from the V00..V95 column (V00 = 00:00–00:15, V01 = 00:15–00:30,
  ..., V95 = 23:45–00:00).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Scats_Data_October_2006.xls"
OUT_TIMESERIES = PROJECT_ROOT / "data" / "processed" / "flow_timeseries.csv"
OUT_LOCATIONS = PROJECT_ROOT / "data" / "processed" / "site_locations.csv"


def load_raw() -> pd.DataFrame:
    """Load the Data sheet of the raw SCATS spreadsheet."""
    df = pd.read_excel(RAW_FILE, sheet_name="Data", header=1)
    return df


def melt_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide V00..V95 layout into a long-format dataframe with one row
    per (site, detector, datetime) and sum across detectors for each site.
    """
    # Identify the V columns (V00 through V95)
    v_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("V") and c[1:].isdigit()]
    assert len(v_cols) == 96, f"Expected 96 V-columns, got {len(v_cols)}"

    keep_cols = ["SCATS Number", "Date"] + v_cols
    long_df = df[keep_cols].melt(
        id_vars=["SCATS Number", "Date"],
        value_vars=v_cols,
        var_name="interval_code",
        value_name="flow_15min",
    )

    # Extract interval index (00..95) from "V00".."V95"
    long_df["interval_idx"] = long_df["interval_code"].str[1:].astype(int)

    # Build proper datetime: take the date from 'Date' (drop its bogus 00:15 time)
    # and add interval_idx * 15 minutes.
    base_date = pd.to_datetime(long_df["Date"]).dt.normalize()
    long_df["datetime"] = base_date + pd.to_timedelta(long_df["interval_idx"] * 15, unit="m")

    # Sum across all detectors at the same (site, datetime)
    grouped = (
        long_df.groupby(["SCATS Number", "datetime"], as_index=False)["flow_15min"]
        .sum()
        .rename(columns={"SCATS Number": "site_id", "flow_15min": "flow"})
    )

    # flow column should be int
    grouped["flow"] = grouped["flow"].astype(int)

    # Format datetime as ISO string for CSV (per INTERFACES.md)
    grouped["datetime"] = grouped["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Sort
    grouped = grouped.sort_values(["site_id", "datetime"]).reset_index(drop=True)

    return grouped[["site_id", "datetime", "flow"]]


def extract_site_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each SCATS site, compute the centroid of all detector lat/lons and grab
    a representative location name.

    Note: some detectors in the raw data have NB_LATITUDE=0, NB_LONGITUDE=0 as
    a missing-data sentinel. We exclude these before averaging.
    """
    # Drop rows with (0,0) coords or NaN — they would skew the centroid
    valid = df[(df["NB_LATITUDE"] != 0) & (df["NB_LONGITUDE"] != 0)].copy()
    valid = valid.dropna(subset=["NB_LATITUDE", "NB_LONGITUDE"])

    locs = (
        valid.groupby("SCATS Number")
        .agg(
            lat=("NB_LATITUDE", "mean"),
            lon=("NB_LONGITUDE", "mean"),
            name=("Location", "first"),
        )
        .reset_index()
        .rename(columns={"SCATS Number": "site_id"})
    )
    return locs


def main() -> None:
    print(f"Loading {RAW_FILE} ...")
    raw = load_raw()
    print(f"  Raw rows: {len(raw)}, sites: {raw['SCATS Number'].nunique()}")

    print("Building timeseries ...")
    ts = melt_to_timeseries(raw)
    print(f"  Timeseries rows: {len(ts)}")
    print(f"  Sites: {ts['site_id'].nunique()}")
    print(f"  Date range: {ts['datetime'].min()} to {ts['datetime'].max()}")

    print("Extracting site locations ...")
    locs = extract_site_locations(raw)
    print(f"  Sites with location: {len(locs)}")

    OUT_TIMESERIES.parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(OUT_TIMESERIES, index=False)
    locs.to_csv(OUT_LOCATIONS, index=False)

    print(f"\nWrote {OUT_TIMESERIES}")
    print(f"Wrote {OUT_LOCATIONS}")


if __name__ == "__main__":
    main()
