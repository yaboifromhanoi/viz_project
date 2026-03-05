from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd

# ----------------------------
# Paths (all in same directory)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

HEALTH_CSV = BASE_DIR /"health_outcomes.csv"
STATES_DIR = BASE_DIR /"states"
STATES_SHP = BASE_DIR /"states"/"tl_2024_us_state.shp"

REGION_NAME = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}

# Census regions (contiguous only)
NORTHEAST = {"CT","ME","MA","NH","RI","VT","NJ","NY","PA"}
MIDWEST   = {"IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD"}
SOUTH     = {"DE","FL","GA","MD","NC","SC","VA","DC","WV","AL","KY","MS","TN","AR","LA","OK","TX"}
WEST      = {"AZ","CO","ID","MT","NV","NM","UT","WY","CA","OR","WA"}

def unzip_states_if_needed():
    if STATES_SHP.exists():
        return

    if not STATES_ZIP.exists():
        raise FileNotFoundError("states.zip not found in project folder.")

    with zipfile.ZipFile(STATES_ZIP, "r") as z:
        z.extractall(BASE_DIR)

    if not STATES_SHP.exists():
        raise FileNotFoundError("Shapefile not found after unzip. Check zip structure.")

def stusps_to_region(st):
    if st in NORTHEAST: return 1
    if st in MIDWEST: return 2
    if st in SOUTH: return 3
    if st in WEST: return 4
    return np.nan

def weighted_mean(x, w):
    mask = x.notna() & w.notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])

def main():

    if not HEALTH_CSV.exists():
        raise FileNotFoundError("health_outcomes.csv not found.")

    unzip_states_if_needed()

    # ---- Load data ----
    df = pd.read_csv(HEALTH_CSV)
    df.columns = df.columns.str.strip()

    df["REGION"] = pd.to_numeric(df["REGION"], errors="coerce")
    df["SAMPWEIGHT"] = pd.to_numeric(df["SAMPWEIGHT"], errors="coerce")

    # Insurance coding
    df["insured"] = np.where(df["HINOTCOV"] == 2, 1,
                      np.where(df["HINOTCOV"] == 1, 0, np.nan))

    # HS or less (your rule)
    df["hs_or_less"] = np.where(
        (df["EDUC"] == 0) | (df["EDUC"] > 900),
        np.nan,
        np.where(df["EDUC"] < 300, 1, 0),
    )

    # ---- Aggregate by region ----
    def agg(g):
        w = g["SAMPWEIGHT"]
        insured_rate = weighted_mean(g["insured"], w)

        return pd.Series({
            "n_obs": len(g),
            "weighted_pop": w.sum(),
            "pct_uninsured": 1 - insured_rate if pd.notna(insured_rate) else np.nan,
            "avg_educ": weighted_mean(g["EDUC"], w),
            "pct_hs_or_less": weighted_mean(g["hs_or_less"], w),
            "avg_povlev": weighted_mean(g["POVLEV"], w),
            "avg_health": weighted_mean(g["HEALTH"], w),
        })

    metrics = df.groupby("REGION", dropna=True).apply(agg).reset_index()
    metrics["region_name"] = metrics["REGION"].map(REGION_NAME)

    metrics.to_parquet(BASE_DIR / "region_metrics.parquet", index=False)

    # ---- Build region geometries ----
    states = gpd.read_file(STATES_SHP)

    drop_st = {"AS","GU","MP","PR","VI","AK","HI"}
    states = states[~states["STUSPS"].isin(drop_st)].copy()

    states["REGION"] = states["STUSPS"].map(stusps_to_region)
    states = states.dropna(subset=["REGION"])
    states["REGION"] = states["REGION"].astype(int)

    regions = states.dissolve(by="REGION", as_index=False)
    regions["region_name"] = regions["REGION"].map(REGION_NAME)
    regions["geometry"] = regions["geometry"].buffer(0)

    regions.to_parquet(BASE_DIR / "regions.gpq", index=False)

    print("Files created in project folder:")
    print(" - region_metrics.parquet")
    print(" - regions.gpq")

if __name__ == "__main__":
    main()