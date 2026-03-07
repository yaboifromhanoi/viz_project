from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="NHIS Health Dashboard", layout="wide")
st.title("NHIS Health Outcomes Dashboard")
st.caption("Interactive dashboard using cleaned NHIS microdata with weighted regional summaries.")

BASE_DIR = Path(__file__).resolve().parent
HEALTH_CSV = BASE_DIR / "health_data_clean.csv"
REGIONS_GPQ = BASE_DIR / "regions.gpq"   # produced by preprocessing.py

REGION_LABELS = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West",
}

HEALTH_ORDER = ["Excellent", "Very good", "Good", "Fair", "Poor"]
REGION_ORDER = ["Northeast", "Midwest", "South", "West"]

# ----------------------------
# Helper functions
# ----------------------------
def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))

def yes_no_label(x):
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan
    if x == 1:
        return "Yes"
    if x == 0:
        return "No"
    return np.nan

def poverty_bin(p):
    """
    Cleaned POVLEV appears to be income-to-poverty ratio.
    Example: 1.00 = poverty line, 2.00 = 200% of poverty line.
    """
    if pd.isna(p):
        return np.nan
    try:
        p = float(p)
    except Exception:
        return np.nan

    if p < 1.0:
        return "<100% poverty"
    elif p < 1.38:
        return "100–137% poverty"
    elif p < 2.0:
        return "138–199% poverty"
    elif p < 4.0:
        return "200–399% poverty"
    else:
        return "400%+ poverty"

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_microdata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put health_data_clean.csv in the same folder as app.py.")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Numeric conversions
    numeric_cols = [
        "YEAR", "REGION", "SAMPWEIGHT", "POVLEV",
        "USUALPL", "DELAYCOST", "HINOTCOV", "HS_OR_LESS"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Region labels
    if "REGION" in df.columns:
        df["REGION_label"] = df["REGION"].map(REGION_LABELS)

    # Binary labels from cleaned numeric fields
    if "HINOTCOV" in df.columns:
        # In this cleaned data, 0 appears to mean insured, 1 means uninsured
        df["insurance_status"] = df["HINOTCOV"].map({0.0: "Insured", 1.0: "Uninsured"})

    if "USUALPL" in df.columns:
        df["USUALPL_label"] = df["USUALPL"].apply(yes_no_label)

    if "DELAYCOST" in df.columns:
        df["DELAYCOST_label"] = df["DELAYCOST"].apply(yes_no_label)

    if "HS_OR_LESS" in df.columns:
        df["HS_OR_LESS_label"] = df["HS_OR_LESS"].apply(yes_no_label)

    # Poverty bins from cleaned ratio
    if "POVLEV" in df.columns:
        df["POVERTY_GROUP"] = df["POVLEV"].apply(poverty_bin)

    return df

@st.cache_data
def load_regions(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run preprocessing.py to generate regions.gpq.")
    return gpd.read_parquet(path)

def compute_region_metrics(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.dropna(subset=["REGION"]).copy()
    temp["REGION"] = temp["REGION"].astype(int)

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["SAMPWEIGHT"]

        uninsured_rate = weighted_mean(g["HINOTCOV"], w) if "HINOTCOV" in g.columns else np.nan
        hs_or_less_rate = weighted_mean(g["HS_OR_LESS"], w) if "HS_OR_LESS" in g.columns else np.nan
        avg_povlev = weighted_mean(g["POVLEV"], w) if "POVLEV" in g.columns else np.nan

        # HEALTH is already labeled, so convert to ordered score for a weighted average
        health_map = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
        health_num = g["HEALTH"].map(health_map) if "HEALTH" in g.columns else pd.Series(np.nan, index=g.index)
        avg_health = weighted_mean(health_num, w)

        return pd.Series({
            "n_obs": len(g),
            "weighted_pop": float(w.sum()),
            "pct_uninsured": uninsured_rate,
            "pct_hs_or_less": hs_or_less_rate,
            "avg_povlev": avg_povlev,
            "avg_health_score": avg_health,
        })

    out = temp.groupby("REGION").apply(agg).reset_index()
    out["region_name"] = out["REGION"].map(REGION_LABELS)
    out = out.sort_values("REGION")
    return out

# ----------------------------
# Load
# ----------------------------
try:
    micro = load_microdata(HEALTH_CSV)
    regions = load_regions(REGIONS_GPQ)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ----------------------------
# Data notes
# ----------------------------
with st.expander("Data notes"):
    st.markdown(
        """
- This version of the dashboard uses a cleaned dataset derived from the original datasets  downloaded from NHIS and TIGER.
- Several variables are already recoded into readable labels, including:
  - `SEX`
  - `RACENEW`
  - `URBRRL`
  - `HEALTH`
  - `EDUC_LVL`
- `POVLEV` is recorded as an **income-to-poverty ratio**:
  - `1.0` = poverty line
  - `2.0` = 200% of poverty
  - `4.0` = 400% of poverty
- `HINOTCOV` is noted here as:
  - `0 = Insured`
  - `1 = Uninsured`
- Regional summaries are weighted using `SAMPWEIGHT`.
        """
    )

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")
df = micro.copy()

# Year
if "YEAR" in df.columns and df["YEAR"].notna().any():
    years = sorted(df["YEAR"].dropna().unique().astype(int).tolist())
    if len(years) == 1:
        st.sidebar.selectbox("Year", years, index=0, disabled=True)
        selected_years = years
    else:
        yr = st.sidebar.multiselect("Year", years, default=years)
        selected_years = yr
    df = df[df["YEAR"].isin(selected_years)]

# Region
if "REGION_label" in df.columns:
    region_opts = [r for r in REGION_ORDER if r in df["REGION_label"].dropna().unique().tolist()]
    sel_regions = st.sidebar.multiselect("Region", region_opts, default=region_opts)
    df = df[df["REGION_label"].isin(sel_regions)]

# Sex
if "SEX" in df.columns:
    sex_opts = sorted(df["SEX"].dropna().unique().tolist())
    if sex_opts:
        sel_sex = st.sidebar.multiselect("Sex", sex_opts, default=sex_opts)
        df = df[df["SEX"].isin(sel_sex)]

# Race
if "RACENEW" in df.columns:
    race_opts = sorted(df["RACENEW"].dropna().unique().tolist())
    if race_opts:
        sel_race = st.sidebar.multiselect("Race", race_opts, default=race_opts)
        df = df[df["RACENEW"].isin(sel_race)]

# Urban/Rural Status
if "URBRRL" in df.columns:
    urb_opts = sorted(df["URBRRL"].dropna().unique().tolist())
    if urb_opts:
        sel_urb = st.sidebar.multiselect("Urban/Rural", urb_opts, default=urb_opts)
        df = df[df["URBRRL"].isin(sel_urb)]

# Education Attainment
if "EDUC_LVL" in df.columns:
    educ_opts = sorted(df["EDUC_LVL"].dropna().unique().tolist())
    if educ_opts:
        sel_educ = st.sidebar.multiselect("Education level", educ_opts, default=educ_opts)
        df = df[df["EDUC_LVL"].isin(sel_educ)]

# Poverty Level
if "POVERTY_GROUP" in df.columns:
    pov_order = [
        "<100% poverty",
        "100–137% poverty",
        "138–199% poverty",
        "200–399% poverty",
        "400%+ poverty",
    ]
    pov_opts = [p for p in pov_order if p in df["POVERTY_GROUP"].dropna().unique().tolist()]
    if pov_opts:
        sel_pov = st.sidebar.multiselect("Poverty level", pov_opts, default=pov_opts)
        df = df[df["POVERTY_GROUP"].isin(sel_pov)]

# Health
if "HEALTH" in df.columns:
    health_opts = [h for h in HEALTH_ORDER if h in df["HEALTH"].dropna().unique().tolist()]
    if health_opts:
        sel_health = st.sidebar.multiselect("Self-rated health", health_opts, default=health_opts)
        df = df[df["HEALTH"].isin(sel_health)]

# Insurance
if "insurance_status" in df.columns:
    ins_opts = [x for x in ["Insured", "Uninsured"] if x in df["insurance_status"].dropna().unique().tolist()]
    if ins_opts:
        sel_ins = st.sidebar.multiselect("Insurance status", ins_opts, default=ins_opts)
        df = df[df["insurance_status"].isin(sel_ins)]

# Usual place for care
if "USUALPL_label" in df.columns:
    usual_opts = [x for x in ["Yes", "No"] if x in df["USUALPL_label"].dropna().unique().tolist()]
    if usual_opts:
        sel_usual = st.sidebar.multiselect("Has usual place for care", usual_opts, default=usual_opts)
        df = df[df["USUALPL_label"].isin(sel_usual)]

# Delayed care due to cost
if "DELAYCOST_label" in df.columns:
    delay_opts = [x for x in ["Yes", "No"] if x in df["DELAYCOST_label"].dropna().unique().tolist()]
    if delay_opts:
        sel_delay = st.sidebar.multiselect("Delayed care due to cost", delay_opts, default=delay_opts)
        df = df[df["DELAYCOST_label"].isin(sel_delay)]

st.sidebar.divider()
st.sidebar.write(f"Rows after filters: **{len(df):,}**")

# ----------------------------
# Downloads
# ----------------------------
st.sidebar.header("⬇Downloads")
st.sidebar.download_button(
    "Download filtered microdata",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_microdata.csv",
    mime="text/csv",
)

# ----------------------------
# Metrics + map
# ----------------------------
metrics = compute_region_metrics(df)

st.sidebar.download_button(
    "Download region summary",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="region_summary.csv",
    mime="text/csv",
)

gdf = regions.merge(metrics, on="REGION", how="left")

metric_map = {
    "Percent Uninsured": "pct_uninsured",
    "Percent HS or Less": "pct_hs_or_less",
    "Average Poverty Ratio": "avg_povlev",
    "Average Health Score": "avg_health_score",
}

selected_metric = st.selectbox("Metric", list(metric_map.keys()))
metric_col = metric_map[selected_metric]

colormap = st.selectbox(
    "Color scheme",
    ["plasma", "viridis", "magma", "YlOrRd", "Blues", "RdYlGn"],
)

left, right = st.columns([3, 1])

with left:
    st.subheader(f"Choropleth by Region — {selected_metric}")
    fig, ax = plt.subplots(figsize=(11, 6))
    gdf.plot(
        column=metric_col,
        legend=True,
        cmap=colormap,
        ax=ax,
        edgecolor="white",
        linewidth=0.8,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
        legend_kwds={"shrink": 0.7, "label": selected_metric},
    )
    gdf.boundary.plot(ax=ax, linewidth=1.0)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Quick stats")
    s = metrics[metric_col].dropna()
    if len(s) == 0:
        st.info("No values available.")
    else:
        st.metric("Mean", f"{s.mean():.3f}")
        st.metric("Min", f"{s.min():.3f}")
        st.metric("Max", f"{s.max():.3f}")

st.divider()

# ----------------------------
# Searchable Table
# ----------------------------
st.subheader("Region Summary Table")
search = st.text_input("Search Table", value="").strip().lower()
table = metrics.copy()

if search:
    mask = table.astype(str).apply(lambda r: search in " ".join(r.values).lower(), axis=1)
    table = table[mask]

st.dataframe(
    table[[
        "REGION",
        "region_name",
        "n_obs",
        "weighted_pop",
        "pct_uninsured",
        "pct_hs_or_less",
        "avg_povlev",
        "avg_health_score",
    ]],
    use_container_width=True,
)

st.divider()


st.markdown(
        """
***VARIABLE DESCRIPTIONS:***

***n_obs = Number of recorded observations for a given region.***

***weighted_pop = The recorded population for each region for each region at time of census.***

***pct_uninsured = The percentage of a given region's population that is not insured.***

***avg_povlev =  Average household income relative to the federal poverty line (displayed as a ratio).***

***pct_hs_or_less = Percentage of those within a region that only have a high school education or less.***

***avg_health = Average health outcomes (1:Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor)***
        """
)
# ----------------------------
# Region detail view
# ----------------------------
st.subheader("Region Detail View")

visible_region_codes = table["REGION"].dropna().astype(int).tolist()
if not visible_region_codes:
    visible_region_codes = [1, 2, 3, 4]

visible_region_names = [REGION_LABELS.get(r, f"Region {r}") for r in visible_region_codes]
selected_region_name = st.selectbox("Select a region", visible_region_names, index=0)

selected_region = {
    REGION_LABELS.get(r, f"Region {r}"): r for r in visible_region_codes
}[selected_region_name]

sub = df[df["REGION"] == selected_region].copy()

def wmean(series):
    return weighted_mean(series, sub["SAMPWEIGHT"])

uninsured_rate = wmean(sub["HINOTCOV"]) if "HINOTCOV" in sub.columns else np.nan
avg_pov = wmean(sub["POVLEV"]) if "POVLEV" in sub.columns else np.nan
hs_or_less = wmean(sub["HS_OR_LESS"]) if "HS_OR_LESS" in sub.columns else np.nan

health_map = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
avg_health = wmean(sub["HEALTH"].map(health_map)) if "HEALTH" in sub.columns else np.nan

st.write(f"**Selected:** {selected_region_name} — rows: **{len(sub):,}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Uninsured (weighted)", f"{uninsured_rate:.3f}" if pd.notna(uninsured_rate) else "NA")
m2.metric("Avg poverty ratio", f"{avg_pov:.2f}" if pd.notna(avg_pov) else "NA")
m3.metric("Avg health score", f"{avg_health:.2f}" if pd.notna(avg_health) else "NA")
m4.metric("HS or less (weighted)", f"{hs_or_less:.3f}" if pd.notna(hs_or_less) else "NA")

c1, c2 = st.columns(2)

with c1:
    if "insurance_status" in sub.columns and sub["insurance_status"].notna().any():
        fig_ins, ax_ins = plt.subplots(figsize=(10, 3))
        sub["insurance_status"].value_counts().reindex(["Insured", "Uninsured"]).dropna().plot(kind="bar", ax=ax_ins)
        ax_ins.set_title("Insurance status (filtered)")
        ax_ins.set_xlabel("")
        ax_ins.set_ylabel("Count")
        st.pyplot(fig_ins)
        plt.close(fig_ins)

with c2:
    if "HEALTH" in sub.columns and sub["HEALTH"].notna().any():
        fig_h, ax_h = plt.subplots(figsize=(10, 3))
        sub["HEALTH"].value_counts().reindex(HEALTH_ORDER).dropna().plot(kind="bar", ax=ax_h)
        ax_h.set_title("Self-rated health (filtered)")
        ax_h.set_xlabel("")
        ax_h.set_ylabel("Count")
        st.pyplot(fig_h)
        plt.close(fig_h)

st.divider()

# ----------------------------
# Bar Chart by Region
# ----------------------------
st.subheader("Bar Chart by Region")
fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
metrics.set_index("region_name")[metric_col].plot(kind="bar", ax=ax_bar)
ax_bar.set_ylabel(selected_metric)
ax_bar.set_xlabel("")
st.pyplot(fig_bar)
plt.close(fig_bar)

st.divider()
st.markdown(
'''
Data Sources:

IPUMS: https://healthsurveys.ipums.org/

TIGER Geodata Base: https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/ 
'''
)