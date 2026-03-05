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
st.caption("Interactive filters on NHIS microdata; outputs aggregated to Census regions using survey weights.")

BASE_DIR = Path(__file__).resolve().parent
HEALTH_CSV = BASE_DIR / "health_outcomes.csv"
REGIONS_GPQ = BASE_DIR / "regions.gpq"  # created by preprocessing.py (dissolved regions)

# ----------------------------
# Label dictionaries / bins
# ----------------------------
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
SEX_LABELS = {1: "Male", 2: "Female"}
HEALTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

# RACENEW labels you provided
RACENEW_LABELS = {
    100: "White only",
    200: "Black/African American only",
    300: "American Indian/Alaska Native only",
    400: "Asian only",
    500: "Other Race and Multiple Race",
    510: "Other Race and Multiple Race (2019+ excl. AI/AN)",
    520: "Other Race",
    530: "Race Group Not Releasable",
    540: "Multiple Race",
    541: "Multiple Race (1999–2018 incl. AI/AN)",
    542: "American Indian/Alaska Native and Any Other Race",
    997: "Unknown/Refused",
    998: "Unknown/Not ascertained",
    999: "Unknown/Don't know",
}

def povlev_bin(p):
    """POVLEV ~ percent of poverty line (100=poverty line)."""
    if pd.isna(p):
        return np.nan
    try:
        p = float(p)
    except Exception:
        return np.nan
    if p < 100:
        return "<100% (Below poverty)"
    if p < 138:
        return "100–137% (Near-poor)"
    if p < 200:
        return "138–199%"
    if p < 400:
        return "200–399%"
    return "400%+"

def educ_group_from_code(e):
    """
    Dashboard-friendly education bins.
    IPUMS EDUC coding varies; these bins are consistent and readable.
    """
    if pd.isna(e):
        return np.nan
    try:
        e = float(e)
    except Exception:
        return np.nan
    if e == 0 or e >= 900:
        return "Missing/Unknown"
    if e < 300:
        return "HS or less"
    if e < 400:
        return "Some college"
    return "Bachelor's+"

def code01_to_yesno(x):
    if pd.isna(x):
        return np.nan
    try:
        x = int(x)
    except Exception:
        return np.nan
    if x == 1:
        return "Yes"
    if x == 0:
        return "No"
    return np.nan

def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    return float(np.average(x[m], weights=w[m]))

# ----------------------------
# Loaders
# ----------------------------
@st.cache_data
def load_microdata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put health_outcomes.csv in the same folder as app.py.")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Standardize numeric types
    for col in ["YEAR", "REGION", "SAMPWEIGHT", "AGE", "POVLEV", "HEALTH", "EDUC", "SEX", "HINOTCOV", "RACENEW"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SAMPWEIGHT" not in df.columns:
        df["SAMPWEIGHT"] = 1.0

    # Derived insured indicator (matching your earlier logic)
    # insured=1 if HINOTCOV==2; uninsured if HINOTCOV==1
    if "HINOTCOV" in df.columns:
        df["insured"] = np.where(df["HINOTCOV"] == 2, 1,
                          np.where(df["HINOTCOV"] == 1, 0, np.nan))
    else:
        df["insured"] = np.nan

    # Derived HS-or-less indicator (matching your earlier rule)
    if "EDUC" in df.columns:
        df["hs_or_less"] = np.where(
            (df["EDUC"] == 0) | (df["EDUC"] > 900),
            np.nan,
            np.where(df["EDUC"] < 300, 1, 0),
        )
    else:
        df["hs_or_less"] = np.nan

    # Labels for UI (do not overwrite original codes)
    df["REGION_label"] = df["REGION"].map(REGION_LABELS)
    if "SEX" in df.columns:
        df["SEX_label"] = df["SEX"].map(SEX_LABELS)
    if "HEALTH" in df.columns:
        df["HEALTH_label"] = df["HEALTH"].map(HEALTH_LABELS)
    if "POVLEV" in df.columns:
        df["POVLEV_bin"] = df["POVLEV"].apply(povlev_bin)
    if "EDUC" in df.columns:
        df["EDUC_group"] = df["EDUC"].apply(educ_group_from_code)
    if "RACENEW" in df.columns:
        df["RACENEW_label"] = df["RACENEW"].map(RACENEW_LABELS)

    df["insured_label"] = df["insured"].apply(lambda x: "Insured" if x == 1 else ("Uninsured" if x == 0 else np.nan))
    df["hs_or_less_label"] = df["hs_or_less"].apply(code01_to_yesno)

    for col in ["USUALPL", "TYPPLSICK", "DELAYCOST"]:
        if col in df.columns:
            df[f"{col}_label"] = df[col].apply(code01_to_yesno)

    return df

@st.cache_data
def load_regions(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing.py to generate regions.gpq in the same folder."
        )
    return gpd.read_parquet(path)

def compute_region_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base = df.dropna(subset=["REGION"]).copy()
    base["REGION"] = base["REGION"].astype(int)

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["SAMPWEIGHT"]
        insured_rate = weighted_mean(g["insured"], w)

        return pd.Series({
            "n_obs": len(g),
            "weighted_pop": float(w.sum()),
            "pct_uninsured": (1 - insured_rate) if pd.notna(insured_rate) else np.nan,
            "avg_povlev": weighted_mean(g["POVLEV"], w),
            "pct_hs_or_less": weighted_mean(g["hs_or_less"], w),
            "avg_health": weighted_mean(g["HEALTH"], w),
            "avg_educ": weighted_mean(g["EDUC"], w),
        })

    out = base.groupby("REGION").apply(agg).reset_index()
    out["region_name"] = out["REGION"].map(REGION_LABELS)
    out = out.sort_values("REGION")
    return out

# ----------------------------
# Load data
# ----------------------------
try:
    micro = load_microdata(HEALTH_CSV)
    regions = load_regions(REGIONS_GPQ)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ----------------------------
# Data notes expander
# ----------------------------
with st.expander("Data notes"):
    st.markdown(
        """
**Dataset**
- NHIS microdata extract with columns: YEAR, REGION, SEX, RACENEW, EDUC, POVLEV, HEALTH, USUALPL, TYPPLSICK, DELAYCOST, HINOTCOV, and others.

**Weights**
- Regional summaries use `SAMPWEIGHT` as person weights.

**Derived dashboard measures**
- **Insured/Uninsured**: derived from `HINOTCOV` using:
  - insured = 1 if `HINOTCOV == 2`
  - insured = 0 if `HINOTCOV == 1`
- **HS or less**: derived from `EDUC`:
  - missing if `EDUC == 0` or `EDUC >= 900`
  - HS or less = 1 if `EDUC < 300`, else 0
- **Poverty bins**: derived from `POVLEV` (% of poverty line):
  - <100%, 100–137%, 138–199%, 200–399%, 400%+
- **Health labels**: `HEALTH` is labeled Excellent → Poor (codes 1–5).
- **Race labels**: `RACENEW` uses the IPUMS code table you provided (White only, Black only, etc.).

**Note**
- IPUMS coding can vary by sample/year; this dashboard standardizes codes into policy-friendly groups for interpretation.
        """
    )

# ----------------------------
# Sidebar filters (all labeled)
# ----------------------------
st.sidebar.header("Filters")

df = micro.copy()

# Year (robust: handles single-year datasets)
if "YEAR" in df.columns and df["YEAR"].notna().any():
    years = sorted(df["YEAR"].dropna().unique().astype(int).tolist())

    if len(years) == 1:
        # Only one year available -> no slider
        selected_years = years
        st.sidebar.selectbox("Year", options=years, index=0, disabled=True)
    else:
        y_min, y_max = years[0], years[-1]
        yr = st.sidebar.slider("Year", y_min, y_max, (y_min, y_max))
        selected_years = list(range(yr[0], yr[1] + 1))

    df = df[df["YEAR"].isin(selected_years)]

# Region
region_opts = [r for r in ["Northeast", "Midwest", "South", "West"] if r in df["REGION_label"].dropna().unique().tolist()]
sel_regions = st.sidebar.multiselect("Region", region_opts, default=region_opts)
df = df[df["REGION_label"].isin(sel_regions)]

# Age
if "AGE" in df.columns and df["AGE"].notna().any():
    a_min = int(df["AGE"].dropna().min())
    a_max = int(df["AGE"].dropna().max())
    age_rng = st.sidebar.slider("Age", a_min, a_max, (a_min, a_max))
    df = df[df["AGE"].between(age_rng[0], age_rng[1], inclusive="both")]

# Sex
if "SEX_label" in df.columns:
    sex_opts = sorted(df["SEX_label"].dropna().unique().tolist())
    if sex_opts:
        sel_sex = st.sidebar.multiselect("Sex", sex_opts, default=sex_opts)
        df = df[df["SEX_label"].isin(sel_sex)]

# Race
if "RACENEW_label" in df.columns:
    race_opts = [r for r in df["RACENEW_label"].dropna().unique().tolist()]
    preferred_order = [
        "White only",
        "Black/African American only",
        "American Indian/Alaska Native only",
        "Asian only",
        "Other Race",
        "Multiple Race",
        "Other Race and Multiple Race",
        "Other Race and Multiple Race (2019+ excl. AI/AN)",
        "Multiple Race (1999–2018 incl. AI/AN)",
        "American Indian/Alaska Native and Any Other Race",
        "Race Group Not Releasable",
        "Unknown/Refused",
        "Unknown/Not ascertained",
        "Unknown/Don't know",
    ]
    race_opts = [r for r in preferred_order if r in race_opts] + [r for r in race_opts if r not in preferred_order]
    if race_opts:
        sel_race = st.sidebar.multiselect("Race (RACENEW)", race_opts, default=race_opts)
        df = df[df["RACENEW_label"].isin(sel_race)]

# Poverty bins
if "POVLEV_bin" in df.columns:
    pov_order = ["<100% (Below poverty)", "100–137% (Near-poor)", "138–199%", "200–399%", "400%+"]
    pov_opts = [p for p in pov_order if p in df["POVLEV_bin"].dropna().unique().tolist()]
    if pov_opts:
        sel_pov = st.sidebar.multiselect("Poverty level", pov_opts, default=pov_opts)
        df = df[df["POVLEV_bin"].isin(sel_pov)]

# Education group
if "EDUC_group" in df.columns:
    educ_order = ["HS or less", "Some college", "Bachelor's+", "Missing/Unknown"]
    educ_opts = [e for e in educ_order if e in df["EDUC_group"].dropna().unique().tolist()]
    if educ_opts:
        sel_educ = st.sidebar.multiselect("Education", educ_opts, default=educ_opts)
        df = df[df["EDUC_group"].isin(sel_educ)]

# Insurance
ins_opts = [x for x in ["Insured", "Uninsured"] if x in df["insured_label"].dropna().unique().tolist()]
if ins_opts:
    sel_ins = st.sidebar.multiselect("Insurance", ins_opts, default=ins_opts)
    df = df[df["insured_label"].isin(sel_ins)]

# Health
if "HEALTH_label" in df.columns:
    health_order = ["Excellent", "Very good", "Good", "Fair", "Poor"]
    health_opts = [h for h in health_order if h in df["HEALTH_label"].dropna().unique().tolist()]
    if health_opts:
        sel_health = st.sidebar.multiselect("Self-rated health", health_opts, default=health_opts)
        df = df[df["HEALTH_label"].isin(sel_health)]

# Access to care yes/no filters
for col, label in [("USUALPL_label", "Has usual place for care"), ("DELAYCOST_label", "Delayed care due to cost")]:
    if col in df.columns:
        opts = [x for x in ["Yes", "No"] if x in df[col].dropna().unique().tolist()]
        if opts:
            sel = st.sidebar.multiselect(label, opts, default=opts)
            df = df[df[col].isin(sel)]

st.sidebar.divider()
st.sidebar.write(f"Rows after filters: **{len(df):,}**")

# ----------------------------
# Downloads
# ----------------------------
st.sidebar.header("Downloads")
st.sidebar.download_button(
    "Download filtered microdata (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_microdata.csv",
    mime="text/csv",
)

# ----------------------------
# Compute metrics and merge for map
# ----------------------------
metrics = compute_region_metrics(df)
st.sidebar.download_button(
    "Download region summary (CSV)",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="region_summary.csv",
    mime="text/csv",
)

gdf = regions.merge(metrics, on="REGION", how="left")

metric_map = {
    "Percent uninsured": "pct_uninsured",
    "Average poverty level (POVLEV)": "avg_povlev",
    "Percent HS or less": "pct_hs_or_less",
    "Average self-rated health (HEALTH)": "avg_health",
    "Average education code (EDUC)": "avg_educ",
}
metric_label = st.selectbox("Metric", list(metric_map.keys()))
metric_col = metric_map[metric_label]
cmap = st.selectbox("Color scheme", ["plasma", "viridis", "magma", "YlOrRd", "Blues", "RdYlGn"], index=0)

# ----------------------------
# Map + stats
# ----------------------------
left, right = st.columns([3, 1])

with left:
    st.subheader(f"Choropleth by Region — {metric_label}")
    fig, ax = plt.subplots(figsize=(11, 6))
    gdf.plot(
        column=metric_col,
        legend=True,
        cmap=cmap,
        ax=ax,
        edgecolor="white",
        linewidth=0.8,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
        legend_kwds={"shrink": 0.7, "label": metric_label},
    )
    gdf.boundary.plot(ax=ax, linewidth=1.0)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Quick Stats")
    s = metrics[metric_col].dropna()
    if len(s) == 0:
        st.info("No values (relax filters).")
    else:
        st.metric("Mean", f"{s.mean():.3f}")
        st.metric("Min", f"{s.min():.3f}")
        st.metric("Max", f"{s.max():.3f}")

st.divider()

# ----------------------------
# Searchable table
# ----------------------------
st.subheader("Region Summary Table")
search = st.text_input("Search Table", value="").strip().lower()
table = metrics.copy()

if search:
    mask = table.astype(str).apply(lambda r: search in " ".join(r.values).lower(), axis=1)
    table = table[mask]

st.dataframe(
    table[["REGION", "region_name", "n_obs", "weighted_pop", "pct_uninsured", "avg_povlev", "pct_hs_or_less", "avg_health"]],
    use_container_width=True
)

st.markdown(
        """
***n_obs = Number of recorded observations for a given region.***

***weighted_pop = The recorded population for each region for each region at time of census.***

***pct_uninsured = The percentage of a given region's population that is not insured.***

***avg_povlev = Ratio of families/observations that live above the poverty level to those that do.***

***pct_hs_or_less = Percentage of those within a region that only have a high school education or less.***

***avg_health = Average health outcomes (1:Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor)***
        """
    )

# ----------------------------
# Region Detail View (dropdown UNDER header + shows names)
# ----------------------------
st.divider()
st.subheader("Region Detail View")

visible_region_codes = table["REGION"].dropna().astype(int).tolist()
if not visible_region_codes:
    visible_region_codes = [1, 2, 3, 4]

visible_region_names = [REGION_LABELS.get(int(r), f"Region {r}") for r in visible_region_codes]
selected_region_name = st.selectbox("Select a region", visible_region_names, index=0)
selected_region = {REGION_LABELS.get(int(r), f"Region {r}"): int(r) for r in visible_region_codes}[selected_region_name]

sub = df[df["REGION"] == selected_region].copy()

# Weighted metrics for region cards
def wmean(series):
    return weighted_mean(series, sub["SAMPWEIGHT"])

insured_rate = wmean(sub["insured"])
uninsured_rate = 1 - insured_rate if pd.notna(insured_rate) else np.nan
avg_pov = wmean(sub["POVLEV"])
avg_health = wmean(sub["HEALTH"])
pct_hs_or_less = wmean(sub["hs_or_less"])

st.write(f"**Selected:** {REGION_LABELS.get(int(selected_region), 'Unknown')} — rows: **{len(sub):,}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Uninsured (weighted)", f"{uninsured_rate:.3f}" if pd.notna(uninsured_rate) else "NA")
m2.metric("Avg POVLEV (weighted)", f"{avg_pov:.1f}" if pd.notna(avg_pov) else "NA")
m3.metric("Avg HEALTH (weighted)", f"{avg_health:.2f}" if pd.notna(avg_health) else "NA")
m4.metric("HS or less (weighted)", f"{pct_hs_or_less:.3f}" if pd.notna(pct_hs_or_less) else "NA")

# Charts
c1, c2 = st.columns(2)

with c1:
    if "AGE" in sub.columns and sub["AGE"].notna().any():
        fig_age, ax_age = plt.subplots(figsize=(10, 3))
        ax_age.hist(sub["AGE"].dropna(), bins=30)
        ax_age.set_title("Age distribution (filtered)")
        ax_age.set_xlabel("Age")
        ax_age.set_ylabel("Count")
        st.pyplot(fig_age)
        plt.close(fig_age)
st.empty()

with c2:
    if "insured_label" in sub.columns and sub["insured_label"].notna().any():
        fig_ins, ax_ins = plt.subplots(figsize=(10, 3))
        sub["insured_label"].value_counts().reindex(["Insured", "Uninsured"]).dropna().plot(kind="bar", ax=ax_ins)
        ax_ins.set_title("Insurance status (filtered)")
        ax_ins.set_xlabel("")
        ax_ins.set_ylabel("Count")
        st.pyplot(fig_ins)
        plt.close(fig_ins)
    else:
        st.info("Insurance not available for this selection.")
st.empty()
st.divider()


# ----------------------------
# Bar chart by region
# ----------------------------
st.subheader("Bar Chart by Region (Filters Applied)")
fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
metrics.set_index("region_name")[metric_col].plot(kind="bar", ax=ax_bar)
ax_bar.set_ylabel(metric_label)
ax_bar.set_xlabel("")
st.pyplot(fig_bar)
plt.close(fig_bar)