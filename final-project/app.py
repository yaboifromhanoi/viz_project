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
st.caption("Interactive filters on NHIS microdata; outputs aggregated to Census regions.")

BASE_DIR = Path(__file__).resolve().parent
HEALTH_CSV = BASE_DIR / "health_outcomes.csv"
REGIONS_GPQ = BASE_DIR / "regions.gpq"  # made by preprocessing.py

# ----------------------------
# Label dictionaries / bins
# ----------------------------
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
SEX_LABELS = {1: "Male", 2: "Female"}
HEALTH_LABELS = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
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
# Urban/Rural (URBRRL) — exact codes can vary, so now we label common values.
URBRRL_LABELS = {
    1: "Large central metro",
    2: "Large fringe metro",
    3: "Medium metro",
    4: "Small metro",
    5: "Micropolitan",
    6: "Noncore / rural",
}

# Citizen (CITIZEN) — codes vary; we'll label conservatively.
CITIZEN_LABELS = {
    1: "Citizen",
    2: "Non-citizen",
}

# Race (RACENEW) — codes vary by extract; label as "Code X" unless you provide the value labels.
# If you later paste the RACENEW code table, we can replace this with exact categories.
def racenew_label(x):
    if pd.isna(x):
        return np.nan
    try:
        return f"Code {int(x)}"
    except Exception:
        return np.nan

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
    Dashboard-friendly education bins using your earlier conventions.
    NOTE: Exact EDUC meanings vary; this is a clean, standard binning.
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

def label_yes_no_01(x):
    if pd.isna(x):
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
    for col in ["REGION", "SAMPWEIGHT", "AGE", "POVLEV", "HEALTH", "EDUC", "SEX", "HINOTCOV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SAMPWEIGHT" not in df.columns:
        df["SAMPWEIGHT"] = 1.0

    # Derived insured (matches your existing logic: HINOTCOV 2 => insured, 1 => uninsured)
    if "HINOTCOV" in df.columns:
        df["insured"] = np.where(df["HINOTCOV"] == 2, 1,
                          np.where(df["HINOTCOV"] == 1, 0, np.nan))
    else:
        df["insured"] = np.nan

    # Derived HS-or-less indicator (matches your earlier rule)
    if "EDUC" in df.columns:
        df["hs_or_less"] = np.where(
            (df["EDUC"] == 0) | (df["EDUC"] > 900),
            np.nan,
            np.where(df["EDUC"] < 300, 1, 0),
        )
    else:
        df["hs_or_less"] = np.nan

    # Add labeled columns for UI
    df["REGION_label"] = df["REGION"].map(REGION_LABELS)
    if "SEX" in df.columns:
        df["SEX_label"] = df["SEX"].map(SEX_LABELS)
    if "HEALTH" in df.columns:
        df["HEALTH_label"] = df["HEALTH"].map(HEALTH_LABELS)
    if "POVLEV" in df.columns:
        df["POVLEV_bin"] = df["POVLEV"].apply(povlev_bin)
    if "EDUC" in df.columns:
        df["EDUC_group"] = df["EDUC"].apply(educ_group_from_code)

    df["insured_label"] = df["insured"].apply(lambda x: "Insured" if x == 1 else ("Uninsured" if x == 0 else np.nan))
    df["hs_or_less_label"] = df["hs_or_less"].apply(label_yes_no_01)
        
    if "URBRRL" in df.columns:
        df["URBRRL_label"] = df["URBRRL"].map(URBRRL_LABELS)

    if "CITIZEN" in df.columns:
        df["CITIZEN_label"] = df["CITIZEN"].map(CITIZEN_LABELS)

    if "RACENEW" in df.columns:
        df["RACENEW_label"] = df["RACENEW"].map(RACENEW_LABELS)

    # Common 0/1 items in your extract (turn into Yes/No)
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
    # group by numeric REGION for stable merges
    base = df.dropna(subset=["REGION"]).copy()
    base["REGION"] = base["REGION"].astype(int)

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["SAMPWEIGHT"]
        insured_rate = weighted_mean(g["insured"], w)
        return pd.Series({
            "n_obs": len(g),
            "weighted_pop": float(w.sum()),
            "pct_uninsured": (1 - insured_rate) if pd.notna(insured_rate) else np.nan,
            "avg_educ": weighted_mean(g["EDUC"], w),
            "pct_hs_or_less": weighted_mean(g["hs_or_less"], w),
            "avg_povlev": weighted_mean(g["POVLEV"], w),
            "avg_health": weighted_mean(g["HEALTH"], w),
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

with st.expander("Data notes"):
    st.markdown(
        """
**What you’re looking at**
- This dashboard uses **NHIS microdata** (individual records) and applies your selected filters.
- After filtering, the app computes **weighted regional summaries** (Census regions) using `SAMPWEIGHT`.

**Weights**
- We use `SAMPWEIGHT` as person-level weights to estimate population-representative means and shares.

**How key metrics are constructed**
- **Percent uninsured**: computed from a derived `insured` indicator created from `HINOTCOV`:
  - `insured = 1` if `HINOTCOV == 2`
  - `insured = 0` if `HINOTCOV == 1`
  - otherwise missing  
  *(If you want this to match the official IPUMS value labels exactly, confirm the `HINOTCOV` code meanings in your codebook.)*

- **HS or less**: derived from `EDUC` using a simple binning rule:
  - missing if `EDUC == 0` or `EDUC >= 900`
  - `HS or less = 1` if `EDUC < 300`, else `0`

- **Poverty bins**: derived from `POVLEV` (percent of poverty line):
  - `<100%` Below poverty
  - `100–137%` Near-poor
  - `138–199%`, `200–399%`, `400%+`

- **Race**: displayed using NHIS RACENEW (post-1997 OMB standards) categories.

- **Self-rated health** (`HEALTH`): labeled as Excellent → Poor for codes 1–5.

**Coded variables**
- Some variables (e.g., `RACENEW`) are displayed as **Code X** unless their value-label table is provided.
  If you paste the `RACENEW` code table from your IPUMS codebook, I can replace those with exact race labels.
        """
    )
# ----------------------------
# Sidebar filters (labeled)
# ----------------------------
st.sidebar.header("Filters")

# Region
region_opts = [r for r in micro["REGION_label"].dropna().unique().tolist()]
region_opts = [r for r in ["Northeast", "Midwest", "South", "West"] if r in region_opts]  # stable order
sel_regions = st.sidebar.multiselect("Region", region_opts, default=region_opts)
df = micro[micro["REGION_label"].isin(sel_regions)].copy()

# Race (RACENEW, labeled)
if "RACENEW_label" in df.columns:
    race_opts = [r for r in df["RACENEW_label"].dropna().unique().tolist()]
    # stable, readable order
    preferred_order = [
        "White only",
        "Black/African American only",
        "American Indian/Alaska Native only",
        "Asian only",
        "Other Race",
        "Multiple Race",
        "Other Race and Multiple Race",
        "American Indian/Alaska Native and Any Other Race",
        "Race Group Not Releasable",
        "Unknown/Refused",
        "Unknown/Not ascertained",
        "Unknown/Don't know",
    ]
    race_opts = [r for r in preferred_order if r in race_opts] + [r for r in race_opts if r not in preferred_order]
    if race_opts:
        sel_race = st.sidebar.multiselect("Race", race_opts, default=race_opts)
        df = df[df["RACENEW_label"].isin(sel_race)]

# Age slider (numeric)
if "AGE" in df.columns and df["AGE"].notna().any():
    a_min = int(df["AGE"].dropna().min())
    a_max = int(df["AGE"].dropna().max())
    age_rng = st.sidebar.slider("Age", a_min, a_max, (a_min, a_max))
    df = df[df["AGE"].between(age_rng[0], age_rng[1], inclusive="both")]

# Sex (labeled)
if "SEX_label" in df.columns:
    sex_opts = sorted(df["SEX_label"].dropna().unique().tolist())
    if sex_opts:
        sel_sex = st.sidebar.multiselect("Sex", sex_opts, default=sex_opts)
        df = df[df["SEX_label"].isin(sel_sex)]

# Poverty bins (labeled)
if "POVLEV_bin" in df.columns:
    pov_opts = [p for p in df["POVLEV_bin"].dropna().unique().tolist()]
    # stable order
    pov_order = ["<100% (Below poverty)", "100–137% (Near-poor)", "138–199%", "200–399%", "400%+"]
    pov_opts = [p for p in pov_order if p in pov_opts]
    if pov_opts:
        sel_pov = st.sidebar.multiselect("Poverty level", pov_opts, default=pov_opts)
        df = df[df["POVLEV_bin"].isin(sel_pov)]

# Education group (labeled)
if "EDUC_group" in df.columns:
    educ_opts = [e for e in df["EDUC_group"].dropna().unique().tolist()]
    educ_order = ["HS or less", "Some college", "Bachelor's+", "Missing/Unknown"]
    educ_opts = [e for e in educ_order if e in educ_opts]
    if educ_opts:
        sel_educ = st.sidebar.multiselect("Education", educ_opts, default=educ_opts)
        df = df[df["EDUC_group"].isin(sel_educ)]

# Insurance (derived, labeled)
ins_opts = [x for x in df["insured_label"].dropna().unique().tolist()]
ins_order = ["Insured", "Uninsured"]
ins_opts = [x for x in ins_order if x in ins_opts]
if ins_opts:
    sel_ins = st.sidebar.multiselect("Insurance", ins_opts, default=ins_opts)
    df = df[df["insured_label"].isin(sel_ins)]

# Health status (labeled)
if "HEALTH_label" in df.columns:
    health_opts = [h for h in df["HEALTH_label"].dropna().unique().tolist()]
    health_order = ["Excellent", "Very good", "Good", "Fair", "Poor"]
    health_opts = [h for h in health_order if h in health_opts]
    if health_opts:
        sel_health = st.sidebar.multiselect("Self-rated health", health_opts, default=health_opts)
        df = df[df["HEALTH_label"].isin(sel_health)]

if "URBRRL_label" in df.columns:
    ur_opts = sorted(df["URBRRL_label"].dropna().unique().tolist())
    if ur_opts:
        sel_ur = st.sidebar.multiselect("Urban/Rural", ur_opts, default=ur_opts)
        df = df[df["URBRRL_label"].isin(sel_ur)]

if "CITIZEN_label" in df.columns:
    cit_opts = sorted(df["CITIZEN_label"].dropna().unique().tolist())
    if cit_opts:
        sel_cit = st.sidebar.multiselect("Citizenship", cit_opts, default=cit_opts)
        df = df[df["CITIZEN_label"].isin(sel_cit)]

if "RACENEW_label" in df.columns:
    race_opts = sorted(df["RACENEW_label"].dropna().unique().tolist())
    if race_opts:
        sel_race = st.sidebar.multiselect("Race (RACENEW)", race_opts, default=race_opts)
        df = df[df["RACENEW_label"].isin(sel_race)]

if "DELAYCOST_label" in df.columns:
    dc_opts = [x for x in ["Yes", "No"] if x in df["DELAYCOST_label"].dropna().unique().tolist()]
    if dc_opts:
        sel_dc = st.sidebar.multiselect("Delayed care due to cost", dc_opts, default=dc_opts)
        df = df[df["DELAYCOST_label"].isin(sel_dc)]

st.sidebar.divider()
st.sidebar.write(f"Rows after filters: **{len(df):,}**")

# ----------------------------
# Compute metrics and merge for map
# ----------------------------
metrics = compute_region_metrics(df)
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
cmap = st.selectbox("Color scheme", ["viridis", "plasma", "magma", "YlOrRd", "Blues"], index=0)

# ----------------------------
# Map + Statistics
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
        st.info("No values, exclude some filters.")
    else:
        st.metric("Mean", f"{s.mean():.3f}")
        st.metric("Min", f"{s.min():.3f}")
        st.metric("Max", f"{s.max():.3f}")

st.divider()

# ----------------------------
# Searchable table + region selection
# ----------------------------
st.subheader("Dynamic Table")

search = st.text_input("Search Table", value="").strip().lower()
table = metrics.copy()

if search:
    mask = table.astype(str).apply(lambda r: search in " ".join(r.values).lower(), axis=1)
    table = table[mask]

st.dataframe(
    table[["REGION", "region_name", "n_obs", "weighted_pop", "pct_uninsured", "avg_povlev", "pct_hs_or_less", "avg_health"]],
    use_container_width=True
)

visible_regions = table["REGION"].dropna().astype(int).tolist()
selected_region = st.selectbox(
    "Select a region for details",
    options=visible_regions if visible_regions else [1, 2, 3, 4],
    index=0
)

st.divider()
st.subheader("Region Detail View")

sub = df[df["REGION"] == selected_region].copy()
st.write(f"**Selected:** {REGION_LABELS.get(int(selected_region), 'Unknown')} — rows: **{len(sub):,}**")

if len(sub) == 0:
    st.info("No rows for this region after filtering.")
else:
    # Simple microdata distributions
    c1, c2 = st.columns(2)

    with c1:
        if "AGE" in sub.columns and sub["AGE"].notna().any():
            fig_age, ax_age = plt.subplots(figsize=(10, 3))
            ax_age.hist(sub["AGE"].dropna(), bins=30)
            ax_age.set_title("Age distribution (filtered)")
            ax_age.set_xlabel("AGE")
            ax_age.set_ylabel("Count")
            st.pyplot(fig_age)
            plt.close(fig_age)

    with c2:
        if "insured_label" in sub.columns and sub["insured_label"].notna().any():
            fig_ins, ax_ins = plt.subplots(figsize=(10, 3))
            sub["insured_label"].value_counts().reindex(["Insured", "Uninsured"]).dropna().plot(kind="bar", ax=ax_ins)
            ax_ins.set_title("Insurance status (filtered)")
            ax_ins.set_xlabel("")
            ax_ins.set_ylabel("Count")
            st.pyplot(fig_ins)
            plt.close(fig_ins)

st.divider()
st.subheader("Bar chart by region (filters applied)")
fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
metrics.set_index("region_name")[metric_col].plot(kind="bar", ax=ax_bar)
ax_bar.set_ylabel(metric_label)
ax_bar.set_xlabel("")
st.pyplot(fig_bar)
plt.close(fig_bar)
