import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Health Outcomes Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 US Health Outcomes Dashboard")
st.markdown("Explore regional health outcomes, education levels, and insurance coverage across the United States.")

# ── Paths ─────────────────────────────────────────────────────────────────────
# All files should live in a `data/` subfolder next to app.py
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"

HEALTH_CSV = DATA_DIR / "health_outcomes.csv"
SHP_FILE   = DATA_DIR / "tl_2024_us_state.shp"
EDUC_IMG   = DATA_DIR / "educ_attainment_chart.png"
INSURE_IMG = DATA_DIR / "uninsured_chart.png"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_health():
    df = pd.read_csv(HEALTH_CSV)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_shapefile():
    gdf = gpd.read_file(SHP_FILE)
    # Drop non-contiguous states/territories for a cleaner map
    exclude = {"02", "15", "60", "66", "69", "72", "78"}
    return gdf[~gdf["STATEFP"].isin(exclude)]

try:
    health_df   = load_health()
    gdf         = load_shapefile()
    data_loaded = True
except FileNotFoundError as e:
    st.error(f"**Missing file:** {e}\n\nMake sure all files are in the `data/` folder next to `app.py`.")
    data_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
if data_loaded:
    st.sidebar.header("⚙️ Controls")

    numeric_cols = health_df.select_dtypes(include="number").columns.tolist()

    # Auto-detect state name column for map join
    possible_keys = [c for c in health_df.columns
                     if c.lower() in ("state", "state_name", "name", "region", "stname")]
    join_col = possible_keys[0] if possible_keys else health_df.columns[0]

    selected_metric = st.sidebar.selectbox("Metric to map", numeric_cols)
    colormap = st.sidebar.selectbox(
        "Color scheme",
        ["plasma", "viridis", "magma", "YlOrRd", "Blues", "RdYlGn"],
    )
    show_table = st.sidebar.checkbox("Show raw data table", value=False)

    # ── Merge shapefile + CSV ─────────────────────────────────────────────────
    merged = gdf.merge(health_df, left_on="NAME", right_on=join_col, how="left")

    # ── Choropleth map + summary stats ────────────────────────────────────────
    map_col, stat_col = st.columns([3, 1])

    with map_col:
        st.subheader(f"Choropleth Map – {selected_metric}")
        fig, ax = plt.subplots(figsize=(12, 7))
        merged.plot(
            column=selected_metric,
            ax=ax,
            legend=True,
            cmap=colormap,
            missing_kwds={"color": "lightgrey", "label": "No data"},
            legend_kwds={"shrink": 0.6, "label": selected_metric},
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(f"{selected_metric} by State", fontsize=15, pad=12)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with stat_col:
        st.subheader("📊 Summary Stats")
        s = health_df[selected_metric].dropna()
        st.metric("Mean",    f"{s.mean():.2f}")
        st.metric("Median",  f"{s.median():.2f}")
        st.metric("Min",     f"{s.min():.2f}")
        st.metric("Max",     f"{s.max():.2f}")
        st.metric("Std Dev", f"{s.std():.2f}")

        st.markdown("**Top 5 States**")
        top5 = (
            health_df[[join_col, selected_metric]]
            .dropna()
            .nlargest(5, selected_metric)
            .reset_index(drop=True)
        )
        top5.index += 1
        st.dataframe(top5, use_container_width=True)

    # ── Histogram ─────────────────────────────────────────────────────────────
    st.subheader(f"Distribution – {selected_metric}")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    health_df[selected_metric].dropna().hist(bins=20, ax=ax2, color="#5A4FCF", edgecolor="white")
    ax2.set_xlabel(selected_metric)
    ax2.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Static regional charts ────────────────────────────────────────────────
    st.divider()
    st.subheader("📌 Pre-computed Regional Charts")
    img1, img2 = st.columns(2)

    if EDUC_IMG.exists():
        img1.image(str(EDUC_IMG), caption="Average Education Level by Region", use_container_width=True)
    else:
        img1.info("Add `educ_attainment_chart.png` to `data/` to display this chart.")

    if INSURE_IMG.exists():
        img2.image(str(INSURE_IMG), caption="Percentage of Uninsured Individuals by Region", use_container_width=True)
    else:
        img2.info("Add `uninsured_chart.png` to `data/` to display this chart.")

    # ── Raw data table (optional) ─────────────────────────────────────────────
    if show_table:
        st.divider()
        st.subheader("📋 Raw Data")
        st.dataframe(health_df, use_container_width=True)




