import pandas as pd
import geopandas as gpd
import numpy as np
import os

import matplotlib.pyplot as plt

current_wd = os.getcwd()
print(f'Working directory is now: {current_wd}.')

# Load data
health_data = pd.read_csv('C:/Users/bdhuy/final-project-tiffany-tu-ben-huynh/data/health_outcomes.csv')
states_gdf = gpd.read_file('C:/Users/bdhuy/final-project-tiffany-tu-ben-huynh/data/states/tl_2024_us_state.shp')

# Filter out non-continental states and territories
all_states_gdf = states_gdf[~states_gdf['STUSPS'].isin(['AS', 'GU', 'MP', 'PR', 'VI'])]
states_gdf = states_gdf[~states_gdf['STUSPS'].isin(['AS', 'GU', 'MP', 'PR', 'VI', 'AK', 'HI'])]

# Define regions; map states to regions for merging
northeast = ['CT','ME','MA','NH','RI','VT','NJ','NY','PA']
midwest = ['IL','IN','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD']
south = ['DE','FL','GA','MD','NC','SC','VA','DC','WV','AL','KY','MS','TN','AR','LA','OK','TX']
west = ['AZ','CO','ID','MT','NV','NM','UT','WY','AK','CA','HI','OR','WA']

def assign_region(state):
    if state in northeast:
        return 1
    elif state in midwest:
        return 2
    elif state in south:
        return 3
    elif state in west:
        return 4

states_gdf['REGION'] = states_gdf['STUSPS'].apply(assign_region)

# Dissolve to regions
regions_gdf = states_gdf.dissolve(by='REGION').reset_index()

# Recoding health insurance status
health_data['INSURANCE'] = np.where(
    health_data['HINOTCOV'] == 1, 0,
    np.where(health_data['HINOTCOV'] == 2, 1, np.nan)
)

# Aggregate data by insurance status by region
region_insurance_summary = (
    health_data
    .groupby('REGION')
    .agg(
        num_insured=('INSURANCE', 'sum'),
        total_pop=('INSURANCE', 'count')
    )
    .reset_index()
)

# Calculate uninsured and percentage uninsured
region_insurance_summary['num_uninsured'] = (
    region_insurance_summary['total_pop'] -
    region_insurance_summary['num_insured']
)

region_insurance_summary['pct_uninsured'] = (
    region_insurance_summary['num_uninsured'] /
    region_insurance_summary['total_pop']
)

# Aggregate data by education level by region
health_data['HS_OR_LESS'] = np.where(
    (health_data['EDUC'] == 0) | (health_data['EDUC'] > 900), np.nan,
    np.where(health_data['EDUC'] < 300, 1, 0)
)

region_educ_summary = (
    health_data
    .groupby('REGION')
    .agg(avg_educ=('EDUC', 'mean'),
         num_hs_or_less=('HS_OR_LESS', 'sum'))
    .reset_index()
)

# Merge with geospatial data
educ_gdf = regions_gdf.merge(region_educ_summary, on='REGION', how='left')
insurance_gdf = regions_gdf.merge(region_insurance_summary, on='REGION', how='left')

# Plot number of insured individuals
fig, ax = plt.subplots(figsize=(12, 8))

insurance_gdf.plot(
    column="pct_uninsured",
    legend=True,
    cmap="viridis",
    ax=ax
)

insurance_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

ax.set_title("Percentage of Uninsured Individuals by Region", fontsize=18)
ax.set_axis_off()

plt.show()

# Plot average educational attainment
fig, ax = plt.subplots(figsize=(12, 8))

educ_gdf.plot(
    column="avg_educ",
    legend=True,
    cmap="plasma",
    ax=ax
)

educ_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

ax.set_title("Average Education Level by Region", fontsize=18)
ax.set_axis_off()

plt.show()


