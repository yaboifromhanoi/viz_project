import pandas as pd
import geopandas as gpd
import numpy as np
import os

current_wd = os.getcwd()
print(f'Working directory is now: {current_wd}.')

# Load data
health_data = pd.read_csv('data/raw-data/health_outcomes.csv')
states_gdf = gpd.read_file('data/raw-data/states/tl_2024_us_state.shp')

# Data Cleaning - health data
# Recoding IPUMS health variables 
health_data['URBRRL'] = health_data['URBRRL'].map({
    1: 'Large central metro',
    2: 'Large fringe metro',
    3: 'Medium/small metro',
    4: 'Nonmetropolitan'
})

health_data['SEX'] = health_data['SEX'].map({1: 'Male', 2: 'Female'})

health_data['RACENEW'] = np.select([
        health_data['RACENEW'] == 100,
        health_data['RACENEW'] == 200,
        health_data['RACENEW'] == 300,
        health_data['RACENEW'] == 400,
        health_data['RACENEW'].between(500,600)],
    ['White','Black','American Indian/Alaska Native','Asian','Other'],
    default=None
)

health_data['CITIZEN'] = health_data['CITIZEN'].map({1: 0, 2: 1}).astype('Int64')

health_data['EDUC_LVL'] = pd.cut(
    health_data['EDUC'],
    bins=[0,111,116,202,301,303,522],
    labels=[
        'Primary school or less',
        'Less than high school',
        'High school graduate',
        'Some college',
        "Associate's degree",
        "Bachelor's degree or higher"
    ]
)

health_data['EDUC_LVL'] = health_data['EDUC_LVL'].cat.add_categories(['Other'])

health_data.loc[health_data['EDUC'].isin([100,103]), 'EDUC_LVL'] = 'Less than high school'
health_data.loc[health_data['EDUC'] == 530, 'EDUC_LVL'] = 'Other'
health_data.loc[(health_data['EDUC'] > 900) | (health_data['EDUC'] == 0), 'EDUC_LVL'] = np.nan

order = [
    'Primary school or less',
    'Less than high school',
    'High school graduate',
    'Some college',
    "Associate's degree",
    "Bachelor's degree or higher",
    'Other'
]

health_data['EDUC_LVL'] = pd.Categorical(
    health_data['EDUC_LVL'],
    categories=order,
    ordered=True
)

health_data['TYPPLSICK'] = np.select([(
        health_data['TYPPLSICK'] == 0) | (health_data['TYPPLSICK'] > 900),
        health_data['TYPPLSICK'] == 116,
        health_data['TYPPLSICK'].between(100, 113),
        health_data['TYPPLSICK'].between(114, 119),
        health_data['TYPPLSICK'].between(120, 130),
        health_data['TYPPLSICK'].between(200, 320),
        health_data['TYPPLSICK'].between(400, 500)],
    [np.nan, 'Urgent/walk-in clinic', 'Clinic', 'Community/rural clinic', "Doctor's office/HMO", 'Hospital', 'Other'],
    default=None
)

health_data['DELAYCOST'] = health_data['DELAYCOST'].map({1: 0, 2: 1}).astype('Int64')
health_data['HINOTCOV'] = health_data['HINOTCOV'].map({1: 0, 2: 1}).astype('Int64')
health_data['USUALPL'] = health_data['USUALPL'].map({1: 0, 2: 1, 3:1}).astype('Int64')

health_data['HEALTH'] = health_data['HEALTH'].map({
    1: 'Excellent',
    2: 'Very good',
    3: 'Good',
    4: 'Fair',
    5: 'Poor'
})

# Data cleaning - spatial data 
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

# Data aggregation
# Insurance status by region
region_insurance_summary = (
    health_data
    .groupby('REGION')
    .agg(
        num_insured=('HINOTCOV', 'sum'),
        total_pop=('HINOTCOV', 'count')
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

# Education level by region
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

# Save processed data
health_data.to_csv('data/derived-data/health_data_clean.csv', index=False)
regions_gdf.to_file('data/derived-data/regions_gdf_clean.geojson', driver='GeoJSON')
educ_gdf.to_file('data/derived-data/educ_gdf_clean.geojson', driver='GeoJSON')
insurance_gdf.to_file('data/derived-data/insurance_gdf_clean.geojson', driver='GeoJSON')