[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)
# Analysis of Healthcare Outcomes in Rural vs Urban Areas

The driving research question behind this project is what are the strongest predictors of regional disparities in hleathcare access in the United States? This project processes health survey data and spatial data and analyzes the relationship, if any, between these insurance coverage and geographic location. Additionally, we include other variables such as age, educational attainment levels, and poverty measures to find their correlation to insurance coverage as a measure of healthcare access. 

## Setup

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Project Structure

```
data/
	raw-data/
		states/ 
			tl_2024_us_state.cpg
			tl_2024_us_state.dbf
			tl_2024_us_state.prj
			tl_2024_us_state.shp
			tl_2024_us_state.shx
			tl_2024_us_state.shp.ea.iso
			tl_2024_us_state.shp.iso
		health_outcomes.csv

	derived-data/
		educ_gdf_clean.geojson
		health_data_clean.csv
		insurance_gdf_clean.geojson
		regions_gdf_clean.geojson 
	preprocessing.py

final-project/
	app.py
	region_metrics.parquet
	regions.gpq

writeup/
	healthratings.png
	uninsured.png
	writeup.pdf
	writeup.qmd

.gitignore
environment.yml
		
```

## Git ignore

Large raw data files are not included in this repository due to GitHub size limits.

Please download the following datasets and place them in:

data/
final-project/

## Required Files

1. health_outcomes.csv  
   Source: IPUMS Health Surveys
   URL: https://healthsurveys.ipums.org/


2. tl_2024_us_state.shp and other relevant files  
   Source: U.S. Census Bureau  
   URL: https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/

After downloading, ensure the folder structure is:

data/
	raw-data/
    	states/
			tl_2024_us_state.cpg
			tl_2024_us_state.dbf
			tl_2024_us_state.prj
			tl_2024_us_state.shp
			tl_2024_us_state.shx
			tl_2024_us_state.shp.ea.iso
			tl_2024_us_state.shp.iso
	health_outcomes.csv
	preprocessing.py

## Health Outcomes Variables

Health survey data was extracted through IPUMS to select custom variables. Each of the variables are defined as follows:  

- YEAR: 4-digit variable reporting the calendar year the survey was conducted and the data was collected 
- SERIAL: IPUMS NHIS-constructed value that uniquely identifies each household in a given year 
- STRATA: IPUMS NHIS-constructed variable representing the impact of the sample design stratification on the estimates of variance and standard errors
- PSU: primary sampling unit representing the impact of the sample design clustering on the estimates of variance and standard errors
- NHISHID: IPUMS NHIS-constructed value that uniquely identifies each household in a given survey year
- REGION: U.S. region where the housing unit of the survey participants was located
- URBRRL: identifies whether the household lives in a large central metro county, a large fringe metro county, a medium and small metro county, or a nonmetropolitan county
- PERNUM: IPUMS NHIS-constructed variable that numbers all persons within each family or household 
- NHISPID: IPUMS NHIS-constructed variable that uniquely identifies each individual in a given survey year
- HHX: for sample adults and sample children, their household number on the original NHIS data
- SAMPWEIGHT: IPUMS NHIS-constructed variable representing the random selection of a sample person in the household to complete the survey
- ASTATFLG: identifies the record of a sample adult
- CSTATFLG: identifies the record of a sample child 
- SEX: identifies the individual as male or female
- RACENEW: identifies the individual's race/ethnicity 
- CITIZEN: identifies whether the individual is a U.S. citizen
- EDUC: identifies the individual highest level of educational attainment
- POVLEV: ratio of family income to the federal poverty line
- HEALTH: an individual's self-reported health on a 5-point scale
- USUALPL: identifies whether an individual has a usual they go to seek medical care or advice
- TYPPLSICK: identifies the specific type of facility or provider an individual uses when seeking medical care or advice
- DELAYCOST: indicates whether an individual delayed seeking medical care due to financial reasons in the last 12 months
- HINOTCOV: identifies whether an indivdual had general health insurance coverage

Note that some of these variables were automatically included in the dataset by IPUMS. 

## Usage

1. Run preprocessing to raw data:
   ```bash
   python ~/viz_project/data/preprocessing.py
   ```

2 Run streamlit
   ```bash
  cd ~/viz_project/final-project
  streamlit run app.py
   ```
## NOTE
Streamlit apps need to be “woken up”
if they have not been run in the last 24 hours.
