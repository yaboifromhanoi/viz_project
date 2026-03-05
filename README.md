[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)
# Analysis of Healthcare Outcomes in Rural vs Urban Areas

This project processes and visualizes data sets related to analysis of the relationship between healthcare outcomes among different regions of the United States and other variables such as having insurance coverage or average poverty levels.

## Setup

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Project Structure

```
data/
states/ 
        tl_2024_us_state.cpg
        tl_2024_us_state.dbf
        tl_2024_us_state.prj
        tl_2024_us_state.shp
        tl_2024_us_state.shx
	tl_2024_us_state.shp.ea.iso
	tl_2024_us_state.shp.iso
.DS_Store
health_outcomes.csv
preprocessing.py

final-project/
	app.py
	health_outcomes.csv
	region_metrics.parquet
	regions.gpq

writeup_html_files/
	libs/
		bootstrap/
				bootstrap.min.css
				bootstrap.min.js
				bootstrap-icons.css
				bootstrap-icons.woff
		clipboard/	
				clipboard.min.js
		quarto-html/ 
				anchor.min.js
				popper.min.js
				quarto.js
				quarto-syntax-highlighting.css
				tippy.css
				tippy.umd.min.js
.DS_Store
.gitignore
educ_attainment_chart.png
environment.yml
uninsured_chart.png
		


```

## Git ignore

Large raw data files are not included in this repository due to GitHub size limits.

Please download the following datasets and place them in:

data/
final-project/

### Required Files

1. health_outcomes.csv  
   Source: IPUMS Health Surveys
   URL: https://healthsurveys.ipums.org/


2. tl_2024_us_state.shp and other relevant files  
   Source: U.S. Census Bureau  
   URL: https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/

After downloading, ensure the folder structure is:

data/
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


## Usage

1. Run preprocessing to raw data:
   ```bash
   python code/preprocessing.py

2 Run streamlit
  streamlit run code/app.py
   ```

