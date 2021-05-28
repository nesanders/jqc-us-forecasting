"""Editable configuration parameters for MPS_severity_forecasting.py
"""
from typing import List, Callable, Optional, Dict

## Maximum year of dataset; first year of forecasting will be subsequent year
max_year: int = 2018
## Maximum year for forecasting simulation
max_year_sim: int = 2040

## Number of stan iterations and chains to run
Niter: int = 2000
Nchains: int = 8

## Minimum sizes of event to model
y_mins: List[int] = [4, 10]

## Case file to lookup source data in; csv format
case_file: str = '2020-10-26_mass_public_shooting_data.csv'

## Variable in datafile with fatalities values; must match to columns of case_file
fat_vars: List[str] = ['Killed Gunfire', 'Total Shot']

## Severity thresholds to use for analysis for each variable
fat_thresholds: Dict[str, List[int]] = {
    'Total Shot': [100, 250, 500, 1000], 
    'Killed Gunfire': [49, 60, 75, 100],
    }

## Number of parallel jobs to run for parallelized calculation 
## and plotting functions
njobs: int=6

## File locations
## Input directory path
idir: str = './'

## Output locations for plots, tables, and data
odir: Dict[str, str] = {
    'plots': "plots/",
    'tables': "tables/",
    'data': "data/",
    }
