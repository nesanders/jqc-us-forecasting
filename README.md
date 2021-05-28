# Forecasting the Severity of US Mass Public Shootings

This repo contains model and analysis code for the paper [Forecasting the Severity of Mass Public Shootings in the United States](https://link.springer.com/article/10.1007/s10940-021-09499-5) in the Journal of Quantitative Criminology by Grant Duwe, [Nathan E. Sanders](https://github.com/nesanders), Michael Rocque, and James Alan Fox.  

## Repository Contents

### Environment

* `environment.yml` specifies the conda environment configuration.

### Analysis

* `MPS_severity_forecasting.py` is the primary analysis script.
* `utils.py` defines a variety of functions and constants invoked in the analysis.
* `config.py` defines several constants invoked by utils.py and MPS_severity_forecasting.py, such as input and output data locations.
* `generate_figure_list.sh` is a shell script which assigns the figures and tables output by MPS_severity_forecasting.py to the names by which they appear in the final paper.

### Models

* `*.stan files` specify the Bayesian tail distribution models in the Stan modeling language.
* `hurwitz_zeta.hpp` a small C++ header needed to enable discrete Pareto modeling in Stan.  See [here](https://github.com/nesanders/stan-discrete-pareto) for more information.

### Input Data

* `2020-10-26_mass_public_shooting_data.csv` is the primary input data file containing mass shooting incident records
* `USCensus_np2017-t1.xlsx` is a data file containing population projections.
* `USPop_World_Development_Indicators.xlsx` is a data file containing historical population counts.

### Outputs

* `plots/` are the output figures
* `tables/` are the output statistical tables
