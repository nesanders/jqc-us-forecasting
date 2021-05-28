## This script copies automated output from the MPS_severity_forecast.py model runner script 
## to renamed figures in the paper_figures_tables/table_3.pdf/ directory.  It specifies a
## mapping of which figures and tables appear where in the submitted paper

mkdir paper_figures_tables

cp "tables/wrap_tab_scen_Killed_Gunfire_Cumulative_10.pdf" paper_figures_tables/table_3.pdf

cp "plots/fig_dist_allcompare_ymin10_ccdf.pdf" paper_figures_tables/figure_2.pdf

cp "plots/fig_model_LOO_compare_>10.pdf" paper_figures_tables/figure_3.pdf

cp "plots/fig_rare_event_analytic_lognormal_Killed Gunfire_60_10.pdf" paper_figures_tables/figure_4.pdf

cp "plots/fig_dist_allcompare_Killed Gunfire_ccdf.pdf" paper_figures_tables/figure_5.pdf

cp "plots/fig_dist_allcompare_Total Shot_ccdf.pdf" paper_figures_tables/figure_c1.pdf

cp "plots/fig_log_odds_diff_average.pdf" paper_figures_tables/figure_c2.pdf

cp "plots/fig_model_LOO_compare_total.pdf" paper_figures_tables/figure_c3.pdf

cp "tables/wrap_tab_scen_Killed_Gunfire_SingleYear_4.pdf" paper_figures_tables/table_c1.pdf

cp "tables/wrap_tab_scen_Total_Shot_SingleYear_4.pdf" paper_figures_tables/table_c2.pdf

cp "tables/wrap_tab_scen_Killed_Gunfire_Cumulative_4.pdf" paper_figures_tables/table_c3.pdf

cp "tables/wrap_tab_scen_Total_Shot_Cumulative_4.pdf" paper_figures_tables/table_c4.pdf

cp "tables/wrap_tab_scen_Killed_Gunfire_SingleYear_10.pdf" paper_figures_tables/table_d1.pdf

cp "tables/wrap_tab_scen_Total_Shot_SingleYear_10.pdf" paper_figures_tables/table_d2.pdf

cp "tables/wrap_tab_scen_Total_Shot_Cumulative_10.pdf" paper_figures_tables/table_d3.pdf



## Tex sources
cp "tab_scen_Killed_Gunfire_Cumulative_10.tex" paper_figures_tables/paper_figures_table_3.tex
cp "tab_scen_Killed_Gunfire_SingleYear_4.tex" paper_figures_tables/paper_figures_table_c1.tex
cp "tab_scen_Total_Shot_SingleYear_4.tex" paper_figures_tables/paper_figures_table_c2.tex
cp "tab_scen_Killed_Gunfire_Cumulative_4.tex" paper_figures_tables/paper_figures_table_c3.tex
cp "tab_scen_Total_Shot_Cumulative_4.tex" paper_figures_tables/paper_figures_table_c4.tex
cp "tab_scen_Killed_Gunfire_SingleYear_10.tex" paper_figures_tables/paper_figures_table_d1.tex
cp "tab_scen_Total_Shot_SingleYear_10.tex" paper_figures_tables/paper_figures_table_d2.tex
cp "tab_scen_Total_Shot_Cumulative_10.tex" paper_figures_tables/paper_figures_table_d3.tex

cp "tab_scen_Killed_Gunfire_Cumulative_10.tex.csv" paper_figures_tables/paper_figures_table_3.csv
cp "tab_scen_Killed_Gunfire_SingleYear_4.tex.csv" paper_figures_tables/paper_figures_table_c1.csv
cp "tab_scen_Total_Shot_SingleYear_4.tex.csv" paper_figures_tables/paper_figures_table_c2.csv
cp "tab_scen_Killed_Gunfire_Cumulative_4.tex.csv" paper_figures_tables/paper_figures_table_c3.csv
cp "tab_scen_Total_Shot_Cumulative_4.tex.csv" paper_figures_tables/paper_figures_table_c4.csv
cp "tab_scen_Killed_Gunfire_SingleYear_10.tex.csv" paper_figures_tables/paper_figures_table_d1.csv
cp "tab_scen_Total_Shot_SingleYear_10.tex.csv" paper_figures_tables/paper_figures_table_d2.csv
cp "tab_scen_Total_Shot_Cumulative_10.tex.csv" paper_figures_tables/paper_figures_table_d3.csv
