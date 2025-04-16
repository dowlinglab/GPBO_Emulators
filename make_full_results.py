import numpy as np
import pandas as pd
import signac
import os
from itertools import combinations
import signac

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis, All_CS_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import Plotters, All_CS_Plotter

# Ignore warnings
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

# Set Stuff
meth_name_val_list = [1, 2, 3, 4, 5, 6, 7]
cs_list = [11, 17, 2, 3, 15, 14, 12, 13, 10, 1]
save_csv = False  # Set to False if you don't want to save/resave csvs
save_figs = False

df_all = []
best_all = []

for projects in ["GPBO_nonoise", "GPBO_rand"]:
    project = signac.get_project(projects)
    for val in cs_list:
        criteria_dict = {
            "cs_name_val": val,
            "ep_enum_val": 1,
            "gp_package": "gpflow",
            "meth_name_val": {"$in": meth_name_val_list},
        }

        analyzer = General_Analysis(criteria_dict, project, "act", save_csv)
        plotters = Plotters(analyzer, save_figs)

        ###Get all data from experiments
        df_all_jobs, job_list, theta_true_data = analyzer.get_df_all_jobs(
            criteria_dict, save_csv
        )
        df_all_jobs["w_noise"] = job_list[0].sp.w_noise
        df_all.append(df_all_jobs)
        ### Get Best Data from ep experiment
        df_best, job_list_best = analyzer.get_best_data()
        df_best["w_noise"] = job_list[0].sp.w_noise
        best_all.append(df_best)


# Combine all dataframes
df_all_combined = pd.concat(df_all, ignore_index=True)
df_best_combined = pd.concat(best_all, ignore_index=True)

df_all_combined.to_csv("full-results.csv", index=False)
df_best_combined.to_csv("results.csv", index=False)


# Make bar chart overalls
df_bar_best = []
df_bar_med = []
for projects in ["GPBO_nonoise", "GPBO_rand"]:
    project = signac.get_project(projects)
    analyzer = All_CS_Analysis(cs_list, meth_name_val_list, project, "act", save_csv)
    plotters = All_CS_Plotter(analyzer, save_figs)

    # Get % true found
    # Change cs_list here to get averages over select case studies
    analyzer.get_percent_true_found(cs_list)

    # Make Derivative Free Bar Charts
    df_med_derivfree, df_best_derivfree = plotters.make_derivfree_bar(
        s_meths=["NLS", "SHGO-Sob", "NM", "GA"], ver="med"
    )
    df_med_derivfree["w_noise"] = False if projects == "GPBO_nonoise" else True
    df_best_derivfree["w_noise"] = False if projects == "GPBO_nonoise" else True
    df_bar_best.append(df_best_derivfree)
    df_bar_med.append(df_med_derivfree)

df_bar_med_comb = pd.concat(df_bar_med, ignore_index=True)
df_bar_best_comb = pd.concat(df_bar_best, ignore_index=True)

df_bar_med_comb.to_csv("opt-meth-comp-med.csv", index=False)
df_bar_best_comb.to_csv("opt-meth-comp-best.csv", index=False)
