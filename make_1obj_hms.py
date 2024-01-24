import numpy as np
import pandas as pd
import signac
import os
import imageio
import glob
import itertools
from itertools import combinations

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import CS_name_enum, Method_name_enum
from bo_methods_lib.bo_methods_lib.analyze_data import get_study_data_signac, get_best_data, open_file_helper, analyze_heat_maps, get_df_all_jobs, make_plot_dict 
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_hms_all_methods
from skimage.transform import resize

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#From signac
import signac
meth_name_val_list = [1, 2, 3, 4, 5, 6]
save_csv = False

criteria_dict = {"cs_name_val" : 1,
                 "ep_enum_val": 1,
                 "meth_name_val": {"$in": meth_name_val_list}}

df, job_list, theta_true = get_df_all_jobs(criteria_dict, save_csv)

### Get Best Data from ep experiment
df_best, job_list_best = get_best_data(criteria_dict, df, job_list, theta_true, save_csv)

#Get Best Data from ep experiment
run_num_list = list(map(int, df_best["Run Number"].to_numpy()))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy()))
bo_method_list = list(df_best["BO Method"].to_numpy())
best_job_results = [job.fn("BO_Results.gz") for job in job_list_best]

#Set z_choices and levels
z_choices = ["sse_sim", "sse_mean", "sse_var", "ei"]
levels = [100,100,100, 100, 100, 100]

#Back out number of parameters
string_val = df_best["Theta Min Obj"].iloc[0]
try:
    numbers = [float(num) for num in string_val.replace('[', '').replace(']', '').split()]
except:
    numbers = [float(num) for num in string_val]
#Create list of parameter pair combinations
dim_theta = len(np.array(numbers).reshape(-1, 1))
dim_list = np.linspace(0, dim_theta-1, dim_theta)
pairs = len((list(combinations(dim_list, 2))))

#Loop over parameter pairs
for pair in range(pairs):
    for z_choice in z_choices:
        #Plot heat maps for one objective with each method in a different subplot
        plot_dict = make_plot_dict(False, None, None, None, line_levels = levels, save_path=None)
        plot_hms_all_methods(best_job_results, run_num_list, bo_iter_list, pair, z_choice, plot_dict)
