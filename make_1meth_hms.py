import numpy as np
import pandas as pd
import signac
import os
from itertools import combinations

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import make_plot_dict, get_best_data, get_df_all_jobs
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_hms_gp_compare
from skimage.transform import resize

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

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
best_job_results = file_path_list = [job.fn("BO_Results.gz") for job in job_list_best]

#Set z_choices and levels
z_choices = ["sse_sim", "sse_mean", "sse_var", "ei"]
levels = [100,100,100,100]

run_num_list = list(map(int, df_best["Run Number"].to_numpy()))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy()))
bo_method_list = list(df_best["BO Method"].to_numpy())
best_job_results = [job.fn("BO_Results.gz") for job in job_list_best]

#Loop over best jobs
for i in range(len(best_job_results)):   
    #Set file path
    file_path = best_job_results[i]
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
        plot_dict = make_plot_dict(False, None, None, None, line_levels = levels, save_path=None)
       
        #Set save path
#         save_path = job_list[i].fn("Results/" + make_dir_name_from_criteria(criteria_dict) + "/")

        plot_hms_gp_compare(file_path, run_num_list[i], bo_iter_list[i], pair, z_choices, plot_dict)
        
