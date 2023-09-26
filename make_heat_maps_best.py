import numpy as np
import pandas as pd
import signac
import os

import itertools
from itertools import combinations

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import CS_name_enum, Method_name_enum
from bo_methods_lib.bo_methods_lib.analyze_data import get_study_data_signac, get_best_data, open_file_helper, analyze_heat_maps
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_heat_maps

#Ignore warnings caused by "nan" values
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

#Set Stuff
meth_name_str_list = [1, 2, 3, 5]
study_id = "ep"
param_name_str = "y0"
cs_name_val = 2  

#Get project
project = signac.get_project()

#Get Best Data from ep experiment
df_best_path = "Results/ep_study/" + CS_name_enum(cs_name_val).name + "/" + param_name_str + "/ep_study_best.csv"

if os.path.exists(df_best_path):
    df_best = pd.read_csv(df_best_path, header = 0, index_col = 0)
else:
    df = pd.DataFrame()
    for meth_name_val in meth_name_str_list:
        df_piece, name_cs_str, theta_true = get_study_data_signac(project, cs_name_val, param_name_str, meth_name_val, 
                                                                  study_id, True)
        df = pd.concat([df, df_piece], ignore_index=True)
        df_best = get_best_data(df, study_id, name_cs_str, theta_true, param_name_str, None, True)

#Make heat maps
#Loop over each method
for meth_val in meth_name_str_list:
    
    #Find jobs and necessary files
    jobs = project.find_jobs({"cs_name_val":cs_name_val, "param_name_str":param_name_str, "num_val_pts": 0, 
                              "meth_name_val":meth_val}) 
    
    for job in jobs:
        job_path = job.fn("")
        file_path = job.fn("BO_Results.gz")

    #Get method name and best iter and run from 
    loaded_results = open_file_helper(file_path)
    meth_name = Method_name_enum(loaded_results[0].configuration["Method Name Enum Value"]).name

    run_num = df_best.loc[df_best['BO Method'].str.contains(meth_name), 'Run Number'].iloc[0] + 1
    bo_iter = df_best.loc[df_best['BO Method'].str.contains(meth_name), 'BO Iter'].iloc[0] + 1

    title_fontsize = 24
    other_fontsize = 20
    xbins = 4
    ybins = 5
    zbins = 900
    save_path = job_path
    cmap = "autumn"
    
    print("save path: ", save_path)

    dim_list = np.linspace(0, loaded_results[0].simulator_class.dim_theta-1, loaded_results[0].simulator_class.dim_theta)
    pairs = len((list(combinations(dim_list, 2))))
    
    for pair in range(pairs):
        analysis_list = analyze_heat_maps(file_path, run_num, bo_iter, pair)
        sim_sse_var_ei, test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, idcs_to_plot = analysis_list
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei

        title = "Heat Map Pair " + "-".join(map(str, plot_axis_names))
        z = [sse_sim, sse_mean, sse_var, ei]
        z_titles = ["sse_sim", "sse", "sse_var", "ei"]
        levels = [100,100,100, 100]

        plot_heat_maps(test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, levels, idcs_to_plot, 
                       z, z_titles, xbins, ybins, zbins, title, title_fontsize, other_fontsize, cmap, save_path)