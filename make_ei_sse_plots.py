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
from bo_methods_lib.bo_methods_lib.analyze_data import get_study_data_signac, get_best_data, open_file_helper, analyze_heat_maps
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_method_sse_one_plot
from skimage.transform import resize

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#Set Stuff
date_time_str = None
meth_name_list = [1,2,3,4,5,6]
study_id = "ep"
log_data = False
save_csv = False
get_ei = False 
save_fig = True

#Set criteria dict
criteria_dict = {"cs_name_val" : 1,
                 "param_name_str" : "t1t2",
                 "ep_enum_val":1,
                 "retrain_GP": 25,
                 "num_x_data": 5,
                 "outputscl": None,
                 "bo_iter_tot": 50,
                 "lenscl": None}

#Set plot details
title_fontsize = 24
other_fontsize = 24
xbins = 5
ybins = 7
zbins = 900
cmap = "autumn"
z_choice = "sse_mean"
path_end = z_choice

#Get project
project = signac.get_project()

#Get Best Data from ep experiment
df = pd.DataFrame()
job_list = []
for meth_name_val in meth_name_list:
    criteria_dict["meth_name_val"] = meth_name_val
    df_piece, jobs, name_cs_str, theta_true = get_study_data_signac(criteria_dict, save_csv)
    job_list += [job for job in jobs]
    df = pd.concat([df, df_piece], ignore_index=True)

df_best = get_best_data(df, name_cs_str, theta_true, job_list, date_time_str, save_csv)

#Get only the jobs which are the best
project = signac.get_project()
job_list_best = []
for meth_name_val in meth_name_list:
    #Get best ep data from previous results if possible    
    criteria_dict_ep = criteria_dict.copy()
    criteria_dict_ep["meth_name_val"] = meth_name_val
    criteria_dict_ep["sep_fact"] = 1.0
    meth_name = Method_name_enum(meth_name_val).name
    
    path_name = job_list[0].fn("ep_study_best_all.csv")
    df_ep_best = pd.read_csv(path_name, header = 0)
    best_ep_enum_val = int(df_ep_best["EP Method Val"][(df_ep_best['BO Method'] == meth_name)])
    criteria_dict_ep["ep_enum_val"] = best_ep_enum_val
    
    #Get all jobs with that ep enum val
    jobs_best = project.find_jobs(criteria_dict_ep)
    job_list_best += [job for job in jobs_best]
    
#Set the save path as the job path
if save_fig == True:
    #Save all jobs to it 
    save_paths = [job.fn("") for job in job_list_best]
else:
    save_paths = None
       
# for i in range(len(job_list_best)): 
#     print(job_list_best[i].id)
    
assert len(meth_name_list) == len(job_list_best), "lens not equal. Check Criteria dict"

run_num_list = list(map(int, df_best["Run Number"].to_numpy() + 1))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy() + 1))
meth_name_str_list = list(df_best["BO Method"]) 

x_label = "BO Iterations"
title = None

file_path_list = [job.fn("BO_Results.gz") for job in job_list_best]

data_names = ["Max EI"]
string_for_df_theta = ["Max EI"]
y_label = "Max " + r"$\mathbf{EI(\theta)}$"

plot_method_sse_one_plot(file_path_list, meth_name_str_list, run_num_list, string_for_df_theta, data_names, xbins, ybins, 
                                   title, x_label, y_label, log_data, title_fontsize, other_fontsize, save_paths)

data_names = ["Min SSE"]
string_for_df_theta = ["Min Obj Cum."]
y_label = r"$\mathbf{e(\theta)}$"

plot_method_sse_one_plot(file_path_list, meth_name_str_list, run_num_list, string_for_df_theta, data_names, xbins, ybins, 
                                   title, x_label, y_label, log_data, title_fontsize, other_fontsize, save_paths)