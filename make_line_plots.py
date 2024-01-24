import numpy as np
import pandas as pd
import signac
import os
from itertools import combinations
import signac

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import get_best_data, make_plot_dict, get_df_all_jobs, make_dir_name_from_criteria, analyze_hypers, analyze_thetas 
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_objs_all_methods, plot_one_obj_all_methods, plot_2D_Data_w_BO_Iter

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#Set Stuff
meth_name_val_list = [1, 2, 3, 4, 5, 6]
save_csv = False
save_fig = False

criteria_dict = {"cs_name_val" : 1,
                 "ep_enum_val": 1,
                 "meth_name_val": {"$in": meth_name_val_list}}

df, job_list, theta_true = get_df_all_jobs(criteria_dict, save_csv)

### Get Best Data from ep experiment
df_best, job_list_best = get_best_data(criteria_dict, df, job_list, theta_true, save_csv)

run_num_list = list(map(int, df_best["Run Number"].to_numpy()))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy()))
bo_method_list = list(df_best["BO Method"].to_numpy())

#Set the save path as the job path
if save_fig == True:
    #Save all figures to study directory
    save_paths = os.makedirs(os.path.join("Results",make_dir_name_from_criteria(criteria_dict)), exist_ok = True)
else:
    save_paths = None
    
#Set results file list
file_path_list = [job.fn("BO_Results.gz") for job in job_list]
run_num_list = list(map(int, df_best["Run Number"].to_numpy()))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy()))

#Loop over z_choices to make comparison line plots
z_choices = ["sse", "sse_sim", "ei"]

#Get best plots for all objectives with all 6 methods on each subplot
x_label = "BO Iterations"
plot_dict = make_plot_dict(False, None, x_label, None, line_levels = None, save_path=None)
plot_objs_all_methods(file_path_list, run_num_list, z_choices, plot_dict)


#Get plot with each method on a different subplot for each obj
y_labels = [r"$\mathbf{e(\theta)}$", "Min " + r"$\mathbf{e(\theta)}$", "Max " + r"$\mathbf{EI(\theta)}$"]
for i in range(len(z_choices)):
    #Set z_choice
    z_choice = z_choices[i]
    #Make plot_dict
    plot_dict = make_plot_dict(False, None, x_label, y_labels[i], line_levels = None, save_path=None)
    #Plot all methods sse, min_sse, and ei
    plot_one_obj_all_methods(file_path_list, run_num_list, z_choice, plot_dict)    

for i in range(len(job_list_best)):
    #Get file path for a best job
    file_path = job_list_best[i].fn("BO_Results.gz")
    
    #Plot hyperparameters
    title = "Hyperparameter Analysis"
    y_label = "Values"
    plot_dict = make_plot_dict(False, title, x_label, y_label, ybins = 7, line_levels = None, save_path=None)
    hps, hp_names, hp_true = analyze_hypers(file_path, save_csv = False)
    plot_2D_Data_w_BO_Iter(hps, hp_names, hp_true, plot_dict)
    
    #Plot param values at min_sse, the best theta_values of min_sse overall, and param values at max ei
    titles = ["Min Obj Parameter Values", "Min Obj Parameter Values Overall", "Max EI Parameter Values"]
    for j in range(len(z_choices)):
        plot_dict = make_plot_dict(False, titles[j], x_label, None, line_levels = None, save_path=None)
        data, data_names, data_true = analyze_thetas(file_path, z_choices[j])
        plot_2D_Data_w_BO_Iter(data, data_names, data_true, plot_dict)