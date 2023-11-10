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
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_heat_maps
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
meth_name_str_list = [1,2,3,4,5]
study_id = "ep"
log_data = False
save_csv = False
get_ei = False 
save_fig = True

#Set criteria dict
criteria_dict = {"cs_name_val" : 9,
                 "param_name_str" : "t1t2t3t4",
                 "retrain_GP": 25,
                 "num_x_data": 5,
                 "outputscl": 25,
                 "num_val_pts": 0,
                 "sep_fact": 1.0,
                 "ep_enum_val": 1,
                 "lenscl": None}

# criteria_dict = {"cs_name_val" : 1,
#                  "param_name_str" : "t1t2",
#                  "retrain_GP": 5,
#                  "num_x_data": 5,
#                  "outputscl": 1,
#                  "num_val_pts": 20,
#                  "sep_fact": 1.0,
#                  "ep_enum_val": 1,
#                  "lenscl": None}

#Set plot details
title_fontsize = 24
other_fontsize = 20
xbins = 4
ybins = 5
zbins = 900
cmap = "autumn"

#Get project
project = signac.get_project()

#Get Best Data from ep experiment
df = pd.DataFrame()
job_list = []
for meth_name_val in meth_name_str_list:
    criteria_dict["meth_name_val"] = meth_name_val
    df_piece, jobs, name_cs_str, theta_true = get_study_data_signac(criteria_dict, study_id, save_csv)
    job_list += [job for job in jobs]
    df = pd.concat([df, df_piece], ignore_index=True)

df_best = get_best_data(df, study_id, name_cs_str, theta_true, job_list, date_time_str, True)

#Get only the jobs which are the best
job_list_best = []
for meth_name_val in meth_name_str_list:
    #Get best ep data from previous results if possible    
    criteria_dict_ep = criteria_dict.copy()
    criteria_dict_ep["meth_name_val"] = meth_name_val
    criteria_dict_ep["sep_fact"] = 1.0
    meth_name = Method_name_enum(meth_name_val).name
    
    path_name = job_list[0].fn(study_id + "_study_best_all.csv")
    df_ep_best = pd.read_csv(path_name, header = 0)
    best_ep_enum_val = int(df_ep_best["EP Method Val"][(df_ep_best['BO Method'] == meth_name)])
    criteria_dict_ep["ep_enum_val"] = best_ep_enum_val
    
    #Get all jobs with that ep enum val
    jobs_best = project.find_jobs(criteria_dict_ep)
    job_list_best += [job for job in jobs_best]

assert len(meth_name_str_list) == len(job_list_best), "lens not equal. Check Criteria dict"

#Get Best Data from ep experiment
df_best_path = job_list_best[0].fn("ep_study_best_all.csv")
df_best = pd.read_csv(df_best_path, header = 0, index_col = 0)

run_num_list = list(map(int, df_best["Run Number"].to_numpy() + 1))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy() + 1))
meth_names = list(df_best["BO Method"])

#Make heat maps
#Loop over best run/iter for each method
for i in range(len(job_list_best)):    
    run_num = run_num_list[i]
    bo_iter = bo_iter_list[i]
    file_path = job_list_best[i].fn("BO_Results.gz")
    string_val = df_best["Theta Min Obj"].iloc[0]
    numbers = [float(num) for num in string_val.replace('[', '').replace(']', '').split()]
    dim_theta = np.array(numbers).reshape(-1, 1)
    dim_theta = len(dim_theta)
    dim_list = np.linspace(0, dim_theta-1, dim_theta)
    method_name = Method_name_enum(meth_name_str_list[i]).name

    #Get Number of pairs
    pairs = len((list(combinations(dim_list, 2))))
    
    #Set the save path as the job path
    if save_fig == True:
        save_path = job_list_best[i].fn("")
    else:
        save_path = None

    #For each pair
    for pair in range(pairs):
        analysis_list = analyze_heat_maps(file_path, run_num, bo_iter, pair, log_data, get_ei)
        sim_sse_var_ei, test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, idcs_to_plot = analysis_list
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        title = "Heat Map Pair " + "-".join(map(str, plot_axis_names))
        title = None
    #     z = [sse_sim, sse_mean, sse_var, ei]
    #     z_titles = ["ln(sse_sim)", "ln(sse)", "ln(sse_var)", "log(ei)"]
    #     levels = [100,100,100,100]
        z = [sse_sim, sse_mean, sse_var]
        z_titles = ["ln("+ r"$\mathbf{e(\theta)_{sim}}$" + ")", 
                    "ln("+ r"$\mathbf{e(\theta)_{gp}}$" + ")", 
                    "ln("+ r"$\mathbf{\sigma^2_{gp}}$" + ")"]
        z_save_names = ["sse_sim", "sse_gp_mean", "sse_var"]
        path_end = '-'.join(z_save_names) 
        levels = [100,100,100]

        plot_heat_maps(test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, levels, idcs_to_plot, 
                    z, z_titles, xbins, ybins, zbins, title, title_fontsize, other_fontsize, cmap, save_path, z_save_names)
        
    #Create mp4/gif files from pngs
    #Initialize filename list
    filenames = []
    
    #Add all Heat map data files to list
    for job in [job_list_best[i]]:
        #Create directory to store Heat Map Movies
        dir_name = job.fn("")
        heat_map_files = glob.glob(job.fn("Heat_Maps/*/*.png"))
        filenames += heat_map_files
  
    if save_fig is True:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        gif_path = dir_name + path_end + ".mp4"

        #Create .mp4 file
        with imageio.get_writer(gif_path, mode='I', fps=0.3) as writer: #Note. For gif use duration instead of fps
            #For each file
            for filename in filenames: 
                #Get image
                image = imageio.imread(filename, pilmode = "RGBA")
                #Get the correct shape for the pngs based on the 1st file
                if filename == filenames[0]: 
                    shape = image.shape
                    #Force image to have XY dims divisible by 16
                    new_shape = (np.ceil(shape[0] / 16) * 16, np.ceil(shape[1] / 16) * 16, shape[2])
                #If item shapes not the same force them to be the same. Fixes issues where pixels are off
                if image.shape is not shape: 
                    image = resize(image, (new_shape))
                #Add file to movie as a uint8 type and multiply array by 255 to get correct coloring
                writer.append_data((image*255).astype(np.uint8)) 