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

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

#Set Stuff
meth_name_str_list = [1, 2, 3, 4, 5]
study_id = "ep"
param_name_str = "y0"
cs_name_val = 2 
cs_name_enum = CS_name_enum(cs_name_val)
log_data = False

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
    jobs = project.find_jobs({"cs_name_val":cs_name_val, "param_name_str":param_name_str, "num_val_pts": 0, "meth_name_val":meth_val}) 
    
    #Get file and job path
    for job in jobs:
        job_path = job.fn("")
        file_path = job.fn("BO_Results.gz")

    #Get method name and best iter and run from results
    loaded_results = open_file_helper(file_path)
    meth_name = Method_name_enum(loaded_results[0].configuration["Method Name Enum Value"]).name
    run_num = df_best.loc[df_best['BO Method'].str.contains(meth_name), 'Run Number'].iloc[0] + 1
    bo_iter = df_best.loc[df_best['BO Method'].str.contains(meth_name), 'BO Iter'].iloc[0] + 1
        
    #Set the save path as the job path
    save_path = job_path
    
    #Get Number of pairs
    dim_list = np.linspace(0, loaded_results[0].simulator_class.dim_theta-1, loaded_results[0].simulator_class.dim_theta)
    pairs = len((list(combinations(dim_list, 2))))
    
    #For each pair
    for pair in range(pairs):
        #Get the heat map data
        analysis_list = analyze_heat_maps(file_path, run_num, bo_iter, pair, log_data)
        sim_sse_var_ei, test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, idcs_to_plot = analysis_list
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei

        #Organize the heat map data
        title = "Heat Map Pair " + "-".join(map(str, plot_axis_names))
        z = [sse_sim, sse_mean, sse_var, ei]
        z_titles = ["sse_sim", "sse", "sse_var", "ei"]
        levels = [100,100,100, 100]

        #Plot and save heat maps in the signac workspace
        plot_heat_maps(test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, levels, idcs_to_plot, 
                       z, z_titles, xbins, ybins, zbins, title, title_fontsize, other_fontsize, cmap, save_path)
        
        
    #Create mp4/gif files from pngs
    #Create directory to store Heat Map Movies
    dir_name = "Results/ep_study/" + cs_name_enum.name + "/" + param_name_str + "/" +  meth_name + "/Heat_Maps/"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    gif_path = dir_name + "param_combos.mp4"

    #Initialize filename list
    filenames = []

    #Add all Heat map data files to list
    for job in jobs:
        heat_map_files = glob.glob(job.fn("Heat_Maps/*/*.png"))
        filenames += heat_map_files

    #Create .mp4 file
    with imageio.get_writer(gif_path, mode='I', fps=0.3) as writer: #Note. For gif use duration instead of fps
        #For each file
        for filename in filenames: 
            #Get image
            image = imageio.imread(filename)
            #Get the correct shape for the pngs based on the 1st file
            if filename == filenames[0]: 
                shape = image.shape
            #If item shapes not the same force them to be the same. Fixes issues where pixels are off by 1
            if image.shape is not shape: 
                image.resize(shape)
            #Add file to movie
            writer.append_data(image) 