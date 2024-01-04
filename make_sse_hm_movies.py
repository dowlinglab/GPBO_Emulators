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
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_heat_maps, compare_method_heat_maps
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
log_data = True
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
xbins = 4
ybins = 5
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
    
# for i in range(len(job_list_best)): 
#     print(job_list_best[i].id)
    
assert len(meth_name_list) == len(job_list_best), "lens not equal. Check Criteria dict"

run_num_list = list(map(int, df_best["Run Number"].to_numpy() + 1))
bo_iter_list = list(map(int, df_best["BO Iter"].to_numpy() + 1))
meth_name_str_list = list(df_best["BO Method"]) 
    
#Loop over best run/iter for each method
string_val = df_best["Theta Min Obj"].iloc[0]
numbers = [float(num) for num in string_val.replace('[', '').replace(']', '').split()]
dim_theta = np.array(numbers).reshape(-1, 1)
dim_theta = len(dim_theta)
dim_list = np.linspace(0, dim_theta-1, dim_theta)

#Get Number of pairs
pairs = len((list(combinations(dim_list, 2))))

#Set the save path as the job path
if save_fig == True:
    #Save all jobs to it 
    save_paths = [job.fn("") for job in job_list_best]
else:
    save_paths = None

#For each pair
for pair in range(pairs):
    title = None

    levels = [100,100,100,100,100,100]
    file_path_list = [job.fn("BO_Results.gz") for job in job_list_best]

    compare_method_heat_maps(file_path_list, meth_name_str_list, run_num_list, bo_iter_list, pair, 
                     z_choice, log_data, levels, xbins, ybins, zbins, title, title_fontsize, other_fontsize, 
                     cmap, save_paths)
        
#Create mp4/gif files from pngs
#Initialize filename list
filenames = []
#Add all Heat map data files to list (All will be the same so just take files from 1st job)
#Create directory to store Heat Map Movies
job = job_list_best[0]
heat_map_files = glob.glob(job.fn("Heat_Maps/"+ z_choice + "_all_methods" + "/*.png"))
filenames += heat_map_files

for job in job_list_best:
    if save_fig is True:
        dir_name = job.fn("")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        gif_path = dir_name + z_choice + "_all_methods" + ".mp4"

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