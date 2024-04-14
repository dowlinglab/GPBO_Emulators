import numpy as np
import pandas as pd
import signac
import os
from itertools import combinations

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import Plotters
from skimage.transform import resize

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

meth_name_val_list = [1, 2, 3, 4, 5, 6, 7]
save_csv = True #Set to False if you don't want to save/resave csvs
save_figs = True

project = signac.get_project()

for val in [2]:
    criteria_dict = {"cs_name_val" : val,
                    "ep_enum_val": 1,
                    "gp_package":"gpflow",
                    "meth_name_val": {"$in": meth_name_val_list}}

    analyzer = General_Analysis(criteria_dict, project, save_csv)
    plotters = Plotters(analyzer, save_figs)

    #Get Best Data from ep experiment
    ### Get Best Data from ep experiment
    df_best, job_list_best = analyzer.get_best_data()

    #Set z_choices and levels
    z_choices = ["sse_sim", "sse_mean", "sse_var", "acq"]
    levels = [100,100,100,100]

    #Loop over best jobs
    for i in range(len(job_list_best)):   
        #Get jobs, runs, and iters to examine
        job = job_list_best[i]
        run_num = df_best["Run Number"].iloc[i]
        bo_iter = df_best["BO Iter"].iloc[i]
        
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
            plotters.plot_hms_gp_compare(job, run_num, bo_iter, pair, z_choices, levels)
            
