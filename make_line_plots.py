import numpy as np
import pandas as pd
import signac
import os
from itertools import combinations
import signac

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import Plotters

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#Set Stuff
meth_name_val_list = [1, 2, 3, 4, 5, 6, 7]
save_csv = True #Set to False if you don't want to save/resave csvs
save_figs = True
modes = ["act", "gp", "acq"]
project = signac.get_project("GPBO_Fix")

for val in [1,11,12,13,10]:
    criteria_dict = {"cs_name_val" : val,
                        "ep_enum_val": 1,
                        "gp_package":"gpflow",
                        "meth_name_val": {"$in": meth_name_val_list}}
    for mode in modes:
        analyzer = General_Analysis(criteria_dict, project, mode, save_csv)
        plotters = Plotters(analyzer, save_figs)

        ###Get all data from experiments
        df_all_jobs, job_list, theta_true_data = analyzer.get_df_all_jobs(criteria_dict, save_csv)
        ### Get Best Data from ep experiment
        df_best, job_list_best = analyzer.get_best_data()

        #Loop over z_choices to make comparison line plots
        z_choices = ["sse", "min_sse", "acq"]
        titles = ["Min SSE Parameter Values", "Min SSE Parameter Values Overall", "Optimal Acq Func Parameter Values"]

        #Make Parity Plots
        plotters.make_parity_plots()

        #Get best plots for all objectives with all 6 methods on each subplot
        plotters.plot_objs_all_methods(z_choices)

        #Get plot with each method on a different subplot for each obj
        for i in range(len(z_choices)):
            plotters.plot_one_obj_all_methods(z_choices[i])    

        # for i in range(len(job_list_best)):   
            #Plot hyperparameters
            # plotters.plot_hypers(job_list_best[i])
            
            #Plot param values at min_sse, the best theta_values of min_sse overall, and param values at max ei
            # for j in range(len(z_choices)):
            #     plotters.plot_thetas(job_list_best[i], z_choices[j], title = titles[j])
