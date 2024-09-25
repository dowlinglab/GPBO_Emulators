import numpy as np
import pandas as pd
import signac
import os
from itertools import combinations
import signac

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import Plotters

# Ignore warnings
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

# Set Stuff
meth_name_val_list = [1, 2, 3, 4, 5, 6, 7]
save_csv = False  # Set to False if you don't want to save/resave csvs
save_figs = False
modes = ["act", "gp", "acq"]
project = signac.get_project("GPBO_Fix")

for job in project:
    command = f"rclone copy gdrv:GPBO_Fix/workspace/{job.id}/BO_Results.gz GPBO_Fix/workspace/{job.id}/BO_Results.gz"
    print(command)
    os.system(command)

job_ids = []
for val in [11,14,2,1,12,13,3,10]:
    criteria_dict = {
        "cs_name_val": val,
        "ep_enum_val": 1,
        "gp_package": "gpflow",
        "meth_name_val": {"$in": meth_name_val_list},
    }
    for mode in modes:
        analyzer = General_Analysis(criteria_dict, project, mode, save_csv)

        ###Get all data from experiments
        df_all_jobs, job_list, theta_true_data = analyzer.get_df_all_jobs(
            criteria_dict, save_csv
        )
        ### Get Best Data from ep experiment
        df_best, job_list_best = analyzer.get_best_data()

        for job in job_list_best:
            job_ids.append(job.id)

#Remove duplicates in list
job_ids = list(set(job_ids))
print(job_ids)
for job_id in job_ids:
    command1 = f"rclone copy GPBO_Fix/workspace/{job_id}/BO_Results_GPs.gz gdrv:GPBO_Fix/workspace/{job_id}/BO_Results_GPs.gz"
    print(command1)
    os.system(command1)