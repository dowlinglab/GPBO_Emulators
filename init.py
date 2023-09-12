import signac
import numpy as np
project = signac.init_project()

#Set Method and Case Study
cs_name_val  = 1
inc_to_consider_lo = 0
inc_to_consider_hi = 1
meth_name_val = 4

#Set Initial Parameters
ep0 = 1
ep_enum_val = [1,2,3,4]
sep_fact_list = np.linspace(0.5,1,6) #For CS1 use 0.5 to 1, for CS2 use 0.5, 0.75, and 1
normalize = False
gen_heat_map_data = True
noise_mean = 0
noise_std = 0.01
kernel_enum_val = 1
lenscl = None
outputscl = 1
retrain_GP = 5
reoptimize_obj = 5
bo_iter_tot = 100
bo_run_tot = 15
save_data = True
seed = 1
ei_tol = 1e-6
obj_tol = 1e-4
num_x_data = 5
gen_meth_theta = 1 
gen_meth_x = 2
gen_meth_theta_val = 2

#Note: Add loop for idc to consider for lo and hi for CS2 based on something
for ep_enum_val in ep_val_list:
    for sep_fact in sep_fact_list:
        sp = {"cs_name_val": cs_name_val, 
              "meth_name_val": meth_name_val, 
              "inc_to_consider_lo" : 0, 
              "inc_to_consider_hi" : 2, 
              "ep0": ep0, 
              "ep_enum_val": ep_enum_val,
              "sep_fact" : sep_fact,
              "normalize" : normalize
              "gen_heat_map_data" : gen_heat_map_data,
              "noise_mean":noise_mean
              "noise_std": noise_std,
              "kernel_enum_val": kernel_enum_val,
              "lenscl": lenscl,
              "outputscl": outputscl,
              "retrain_GP": retrain_GP,
              "reoptimize_obj": reoptimize_obj,
              "bo_iter_tot": bo_run_tot,
              "bo_run_tot":bo_run_tot,
              "save_data": save_data,
              "seed":seed,
              "ei_tol":ei_tol,
              "obj_tol":obj_tol,
              "num_x_data":num_x_data,
              "gen_meth_theta":gen_meth_theta,
              "gen_meth_x":gen_meth_x,
              "gen_meth_theta_val":gen_meth_theta_val}

        job = project.open_job(sp).init()