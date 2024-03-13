import signac
import numpy as np
import os
import json
import bo_methods_lib

project = signac.init_project()

#Set Method and Case Study (Start w/ just 1 and 2 for now)
gp_pack = "gpflow"
cs_val_list  = [17]
meth_val_list = [1, 2, 3, 4, 5, 6] #1A, 1B, 2A, 2B, 2C, 2D

#Set Initial Parameters
ep0 = 1 #Set initial ep as an even mix between exploration and exploitation
ep_val_list = [1]
sep_fact = 1.0
gen_heat_map_data = True
normalize = True
noise_mean = 0
noise_std = None
kernel_enum_val = 1
lenscl = None #list([0.136113749, 221.573761, 830.968019, 1.67917241, 0.3, 0.2])
if isinstance(lenscl, list):
    lenscl = json.dumps(lenscl)
outputscl = None
retrain_GP = 25
reoptimize_obj = 25
bo_iter_tot = 50
bo_run_total = 5
runs_per_job_max = 2
save_data = False
ei_tol = 1e-7
obj_tol = 1e-7
num_x_data = 12
gen_meth_theta = 1 
gen_meth_x = 2
gen_meth_theta_val = 2
num_val_pts = 20
num_theta_multiplier = 10 #How many simulation data points to generate is equal to num_theta_multiplier*number of parameters
initial_seed = 1

assert bo_run_total >= runs_per_job_max, "bo_run_total must be greater than or equal to runs_per_job_max"

#Note: Add loop for idc to consider for lo and hi for CS2 based on something
#Loop over Case Studies
for cs_name_val in cs_val_list:   
    #If CS1, Run the full exploration parameter study, otherwise, just use the constant method
    #If cs > 1, do not generate validation data
    if cs_name_val > 1:
        ep_val_list = [1]
        num_val_pts = 0
        gen_meth_theta_val = None
        
    #If cs > 2 do not generate heat map data either
    if cs_name_val > 2:
        gen_heat_map_data = False
        
    #Loop over methods
    for meth_name_val in meth_val_list:
        #Loop over exploration parameter methods
        for ep_enum_val in ep_val_list:
            #Loop over number of runs
            for bo_run_num in range(1, bo_run_total+1, runs_per_job_max):
                #Note: bo_run_num is the run number of the first run in the job
                #If adding the max number of runs to the run number does not exceed the max range, use it
                if bo_run_num + runs_per_job_max <= bo_run_total+1:
                    runs_per_job = runs_per_job_max
                #Otherwise, the number of runs in the job is the difference between the range max and the run number
                else:
                    runs_per_job = bo_run_total + 1 - bo_run_num
                #Create job parameter dict
                sp = {"cs_name_val": cs_name_val, 
                    "meth_name_val": meth_name_val,
                    "gp_package": gp_pack,
                    "ep0": ep0, 
                    "ep_enum_val": ep_enum_val,
                    "sep_fact" : sep_fact,
                    "normalize" : normalize,
                    "gen_heat_map_data" : gen_heat_map_data,
                    "noise_mean":noise_mean,
                    "noise_std": noise_std,
                    "kernel_enum_val": kernel_enum_val,
                    "lenscl": lenscl,
                    "outputscl": outputscl,
                    "retrain_GP": retrain_GP,
                    "reoptimize_obj": reoptimize_obj,
                    "bo_iter_tot": bo_iter_tot,
                    "bo_run_tot": bo_run_total,
                    "bo_runs_in_job": runs_per_job, 
                    "bo_run_num": bo_run_num,
                    "save_data": save_data,
                    "seed": int(initial_seed + 2*(bo_run_num - 1)), 
                    "ei_tol":ei_tol,
                    "obj_tol":obj_tol,
                    "num_x_data":num_x_data,
                    "gen_meth_theta":gen_meth_theta,
                    "gen_meth_x":gen_meth_x,
                    "gen_meth_theta_val":gen_meth_theta_val,
                    "num_theta_multiplier": num_theta_multiplier,
                    "num_val_pts":num_val_pts}
                #Create jobs for exploration bias study
                job = project.open_job(sp).init()