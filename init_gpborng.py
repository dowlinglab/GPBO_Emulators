import signac
import numpy as np
import os
import json
import bo_methods_lib

project = signac.init_project("GPBO_RNG")

# Set Method and Case Study
gp_pack = "gpflow"
cs_val_list = [1,2,3,10,11,12,13,14,15,16,17] #See bo_methods_lib/bo_methods_lib/GPBO_Class_fxns.py for more details
meth_val_list = [1, 2, 3, 4, 5, 6, 7]  # Conv, Log Conv., Ind., Log Ind., Sparse Grid, Monte Carlo, E[SSE]

# Set Initial Parameters
ep0 = 1  # Set initial ep as an even mix between exploration and exploitation
ep_enum_val = 1 #Set explotation bias to constant value
sep_fact = 1.0 #Set exploration bias init value (alpha) to 1.0
gen_heat_map_data = False #Do not generate heat map data
normalize = True #Standardize data w/ RobustScaler
noise_mean = 0 #Set noise mean to 0
noise_std = None #Set noise std to None (calculated automatically)
kernel_enum_val = 1 #Set kernel to Matern 5/2
lenscl = None #Set lenscl to None (trainable)
if isinstance(lenscl, list):
    lenscl = json.dumps(lenscl)
outputscl = None #Set outputscl to None (trainable)
retrain_GP = 25 #Retrain GP 25 times per iteration
reoptimize_obj = 25 #Reoptimize objective optimizations 25 times per iteration
bo_iters_tot = {1: 50,
                2: 75,
                3: 75,
                10: 50,
                11: 50,
                12: 50,
                13: 50,
                14: 50,
                15: 50,
                16: 50,
                17: 50} #Total number of iterations
bo_runs_total = {1: 5,
                2: 10,
                3: 10,
                10: 5,
                11: 5,
                12: 5,
                13: 5,
                14: 5,
                15: 5,
                16: 5,
                17: 5,} #Total number of runs (restarts)
runs_per_jobs_max = {1: 5,
                2: 1,
                3: 1,
                10: 1,
                11: 5,
                12: 5,
                13: 3,
                14: 1,
                15: 5,
                16: 5,
                17: 5} #Number of runs per job
save_data = False #Do not save extra ei data
ei_tol = 1e-7 #Set EI tolerance to 1e-7
obj_tol = 1e-7 #Set objective tolerance to 1e-7
num_x_datas = {1: 5,
            2: 5,
            3: 5,
            10: 5,
            11: 10,
            12: 10,
            13: 10,
            14: 5,
            15: 10,
            16: 10,
            17: 10} #Number of x data points
gen_meth_theta = 1 #Generate parameter sets using LHS
gen_meth_x = 2 #Generate x data using a grid
gen_meth_theta_val = None #Don't generate validation data
num_val_pts = 0 #Number of validation points
num_theta_multiplier = 10  # How many simulation data points to generate is equal to num_theta_multiplier*number of parameters
initial_seed = 1 # Initial seed for random number generator

# Loop over Case Studies
for cs_name_val in cs_val_list:
    bo_iter_tot = bo_iters_tot[cs_name_val]
    bo_run_total = bo_runs_total[cs_name_val]
    runs_per_job_max = runs_per_jobs_max[cs_name_val]
    num_x_data = num_x_datas[cs_name_val]
    ep_val_list = [1] #Set explotation bias to constant value

    # Loop over methods
    for meth_name_val in meth_val_list:
        # Loop over number of runs
        for bo_run_num in range(1, bo_run_total + 1, runs_per_job_max):
            # Note: bo_run_num is the run number of the first run in the job
            # If adding the max number of runs to the run number does not exceed the max range, use it
            if bo_run_num + runs_per_job_max <= bo_run_total + 1:
                runs_per_job = runs_per_job_max
            # Otherwise, the number of runs in the job is the difference between the range max and the run number
            else:
                runs_per_job = bo_run_total + 1 - bo_run_num
            # Create job parameter dict
            sp = {
                "cs_name_val": cs_name_val,
                "meth_name_val": meth_name_val,
                "gp_package": gp_pack,
                "ep0": ep0,
                "ep_enum_val": ep_enum_val,
                "sep_fact": sep_fact,
                "normalize": normalize,
                "gen_heat_map_data": gen_heat_map_data,
                "noise_mean": noise_mean,
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
                "seed": int(initial_seed + 2 * (bo_run_num - 1)), #Set seed is different for each job's case studies
                "ei_tol": ei_tol,
                "obj_tol": obj_tol,
                "num_x_data": num_x_data,
                "gen_meth_theta": gen_meth_theta,
                "gen_meth_x": gen_meth_x,
                "gen_meth_theta_val": gen_meth_theta_val,
                "num_theta_multiplier": num_theta_multiplier,
                "num_val_pts": num_val_pts,
            }
            # Create jobs for exploration bias study
            job = project.open_job(sp).init()
