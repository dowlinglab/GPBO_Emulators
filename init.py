import signac
import numpy as np
import os
import json
import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import get_study_data_signac, get_best_data
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import set_param_str

project = signac.init_project()

#Set Method and Case Study (Start w/ just 1 and 2 for now)
cs_val_list  = [1] #Corresponds to CS1 and all subproblems of CS2. Full list is [1, 2, 3, 4, 5, 6, 7, 8, 9]
meth_val_list = [1, 2, 3, 4, 5]

#Set Initial Parameters
ep0 = 1 #Set initial ep as an even mix between exploration and exploitation
sep_fact_list = np.linspace(0.5,0.9,5) #For CS1 use 0.5 to 1, for CS2 use 0.5, 0.75, and 1
ep_val_list = [1]
gen_heat_map_data = True
normalize = True
noise_mean = 0
noise_std = 0.01
kernel_enum_val = 1
lenscl = None #list([0.136113749, 221.573761, 830.968019, 1.67917241, 0.3, 0.2])
if isinstance(lenscl, list):
    lenscl = json.dumps(lenscl)
outputscl = 1
retrain_GP = 25
reoptimize_obj = 5
bo_iter_tot = 10
bo_run_tot = 5
save_data = False
seed = 1
ei_tol = 1e-6
obj_tol = 1e-4
num_x_data = 5
gen_meth_theta = 1 
gen_meth_x = 2
gen_meth_theta_val = 2
num_val_pts = 20
num_theta_multiplier = 10 #How many simulation data points to generate is equal to num_theta_multiplier*number of parameters


#Note: Add loop for idc to consider for lo and hi for CS2 based on something
#Loop over Case Studies
for cs_name_val in cs_val_list:
    #Set idcs to consider
    param_name_str = set_param_str(cs_name_val)
    
    #If CS1, Run the full exploration parameter study and/or the sf study, otherwise, just use the constant method
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
            #Set SF to 1 for 1st job
            sep_fact = 1.0
            #Create job parameter dict
            sp = {"cs_name_val": cs_name_val, 
                  "meth_name_val": meth_name_val,
                  "param_name_str": param_name_str,
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
                  "bo_run_tot":bo_run_tot,
                  "save_data": save_data,
                  "seed":seed,
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
        
        #Check that sep_fact study has not already been completed
        #Initialize flag
        run_sf_study = True
        #Loop over jobs in project for each case study and method name
        for job in project.find_jobs({"cs_name_val": cs_name_val, "meth_name_val" : meth_name_val}):
            #Don't create the SF jobs if any of the sep facts are < 1 or the job files for the ep study do not exist
            if job.sp.sep_fact < 1.0 or not os.path.exists(job.fn("BO_Results.gz")):
                run_sf_study = False
                break
                
        #If we are creating jobs for the SF study (CS1 only)      
        if run_sf_study and cs_name_val == 1:
            #Get best data from signac project jobs
            study_id = "ep"
            df, cs_name, theta_true = get_study_data_signac(project, cs_name_val, param_name_str, meth_name_val, study_id, save_csv = True)
            df_best = get_best_data(df, study_id, cs_name, theta_true, param_name_str, date_time_str = None, save_csv = True)

            #Set ep enum val to the best one for that cs and method
            ep_enum_val = df_best["EP Method Val"].iloc[0]

            #Loop over separation factors
            for sep_fact in sep_fact_list:
                #Create job parameter dict
                sp = {"cs_name_val": cs_name_val, 
                      "meth_name_val": meth_name_val,
                      "param_name_str": param_name_str,
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
                      "bo_run_tot":bo_run_tot,
                      "save_data": save_data,
                      "seed":seed,
                      "ei_tol":ei_tol,
                      "obj_tol":obj_tol,
                      "num_x_data":num_x_data,
                      "gen_meth_theta":gen_meth_theta,
                      "gen_meth_x":gen_meth_x,
                      "gen_meth_theta_val":gen_meth_theta_val,
                      "num_theta_multiplier": num_theta_multiplier,
                      "num_val_pts":num_val_pts}

                #Run job
                job = project.open_job(sp).init()