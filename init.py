import signac
import numpy as np
project = signac.init_project()

#Set Method and Case Study
cs_val_list  = [1, 2, 3, 4, 5, 6, 7] #Corresponds to CS1 and all subproblems of CS2

def set_idcs_to_consider(cs_name_val):
    """
    Sets indecies to consider based on problem name
    
    Parameters
    ----------
    cs_name_val: int, the string of the case study name
    
    Returns
    -------
    indecies_to_consider
    """
    cs_val_idx = cs_name_val - 1
    inc_to_consider_lo = [0, 0, 0, 0, 0, 0, 0]
    inc_to_consider_hi = [2, 4, 8, 12, 16, 20, 24]
        
    idc_lo = inc_to_consider_lo[cs_val_idx]
    idc_hi = inc_to_consider_hi[cs_val_idx]
    
    return idc_lo, idc_hi

meth_val_list = [1,2,3,5,4] #Put 2B last because it takes the longest

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
num_theta_multiplier = 10 #How many simulation data points to generate is equal to num_theta_multiplier*number of parameters
num_val_pts = 20

#Note: Add loop for idc to consider for lo and hi for CS2 based on something
#Loop over Case Studies
for cs_name_val in cs_val_list:
    idcs_to_consider_rng = set_idcs_to_consider(cs_name_val)
    #Loop over methods
    for meth_name_val in meth_val_list:
        #Loop over exploration parameter methods
        for ep_enum_val in ep_val_list:
            #Loop over separation factors
            for sep_fact in sep_fact_list:
                #Create job parameter dict
                sp = {"cs_name_val": cs_name_val, 
                      "meth_name_val": meth_name_val,
                      "idc_lo": idcs_to_consider_rng[0]
                      "idc_hi": idcs_to_consider_rng[1]
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
                      "gen_meth_theta_val":gen_meth_theta_val,
                      "num_theta_multiplier": num_theta_multiplier,
                      "num_val_pts":num_val_pts}
                #Run job
                job = project.open_job(sp).init()