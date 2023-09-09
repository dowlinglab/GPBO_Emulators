import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.bo_functions_generic import gen_theta_set, clean_1D_arrays
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import * #Fix this later
import pickle

#Ignore warnings caused by "nan" values
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

#Ignore warning from scikit learn hp tuning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

#Define Simulator Class Helper
def simulator_helper_test_fxns(cs_name, indecies_to_consider, noise_mean, noise_std, normalize, seed):
    """
    Sets the model for calculating y based off of the case study identifier.

    Parameters
    ----------
    cs_name: Class, The name/enumerator associated with the case study being evaluated

    Returns
    -------
    calc_y_fxn: function, the function used for calculation is case study cs_name.name
    """
    #Note: Add your function name from GPBO_Class_fxns.py here
    if cs_name.value == 1:      
        theta_names = ['theta_1', 'theta_2']
        bounds_x_l = [-2]
        bounds_x_u = [2]
        bounds_theta_l = [-2, -2]
        bounds_theta_u = [ 2,  2]
        theta_ref = np.array([1.0, -1.0])     
        calc_y_fxn = calc_cs1_polynomial
        
    elif cs_name.value == 2:                          
        theta_names = ['A_1', 'A_2', 'A_3', 'A_4', 'a_1', 'a_2', 'a_3', 'a_4', 'b_1', 'b_2', 'b_3', 'b_4', 'c_1', 
                       'c_2', 'c_3', 'c_4', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'x1_1', 'x1_2', 'x1_3', 'x1_4']
        bounds_x_l = [-1.5, -0.5]
        bounds_x_u = [1, 2]
        bounds_theta_l = [-300,-200,-250, 5,-2,-2,-10, -2, -2,-2,5,-2,-20,-20, -10,-1 ,-2,-2,-2, -2,-2,-2,0,-2]
        bounds_theta_u = [-100,  0, -150, 20,2, 2, 0,  2,  2,  2, 15,2, 0,0   , 0,  2, 2,  2, 2, 2 ,2 , 2, 2,2]
        theta_ref = np.array([-200,-100,-170,15,-1,-1,-6.5,0.7,0,0,11,0.6,-10,-10,-6.5,0.7,1,0,-0.5,-1,0,0.5,1.5,1])      
#         theta_ref = np.array([0.5, 0.5, 0.8, 2/3, 0.25, 0.25, 0.35, 0.675, 0.5, 0.5, 0.6, 0.65, 0.5, 0.5, 0.35, 28333/50000, 0.75, 0.5,
#     0.375, 0.25, 0.5, 0.625, 0.75, 0.75])
        calc_y_fxn = calc_muller
        
    else:
        raise ValueError("self.CaseStudyParameters.cs_name.value must exist!")

    return Simulator(indecies_to_consider, 
                     theta_ref,
                     theta_names,
                     bounds_theta_l, 
                     bounds_x_l, 
                     bounds_theta_u, 
                     bounds_x_u, 
                     noise_mean,
                     noise_std,
                     normalize,
                     seed,
                     calc_y_fxn)

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")

#Set Method and Case Study
cs_name_enum  = CS_name_enum(1)
indecies_to_consider = list(range(0, 2)) #This is what changes for different subproblems of CS1
meth_name_vals = [4]

#Set Initial Parameters
ep0 = 1
ep_enum_list = [Ep_enum(1)]
sep_fact_list = np.linspace(0.5,1,6) #For CS1 use 0.5 to 1, for CS2 use 0.5, 0.75, and 1
normalize = False
gen_heat_map_data = True
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = None
outputscl = 1 #We know the scale on this test problem
retrain_GP = 5
reoptimize_obj = 5
bo_iter_tot = 100
bo_run_tot = 15
save_data = True
seed = 1
ei_tol = 1e-6
obj_tol = 1e-4

num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data = 30 #Use 1.5x amount of training data in ep bias study
gen_meth_theta = Gen_meth_enum(1)
num_theta_data_val = 20
gen_meth_theta_val = Gen_meth_enum(2)

#Define Simulator Class
simulator = simulator_helper_test_fxns(cs_name_enum, indecies_to_consider, noise_mean, noise_std, normalize, seed)

#Generate Exp Data
exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)

#Loop over methods
for j in range(len(meth_name_vals)):
    meth_name = Method_name_enum(meth_name_vals[j])
    method = GPBO_Methods(meth_name)
    ep_enum = ep_enum_list[j]
    #Create Exploration Bias Class
    if ep_enum.value == 1:
        ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)
    elif ep_enum.value == 2:
        ep_bias = Exploration_Bias(ep0, None, ep_enum, None, bo_iter_tot, None, 0, None, None, None)
    elif ep_enum.value == 3:
        ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, 1.5, None, None, None, None)
    else:
        ep_bias = Exploration_Bias(None, None, ep_enum, None, None, None, None, None, None, None)
        
    #Loop over number of number of separation factors
    for i in range(len(sep_fact_list)):
        sep_fact = sep_fact_list[i]
        #Generate Sim Data
        sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, False)
        #Generate Validation Data
        val_data = simulator.gen_sim_data(num_theta_data_val, num_x_data, gen_meth_theta_val, gen_meth_x, sep_fact, True)
        #Gen sse_sim_data and sse_sim_val_data
        sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, False)
        val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, sep_fact, True)

        #Define cs_name and cs_params class
        cs_name = "CS1_BO_method_" + meth_name.name + "_sep_fact_" + str(round(sep_fact_list[i],2))
        cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, 
                                    reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, 
                                    seed, ei_tol, obj_tol)
        #Initialize driver class
        driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, ep_bias, gen_meth_theta_val)
        #Get results
        restart_bo_results = driver.run_bo_restarts()