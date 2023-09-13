# project.py
import signac
from flow import FlowProject, aggregator, with_job

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

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")

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

def converter(instr):
    """
    Converts strings to arrays when loading pandas dataframes
    
    Parameters
    ----------
    instr: str, The string form of the array
    
    Returns 
    -------
    outstr: ndarry (object type), array of the string
    
    """
    outstr = np.fromstring(instr[1:-1],sep=' ')
    return outstr

class Project(FlowProject):
    pass

sf_group = Project.make_group(name="sf_exp_group")

@Project.label
def ep_computed(job):
    #Write script that checks whether the .pickle file is there
    return job.isfile("ep_method_experiment.pickle")

@Project.post(ep_computed)
@Project.operation(with_job = True)
def run_ep_exp(job):
    #For these jobs, sf is ALWAYS 1 
    #How do I ensure duplicate results won't be generated if I'm setting this value here? Technically, this should run 5 times
    job.sp.sep_fact = 1
    #Define method, ep_enum classes, indecies to consider, and kernel
    meth_name = Method_name_enum(meth_name_vals[job.sp.meth_name_val])
    method = GPBO_Methods(meth_name)
    ep_enum = Ep_enum(job.sp.ep_enum_val)
    cs_name_enum = CS_name_enum(job.sp.cs_name_val)
    indecies_to_consider = list(range(job.sp.idc_lo, j0b.sp.idc_hi))
    indecies_to_consider = set_idcs_to_consider(job.sp.cs_name_val)
    kernel = Kernel_enum(job.sp.kernel_enum_value)
    
    #Define Simulator Class
    simulator = simulator_helper_test_fxns(cs_name_enum, indecies_to_consider, job.sp.noise_mean, job.sp.noise_std, job.sp.normalize, 
                                           job.sp.seed)

    #Generate Exp Data
    exp_data = simulator.gen_exp_data(job.sp.num_x_data, job.sp.gen_meth_x)
    
    #Create Exploration Bias Class
    if ep_enum.value == 1:
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, None, None, None, None, None, None)
    elif ep_enum.value == 2:
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, job.sp.bo_iter_tot, None, 0, None, None, None)
    elif ep_enum.value == 3:
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, None, 1.5, None, None, None, None)
    else:
        ep_bias = Exploration_Bias(None, None, ep_enum, None, None, None, None, None, None, None)
        
    #Generate Sim Data
    num_theta_data = len(indecies_to_consider)*job.sp.num_theta_multiplier
    gen_meth_theta = Gen_meth_enum(job.sp.gen_meth_theta)
    gen_meth_x = Gen_meth_enum(job.sp.gen_meth_x)
    sim_data = simulator.gen_sim_data(num_theta_data, job.sp.num_x_data, gen_meth_theta, gen_meth_x, sep_fact, False)
    #Generate Validation Data
    gen_meth_theta_val = Gen_meth_enum(job.sp.gen_meth_theta_val)
    val_data = simulator.gen_sim_data(job.sp.num_val_pts, num_x_data, gen_meth_theta_val, gen_meth_x, sep_fact, True)
    #Gen sse_sim_data and sse_sim_val_data
    sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, False)
    val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, sep_fact, True)

    #Define cs_name and cs_params class
    #Do I need a different name for each experiment to save its results to or will signac take care of that?
#     cs_name = cs_name_enum.name + "_BO_method_" + meth_name.name + "_sep_fact_" + str(round(job.sp.sep_fact,2))
    cs_name = "ep_method_experiment"
    cs_params = CaseStudyParameters(cs_name, job.sp.ep0, job.sp.sep_fact, job.sp.normalize, kernel, job.sp.lenscl, job.sp.outputscl,
                                    job.sp.retrain_GP, job.sp.reoptimize_obj, job.sp.gen_heat_map_data, job.sp.bo_iter_tot,
                                    job.sp.bo_run_tot, job.sp.save_data, job.sp.DateTime, job.sp.seed, job.sp.ei_tol, job.sp.obj_tol)
    #Initialize driver class
    driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, ep_bias, gen_meth_theta_val)
    #Get results
    restart_bo_results = driver.run_bo_restarts()

@Project.label
def ep_analysis_computed(*jobs):
    #Write script that checks whether the .pickle file is there
    
    return all(job.isfile("best_ep.pickle") for job in jobs)  

@sf_group
@Project.pre.after(run_ep_exp)
@Project.post("ep_analysis_computed")
@Project.operation(aggregator=aggregator.groupby("cs_name_val", "meth_name_val"))
@Project.label
def analyze_ep_data(*jobs):
    #Analyze ep data
    
    #Initialize df for all ep method data for each case study and method
    df = pd.DataFrame()
    #For each job in that category:
    for job in jobs:  
        #Read data
        df_job = pd.read_csv("best_ep.csv", header = 0, converters={'Theta Min Obj' : converter, 'Theta Max EI' : converter})
        #Add the value of the EP enum as a coumn
        df_job["EP Method Val"] = job.sp.ep_enum_val
        #Add data to the dataframe with all the data
        df = pd.concat([df, df_job])

    #Find the lowest sse
    index_min_sse = np.argmin(df["Min Obj Act"])
            
    #The best ep method is the one where the sse min is lowest  
    ep_val_best = df["EP Method Val"].iloc[index_min_sse]

    #Save that ep method marker in a csv
    #How should I actually be saving this such that the sep fact experiment can access it?
    with open(job.fn("best_ep.csv"), "w") as file:
        file.write(ep_val_best)

@Project.label
def sf_computed(job):
    #Write script that checks whether the .pickle file is there
    return job.isfile("sf_experiment.pickle")

@sf_group
@Project.pre.after(analyze_ep_data)
@Project.post.isfile("sf_experiment.pickle")
@Project.operation(with_job = True)
def run_sf_exp(job):
    #Get results of type from analyze best
    #How do I ensure duplicate results won't be generated if I'm setting this value here?
    job.sp.ep_enum_val = genfromtxt('best_ep.csv', delimiter=',')
    
    #Run SF using correct ep for that method
    #Define method, ep_enum classes, indecies to consider, and kernel
    meth_name = Method_name_enum(meth_name_vals[job.sp.meth_name_val])
    method = GPBO_Methods(meth_name)
    ep_enum = Ep_enum(job.sp.ep_enum_val)
    cs_name_enum = CS_name_enum(job.sp.cs_name_val)
    indecies_to_consider = set_idcs_to_consider(job.sp.cs_name_val)
    kernel = Kernel_enum(job.sp.kernel_enum_value)
    
    #Define Simulator Class
    simulator = simulator_helper_test_fxns(cs_name_enum, indecies_to_consider, job.sp.noise_mean, job.sp.noise_std, job.sp.normalize, 
                                           job.sp.seed)

    #Generate Exp Data
    exp_data = simulator.gen_exp_data(job.sp.num_x_data, job.sp.gen_meth_x)
    
    #Create Exploration Bias Class
    if ep_enum.value == 1:
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, None, None, None, None, None, None)
    elif ep_enum.value == 2:
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, job.sp.bo_iter_tot, None, 0, None, None, None)
    elif ep_enum.value == 3:
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, None, 1.5, None, None, None, None)
    else:
        ep_bias = Exploration_Bias(None, None, ep_enum, None, None, None, None, None, None, None)
        
    #Generate Sim Data
    num_theta_data = len(indecies_to_consider)*job.sp.num_theta_multiplier
    gen_meth_theta = Gen_meth_enum(job.sp.gen_meth_theta)
    gen_meth_x = Gen_meth_enum(job.sp.gen_meth_x)
    sim_data = simulator.gen_sim_data(num_theta_data, job.sp.num_x_data, gen_meth_theta, gen_meth_x, sep_fact, False)
    #Generate Validation Data
    gen_meth_theta_val = Gen_meth_enum(job.sp.gen_meth_theta_val)
    val_data = simulator.gen_sim_data(job.sp.num_val_pts, num_x_data, gen_meth_theta_val, gen_meth_x, sep_fact, True)
    #Gen sse_sim_data and sse_sim_val_data
    sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, False)
    val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, sep_fact, True)

    #Define cs_name and cs_params class
    #Do I need a different name for each experiment to save its results to or will signac take care of that?
#     cs_name = cs_name_enum.name + "_BO_method_" + meth_name.name + "_sep_fact_" + str(round(job.sp.sep_fact,2))
    cs_name = "sf_experiment"
    cs_params = CaseStudyParameters(cs_name, job.sp.ep0, job.sp.sep_fact, job.sp.normalize, kernel, job.sp.lenscl, job.sp.outputscl,
                                    job.sp.retrain_GP, job.sp.reoptimize_obj, job.sp.gen_heat_map_data, job.sp.bo_iter_tot,
                                    job.sp.bo_run_tot, job.sp.save_data, job.sp.DateTime, job.sp.seed, job.sp.ei_tol, job.sp.obj_tol)
    #Initialize driver class
    driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, ep_bias, gen_meth_theta_val)
    #Get results
    restart_bo_results = driver.run_bo_restarts()

if __name__ == "__main__":
    Project().main()