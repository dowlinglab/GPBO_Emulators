import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import copy

import bo_methods_lib
# from bo_methods_lib.bo_methods_lib.bo_functions_generic import gen_theta_set, clean_1D_arrays
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import * #Fix this later
from bo_methods_lib.bo_methods_lib.analyze_data import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import * #Fix this later
import pympler
import pickle
print(scipy.__version__)

from pympler import asizeof
import gpflow
import tensorflow_probability as tfp

from matplotlib import pyplot as plt

#Set Parameters
cs_name_val = 1
ep0 = 1 #Set initial ep as an even mix between exploration and exploitation
ep_enum_val = 1
meth_name_val = 1
sep_fact = 1.0
gen_heat_map_data = False
normalize = True
noise_mean = 0
noise_std = None
kernel_enum_val = 1
lenscl = None #list([0.136113749, 221.573761, 830.968019, 1.67917241, 0.3, 0.2])
outputscl = None
retrain_GP = 5
reoptimize_obj = 5
bo_iter_tot = 5
bo_run_total = 2
runs_per_job_max = 2
bo_runs_in_job = bo_run_total
save_data = False
ei_tol = 1e-7
obj_tol = 1e-7
num_x_data = 5
gen_meth_theta = 1
gen_meth_x = 2
gen_meth_theta_val = 1
num_val_pts = 0
num_theta_multiplier = 10 #How many simulation data points to generate is equal to num_theta_multiplier*number of parameters
seed = 1
sim_seed = 1
noise_std_pct = 0.01

# Define method, ep_enum classes, indecies to consider, and kernel
meth_name = Method_name_enum(meth_name_val)
method = GPBO_Methods(meth_name)
ep_enum = Ep_enum(ep_enum_val)
kernel = Kernel_enum(kernel_enum_val)
lenscl = lenscl
try:
    lenscl = json.loads(lenscl)
except:
    lenscl = lenscl


#All simulator objects will have the same seed. This keeps restarts/jobs consistent for data generation
simulator = simulator_helper_test_fxns(
    cs_name_val, noise_mean, noise_std, sim_seed
)

# Generate Exp Data (OR Add your own experimental data as a Data class object)
gen_meth_x = Gen_meth_enum(gen_meth_x)
if cs_name_val == 16:
    x_vals = np.array([0.0,0.1115,0.2475,0.4076,0.5939,0.8230,0.9214,0.9296,0.985,1.000])
elif cs_name_val == 17:
    x_vals = np.array([0.0087,0.0269,0.0568,0.1556,0.2749,0.4449,0.661,0.8096,0.9309,0.9578])    
else:
    x_vals = None

# Set simulator noise_std artifically as 1% of y_exp median (So that noise will be set rather than trained)
exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x, x_vals, noise_std_pct)
simulator.noise_std = np.abs(np.median(exp_data.y_vals)) * noise_std_pct #Manually set noise std

# Create Exploration Bias Class
if ep_enum.value == 1:
    # Constant value stays constant
    ep_bias = Exploration_Bias(
        ep0, None, ep_enum, None, None, None, None, None, None, None
    )
elif ep_enum.value == 2:
    # For decay method, decay from mixed to full exploitation (alpha of 0.5) for this example
    ep_bias = Exploration_Bias(
        ep0,
        None,
        ep_enum,
        None,
        bo_iter_tot,
        None,
        0.5,
        None,
        None,
        None,
    )
elif ep_enum.value == 3:
    # Set ep multiplier to 1.5 as recommended in Boyle
    ep_bias = Exploration_Bias(
        ep0, None, ep_enum, None, None, 1.5, None, None, None, None
    )
elif ep_enum.value == 4:
    # Jasrasaria method will take care of itself
    ep_bias = Exploration_Bias(
        None, None, ep_enum, None, None, None, None, None, None, None
    )
else:
    raise Warning("Ep_enum value must be between 1 and 4!")

# Generate Sim (Training) Data (OR Add your own training data here as a Data class object)
num_theta_data = len(simulator.indices_to_consider) * num_theta_multiplier
gen_meth_theta = Gen_meth_enum(gen_meth_theta)
# Note at present, training data is always the same between jobs since we set the data generation seed to 1    
sim_data = simulator.gen_sim_data(
    num_theta_data,
    num_x_data,
    gen_meth_theta,
    gen_meth_x,
    sep_fact,
    sim_seed,
    False,
    x_vals
)

y_to_sse = False
# Gen sse_sim_data and sse_sim_val_data
sim_sse_data = simulator.sim_data_to_sse_sim_data(
    method, sim_data, exp_data, sep_fact, y_to_sse
)

# Generate validation data if applicable. This is only useful for small (<4 Params + 1 State Point). Otherwise this takes up too much memory
if num_val_pts > 0:
    gen_meth_theta_val = Gen_meth_enum(
        gen_meth_theta_val
    )  # input is an integer (1 or 2)
    val_data = simulator.gen_sim_data(
        num_val_pts,
        num_x_data,
        gen_meth_theta_val,
        gen_meth_x,
        sep_fact,
        None,
        True,
        x_vals
    )
    y_to_sse = True
    val_sse_data = simulator.sim_data_to_sse_sim_data(
        method, val_data, exp_data, sep_fact, y_to_sse
    )
# Set validation data to None if not generating it
else:
    val_data = None
    val_sse_data = None
    gen_meth_theta_val = gen_meth_theta_val  # Value is None

# Define cs_name and cs_params class
cs_name = get_cs_class_from_val(cs_name_val).name #Save name of case study here
# Signac saves all BO_Results in different folders, so they can have the same name
cs_params = CaseStudyParameters(
    cs_name,
    ep0,
    sep_fact,
    normalize,
    kernel,
    lenscl,
    outputscl,
    retrain_GP,
    reoptimize_obj,
    gen_heat_map_data,
    bo_iter_tot,
    bo_runs_in_job,
    save_data,
    None,
    seed,
    ei_tol,
    obj_tol,
)
# Initialize driver class
driver = GPBO_Driver(
    cs_params,
    method,
    simulator,
    exp_data,
    sim_data,
    sim_sse_data,
    val_data,
    val_sse_data,
    None,
    ep_bias,
    gen_meth_theta,
)
# Get results
gpbo_res_simple, gpbo_res_GP = driver.run_bo_restarts()