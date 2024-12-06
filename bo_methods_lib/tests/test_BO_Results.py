import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import copy

import pytest
from bo_methods_lib.GPBO_Classes_New import *  # Fix this later
from bo_methods_lib.GPBO_Class_fxns import *  # Fix this later

# Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
# print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
DateTime = None  ##For Testing


def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules

cs_name1 = "Simple Linear"

cs_val1 = 1
num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data1 = 5
gen_meth_theta = Gen_meth_enum(2)

ep0 = 1
sep_fact = 0.8
normalize = True
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 2
seed = 1
method = GPBO_Methods(Method_name_enum(1))  # 1A

# Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_val1, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(
    method, sim_data1, exp_data1, sep_fact
)
val_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(
    method, val_data1, exp_data1, sep_fact, True
)
gp_emulator1_s = Type_1_GP_Emulator(
    sim_sse_data1,
    val_sse_data1,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)

configuration = {
    "num_x_data": 5,
    "gen_meth_x": Gen_meth_enum(2),
    "sep_fact": 0.8,
    "normalize": False,
    "noise_mean": 0,
    "noise_std": 0.01,
    "seed": 3,
}

results_df = pd.DataFrame()
max_ei_details_df = pd.DataFrame()
why_term = "Test"
heat_map_data_dict = {}
em_list = [gp_emulator1_s, gp_emulator1_s]
#Asserts that init of BO_Results throws correct errors
init_BO_Results_err_list = [
    [configuration,simulator1,exp_data1,em_list,results_df, max_ei_details_df,why_term,""],
    [configuration,simulator1,exp_data1,em_list,results_df, max_ei_details_df,-2, heat_map_data_dict],
    [configuration,simulator1,exp_data1,em_list,results_df, "",why_term, heat_map_data_dict],
    [configuration,simulator1,exp_data1,em_list,"results_df", max_ei_details_df, why_term, heat_map_data_dict],
    [configuration,simulator1,exp_data1,"em_list",results_df, max_ei_details_df, why_term, heat_map_data_dict],
    [configuration,simulator1,"exp_data1",em_list,results_df, max_ei_details_df, why_term, heat_map_data_dict],
    [configuration,"simulator1",exp_data1,em_list,results_df, max_ei_details_df, why_term, heat_map_data_dict],
    ["configuration",simulator1,exp_data1,em_list,results_df, max_ei_details_df, why_term, heat_map_data_dict],
]
@pytest.mark.parametrize(
    "config, sim, exp_data, em_list, res_df, ei_df, why_term, hm_dict",
    init_BO_Results_err_list,
)
def test_init_BO_Results_err(
    config, sim, exp_data, em_list, res_df, ei_df, why_term, hm_dict):
    with pytest.raises((AssertionError, ValueError)):
        results = BO_Results(config, sim, exp_data, em_list, res_df, ei_df, why_term, hm_dict)