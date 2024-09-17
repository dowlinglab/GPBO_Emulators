import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations

import bo_methods_lib
import pytest
from bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.GPBO_Class_fxns import * #Fix this later

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
# print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
DateTime = None ##For Testing

def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules
    

ep0 = 1
sep_fact =0.8
seed = 1
normalize = False
noise_mean = 0
noise_std = 0.01
num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
ep_enum = Ep_enum(1)

#Generate some experimental data
cs_name1  = CS_name_enum(1)
indecies_to_consider1 = list(range(0, 2)) #This is what changes for different subproblems of CS1
simulator1 = simulator_helper_test_fxns(cs_name1.value, noise_mean, noise_std, seed)
exp_data = simulator1.gen_exp_data(num_x_data, gen_meth_x)

#Generate exploration bias class    
ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)
ep_bias.set_ep()

#This test function tests whether type_1 works as intended
               ## gp_mean, gp_var, best_error, ei_expected
type_1_list = [[np.array([1]), np.diag(np.array([0.05**2])), 0.5, 3.737e-26],
               [np.array([1]), np.diag(np.array([0])), 0.5, 0],
               [np.array([0.4]), np.diag(np.array([0.01**2])), 0.5, 0.1], #check me
               [np.array([0.5]), np.diag(np.array([0.5**2])), 0.2, 0.084336]] #Check me
@pytest.mark.parametrize("gp_mean, gp_var, best_error, ei_expected", type_1_list)
def test_type_1_list(gp_mean, gp_var, best_error, ei_expected):
    best_error_metrics = tuple([best_error, np.zeros(5), None])
    acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error_metrics, seed)
    ei = acq_func.type_1()[0]
    assert np.isclose(ei, ei_expected, rtol = 1e-2)
    
#This test function tests whether Expected_Improvement throws the correct errors on initialization
               ## ep_bias, gp_mean, gp_var, exp_data, best_error, method, depth
ei_init_list = [["ep_bias", np.array([0.5]), np.diag(np.array([0.02])), exp_data, 0.2, Method_name_enum(3), None],
               [ep_bias, [0.5], np.diag(np.array([0.02])), exp_data, 0.2, Method_name_enum(3), 1],
               [ep_bias, np.array([0.5]), np.diag(np.array([0.02])), exp_data, 0.2, Method_name_enum(7), 1],
               [ep_bias, np.array([0.5]), np.diag(np.array([0.02])), "exp_data", 0.2, Method_name_enum(3), 1],
               [ep_bias, np.array([0.5]), np.diag(np.array([0.02])), exp_data, "best_error", Method_name_enum(3), 1],
               [ep_bias, np.array([0.5]), np.diag(np.array([0.02])), exp_data, 0.2, 3, 1],
               [ep_bias, np.array([0.5]), np.diag(np.array([0.02])), exp_data, 0.2, Method_name_enum(3), 3.2]]
@pytest.mark.parametrize("ep_bias, gp_mean, gp_var, exp_data, best_error, method, depth", ei_init_list)
def test_ei_init_err_list(ep_bias, gp_mean, gp_var, exp_data, best_error, method, depth):
    with pytest.raises((AssertionError, AttributeError, ValueError)):   
        best_error_metrics = tuple([best_error, np.zeros(5), None])
        acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error_metrics, seed, depth)
        ei = acq_func.type_2(method)[0]      

#This test function tests whether type_2 works as intended
               ## gp_mean, gp_var, best_error, method, ei_expected
type_2_list = [[np.array([-14, -3, 0, 1, 6]), np.diag(np.ones(5)*0.05**2), 0.2, GPBO_Methods(Method_name_enum(3)), 0.98698],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.ones(5)*0.04**2), 0.5, GPBO_Methods(Method_name_enum(4)), 34.756136],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.ones(5)*0.05**2), 0.2, GPBO_Methods(Method_name_enum(5)), 0.19948094],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.ones(5)*0.05**2), 0.2, GPBO_Methods(Method_name_enum(6)), 0.19747919],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.zeros(5)), 0.2, GPBO_Methods(Method_name_enum(3)), 0],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.zeros(5)), 0.5, GPBO_Methods(Method_name_enum(4)), 0],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.zeros(5)), 0.5, GPBO_Methods(Method_name_enum(5)), 0.49948086],
               [np.array([-14, -3, 0, 1, 6]), np.diag(np.zeros(5)), 0.5, GPBO_Methods(Method_name_enum(6)), 0.49948086]]

@pytest.mark.parametrize("gp_mean, gp_var, best_error, method, ei_expected", type_2_list)
def test_type_2_list(gp_mean, gp_var, best_error, method, ei_expected):
    best_error_metrics = tuple([best_error, np.zeros(5), np.full(5, best_error)])
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error_metrics, seed, depth)
    ei = acq_func.type_2(method)[0]
    assert np.isclose(ei, ei_expected, atol = 1e-2)
    
#Test Error Case for giving a sparse grid value that is not an integer to the sparse grid method
#This test function tests whether Expected_Improvement throws the correct errors on initialization
               ## method, depth
type_2_err_list = [[GPBO_Methods(Method_name_enum(5)), None],
                   [GPBO_Methods(Method_name_enum(5)), 0.8],
                   [5, 1],
                   ["C2", 1]]
@pytest.mark.parametrize("method, depth", type_2_err_list)
def test_type_2_err_list(method, depth):
    with pytest.raises((AssertionError, AttributeError, ValueError)):   
        best_error_metrics = tuple([np.ones(5), np.zeros(5), None])
        gp_mean = np.array([-14, -3, 0, 1, 6])
        gp_var = np.ones(5)*0.05**2
        acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error_metrics, seed, depth)
        ei = acq_func.type_2(method)[0]  