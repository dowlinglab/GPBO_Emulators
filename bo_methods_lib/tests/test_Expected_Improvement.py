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
    
#Defining this function intentionally here to test function behavior for test cases
def simulator_helper_test_fxns(cs_name, indecies_to_consider, noise_mean, noise_std, seed):
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
                     seed,
                     calc_y_fxn)

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
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, seed)
exp_data = simulator1.gen_exp_data(num_x_data, gen_meth_x)

#Generate exploration bias class    
ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)
ep_bias.set_ep()

#This test function tests whether type_1 works as intended
               ## gp_mean, gp_var, best_error, ei_expected
type_1_list = [[np.array([1]), np.array([0.05**2]), 0.5, 3.737e-26],
               [np.array([1]), np.array([0]), 0.5, 0],
               [np.array([0.4]), np.array([0.01**2]), 0.5, 0.1], #check me
               [np.array([0.5]), np.array([0.5**2]), 0.2, 0.084336]] #Check me
@pytest.mark.parametrize("gp_mean, gp_var, best_error, ei_expected", type_1_list)
def test_type_1_list(gp_mean, gp_var, best_error, ei_expected):
    best_error_metrics = tuple([best_error, np.zeros(5), None])
    acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error_metrics, seed)
    ei = acq_func.type_1()[0]
    assert np.isclose(ei, ei_expected, rtol = 1e-2)
    
#This test function tests whether Expected_Improvement throws the correct errors on initialization
               ## ep_bias, gp_mean, gp_var, exp_data, best_error, method, depth
ei_init_list = [["ep_bias", np.array([0.5]), np.array([0.02]), exp_data, 0.2, Method_name_enum(3), None],
               [ep_bias, [0.5], np.array([0.02]), exp_data, 0.2, Method_name_enum(3), 1],
               [ep_bias, np.array([0.5]), np.array([0.02]), "exp_data", 0.2, Method_name_enum(3), 1],
               [ep_bias, np.array([0.5]), np.array([0.02]), exp_data, "best_error", Method_name_enum(3), 1],
               [ep_bias, np.array([0.5]), np.array([0.02]), exp_data, 0.2, 3, 1],
               [ep_bias, np.array([0.5]), np.array([0.02]), exp_data, 0.2, Method_name_enum(3), 3.2]]
@pytest.mark.parametrize("ep_bias, gp_mean, gp_var, exp_data, best_error, method, depth", ei_init_list)
def test_ei_init_err_list(ep_bias, gp_mean, gp_var, exp_data, best_error, method, depth):
    with pytest.raises((AssertionError, AttributeError, ValueError)):   
        best_error_metrics = tuple([best_error, np.zeros(5), None])
        acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error_metrics, seed, depth)
        ei = acq_func.type_2(method)[0]      

#This test function tests whether type_2 works as intended
               ## gp_mean, gp_var, best_error, method, ei_expected
type_2_list = [[np.array([-14, -3, 0, 1, 6]), np.ones(5)*0.05**2, 0.2, GPBO_Methods(Method_name_enum(3)), 0.98698],
               [np.array([-14, -3, 0, 1, 6]), np.ones(5)*0.04**2, 0.5, GPBO_Methods(Method_name_enum(4)), 40.721],
               [np.array([-14, -3, 0, 1, 6]), np.ones(5)*0.05**2, 0.2, GPBO_Methods(Method_name_enum(5)), 1.04117048],
               [np.array([-14, -3, 0, 1, 6]), np.ones(5)*0.05**2, 0.2, GPBO_Methods(Method_name_enum(6)), 0.18682241],
               [np.array([-14, -3, 0, 1, 6]), np.zeros(5), 0.2, GPBO_Methods(Method_name_enum(3)), 0],
               [np.array([-14, -3, 0, 1, 6]), np.zeros(5), 0.5, GPBO_Methods(Method_name_enum(4)), 0],
               [np.array([-14, -3, 0, 1, 6]), np.zeros(5), 0.5, GPBO_Methods(Method_name_enum(5)), 2.78127298],
               [np.array([-14, -3, 0, 1, 6]), np.zeros(5), 0.5, GPBO_Methods(Method_name_enum(6)), 0.49948086]]

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