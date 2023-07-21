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
def simulator_helper_test_fxns(cs_name, indecies_to_consider, noise_mean, noise_std, case_study_parameters):
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
        theta_ref = np.array([1.0, -1.0])
        theta_names = ['theta_1', 'theta_2']
        bounds_x_l = [-2]
        bounds_x_u = [2]
        bounds_theta_l = [-2, -2]
        bounds_theta_u = [ 2,  2]
        calc_y_fxn = calc_cs1_polynomial
        
    elif cs_name.value == 2:     
        theta_ref = np.array([-200,-100,-170,15,-1,-1,-6.5,0.7,0,0,11,0.6,-10,-10,-6.5,0.7,1,0,-0.5,-1,0,0.5,1.5,1])
                             
        theta_names = ['A_1', 'A_2', 'A_3', 'A_4', 'a_1', 'a_2', 'a_3', 'a_4', 'b_1', 'b_2', 'b_3', 'b_4', 'c_1', 
                       'c_2', 'c_3', 'c_4', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'x1_1', 'x1_2', 'x1_3', 'x1_4']
        bounds_x_l = [-1.5, -0.5]
        bounds_x_u = [1, 2]
        bounds_theta_l = [-300,-200,-250, 5,-2,-2,-10, -2, -2,-2,5,-2,-20,-20, -10,-1 ,-2,-2,-2, -2,-2,-2,0,-2]
        bounds_theta_u = [-100,  0, -150, 20,2, 2, 0,  2,  2,  2, 15,2, 0,0   , 0,  2, 2,  2, 2, 2 ,2 , 2, 2,2]
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
                     case_study_parameters,
                     calc_y_fxn)

cs_name1  = CS_name_enum(1)
cs_name2  = CS_name_enum(2)
indecies_to_consider1 = list(range(0, 2)) #This is what changes for different subproblems of CS1
indecies_to_consider2 = list(range(4, 12)) #This is what changes for different subproblems of CS2

num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
ep0 = 1
sep_fact = 0.8
normalize = False
lhs_gen_theta = True
eval_all_pairs = False
noise_mean = 0
noise_std = 0.01
noise_std = 0
kernel = Kernel_enum(1)
set_lenscl = 1
outputscl = False
retrain_GP = 2
GP_train_iter = 300
bo_iter_tot = 3
bo_run_tot = 2
save_fig = False
save_data = False
num_data = None
seed = 3

#Define cs_params, simulator, and exp_data for CS1
cs_params1 = CaseStudyParameters(cs_name1, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, save_fig, save_data, 
                                DateTime, seed)
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, cs_params1)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)

#Define cs_params, simulator, and exp_data for CS2
cs_params2 = CaseStudyParameters(cs_name2, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, save_fig, save_data, 
                                DateTime, seed)
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, cs_params2)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)

#This test function tests whether get_num_theta checker works correctly
                    #exp_data, expected
get_num_theta_list = [[exp_data1, 5],
                      [exp_data2, 5**2]] #Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x
@pytest.mark.parametrize("exp_data, expected", get_num_theta_list)
def test_get_num_theta(exp_data, expected):
    assert exp_data.get_num_theta() == expected

#This test function tests whether get_dim_theta checker works correctly
                    #exp_data, expected
get_dim_theta_list = [[exp_data1, 2],
                      [exp_data2, 8]]
@pytest.mark.parametrize("exp_data, expected", get_dim_theta_list)
def test_get_dim_theta(exp_data, expected):
    assert exp_data.get_dim_theta() == expected
    
#This test function tests whether get_num_x_vals checker works correctly
                    #exp_data, expected
get_num_x_vals_list = [[exp_data1, 5],
                       [exp_data2, 5**2]] #Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x
@pytest.mark.parametrize("exp_data, expected", get_num_x_vals_list)
def test_get_num_x_vals(exp_data, expected):
    assert exp_data.get_num_x_vals() == expected

#This test function tests whether get_dim_x_vals checker works correctly
                    #exp_data, expected
get_dim_x_vals_list = [[exp_data1, 1],
                       [exp_data2, 2]]
@pytest.mark.parametrize("exp_data, expected", get_dim_x_vals_list)
def test_get_dim_x_vals(exp_data, expected):
    assert exp_data.get_dim_x_vals() == expected
    
#This test function tests whether norm_feature_data checker works correctly
                    #exp_data, 
                    #expected_theta
                    #expected_x
norm_feature_data_list = [[exp_data1, 
                         np.array([[0.75, 0.25], 
                                   [0.75, 0.25], 
                                   [0.75, 0.25], 
                                   [0.75, 0.25], 
                                   [0.75, 0.25]]), 
                         np.array([[0.0], 
                                   [0.25], 
                                   [0.5], 
                                   [0.75], 
                                   [1.0]])]
                       ]
@pytest.mark.parametrize("exp_data, expected_theta, expected_x", norm_feature_data_list)
def test_norm_feature_data(exp_data, expected_theta, expected_x):
    scaled_exp_data = exp_data.norm_feature_data()
    assert np.allclose(expected_theta, scaled_exp_data.theta_vals) 
    assert np.allclose(expected_x, scaled_exp_data.x_vals) 
    
    
#This test function tests whether unnorm_feature_data checker works correctly
                    #exp_data
unnorm_feature_data_list = [exp_data1, exp_data2]
@pytest.mark.parametrize("exp_data", unnorm_feature_data_list)
def test_unnorm_feature_data(exp_data):
    scaled_exp_data = exp_data.norm_feature_data()
    exp_data_reg = scaled_exp_data.unnorm_feature_data()
    assert np.allclose(exp_data.theta_vals, exp_data_reg.theta_vals)
    assert np.allclose(exp_data.x_vals, exp_data_reg.x_vals)
    
#This test function tests whether train_test_idx_split works correctly
                    #exp_data
train_test_idx_split_list = [[cs_params1, exp_data1], 
                             [cs_params2, exp_data2]]
@pytest.mark.parametrize("cs_params, exp_data", train_test_idx_split_list)
def test_train_test_idx_split(cs_params, exp_data):
    train_idx, test_idx = exp_data.train_test_idx_split(cs_params)
    union_set = set(train_idx).union(test_idx)
    assert (len(train_idx) + len(test_idx) == exp_data.get_num_theta())
    assert set(range(exp_data.get_num_theta())).issubset(union_set)