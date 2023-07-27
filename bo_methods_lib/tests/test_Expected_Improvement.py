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

ep0 = 1
sep_fact =0.8
bo_iter_tot = bo_run_tot = seed = 1
save_fig = save_data = normalize = eval_all_pairs = False
DateTime = None
noise_mean = 0
noise_std = 0.01
num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
ep_enum = Ep_enum(1)

#Generate some experimental data
cs_name1  = CS_name_enum(1)
indecies_to_consider1 = list(range(0, 2)) #This is what changes for different subproblems of CS1
cs_params1 = CaseStudyParameters(cs_name1, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, save_fig, save_data, 
                                DateTime, seed)
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, cs_params1)
exp_data = simulator1.gen_exp_data(num_x_data, gen_meth_x)

#Generate exploration bias class    
ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)
ep_bias.set_ep()

#This test function tests whether type_1 works as intended
               ## gp_mean, gp_var, best_error, ei_expected
type_1_list = [[[0.5], [0.02], 0.2, 8.622e-4],
               [[0.5], [0], 0.2, 0],
               [[0.5], [0.5], 0.2, 0.065126]]
@pytest.mark.parametrize("gp_mean, gp_var, best_error, ei_expected", type_1_list)
def test_set_ep_list(gp_mean, gp_var, best_error, ei_expected):
    acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error)
    ei = acq_func.type_1(exp_data, ep_bias, best_error)
    assert np.isclose(ep_bias.ep_curr, expected, atol = 1e-2)
    
#This test function tests whether Expected_Improvement throws the correct errors on initialization
               ## ep_bias, gp_mean, gp_var, exp_data, best_error
ei_init_list = [["ep_bias", [0.5], [0.02], exp_data, 0.2],
               [ep_bias, [0.5], [0.02], exp_data, 0.2],
               [ep_bias, [0.5], [0.02], "exp_data", 0.2],
               [ep_bias, [0.5], [0.02], exp_data, "best_error"]]
@pytest.mark.parametrize("ep_bias, gp_mean, gp_var, exp_data, best_error", ei_init_list)
def test_set_ep_list(ep_bias, gp_mean, gp_var, exp_data, best_error):
    with pytest.raises((AssertionError, AttributeError, ValueError)):   
        acq_func = Expected_Improvement(ep_bias, gp_mean, gp_var, exp_data, best_error)