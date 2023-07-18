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

cs_name  = CS_name_enum(1)
indecies_to_consider = list(range(0, 2)) #This is what changes for different subproblems of CS1

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
seed = 1

cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, 
                         save_fig, save_data, DateTime, seed)

simulator = simulator_helper_test_fxns(cs_name, indecies_to_consider, noise_mean, noise_std, cs_params)

#How to combine into 1 test function?
sim_num_theta_data_list = [[1,Gen_meth_enum(1), 1],
                           [5,Gen_meth_enum(1), 5],
                           [1,Gen_meth_enum(2), 1**len(indecies_to_consider)],
                           [5,Gen_meth_enum(2), 5**len(indecies_to_consider)]]
@pytest.mark.parametrize("num_theta_data, gen_meth_theta, expected", sim_num_theta_data_list)
def test_get_sim_num_theta_data(num_theta_data, gen_meth_theta, expected):
    sim_num_theta = simulator.get_sim_num_theta_data(num_theta_data, gen_meth_theta)
    assert sim_num_theta == expected, "sim_num_theta should equal num_theta_data"
    
sim_num_theta_data_err_list = [[0,Gen_meth_enum(1)],
                               [0,Gen_meth_enum(2)],
                               [-1,Gen_meth_enum(1)],
                               [-1,Gen_meth_enum(2)]]
@pytest.mark.parametrize("num_theta_data, gen_meth_theta", sim_num_theta_data_err_list)
def test_get_sim_num_theta_data_err(num_theta_data, gen_meth_theta):
    with pytest.raises(ValueError):
        sim_num_theta = simulator.get_sim_num_theta_data(num_theta_data, gen_meth_theta)
        
    
gen_exp_data_list = [[1,Gen_meth_enum(1), 1],
                     [5,Gen_meth_enum(1), 5],
                     [1,Gen_meth_enum(2), 1],
                     [5,Gen_meth_enum(2), 5]]
@pytest.mark.parametrize("num_x_data, gen_meth_x, expected", gen_exp_data_list)
def test_gen_exp_data(num_x_data, gen_meth_x, expected):
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)
    assert len(exp_data.theta_vals) == len(exp_data.x_vals) == len(exp_data.y_vals) == expected, "y_vals, theta_vals and x_vals should be same length"

    
gen_exp_data_err_list = [[0,Gen_meth_enum(1)],
                         [0,Gen_meth_enum(2)],
                         [-1,Gen_meth_enum(1)],
                         [-1,Gen_meth_enum(2)]]
@pytest.mark.parametrize("num_x_data, gen_meth_x", gen_exp_data_err_list)
def test_gen_exp_data_err(num_x_data, gen_meth_x):
    with pytest.raises(ValueError):
        exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)
    

gen_sim_data_list = [[1, 1, Gen_meth_enum(1), Gen_meth_enum(1), 1],
                     [1, 1, Gen_meth_enum(1), Gen_meth_enum(2), 1],
                     [1, 1, Gen_meth_enum(2), Gen_meth_enum(1), 1],
                     [1, 1, Gen_meth_enum(2), Gen_meth_enum(2), 1],
                     [5, 5, Gen_meth_enum(1), Gen_meth_enum(1), 25],
                     [5, 5, Gen_meth_enum(1), Gen_meth_enum(2), 25],
                     [5, 5, Gen_meth_enum(2), Gen_meth_enum(1), 125],
                     [5, 5, Gen_meth_enum(2), Gen_meth_enum(2), 125],
                     [5, 1, Gen_meth_enum(1), Gen_meth_enum(1), 5],
                     [5, 1, Gen_meth_enum(1), Gen_meth_enum(2), 5],
                     [5, 1, Gen_meth_enum(2), Gen_meth_enum(1), 25],
                     [5, 1, Gen_meth_enum(2), Gen_meth_enum(2), 25],
                     [1, 5, Gen_meth_enum(1), Gen_meth_enum(1), 5],
                     [1, 5, Gen_meth_enum(1), Gen_meth_enum(2), 5],
                     [1, 5, Gen_meth_enum(2), Gen_meth_enum(1), 5],
                     [1, 5, Gen_meth_enum(2), Gen_meth_enum(2), 5]]
@pytest.mark.parametrize("num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected", gen_sim_data_list)
def test_gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected):
    sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x)
    assert len(sim_data.theta_vals) == len(sim_data.y_vals) == expected, "Need same number of theta_vals and y_vals generated"
    
#How can I generalize this to not have to generate a data class every time?
gen_y_data_list = [[1, 1, Gen_meth_enum(2), Gen_meth_enum(2), [-12]],
                   [2, 2, Gen_meth_enum(2), Gen_meth_enum(2), [-12,  -4,   4,  12, -20,   4,  -4,  20]],
                   [1, 2, Gen_meth_enum(2), Gen_meth_enum(2), [-12,  -4]],
                   [2, 1, Gen_meth_enum(2), Gen_meth_enum(2), [-12,   4,  -20,  -4]] ]
@pytest.mark.parametrize("num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected", gen_y_data_list)
def test_gen_y_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected):
    data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x)
    y_data = simulator.gen_y_data(data, 0, 0)
    assert np.allclose(y_data, expected), "Check that y_data is outputting the correct values"
    
#What other tests look good to run?
sim_data = simulator.gen_sim_data(5, 2, Gen_meth_enum(2), Gen_meth_enum(2))
exp_data = simulator.gen_exp_data(2, Gen_meth_enum(2))
sim_data_to_sse_sim_data_list = [[GPBO_Methods(Method_name_enum(1)), sim_data, exp_data, [1,-1], 0],
                                 [GPBO_Methods(Method_name_enum(2)), sim_data, exp_data, [1,-1], -np.inf]]
@pytest.mark.parametrize("method, sim_data, exp_data, expected_arr, expected_val", sim_data_to_sse_sim_data_list)
def test_sim_data_to_sse_sim_data(method, sim_data, exp_data, expected_arr, expected_val):
    sse_sim_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data)
    assert np.allclose(sse_sim_data.theta_vals[np.argmin(sse_sim_data.y_vals)], expected_arr, atol = 0.01)
    assert np.isclose(np.min(sse_sim_data.y_vals), expected_val, 0.01)