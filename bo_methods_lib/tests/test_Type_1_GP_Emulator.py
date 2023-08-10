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
    
#Create sample test data for gp_emulator
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
num_theta_data1 = 5
num_theta_data2 = 2
gen_meth_theta = Gen_meth_enum(2)

ep0 = 1
sep_fact = 0.8
normalize = False
lhs_gen_theta = True
eval_all_pairs = False
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 2
GP_train_iter = 300
bo_iter_tot = 3
bo_run_tot = 2
save_fig = False
save_data = False
num_data = None
seed = 1
ei_tol = 1e-6
method = GPBO_Methods(Method_name_enum(1)) #1A

#Define cs_params, simulator, and exp_data for CS1
cs_params1 = CaseStudyParameters(cs_name1, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, save_fig, save_data, 
                                DateTime, seed, ei_tol)
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, cs_params1)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, sim_data1, exp_data1)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, True)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, val_data1, exp_data1, True)
gp_emulator1_s = Type_1_GP_Emulator(sim_sse_data1, val_sse_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
cs_params2 = CaseStudyParameters(cs_name2, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, save_fig, save_data, 
                                DateTime, seed, ei_tol)
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, cs_params2)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, sim_data2, exp_data2)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, True)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, val_data2, exp_data2, True)
gp_emulator2_s = Type_1_GP_Emulator(sim_sse_data2, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#This test function tests whether get_num_gp_data checker works correctly
                    #emulator class, expected value
get_num_gp_data_list = [[gp_emulator1_s, 25],
                        [gp_emulator1_e, 125],
                        [gp_emulator2_s, 256],
                        [gp_emulator2_e, 6400]]
@pytest.mark.parametrize("gp_emulator, expected", get_num_gp_data_list)
def test_get_num_gp_data(gp_emulator, expected):
    assert gp_emulator.get_num_gp_data() == expected
                        
#This test function tests whether set_gp_model works correctly
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
set_gp_model_list = [[Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, 1, 1, 1],
                     [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, 1, 1, 1, 1],
                     [Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, 1, 1],
                     [Type_2_GP_Emulator, sim_data2, val_data2, 1, 1, 1, 1],
                     [Type_2_GP_Emulator, sim_data2, val_data2, 2, 1, 2, 1],
                     [Type_2_GP_Emulator, sim_data2, val_data2, 2, 2, 2, 2]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", set_gp_model_list)
def test_set_gp_model(gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert gp_emulator.lenscl == exp_lenscl
    assert gp_emulator.outputscl == exp_ops

#This test function tests whether correct errors get thrown on initialization
                       #sim_data, val_data, kernel, lenscl, outputscl, retrain_GP
set_gp_model_err_list = [[sim_data1, val_data1, "string", 1, 1, 1],
                         [sim_data1, val_data1, Kernel_enum(1), 0, 1, 1],
                         [sim_data1, val_data1, Kernel_enum(1), 1, 0, 1],
                         [sim_data1, val_data1, Kernel_enum(1), 1, 1, -2],
                         [sim_data1, val_data1, Kernel_enum(1), -1, 1, 1],
                         [sim_data1, val_data1, Kernel_enum(1), 1, -1, 1],
                         [sim_data1, val_data1, Kernel_enum(1), 1, 1, -1],
                         ["string", val_data1, Kernel_enum(1), 1, 1, 1],
                         [sim_data1, "string", Kernel_enum(1), 1, 1, 1]]
@pytest.mark.parametrize("sim_data, val_data, kernel, lenscl, outputscl, retrain_GP", set_gp_model_err_list)
def test_set_gp_model_err(sim_data, val_data, kernel, lenscl, outputscl, retrain_GP):
    with pytest.raises((AssertionError, ValueError)):   
        gp_emulator = Type_2_GP_Emulator(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#This test function tests whether get_dim_gp_data checker works correctly
                        #Emulator class, number of GP training dims
get_dim_gp_data_list = [[gp_emulator1_s, 2],
                        [gp_emulator1_e, 3],
                        [gp_emulator2_s, 8],
                        [gp_emulator2_e, 10]]
@pytest.mark.parametrize("gp_emulator, expected", get_dim_gp_data_list)
def test_get_dim_gp_data(gp_emulator, expected):
    assert gp_emulator.get_dim_gp_data() == expected

#This test function tests whether set_train_test_data checker works correctly
                            #gp emulator, cs_params
set_train_test_data_list = [[gp_emulator1_s, cs_params1],
                            [gp_emulator1_e, cs_params1],
                            [gp_emulator2_s, cs_params2],
                            [gp_emulator2_e, cs_params2]]
@pytest.mark.parametrize("gp_emulator, cs_params", set_train_test_data_list)
def test_set_train_test_data(gp_emulator, cs_params):
    train_data, test_data = gp_emulator.set_train_test_data(cs_params)
    assert len(train_data.theta_vals) + len(test_data.theta_vals) == len(gp_emulator.gp_sim_data.theta_vals)
    

#This test function tests whether train_gp checker works correctly
                    #cs_params, emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
train_gp_list = [[cs_params1, Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, 1, np.ones(2), 1],
                 [cs_params2, Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, 1, 1, np.ones(8), 1],
                 [cs_params1, Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, 1, np.ones(2)*2, 1],
                 [cs_params1, Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, 2, np.ones(2)*2, 2]]
@pytest.mark.parametrize("cs_params, gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", train_gp_list)
def test_train_gp(cs_params, gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    train_data, test_data = gp_emulator.set_train_test_data(cs_params)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all(gp_emulator.lenscl == exp_lenscl)
    assert trained_ops == exp_ops
    
#This test function tests whether train_gp checker works correctly (optimizes None parameters between bounds)
                    #cs_params, emulator class type, sim data, val_data, lenscl, outputscl, exp_ops
train_gp_opt_list = [[cs_params1, Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, None, 1],
                     [cs_params2, Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, None, 1],
                     [cs_params2, Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, None, 2],
                     [cs_params1, Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, None],
                     [cs_params1, Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, None]]
@pytest.mark.parametrize("cs_params, gp_type, sim_data, val_data, lenscl, outputscl", train_gp_opt_list)
def test_train_gp_opt(cs_params, gp_type, sim_data, val_data, lenscl, outputscl):
    tol = 1e-7
    gp_emulator = gp_type(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    train_data, test_data = gp_emulator.set_train_test_data(cs_params)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all( 1e-5 - tol <= element <= 1e5 + tol for element in trained_lenscl )
    assert 1e-5 - tol <= trained_ops <= 1e2 + tol        
    
#This test function tests whether calc_best_error checker works correctly
                            #gp emulator, cs_params
calc_best_error_list = [[gp_emulator1_s, cs_params1],
                        [gp_emulator2_s, cs_params2]]
@pytest.mark.parametrize("gp_emulator, cs_params", calc_best_error_list)
def test_calc_best_error(gp_emulator, cs_params):
    train_data, test_data = gp_emulator.set_train_test_data(cs_params)
    best_error = gp_emulator.calc_best_error()
    assert np.isclose(best_error, min(train_data.y_vals), rtol = 1e-6)
    
#This test function tests whether eval_gp_ei works correctly
#Define exploration bias and set ep_curr
ep_bias = Exploration_Bias(ep0, None, Ep_enum(1), None, None, None, None, None, None, None)
ep_bias.set_ep()

                 #gp_emulator, exp_data, expected_len
eval_gp_ei_list = [[gp_emulator1_s, exp_data1, 25],
                   [gp_emulator2_s, exp_data2, 256]]
@pytest.mark.parametrize("gp_emulator, exp_data, expected_l", eval_gp_ei_list)
def test_eval_gp_ei(gp_emulator, exp_data, expected_l):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error() #Calc best error
    ei = gp_emulator.eval_gp_ei(exp_data, ep_bias, best_error)
    
    assert len(ei) == expected_l

##How to write these tests given stochasticity of GP?

#This test function tests whether eval_gp_mean_var checker works correctly

#This test function tests whether eval_gp_sse_var checker works correctly

