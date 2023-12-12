import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import copy

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
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 2
seed = 1
method = GPBO_Methods(Method_name_enum(1)) #1A

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, normalize, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, sim_data1, exp_data1, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, val_data1, exp_data1, sep_fact, True)
gp_emulator1_s = Type_1_GP_Emulator(sim_sse_data1, val_sse_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, normalize, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, sim_data2, exp_data2, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, val_data2, exp_data2, sep_fact, True)
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
    
#This test function tests whether get_num_gp_data throws correct errors
                            #sim_data
get_num_gp_data_err_list =   ["sim_data", None, 1]
@pytest.mark.parametrize("sim_data", get_num_gp_data_err_list)
def test_get_num_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        gp_emulator_fail = Type_1_GP_Emulator(sim_data, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
        gp_emulator_fail.get_num_gp_data()
                                
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
                         [sim_data1, "string", Kernel_enum(1), 1, 1, 1],
                         [sim_data1, val_data1, Kernel_enum(1), "1", 1, 1],
                         [sim_data1, val_data1, Kernel_enum(1), 1, "1", 1],
                         [sim_data1, val_data1, Kernel_enum(1), 1, 1, "1"],
                         [sim_data1, val_data1, 1, 1, 1, 1]]
@pytest.mark.parametrize("sim_data, val_data, kernel, lenscl, outputscl, retrain_GP", set_gp_model_err_list)
def test_set_gp_model_err(sim_data, val_data, kernel, lenscl, outputscl, retrain_GP):
    with pytest.raises((AssertionError, ValueError)):   
        gp_emulator = Type_1_GP_Emulator(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#This test function tests whether get_dim_gp_data checker works correctly
                        #Emulator class, number of GP training dims
get_dim_gp_data_list = [[gp_emulator1_s, 2],
                        [gp_emulator1_e, 3],
                        [gp_emulator2_s, 8],
                        [gp_emulator2_e, 10]]
@pytest.mark.parametrize("gp_emulator, expected", get_dim_gp_data_list)
def test_get_dim_gp_data(gp_emulator, expected):
    assert gp_emulator.get_dim_gp_data() == expected
    
#This test function tests whether get_dim_gp_data throws correct errors
                            #sim_data
get_dim_gp_data_err_list =   ["sim_data", None, 1]
@pytest.mark.parametrize("sim_data", get_dim_gp_data_err_list)
def test_get_dim_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        gp_emulator_fail = Type_1_GP_Emulator(sim_data, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
        gp_emulator_fail.get_dim_gp_data()

#This test function tests whether set_train_test_data checker works correctly
                            #gp emulator
set_train_test_data_list = [gp_emulator1_s, gp_emulator1_e, gp_emulator2_s, gp_emulator2_e]
@pytest.mark.parametrize("gp_emulator", set_train_test_data_list)
def test_set_train_test_data(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    assert len(train_data.theta_vals) + len(test_data.theta_vals) == len(gp_emulator.gp_sim_data.theta_vals)
    
#This test function tests whether set_train_test_data throws correct errors
                                  #theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed
set_train_test_data_err_list = [[None ,sim_data1.x_vals, sim_data1.y_vals, sim_data1.bounds_x, sim_data1.bounds_theta,1, 1],
                                [sim_data1.theta_vals, None, sim_data1.y_vals, sim_data1.bounds_x, sim_data1.bounds_theta,1, 1],
                                [sim_data1.theta_vals,sim_data1.x_vals, None, sim_data1.bounds_x, sim_data1.bounds_theta,1, 1],
                                [sim_data1.theta_vals,sim_data1.x_vals, sim_data1.y_vals, None, sim_data1.bounds_theta,1, 1],
                                [sim_data1.theta_vals,sim_data1.x_vals, sim_data1.y_vals, sim_data1.bounds_x, None ,1, 1],
                            [sim_data1.theta_vals,sim_data1.x_vals, sim_data1.y_vals, sim_data1.bounds_x, sim_data1.bounds_theta, None, 1],
                            [sim_data1.theta_vals,sim_data1.x_vals, sim_data1.y_vals, sim_data1.bounds_x, sim_data1.bounds_theta, 1, None],
                            [None, None, None, None, None, None, None],
                            [sim_data2.theta_vals,sim_data2.x_vals, sim_data2.y_vals, sim_data2.bounds_x, sim_data2.bounds_theta, 1, None]]
                                
@pytest.mark.parametrize("theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed", set_train_test_data_err_list)
def test_get_dim_gp_data_err(theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        sim_data_fail =  Data(theta_vals, x_vals, y_vals, None, None, None, None, None, bounds_theta, bounds_x, sep_fact, seed)
        
        if all(var is None for var in [theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed]):
            sim_data_fail = "string"
        
        gp_emulator_fail = Type_1_GP_Emulator(sim_data_fail, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
        train_data, test_data = gp_emulator_fail.set_train_test_data(sim_data_fail.sep_fact, sim_data_fail.seed)
           

#This test function tests whether train_gp checker works correctly
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
train_gp_list = [[Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, 1, np.ones(2), 1],
                 [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, 1, 1, np.ones(8), 1],
                 [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, 1, np.ones(2)*2, 1],
                 [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, 2, np.ones(2)*2, 2]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", train_gp_list)
def test_train_gp(gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all(gp_emulator.lenscl == exp_lenscl)
    assert trained_ops == exp_ops
    
#This test function tests whether train_gp checker works correctly (optimizes None parameters between bounds)
                    # emulator class type, sim data, val_data, lenscl, outputscl, exp_ops
train_gp_opt_list = [[Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, None, 1],
                     [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, None, 1],
                     [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, None, 2],
                     [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, None],
                     [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, None]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl", train_gp_opt_list)
def test_train_gp_opt(gp_type, sim_data, val_data, lenscl, outputscl):
    tol = 1e-7
    gp_emulator = gp_type(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all( 1e-5 - tol <= element <= 1e5 + tol for element in trained_lenscl )
    assert 1e-5 - tol <= trained_ops <= 1e2 + tol     
    
#This test function tests whether train_gp throws correct errors
                  # gp_emulator, feature_train_data, set_model_val
train_gp_err_list = [[gp_emulator1_s, None, True],
                     [gp_emulator1_s, True, None],
                     [gp_emulator1_s, True, "string"],
                     [gp_emulator2_s, None, True],
                     [gp_emulator2_s, True, None]]
                                
@pytest.mark.parametrize("gp_emulator, feature_train_data, set_model_val", train_gp_err_list)
def test_train_gp_err(gp_emulator, feature_train_data, set_model_val):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed) 
    with pytest.raises((AssertionError, ValueError)): 
        if set_model_val == True:
            gp_model = gp_emulator_fail.set_gp_model()
        else:
            gp_model = set_model_val
        if feature_train_data is not True:
            gp_emulator_fail.feature_train_data = feature_train_data
            
        gp_emulator_fail.train_gp(gp_model)

    
#This test function tests whether calc_best_error checker works correctly
                            #gp emulator
calc_best_error_list = [gp_emulator1_s, gp_emulator2_s]
@pytest.mark.parametrize("gp_emulator", calc_best_error_list)
def test_calc_best_error(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error = gp_emulator.calc_best_error()[0]
    assert np.isclose(best_error, min(train_data.y_vals), rtol = 1e-6)

#This test function tests whether calc_best_error throws correct errors
                          #gp_emulator, train_data, y_vals
calc_best_error_err_list = [[gp_emulator1_s, None, True],
                            [gp_emulator1_s, True, None],
                            [gp_emulator2_s, None, True],
                            [gp_emulator2_s, True, None]]
                                
@pytest.mark.parametrize("gp_emulator, train_data, y_vals", calc_best_error_err_list)
def test_calc_best_error_err(gp_emulator, train_data, y_vals):
    with pytest.raises((AssertionError, AttributeError, ValueError)):               
        gp_emulator_fail = copy.copy(gp_emulator)
        if train_data is True:
            train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed) 
        else:
            gp_emulator_fail.train_data = None
            
        if y_vals is None:
            gp_emulator_fail.train_data.y_vals = y_vals
            
        best_error = gp_emulator_fail.calc_best_error()

#Define exploration bias and set ep_curr
ep_bias = Exploration_Bias(ep0, None, Ep_enum(1), None, None, None, None, None, None, None)
ep_bias.set_ep()

                 #gp_emulator, exp_data, expected_len
eval_ei_test_list = [[gp_emulator1_s, exp_data1],
                     [gp_emulator2_s, exp_data2]]
@pytest.mark.parametrize("gp_emulator, exp_data", eval_ei_test_list)
def test_eval_ei_test(gp_emulator, exp_data):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    best_error_metrics = gp_emulator.calc_best_error() #Calc best error
    if len(best_error_metrics) < 3:
        best_error_metrics = best_error_metrics + (None,)
    ei = gp_emulator.eval_ei_test(exp_data, ep_bias, best_error_metrics)
    
    assert len(ei[0]) == len(gp_emulator.test_data.theta_vals)
    
                 #gp_emulator, exp_data, expected_len
eval_ei_val_list = [[gp_emulator1_s, exp_data1],
                   [gp_emulator2_s, exp_data2]]
@pytest.mark.parametrize("gp_emulator, exp_data", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    best_error_metrics = gp_emulator.calc_best_error() #Calc best error
    if len(best_error_metrics) < 3:
        best_error_metrics = best_error_metrics + (None,)
    ei = gp_emulator.eval_ei_val(exp_data, ep_bias, best_error_metrics)
    
    assert len(ei[0]) == len(gp_emulator.gp_val_data.theta_vals)

                 #gp_emulator, exp_data, expected_len
eval_ei_cand_list = [[gp_emulator1_s, simulator1, exp_data1],
                   [gp_emulator2_s, simulator2, exp_data2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data", eval_ei_cand_list)
def test_eval_ei_cand(gp_emulator, simulator, exp_data):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model 
    candidate = Data(None, exp_data.x_vals, None, None, None, None, None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    candidate.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set candidate thetas
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    best_error_metrics = gp_emulator.calc_best_error() #Calc best error
    if len(best_error_metrics) < 3:
        best_error_metrics = best_error_metrics + (None,)
    ei = gp_emulator.eval_ei_cand(exp_data, ep_bias, best_error_metrics)
    
    assert len(ei[0]) == len(gp_emulator.cand_data.theta_vals)
    
                 #gp_emulator, simulator, exp_data
eval_ei_misc_list = [[gp_emulator1_s, simulator1, exp_data1],
                   [gp_emulator2_s, simulator2, exp_data2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data", eval_ei_misc_list)
def test_eval_ei_misc(gp_emulator, simulator, exp_data):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model 
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    best_error_metrics = gp_emulator.calc_best_error() #Calc best error
    if len(best_error_metrics) < 3:
        best_error_metrics = best_error_metrics + (None,)
    ei = gp_emulator.eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics)
    
    assert len(ei[0]) == len(misc_data.theta_vals)

#This test function tests whether eval_ei_cand/val/test/and misc throw correct errors
                          #gp_emulator, simualtor, exp_data, ep_bias, best_error, data
calc_best_error_err_list = [[gp_emulator1_s, simulator1, exp_data1, ep_bias, 1, None],
                            [gp_emulator1_s, simulator1, None, ep_bias, 1, True],
                            [gp_emulator1_s, simulator1, exp_data1, None, 1, True],
                            [gp_emulator2_s, simulator2, exp_data2, None, 1, True]]
                                
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, ep_bias, best_error, data", calc_best_error_err_list)
def test_calc_ei_err(gp_emulator, simulator, exp_data, ep_bias, best_error, data):
    gp_emulator_fail = copy.copy(gp_emulator)
    gp_model = gp_emulator_fail.set_gp_model()#Set model
    gp_emulator_fail.train_gp(gp_model) #Train model 

    if data is True and exp_data is not None:
        misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
        misc_data.theta_vals = gp_emulator_fail.gp_sim_data.theta_vals[0].reshape(1,-1) #Set misc thetas
        feature_misc_data = gp_emulator_fail.featurize_data(misc_data) #Set feature vals
    else:
        data = None

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.cand_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_cand(exp_data, ep_bias, best_error)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        gp_emulator_fail.test_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        gp_emulator_fail.gp_val_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error)
    
#Test that featurize_data works as intended    
                     #gp_emulator, simulator, exp_data
featurize_data_list = [[gp_emulator1_s, simulator1, exp_data1],
                     [gp_emulator2_s, simulator2, exp_data2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data", featurize_data_list)
def test_featurize_data(gp_emulator, simulator, exp_data):
    #Make Fake Data class
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    
    assert gp_emulator.get_dim_gp_data() == feature_misc_data.shape[1]
    
#Test that featurize_data throws the correct errors  
                     #gp_emulator, simulator, exp_data, bad_data_val
featurize_data_err_list = [[gp_emulator1_s, simulator1, exp_data1, None],
                           [gp_emulator1_s, simulator1, exp_data1, True],
                           [gp_emulator2_s, simulator2, exp_data2, None],
                           [gp_emulator2_s, simulator2, exp_data2, True],]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, bad_data_val", featurize_data_err_list)
def test_featurize_data_err(gp_emulator, simulator, exp_data, bad_data_val):
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        if bad_data_val is None:
            bad_data = None
        else:
            bounds_theta = simulator.bounds_theta_reg
            bounds_x = simulator.bounds_x
            bad_data = Data(None, exp_data.x_vals, None, None, None, None, None, None, bounds_theta, bounds_x, sep_fact, seed)

        gp_emulator.featurize_data(bad_data) #Set feature vals
    
#Define small case study
num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data1 = 20
num_theta_data2 = 20
gen_meth_theta = Gen_meth_enum(1)

ep0 = 1
sep_fact = 0.8
normalize = False
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(1)) #1A

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, normalize, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, sim_data1, exp_data1, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, val_data1, exp_data1, sep_fact, True)
gp_emulator1_s = Type_1_GP_Emulator(sim_sse_data1, val_sse_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, normalize, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, sim_data2, exp_data2, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, val_data2, exp_data2, sep_fact, True)
gp_emulator2_s = Type_1_GP_Emulator(sim_sse_data2, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    
#This test function tests whether eval_gp_mean_var checker works correctly
expected_mean1_test = np.array([123.59112244, 140.70741161, 105.93285795,  11.12967871])
expected_var1_test = np.array([0.00216589, 0.00041544, 0.00091293, 0.00109246])
expected_mean2_test = np.array([1.04731139e+09, 7.22056709e+08, 1.04245927e+09, 7.56226959e+08])
expected_var2_test = np.array([0.00268639, 0.00265817, 0.00268664, 0.00268468])
                             #gp_emulator, expected_mean, expected_var
eval_gp_mean_var_test_list = [[gp_emulator1_s, expected_mean1_test, expected_var1_test],
                              [gp_emulator2_s, expected_mean2_test, expected_var2_test]]
@pytest.mark.parametrize("gp_emulator, expected_mean, expected_var", eval_gp_mean_var_test_list)
def test_eval_gp_mean_var_test(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 

    assert len(gp_mean) == len(test_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    assert np.allclose(gp_var, expected_var, rtol=1e-02)

expected_mean1_val = np.array([
    91.03770286, 252.05496905, 244.64669534, 17.24441463, 135.96836722, 
    114.54428337, 25.27605883, 188.68054902, 50.6193616, 9.14250992, 
    28.88244341, 71.31846296, 104.04907344, 16.72010299, 32.44833715, 
    149.23921143, 9.69308982, 61.99328525, 234.87968482, 94.41947434
])

expected_var1_val = np.array([
    2.57270099e-03, 4.84143114e-04, 5.26721929e-04, 5.35977028e-04, 
    1.49147780e-03, 4.30235387e-04, 3.24859117e-03, 1.63675521e-03, 
    1.02659772e-04, 1.38383954e-03, 1.09036309e-04, 3.26543531e-04, 
    1.27192898e-03, 1.53256475e-04, 5.45320033e-03, 7.47741787e-04, 
    1.24712340e-03, 4.92944192e-04, 7.10795230e-04, 5.99989984e-05
])

expected_mean2_val = np.array([
    4.07610689e+08, 4.99447280e+08, 9.58283962e+08, 8.52787856e+08, 
    6.76210101e+08, 8.81438172e+09, 1.21751739e+09, 5.66740297e+08, 
    1.02410215e+09, 5.03289027e+08, 7.91898726e+08, 1.32779442e+09, 
    2.93216625e+09, 1.34979052e+09, 3.38146097e+08, 1.19054293e+09, 
    7.35364090e+08, 1.27715655e+09, 2.82220892e+08, 4.96493568e+08
])

expected_var2_val = np.array([
    0.00266814, 0.00267209, 0.00268617, 0.00267093, 0.00268059,
    0.00251564, 0.0026791, 0.0026501, 0.00264348, 0.00266669, 
    0.00267, 0.00268239, 0.0026484, 0.00268539, 0.00265416,
    0.00267798, 0.00268411, 0.00267194, 0.00263841, 0.00264614
])

                             #gp_emulator, expected_mean, expected_var
eval_gp_mean_var_val_list = [[gp_emulator1_s, expected_mean1_val, expected_var1_val],
                              [gp_emulator2_s, expected_mean2_val, expected_var2_val]]
@pytest.mark.parametrize("gp_emulator, expected_mean, expected_var", eval_gp_mean_var_val_list)
def test_eval_gp_mean_var_val(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 

    assert len(gp_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    
expected_mean1 = np.array([173.18646196])
expected_var1 = np.array([1.66493029e-06])
expected_mean2 = np.array([3094950.91217347])
expected_var2 = np.array([5.64219813e-07])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_mean_var_misc_list = [[gp_emulator1_s, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator2_s, simulator2, exp_data2, expected_mean2, expected_var2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, expected_mean, expected_var", eval_gp_mean_var_misc_list)
def test_eval_gp_mean_var_misc_cand(gp_emulator, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    candidate.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set candidate thetas
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_mean_cand, gp_var_cand = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 

    assert len(gp_mean) == len(misc_data.theta_vals) == len(gp_var) == len(gp_mean_cand) == len(gp_var_cand)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    assert np.allclose(gp_mean_cand, expected_mean, rtol=1e-02)
    assert np.allclose(gp_var_cand, expected_var, rtol=1e-02)
    
#This function tests whether eval_gp_mean_var_test/val/cand/misc throw the correct errors     
                             #gp_emulator
eval_gp_mean_var_err_list = [gp_emulator1_s, gp_emulator2_s]
@pytest.mark.parametrize("gp_emulator", eval_gp_mean_var_err_list)
def test_eval_gp_mean_var_err(gp_emulator):
    gp_emulator_fail = copy.copy(gp_emulator)
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_emulator_fail.feature_val_data = None
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_val() #Calc mean, var of gp 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_emulator_fail.feature_test_data = None
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test() #Calc mean, var of gp 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_emulator_fail.feature_cand_data = None
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand() #Calc mean, var of gp 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        misc_data = None
        feat_misc_data = None
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(misc_data, feat_misc_data) #Calc mean, var of gp 
    
expected_mean1 = np.array([173.18646196])
expected_var1 = np.array([1.66493029e-06])
expected_mean2 = np.array([3094950.91217347])
expected_var2 = np.array([5.64219813e-07])
    
#This test function tests whether eval_gp_sse_var checker works correctly
#Define exploration bias and set ep_curr
                             #gp_emulator, expected_mean, expected_var
eval_gp_sse_var_test_list = [[gp_emulator1_s, expected_mean1_test, expected_var1_test],
                              [gp_emulator2_s, expected_mean2_test, expected_var2_test]]
@pytest.mark.parametrize("gp_emulator, expected_mean, expected_var", eval_gp_sse_var_test_list)
def test_eval_gp_sse_var_test(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test()

    assert len(sse_mean) == len(test_data.theta_vals) == len(sse_var)
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)

                             #gp_emulator, expected_mean, expected_var
eval_gp_sse_var_val_list = [[gp_emulator1_s, expected_mean1_val, expected_var1_val],
                              [gp_emulator2_s, expected_mean2_val, expected_var2_val]]
@pytest.mark.parametrize("gp_emulator, expected_mean, expected_var", eval_gp_sse_var_val_list)
def test_eval_gp_sse_var_val(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val()

    assert len(sse_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(sse_var)
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)
    
                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_misc_list = [[gp_emulator1_s, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator2_s, simulator2, exp_data2, expected_mean2, expected_var2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_misc_list)
def test_eval_gp_sse_var_misc_cand(gp_emulator, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_misc(misc_data)
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    candidate.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(1,-1) #Set candidate thetas
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_mean_cand, gp_var_cand = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    sse_mean_cand, sse_var_cand = gp_emulator.eval_gp_sse_var_cand()

    assert len(sse_mean) == len(misc_data.theta_vals) == len(sse_var) == len(sse_mean_cand) == len(sse_var_cand)
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var_cand, expected_var, rtol=1e-02)
    
#This function tests whether eval_gp_sse_var_test/val/cand/misc throw the correct errors     
                             #gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var
eval_gp_sse_var_err_list = [[gp_emulator1_s, simulator1, exp_data1, False, True, True],
                            [gp_emulator1_s, simulator1, exp_data1, True, False, True],
                            [gp_emulator1_s, simulator1, exp_data1, True, True, False],
                            [gp_emulator2_s, simulator2, exp_data2, True, True, False]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var", eval_gp_sse_var_err_list)
def test_eval_gp_sse_var_err(gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator_fail.set_gp_model()
    gp_emulator_fail.train_gp(gp_model)
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    candidate.theta_vals = gp_emulator_fail.gp_sim_data.theta_vals[0].reshape(1,-1) #Set candidate thetas
    gp_emulator_fail.cand_data = candidate #Set candidate point
    gp_emulator_fail.feature_cand_data = gp_emulator_fail.featurize_data(gp_emulator_fail.cand_data) #Set feature vals
    
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_val() 
        if set_data is False:
            gp_emulator_fail.gp_val_data = None
        if set_gp_mean is False:
            gp_emulator_fail.gp_val_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.gp_val_data.gp_var = None
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val()        
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test() 
        if set_data is False:
            gp_emulator_fail.test_data = None
        if set_gp_mean is False:
            gp_emulator_fail.test_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.test_data.gp_var = None
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test()
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand() 
        if set_data is False:
            gp_emulator_fail.cand_data = None
        if set_gp_mean is False:
            gp_emulator_fail.cand_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.cand_data.gp_var = None
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand()
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(misc_data, feat_misc_data) #Calc mean, var of gp 
        if set_data is False:
            misc_data = None
        if set_gp_mean is False:
            misc_data.gp_mean = None
        if set_gp_var is False:
            misc_data.gp_var = None
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data)

    
#Test that add_next_theta_to_train_data(theta_best_sse_data) works correctly
                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
add_next_theta_to_train_data_list = [[gp_emulator1_s, simulator1, exp_data1],
                                     [gp_emulator2_s, simulator2, exp_data2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data", add_next_theta_to_train_data_list)
def test_add_next_theta_to_train_data(gp_emulator, simulator, exp_data):
    #Get number of training data before
    theta_before = len(gp_emulator.train_data.theta_vals)
    #Create fake theta_best_sse_data
    theta_best = gp_emulator.gp_sim_data.theta_vals[0]
    theta_best_repeated = np.repeat(theta_best.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    #Add instance of Data class to theta_best
    theta_best_data = Data(theta_best_repeated, exp_data.x_vals, None, None, None, None, None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    #Calculate y values and sse for theta_best with noise
    theta_best_data.y_vals = simulator.gen_y_data(theta_best_data, simulator.noise_mean, simulator.noise_std)  
    #Set the best data to be in sse form if using a type 1 GP
    theta_best_data = simulator.sim_data_to_sse_sim_data(method, theta_best_data, exp_data, sep_fact, False)
    
    #Append training data
    gp_emulator.add_next_theta_to_train_data(theta_best_data)

    assert len(gp_emulator.train_data.theta_vals) == theta_before + 1
    
#Test that add_next_theta_to_train_data(theta_best_sse_data) throws correct errors correctly
train_data1 =  Data(sim_data1.theta_vals, sim_data1.x_vals, None,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
train_data2 =  Data(None, sim_data1.x_vals, sim_data1.y_vals,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
train_data3 = "str"
theta_best_data1 =  Data(sim_data1.theta_vals, sim_data1.x_vals, None,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
theta_best_data2 =  Data(None, sim_data1.x_vals, sim_data1.y_vals,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
theta_best_data3 = "str"


#Test that add_next_theta_to_train_data(theta_best_sse_data) works correctly
                             #gp_emulator, simulator, exp_data, bad_new_data, bad_train_data
add_next_theta_to_train_data_list = [[gp_emulator1_s, simulator1, exp_data1, None, train_data1],
                                     [gp_emulator1_s, simulator1, exp_data1, None, train_data2],
                                     [gp_emulator1_s, simulator1, exp_data1, None, train_data3],
                                     [gp_emulator1_s, simulator1, exp_data1, theta_best_data1, None],
                                     [gp_emulator1_s, simulator1, exp_data1, theta_best_data2, None],
                                     [gp_emulator1_s, simulator1, exp_data1, theta_best_data3, None]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, bad_new_data, bad_train_data", add_next_theta_to_train_data_list)
def test_add_next_theta_to_train_data(gp_emulator, simulator, exp_data, bad_new_data, bad_train_data):
    gp_emulator_fail = copy.copy(gp_emulator)
    #Create fake theta_best_sse_data
    theta_best = gp_emulator_fail.gp_sim_data.theta_vals[0]
    theta_best_repeated = np.repeat(theta_best.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    #Add instance of Data class to theta_best
    theta_best_data = Data(theta_best_repeated, exp_data.x_vals, None, None, None, None, None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    #Calculate y values and sse for theta_best with noise
    theta_best_data.y_vals = simulator.gen_y_data(theta_best_data, simulator.noise_mean, simulator.noise_std)  
    #Set the best data to be in sse form if using a type 1 GP
    theta_best_data = simulator.sim_data_to_sse_sim_data(method, theta_best_data, exp_data, sep_fact, False)
    
    if bad_new_data is not None:
        theta_best_data = bad_new_data
        
    if bad_train_data is not None:
        gp_emulator_fail.train_data = bad_train_data
        
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        #Append training data
        gp_emulator_fail.add_next_theta_to_train_data(theta_best_data)