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

cs_name1  = CS_name_enum(1)
cs_name2  = CS_name_enum(2)

num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data1 = 5
num_theta_data2 = 2
gen_meth_theta = Gen_meth_enum(2)

ep0 = 1
sep_fact = 0.8
normalize = True
noise_mean = 0
noise_std = 0.01
noise_std = 0
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(5)) #2C

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1.value, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, sim_data1, exp_data1, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, val_data1, exp_data1, sep_fact, True)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2.value, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, sim_data2, exp_data2, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, val_data2, exp_data2, sep_fact, True)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#This test function tests whether get_num_gp_data checker works correctly
                    #emulator class, expected value
get_num_gp_data_list = [[gp_emulator1_e, 125],
                        [gp_emulator2_e, 400]]
@pytest.mark.parametrize("gp_emulator, expected", get_num_gp_data_list)
def test_get_num_gp_data(gp_emulator, expected):
    assert gp_emulator.get_num_gp_data() == expected
    
#This test function tests whether get_num_gp_data throws correct errors
                            #sim_data
get_num_gp_data_err_list =   ["sim_data", None, 1]
@pytest.mark.parametrize("sim_data", get_num_gp_data_err_list)
def test_get_num_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        gp_emulator_fail = Type_2_GP_Emulator(sim_data, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)
        gp_emulator_fail.get_num_gp_data()
                        
#This test function tests whether set_gp_model works correctly
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
set_gp_model_list = [[Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, 1, 1],
                     [Type_2_GP_Emulator, sim_data2, val_data2, 1, 1, 1, 1],
                     [Type_2_GP_Emulator, sim_data2, val_data2, 2, 1, 2, 1],
                     [Type_2_GP_Emulator, sim_data2, val_data2, 2, 2, 2, 2]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", set_gp_model_list)
def test_set_gp_model(gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)
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
        gp_emulator = Type_2_GP_Emulator(sim_data, val_data, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#This test function tests whether get_dim_gp_data checker works correctly
                        #Emulator class, number of GP training dims
get_dim_gp_data_list = [[gp_emulator1_e, 3],
                        [gp_emulator2_e, 6]]
@pytest.mark.parametrize("gp_emulator, expected", get_dim_gp_data_list)
def test_get_dim_gp_data(gp_emulator, expected):
    assert gp_emulator.get_dim_gp_data() == expected
    
#This test function tests whether get_dim_gp_data throws correct errors
                            #sim_data
get_dim_gp_data_err_list =   ["sim_data", None, 1]
@pytest.mark.parametrize("sim_data", get_dim_gp_data_err_list)
def test_get_dim_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        gp_emulator_fail = Type_2_GP_Emulator(sim_data, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)
        gp_emulator_fail.get_dim_gp_data()

#This test function tests whether set_train_test_data checker works correctly
                            #gp emulator, cs_params
set_train_test_data_list = [gp_emulator1_e, gp_emulator2_e]
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
                            [None, None, None, None, None, None, None]]
                                
@pytest.mark.parametrize("theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed", set_train_test_data_err_list)
def test_get_dim_gp_data_err(theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        sim_data_fail =  Data(theta_vals, x_vals, y_vals, None, None, None, None, None, bounds_theta, bounds_x, sep_fact, seed)
        
        if all(var is None for var in [theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed]):
            sim_data_fail = "string"
        
        gp_emulator_fail = Type_2_GP_Emulator(sim_data_fail, val_sse_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)
        train_data, test_data = gp_emulator_fail.set_train_test_data(sim_data_fail.sep_fact, sim_data_fail.seed)
        
#This test function tests whether train_gp checker works correctly
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
#For time sake does not consider cs2 when testing
train_gp_list = [[Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, np.ones(3), 1],
                 [Type_2_GP_Emulator, sim_data1, val_data1, 2, 1, np.ones(3)*2, 1],
                 [Type_2_GP_Emulator, sim_data1, val_data1, 2, 2, np.ones(3)*2, 2]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", train_gp_list)
def test_train_gp(gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(sim_data, val_data, None,None,None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None,None,None,None)
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all(gp_emulator.lenscl == exp_lenscl)
    assert trained_ops == exp_ops
    
#This test function tests whether train_gp checker works correctly (optimizes None parameters between bounds)
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_ops
train_gp_opt_list = [[Type_2_GP_Emulator, sim_data1, val_data1, None, 1],
                     [Type_2_GP_Emulator, sim_data1, val_data1, 1, None],
                     [Type_2_GP_Emulator, sim_data1, val_data1, 2, None]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl", train_gp_opt_list)
def test_train_gp_opt(gp_type, sim_data, val_data, lenscl, outputscl):
    tol = 1e-7
    gp_emulator = gp_type(sim_data, val_data, None,None,None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None,None,None,None)
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()

#This test function tests whether train_gp throws correct errors
                  # gp_emulator, feature_train_data
train_gp_err_list = [[gp_emulator1_e, None],
                     [gp_emulator2_e, None]]
                                
@pytest.mark.parametrize("gp_emulator, feature_train_data", train_gp_err_list)
def test_train_gp_err(gp_emulator, feature_train_data):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed) 
    with pytest.raises((AssertionError, ValueError)): 
        if feature_train_data is not True:
            gp_emulator_fail.feature_train_data = feature_train_data
            
        gp_emulator_fail.train_gp()
        
#This test function tests whether calc_best_error checker works correctly
                            #gp emulator, exp_data, sim_sse_data
calc_best_error_list = [[gp_emulator1_e, exp_data1, sim_sse_data1],
                        [gp_emulator2_e, exp_data2, sim_sse_data2]]
@pytest.mark.parametrize("gp_emulator, exp_data, sim_sse_data", calc_best_error_list)
def test_calc_best_error(gp_emulator, exp_data, sim_sse_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(method, exp_data)
    assert np.isclose(best_error, float(min(sim_sse_data.y_vals)), rtol = 1e-6)
    assert np.allclose(be_theta, sim_sse_data.theta_vals[np.argmin(sim_sse_data.y_vals)], rtol = 1e-6)
    assert np.isclose(sum(best_sq_error), float(min(sim_sse_data.y_vals)), rtol = 1e-6)

#This test function tests whether calc_best_error throws correct errors
                          #gp_emulator, exp_data, train_data, y_vals, method
calc_best_error_err_list = [[gp_emulator1_e, exp_data1, None, True, method],
                            [gp_emulator1_e, exp_data1, True, None, method],
                            [gp_emulator1_e, None, True, True, method],
                            [gp_emulator2_e, exp_data2, None, True, method],
                            [gp_emulator2_e, exp_data2, True, None, method],
                            [gp_emulator2_e, exp_data2, True, True, None]]
                                
@pytest.mark.parametrize("gp_emulator, exp_data, train_data, y_vals, method", calc_best_error_err_list)
def test_calc_best_error_err(gp_emulator, exp_data, train_data, y_vals, method):
    with pytest.raises((AssertionError, AttributeError, ValueError)):               
        gp_emulator_fail = copy.copy(gp_emulator)
        if train_data is True:
            train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed) 
        else:
            gp_emulator_fail.train_data = None
            
        if y_vals is None:
            gp_emulator_fail.train_data.y_vals = y_vals
            
        best_error = gp_emulator_fail.calc_best_error(method, exp_data)
        
#This test function tests whether eval_gp_ei works correctly
#Define exploration bias and set ep_curr
ep_bias = Exploration_Bias(ep0, None, Ep_enum(1), None, None, None, None, None, None, None)
ep_bias.set_ep()

                 #gp_emulator, exp_data, method
eval_ei_test_list = [[gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, exp_data, method", eval_ei_test_list)
def test_eval_ei_test(gp_emulator, exp_data, method):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() #Train model 
    #Set testing data to training data for example
    gp_emulator.feature_test_data = gp_emulator.featurize_data(gp_emulator.train_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(method, exp_data) #Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_test(exp_data, ep_bias, best_error_metrics, method, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(gp_emulator.train_data.theta_vals)

                 #gp_emulator, exp_data, method
eval_ei_val_list = [[gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(5))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(6))]]
@pytest.mark.parametrize("gp_emulator, exp_data, method_val", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data, method_val):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(method_val, exp_data) #Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_val(exp_data, ep_bias, best_error_metrics, method_val, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(gp_emulator.gp_val_data.theta_vals)
    
    
                 #gp_emulator, exp_data, method
eval_ei_cand_list = [[gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", eval_ei_cand_list)
def test_eval_ei_cand(gp_emulator, simulator, exp_data, method):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() #Train model   
    candidate = Data(None, exp_data.x_vals, None, None, None, None, None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(method, exp_data) #Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_cand(exp_data, ep_bias, best_error_metrics, method, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(gp_emulator.cand_data.theta_vals)
    
    
                 #gp_emulator, exp_data, method
eval_ei_misc_list = [[gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", eval_ei_misc_list)
def test_eval_ei_misc(gp_emulator, simulator, exp_data, method):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() #Train model  
    misc_data = Data(None, exp_data.x_vals, None,None,None,None,None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "misc_data thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals
    feature_misc_data = gp_emulator.featurize_data(misc_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(method, exp_data) #Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics, method, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(misc_data.theta_vals)

#This test function tests whether eval_ei_cand/val/test/and misc throw correct errors
                          #gp_emulator, simualtor, exp_data, ep_bias, method, data
calc_ei_err_list = [[gp_emulator1_e, simulator1, exp_data1, ep_bias, method, None],
                            [gp_emulator1_e, simulator1, None, ep_bias, method, True],
                            [gp_emulator1_e, simulator1, exp_data1, ep_bias, GPBO_Methods(Method_name_enum(1)), True],
                            [gp_emulator1_e, simulator1, exp_data1, ep_bias, "str", True],
                            [gp_emulator1_e, simulator1, exp_data1, None, method, True],
                            [gp_emulator2_e, simulator2, exp_data2, None, method, True],
                            [gp_emulator2_e, simulator2, exp_data2, ep_bias, None, True]]
                                
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, ep_bias, method, data", calc_ei_err_list)
def test_calc_ei_err(gp_emulator, simulator, exp_data, ep_bias, method, data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator_fail = copy.copy(gp_emulator)
    gp_emulator_fail.train_gp() #Train model 

    if data is True and exp_data is not None:
        misc_data = Data(None, exp_data.x_vals, None,None,None,None,None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
        theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "misc_data thetas"
        theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
        misc_data.theta_vals = theta_vals
        feature_misc_data = gp_emulator.featurize_data(misc_data)
        if isinstance(method, GPBO_Methods):
            best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(method, exp_data) #Calc best error
            best_error_metrics = best_error, be_theta, best_sq_error
        else:
            best_error_metrics = None
    else:
        data = None
        best_error_metrics = None

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.cand_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_cand(exp_data, ep_bias, best_error_metrics, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        gp_emulator_fail.test_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error_metrics, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        gp_emulator_fail.gp_val_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_val(exp_data, ep_bias, best_error_metrics, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error_metrics, method)
        
    if method == GPBO_Methods(Method_name_enum(5)):
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.cand_data = data #Set candidate point
            ei = gp_emulator_fail.eval_ei_cand(exp_data, ep_bias, best_error_metrics, method, 0)
        with pytest.raises((AssertionError, AttributeError, ValueError)):        
            gp_emulator_fail.test_data = data #Set candidate point
            ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error_metrics, method, 0.8)
        with pytest.raises((AssertionError, AttributeError, ValueError)):        
            gp_emulator_fail.gp_val_data = data #Set candidate point
            ei = gp_emulator_fail.eval_ei_val(exp_data, ep_bias, best_error_metrics, method, "1")
        with pytest.raises((AssertionError, AttributeError, ValueError)):        
            ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error_metrics, method, None)
        
        
                 #gp_emulator, exp_data, method
featurize_data_list = [[gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(6))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(7))]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", featurize_data_list)
def test_featurize_data(gp_emulator, simulator, exp_data, method):
    misc_data = Data(None, exp_data.x_vals, None,None,None,None,None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "misc_data thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals
    feature_misc_data = gp_emulator.featurize_data(misc_data)
    
    #Multiply by 5 because there is 1 prediction for each x data point
    assert gp_emulator.get_dim_gp_data() == feature_misc_data.shape[1]

#Test that featurize_data throws the correct errors  
                     #gp_emulator, simulator, exp_data, bad_data_val
featurize_data_err_list = [[gp_emulator1_e, simulator1, exp_data1, None],
                           [gp_emulator1_e, simulator1, exp_data1, True],
                           [gp_emulator2_e, simulator2, exp_data2, None],
                           [gp_emulator2_e, simulator2, exp_data2, True],]
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
num_theta_data1 = 10
num_theta_data2 = 5
gen_meth_theta = Gen_meth_enum(1)

ep0 = 1
sep_fact = 0.8
normalize = True
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(5)) #2C

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1.value, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2.value, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)
    
#This test function tests whether eval_gp_mean_var checker works correctly
expected_mean1_test = np.array([
    -4.72258552, -2.43031039, 0.54347301, 4.03296498, 6.9245544,
-4.15991685, -2.32636575, -0.57124432, 0.59673766, 1.74169483

])
expected_var1_test = np.array([
    4.33736851, 3.93959677, 3.93170407, 3.93959677, 4.33736851,
4.90243459, 4.47375484, 4.455128, 4.47375484, 4.90243459

])

expected_mean2_test = np.array([
       5.67435502, 5.63441328, 5.57075905, 5.52334012, 5.50152108, 5.62957301,
5.58619736, 5.5284916, 5.48960623, 5.47616581, 5.5580813, 5.51836324,
5.48034238, 5.46203136, 5.46037859, 5.49701705, 5.46268491, 5.44565487,
5.44994169, 5.4603697, 5.46030582, 5.43272097, 5.43008552, 5.44833164,
5.46609181
])

expected_var2_test = np.array([
       0.65430736, 0.65387554, 0.6538284, 0.65387554, 0.65430736,
0.65387554, 0.65346313, 0.65342539, 0.65346313, 0.65387554,
0.6538284, 0.65342539, 0.65338965, 0.65342539, 0.6538284,
0.65387554, 0.65346313, 0.65342539, 0.65346313, 0.65387554,
0.65430736, 0.65387554, 0.6538284, 0.65387554, 0.65430736
])
                             #gp_emulator, expected_mean, expected_var
eval_gp_mean_var_test_list = [[gp_emulator1_e, False, expected_mean1_test, expected_var1_test],
                              [gp_emulator1_e, True, expected_mean1_test, expected_var1_test],
                              [gp_emulator2_e, False, expected_mean2_test, expected_var2_test]]
@pytest.mark.parametrize("gp_emulator, covar, expected_mean, expected_var", eval_gp_mean_var_test_list)
def test_eval_gp_mean_var_test(gp_emulator, covar, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test(False) #Calc mean, var of gp
    gp_mean, gp_covar = gp_emulator.eval_gp_mean_var_test(True) #Calc mean, var of gp  

    assert len(gp_mean) == len(test_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    assert len(gp_covar.shape) == 2 
    assert gp_covar.shape[0] == gp_covar.shape[1]

expected_mean1_val = np.array([
    -5.11516336, -2.7389, 0.5171214, 4.36658836, 7.46267137, -6.41038645,
-3.66877136, 0.01010446, 3.72908251, 6.49800347, -0.60563402, 0.09891939,
0.86366737, 3.50048307, 6.69105288, -5.18541074, -2.61091171, -0.09273532,
2.20548939, 4.48224462, -4.40770396, -2.50470561, -1.03574225, -0.86884226,
-0.58609131, -4.72258552, -2.43031039, 0.54347301, 4.03296498, 6.9245544,
-8.09403318, -4.66575609, -0.87095501, 1.49401502, 3.08378186, -7.72233521,
-4.20488713, -0.66036072, 1.63383651, 3.51805956, -4.15991685, -2.32636575,
-0.57124432, 0.59673766, 1.74169483, -0.29783504, 0.30993434, 1.10472017,
3.94421232, 7.24198702
])

expected_var1_val = np.array([
    3.53945767, 3.03820865, 3.02868304, 3.03820865, 3.53945767,
3.52289855, 3.01932453, 3.01005975, 3.01932453, 3.52289855,
3.25824407, 2.76660456, 2.76141739, 2.76660456, 3.25824407,
3.39344733, 2.9181482, 2.91191561, 2.9181482, 3.39344733,
4.14182633, 3.5649589, 3.53921803, 3.5649589, 4.14182633,
4.33736851, 3.93959677, 3.93170407, 3.93959677, 4.33736851,
3.56987876, 3.06714704, 3.05628474, 3.06714704, 3.56987876,
3.15712332, 2.69056212, 2.68762859, 2.69056212, 3.15712332,
4.90243459, 4.47375484, 4.455128, 4.47375484, 4.90243459,
3.33344531, 2.83572307, 2.82895404, 2.83572307, 3.33344531
])

expected_mean2_val = np.array([
       5.04794457, 3.89509766, 4.7022203, 5.34590467, 5.47910491, 5.35335785,
4.62111819, 5.01090039, 5.48881249, 5.77542754, 5.71946639, 5.38987394,
5.49084607, 6.00649593, 6.68402581, 6.46529358, 6.40154033, 6.61616433,
7.56476311, 8.60281383, 7.44288019, 7.72684438, 8.22378017, 9.41646894,
10.32887774, 12.84498193, 11.68319488, 9.91332587, 8.88745988, 8.13856966,
10.96221103, 9.46933066, 7.89774895, 7.16291646, 6.80761502, 8.3741007,
6.95925001, 6.0554951, 5.83166358, 5.77956395, 6.65212084, 5.44761328,
5.11442378, 5.27578893, 5.45018381, 5.79648217, 4.99817648, 4.75214991,
4.813213, 5.25731305, 6.84012168, 6.11818437, 5.42133585, 5.17800959,
5.38042259, 6.16325018, 5.64064368, 5.19221882, 4.86490969, 5.06782293,
5.62701546, 5.27645363, 5.13935951, 5.13398931, 5.1517146, 5.42042472,
4.97430923, 5.06483686, 5.35056628, 5.43994308, 5.2821685, 4.71900688,
5.04702998, 5.47030808, 5.65664941, 5.67435502, 5.63441328, 5.57075905,
5.52334012, 5.50152108, 5.62957301, 5.58619736, 5.5284916, 5.48960623,
5.47616581, 5.5580813, 5.51836324, 5.48034238, 5.46203136, 5.46037859,
5.49701705, 5.46268491, 5.44565487, 5.44994169, 5.4603697, 5.46030582,
5.43272097, 5.43008552, 5.44833164, 5.46609181, 5.97645185, 5.50183379,
5.19853615, 5.31506302, 5.40118337, 5.68983853, 5.29772741, 5.0741676,
5.22119026, 5.38726087, 5.47321967, 5.13272867, 4.98292621, 5.0034509,
5.33793493, 5.38537059, 4.91815987, 5.09089495, 5.11267307, 5.5217863,
5.39603634, 4.86094875, 5.36412312, 5.79615578, 6.15729524
])

expected_var2_val = np.array(
    [0.09332846, 0.07986269, 0.07939828, 0.07986269, 0.09332846, 0.07986269,
0.07061119, 0.06997318, 0.07061119, 0.07986269, 0.07939828, 0.06997318,
0.06939585, 0.06997318, 0.07939828, 0.07986269, 0.07061119, 0.06997318,
0.07061119, 0.07986269, 0.09332846, 0.07986269, 0.07939828, 0.07986269,
0.09332846, 0.09333151, 0.07986391, 0.07939962, 0.07986391, 0.09333151,
0.07986391, 0.07061155, 0.06997358, 0.07061155, 0.07986391, 0.07939962,
0.06997358, 0.06939631, 0.06997358, 0.07939962, 0.07986391, 0.07061155,
0.06997358, 0.07061155, 0.07986391, 0.09333151, 0.07986391, 0.07939962,
0.07986391, 0.09333151, 0.09323977, 0.07982304, 0.07935634, 0.07982304,
0.09323977, 0.07982304, 0.07059517, 0.06995687, 0.07059517, 0.07982304,
0.07935634, 0.06995687, 0.06937901, 0.06995687, 0.07935634, 0.07982304,
0.07059517, 0.06995687, 0.07059517, 0.07982304, 0.09323977, 0.07982304,
0.07935634, 0.07982304, 0.09323977, 0.65430736, 0.65387554, 0.6538284,
0.65387554, 0.65430736, 0.65387554, 0.65346313, 0.65342539, 0.65346313,
0.65387554, 0.6538284, 0.65342539, 0.65338965, 0.65342539, 0.6538284,
0.65387554, 0.65346313, 0.65342539, 0.65346313, 0.65387554, 0.65430736,
0.65387554, 0.6538284, 0.65387554, 0.65430736, 0.09323339, 0.07982063,
0.07935364, 0.07982063, 0.09323339, 0.07982063, 0.07059454, 0.06995613,
0.07059454, 0.07982063, 0.07935364, 0.06995613, 0.06937813, 0.06995613,
0.07935364, 0.07982063, 0.07059454, 0.06995613, 0.07059454, 0.07982063,
0.09323339, 0.07982063, 0.07935364, 0.07982063, 0.09323339
    ])

#                              #gp_emulator, covar expected_mean, expected_var
# eval_gp_mean_var_val_list = [[gp_emulator1_e, False, expected_mean1_val, expected_var1_val],
#                              [gp_emulator1_e, True, expected_mean1_val, expected_var1_val],
#                              [gp_emulator2_e, False, expected_mean2_val, expected_var2_val]]
# @pytest.mark.parametrize("gp_emulator, covar, expected_mean, expected_var", eval_gp_mean_var_val_list)
# def test_eval_gp_mean_var_val(gp_emulator, covar, expected_mean, expected_var):
#     train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
#     gp_emulator.train_gp() 
#     gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val(False) #Calc mean, var of gp 
#     gp_mean, gp_covar = gp_emulator.eval_gp_mean_var_val(True) #Calc mean, var of gp

#     assert len(gp_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(gp_var)
#     assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    
#     #If covar is false, check variance values are correct
#     assert np.allclose(gp_var, expected_var, rtol=1e-02)
#     #Otherwise check that square covariance matrix is returned
#     assert len(gp_covar.shape) == 2 
#     assert gp_covar.shape[0] == gp_covar.shape[1]
    
expected_mean1 = np.array([-5.11516336, -2.7389, 0.5171214, 4.36658836, 7.46267137])
expected_var1 = np.array([3.53945767, 3.03820865, 3.02868304, 3.03820865, 3.53945767])
expected_mean2 = np.array([
       5.04794457, 3.89509766, 4.7022203, 5.34590467, 5.47910491, 5.35335785,
4.62111819, 5.01090039, 5.48881249, 5.77542754, 5.71946639, 5.38987394,
5.49084607, 6.00649593, 6.68402581, 6.46529358, 6.40154033, 6.61616433,
7.56476311, 8.60281383, 7.44288019, 7.72684438, 8.22378017, 9.41646894,
10.32887774
])
expected_var2 = np.array([
       0.09332846, 0.07986269, 0.07939828, 0.07986269, 0.09332846, 0.07986269,
0.07061119, 0.06997318, 0.07061119, 0.07986269, 0.07939828, 0.06997318,
0.06939585, 0.06997318, 0.07939828, 0.07986269, 0.07061119, 0.06997318,
0.07061119, 0.07986269, 0.09332846, 0.07986269, 0.07939828, 0.07986269,
0.09332846
])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_mean_var_misc_list = [[gp_emulator1_e, False, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator1_e, True, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator2_e, False, simulator2, exp_data2, expected_mean2, expected_var2]]
@pytest.mark.parametrize("gp_emulator, covar, simulator, exp_data, expected_mean, expected_var", eval_gp_mean_var_misc_list)
def test_eval_gp_mean_var_misc_cand(gp_emulator, covar, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data, covar) #Calc mean, var of gp 
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_mean_cand, gp_var_cand = gp_emulator.eval_gp_mean_var_cand(covar) #Calc mean, var of gp 

    assert len(gp_mean) == len(misc_data.theta_vals) == len(gp_var) == len(gp_mean_cand) == len(gp_var_cand)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    assert np.allclose(gp_mean_cand, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(gp_var, expected_var, rtol=1e-02)
        assert np.allclose(gp_var_cand, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(gp_var.shape) == 2 
        assert len(gp_var_cand.shape) == 2 
        assert gp_var.shape[0] == gp_var.shape[1]   
        assert gp_var_cand.shape[0] == gp_var_cand.shape[1]

#This function tests whether eval_gp_mean_var_test/val/cand/misc throw the correct errors     
                             #gp_emulator
eval_gp_mean_var_err_list = [gp_emulator1_e]
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
        
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_val(0) #Calc mean, var of gp 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test("False") #Calc mean, var of gp 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand(1) #Calc mean, var of gp 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(misc_data, feat_misc_data, "True") #Calc mean, var of gp 
        
#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_test_sse = np.array([96.50492497, 115.65122575])
expected_var1_test_sse = np.array([2066.8154584, 2638.07266705])
expected_mean2_test_sse = np.array([9.2851424])
expected_var2_test_sse = np.array([74.90450297])
                             #gp_emulator, exp_data, method, expected_mean, expected_var
eval_gp_sse_var_test_list = [[gp_emulator1_e, False, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator1_e, True, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator1_e, False, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator2_e, False, exp_data2, method, expected_mean2_test_sse, expected_var2_test_sse]]
@pytest.mark.parametrize("gp_emulator, covar, exp_data, method, expected_mean, expected_var", eval_gp_sse_var_test_list)
def test_eval_gp_sse_var_test(gp_emulator, covar, exp_data, method, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() 
    gp_emulator.test_data.gp_mean, gp_emulator.test_data.gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test(method, exp_data, covar) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert len(sse_mean)*mult_factor == len(test_data.theta_vals) == len(sse_var)*mult_factor
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2 
        assert sse_var.shape[0] == sse_var.shape[1]
    
#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_val_sse = np.array([
    92.51669873, 65.54131517, 196.14234711, 81.38383942, 139.9572951,
96.50492497, 47.00227637, 47.69010072, 115.65122575, 209.78554424
])
expected_var1_val_sse = np.array([
   1522.28331489, 953.84253059, 3207.44499074, 1215.95511864, 2770.90054159,
2066.81545842, 621.23569709, 576.50909848, 2638.07266705, 3578.00424117
])
expected_mean2_val_sse = np.array([56.67628699, 222.91845154, 8.8533323, 9.2851424, 4.02436433])
expected_var2_val_sse = np.array([31.04599607, 122.06645035, 4.65206117, 74.90450297, 2.18940686])
                             #gp_emulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_val_list = [[gp_emulator1_e, False, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
                            [gp_emulator1_e, True, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
                            [gp_emulator2_e, False, exp_data2, expected_mean2_val_sse, expected_var2_val_sse]]
@pytest.mark.parametrize("gp_emulator, covar, exp_data, expected_mean, expected_var", eval_gp_sse_var_val_list)
def test_eval_gp_sse_var_val(gp_emulator, covar, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp() 
    gp_emulator.gp_val_data.gp_mean, gp_emulator.gp_val_data.gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val(method, exp_data, covar) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()   
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert len(sse_mean)*mult_factor == len(gp_emulator.gp_val_data.theta_vals) == len(sse_var)*mult_factor
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    #Otherwise check that no covariance matrix is returned
    else:
        assert sse_var == None

expected_mean1_sse = np.array([92.51669873])
expected_var1_sse = np.array([1522.28331489])
expected_mean2_sse = np.array([56.67628699])
expected_var2_sse = np.array([31.04599607])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_misc_list = [[gp_emulator1_e, False, simulator1, exp_data1, expected_mean1_sse, expected_var1_sse],
                             [gp_emulator1_e, True, simulator1, exp_data1, expected_mean1_sse, expected_var1_sse],
                             [gp_emulator2_e, False, simulator2, exp_data2, expected_mean2_sse, expected_var2_sse]]
@pytest.mark.parametrize("gp_emulator, covar, simulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_misc_list)
def test_eval_gp_sse_var_misc_cand(gp_emulator, covar, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    misc_data.gp_mean, misc_data.gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_misc(misc_data, method, exp_data, covar) #Calc mean, var of gp sse
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_emulator.cand_data.gp_mean, gp_emulator.cand_data.gp_var = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    sse_mean_cand, sse_var_cand = gp_emulator.eval_gp_sse_var_cand(method, exp_data, covar) #Calc mean, var of gp sse

    mult_factor = exp_data.get_num_x_vals()
    
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert len(sse_mean)*mult_factor == len(misc_data.theta_vals) == len(sse_var)*mult_factor == len(sse_mean_cand)*mult_factor == len(sse_var_cand)*mult_factor
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
        assert np.allclose(sse_var_cand, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2 
        assert len(sse_var_cand.shape) == 2 
        assert sse_var.shape[0] == sse_var.shape[1]   
        assert sse_var_cand.shape[0] == sse_var_cand.shape[1]

#This function tests whether eval_gp_sse_var_test/val/cand/misc throw the correct errors     
                             #gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method, set_covar
eval_gp_sse_var_err_list = [[gp_emulator1_e, simulator1, exp_data1, False, True, True, method, True],
                            [gp_emulator1_e, simulator1, exp_data1, True, False, True, method, True],
                            [gp_emulator1_e, simulator1, exp_data1, True, True, False, method, True],
                            [gp_emulator1_e, simulator1, exp_data1, True, True, False, method, False],
                            [gp_emulator1_e, simulator1, exp_data1, True, True, False, None, True]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method, set_covar", eval_gp_sse_var_err_list)
def test_eval_gp_sse_var_err(gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method, set_covar):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
    gp_emulator_fail.train_gp()
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator_fail.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
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
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val(method, exp_data, "False") 
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val(method, exp_data)      
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test() 
        if set_data is False:
            gp_emulator_fail.test_data = None
        if set_gp_mean is False:
            gp_emulator_fail.test_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.test_data.gp_var = None
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test(method, exp_data, 0) 
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test(method, exp_data) 
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand() 
        if set_data is False:
            gp_emulator_fail.cand_data = None
        if set_gp_mean is False:
            gp_emulator_fail.cand_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.cand_data.gp_var = None
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand(method, exp_data, "True") 
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand(method, exp_data) 
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
            
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data, method, exp_data, 1) 
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data, method, exp_data) 
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(misc_data, feat_misc_data) #Calc mean, var of gp 
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data, method, None)
        
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(misc_data, feat_misc_data) #Calc mean, var of gp 
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data, method, exp_data, None)

#Test that add_next_theta_to_train_data(theta_best_sse_data) works correctly
                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
add_next_theta_to_train_data_list = [[gp_emulator1_e, simulator1, exp_data1],
                                     [gp_emulator2_e, simulator2, exp_data2]]
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
    
    #Append training data
    gp_emulator.add_next_theta_to_train_data(theta_best_data)

    assert len(gp_emulator.train_data.theta_vals) == theta_before + exp_data.get_num_x_vals()
    
#Test that add_next_theta_to_train_data(theta_best_data) throws correct errors correctly
train_data1 =  Data(sim_data1.theta_vals, sim_data1.x_vals, None,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
train_data2 =  Data(None, sim_data1.x_vals, sim_data1.y_vals,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
train_data3 = "str"
theta_best_data1 =  Data(sim_data1.theta_vals, sim_data1.x_vals, None,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
theta_best_data2 =  Data(None, sim_data1.x_vals, sim_data1.y_vals,None,None,None,None,None, simulator1.bounds_theta_reg, simulator1.bounds_x, sep_fact, seed)
theta_best_data3 = "str"
#Test that add_next_theta_to_train_data(theta_best_data) works correctly
                             #gp_emulator, simulator, exp_data, bad_new_data, bad_train_data
add_next_theta_to_train_data_list = [[gp_emulator1_e, simulator1, exp_data1, None, train_data1],
                                     [gp_emulator1_e, simulator1, exp_data1, None, train_data2],
                                     [gp_emulator1_e, simulator1, exp_data1, None, train_data3],
                                     [gp_emulator1_e, simulator1, exp_data1, theta_best_data1, None],
                                     [gp_emulator1_e, simulator1, exp_data1, theta_best_data2, None],
                                     [gp_emulator1_e, simulator1, exp_data1, theta_best_data3, None]]
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
    
    if bad_new_data is not None:
        theta_best_data = bad_new_data
        
    if bad_train_data is not None:
        gp_emulator_fail.train_data = bad_train_data
        
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        #Append training data
        gp_emulator_fail.add_next_theta_to_train_data(theta_best_data)