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
        calc_y_fxn_args = None
        
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
        calc_y_fxn_args = calc_y_fxn_args = {"min muller": solve_pyomo_Muller_min(set_param_str(cs_name.value))}
        
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
                     calc_y_fxn,
                     calc_y_fxn_args)

cs_name1  = CS_name_enum(1)
cs_name2  = CS_name_enum(2)
indecies_to_consider1 = list(range(0, 2)) #This is what changes for different subproblems of CS1
indecies_to_consider2 = list(range(16, 24)) #This is what changes for different subproblems of CS2

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
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, sim_data1, exp_data1, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, val_data1, exp_data1, sep_fact, True)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, sim_data2, exp_data2, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, val_data2, exp_data2, sep_fact, True)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#This test function tests whether get_num_gp_data checker works correctly
                    #emulator class, expected value
get_num_gp_data_list = [[gp_emulator1_e, 125],
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
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
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
train_gp_err_list = [[gp_emulator1_e, None, True],
                     [gp_emulator1_e, True, None],
                     [gp_emulator1_e, True, "string"],
                     [gp_emulator2_e, None, True],
                     [gp_emulator2_e, True, None]]
                                
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
                            #gp emulator, exp_data, sim_sse_data
calc_best_error_list = [[gp_emulator1_e, exp_data1, sim_sse_data1],
                        [gp_emulator2_e, exp_data2, sim_sse_data2]]
@pytest.mark.parametrize("gp_emulator, exp_data, sim_sse_data", calc_best_error_list)
def test_calc_best_error(gp_emulator, exp_data, sim_sse_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error = gp_emulator.calc_best_error(method, exp_data)
    assert np.isclose(best_error[0], min(sim_sse_data.y_vals), rtol = 1e-6)

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
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model 
    #Set testing data to training data for example
    gp_emulator.feature_test_data = gp_emulator.featurize_data(gp_emulator.train_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error(method, exp_data) #Calc best error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_test(exp_data, ep_bias, best_error, method, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(gp_emulator.train_data.theta_vals)

                 #gp_emulator, exp_data, method
eval_ei_val_list = [[gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, exp_data, method_val", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data, method_val):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error(method_val, exp_data) #Calc best error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_val(exp_data, ep_bias, best_error, method_val, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(gp_emulator.gp_val_data.theta_vals)
    
    
                 #gp_emulator, exp_data, method
eval_ei_cand_list = [[gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", eval_ei_cand_list)
def test_eval_ei_cand(gp_emulator, simulator, exp_data, method):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model 
    candidate = Data(None, exp_data.x_vals, None, None, None, None, None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error(method, exp_data) #Calc best error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_cand(exp_data, ep_bias, best_error, method, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(gp_emulator.cand_data.theta_vals)
    
    
                 #gp_emulator, exp_data, method
eval_ei_misc_list = [[gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", eval_ei_misc_list)
def test_eval_ei_misc(gp_emulator, simulator, exp_data, method):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model 
    misc_data = Data(None, exp_data.x_vals, None,None,None,None,None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "misc_data thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals
    feature_misc_data = gp_emulator.featurize_data(misc_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error(method, exp_data) #Calc best error
    if method.sparse_grid == True:
        depth = 5
    else:
        depth = None
    ei = gp_emulator.eval_ei_misc(misc_data, exp_data, ep_bias, best_error, method, depth)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0])*num_x_data == len(misc_data.theta_vals)

#This test function tests whether eval_ei_cand/val/test/and misc throw correct errors
                          #gp_emulator, simualtor, exp_data, ep_bias, best_error, method, data
calc_ei_err_list = [[gp_emulator1_e, simulator1, exp_data1, ep_bias, 1, method, None],
                            [gp_emulator1_e, simulator1, None, ep_bias, 1, method, True],
                            [gp_emulator1_e, simulator1, exp_data1, ep_bias, 1, GPBO_Methods(Method_name_enum(1)), True],
                            [gp_emulator1_e, simulator1, exp_data1, ep_bias, 1, "str", True],
                            [gp_emulator1_e, simulator1, exp_data1, None, 1, method, True],
                            [gp_emulator2_e, simulator2, exp_data2, None, 1, method, True],
                            [gp_emulator2_e, simulator2, exp_data2, ep_bias, 1, None, True]]
                                
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, ep_bias, best_error, method, data", calc_ei_err_list)
def test_calc_ei_err(gp_emulator, simulator, exp_data, ep_bias, best_error, method, data):
    gp_emulator_fail = copy.copy(gp_emulator)
    gp_model = gp_emulator_fail.set_gp_model()#Set model
    gp_emulator_fail.train_gp(gp_model) #Train model 

    if data is True and exp_data is not None:
        misc_data = Data(None, exp_data.x_vals, None,None,None,None,None, None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
        theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "misc_data thetas"
        theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
        misc_data.theta_vals = theta_vals
        feature_misc_data = gp_emulator.featurize_data(misc_data)
    else:
        data = None

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.cand_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_cand(exp_data, ep_bias, best_error, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        gp_emulator_fail.test_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        gp_emulator_fail.gp_val_data = data #Set candidate point
        ei = gp_emulator_fail.eval_ei_val(exp_data, ep_bias, best_error, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error, method)
        
    if method == GPBO_Methods(Method_name_enum(5)):
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.cand_data = data #Set candidate point
            ei = gp_emulator_fail.eval_ei_cand(exp_data, ep_bias, best_error, method, 0)
        with pytest.raises((AssertionError, AttributeError, ValueError)):        
            gp_emulator_fail.test_data = data #Set candidate point
            ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error, method, 0.8)
        with pytest.raises((AssertionError, AttributeError, ValueError)):        
            gp_emulator_fail.gp_val_data = data #Set candidate point
            ei = gp_emulator_fail.eval_ei_val(exp_data, ep_bias, best_error, method, "1")
        with pytest.raises((AssertionError, AttributeError, ValueError)):        
            ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error, method, None)
        
        
                 #gp_emulator, exp_data, method
featurize_data_list = [[gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))]]
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
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, normalize, None, None, None, None)
    
#This test function tests whether eval_gp_mean_var checker works correctly
expected_mean1_test = np.array([
    -5.69768671, -2.29834478, -0.14384735,  2.66129928,  7.42554651, -0.19770476,  0.96579034,  0.03521839,  2.4431893 , 11.82000324
])
expected_var1_test = np.array([
    0.62858822, 0.62123159, 0.62119029, 0.62123159, 0.62858822, 0.26891116, 0.2665345 , 0.26647474, 0.2665345 , 0.26891116
])

expected_mean2_test = np.array([
       6.03039151, 6.03566037, 6.04234395, 6.04839094, 6.0522504 ,
       6.02616909, 6.03290993, 6.04176482, 6.04971419, 6.05470286,
       6.02656448, 6.03353238, 6.04312748, 6.05188341, 6.05723479,
       6.03090104, 6.03729112, 6.0463477 , 6.05467673, 6.0594715 ,
       6.03611423, 6.04170528, 6.04939868, 6.0562607 , 6.05979425])

expected_var2_test = np.array([
       0.79938704, 0.7993841 , 0.79938372, 0.79938437, 0.799387  ,
       0.79938432, 0.79938127, 0.79938089, 0.79938155, 0.79938428,
       0.79938374, 0.79938069, 0.79938032, 0.79938096, 0.79938369,
       0.79938422, 0.79938117, 0.79938079, 0.79938145, 0.79938418,
       0.79938705, 0.79938412, 0.79938374, 0.79938439, 0.79938702
])
                             #gp_emulator, expected_mean, expected_var
eval_gp_mean_var_test_list = [[gp_emulator1_e, False, expected_mean1_test, expected_var1_test],
                              [gp_emulator1_e, True, expected_mean1_test, expected_var1_test],
                              [gp_emulator2_e, False, expected_mean2_test, expected_var2_test]]
@pytest.mark.parametrize("gp_emulator, covar, expected_mean, expected_var", eval_gp_mean_var_test_list)
def test_eval_gp_mean_var_test(gp_emulator, covar, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test(covar) #Calc mean, var of gp 

    assert len(gp_mean) == len(test_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(gp_var, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(gp_var.shape) == 2 
        assert gp_var.shape[0] == gp_var.shape[1]

expected_mean1_val = np.array([
    -4.09101293e+00, -1.45927444e-01, 3.06356251e-02, 2.59794613e+00, 1.22681324e+01, -1.02457629e+01, -2.20195064e+00, -1.62709135e-01,
    -4.61825136e-01, 2.30292328e+00, -4.96144100e+00, -1.22965035e+00, 1.90971165e-01, 3.41677427e+00, 1.10782651e+01, -6.65076842e+00, 
    -1.06114406e+00, 2.48216697e-02, 2.64770818e+00, 1.24109057e+01, -8.00854083e+00, -7.34984655e-01, 1.04151757e-02, 2.29024337e-01, 
    5.93860188e+00, -3.40463000e+00, -1.00109374e-01, 1.19008570e-01, 1.11516784e+00, 5.49091795e+00, -1.18823701e+01, -3.46568206e+00,
    -1.41568406e-02, 1.84240971e+00, 6.63677947e+00, -9.57085135e+00, -1.26988663e+00, -3.74977702e-02, -1.18892098e+00, 1.58681497e+00, 
    -5.71892773e+00, -1.44463876e+00, 9.64337327e-02, 3.32006550e+00, 1.16337473e+01, -1.07405931e+01, -8.74989545e-01, -2.58327285e-02,
    -2.41856121e+00, -2.12656067e+00
])

expected_var1_val = np.array([
    0.14894872, 0.14791648, 0.14785142, 0.14791648, 0.14894872, 0.34246146, 0.33877762, 0.33869666, 0.33877762, 0.34246146, 0.3911121, 
    0.3879422, 0.38790525, 0.3879422, 0.3911121, 0.06141792, 0.06108165, 0.06105769, 0.06108165, 0.06141792, 0.02340763, 0.02335528, 
    0.02334802, 0.02335528, 0.02340763, 0.48494208, 0.47891568, 0.47889096, 0.47891568, 0.48494208, 0.44754412, 0.44423061, 0.4441141,
    0.44423061, 0.44754412, 0.25445063, 0.25269969, 0.25256328, 0.25269969, 0.25445063, 0.22865051, 0.22577588, 0.22576584, 0.22577588,
    0.22865051, 0.01247869, 0.01243313, 0.0124329, 0.01243313, 0.01247869
])

expected_mean2_val = np.array([
       6.05038012, 6.05203312, 6.05740349, 6.06678599, 6.07476357,
       6.01882622, 6.01897548, 6.03213567, 6.05175902, 6.06783476,
       5.99448771, 5.9900754 , 6.00915574, 6.0385712 , 6.06196257,
       5.98936119, 5.98261162, 6.00397291, 6.03737643, 6.06315429,
       6.00193161, 5.99938411, 6.02020785, 6.04871292, 6.06918728,
       6.09185601, 6.09131879, 6.09194771, 6.09885189, 6.10509011,
       6.04100732, 6.03599848, 6.04699402, 6.06876176, 6.08768714,
       5.9944894 , 5.98124535, 6.00133423, 6.03867232, 6.06989231,
       5.97528619, 5.95858151, 5.98332094, 6.02819936, 6.06457289,
       5.98540668, 5.97636388, 6.00221803, 6.04110845, 6.07063416,
       6.03831991, 6.03995511, 6.04198814, 6.04381453, 6.04492453,
       6.03653664, 6.03850534, 6.04121568, 6.04373259, 6.0453278 ,
       6.03585644, 6.03785778, 6.04087809, 6.04378205, 6.04564755,
       6.03649567, 6.0383449 , 6.0412829 , 6.0441506 , 6.04595189,
       6.03785247, 6.03950322, 6.04205351, 6.04448084, 6.0459055 ,
       6.05399251, 6.05011787, 6.0474433 , 6.04751163, 6.04867778,
       6.04057905, 6.0345079 , 6.03384595, 6.03776643, 6.04257058,
       6.02643989, 6.01835115, 6.02023325, 6.02851144, 6.03703029,
       6.01918839, 6.01095201, 6.01474203, 6.02541052, 6.03564623,
       6.02134484, 6.0158285 , 6.02050869, 6.03013356, 6.03878703,
       6.17390659, 6.16299299, 6.14113316, 6.12443833, 6.11142054,
       6.13449029, 6.11319423, 6.09148766, 6.08163175, 6.07848532,
       6.07398292, 6.04592085, 6.02996836, 6.03077379, 6.03919819,
       6.02443479, 5.99709481, 5.98899907, 5.99821374, 6.01391571,
       6.00391356, 5.98410553, 5.98239677, 5.99365121, 6.00932539
])

expected_var2_val = np.array(
    [0.7988031 , 0.79874583, 0.79874076, 0.79875021, 0.79880265,
       0.79874926, 0.79869181, 0.79868713, 0.79869603, 0.79874857,
       0.79874108, 0.79868415, 0.79867959, 0.79868827, 0.79874031,
       0.79874767, 0.79869029, 0.79868563, 0.79869449, 0.79874697,
       0.79880328, 0.79874611, 0.79874107, 0.79875048, 0.79880284,
       0.79819774, 0.79808905, 0.79808031, 0.79809693, 0.798197  ,
       0.79809515, 0.79798707, 0.79797914, 0.79799456, 0.79809394,
       0.79808086, 0.79797388, 0.79796619, 0.7979812 , 0.7980795 ,
       0.79809229, 0.79798435, 0.79797647, 0.79799182, 0.79809106,
       0.79819805, 0.79808955, 0.79808085, 0.79809741, 0.79819731,
       0.79940923, 0.79940877, 0.7994087 , 0.79940882, 0.79940923,
       0.79940881, 0.79940832, 0.79940825, 0.79940837, 0.7994088 ,
       0.7994087 , 0.79940821, 0.79940814, 0.79940826, 0.79940869,
       0.79940879, 0.7994083 , 0.79940823, 0.79940835, 0.79940878,
       0.79940924, 0.79940877, 0.7994087 , 0.79940882, 0.79940923,
       0.79926866, 0.79925381, 0.79925221, 0.79925506, 0.79926852,
       0.79925481, 0.79923965, 0.79923811, 0.79924088, 0.7992546 ,
       0.79925231, 0.79923723, 0.79923572, 0.79923844, 0.79925208,
       0.79925436, 0.7992392 , 0.79923767, 0.79924044, 0.79925415,
       0.79926872, 0.79925389, 0.7992523 , 0.79925514, 0.79926858,
       0.79838619, 0.79829349, 0.79828599, 0.79830025, 0.79838555,
       0.79829872, 0.79820639, 0.79819957, 0.79821283, 0.79829769,
       0.79828645, 0.79819504, 0.79818841, 0.79820134, 0.79828529,
       0.79829627, 0.79820406, 0.79819727, 0.79821048, 0.79829522,
       0.79838645, 0.79829392, 0.79828645, 0.79830066, 0.79838582
    ])

                             #gp_emulator, covar expected_mean, expected_var
eval_gp_mean_var_val_list = [[gp_emulator1_e, False, expected_mean1_val, expected_var1_val],
                             [gp_emulator1_e, True, expected_mean1_val, expected_var1_val],
                             [gp_emulator2_e, False, expected_mean2_val, expected_var2_val]]
@pytest.mark.parametrize("gp_emulator, covar, expected_mean, expected_var", eval_gp_mean_var_val_list)
def test_eval_gp_mean_var_val(gp_emulator, covar, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val(covar) #Calc mean, var of gp 

    assert len(gp_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(gp_var, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(gp_var.shape) == 2 
        assert gp_var.shape[0] == gp_var.shape[1]
    
expected_mean1 = np.array([-4.09101293, -0.14592744,  0.03063563,  2.59794613, 12.26813236])
expected_var1 = np.array([0.14894872, 0.14791648, 0.14785142, 0.14791648, 0.14894872])
expected_mean2 = np.array([
       6.05038012, 6.05203312, 6.05740349, 6.06678599, 6.07476357,
       6.01882622, 6.01897548, 6.03213567, 6.05175902, 6.06783476,
       5.99448771, 5.9900754 , 6.00915574, 6.0385712 , 6.06196257,
       5.98936119, 5.98261162, 6.00397291, 6.03737643, 6.06315429,
       6.00193161, 5.99938411, 6.02020785, 6.04871292, 6.06918728
])
expected_var2 = np.array([
       0.7988031 , 0.79874583, 0.79874076, 0.79875021, 0.79880265,
       0.79874926, 0.79869181, 0.79868713, 0.79869603, 0.79874857,
       0.79874108, 0.79868415, 0.79867959, 0.79868827, 0.79874031,
       0.79874767, 0.79869029, 0.79868563, 0.79869449, 0.79874697,
       0.79880328, 0.79874611, 0.79874107, 0.79875048, 0.79880284
])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_mean_var_misc_list = [[gp_emulator1_e, False, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator1_e, True, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator2_e, False, simulator2, exp_data2, expected_mean2, expected_var2]]
@pytest.mark.parametrize("gp_emulator, covar, simulator, exp_data, expected_mean, expected_var", eval_gp_mean_var_misc_list)
def test_eval_gp_mean_var_misc_cand(gp_emulator, covar, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
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
expected_mean1_test_sse = np.array([73.98236105, 241.7185761])
expected_var1_test_sse = np.array([217.19806523, 345.59589243])
expected_mean2_test_sse = np.array([17.78391007])
expected_var2_test_sse = np.array([309.50112594])
                             #gp_emulator, exp_data, method, expected_mean, expected_var
eval_gp_sse_var_test_list = [[gp_emulator1_e, False, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator1_e, True, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator1_e, False, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator2_e, False, exp_data2, method, expected_mean2_test_sse, expected_var2_test_sse]]
@pytest.mark.parametrize("gp_emulator, covar, exp_data, method, expected_mean, expected_var", eval_gp_sse_var_test_list)
def test_eval_gp_sse_var_test(gp_emulator, covar, exp_data, method, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_emulator.test_data.gp_mean, gp_emulator.test_data.gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test(method, exp_data, covar) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(test_data.theta_vals) == len(sse_var)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2 
        assert sse_var.shape[0] == sse_var.shape[1]
    
#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_val_sse = np.array([
    147.81726003, 30.48218282, 116.1902601, 101.29578505, 41.44437525, 120.6621511, 5.74949159, 46.78644633, 107.83023785, 92.85527271
])
expected_var1_val_sse = np.array([
    112.78340684,  52.91127333, 249.39080888,  30.41774291, 4.7530246 , 306.86169299,   9.39637032,  63.64525288, 130.65995194, 6.12767308
])
expected_mean2_val_sse = np.array([17.26686418, 17.47243606, 17.7744707 , 17.45821545, 18.70787708])
expected_var2_val_sse = np.array([298.67733198, 303.13967843, 308.16645083, 300.63012781,325.36403632])
                             #gp_emulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_val_list = [[gp_emulator1_e, False, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
                            [gp_emulator1_e, True, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
                            [gp_emulator2_e, False, exp_data2, expected_mean2_val_sse, expected_var2_val_sse]]
@pytest.mark.parametrize("gp_emulator, covar, exp_data, expected_mean, expected_var", eval_gp_sse_var_val_list)
def test_eval_gp_sse_var_val(gp_emulator, covar, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_emulator.gp_val_data.gp_mean, gp_emulator.gp_val_data.gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val(method, exp_data, covar) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(gp_emulator.gp_val_data.theta_vals) == len(sse_var)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    #Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2 
        assert sse_var.shape[0] == sse_var.shape[1]

expected_mean1_sse = np.array([148.09705927])
expected_var1_sse = np.array([112.78340684])
expected_mean2_sse = np.array([17.26686418])
expected_var2_sse = np.array([298.67733198])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_misc_list = [[gp_emulator1_e, False, simulator1, exp_data1, expected_mean1_sse, expected_var1_sse],
                             [gp_emulator1_e, True, simulator1, exp_data1, expected_mean1_sse, expected_var1_sse],
                             [gp_emulator2_e, False, simulator2, exp_data2, expected_mean2_sse, expected_var2_sse]]
@pytest.mark.parametrize("gp_emulator, covar, simulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_misc_list)
def test_eval_gp_sse_var_misc_cand(gp_emulator, covar, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
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
    
    assert len(sse_mean)*mult_factor == len(misc_data.theta_vals) == len(sse_var)*mult_factor == len(sse_mean_cand)*mult_factor == len(sse_var_cand)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)
    
    #If covar is false, check variance values are correct
    if covar == False:
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
    gp_model = gp_emulator_fail.set_gp_model()
    gp_emulator_fail.train_gp(gp_model)
    
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