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
    print(gp_emulator.feature_test_data)
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
calc_best_error_err_list = [[gp_emulator1_e, simulator1, exp_data1, ep_bias, 1, method, None],
                            [gp_emulator1_e, simulator1, None, ep_bias, 1, method, True],
                            [gp_emulator1_e, simulator1, exp_data1, ep_bias, 1, GPBO_Methods(Method_name_enum(1)), True],
                            [gp_emulator1_e, simulator1, exp_data1, ep_bias, 1, "str", True],
                            [gp_emulator1_e, simulator1, exp_data1, None, 1, method, True],
                            [gp_emulator2_e, simulator2, exp_data2, None, 1, method, True],
                            [gp_emulator2_e, simulator2, exp_data2, ep_bias, 1, None, True]]
                                
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, ep_bias, best_error, method, data", calc_best_error_err_list)
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
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error, method)
    with pytest.raises((AssertionError, AttributeError, ValueError)):        
        ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error, method)
        
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
    946.76837845, 941.41316789, 941.57227279, 945.46118642, 950.03203404, 949.72422723, 945.31917045, 944.84504391, 947.02029813, 
    949.92714433, 953.55778956, 950.04090153, 948.61048944, 948.54631417, 949.25791276, 957.08453628, 953.83456312, 951.1950442, 
    948.84282129, 947.31726465, 959.99857884, 956.76357079, 953.1418915, 949.08673027, 945.9199195
])

expected_var2_test = np.array([
    1.35737097, 1.35736639, 1.35736579, 1.35736682, 1.35737092, 1.35736674, 1.35736198, 1.35736138, 1.35736241, 1.35736667, 1.35736583, 
    1.35736106, 1.35736047, 1.35736149, 1.35736575, 1.35736659, 1.35736182, 1.35736122, 1.35736225, 1.35736651, 1.35737099, 1.35736642, 
    1.35736582, 1.35736685, 1.35737094
])
                             #gp_emulator, expected_mean, expected_var
eval_gp_mean_var_test_list = [[gp_emulator1_e, expected_mean1_test, expected_var1_test],
                              [gp_emulator2_e, expected_mean2_test, expected_var2_test]]
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
    653.00104023, 675.11491553, 754.03000519, 832.93770111, 887.12765806, 745.75190155, 757.51106716, 807.2204481, 858.46645965, 
    895.73289461, 840.50286295, 841.10424407, 862.69244247, 888.13624794, 909.79816287, 904.30588038, 897.39342577, 903.41091011, 
    915.89819405, 929.71812743, 940.14204562, 931.27717484, 933.00135737, 942.82331435, 954.46786827, 370.25352663, 437.71531275, 
    615.01949805, 773.68254355, 874.46347615, 601.94879621, 636.32363983, 747.16427353, 859.32953657, 941.78966495, 875.84897788, 
    881.08817845, 943.28037767, 1036.56225177, 1123.59155849, 1154.03287202, 1147.96963411, 1206.71891277, 1336.54263997, 1463.66773711, 
    1392.6434455, 1397.03096858, 1480.54861191, 1665.17419545, 1832.35346219, 960.25929144, 958.51929884, 958.36450378, 959.45903813, 
    960.96748851, 962.36125727, 961.03674087, 961.0808913, 962.23174777, 963.71127807, 966.29503849, 965.66719191, 966.15804269, 
    967.58241416, 969.10357586, 971.70239683, 971.93695155, 973.10317352, 975.05415687, 976.70859482, 976.88290829, 978.00063883, 
    979.85659586, 982.30410229, 984.00337517, 721.40589926, 751.68137157, 819.16548737, 880.40168045, 920.30388626, 796.49823054, 
    815.68101469, 860.06868777, 901.4576097, 929.76541452, 874.18351191, 881.95427489, 903.95347007, 926.84461332, 944.68401121, 
    926.61647234, 927.72302297, 937.6774284, 951.60700111, 964.65655453, 954.18549201, 953.71078483, 960.64624497, 973.10847215, 
    985.28287743, 500.22107796, 566.85614809, 713.62190276, 843.75158936, 929.28373726, 706.8751627, 735.84632913, 834.07144224, 
    945.73147854, 1042.01401015, 1057.90366091, 1043.83577046, 1117.27920577, 1275.61426303, 1448.49201551, 1615.59713538, 1569.40456802, 
    1676.37131277, 2004.43781143, 2350.40456386, 2217.55957641, 2179.28143515, 2358.29875159, 2882.98051492, 3387.3062875
])

expected_var2_val = np.array(
    [1.35646647, 1.35637689, 1.3563688, 1.35638381, 1.35646577, 1.35638232, 1.3562923, 1.3562848, 1.35629898, 1.35638123, 1.35636931,
     1.35628006, 1.35627276, 1.3562866, 1.35636809, 1.35637981, 1.35628989, 1.35628242, 1.35629654, 1.3563787, 1.35646676, 1.35637734,
     1.35636928, 1.35638425, 1.35646606, 1.35538367, 1.35520212, 1.35518748, 1.35521529, 1.35538242, 1.35521232, 1.35503175, 1.35501846,
     1.35504428, 1.3552103, 1.35518839, 1.35500965, 1.35499676, 1.35502191, 1.35518612, 1.35520754, 1.35502721, 1.35501399, 1.3550397, 
     1.35520548, 1.35538419, 1.35520296, 1.35518839, 1.3552161, 1.35538295, 1.35740555, 1.35740485, 1.35740474, 1.35740492, 1.35740554, 
     1.35740491, 1.35740416, 1.35740405, 1.35740424, 1.35740489, 1.35740474, 1.35740399, 1.35740388, 1.35740407, 1.35740473, 1.35740488, 
     1.35740414, 1.35740402, 1.35740421, 1.35740487, 1.35740555, 1.35740485, 1.35740474, 1.35740492, 1.35740554, 1.35717538, 1.35715104, 
     1.35714841, 1.35715311, 1.35717515, 1.35715269, 1.35712783, 1.3571253, 1.35712986, 1.35715235, 1.35714858, 1.35712384, 1.35712136, 
     1.35712584, 1.35714821, 1.35715195, 1.3571271, 1.35712458, 1.35712913, 1.35715161, 1.35717548, 1.35715118, 1.35714856, 1.35715324, 
     1.35717525, 1.35552408, 1.3553542, 1.35534054, 1.35536655, 1.35552291, 1.35536375, 1.35519466, 1.35518226, 1.35520641, 1.35536187, 
     1.35534138, 1.355174, 1.35516196, 1.35518549, 1.35533927, 1.35535927, 1.3551904, 1.35517807, 1.35520212, 1.35535736, 1.35552456, 
     1.35535498, 1.35534138, 1.3553673, 1.3555234
    ])

                             #gp_emulator, expected_mean, expected_var
eval_gp_mean_var_val_list = [[gp_emulator1_e, expected_mean1_val, expected_var1_val],
                              [gp_emulator2_e, expected_mean2_val, expected_var2_val]]
@pytest.mark.parametrize("gp_emulator, expected_mean, expected_var", eval_gp_mean_var_val_list)
def test_eval_gp_mean_var_val(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 

    assert len(gp_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    
expected_mean1 = np.array([-4.09101293, -0.14592744,  0.03063563,  2.59794613, 12.26813236])
expected_var1 = np.array([0.14894872, 0.14791648, 0.14785142, 0.14791648, 0.14894872])
expected_mean2 = np.array([
    653.00104023, 675.11491553, 754.03000519, 832.93770111, 887.12765806, 745.75190155, 757.51106716, 807.2204481, 858.46645965,
    895.73289461, 840.50286295, 841.10424407, 862.69244247, 888.13624794, 909.79816287, 904.30588038, 897.39342577, 903.41091011,
    915.89819405, 929.71812743, 940.14204562, 931.27717484, 933.00135737, 942.82331435, 954.46786827
])
expected_var2 = np.array([
    1.35646647, 1.35637689, 1.3563688, 1.35638381, 1.35646577, 1.35638232, 1.3562923, 1.3562848, 1.35629898, 1.35638123, 1.35636931, 
    1.35628006, 1.35627276, 1.3562866, 1.35636809, 1.35637981, 1.35628989, 1.35628242, 1.35629654, 1.3563787, 1.35646676, 1.35637734, 
    1.35636928, 1.35638425, 1.35646606
])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_mean_var_misc_list = [[gp_emulator1_e, simulator1, exp_data1, expected_mean1, expected_var1],
                              [gp_emulator2_e, simulator2, exp_data2, expected_mean2, expected_var2]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, expected_mean, expected_var", eval_gp_mean_var_misc_list)
def test_eval_gp_mean_var_misc_cand(gp_emulator, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
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
        
#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_test_sse = np.array([73.98236105, 241.7185761])
expected_var1_test_sse = np.array([217.19806523, 345.59589243])
expected_mean2_test_sse = np.array([20335676.71874536])
expected_var2_test_sse = np.array([7.56132822e+08])
                             #gp_emulator, exp_data, method, expected_mean, expected_var
eval_gp_sse_var_test_list = [[gp_emulator1_e, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator1_e, exp_data1, method, expected_mean1_test_sse, expected_var1_test_sse],
                             [gp_emulator2_e, exp_data2, method, expected_mean2_test_sse, expected_var2_test_sse]]
@pytest.mark.parametrize("gp_emulator, exp_data, method, expected_mean, expected_var", eval_gp_sse_var_test_list)
def test_eval_gp_sse_var_test(gp_emulator, exp_data, method, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_emulator.test_data.gp_mean, gp_emulator.test_data.gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test(method, exp_data) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(test_data.theta_vals) == len(sse_var)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)
    
#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_val_sse = np.array([
    147.81726003, 30.48218282, 116.1902601, 101.29578505, 41.44437525, 120.6621511, 5.74949159, 46.78644633, 107.83023785, 92.85527271
])
expected_var1_val_sse = np.array([
    112.78340684,  52.91127333, 249.39080888,  30.41774291, 4.7530246 , 306.86169299,   9.39637032,  63.64525288, 130.65995194, 6.12767308
])
expected_mean2_val_sse = np.array([16488667.09932764, 23926537.8891193, 21066297.06571797, 18022075.39720873, 52649112.95833728])
expected_var2_val_sse = np.array([6.05317495e+08, 8.94804049e+08, 7.87682558e+08, 6.69138828e+08, 1.86135075e+09])
                             #gp_emulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_val_list = [[gp_emulator1_e, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
                              [gp_emulator2_e, exp_data2, expected_mean2_val_sse, expected_var2_val_sse]]
@pytest.mark.parametrize("gp_emulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_val_list)
def test_eval_gp_sse_var_val(gp_emulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_emulator.gp_val_data.gp_mean, gp_emulator.gp_val_data.gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val(method, exp_data) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(gp_emulator.gp_val_data.theta_vals) == len(sse_var)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)

expected_mean1_sse = np.array([148.09705927])
expected_var1_sse = np.array([112.78340684])
expected_mean2_sse = np.array([16488667.09932764])
expected_var2_sse = np.array([6.05317495e+08])

                             #gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_misc_list = [[gp_emulator1_e, simulator1, exp_data1, expected_mean1_sse, expected_var1_sse],
                              [gp_emulator2_e, simulator2, exp_data2, expected_mean2_sse, expected_var2_sse]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_misc_list)
def test_eval_gp_sse_var_misc_cand(gp_emulator, simulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    misc_data = Data(None, exp_data.x_vals, None, None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    misc_data.theta_vals = theta_vals #Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data) #Set feature vals
    misc_data.gp_mean, misc_data.gp_var = gp_emulator.eval_gp_mean_var_misc(misc_data, feature_misc_data) #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_misc(misc_data, method, exp_data) #Calc mean, var of gp sse
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_emulator.cand_data.gp_mean, gp_emulator.cand_data.gp_var = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    sse_mean_cand, sse_var_cand = gp_emulator.eval_gp_sse_var_cand(method, exp_data) #Calc mean, var of gp sse

    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(misc_data.theta_vals) == len(sse_var)*mult_factor == len(sse_mean_cand)*mult_factor == len(sse_var_cand)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var_cand, expected_var, rtol=1e-02)

#This function tests whether eval_gp_sse_var_test/val/cand/misc throw the correct errors     
                             #gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method
eval_gp_sse_var_err_list = [[gp_emulator1_e, simulator1, exp_data1, False, True, True, method],
                            [gp_emulator1_e, simulator1, exp_data1, True, False, True, method],
                            [gp_emulator1_e, simulator1, exp_data1, True, True, False, method],
                            [gp_emulator1_e, simulator1, exp_data1, True, True, False, None]]
@pytest.mark.parametrize("gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method", eval_gp_sse_var_err_list)
def test_eval_gp_sse_var_err(gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method):
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
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val(method, exp_data)        
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test() 
        if set_data is False:
            gp_emulator_fail.test_data = None
        if set_gp_mean is False:
            gp_emulator_fail.test_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.test_data.gp_var = None
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test(method, exp_data)
    with pytest.raises((AssertionError, AttributeError, ValueError)):  
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand() 
        if set_data is False:
            gp_emulator_fail.cand_data = None
        if set_gp_mean is False:
            gp_emulator_fail.cand_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.cand_data.gp_var = None
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
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data, method, exp_data)
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(misc_data, feat_misc_data) #Calc mean, var of gp 
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data, method, None)

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