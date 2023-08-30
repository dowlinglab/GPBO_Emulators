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
noise_std = 0
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(5)) #1A

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, normalize, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, sim_data1, exp_data1, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(method, val_data1, exp_data1, sep_fact, True)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, normalize, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, sim_data2, exp_data2, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(method, val_data2, exp_data2, sep_fact, True)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#This test function tests whether get_num_gp_data checker works correctly
                    #emulator class, expected value
get_num_gp_data_list = [[gp_emulator1_e, 125],
                        [gp_emulator2_e, 6400]]
@pytest.mark.parametrize("gp_emulator, expected", get_num_gp_data_list)
def test_get_num_gp_data(gp_emulator, expected):
    assert gp_emulator.get_num_gp_data() == expected
                        
#This test function tests whether set_gp_model works correctly
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
set_gp_model_list = [[Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, 1, 1],
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
get_dim_gp_data_list = [[gp_emulator1_e, 3],
                        [gp_emulator2_e, 10]]
@pytest.mark.parametrize("gp_emulator, expected", get_dim_gp_data_list)
def test_get_dim_gp_data(gp_emulator, expected):
    assert gp_emulator.get_dim_gp_data() == expected

#This test function tests whether set_train_test_data checker works correctly
                            #gp emulator, cs_params
set_train_test_data_list = [gp_emulator1_e, gp_emulator2_e]
@pytest.mark.parametrize("gp_emulator", set_train_test_data_list)
def test_set_train_test_data(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    assert len(train_data.theta_vals) + len(test_data.theta_vals) == len(gp_emulator.gp_sim_data.theta_vals)
    

#This test function tests whether train_gp checker works correctly
                    #emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
#For time sake does not consider cs2 when testing
train_gp_list = [[Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, np.ones(3), 1],
                 [Type_2_GP_Emulator, sim_data1, val_data1, 2, 1, np.ones(3)*2, 1],
                 [Type_2_GP_Emulator, sim_data1, val_data1, 2, 2, np.ones(3)*2, 2]]
@pytest.mark.parametrize("gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", train_gp_list)
def test_train_gp(gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(sim_data, val_data, None,None,None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None,None,None,None)
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
    gp_emulator = gp_type(sim_data, val_data, None,None,None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None,None,None,None)
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model)
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all( 1e-5 - tol <= element <= 1e5 + tol for element in trained_lenscl )
    assert 1e-5 - tol <= trained_ops <= 1e2 + tol
    
#This test function tests whether calc_best_error checker works correctly
                            #gp emulator, exp_data, sim_sse_data
calc_best_error_list = [[gp_emulator1_e, exp_data1, sim_sse_data1],
                        [gp_emulator2_e, exp_data2, sim_sse_data2]]
@pytest.mark.parametrize("gp_emulator, exp_data, sim_sse_data", calc_best_error_list)
def test_calc_best_error(gp_emulator, exp_data, sim_sse_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error = gp_emulator.calc_best_error(exp_data)
    assert np.isclose(best_error, min(sim_sse_data.y_vals), rtol = 1e-6)
    
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
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error(exp_data) #Calc best error
    ei = gp_emulator.eval_ei_test(exp_data, ep_bias, best_error, method)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei)*num_x_data == len(gp_emulator.test_data.theta_vals)
    
                 #gp_emulator, exp_data, method
eval_ei_val_list = [[gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(3))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(4))],
                   [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(5))]]
@pytest.mark.parametrize("gp_emulator, exp_data, method", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data, method):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error(exp_data) #Calc best error
    ei = gp_emulator.eval_ei_val(exp_data, ep_bias, best_error, method)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei)*num_x_data == len(gp_emulator.gp_val_data.theta_vals)
    
    
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
    best_error = gp_emulator.calc_best_error(exp_data) #Calc best error
    ei = gp_emulator.eval_ei_cand(exp_data, ep_bias, best_error, method)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei)*num_x_data == len(gp_emulator.cand_data.theta_vals)
    
    
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
    best_error = gp_emulator.calc_best_error(exp_data) #Calc best error
    ei = gp_emulator.eval_ei_misc(misc_data, exp_data, ep_bias, best_error, method)
    #Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei)*num_x_data == len(misc_data.theta_vals)
    
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

#Define small case study
num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data1 = 10
num_theta_data2 = 5
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
method = GPBO_Methods(Method_name_enum(5)) #2C

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1, indecies_to_consider1, noise_mean, noise_std, normalize, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
val_data1 = simulator1.gen_sim_data(num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
gp_emulator1_e = Type_2_GP_Emulator(sim_data1, val_data1, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2, indecies_to_consider2, noise_mean, noise_std, normalize, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact)
val_data2 = simulator2.gen_sim_data(num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True)
gp_emulator2_e = Type_2_GP_Emulator(sim_data2, val_data2, None, None, None, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, None, None, None, None)
    
#This test function tests whether eval_gp_mean_var checker works correctly
expected_mean1_test = np.array([
    -6.19827597, -2.21283858, -0.04928079, 2.45392956, 7.40772254,
    -0.17640513, 0.89647328, 0.05546933, 2.36676682, 10.97876505
])
expected_var1_test = np.array([
    0.67392875, 0.66882892, 0.66878174, 0.66882892, 0.67392875,
    0.42719602, 0.42473066, 0.42466323, 0.42473066, 0.42719602
])
expected_mean2_test = np.array([
    -412.67259591, -366.50258467, -277.71554583, -178.92404052, -97.31817695,
    -312.67511083, -276.79594741, -209.5101477, -135.28478672, -74.2183414,
    -201.75991247, -178.5618176, -136.13094672, -89.66582848, -51.41773351,
    -113.34653057, -101.07824996, -78.98098986, -54.94340661, -35.05607031,
    -58.55766164, -53.56624918, -44.30804932, -34.28475556, -25.87798368
])
expected_var2_test = np.array([
    1.00009639, 1.000096, 1.0000959, 1.000096, 1.00009639,
    1.000096, 1.0000956, 1.0000955, 1.0000956, 1.000096,
    1.0000959, 1.0000955, 1.00009539, 1.0000955, 1.0000959,
    1.000096, 1.0000956, 1.0000955, 1.0000956, 1.000096,
    1.00009639, 1.000096, 1.0000959, 1.000096, 1.00009639
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
    -3.96634113e+00, -1.93155494e-01, 3.92182728e-02, 2.70703272e+00,
    1.20817920e+01, -9.77887130e+00, -1.85950869e+00, -6.54711129e-02,
    -3.24660068e-01, 2.80897932e+00, -3.58793021e+00, -8.94566548e-01,
    1.87633996e-01, 3.08127166e+00, 9.26512409e+00, -6.63629973e+00,
    -1.00029172e+00, 7.32817284e-03, 2.62458161e+00, 1.27570235e+01,
    -7.94985923e+00, -7.30270554e-01, 2.11714851e-03, 2.61664024e-01,
    6.03972615e+00, -3.30137243e+00, -1.25305529e-01, 1.07481246e-01,
    7.89824679e-01, 4.32552424e+00, -1.21829101e+01, -3.12563315e+00,
    -1.47484277e-03, 1.34114226e+00, 5.93520103e+00, -9.15165225e+00,
    -1.03437928e+00, -2.87538732e-02, -1.08113798e+00, 2.04174626e+00,
    -5.20898074e+00, -1.29529772e+00, 1.11764021e-01, 3.25305236e+00,
    1.11223872e+01, -1.07692174e+01, -8.44014209e-01, -8.53716232e-03,
    -2.41516184e+00, -2.09272908e+00
])
expected_var1_val = np.array([
    0.26641528, 0.2647776, 0.26470579, 0.2647776, 0.26641528,
    0.36721775, 0.36458752, 0.3645194, 0.36458752, 0.36721775,
    0.63457698, 0.63028091, 0.63023327, 0.63028091, 0.63457698,
    0.06599062, 0.06586324, 0.06585793, 0.06586324, 0.06599062,
    0.0334669, 0.03340724, 0.03340308, 0.03340724, 0.0334669,
    0.61248436, 0.60729495, 0.60725478, 0.60729495, 0.61248436,
    0.44059212, 0.4386921, 0.43860923, 0.4386921, 0.44059212,
    0.29323482, 0.2919453, 0.29188343, 0.2919453, 0.29323482,
    0.3421327, 0.33919872, 0.33916261, 0.33919872, 0.3421327,
    0.01326301, 0.01324496, 0.0132448, 0.01324496, 0.01326301
])

expected_mean2_val = np.array([
    -3.69303089e+01, -3.43785895e+01, -2.78073786e+01, -1.93821436e+01,
    -1.15604437e+01, -2.83374402e+01, -2.63613641e+01, -2.12263202e+01,
    -1.45941287e+01, -8.42443286e+00, -1.84159458e+01, -1.70536473e+01,
    -1.34909438e+01, -8.80898421e+00, -4.44619679e+00, -9.62853678e+00,
    -8.72708550e+00, -6.38828928e+00, -3.21497827e+00, -2.98820240e-01,
    -3.37617154e+00, -2.72015568e+00, -1.12193691e+00, 1.11171498e+00,
    3.06201983e+00, -4.20625313e+03, -3.60028123e+03, -2.50863152e+03,
    -1.41056900e+03, -6.20063301e+02, -2.96272197e+03, -2.51752882e+03,
    -1.74223546e+03, -9.69046546e+02, -4.12891623e+02, -1.68848123e+03,
    -1.42552106e+03, -9.84046222e+02, -5.44035457e+02, -2.23367277e+02,
    -8.07329228e+02, -6.78734697e+02, -4.68601310e+02, -2.56366470e+02,
    -9.72055549e+01, -3.60533400e+02, -3.04332945e+02, -2.10413498e+02,
    -1.11484278e+02, -3.40198869e+01, -3.38761631e+01, -3.04416147e+01,
    -2.39622301e+01, -1.64397879e+01, -9.73490770e+00, -2.68239013e+01,
    -2.39960810e+01, -1.87779871e+01, -1.27607280e+01, -7.41896965e+00,
    -1.85090471e+01, -1.64575321e+01, -1.27708490e+01, -8.54600047e+00,
    -4.80630342e+00, -1.11477552e+01, -9.81205457e+00, -7.48853945e+00,
    -4.84342999e+00, -2.51248315e+00, -5.88950022e+00, -5.08106896e+00,
    -3.73005897e+00, -2.20535754e+00, -8.76596014e-01, -2.60124936e+02,
    -2.32809723e+02, -1.79230452e+02, -1.18496682e+02, -6.71396232e+01,
    -1.99733850e+02, -1.78400090e+02, -1.37557239e+02, -9.17014483e+01,
    -5.31075264e+01, -1.31992181e+02, -1.18198258e+02, -9.23719876e+01,
    -6.36650422e+01, -3.95224641e+01, -7.69677355e+01, -6.99204308e+01,
    -5.68010549e+01, -4.24088233e+01, -3.02402298e+01, -4.19297678e+01,
    -3.95872550e+01, -3.48378106e+01, -2.97471136e+01, -2.53159214e+01,
    2.62573322e+01, 2.71052885e+01, 3.10536345e+01, 3.88123144e+01,
    4.75169562e+01, 6.84504588e+01, 6.73971912e+01, 7.49978230e+01,
    9.47600149e+01, 1.18786133e+02, 1.80713139e+02, 1.80603652e+02,
    2.00692513e+02, 2.51552462e+02, 3.08499211e+02, 3.82455569e+02,
    3.91228795e+02, 4.36928335e+02, 5.42518588e+02, 6.48789325e+02,
    6.06123609e+02, 6.34623954e+02, 7.12978546e+02, 8.74459125e+02,
    1.02089465e+03
])
expected_var2_val = np.array([
    1.00008719, 1.00008589, 1.00008557, 1.00008589, 1.00008719,
    1.00008589, 1.00008453, 1.00008421, 1.00008453, 1.00008589,
    1.00008557, 1.00008421, 1.00008389, 1.00008421, 1.00008557,
    1.00008589, 1.00008453, 1.00008421, 1.00008453, 1.00008589,
    1.00008719, 1.00008589, 1.00008557, 1.00008589, 1.00008719,
    1.00000324, 0.99999412, 0.99999219, 0.99999412, 1.00000324,
    0.99999412, 0.99998478, 0.99998288, 0.99998478, 0.99999412,
    0.99999219, 0.99998288, 0.999981, 0.99998288, 0.99999219,
    0.99999412, 0.99998478, 0.99998288, 0.99998478, 0.99999412,
    1.00000324, 0.99999412, 0.99999219, 0.99999412, 1.00000324,
    1.00009995, 1.00009994, 1.00009994, 1.00009994, 1.00009995,
    1.00009994, 1.00009993, 1.00009993, 1.00009993, 1.00009994,
    1.00009994, 1.00009993, 1.00009993, 1.00009993, 1.00009994,
    1.00009994, 1.00009993, 1.00009993, 1.00009993, 1.00009994,
    1.00009995, 1.00009994, 1.00009994, 1.00009994, 1.00009995,
    1.00009989, 1.00009988, 1.00009988, 1.00009988, 1.00009989,
    1.00009988, 1.00009987, 1.00009986, 1.00009987, 1.00009988,
    1.00009988, 1.00009986, 1.00009986, 1.00009986, 1.00009988,
    1.00009988, 1.00009987, 1.00009986, 1.00009987, 1.00009988,
    1.00009989, 1.00009988, 1.00009988, 1.00009988, 1.00009989,
    0.99973751, 0.99970615, 0.99970046, 0.99970615, 0.99973751,
    0.99970615, 0.99967464, 0.99966921, 0.99967464, 0.99970615,
    0.99970046, 0.99966921, 0.99966389, 0.99966921, 0.99970046,
    0.99970615, 0.99967464, 0.99966921, 0.99967464, 0.99970615,
    0.99973751, 0.99970615, 0.99970046, 0.99970615, 0.99973751
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
    
expected_mean1 = np.array([-3.96634113, -0.19315549,  0.03921827,  2.70703272, 12.08179199])
expected_var1 = np.array([0.26641528, 0.2647776,  0.26470579, 0.2647776,  0.26641528])
expected_mean2 = np.array([
    -36.93030887, -34.37858952, -27.80737858, -19.38214356, -11.56044371,
    -28.3374402, -26.36136407, -21.22632016, -14.59412871, -8.42443286,
    -18.41594578, -17.05364728, -13.49094379, -8.80898421, -4.44619679,
    -9.62853678, -8.7270855, -6.38828928, -3.21497827, -0.29882024,
    -3.37617154, -2.72015568, -1.12193691, 1.11171498, 3.06201983
])
expected_var2 = np.array([
    1.00008719, 1.00008589, 1.00008557, 1.00008589, 1.00008719,
    1.00008589, 1.00008453, 1.00008421, 1.00008453, 1.00008589,
    1.00008557, 1.00008421, 1.00008389, 1.00008421, 1.00008557,
    1.00008589, 1.00008453, 1.00008421, 1.00008453, 1.00008589,
    1.00008719, 1.00008589, 1.00008557, 1.00008589, 1.00008719
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

#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_test_sse = np.array([65.34755447, 232.47635892])
expected_var1_test_sse = np.array([15.3409606, 20.5801417])
expected_mean2_test_sse = np.array([4198532.10241006])
expected_var2_test_sse = np.array([-12768.34763098])
                             #gp_emulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_test_list = [[gp_emulator1_e, exp_data1, expected_mean1_test_sse, expected_var1_test_sse],
                              [gp_emulator2_e, exp_data2, expected_mean2_test_sse, expected_var2_test_sse]]
@pytest.mark.parametrize("gp_emulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_test_list)
def test_eval_gp_sse_var_test(gp_emulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_emulator.test_data.gp_mean, gp_emulator.test_data.gp_var = gp_emulator.eval_gp_mean_var_test() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test(exp_data) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(test_data.theta_vals) == len(sse_var)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)
    
#This test function tests whether eval_gp_sse_var checker works correctly
expected_mean1_val_sse = np.array([
    148.09705927, 30.96372618, 127.55013595, 106.22335144, 42.11782753,
    125.29782846, 3.38655573, 47.26048286, 111.21227011, 92.2305308
])
expected_var1_val_sse = np.array([
    10.99636609, 0.57237519, 22.86886745, 2.34167966, 0.50989066,
    14.41751719, 1.72935713, 0.43609671, 12.27918108, -0.16262392
])
expected_mean2_val_sse = np.array([3089941.63524772, 71489867.45474209, 3106795.59862653, 3660414.45898299, 2804455.57688683])
expected_var2_val_sse = np.array([-6214.3944027, -63857.74441299, -6207.2523478, -10393.46744796, 10364.33372449])
                             #gp_emulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_val_list = [[gp_emulator1_e, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
                              [gp_emulator2_e, exp_data2, expected_mean2_val_sse, expected_var2_val_sse]]
@pytest.mark.parametrize("gp_emulator, exp_data, expected_mean, expected_var", eval_gp_sse_var_val_list)
def test_eval_gp_sse_var_val(gp_emulator, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_model = gp_emulator.set_gp_model()
    gp_emulator.train_gp(gp_model) 
    gp_emulator.gp_val_data.gp_mean, gp_emulator.gp_val_data.gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val(exp_data) #Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(gp_emulator.gp_val_data.theta_vals) == len(sse_var)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)

expected_mean1_sse = np.array([148.09705927])
expected_var1_sse = np.array([10.99636609])
expected_mean2_sse = np.array([3089941.63524772])
expected_var2_sse = np.array([-6214.3944027])

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
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_misc(misc_data, exp_data) #Calc mean, var of gp sse
    
    candidate = Data(None, exp_data.x_vals, None,None,None,None,None,None, simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(1,-1) #Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate #Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data) #Set feature vals
    gp_emulator.cand_data.gp_mean, gp_emulator.cand_data.gp_var = gp_emulator.eval_gp_mean_var_cand() #Calc mean, var of gp 
    sse_mean_cand, sse_var_cand = gp_emulator.eval_gp_sse_var_cand(exp_data) #Calc mean, var of gp sse

    mult_factor = exp_data.get_num_x_vals()
    
    assert len(sse_mean)*mult_factor == len(misc_data.theta_vals) == len(sse_var)*mult_factor == len(sse_mean_cand)*mult_factor == len(sse_var_cand)*mult_factor
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var, expected_var, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)
    assert np.allclose(sse_var_cand, expected_var, rtol=1e-02)
    
    
#Test that errors get raised if you don't put the correct data into these functions
#Need test for add_next_theta_to_train_data(theta_best_data)