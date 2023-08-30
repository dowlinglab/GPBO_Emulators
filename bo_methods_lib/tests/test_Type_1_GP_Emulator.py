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
                            #gp emulator
set_train_test_data_list = [gp_emulator1_s, gp_emulator1_e, gp_emulator2_s, gp_emulator2_e]
@pytest.mark.parametrize("gp_emulator", set_train_test_data_list)
def test_set_train_test_data(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    assert len(train_data.theta_vals) + len(test_data.theta_vals) == len(gp_emulator.gp_sim_data.theta_vals)
    

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
    
#This test function tests whether calc_best_error checker works correctly
                            #gp emulator
calc_best_error_list = [gp_emulator1_s, gp_emulator2_s]
@pytest.mark.parametrize("gp_emulator", calc_best_error_list)
def test_calc_best_error(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error = gp_emulator.calc_best_error()
    assert np.isclose(best_error, min(train_data.y_vals), rtol = 1e-6)
    
#This test function tests whether eval_gp_ei works correctly
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
    best_error = gp_emulator.calc_best_error() #Calc best error
    ei = gp_emulator.eval_ei_test(exp_data, ep_bias, best_error)
    
    assert len(ei) == len(gp_emulator.test_data.theta_vals)
    
                 #gp_emulator, exp_data, expected_len
eval_ei_val_list = [[gp_emulator1_s, exp_data1],
                   [gp_emulator2_s, exp_data2]]
@pytest.mark.parametrize("gp_emulator, exp_data", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data):
    gp_model = gp_emulator.set_gp_model()#Set model
    gp_emulator.train_gp(gp_model) #Train model    
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val() #Calc mean, var of gp 
    best_error = gp_emulator.calc_best_error() #Calc best error
    ei = gp_emulator.eval_ei_val(exp_data, ep_bias, best_error)
    
    assert len(ei) == len(gp_emulator.gp_val_data.theta_vals)

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
    best_error = gp_emulator.calc_best_error() #Calc best error
    ei = gp_emulator.eval_ei_cand(exp_data, ep_bias, best_error)
    
    assert len(ei) == len(gp_emulator.cand_data.theta_vals)
    
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
    best_error = gp_emulator.calc_best_error() #Calc best error
    ei = gp_emulator.eval_ei_misc(misc_data, exp_data, ep_bias, best_error)
    
    assert len(ei) == len(misc_data.theta_vals)
    
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
expected_mean1_test = np.array([109.86605125, 142.86310602, 107.12050241, 9.5246127])
expected_var1_test = np.array([0.39072597, 0.12714247, 0.23999174, 0.26761744])
expected_mean2_test = np.array([1.09616022e+12, 1.00703173e+12, 5.78004244e+10, 2.14920635e+09])
expected_var2_test = np.array([1.00009055, 1.0000184, 1.00009929, 1.00009913])
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
    75.3460122, 244.69819064, 233.55052228, 15.91201763, 140.9495282,
    106.67860777, 13.77534612, 185.11151249, 50.19062414, 7.84458439,
    28.28233413, 70.37626277, 105.22554349, 16.98927078, 6.88380843,
    151.34998606, 8.27437681, 68.07934596, 236.37295065, 92.63967388
])
expected_var1_val = np.array([
    0.44208099, 0.10684535, 0.14387059, 0.12657987, 0.34553957,
    0.10260152, 0.58958937, 0.22827154, 0.02040543, 0.32400625,
    0.01954792, 0.07092983, 0.27665981, 0.04301436, 0.83232255,
    0.21223588, 0.30111541, 0.08190393, 0.18944042, 0.00883387
])

expected_mean2_val = np.array([
    6.38925047e+09, 8.43311020e+09, 2.55419487e+11, 3.93540197e+11,
    2.32822199e+10, 5.96842025e+13, 7.18153772e+09, 1.25074877e+11,
    1.92250057e+11, 1.07584469e+11, 9.48061765e+10, 1.69827445e+09,
    3.28189573e+13, 2.87840175e+10, 9.21652161e+10, 1.59283264e+11,
    1.31744137e+10, 9.61169596e+09, 1.60572337e+11, 1.29709996e+11
])
expected_var2_val = np.array([
    0.99790284, 0.99976308, 1.00009311, 0.99986484, 0.99972573,
    0.99137021, 0.99962927, 0.99740074, 0.99983385, 0.99953015,
    1.0000976,  1.00009855, 0.99756636, 1.00009986, 0.99967851,
    1.0000972,  1.00002411, 0.99981033, 0.99674959, 0.99957733
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
expected_var1 = np.array([0.00029971])
expected_mean2 = np.array([7632080.35503387])
expected_var2 = np.array([0.00029996])

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
    
#Test that errors get raised if you don't put the correct data into these functions
#Need test for add_next_theta_to_train_data(theta_best_sse_data)