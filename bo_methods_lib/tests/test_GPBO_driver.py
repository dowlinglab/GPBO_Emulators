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

CS_name  = "Simple Linear"

num_x_data = 5
gen_meth_x = Gen_meth_enum(2) #Note: Has to be the same for validation and sim data
num_theta_data = 20
gen_meth_theta = Gen_meth_enum(1)
gen_heat_map_data = True
num_theta_data_val = 20
gen_meth_theta_val = Gen_meth_enum(2)

ep0 = 1
ep_enum = Ep_enum(1)
sep_fact = 1.0
normalize = True
noise_mean = 0
noise_std = 0.01
# noise_std = 0.0
kernel = Kernel_enum(1)
lenscl = None
outputscl = 1
retrain_GP = 1
reoptimize_obj = 1
bo_iter_tot = 2
bo_run_tot = 2
save_data = False
seed = 1
ei_tol = 1e-6
obj_tol = 1e-4
DateTime = None

method1 = GPBO_Methods(Method_name_enum(1)) #1A
method2 = GPBO_Methods(Method_name_enum(2)) #1B
method3 = GPBO_Methods(Method_name_enum(3)) #2A
method4 = GPBO_Methods(Method_name_enum(4)) #2B
method5 = GPBO_Methods(Method_name_enum(5)) #2C
method6 = GPBO_Methods(Method_name_enum(6)) #2D
method7 = GPBO_Methods(Method_name_enum(7)) #3A

#This test function tests whether run_bo_restarts,  works correctly
                    #method expected value
run_bo_restarts_list = [method1, method2, method3, method4, method5, method6, method7]
@pytest.mark.parametrize("method", run_bo_restarts_list)
def test_run_bo_restarts(method):
    #Define cs_params, simulator, and exp_data for CS1
    simulator = simulator_helper_test_fxns(CS_name.value, noise_mean, noise_std, seed)
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)
    sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, False)
    sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, False)
    val_data = simulator.gen_sim_data(num_theta_data_val, num_x_data, gen_meth_theta_val, gen_meth_x, sep_fact, True)
    val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, sep_fact, True)

    #Set Ep_Bias
    ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)

    #Set Cs_params and Simulator
    cs_name = "test"
    cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, 
                                    reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, 
                                    seed, ei_tol, obj_tol)

    #Initialize Driver
    driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, 
                         ep_bias, gen_meth_theta)
    
    gpbo_res_simple, gpbo_res_GP = driver.run_bo_restarts()
    one_run_bo_results = gpbo_res_simple[0]
    one_run_gp_results = gpbo_res_GP[0]
    
    assert len(gpbo_res_simple) == len(gpbo_res_GP) == bo_run_tot
    assert isinstance(one_run_bo_results.configuration, dict)
    assert len(one_run_bo_results.configuration.keys()) == 21
    assert isinstance(one_run_bo_results.simulator_class, Simulator)
    assert isinstance(one_run_bo_results.why_term, str)
    assert isinstance(one_run_bo_results.exp_data_class, Data)
    assert isinstance(one_run_bo_results.results_df, pd.DataFrame) 
    assert isinstance(one_run_gp_results.max_ei_details_df, pd.DataFrame) or one_run_gp_results.max_ei_details_df is None
    assert len(one_run_bo_results.results_df) == bo_iter_tot
    assert isinstance(one_run_gp_results.list_gp_emulator_class, list)
    if method.emulator == False:
        assert all(isinstance(var, Type_1_GP_Emulator) for var in one_run_gp_results.list_gp_emulator_class)
    else:
        assert all(isinstance(var, Type_2_GP_Emulator) for var in one_run_gp_results.list_gp_emulator_class)        
    assert isinstance(one_run_gp_results.heat_map_data_dict, dict)    
    
    
#This test function tests whether create_heat_map_param_data,  works correctly
                    #method expected value
create_heat_map_param_data_list = [method1, method2, method3, method4, method5, method6, method7]
@pytest.mark.parametrize("method", create_heat_map_param_data_list)
def test_create_heat_map_param_data(method):
    #Define cs_params, simulator, and exp_data for CS1
    simulator = simulator_helper_test_fxns(CS_name.value, noise_mean, noise_std, seed)
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)
    sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, False)
    sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, False)
    val_data = simulator.gen_sim_data(num_theta_data_val, num_x_data, gen_meth_theta_val, gen_meth_x, sep_fact, True)
    val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, sep_fact, True)

    #Set Ep_Bias
    ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)

    #Set Cs_params and Simulator
    cs_name = "test"
    cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, 
                                    reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, 
                                    seed, ei_tol, obj_tol)

    #Initialize Driver
    driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, 
                         ep_bias, gen_meth_theta)
    
    #Initialize gp_emualtor class
    gp_emulator = driver._GPBO_Driver__gen_emulator()
    driver.gp_emulator = gp_emulator
    
    hm_data_dict = driver.create_heat_map_param_data()
    assert len(hm_data_dict.keys()) == 1
    
    
theta_array = np.array([1,2])
#This test function tests whether create_data_instance_from_theta,  works correctly
                    #method expected value
create_data_instance_from_theta_list = [method1, method2, method3, method4, method5, method6, method7]
@pytest.mark.parametrize("method", create_data_instance_from_theta_list)
def test_create_data_instance_from_theta(method):
    #Define cs_params, simulator, and exp_data for CS1
    simulator = simulator_helper_test_fxns(CS_name.value, noise_mean, noise_std, seed)
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)
    sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, False)
    sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, False)
    val_data = simulator.gen_sim_data(num_theta_data_val, num_x_data, gen_meth_theta_val, gen_meth_x, sep_fact, True)
    val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, sep_fact, True)

    #Set Ep_Bias
    ep_bias = Exploration_Bias(ep0, None, ep_enum, None, None, None, None, None, None, None)

    #Set Cs_params and Simulator
    cs_name = "test"
    cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, 
                                    reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, 
                                    seed, ei_tol, obj_tol)

    #Initialize Driver
    driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, 
                         ep_bias, gen_meth_theta)
    
    data = driver.create_data_instance_from_theta(theta_array)
    assert isinstance(data, Data)
    if method.emulator == False:
        assert len(data.theta_vals) == 1
    else:
        assert len(data.theta_vals) == exp_data.get_num_x_vals()