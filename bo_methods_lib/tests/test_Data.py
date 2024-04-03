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

cs_name1  = CS_name_enum(1)
cs_name2  = CS_name_enum(2)

num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
sep_fact = 0.8
normalize = False
noise_mean = 0
noise_std = 0.01
noise_std = 0
seed = 3

#Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1.value, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)

#Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2.value, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)

#This test function tests whether get_num_theta checker works correctly
                    #exp_data, expected
get_num_theta_list = [[exp_data1, 5],
                      [exp_data2, 5**2]] #Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x
@pytest.mark.parametrize("exp_data, expected", get_num_theta_list)
def test_get_num_theta(exp_data, expected):
    assert exp_data.get_num_theta() == expected

#This test function tests whether get_dim_theta checker works correctly
                    #exp_data, expected
get_dim_theta_list = [[exp_data1, 2],
                      [exp_data2, 4]]
@pytest.mark.parametrize("exp_data, expected", get_dim_theta_list)
def test_get_dim_theta(exp_data, expected):
    assert exp_data.get_dim_theta() == expected
    
#This test function tests whether get_num_x_vals checker works correctly
                    #exp_data, expected
get_num_x_vals_list = [[exp_data1, 5],
                       [exp_data2, 5**2]] #Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x
@pytest.mark.parametrize("exp_data, expected", get_num_x_vals_list)
def test_get_num_x_vals(exp_data, expected):
    assert exp_data.get_num_x_vals() == expected

#This test function tests whether get_dim_x_vals checker works correctly
                    #exp_data, expected
get_dim_x_vals_list = [[exp_data1, 1],
                       [exp_data2, 2]]
@pytest.mark.parametrize("exp_data, expected", get_dim_x_vals_list)
def test_get_dim_x_vals(exp_data, expected):
    assert exp_data.get_dim_x_vals() == expected
    
#This test function tests whether train_test_idx_split works correctly
#Add sep_fact to exp_data to pretend it's simulation data
exp_data1.sep_fact = sep_fact
exp_data2.sep_fact = sep_fact
                            #exp_data
train_test_idx_split_list = [exp_data1, exp_data2]
@pytest.mark.parametrize("exp_data", train_test_idx_split_list)
def test_train_test_idx_split(exp_data):
    train_idx, test_idx = exp_data.train_test_idx_split()
    union_set = set(train_idx).union(test_idx)
    assert (len(train_idx) + len(test_idx) == len(exp_data.get_unique_theta()))
    assert set(range(len(exp_data.get_unique_theta()))).issubset(union_set)

#This test function tests whether get_unique_theta() and get_unique_x() work correctly
                    #exp_data, expected
get_unique_theta_x =   [[exp_data1, 5, 1],
                       [exp_data2, 5**2, 1]] #Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x
@pytest.mark.parametrize("exp_data, expected_x, expected_theta", get_unique_theta_x)
def test_get_unique_vals(exp_data, expected_x, expected_theta):
    assert len(exp_data.get_unique_x()) == expected_x
    assert len(exp_data.get_unique_theta()) == expected_theta

#This test function tests whether get_unique_theta() and get_unique_x() throw correct errors
                    #exp_data, expected
get_unique_theta_x_err =   [["string", 5, 1]] #Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x
@pytest.mark.parametrize("exp_data, expected_x, expected_theta", get_unique_theta_x_err)
def test_get_unique_vals_err(exp_data, expected_x, expected_theta):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        assert len(exp_data.get_unique_x()) == expected_x
        assert len(exp_data.get_unique_theta()) == expected_theta
  