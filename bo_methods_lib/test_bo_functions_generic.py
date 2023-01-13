import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
import torch
import csv
import scipy.optimize as optimize
import itertools
from itertools import combinations_with_replacement
from itertools import combinations
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import time
import Tasmanian
import pytest

from .bo_functions_generic import clean_1D_arrays, gen_theta_set, LHS_Design, ei_approx_ln_term, calc_ei_emulator, eval_GP_sparse_grid, calc_ei_basic 
Theta_True = np.array([1,-1])
d = len(Theta_True)
noise_std = 0.1
exp_d = 1
n = 5
exp_data_doc = '/scratch365/mcarlozo/Toy_Problem/Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

clean_array_1 = np.array([[1,2],[3,4]])
clean_array_2 = np.array([1,2])
@pytest.mark.parametrize("array, param_clean, array_expect", [
    (clean_array_1, False, clean_array_1),
    (clean_array_1, True, clean_array_1),
    (clean_array_2, False, clean_array_2.reshape(-1,1)),
    (clean_array_2, True, clean_array_2.reshape(1,-1))
])

def test_clean_1D_arrays(array, param_clean, array_expect):
    array_exp = clean_1D_arrays(array, param_clean)
    assert pytest.approx(array_expect , abs = 1e-2) == array_exp
    
#Note testing gen_theta_set and normalize_bounds together since this is the way these functions work practically
#Note: Not testing "True" case because it is covered in another test
LHS_exp1 = np.array([[-1., -1.], [-1.,  1.], [ 1., -1.], [ 1. , 1.]])
LHS_exp2 = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

@pytest.mark.parametrize("LHS, bounds, theta_set_expected", [
    (False, np.array([[-1,-1], [1,1]]), LHS_exp1),
    (False, np.array([[0, 0], [1,1]]), LHS_exp2),
    (False, None, LHS_exp2)
])    
def test_gen_theta_set(LHS, bounds, theta_set_expected):
    theta_set = gen_theta_set(LHS, n_points = 2, dimensions = 2, bounds = bounds)
    assert pytest.approx(theta_set , abs = 1e-2) == theta_set_expected
    
@pytest.mark.parametrize("bounds, expected_upper, expected_lower", [
    (None, 1, 0),
    (np.array([[0, 0], [1,1]]), 1, 0),
    (np.array([[-2, -2], [2,2]]), 2, -2)
])        
def test_LHS_Design(bounds, expected_upper, expected_lower):
    LHS = LHS_Design(100, 2, 9, bounds)
    assert np.amax(LHS) <= expected_upper and np.amin(LHS) >= expected_lower
    assert LHS.shape == (100,2)
    
def test_ei_approx_ln_term():
    ei_approx_ln = ei_approx_ln_term(0.1, 0.1, 1, 0.1, 0.12, 0)
    assert pytest.approx(-0.0462584 , abs = 1e-3) == ei_approx_ln
    
#Not testing true because it's usually not used     
def test_calc_ei_basic():
    ei_basic_components = calc_ei_basic(f_best =0.4,pred_mean=0.5,pred_var=0.25, explore_bias=1, verbose=False)
    assert pytest.approx(0.1534 , abs = 1e-2) == ei_basic_components

#Hard to calculate LN_obj by hand, but this function was checked a while back, works fine and hasn't changed since
def test_calc_ei_emulator():
    ei_emul = calc_ei_emulator(error_best=0.4,pred_mean=0.5,pred_var=0.25**2,y_target=0.45, explore_bias=torch.tensor(1), obj = "obj")
    assert pytest.approx(0.33653, abs = 1e-2) == ei_emul

#Tests sparse grid creater too
def test_eval_GP_sparse_grid():
    GP_mean = np.array([-12, -5, 0, 3, 7])
    GP_stdev = np.array([1.2, 1, 0.8, 0.9, 1.1])
    EI_Temp = eval_GP_sparse_grid(Xexp, Yexp, GP_mean, GP_stdev, 0.05 , ep = torch.tensor([1]), verbose = False)
    assert pytest.approx(0 , abs = 1e-2) == EI_Temp