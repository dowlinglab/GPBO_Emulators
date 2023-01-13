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

from .CS2_bo_functions_multi_dim import argmax_multiple, find_opt_and_best_arg, eval_GP_emulator_set, eval_GP_basic_set, eval_GP_scipy, eval_GP  
#How to write tests for if scipy works correctly (optimize_theta_set)? How to write unit tests for bo_iter and bo_iter_w_runs?
#How to write unit test for: eval_GP_emulator_set, eval_GP_basic_set, eval_GP_scipy, eval_GP: Need a simple model and there's no way to know in advance what the GP will output
Theta_True = np.array([1,-1])
d = len(Theta_True)
noise_std = 0.1
exp_d = 1
n = 5
exp_data_doc = '/scratch365/mcarlozo/Toy_Problem/Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]


argmax_org= np.array([0,1,2,3])
train_p = np.array([[-1.5, -1.5], [-1.6, -1.6], [-1.7, -1.7], [-1.8, -1.8]])
theta_set = np.array([[1.5, 1.5], [1.6, 1.6], [1.7, 1.7], [1.8, 1.8]])
sse_test_1 = np.array([0,0,1,4])
sse_test_2 = np.array([0,1,2,3])
ei_test_1 = np.array([4,4,1,0])
ei_test_2 = np.array([3,2,1,0])

def test_argmax_multiple():
    argmax = argmax_multiple(argmax_org, train_p, theta_set)
    assert pytest.approx(np.array([3]) , abs = 1e-2) == argmax

@pytest.mark.parametrize("sse, ei, sse_exp_1, sse_exp_2, ei_exp_1, ei_exp_2", [
    (sse_test_1, ei_test_1, np.array([[1.5, 1.5]]), np.array([[1.6, 1.6]]), np.array([[1.5, 1.5]]), np.array([[1.6, 1.6]]) ),
    (sse_test_2, ei_test_2, np.array([[1.5, 1.5]]), np.array([[1.5, 1.5]]), np.array([[1.5, 1.5]]), np.array([[1.5, 1.5]]) )
])

def test_find_opt_and_best_arg(sse, ei, sse_exp_1, sse_exp_2, ei_exp_1, ei_exp_2):
    theta_b_0, theta_o_0 = find_opt_and_best_arg(theta_set, sse, ei, train_p)
    assert pytest.approx(theta_b_0 , abs = 1e-2) == ei_exp_1 or pytest.approx(theta_b_0 , abs = 1e-2) == ei_exp_2, "Theta_b incorrect"
    assert pytest.approx(theta_o_0 , abs = 1e-2) == sse_exp_1 or pytest.approx(theta_b_0 , abs = 1e-2) == sse_exp_2, "Theta_o incorrect"

# def test_eval_GP_emulator_set():    