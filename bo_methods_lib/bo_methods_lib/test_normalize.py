import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
import torch
import csv
import gpytorch
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

from .bo_functions_generic import test_train_split as tt_split
from .bo_functions_generic import norm_unnorm, clean_1D_arrays, gen_theta_set, find_train_doc_path

from .normalize import normalize_x, normalize_p_data, normalize_p_bounds, normalize_p_set, normalize_p_true, normalize_constants, normalize_general

Theta_True = np.array([1,-1])
CS = 1
d = len(Theta_True)
noise_std = 0.1
exp_d = 1
n = 5
shuffle_seed = 9
exp_data_doc = '/scratch365/mcarlozo/Toy_Problem/Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

Xexp = clean_1D_arrays(Xexp)
m = Xexp.shape[1]
bounds_x = np.array([[-2],[2]])
# print(bounds_x.shape)

#Define GP Testing space
LHS = False
p=20
bounds_p = np.array([[-2,-2],
                     [2 , 2]])

theta_set = gen_theta_set(LHS = LHS, n_points = p, dimensions = d, bounds = bounds_p)
skip_params = 0

param_space = np.array([[1,-1,0], [1,-1,2]]) 
expected_y_data = np.array([0,6])

train_p = np.array([[1,-1], [1,-2], [0,0], [-1,1]])

@pytest.mark.parametrize("emulator, t, len_train, len_test", [
    (False, 20, 16, 4),
    (True, 100, 80, 20)
])

def test_test_train_split(emulator, t, len_train, len_test):
    all_data_doc = '/scratch365/mcarlozo/Toy_Problem/' + find_train_doc_path(emulator, "obj", d, t)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
#     train_data, test_data = test_train_split(all_data, sep_fact=0.8, runs = 0, shuffle_seed = 9)
    train_data, test_data = tt_split(all_data, sep_fact=0.8, runs = 0, shuffle_seed = 9)
    assert len(train_data) == len_train and len(test_data) == len_test, "Training and testing data incorrect lengths"


def test_normalize_x():
    bounds_x_scl, scaler_x = normalize_x(bounds_x, norm = True)
    X_norm = normalize_x(Xexp, None, True, scaler_x)[0]
    X_org = normalize_x(X_norm, None, False, scaler_x)[0]
    bounds_x_org = normalize_x(bounds_x_scl, None, False, scaler_x)[0]
    assert pytest.approx(Xexp, abs = 1e-2) == X_org, "Xexp not normalized correctly"
    assert pytest.approx(bounds_x, abs = 1e-2) == bounds_x_org, "X bounds not normalized correctly"

def test_normalize_p_bounds():
    bounds_p_scl, scaler_theta = normalize_p_bounds(bounds_p, norm = True, scaler = None)
    bounds_p_reg = normalize_p_bounds(bounds_p_scl, norm = False, scaler = scaler_theta)[0]
    assert pytest.approx(bounds_p, abs = 1e-2) == bounds_p_reg
    
bounds_p_scl, scaler_theta = normalize_p_bounds(bounds_p, norm = True, scaler = None)

def test_normalize_p_true():
    Theta_True_scl = normalize_p_true(Theta_True, scaler_theta, True)
    Theta_True_reg = normalize_p_true(Theta_True_scl, scaler_theta, False)
    assert pytest.approx(Theta_True, abs = 1e-2) == Theta_True_reg

def test_normalize_p_set():
    theta_set_scl = normalize_p_set(theta_set, scaler_theta, True)
    theta_set_reg = normalize_p_set(theta_set_scl, scaler_theta, False)
    assert pytest.approx(theta_set, abs = 1e-2) == theta_set_reg

@pytest.mark.parametrize("emulator, obj_func, t", [
    (False, "obj", 20),
    (False, "LN_obj", 20),
    (True, "obj", 100)
])

def test_normalize_p_data(emulator, obj_func, t):
    all_data_doc = '/scratch365/mcarlozo/Toy_Problem/' + find_train_doc_path(emulator, obj_func, d, t)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
    train_data, test_data = tt_split(all_data, runs = 0, sep_fact = 1, shuffle_seed=shuffle_seed)
#     train_data, test_data = test_train_split(all_data, runs = 0, sep_fact = 1, shuffle_seed=shuffle_seed)
    q = d
    if emulator == True:
        train_p = train_data[:,1:(q+m+1)]
        test_p = test_data[:,1:(q+m+1)]
    else:
        train_p = train_data[:,1:(q+1)]
        test_p = test_data[:,1:(q+1)]
    #Write test here
    train_p_scl = normalize_p_data(train_p, m, emulator, norm = True, scaler = scaler_theta)
    train_p_reg = normalize_p_data(train_p_scl, m, emulator, norm = False, scaler = scaler_theta)
    
    assert pytest.approx(train_p[:,0:q] , abs = 1e-2) == train_p_reg

CS_2 = 2.2
Constants_2 = np.array([[-200,-100,-170,15],
                      [-1,-1,-6.5,0.7],
                      [0,0,11,0.6],
                      [-10,-10,-6.5,0.7],
                      [1,0,-0.5,-1],
                      [0,0.5,1.5,1]])

Theta_True_2 = Constants_2[1:3].flatten()
skip_params_2 = 1

bounds_p_2 = np.array([[-2, -2, -10, -2, -2, -2,  5, -2],
                   [ 2,  2,   0,  2,  2,  2, 15,  2]])
bounds_p_2_scl, scaler_theta_2 = normalize_p_bounds(bounds_p_2, norm = True)

@pytest.mark.parametrize("Constants, p_true, skip_params, CS_use, scaler_thetas", [
    (Theta_True, Theta_True, skip_params, CS, scaler_theta),
    (Constants_2, Theta_True_2, skip_params_2, CS_2, scaler_theta_2)
])    

def test_normalize_constants(Constants, p_true, skip_params, CS_use, scaler_thetas):
    Constants_scl, scaler_C_before, scaler_C_after = normalize_constants(Constants, p_true, scaler_thetas, skip_params, CS_use, norm = True, scaler_C_before = None, scaler_C_after = None)
    Theta_True_scl = normalize_p_true(p_true, scaler_thetas, True)
    Constants_reg = normalize_constants(Constants_scl, Theta_True_scl, scaler_thetas, skip_params, CS_use, False, scaler_C_before, scaler_C_after)[0]
    assert pytest.approx(Constants, abs = 1e-2) == Constants_reg

@pytest.mark.parametrize("emulator, obj_func, d, n ", [
    (False, "obj", 2, 5),
    (False, "LN_obj", 2, 5),
    (True, "obj", 2, 5)
])    

def test_normalize_general(emulator, obj_func, d, n):
    #Find and split training/testing data
    t = 20
    if emulator == True:
        t = 20*n
    q = d
    all_data_doc = '/scratch365/mcarlozo/Toy_Problem/' + find_train_doc_path(emulator, obj_func, d, t)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
#     train_data, test_data = test_train_split(all_data, runs = 0, sep_fact = 1, shuffle_seed=shuffle_seed)
    train_data, test_data = tt_split(all_data, runs = 0, sep_fact = 1, shuffle_seed=shuffle_seed)
    if emulator == True:
        train_p = train_data[:,1:(q+m+1)]
        test_p = test_data[:,1:(q+m+1)]
    else:
        train_p = train_data[:,1:(q+1)]
        test_p = test_data[:,1:(q+1)]

    train_y = train_data[:,-1]

    norm_vals, norm_scalers = normalize_general(bounds_p, train_p, test_p, bounds_x, Xexp, theta_set, Theta_True, Theta_True, emulator, skip_params, CS)

    bounds_p_scl, train_p_scl, test_p_scl, bounds_x_scl, Xexp_scl, theta_set_scl, Theta_True_scl, Constants_scl = norm_vals
    scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers

    norm = False
    train_p_unscl = train_p_scl.copy()
    scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
    bounds_p_unscl = normalize_p_bounds(bounds_p_scl, norm, scaler = scaler_theta)[0] 
    if emulator == True:
        train_p_unscl[:,0:-m] = normalize_p_data(train_p_scl[:,0:-m], m, emulator, norm, scaler_theta)
        train_p_unscl[:,-m:] = normalize_x(Xexp, train_p_scl[:,-m:], norm, scaler_x)[0]
    else:
        train_p_unscl = normalize_p_data(train_p_scl, m, emulator, norm, scaler_theta)  
        
    bounds_x_unscl = normalize_x(bounds_x_scl, None, norm, scaler_x)[0]
    Xexp_unscl = normalize_x(Xexp_scl, None, norm, scaler_x)[0]
    theta_set_unscl = normalize_p_set(theta_set_scl, scaler_theta, norm)
    Theta_True_unscl =  normalize_p_true(Theta_True_scl, scaler_theta, norm)
    Constants_unscl = normalize_constants(Constants_scl, Theta_True_scl, scaler_theta, skip_params,CS, False,scaler_C_before, scaler_C_after)[0]

    assert pytest.approx(train_p, abs = 1e-2) == train_p_unscl, "Training data not scaled correctly"
    assert pytest.approx(bounds_p, abs = 1e-2) == bounds_p_unscl, "Param bounds not scaled correctly"
    assert pytest.approx(bounds_x, abs = 1e-2) == bounds_x_unscl, "X bounds not scaled correctly"
    assert pytest.approx(Xexp, abs = 1e-2) == Xexp_unscl, "Xexp not scaled correctly"
    assert pytest.approx(theta_set, abs = 1e-2) == theta_set_unscl, "Theta set not scaled correctly"
    assert pytest.approx(Theta_True, abs = 1e-2) == Theta_True_unscl, "Theta_true not scaled correctly"
    assert pytest.approx(Theta_True, abs = 1e-2) == Constants_unscl, "Constants not scaled correctly"