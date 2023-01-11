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

from .bo_functions_generic import test_train_split, norm_unnorm

from .normalize import normalize_x, normalize_p_data, normalize_p_bounds, normalize_p_set, normalize_p_true, normalize_constants, normalize_general

Theta_True = np.array([1,-1])
d = len(Theta_True)
noise_std = 0.1
exp_d = 1
n = 5
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

param_space = np.array([[1,-1,0], [1,-1,2]]) 
expected_y_data = np.array([0,6])

train_p = np.array([[1,-1], [1,-2], [0,0], [-1,1]])

@pytest.mark.parametrize("emulator, t, len_train, len_test", [
    (False, 20, 16, 4),
    (True, 100, 80, 20)
])

def test_test_train_split():
    all_data_doc = find_train_doc_path(emulator, obj_func, d, t)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
    train_data, test_data = test_train_split(all_data, sep_fact=0.8, runs = 0, shuffle_seed)
    assert len(train_data) == len_train and len(test_data) == len_test, "Training and testing data incorrect lengths"


def test_normalize_x():
    bounds_x_scl, scaler_x = normalize_x(bounds_x, True)
    X_norm = normalize_x(Xexp, None, True, scaler_x)[0]
    X_org = normalize_x(X_norm, None, False, scaler_x)[0]
    bounds_x_org = normalize_x(bounds_x, None, False, scaler_x)[0]
    assert pytest.approx(Xexp, abs = 1e-2) == X_norm
    assert pytest.approx(bounds_x, abs = 1e-2) == bounds_x_org

def test_normalize_p_bounds():
    bounds_p_scl, scaler_theta = normalize_p_bounds(bounds_p, norm = True, scaler = None)
    bounds_p_reg = normalize_p_bounds(bounds_p_scl, norm = False, scaler = scaler_theta)
    assert pytest.approx(bounds_p, abs = 1e-2) == bounds_p_reg
    
bounds_p_scl, scaler_theta = normalize_p_bounds(bounds_p, norm = True, scaler = None)

def test_normalize_p_true():
    Theta_True_scl = normalize_p_true(Theta_True, scaler_theta, True)
    Theta_True_reg = normalize_p_true(Theta_True_scl, scaler_theta, False)
    assert pytest.approx(Theta_True, abs = 1e-2) == Theta_True_reg

def test_normalize_p_set():
    theta_set_scl = normalize_p_set(theta_set, scaler_theta, True)
    theta_set_reg = normalize_p_set(theta_set_scl, scaler_theta, True)
    assert pytest.approx(theta_set, abs = 1e-2) == theta_set_reg

@pytest.mark.parametrize("emulator, obj_func, t, train_p_compare", [
    (False, "obj", 20, train_p_unscl[:,0:-m]),
    (False, "LN_obj", 20),
    (True, "obj", 100)
])

def test_normalize_p_data(emulator, obj_func, t):
    all_data_doc = find_train_doc_path(emulator, obj_func, d, t)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
    train_data, test_data = test_train_split(all_data, runs = 0, sep_fact = 1, shuffle_seed=shuffle_seed)
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
    
#Write tests for normalize_constants & normalize_general
    