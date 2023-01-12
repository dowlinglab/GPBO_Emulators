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

#Note: COuld you isclose or all_close instead of pytest.approx()
from .CS2_create_data import calc_muller, calc_y_exp, create_sse_data_GP_val, create_sse_data, create_y_data, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point

CS = 2.2
minima = np.array([[-0.558,1.442],
                  [-0.050,0.467],
                  [0.623,0.028]])
Constants = np.array([[-200,-100,-170,15],
                      [-1,-1,-6.5,0.7],
                      [0,0,11,0.6],
                      [-10,-10,-6.5,0.7],
                      [1,0,-0.5,-1],
                      [0,0.5,1.5,1]])

Theta_True = Constants[1:3].flatten()
param_space_test = Theta_True
d = len(Theta_True)
noise_std = 0.1
exp_d = 2
n = 15
exp_data_doc = '/scratch365/mcarlozo/Toy_Problem/Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

bounds_x = np.array([[-1.5, -0.5],
                     [   1,    2]])
# print(Xexp)
# print(Yexp)
#Define GP Testing space
LHS = True
p_train = 20
p=20
bounds_p = np.array([[-2, -2, -10, -2, -2, -2,  5, -2],
                   [ 2,  2,   0,  2,  2,  2, 15,  2]])

theta_mesh = gen_theta_set(LHS = LHS, n_points = p, dimensions = d, bounds = bounds_p)
skip_params = 1


@pytest.mark.parametrize("X_val, Mul_pred", [
    (np.array([-0.558,1.442]), -146.6995),
    (np.array([-0.050,0.467]), -80.7677)
    (np.array([0.623,0.028]),-108.1667)
])

def test_calc_muller():
    mul_pot = calc_muller(x, Constants, noise = 0)
    assert pytest.approx(Mul_pred, abs = 1e-2) == mul_pot, "Muller potential minima not correct"

def test_calc_y_exp():
    y_exp = calc_y_exp(Constants, minima, noise_std, noise_mean=0,random_seed=9)
    assert pytest.approx(y_exp, abs = 1e-2) == np.array([-146.6994, -80.7676, -108.1665], "Yexp and y_exp are not the same values" 

@pytest.mark.parametrize("obj, sse_pred", [
    ("obj", np.array([0])),
    ("LN_obj", np.array([-20.1117]))
])

def test_create_sse_data_GP_val():
    sse = create_sse_data(param_space_test, Xexp, Yexp, Constants, obj, skip_param_types = 1)
    assert pytest.approx(sse_pred, abs = 0.1) == sse.flatten(), "SSE should be zero at the true parameter set" 

                                                         
#STOPPED HERE 1:37 pm 1/12/23
param_space = np.array([[-1,-1,-6.5,0.7, 0,0,11,0.6,-0.558,1.442],
                        [-1,-1,-6.5,0.7, 0,0,11,0.6,-0.050,0.467],
                        [-1,-1,-6.5,0.7, 0,0,11,0.6,0.623,0.028]])
expected_y_data = np.array([-146.6995, -80.7677,  -108.1667])

def test_create_y_data():
    y_data = create_y_data(param_space)
    assert pytest.approx(expected_y_data, atol = 1e-2) == y_data, "Minimum Ys should be gnerated when Theta_True is used for param values"

def test_gen_y_Theta_GP():
    #Note: Need to add test for norm_scalers w/parametrization once normalization code has tests
    y_data_GP = gen_y_Theta_GP(minima, Theta_True, Constants, Theta_True, CS, skip_params, norm_scalers = None, emulator = True)
    assert pytest.approx(expected_y_data, abs = 0.1) == y_data_GP

train_p = np.array([[-1,-1,-6.5,0.7, 0,0,11,0.6],
                   [-1,1,6.5,-0.7, 0,0,-11,-0.6],
                   [1, 2, 3, 0, 1, 3, 11 , 0]])

#Note, not tested w/ LN_obj because this function is always used with obj = "obj" and emulator = True
def test_eval_GP_emulator_BE(): 
    #Note: Need to add test for norm_scalers w/parametrization once normalization code has tests
    BE = eval_GP_emulator_BE(Xexp, Yexp, train_p, Theta_True, emulator=True, obj = "obj", skip_param_types = 1)
    assert pytest.approx(np.array([0]), abs = 1e-2) == BE.flatten(), "Best Error should be 0 when Theta_True is in train_p" 

train_p_org_std = np.array([[1,-2], [0,0], [-1,1]])
train_p_org_emul = np.array([[1,-2, 0], [0,0,1], [-1,1,2]])
train_y_org_std = np.array([34, 44, 176])
train_y_org_std_log = np.log(np.array([34, 44, 176]))
train_y_org_emul = np.array([0, 1, 10])

train_p_new_std = np.array([[1,-2], [0,0], [-1,1], [1,-1]])
train_p_new_emul = np.array([[1,-2, 0], [0,0,1], [-1,1,2], [1,-1,-2], [1,-1,-1], [1,-1,0], [1,-1,1], [1,-1,2]])
train_y_new_std = np.array([34, 44, 176, 0])
train_y_new_std_log = np.log(np.array([34, 44, 176, 0.000767]))
train_y_new_emul = np.array([0, 1, 10, -14, -3, 0, 1, 6])

theta_b = Theta_True

@pytest.mark.parametrize("emulator, obj, train_p_org, train_y_org, n_mult", [
    (False, "obj", train_p_org_std, train_y_org_std, 1),
    (False, "LN_obj", train_p_org_std, train_y_org_std_log, 1),
    (True, "obj", train_p_org_emul, train_y_org_emul, n)
])

# def test_make_next_point(emulator, train_p_org, train_y_org, train_p_new, train_y_new)
def test_make_next_point(emulator, train_p_org, train_y_org, n_mult):
    train_p_after, train_y_after = make_next_point(train_p_org, train_y_org, theta_b, Xexp, Yexp, emulator, Theta_True, obj, dim_param, skip_param_types=1, noise_std=None, norm_scalers = None)
    assert len(train_p_after) == len(train_p_org) + 1*n_mult , "train_p not updated correctly"
    assert len(train_y_after) == len(train_y_org) + 1*n_mult, "train_y not updated correctly"