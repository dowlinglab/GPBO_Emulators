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
from .CS1_create_data import calc_y_exp, create_sse_data_GP_val, create_sse_data, create_y_data, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point

Theta_True = np.array([1,-1])
d = len(Theta_True)
noise_std = float(0.1)
exp_d = 1
n = 5
exp_data_doc = '/scratch365/mcarlozo/Toy_Problem/Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

def test_calc_y_exp():
    y_exp = calc_y_exp(Theta_True, Xexp, noise_std, noise_mean=0,random_seed=9)
    assert pytest.approx(Yexp.flatten(), abs = 0.1) == y_exp.flatten(), "Yexp and y_exp are not the same values" 

@pytest.mark.parametrize("obj, sse_pred", [
    ("obj", np.array([0])),
    ("LN_obj", np.array([-7.2]))
])

def test_create_sse_data_GP_val(obj, sse_pred):
    q = 2
    sse = create_sse_data_GP_val(q,Theta_True.reshape(1,-1), Xexp, Yexp, obj)
    assert pytest.approx(sse_pred, abs = 0.1) == sse.flatten(), "SSE should be zero at the true parameter set" 

param_space = np.array([[1,-1,0], [1,-1,2]]) 
expected_y_data = np.array([0,6])

def test_create_y_data():
    y_data = create_y_data(param_space)
    assert pytest.approx(expected_y_data, abs = 0.1) == y_data

def test_gen_y_Theta_GP():
    #Note: Need to add test for norm_scalers w/parametrization once normalization code has tests
    y_data_GP = gen_y_Theta_GP(Xexp, Theta_True, Theta_True, skip_param_types = 0, norm_scalers = None, emulator = False)
    assert pytest.approx(Yexp, abs = 0.1) == y_data_GP, "Given Xexp and true theta, you should get roughly the experimental data"

train_p = np.array([[1,-1], [1,-2], [0,0], [-1,1]])

#Note, not tested w/ LN_obj because this function is always used with obj = "obj" and emulator = True
def test_eval_GP_emulator_BE(): 
    #Note: Need to add test for norm_scalers w/parametrization once normalization code has tests
    BE = eval_GP_emulator_BE(Xexp, Yexp, train_p, Theta_True, emulator=True, obj = "obj", skip_param_types = 0, norm_scalers=None)
    assert pytest.approx(np.array([0]), abs = 0.1) == BE.flatten(), "Best Error should be 0 when Theta_True in train_p" 

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
def test_make_next_point(emulator, obj, train_p_org, train_y_org, n_mult):
    train_p_after, train_y_after = make_next_point(train_p_org, train_y_org, theta_b, Xexp, Yexp, emulator, Theta_True, obj, 2, skip_param_types=0, noise_std  = noise_std, norm_scalers = None)
    assert len(train_p_after) == len(train_p_org) + 1*n_mult , "train_p not updated correctly"
    assert len(train_y_after) == len(train_y_org) + 1*n_mult, "train_y not updated correctly"