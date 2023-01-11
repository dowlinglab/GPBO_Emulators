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

from .bo_functions_generic import clean_1D_arrays, normalize_bounds, gen_theta_set, LHS_Design, test_train_split, ei_approx_ln_term, calc_ei_emulator, eval_GP_sparse_grid, calc_ei_basic 
Theta_True = np.array([1,-1])
d = len(Theta_True)
noise_std = 0.1
exp_d = 1
n = 5
exp_data_doc = '/scratch365/mcarlozo/Toy_Problem/Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

param_space = np.array([[1,-1,0], [1,-1,2]]) 
expected_y_data = np.array([0,6])

train_p = np.array([[1,-1], [1,-2], [0,0], [-1,1]])

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