import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from scipy.stats import qmc

from bo_methods_lib.Parm_Sens_Multi_Theta import Param_Sens_Multi_Theta
from bo_methods_lib.bo_functions_generic import round_time, gen_theta_set, gen_x_set, find_train_doc_path, set_ep, clean_1D_arrays

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
# simplefilter("ignore", category=ConvergenceWarning)

#Set Date and Time
dateTimeObj = round_time()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
print("Date and Time Saved: ", DateTime)
# DateTime = None ##For Testing

#Set Parameters
#Need to run at a and b, need 2 arrays to test that this will work
CS = 2.2

Bound_Cut = True
denseX = True
eval_Train = True
eval_theta_num = [-4, -3, -2, -1]

Constants = np.array([[-200,-100,-170,15],
                      [-1,-1,-6.5,0.7],
                      [0,0,11,0.6],
                      [-10,-10,-6.5,0.7],
                      [1,0,-0.5,-1],
                      [0,0.5,1.5,1]])
if CS == 2.2:
    skip_param_types = 1 #This is what changes for subpoint
    true_p = Constants[skip_param_types:skip_param_types+2].flatten()
    param_dict = {0 : 'a_1', 1 : 'a_2', 2 : 'a_3', 3 : 'a_4',
                  4 : 'b_1', 5 : 'b_2', 6 : 'b_3', 7 : 'b_4'}
    exp_d = 2
    if Bound_Cut == True:
        bounds_x = np.array([[-1.0, 0.0],
                            [   0.5, 1.5]])
        if denseX == True:
            n = 30
        else:
            n = 25 #Number of experimental data points to use
    else:    
        bounds_x = np.array([[-1.5, -0.5],
                     [   1,    2]])
        n = 27 #Number of experimental data points to use
    bounds_p = np.array([[-2, -2, -10, -2, -2, -2,  5, -2],
                   [ 2,  2,   0,  2,  2,  2, 15,  2]])

else:
    Constants = true_p = np.array([1,-1])
    skip_param_types = 0
    param_dict = {0 : '\\theta_1', 1 : '\\theta_2'}
    exp_d = 1
    n = 5
    bounds_x = np.array([[-2], [2]])
    bounds_p = np.array([[-2, -2],
                         [ 2,  2]])

# print(Theta_True)
t_list = [20, 200, 600, 1000]
d = len(true_p)
kernel_func = "Mat_52"
pckg_list = "scikit_learn"
train_iter = 300
initialize = 10
noise_std = 0.01 #Numerically 0 (1e-4**2) or actual noise 0.01
outputscl = [False, True]
set_lengthscale = 1
verbose = False
norm = False

obj = "obj"

emulator =  True
save_figure = True
save_csvs = True

if Bound_Cut == True:
    cut_bounds = '_cut_bounds'
else:
    cut_bounds = ""
    
if denseX == True:
    dense = "_dense"
else:
    dense = ""

#Pull Experimental data from CSV
exp_data_doc = 'Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+cut_bounds+dense+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

Xexp = clean_1D_arrays(Xexp)

#Define parameter sensitivity analysis space
value_num = 101 #Number of parameter values for each parameter to evaluate within the bounds
x_space_points = [2,12,23,26]

print("Case Study: ", CS)
print("Number of Training Thetas: ", t)
print("Number of Experimental Data Points: ", n)
print("GP Emulating Function Output (T) or SSE (F)? ", emulator)
print("Scaling of Objective Function? ", obj)
print("Bounds On X Cut (T) or Normal (F)? ", Bound_Cut)
print("Dense Grid for Xexp?", denseX)
print("Evaluating Near Test Point (T) or True Parameter Set (F)? ", eval_Train)
print("GP Training Package: ", package)
print("GP Training Iterations (Gpytorch only): ", train_iter)
print("GP Kernel Function: ", kernel_func)
print("GP Kernel lengthscale: ", lenscl)
print("GP Training Restarts (when lengthscale/outputscale not set): ", initialize)
print("Training Data Noise st.dev: ", noise_std)
print("Percentiles: ", percentiles)
print("\n")

#Define GP Testing space
p=20
X_space = gen_x_set(LHS = False, n_points = p, dimensions = exp_d, bounds = bounds_x)

#Create empty list to store data values in for each number of training points
all_data_list = []

for t in t_list:
    t_use = int(t*n)
    all_data_doc = find_train_doc_path(emulator, obj, d, t_use, Bound_Cut, denseX)
    print("All Data Path: ", all_data_doc)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
    all_data_list.append(all_data)

for op_scl in outputscl:
    print("GP Kernel has outputscale?: ", op_scl)
    Param_Sens_Multi_Theta(all_data_list, x_space_points, eval_theta_num, Xexp, Yexp, Constants, true_p, CS, 
                           bounds_p, value_num, skip_param_types, kernel_func, set_lengthscale, 
                           outputscl, train_iter, initialize, noise_std, verbose, DateTime, save_csvs, 
                           save_figure, eval_Train, Bound_Cut, package = package)
    print("\n")