import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from scipy.stats import qmc

# from bo_methods_lib.GP_Vs_True import Compare_GP_True
from bo_methods_lib.GP_Vs_True_Sensitivity import Compare_GP_True_Movie
# from bo_methods_lib.GP_Validation import LOO_Analysis
from bo_methods_lib.bo_functions_generic import round_time, gen_theta_set,gen_x_set, find_train_doc_path, set_ep, clean_1D_arrays

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

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
eval_Train = True

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
t_list = [200]
d = len(true_p)
train_iter = 300
noise_std = 0.1
set_lengthscale = None
emulator = True
obj = "obj"
verbose = False

save_figure = True
save_csvs = True

if Bound_Cut == True:
    cut_bounds = '_cut_bounds'
else:
    cut_bounds = ""

#Pull Experimental data from CSV
exp_data_doc = 'Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+cut_bounds+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

Xexp = clean_1D_arrays(Xexp)
percentiles = np.linspace(-1.0,1.0,41)

print("Case Study: ", CS)
print("Number of Training Thetas: ", t_list[0])
print("Number of Experimental Data Points: ", n)
print("GP Emulating Function Output (T) or SSE (F)? ", emulator)
print("Scaling of Objective Function? ", obj)
print("Bounds On X Cut (T) or Normal (F)? ", Bound_Cut)
print("Evaluating Near Test Point (T) or True Parameter Set (F)? ", eval_Train)
print("GP RBF Kernel lengthscale: ", set_lengthscale)
print("GP Training Iterations: ", train_iter)
print("Training Data Noise st.dev: ", noise_std)
print("Percentiles: ", percentiles)

#Define GP Testing space
p=20
# print(bounds_x)
X_space = gen_x_set(LHS = False, n_points = p, dimensions = exp_d, bounds = bounds_x)
# print(X_space[-5:,:])

for t in t_list:
    t_use = int(t*n)
    obj = "obj"
    all_data_doc = find_train_doc_path(emulator, obj, d, t_use, bound_cut = Bound_Cut)
    print("All Data Path: ", all_data_doc)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
#     Compare_GP_True(all_data, X_space, Xexp, Yexp, Constants, true_p, obj, CS, 
#                 skip_param_types, set_lengthscale, train_iter, noise_std, verbose, DateTime, 
#                 save_figure, eval_Train = eval_Train)
    Compare_GP_True_Movie(all_data, X_space, Xexp, Yexp, Constants, true_p, CS, bounds_p, percentiles,
                skip_param_types, set_lengthscale, train_iter, noise_std, verbose, DateTime, 
                save_csvs, save_figure, eval_Train = eval_Train, CutBounds = Bound_Cut)