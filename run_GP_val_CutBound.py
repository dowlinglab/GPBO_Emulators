import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc

from bo_methods_lib.GP_Validation_CutBounds import LOO_Analysis
from bo_methods_lib.bo_functions_generic import gen_theta_set, find_train_doc_path, set_ep, clean_1D_arrays

import matplotlib as mpl

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
print("Date and Time Saved: ", DateTime)
# DateTime = None ##For Testing

#Set Parameters
#Need to run at a and b, need 2 arrays to test that this will work
CS = 2.2

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
    n = 25 #Number of experimental data points to use
    Bound_Cut = True
    bounds_x = np.array([[-1.0, 0.0],
                     [   0.5,    1.5]])
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
# t_list = np.array([20,40,100,200,300])
t_list = np.array([40])
d = len(true_p)
train_iter = 300
noise_std = 0.1
sep_fact = np.linspace(1,1,1)
set_lengthscale = None
explore_bias = 1
plot_axis = np.array([1,0])

# normalizing = np.array([False])
# normalizing = np.array([True])
normalizing = np.array([False,True])

obj = np.array(["obj", "LN_obj"])
# obj = np.array(["obj"])

emulator = np.array([False, True])
# emulator = np.array([True])
# emulator =  np.array([False])
save_figure = True
# save_figure = False

#Pull Experimental data from CSV
if Bound_Cut == True:
    cut_bounds = '_cut_bounds'
else:
    cut_bounds = ""
    
exp_data_doc = 'Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+cut_bounds+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]

Xexp = clean_1D_arrays(Xexp)

for t in t_list:
    for norm in normalizing:
        print("Norm:", norm)
        for emul in emulator:
            print("Emulator =", emul)
            if emul == False:
                t_use = int(t)
                obj_use = obj
            else:
                t_use = int(t*n)
                obj_use = np.array(["obj"])
            for obj_func in obj_use:
                print("Objective Function =", obj_func)
                all_data_doc = find_train_doc_path(emul, obj_func, d, t_use, bound_cut = Bound_Cut)
                all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
                LOO_Analysis(all_data, Xexp, Yexp, Constants, true_p, emul, obj_func, CS,  
                             skip_param_types = skip_param_types, noise_std = noise_std, DateTime = DateTime, 
                             save_figure= save_figure, plot_axis = plot_axis, normalize = norm, bounds_p = bounds_p, 
                             bounds_x = bounds_x)        