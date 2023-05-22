import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc

import bo_methods_lib
from bo_methods_lib.bo_functions_generic import gen_theta_set, clean_1D_arrays
from bo_methods_lib.CS2_bo_functions_multi_dim import bo_iter_w_runs, find_train_doc_path, set_ep

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#Ignore warnings caused by "nan" values
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

#Ignore warning from scikit learn hp tuning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
#----------------------------------------------
CS = 1
Bound_Cut = True
denseX = True

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
# DateTime = None ##For Testing

#Set Parameters
#Need to run at a and b, need 2 arrays to test that this will work
if CS == 2.2:
    Bound_Cut = True
    denseX = True
    Constants = np.array([[-200,-100,-170,15],
                          [-1,-1,-6.5,0.7],
                          [0,0,11,0.6],
                          [-10,-10,-6.5,0.7],
                          [1,0,-0.5,-1],
                          [0,0.5,1.5,1]])

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
    Bound_Cut = False
    denseX = False
    Constants = true_p = np.array([1,-1])
    skip_param_types = 0
    param_dict = {0 : '\\theta_1', 1 : '\\theta_2'}
    exp_d = 1
    n = 5
    bounds_x = np.array([[-2], [2]])
    bounds_p = np.array([[-2, -2],
                         [ 2,  2]])

d = len(true_p)
BO_iters = 15
runs = 1
train_iter = 300
noise_std = 0.01
shuffle_seed = 9
sep_fact = np.linspace(0.1,1.0,10)
set_lengthscale = [1, None]
explore_bias = 1

eval_all_pairs = True
# eval_all_pairs = False
package = "scikit_learn"
kernel = "Mat_52"
outputscl = [False, True]
initialize = 5

obj = np.array(["LN_obj"])
emulator = np.array([False])
sparse_grid = np.array([False])
normalize = [False]
verbose = False
save_fig = True
save_CSV = True

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

#Define GP Testing space
LHS = True
p_train = 20
p=20
theta_mesh = gen_theta_set(LHS = LHS, n_points = p, dimensions = d, bounds = bounds_p, seed =1)

print("Case Study: ", CS)
print("Bounds On X Cut (T) or Normal (F)? ", Bound_Cut)
print("Dense Grid for Xexp?", denseX)
print("Number of Training Thetas: ", p_train)
print("Number of Experimental Data Points: ", n)
print("GP Training Package: ", package)
print("GP Training Iterations (Gpytorch only): ", train_iter)
print("GP Kernel Function: ", kernel)
print("GP Training Restarts (when lengthscale not set): ", initialize)
print("Training Data Noise st.dev: ", noise_std)
print("Runs:", runs)
print("BO Iterations:",BO_iters)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
for norm in normalize:
    print("Norm:", norm)
    for emul in emulator: 
        sys.stdout.flush()
        obj_use = obj
        print("-------------------")
        print("GP Emulating Function Output (T) or SSE (F)?", emul)
        if emul == True: #Change this based on number of TP for each test
            t = p_train*n
            sparse_grid_use = sparse_grid
        else:
            t = p_train
            sparse_grid_use = np.array([sparse_grid[0]]) #Sparse Grid will always be False for 2-Input

        for sparse in sparse_grid_use:
            #Can set ep to 1 for sparse grid if wanted
            if sparse == True:
                obj_use =  np.array(["obj"])
            else:
                obj_use =  obj
            print("______________________________")
            print("Sparse Grid?:", sparse)  

            for obj_func in obj_use:
                all_data_doc = find_train_doc_path(emul, obj_func, d, t, Bound_Cut, denseX)
                all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=",")) 
                print("Scaling of Objective Function? ", obj_func)
                print("-  -  -  -  -  -  -  -  -  -  -")
                for i in range(len(sep_fact)):
    #                 explore_bias = set_ep(emul, obj_func, sparse)
                    ep = torch.tensor([float(explore_bias)])
                    print("Separation Factor Train/Test:", str(np.round(sep_fact[i],3)))
                    for j in range(len(set_lengthscale)):
                        print("GP Kernel lengthscale: ", set_lengthscale[j])
                        print("GP Kernel Has Trained outputscale?: ", outputscl[j])
                        print("Explore Bias Multiplier:", str(np.round(float(ep),3)))
                        results = bo_iter_w_runs(BO_iters,all_data_doc,t,theta_mesh,true_p,train_iter,ep, Xexp, Yexp, noise_std, obj_func,
                                                 runs, sparse, emul, package, kernel, set_lengthscale[j], outputscl[j], initialize, 
                                                 Constants, param_dict, bounds_p, bounds_x, verbose, save_fig, save_CSV, shuffle_seed,
                                                 DateTime, sep_fact = sep_fact[i], LHS = LHS, skip_param_types =skip_param_types, 
                                                 eval_all_pairs = eval_all_pairs, normalize = norm,case_study = CS)
                        print("The GP predicts the lowest SSE of", "{:.3e}".format(np.exp(results[3])), "occurs at \u03B8 =", results[2], 
                                  "during run", results[1], "at BO iteration", results[0])
                        print("At this point, the highest EI occurs at \u03B8 =", results[4])
                        print("True p: ", true_p)
                        print("\n")