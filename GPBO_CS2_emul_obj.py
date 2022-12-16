import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc

import bo_methods_lib
from bo_methods_lib.bo_functions_generic import gen_theta_set
from bo_methods_lib.CS2_bo_functions_multi_dim import bo_iter_w_runs, find_train_doc_path, set_ep

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#----------------------------------------------
CS = 2.2

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
# DateTime = None ##For Testing

#Set Parameters
#Need to run at a and b, need 2 arrays to test that this will work
Constants = np.array([[-200,-100,-170,15],
                      [-1,-1,-6.5,0.7],
                      [0,0,11,0.6],
                      [-10,-10,-6.5,0.7],
                      [1,0,-0.5,-1],
                      [0,0.5,1.5,1]])

Theta_True = Constants[1:3].flatten()
# print(Theta_True.shape)
param_dict = {0 : 'a_1', 1 : 'a_2', 2 : 'a_3', 3 : 'a_4',
              4 : 'b_1', 5 : 'b_2', 6 : 'b_3', 7 : 'b_4'}

# print(Theta_True)

d = len(Theta_True)
BO_iters = 10
runs = 3
train_iter = 300
noise_std = 0.1
shuffle_seed = 9
sep_fact = np.linspace(1,1,1)
set_lengthscale = None
explore_bias = 1

skip_param_types = 1
eval_all_pairs = True
# eval_all_pairs = False

obj = np.array(["obj"])
# obj = np.array(["obj","LN_obj"])
# emulator = False
# emulator = np.array([False])
emulator = np.array([True])
# sparse_grid = np.array([True])
sparse_grid = np.array([False])
normalize = np.array([False,True])
# sparse_grid = np.array([False,True])
verbose = False
# verbose = True
save_fig = True
# save_fig = False

#Pull Experimental data from CSV
exp_d = 2
n = 15 #Number of experimental data points to use
exp_data_doc = 'Input_CSVs/Exp_Data/d='+str(exp_d)+'/n='+str(n)+'.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1:exp_d+1]
Yexp = exp_data[:,-1]
bounds_x = np.array([[-1.5, -0.5],
                     [   1,    2]])
# print(Xexp)
# print(Yexp)
#Define GP Testing space
LHS = True
p=20
bounds_p = np.array([[-2, -2, -10, -2, -2, -2,  5, -2],
                   [ 2,  2,   0,  2,  2,  2, 15,  2]])
# Theta1 =  np.linspace(-2,2,p) #1x10
# Theta2 =  np.linspace(-2,2,p) #1x10
# theta_mesh = np.array(np.meshgrid(Theta1, Theta2)) #2 Uniform 5x5 arrays 
theta_mesh = gen_theta_set(LHS = LHS, n_points = p, dimensions = d, bounds = bounds_p)
# print(theta_mesh.shape)

print("Runs:", runs)
print("BO Iterations:",BO_iters)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
for norm in normalize:
    for emul in emulator: 
        sys.stdout.flush()
        obj_use = obj
        print("-------------------")
        print("Emulator?:", emul)
        if emul == True: #Change this based on number of TP for each test
            t = 300
            sparse_grid_use = sparse_grid
        else:
            t = 20
            sparse_grid_use = np.array([sparse_grid[0]]) #Sparse Grid will always be False for 2-Input

        for sparse in sparse_grid_use:
    #         #Can set ep to 1 for sparse grid if wanted
            if sparse == True:
                obj_use =  np.array(["obj"])
            else:
                obj_use =  obj
    #             ep_use = torch.tensor([1]) 
    #         else:
    #             ep_use = explore_bias
            print("______________________________")
            print("Sparse Grid?:", sparse)  

            for obj_func in obj_use:
                all_data_doc = find_train_doc_path(emul, obj_func, d, t)
                all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=",")) 
                print("Objective Function:", obj_func)
                print("-  -  -  -  -  -  -  -  -  -  -")
                for i in range(len(sep_fact)):
    #                 explore_bias = set_ep(emul, obj_func, sparse)
                    ep = torch.tensor([float(explore_bias)])
                    print("Separation Factor Train/Test:", str(np.round(sep_fact[i],3)))
                    print("Lengthscale Set To:", set_lengthscale)
                    print("Explore Bias Multiplier:", str(np.round(float(ep),3)))
                    results = bo_iter_w_runs(BO_iters,all_data_doc,t,theta_mesh,Theta_True,train_iter,ep, Xexp, Yexp,
                                                     noise_std, obj_func, runs, sparse, emul, set_lengthscale, Constants, 
                                                     param_dict, bounds_p, bounds_x, verbose, save_fig, shuffle_seed, DateTime, 
                                                     sep_fact = sep_fact[i], LHS = LHS, skip_param_types = skip_param_types, 
                                                     eval_all_pairs = eval_all_pairs, normalize = norm, case_study = CS)
    #                 results = bo_iter_w_runs(BO_iters,all_data_doc,t,theta_mesh,Theta_True,train_iter,ep, Xexp, Yexp,
    #                                              noise_std, obj_func, runs, sparse, emul, set_lengthscale, Constants, 
    #                                              param_dict, verbose, save_fig, shuffle_seed, DateTime, sep_fact = sep_fact[i], 
    #                                              LHS = LHS, skip_param_types = skip_param_types, eval_all_pairs = eval_all_pairs)
                    print("The GP predicts the lowest SSE of", "{:.3e}".format(np.exp(results[3])), "occurs at \u03B8 =", results[2], 
                              "during run", results[1], "at BO iteration", results[0])
                    print("At this point, the highest EI occurs at \u03B8 =", results[4])
                    print("\n")