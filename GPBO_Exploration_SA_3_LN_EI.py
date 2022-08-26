import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from bo_functions import bo_iter_w_runs
from bo_functions import find_train_doc_path

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#----------------------------------------------

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
# DateTime = None ##For Testing

#Set Parameters
Theta_True = np.array([1,-1])
BO_iters = 100
runs = 1
train_iter = 300
noise_std = 0.1
shuffle_seed = 9
sep_fact = 1
explore_bias = np.linspace(0.1,1,10)
set_lengthscale = None
t = 100

obj = "LN_obj"
# obj = np.array(["obj","LN_obj"])
emulator = True
# emulator = np.array([False,True])
sparse_grid = False
# sparse_grid = np.array([False,True])
verbose = False
save_fig = True

#Pull Experimental data from CSV
exp_data_doc = 'Input_CSVs/Exp_Data/n=5.csv'
exp_data = np.array(pd.read_csv(exp_data_doc, header=0,sep=","))
Xexp = exp_data[:,1]
Yexp = exp_data[:,2]
n = len(Xexp)

#Define GP Testing space
p=20
Theta1 =  np.linspace(-2,2,p) #1x10
Theta2 =  np.linspace(-2,2,p) #1x10
theta_mesh = np.array(np.meshgrid(Theta1, Theta2)) #2 Uniform 5x5 arrays  

print("Runs:", runs)
print("BO Iterations:",BO_iters)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%")

print("-------------------")
print("Emulator?:", emulator)
print("______________________________")
print("Sparse Grid?:", sparse_grid)  


all_data_doc = find_train_doc_path(emulator, obj, t)
all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=",")) 
print("Objective Function:", obj)
print("-  -  -  -  -  -  -  -  -  -  -")
for i in range(len(explore_bias)):
    print("Separation Factor Train/Test:", str(np.round(sep_fact,3)))
    print("Lengthscale Set To:", set_lengthscale)
    ep = torch.tensor([float(explore_bias[i])])
    print("Explore Bias Multiplier:", str(np.round(float(ep),3)))
    results = bo_iter_w_runs(BO_iters,all_data_doc,t,theta_mesh,Theta_True,train_iter,ep, Xexp, Yexp,
                                 noise_std, obj, runs, sparse_grid, emulator, set_lengthscale, verbose, 
                                 save_fig, shuffle_seed, DateTime, sep_fact = sep_fact)
    print("The GP predicts the lowest SSE of", "{:.3e}".format(np.exp(results[3])), "occurs at \u03B8 =", results[2][0], 
              "during run", results[1], "at BO iteration", results[0])
    print("At this point, the highest EI occurs at \u03B8 =", results[4][0])
    print("\n")