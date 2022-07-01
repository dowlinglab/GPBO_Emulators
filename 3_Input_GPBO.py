import numpy as np
import pandas as pd
import torch
from bo_functions import bo_iter_w_restarts
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#Set Parameters
Theta_True = np.array([1,-1])
BO_iters = 100
train_iter = 300
noise_std = 0.1
restarts = 15
shuffle_seed = 6
t=4
explore_bias = torch.tensor([0.75])
set_lengthscale = np.array([None, 0.5, 1, 5])

obj = "LN_obj"
emulator = True
verbose = False
save_fig=True
# sparse_grid = True
sparse_grid = False

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

all_data_doc = "Input_CSVs/Train_Data/all_3_data/t=25.csv"
all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))   

for i in range(len(set_lengthscale)):
    for j in range(len(explore_bias)):
        print("Lengthscale Set As:", set_lengthscale[i])
        print("Explore Bias:", str(np.round(float(explore_bias[j]),3)))
        results = bo_iter_w_restarts(BO_iters,all_data_doc,t,theta_mesh,Theta_True,train_iter,explore_bias[j], Xexp, Yexp,
                                     noise_std, obj, restarts, sparse_grid, emulator, set_lengthscale[i], verbose, 
                                     save_fig, shuffle_seed)
        print("The GP predicts the lowest SSE of", "{:.3e}".format(np.exp(results[3])), "occurs at \u03B8 =", results[2][0], 
                  "during restart", results[1], "at BO iteration", results[0])
        print(" \n")