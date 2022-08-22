import numpy as np
import pandas as pd
import torch
from bo_functions import bo_iter_w_runs
from bo_functions import find_train_doc_path
import matplotlib as mpl
from datetime import datetime
mpl.rcParams['figure.dpi'] = 200

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
# DateTime = None ##For Testing

#Set Parameters
Theta_True = np.array([1,-1])
BO_iters = 100
runs = 15
train_iter = 300
noise_std = 0.1
shuffle_seed = 9
sep_fact = 1
explore_bias = torch.tensor([0.5, 0.75, 1, 5])
set_lengthscale = np.array([None, 0.1, 0.25, 0.5])

# obj = "obj"
obj = np.array(["obj","LN_obj"])
# emulator = False
emulator = np.array([False,True])
# sparse_grid = np.array([False])
sparse_grid = np.array([False,True])

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

# all_data_doc = find_train_doc_path(emulator, obj)
# all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))   

print("Runs:", runs)
print("BO Iterations:",BO_iters)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
for emul in emulator: 
    print("-------------------")
    print("Emulator?:", emul)
    if emul == True:
        t = 100
        obj_use = np.array(["obj"])   
        sparse_grid_use = sparse_grid
    else:
        t = 20
        obj_use = obj
        sparse_grid_use = np.array([sparse_grid[1]])
        
    for sparse in sparse_grid_use:
        #Can use only 0 for sparse grid exploration parameter
        if sparse == True:
            ep_use = torch.tensor([0]) 
        else:
            ep_use = explore_bias
        print("______________________________")
        print("Sparse Grid?:", sparse)  

        for obj_func in obj_use:
            all_data_doc = find_train_doc_path(emul, obj_func)
            all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))
            print("Objective Function:", obj_func)

            for i in range(len(set_lengthscale)):
                for j in range(len(ep_use)):
                    print(" \n")
                    print("Lengthscale Set As:", set_lengthscale[i])
                    print("Explore Bias:", str(np.round(float(ep_use[j]),3)))
                    results = bo_iter_w_runs(BO_iters,all_data_doc,t,theta_mesh,Theta_True,train_iter,ep_use[j], Xexp, Yexp,
                                                 noise_std, obj_func, runs, sparse, emul, set_lengthscale[i], verbose, 
                                                 save_fig, shuffle_seed, DateTime)
                    print("The GP predicts the lowest SSE of", "{:.3e}".format(np.exp(results[3])), "occurs at \u03B8 =", results[2][0], 
                              "during Run", results[1], "at BO iteration", results[0])
                    print("At this point, the highest EI occurs at \u03B8 =", results[4][0])