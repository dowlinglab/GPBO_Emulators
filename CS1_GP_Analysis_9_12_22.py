import gpytorch
import numpy as np
import pandas as pd
import torch

from bo_functions_GP_Analysis import LSO_LOO_Analysis
from datetime import datetime

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#--------------------------------

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
print("Date and Time: ", timestampStr)
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
# DateTime = None ##For Testing

#Set Parameters
# emulator = True
emulator = np.array([True, False])
# obj = "obj"
obj = np.array(["obj", "LN_obj"])
save_fig = True
verbose = True

LOO = False
LSO = True

Theta_True = np.array([1,-1])
train_iter = 300
noise_std = 0.1
shuffle_seed = 9
set_lengthscale = None

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

for emul in emulator:
    
    print("Emulator?:", emul)
    print("-------------------")
    if emul == True:
        len_data = 100
        obj_use = np.array(["obj"])
    else:
        obj_use = obj
        len_data = 20
    for obj_func in obj_use:
        print("Objective Function:", obj_func)
        print("Lengthscale Set To:", set_lengthscale)
        results = LSO_LOO_Analysis(theta_mesh,Theta_True,train_iter, Xexp, Yexp,
                                         noise_std, obj_func, emul, set_lengthscale, len_data, verbose, 
                                         save_fig, shuffle_seed, DateTime, LOO, LSO)
        print("-  -  -  -  -  -  -  -  -  -  -")