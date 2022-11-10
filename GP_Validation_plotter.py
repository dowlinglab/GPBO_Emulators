from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pandas as pd
import os
import matplotlib.pyplot as plt

def LOO_Plots_3_Input(X_space, Y_sim, GP_mean, GP_stdev, Theta, Xexp, train_p = None, train_y = None, test_p = None, test_y = None, verbose = True):
#     if verbose == True:
#         print("GP Mean",GP_mean)
#         print("GP Stdev",GP_stdev)
#         print("SSE",sum(GP_mean-Y_sim)**2)
#         plt.close()
    fig, ax = plt.subplots()
    
    ax.plot(X_space, GP_mean, lw=2, label="GP_mean")
    ax.plot(X_space, Y_sim, color = "green", label = "Y_sim")
#     if train_p != None:
#         ax.scatter(train_p[:,-1], train_y, color = "black", label = "Training")
    if test_p != None:
        ax.scatter(Xexp, test_y, color = "red", label = "Testing")
    
    ax.fill_between(
        X_space,
        GP_mean - 1.96 * GP_stdev,
        GP_mean + 1.96 * GP_stdev,
        alpha=0.3
    )
#     ax.set_title("GP Mean + confidence interval at"+ str(Theta))
    ax.set_xlabel("Xexp")
    ax.set_ylabel("Function Value")
    ax.legend()
    
    return plt.show()