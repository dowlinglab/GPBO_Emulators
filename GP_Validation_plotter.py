from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pandas as pd
import os
import matplotlib.pyplot as plt

def LOO_Plots(X_space, Y_space, Xexp, Yexp, GP_mean, GP_stdev, Theta):
    X1, X2 = X_space[:,0], X_space[:,1]
    # Compare the experiments to the true model
    fig = plt.figure(figsize = (6.4,4))
    ax = plt.axes(projection='3d')
    ax.contour3D(X1, X2, GP_mean, 100, cmap='Blues') #Ysim
    ax.contour3D(X1, X2, Y, 100, cmap='Reds') #Yexp
    ax.scatter3D(Xexp[:,0], Xexp[:,1], Yexp, c=Yexp, cmap='Greens', edgecolors = "k") #Yexp
    ax.plot(1000,1000,1000, label = "$y_{sim}$", color = 'blue')
    ax.plot(1000,1000,1000, label = "$y_{exp}$", color = 'red')
    ax.scatter(1000,1000,1000, label = "Exp Data", color = 'green', edgecolors = "k")
    plt.legend(fontsize=10,bbox_to_anchor=(0, 1.0, 1, 0.2),borderaxespad=0, loc = "lower right")
    
#     ax.fill_between(
#         X_space,
#         GP_mean - 1.96 * GP_stdev,
#         GP_mean + 1.96 * GP_stdev,
#         alpha=0.3)
        
    ax.minorticks_on() # turn on minor ticks
    ax.tick_params(direction="in",top=True, right=True) 
    ax.tick_params(which="minor",direction="in",top=True, right=True)


    ax.zaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.grid(False)


    ax.set_xlim((np.amin(X1),np.amax(X1)))
    ax.set_ylim((np.amin(X2),np.amax(X2)))

    ax.set_xlabel('X1', fontsize=16,fontweight='bold')
    ax.set_ylabel('X2', fontsize=16,fontweight='bold')
    ax.set_zlabel('Muller Potential',fontsize=16,fontweight='bold');

    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', nbins=5)
    # ax.locator_params(axis='z', nbins=5)
#     ax.set_title("GP Mean + confidence interval at"+ str(Theta))
    return plt.show()