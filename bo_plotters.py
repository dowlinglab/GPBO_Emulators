from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from bo_functions import calc_y_exp

def plot_hyperparams(iterations, hyperparam, title):
    '''
    Plots Hyperparameters
    Parameters
    ----------
        Iterations: Float, number of training iterations
        hyperparam: ndarray, array of hyperparameter value at each training iteration 
        title: string, title of the graph
     
    Returns
    -------
        plt.show(), A plot of iterations and hyperparameter
    '''
    assert isinstance(title, str)==True, "Title must be a string"
    iters_axis = np.linspace(0,iterations, iterations)
    assert len(iters_axis) == len(hyperparam), "Hyperparameter array must have length of # of training iterations"
    
    plt.figure()
    plt.plot(iters_axis, hyperparam)
    plt.grid(True)
    plt.xlabel('Iterations',weight='bold')
    plt.ylabel('Hyperparameter Value',weight='bold')
    plt.title("Plot of "+title, weight='bold',fontsize = 16)
    return plt.show()

def plot_xy(x, y_exp, y_GP,Theta_True,title):
    '''
    Plots Hyperparameters
    Parameters
    ----------
        x_exp: ndarray, array of X_exp values
        y_exp: ndarray, array of Y_exp values
        y_GP: ndarray, array of y_GP values given based on GP Theta_Best
     
    Returns
    -------
        plt.show(), A plot of iterations and hyperparameter
    '''
    assert isinstance(title, str)==True, "Title must be a string"
    assert len(x) == len(y_exp) == len(y_GP), "Xexp, Yexp, and Y_GP must be the same length"
#     assert len(iters_axis) == len(hyperparam), "Hyperparameter array must have length of # of training iterations"
    x2 = np.linspace(-2,2,100)
    noise_std = 0
    y_true = calc_y_exp(Theta_True, x2, noise_std, noise_mean=0)
    
    plt.figure()
    plt.scatter(x, y_exp, label = "y_true", color = "orange")
    plt.plot(x, y_GP, label = "y_GP")
    plt.plot(x2, y_true, color = "orange")
    plt.grid(True)
    plt.legend(loc = "best")
    plt.xlabel('X Value',weight='bold')
    plt.ylabel('Y Value',weight='bold')
    plt.title("Plot of "+title, weight='bold',fontsize = 16)
    return plt.show()

def value_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train=True):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        title: str, A string containing the title of the plot
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Defines the x and y coordinates that will be used to generate the heat map, this step isn't
    #necessary, but streamlines the process
    xx , yy = test_mesh #NxN, NxN
#     print(p_true,p_GP_opt)
    #Assert that test_mesh and z are NxN, that p_true and p_GP_opt are 2x1, and the title is a string
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==2, "p_true and p_GP_opt must be 2x1 for a 2 input GP"
    assert isinstance(title, str)==True, "Title must be a string"
    assert len(train_p.T) >= 2, "Train_p must have at least 2 columns"
    
    plt.contourf(xx, yy,z)
    plt.colorbar()
#     print(p_GP_opt[0],p_GP_opt[1])
    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True Optimal Value", s=50, marker = (5,1))
    
    if plot_train == True:
        plt.scatter(train_p[:,0],train_p[:,1], color="blue", label = "Training Data", s=25, marker = ".")
        
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", s=50, label = "GP Optimal Value", marker = ".")
    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')

    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'upper left')

    #Creates axis labels and title
    plt.xlabel('$\\theta_1$',weight='bold')
    plt.ylabel('$\\theta_2$',weight='bold')
    plt.title("Heat Map of "+title +" Points = "+str(len(train_p)), weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()

def ei_plotter(parameter_space, z, p_true, p_GP_opt,train_p,plot_train=True):
    """
    Plots the expected improvement of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP expected improvement values
    
    Returns
    -------
        A 3D Heat map of the values of expected improvement predicted by the GP
    """
    title = "Expected Improvement"
    return value_plotter(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train)

def y_plotter(test_mesh, z, p_true, p_GP_opt,train_p,title="y",plot_train=True):
    '''
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting y values.

    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        title: str, A string containing the title of the plot
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (y values)
    '''
    return value_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train)


def stdev_plotter(test_mesh, z, p_true, p_GP_opt,train_p,plot_train=True):
    '''
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting standard deviation values.

    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (standard deviation values)
    '''
    title = "Standard Deviation"
    return value_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train)

def error_plotter(parameter_space, z, p_true, p_GP_opt,train_p,plot_train=True):
    """
    Plots the error^2 of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP expected improvement values
    
    Returns
    -------
        A 3D Heat map of the values of expected improvement predicted by the GP
    """
    title = "Error Magnitude"
    
    if isinstance(z,ndarray)!=True:
        z = np.asarray(z)
        
    error = z
    return value_plotter(parameter_space, error, p_true, p_GP_opt,title,train_p,plot_train=True)
 
def ei_plotter_adv_test(parameter_space, z, p_true, train_p,Xexp,p_GP_opt = None,plot_train=True):
    """
    Plots the expected improvement of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP expected improvement values
    
    Returns
    -------
        A 3D Heat map of the values of expected improvement predicted by the GP
    """
    title = "Expected Improvement: Xexp = " + str(Xexp)
    return value_plotter(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True)

