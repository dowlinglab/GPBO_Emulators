from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def plotter_adv(parameter_space, z,plot_title="Model Output"):
    """
    Plots the values of the GP given by the user
    Parameters
    ----------
        parameter_space: tensor or ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  tensor or ndarray, nx1 array of values
        title: str, The title for the graph
    
    Returns
    -------
        A 3D Heat map of the values of z predicted by the GP
    """
    #Converts tensors and tuples to ndarrays
    if torch.is_tensor(parameter_space)==True:
        parameter_space= parameter_space.numpy()
        
    if isinstance(z,ndarray)!=True:
        z = np.asarray(z)

        
    #Asserts that the parameter space is 3 inuts, the data to be plotted is an array, and the plot title is a string
    assert len(parameter_space.T) == 3, "The GP is a 3 input GP. Please include only 3 input parameters to plot."
    assert isinstance(plot_title,str) == True, "Plot title must be a string."
    
    #Breaks Parameter space into separate componenets
    p_1 = parameter_space[:,0] #Theta1 #1xn
    p_2 = parameter_space[:,1] #Theta2 #1xn
    p_3 = parameter_space[:,2] #x #1xn
    
    #Sets what data will be within the graph as the heat map points
    color = z

    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cm.Greens_r)
    color_map.set_array(color)

    # creating the heatmap
    img = ax.scatter(p_1, p_2, p_3, marker='s',
                     s=200, color='green')
    plt.colorbar(color_map)

    # adding title and labels
    ax.set_title("Heat Map of "+plot_title)
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')
    ax.set_zlabel('x coordinate')
    
    # displaying plot
    return plt.show()

def y_plotter_adv(parameter_space, z,plot_title):
    """
    Plots the y values of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of values the GP predicted function values
        title: str, The title for the graph
    
    Returns
    -------
        A 3D Heat map of the values of z predicted by the GP
    """
    return plotter_adv(parameter_space, z,plot_title)

def stdev_plotter_adv(parameter_space, z):
    """
    Plots the standard deviation alues of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP predicted standard deviation values
    
    Returns
    -------
        A 3D Heat map of the values of standard deviation predicted by the GP
    """
    title = "Standard Deviation"
    return plotter_adv(parameter_space, z,title)

def ei_plotter_adv(parameter_space, z):
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
    return plotter_adv(parameter_space, z,title)

def improvement_plot(parameter_space, z):
    """
    Plots the improvement of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP improvement values
    
    Returns
    -------
        A 3D Heat map of the values of improvement predicted by the GP
    """
    title = "(e* - (f-mu-sig*eps)^2*pdf(eps)) Improvement"
    return plotter_adv(parameter_space, z,title)
