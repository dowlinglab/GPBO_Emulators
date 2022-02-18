from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def plotter_adv(parameter_space, z,plot_title="Model Output"):
    """
    Plots the values of the GP given by the user
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of values
        title: str, The title for the graph
    
    Returns
    -------
        A 3D Heat map of the values of z predicted by the GP
    """

    assert len(parameter_space) == 3, "The GP is a 3 input GP. Please include only 3 parameters to plot."
    assert isinstance(y_model, ndarray) == True, "The data to plot must be a 1xn ndarray."
    assert isinstance(plot_title,str) == True, "Plot title must be a string."
    p_1 = parameter_space[:,0].numpy() #Theta1 #1xn
    p_2 = parameter_space[:,1].numpy() #Theta2 #1xn
    p_3 = parameter_space[:,2].numpy() #x #1xn

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
        z:  ndarray, nx1 array of values
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
        z:  ndarray, nx1 array of values
    
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
        z:  ndarray, nx1 array of values
    
    Returns
    -------
        A 3D Heat map of the values of expected improvement predicted by the GP
    """
    title = "Expected Improvement"
    return plotter_adv(parameter_space, z,title)
