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
        plot_title: str, The title for the graph
    
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

    #https://stackoverflow.com/questions/17756925/how-to-plot-heatmap-colors-in-3d-in-matplotlib
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')

    xs = p_1
    ys = p_2
    zs = p_3

    the_fourth_dimension = z

    colors = cm.viridis(the_fourth_dimension/max(the_fourth_dimension))

    colmap = cm.ScalarMappable(cmap=cm.viridis)
    colmap.set_array(the_fourth_dimension)

    yg = ax.scatter(xs, ys, zs, c=colmap.to_rgba(the_fourth_dimension)[:,0:3], marker='o',s=200)
    cb = fig.colorbar(colmap)

    # adding title and labels
    ax.set_title("Heat Map of "+plot_title, fontsize = 18)
    ax.set_xlabel('$\\theta_1$', fontsize = 15)
    ax.set_ylabel('$\\theta_2$', fontsize = 15)
    ax.set_zlabel('x coordinate', fontsize = 15)
    
    # displaying plot
    plt.savefig(plot_title+'.png')
    plt.show()
    return 

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

def improvement_integral_plot(parameter_space, z):
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
    title = "Improvement Integrand" #(e* - (f-mu-sig*eps)^2*pdf(eps))
    return plotter_adv(parameter_space, z,title)

def improvement_int_terms_plot(z, term_num, index_num):
    """
    Plots a term of the improvement of the GP
    Parameters
    ----------
        z: ndarray, the values of epsilon and all improvement term for each iterations
        term_num: The improvement integral term being printed
        index_num: Number of indexes to print
    
    Returns
    -------
        A 3D Heat map of the values of improvement predicted by the GP
    """
    assert isinstance(term_num, (int))==True, "Term number must be an integer 1, 2, or 3"
    assert isinstance(index_num, (int))==True, "Index number must be an integer less than len(parameter_space)"
    assert 1<= term_num <= 3, "Term number must be an integer 1, 2, or 3"
    assert index_num<=len(z[0]), "Index number must be less than or equal to len(parameter_space)"
    
    title_options = ["Term 1: (e* - (f-mu)^2)*pdf(eps)", "Term 2: 2*(f-mu)*sigma*eps*pdf(eps)", "Term 3:-var*eps^2*pdf(eps)"]
#     title = "Term 1: (e* - (f-mu)^2*pdf(eps))"
    title = title_options[term_num-1]
    
    for i in range(index_num):
        print("Index: ", i+1)
        plt.figure()
        plt.title(title)
        plt.xlabel("Epsilon Value")
        plt.ylabel(str("Improvement Integral Term " +str(term_num)))
        epsilon = z[0][:,i]
        I_term = z[term_num][:,i]
        plt.plot(epsilon,I_term)
        plt.show()
    return 