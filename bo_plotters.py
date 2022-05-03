from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

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
def plotter_adv(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train=True):
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
    
    #Assert that test_mesh and z are NxN, that p_true and p_GP_opt are 2x1, and the title is a string
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
#     assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==2, "p_true and p_GP_opt must be 2x1 for a 2 input GP"
    assert isinstance(title, str)==True, "Title must be a string"
    
    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot z.T because test_mesh.T was used to calculate z
    plt.contourf(xx, yy,z)
    plt.colorbar()
#     print(p_GP_opt[0],p_GP_opt[1])
    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True Optimal Value", s=50, marker = (5,1))
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP Optimal Value", marker = ".")
    
    if plot_train == True:
        plt.scatter(train_p[:,0],train_p[:,1], color="blue", label = "Training Data", s=25, marker = ".")

    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')

    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('$\\theta_1$',weight='bold')
    plt.ylabel('$\\theta_2$',weight='bold')
    plt.title("Heat Map of "+title, weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()

def y_plotter_adv(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True):
    """
    Plots the y values of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of values the GP predicted function values
        title: str, The title for the graph
        yval: True or False, will determine whether true values will be plotted with y model values
    
    Returns
    -------
        A 3D Heat map of the values of z predicted by the GP
    """
    return plotter_adv(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True)


def stdev_plotter_adv(parameter_space, z, p_true, p_GP_opt, train_p,plot_train=True):
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
    return plotter_adv(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True)

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
    return plotter_adv(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True)

def ei_plotter_adv(parameter_space, z, p_true, train_p, p_GP_opt = None, plot_train=True):
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
    return plotter_adv(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True)


def error_plotter_adv(parameter_space, z, p_true, p_GP_opt,title,train_p,plot_train=True):
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
    if isinstance(z2,ndarray)!=True:
        z2 = np.asarray(z2)
        
    error = (z2 - z)**2
    return plotter_adv(parameter_space, error,title)

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

def basic_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train=True):
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
    
    #Assert that test_mesh and z are NxN, that p_true and p_GP_opt are 2x1, and the title is a string
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==2, "p_true and p_GP_opt must be 2x1 for a 2 input GP"
    assert isinstance(title, str)==True, "Title must be a string"
    
    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot z.T because test_mesh.T was used to calculate z
    plt.contourf(xx, yy,z.T)
    plt.colorbar()

    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True Optimal Value", s=50, marker = (5,1))
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP Optimal Value", marker = ".")
    
    if plot_train == True:
        plt.scatter(train_p[:,0],train_p[:,1], color="blue", label = "Training Data", s=25, marker = ".")

    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')

    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('$\\theta_1$',weight='bold')
    plt.ylabel('$\\theta_2$',weight='bold')
    plt.title("Heat Map of "+title +"Points = "+str(len(train_p)), weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()

def y_plotter_basic(test_mesh, z, p_true, p_GP_opt,train_p,title="y",plot_train=True):
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
    return basic_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train)

def stdev_plotter_basic(test_mesh, z, p_true, p_GP_opt,train_p,plot_train=True):
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
    return basic_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train)

def ei_plotter_basic(test_mesh, z, p_true, p_GP_opt,train_p,plot_train=True):
    """
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting expected improvement values.

    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (expected improvement values)
    """
    title = "Expected Improvement"
    return basic_plotter(test_mesh, z, p_true, p_GP_opt,title,train_p,plot_train)

                        
    
