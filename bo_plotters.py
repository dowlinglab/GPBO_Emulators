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

def plotter_4D(parameter_space,z, plot_title="Model Output"):
    """
    Plots the values of the GP given by the user
    Parameters
    ----------
        parameter_space: tensor or ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  tensor or ndarray, nx1 array of values
        plot_title: str, The title for the graph
    
    Returns
    -------
        A 4D Heat map of the values of z predicted by the GP
    """
    #Converts tensors and tuples to ndarrays
    if torch.is_tensor(parameter_space)==True:
        parameter_space= parameter_space.numpy()
        
#     if isinstance(z,ndarray)!=True:
#         z = np.asarray(z)
   
    #Asserts that the parameter space is 3 inuts, the data to be plotted is an array, and the plot title is a string
    assert isinstance(plot_title,str) == True, "Plot title must be a string."

    #https://stackoverflow.com/questions/17756925/how-to-plot-heatmap-colors-in-3d-in-matplotlib
    
    # Define dimensions
    X, Y, Z = parameter_space

    # Create data
#     point_num = point_num
#     data = z.reshape(point_num,point_num,point_num).T
    data = z
    kw = {
        'vmin': data.min(),
        'vmax': data.max(),
        'levels': np.linspace(data.min(), data.max()),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, -1], Y[:, :, -1], data[:, :, -1],
        zdir='z', offset=Z.max(), **kw
    ) 
    _ = ax.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=Y.min(), **kw
    )
    C = ax.contourf(
        data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw
    )
    #CHange these
#     _ = ax.contourf(
#     X[:, :, 0], Y[:, :, 0], data[:, :, 0],
#     zdir='z', offset=Z.min(), **kw
#     )
#     _ = ax.contourf(
#         X[-1, :, :], data[-1, :, :], Z[-1, :, :],
#         zdir='y', offset=Y.max(), **kw
#     )
#     C = ax.contourf(
#         data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
#         zdir='x', offset=X.min(), **kw
#     )
    # --


    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel='$\Theta_1$',
        ylabel='$\Theta_2$',
        zlabel='x coord',
    )
    ax.set_title("Heat Map of "+plot_title, fontsize = 18)
    # Set distance and angle view
    ax.view_init(40, -30)
    ax.dist = 11

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=plot_title)

    # Show Figure
#     plt.savefig(plot_title+'_4D'+'.png')
    plt.show()
    
    return 

def error_plotter_4D(parameter_space, z):
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
    return plotter_4D(parameter_space,z, title)

def y_plotter_4D(parameter_space, z,title="y"):
    '''
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting y values.

    Parameters
    ----------
        parameter_space: ndarray, n NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        title: str, A string containing the title of the plot
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (y values)
    '''
    title = "Model Y Values"
    return plotter_4D(parameter_space, z,title)


def stdev_plotter_4D(parameter_space, z):
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
    return plotter_4D(parameter_space, z,title)

# def plotter_4D_2(parameter_space,z, plot_title="Model Output"):
#     """
#     Plots the values of the GP given by the user
#     Parameters
#     ----------
#         parameter_space: tensor or ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
#         z:  tensor or ndarray, nx1 array of values
#         plot_title: str, The title for the graph
    
#     Returns
#     -------
#         A 4D Heat map of the values of z predicted by the GP
#     """
#     #Converts tensors and tuples to ndarrays
#     if torch.is_tensor(parameter_space)==True:
#         parameter_space= parameter_space.numpy()
        
# #     if isinstance(z,ndarray)!=True:
# #         z = np.asarray(z)
   
#     #Asserts that the parameter space is 3 inuts, the data to be plotted is an array, and the plot title is a string
#     assert isinstance(plot_title,str) == True, "Plot title must be a string."

#     #https://stackoverflow.com/questions/17756925/how-to-plot-heatmap-colors-in-3d-in-matplotlib
    
#     # Define dimensions
#     X, Y, Z = parameter_space

#     # Create data
# #     point_num = point_num
# #     data = z.reshape(point_num,point_num,point_num).T
#     data = z
#     kw = {
#         'vmin': data.min(),
#         'vmax': data.max(),
#         'levels': np.linspace(data.min(), data.max()),
#     }

#     fig = plt.figure(figsize=(5, 4))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot contour surfaces
#     for i in range(int(len(Z))):
#         ranges = int(len(Z)/2)
#         _ = ax.contourf(
#             X[:, :, i], Y[:, :, i], data[:, :, i],
#             zdir='z', offset=Z[0,0,i], **kw, cmap = "viridis"
#         ) 

#     _ = ax.contourf(
#         X[0, :, :], data[0, :, :], Z[0, :, :],
#         zdir='y', offset=Y[0,0,0], **kw, cmap = "viridis"
#     )
#     C = ax.contourf(
#         data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
#         zdir='x', offset=X[0,0,0], **kw, cmap = "viridis"
#     )

#     #     C = ax.contourf(
#     #         data[:, -1, :], Y[:, -1, :], Z[:, i, :],
#     #         zdir='x', offset=X[0,-1,0], **kw,cmap = "viridis"
#     #     )


#     # Set limits of the plot from coord limits
#     xmin, xmax = X.min(), X.max()
#     ymin, ymax = Y.min(), Y.max()
#     zmin, zmax = Z.min(), Z.max()
#     ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

#     # Plot edges
#     edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
#     ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], **edges_kw)
#     ax.plot([xmin, xmax], [ymax, ymax], [zmax, zmax], **edges_kw)
#     ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
#     ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)
#     ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)

#     # Set labels and zticks
#     ax.set(
#         xlabel='$\Theta_1$',
#         ylabel='$\Theta_2$',
#         zlabel='x coord',
#     )

#     # Set distance and angle view
#     ax.view_init(40, 30)
#     ax.dist = 11

#     # Colorbar
#     fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=":)")

#     # Show Figure
#     plt.show()

#     # Plot contour surfaces
#     shrink = len(Z)/2
#     for i in range(3):
#         # Create a figure with 3D ax
#         fig = plt.figure(figsize=(5, 4))
#         ax = fig.add_subplot(111, projection='3d')
#         for i in range(int(len(Z)/2)):
#             up_lim = len(Z) - int(len(Z)/shrink)
#             low_lim = int(len(Z)/shrink)
#     #         print(low_lim,up_lim)
#             _ = ax.contourf(
#                 X[low_lim:up_lim, low_lim:up_lim, i], 
#                 Y[low_lim:up_lim, low_lim:up_lim, i], 
#                 data[low_lim:up_lim, low_lim:up_lim, i],
#                 zdir='z',offset=Z[-1,-1,i+low_lim],  **kw, cmap = "viridis"
#             ) 

#         # Set limits of the plot from coord limits
#         xmin, xmax = test_p1.min(), test_p1.max()
#         ymin, ymax = test_p2.min(), test_p2.max()
#         zmin, zmax = test_p3.min(), test_p3.max()
#         ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

#         # Plot edges
#         edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)

#         # Set labels and zticks
#         ax.set(
#             xlabel='$\Theta_1$',
#             ylabel='$\Theta_2$',
#             zlabel='x coord',
#         )

#         # Set distance and angle view
#         ax.view_init(40, 30)
#         ax.dist = 11

#         # Colorbar
#         fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=":)")

#         # Show Figure
#         plt.show()
#         shrink = shrink/1.75
#     return

# def y_plotter_4D_2(parameter_space, z,title="y"):
#     '''
#     Helper function for basic_plotter. Calls basic_plotter specifically for plotting y values.

#     Parameters
#     ----------
#         parameter_space: ndarray, n NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
#         z: ndarray, An NxN Array containing all points that will be plotted. Y-values
#         title: str, A string containing the title of the plot
     
#     Returns
#     -------
#         plt.show(), A heat map of test_mesh and z (y values)
#     '''
#     title = "Model Y Values"
#     return plotter_4D_2(parameter_space, z,title)