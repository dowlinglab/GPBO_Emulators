from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def plotter_adv_4D(parameter_space, z, point_num, plot_title="Model Output",yval = False):
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
        
    if isinstance(z,ndarray)!=True:
        z = np.asarray(z)
   
    #Asserts that the parameter space is 3 inuts, the data to be plotted is an array, and the plot title is a string
    assert isinstance(plot_title,str) == True, "Plot title must be a string."

    #https://stackoverflow.com/questions/17756925/how-to-plot-heatmap-colors-in-3d-in-matplotlib
    
    # Define dimensions
    X, Y, Z = parameter_space

    # Create data
    point_num = point_num
    data = z.reshape(point_num,point_num,point_num).T

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

    # Set distance and angle view
    ax.view_init(40, -30)
    ax.dist = 11

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=plot_title)

    # Show Figure
    plt.show()
    return 

def plotter_adv(parameter_space, z,plot_title="Model Output",yval = False):
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

    yg = ax.scatter(xs, ys, zs, c=colmap.to_rgba(the_fourth_dimension)[:,0:3], marker='o')

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

def y_plotter_adv(parameter_space, z,plot_title,yval):
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
    return plotter_adv(parameter_space, z,plot_title,yval)

def y_plotter_adv_4D(parameter_space, z,point_num,plot_title,yval):
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
    return plotter_adv_4D(parameter_space, z,point_num,plot_title,yval)

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

def stdev_plotter_adv_4D(parameter_space, z,point_num):
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
    return plotter_adv_4D(parameter_space, z,point_num, title)

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
    return plotter_adv(parameter_space, z, title)

def ei_plotter_adv_4D(parameter_space, z, point_num):
    """
    Plots the expected improvement of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP expected improvement values
        point_num: The amount of points of each input parameter used to generate the initial meshgrid
    
    Returns
    -------
        A 3D Heat map of the values of expected improvement predicted by the GP
    """
    title = "Expected Improvement"
    return plotter_adv_4D(parameter_space, z, point_num, title)


def error_plotter_adv(parameter_space, z, z2):
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
    title = "Error Magnitude"
    
    if isinstance(z,ndarray)!=True:
        z = np.asarray(z)
    if isinstance(z2,ndarray)!=True:
        z2 = np.asarray(z2)
        
    error = np.sqrt((z2 - z)**2)
    return plotter_adv(parameter_space, error,title)

def error_plotter_adv_4D(parameter_space, z, z2, point_num):
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
    title = "Error Magnitude"
    
    if isinstance(z,ndarray)!=True:
        z = np.asarray(z)
    if isinstance(z2,ndarray)!=True:
        z2 = np.asarray(z2)
        
    error = np.sqrt((z2 - z)**2)
    return plotter_adv_4D(parameter_space, error, point_num, title)

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