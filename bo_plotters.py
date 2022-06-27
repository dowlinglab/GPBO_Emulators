from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import os
import matplotlib.pyplot as plt

def save_fig(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
    
    
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
    #Assert Statements and making an axis for training iterations
    assert isinstance(title, str)==True, "Title must be a string"
    iters_axis = np.linspace(0,iterations, iterations)
    assert len(iters_axis) == len(hyperparam), "Hyperparameter array must have length of # of training iterations"
    
    #Plotting hyperparameters vs training iterations
    plt.figure()
    plt.plot(iters_axis, hyperparam)
    plt.grid(True)
    plt.xlabel('Iterations',weight='bold')
    plt.ylabel('Hyperparameter Value',weight='bold')
    plt.title("Plot of "+title, weight='bold',fontsize = 16)
    return plt.show()

def plot_org_train(test_mesh,train_p,p_true):
    '''
    Plots original training data with true value
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        train_p: tensor or ndarray, The training parameter space data
        p_true: ndarray, A 2x1 containing the true input parameters
     
    Returns
    -------
        plt.show(), A plot of the original training data points and the true value
    '''
    #xx and yy are the values of the parameter sets
    xx,yy = test_mesh
    plt.figure()
    #plot training data and true values
    plt.scatter(train_p[:,0],train_p[:,1], color="green",s=25, label = "Training Data", marker = "x")
    plt.scatter(p_true[0],p_true[1], color="blue", label = "True Optimal Value", s=100, marker = (5,1))
    #Set plot details
    plt.legend(loc = "best")
    plt.xlabel("$\Theta_1$")
    plt.ylabel("$\Theta_2$")
    #Set axis limits based on the maximum and minimum of the parameter search space
    plt.xlim((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy), np.amax(yy)))
    plt.title("Starting Training Data")
    plt.grid(True)
    return plt.show()

def plot_obj_abs_min(bo_iters, obj_abs_min, restarts, emulator):
    '''
    Plots the absolute minimum of the objective over BO iterations
    Parameters
    ----------
        BO_iters: integer, number of BO iteratiosn
        obj_abs_min: ndarray, An array containing the absolute minimum of SSE found so far at each iteration
        restarts: int, The number of times to choose new training points
        emulator: True/False, Determines if GP will model the function or the function error
     
    Returns
    -------
        plt.show(), A plot of the minimum SSE vs BO iteration for each restart
    '''
    #Create bo_iters as an axis
    bo_space = np.linspace(1,bo_iters,bo_iters)
    
    #Plot Minimum SSE value at each restart
    plt.figure()
    if restarts == None:
        plt.step(bo_space,obj_abs_min, label = "Minimum SSE Value Found")
    else:
        for i in range(restarts):
            plt.step(bo_space, obj_abs_min[i], label = "Restart: "+str(i+1))
    #Set plot details        
    plt.legend(loc = "best")
    plt.xlabel("BO Iterations")
    plt.ylabel("SSE")
    plt.title("BO Iteration Results: Lowest Overall SSE")
    plt.grid(True)
    
    #Save figure path
    if emulator == True:
        path = "Figures/Convergence_Figs/GP_Emulator/Min_SSE_Conv/"
    else:
        path = "Figures/Convergence_Figs/GP_Error_Emulator/Min_SSE_Conv/"
    save_fig(path, ext='png', close=False, verbose=False)
    
    return plt.show()

def plot_xy(x_line, x_exp, y_exp, y_GP,y_GP_long,y_true,title = "XY Comparison"):
    '''
    Plots Hyperparameters
    Parameters
    ----------
        x_line: ndarray, array of many values for which to graph y_true
        x_exp: ndarray, array of X_exp values
        y_exp: ndarray, array of Y_exp values
        y_GP: ndarray, array of y_GP values given based on GP Theta_Best
        y_GP_long: ndarray, array of y_GP values given based on GP Theta_Best using x_line
        y_true: ndarray, array of y_true values at all points in x_line
        title: str, Title of the plot
     
    Returns
    -------
        plt.show(), A plot of iterations and hyperparameter
    '''
    #assert statments
    assert isinstance(title, str)==True, "Title must be a string"
    assert len(x_exp) == len(y_exp) == len(y_GP), "Xexp, Yexp, and Y_GP must be the same length"
    
    #Plot x vs Y for experimental and extrapolated data
    plt.figure()
    plt.scatter(x_exp, y_exp, label = "y $\Theta_{true}$", color = "orange")
    plt.scatter(x_exp, y_GP)
    plt.plot(x_line, y_true, color = "orange")
    plt.plot(x_line, y_GP_long, "--", label = "y $\Theta_{GP}$")
    
    #Set plot details
    plt.grid(True)
    plt.legend(loc = "best")
    plt.xlabel('X Value',weight='bold')
    plt.ylabel('Y Value',weight='bold')
    plt.title("Plot of "+title, weight='bold',fontsize = 16)
    
    return plt.show()

def plot_obj_Theta(obj_array, Theta_array, Theta_True, t, bo_iters, obj, ep, emulator, restarts=0):
    """
    Plots the objective function and Theta values vs BO iteration
    
    Parameters
    ----------
        obj_array: ndarry, (nx1): The output array containing objective function values
        Theta_array: ndarray, (nxq): The output array containing objective function values
        Theta_True: ndarray, Used for plotting Theta Values
        t: int, Number of training points to use
        bo_iters: integer, number of BO iterations
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        restarts: int, The number of times to choose new training points
    
    Returns:
    --------
        Plots of obj vs BO_iter and Plots of Theta vs BO_iter
    """
    assert len(obj_array) == len(Theta_array), "obj_array and Theta_array must be the same length"
    assert isinstance(obj,str)==True, "Objective function name must be a string" 
    
    #Find value of q from given information
    q = len(Theta_True)
    #Create x axis as # of bo iterations
    bo_space = np.linspace(1,bo_iters,bo_iters)
    
    #Set a string for exploration parameter and initial number of training points
    ep = str(np.round(float(ep),1))
    org_TP = str(t)

    plt.figure() 
    
    #Plots either 1 or multiple lines for objective function values depending on whether there are restarts
    if restarts !=0:
        #Loop over number of restarts
        for i in range(restarts):
            #Plot data
            plt.step(bo_space, obj_array[i], label = "Restart: "+str(i+1))
    else:
        plt.step(bo_space, obj_array, label = "SSE")
    
    #Set plot details
    plt.xlabel("BO Iterations")
    plt.ylabel("SSE")
    plt.title("BO Iteration Results: SSE Metric")
    plt.grid(True)
    plt.legend(loc = "upper right")
    
    #Save path and figure
    if emulator == True:
        path = "Figures/Convergence_Figs/GP_Emulator/"+"SSE_Conv/"+"TP_"+str(org_TP)+"/"+str(obj)+"/"+"Iter_"+str(bo_iters)
    else:
        path = "Figures/Convergence_Figs/GP_Error_Emulator/"+"SSE_Conv/"+"TP_"+str(org_TP)+"/"+str(obj)+"/"+"ep_"+str(ep)+"/"+"Iter_"+str(bo_iters)
    save_fig(path, ext='png', close=False, verbose=False)
    
    #Make multiple plots for each parameter
    #Loop over number of parameters
    for j in range(q):
        plt.figure()
        #Plot more than 1 line if there are many restarts
        if restarts != 0:
            #Loop over restarts and plot
            for i in range(restarts):
                plt.step(bo_space, Theta_array[i,:,j], label = "$\Theta_" +str({j+1})+"$"+" Restart: "+str(i+1))
        else:   
            plt.step(bo_space, Theta_array[:,j], label = "$\Theta_" +str({j+1})+"$")
        
        #Set plot details
        plt.step(bo_space, np.repeat(Theta_True[j],bo_iters), label = "$\Theta_{true,"+str(j+1)+"}$")
        plt.xlabel("BO Iterations")
        plt.ylabel("$\Theta_" + str({j+1})+"$")
        plt.title("BO Iteration Results: "+"$\Theta_"+str({j+1})+"$")
        plt.grid(True)
        plt.legend(loc = "upper left")
        
        #Save path and figure
        if emulator == True:
            path = "Figures/Convergence_Figs/GP_Emulator/Theta_Conv/"+"TP_"+str(org_TP)+"/"+str(obj)+"/"+"Theta_"+str(j)+"/"+"Iter_"+str(bo_iters)
        else:
            path = "Figures/Convergence_Figs/GP_Error_Emulator/Theta_Conv/"+"TP_"+str(org_TP)+"/"+str(obj)+"/"+"ep_"+str(ep)+"/"+"Theta_"+str(j)+"/"+"Iter_"+str(bo_iters)
        save_fig(path, ext='png', close=False, verbose=False)
#         plt.savefig("Figures/Convergence_Figs/Theta_Conv/"+str(org_TP)+"/"+str(obj)+"/"+str(ep)+"Iter_"+str(bo_iters)+".png",dpi = 600)
        plt.show()
    
    return


def value_plotter(test_mesh, z, p_true, p_GP_opt, p_GP_best, train_p,title,title_save, obj,ep, emulator, Bo_iter = None, restart = 0):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        p_GP_Best: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        title: str, A string containing the title of the plot
        title_save: str, A string containing the title of the file of the plot
        obj: str, The name of the objective function. Used for saving figures
        ep: int or float, the exploration parameter
        emulator: True/False, Determines if GP will model the function or the function error
        Bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Backtrack out number of parameters from given information
    q=len(p_true)
    xx , yy = test_mesh #NxN, NxN

    #Assert sattements
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==len(p_GP_best)==q, "p_true, p_GP_opt, and p_GP_best must be qx1 for a q input GP"
    assert isinstance(title, str)==True, "Title must be a string"
    assert len(train_p.T) >= q, "Train_p must have at least q columns"
    assert isinstance(Bo_iter,int) == True or Bo_iter == None, "Bo_iter must be an integer or None"
    assert isinstance(obj,str)==True, "Objective function name must be a string" 
    
    #Set plot details
#     plt.figure(figsize=(8,4))
    plt.contourf(xx, yy,z,cmap = "autumn")
    plt.colorbar()
#     print(p_GP_opt[0],p_GP_opt[1])
    
    #Can only plot np arrays
    if torch.is_tensor(train_p) == True:
        train_p = train_p.numpy()
        
    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')    
    
    #Plots the true optimal value, the best EI value, the GP value, and all training points
    plt.scatter(p_true[0],p_true[1], color="blue", label = "True Optimal Value", s=100, marker = (5,1))
        
    plt.scatter(train_p[:,0],train_p[:,1], color="green",s=25, label = "Training Data", marker = "x")
    
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="white", s=50, label = "GP min(SSE) Value", marker = ".")
    
    plt.scatter(p_GP_best[0],p_GP_best[1], color="black", s=25, label = "GP Best EI Value", marker = ".")
    
    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')
    
    #Plots grid and legend
#     plt.grid()
    plt.legend(loc = 'upper right')

    #Creates axis labels and title
    plt.xlabel('$\\theta_1$',weight='bold')
    plt.ylabel('$\\theta_2$',weight='bold')
    plt.xlim((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy),np.amax(yy)))   
    
    #Back out number of original training points for saving figures
    if Bo_iter != None:
        plt.title(title+" BO iter "+str(Bo_iter+1), weight='bold',fontsize=16)
        ep = str(np.round(float(ep),1))
        
        if emulator == True:
            org_TP = str(len(train_p) - 5*(Bo_iter) )
        else:
            org_TP = str(len(train_p) - Bo_iter )
        
        #Generate path and save figures 
        #Separate by iteration, org_TP, and ep
        if emulator == True:
            path = "Figures/"+"GP_Emulator/"+"TP_"+str(org_TP)+"/"+str(obj)+"/"+title_save+"/"+"Restart_"+str(restart+1)+"/Iter_"+str(Bo_iter+1)
        else:
            path = "Figures/"+"GP_Error_Emulator/"+"TP_"+str(org_TP)+"/"+str(obj)+"/"+"ep_"+str(ep)+"/"+title_save+"/"+"Restart_"+str(restart+1)+"/Iter_"+str(Bo_iter+1)
        save_fig(path, ext='png', close=False, verbose=False)
    #Don't save if there's only 1 BO iteration
    else:
        plt.title("Heat Map of "+title, weight='bold',fontsize=16)     
           
    return plt.show()


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

def value_plotter_4D(parameter_space, z, title):
    """
    Plots a value of the GP in 4D
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of the GP expected improvement values
    
    Returns
    -------
        A 3D Heat map of the values of expected improvement predicted by the GP
    """    
    if isinstance(z,ndarray)!=True:
        z = np.asarray(z)
        
    error = z
    return plotter_4D(parameter_space,z, title)


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
#             zdir='z', offset=Z[0,0,i], **kw, cmap = ""
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