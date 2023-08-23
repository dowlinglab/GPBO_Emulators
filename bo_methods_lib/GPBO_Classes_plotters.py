from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pandas as pd
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
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
        

def create_subplots(num_subplots):
    """
    Creates Subplots based on the amount of data
    """
    #Make figures and define number of subplots  
    row_num = int(np.floor(np.sqrt(num_subplots)))
    col_num = int(np.ceil(num_subplots/row_num))
#     print(row_num, col_num) 
    assert row_num * col_num >= num_subplots, "row * col numbers must be at least equal to number of graphs"
    fig, axes = plt.subplots(nrows = row_num, ncols = col_num, figsize = (col_num*6,row_num*6), squeeze = False, sharex = "row")
    ax = axes.reshape(row_num * col_num)
    
    return fig, ax, len(ax)
    
def plot_details(ax, plot_x, plot_y, xlabel, ylabel, title, xbins, ybins, fontsize):
    """
    Function for setting plot settings
    """
    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontweight='bold')   
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize,fontweight='bold')
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold')
    ax.xaxis.set_tick_params(labelsize=fontsize, direction = "in")
    ax.yaxis.set_tick_params(labelsize=fontsize, direction = "in")
    ax.locator_params(axis='y', nbins=ybins)
    ax.locator_params(axis='x', nbins=xbins)
    ax.minorticks_on() # turn on minor ticks
    ax.tick_params(which="minor",direction="in",top=True, right=True)
    
    if np.isclose(np.min(plot_x), np.max(plot_x), rtol =1e-6) == False:        
        ax.set_xlim(left = np.min(plot_x), right = np.max(plot_x))
 
    ax.set_box_aspect(1)
    
    if np.min(plot_y) == 0:
        ax.set_ylim(bottom = np.min(plot_y)-0.05, top = np.max(plot_y)+0.05)
#     elif np.isclose(np.min(plot_y), np.max(plot_y), rtol =1e-6) == False:
#         ax.set_ylim(bottom = np.min(plot_y)-abs(np.min(plot_y)*0.05), top = np.max(plot_y)+abs(np.min(plot_y)*0.05))
#         print(np.min(plot_y)-abs(np.min(plot_y)*0.05), np.max(plot_y)+abs(np.min(plot_y)*0.05))
#     else:
#         ax.set_ylim(bottom = np.min(plot_y)/2, top = np.max(plot_y)*2)
    
    
    return ax
         
    
def plot_2D_Data(data, data_names, data_true, xbins, ybins, title, x_label, y_label, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots 2D values of the same value on multiple subplots
    
    Parameters
    -----------
    data: ndarray (n_runs x n_iters x n_params), Array of data from bo workflow runs
    data_names: list of str, List of data names
    data_true: list of float/int, The true values of each parameter
    xbins: int, Number of bins for x
    ybins: int, Number of bins for y
    title: str, Title of graph
    x_label: str, title of x-axis
    y_label: str, title of y-axis
    title_fontsize: int, fontisize for title. Default 24
    other_fontsize: int, fontisize for other values. Default 20
    save_path: None or str, Path to save figure to. Default None (do not save figure).
    """
    #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
    subplots_needed = data.shape[-1]
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    #Print the title
    if title_fontsize is not None:
        fig.suptitle(title, weight='bold', fontsize=title_fontsize)
    fig.supxlabel(x_label, fontsize=other_fontsize,fontweight='bold')
    fig.supylabel(y_label, fontsize=other_fontsize,fontweight='bold')
    
    
    #Loop over different hyperparameters (number of subplots)
    for i in range(num_subplots):
        if i < data.shape[-1]:
            true_val_idx = i
            one_data_type = data[:,:,i]
            #Loop over all runs
            for j in range(one_data_type.shape[0]):
                label = "Run: "+str(j+1) 
                data_df_run = pd.DataFrame(data = one_data_type[j])
                data_df_j = data_df_run.loc[(abs(data_df_run) > 1e-6).any(axis=1),0]
                data_df_i = data_df_run.loc[:,0]
                if len(data_df_j) < 2:
                    data_df_j = data_df_i[0:int(len(data_df_j)+2)] #+2 for stopping criteria + 1 to include last point
                bo_len = len(data_df_j)
                bo_space = np.linspace(1,bo_len,bo_len) 
                if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                    ax[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
                ax[i].step(bo_space, data_df_j, label = label)
                if data_true is not None and j == one_data_type.shape[0] - 1:
                    ax[i].axhline(y=data_true[true_val_idx], color = "red", linestyle='-', label = "True Value")
                #Set plot details 
                y_label = None
                title = r'$'+ data_names[i]+ '$'
                plot_details(ax[i], bo_space, data_df_j, None, y_label, title, xbins, ybins, other_fontsize)

        #Set axes off if it's an extra
        else:
            ax[i].set_axis_off()
        #Fetch handles and labels on last iteration
        if i == num_subplots-1:
            handles, labels = ax[i].get_legend_handles_labels()
            
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
#     if verbose == True and save_figure == False:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  

    return

#This needs to take indecies as an argument and link indecies to a list of parameters
def plot_train_test_val_data(train_data, test_data, val_data, param_names, idcs_to_plot, x_exp, xbins, ybins, zbins, title, title_fontsize = 24, other_fontsize = 20, save_path = None):
    '''
    Plots original training/testing data with the true value
    Parameters
    ----------
        train_data: ndarray, The training parameter space data. Must be 3D Max
        test_data: ndarray, The testing parameter space data
        val_data: ndarray, The testing parameter space data
        param_names: list of str, List of parameter names
        true_params: tensor or ndarray, The true parameter space data
        idcs_to_plot: ndarray, The list of indecies that will be plotted in ascending order
        x_exp: The experimental x data used
        xbins: int, Number of bins for x
        ybins: int, Number of bins for y
        zbins: int, Number of bins for z
        title: str, Title of graph
        title_fontsize: int, fontisize for title. Default 24
        other_fontsize: int, fontisize for other values. Default 20
        save_path: None or str, Path to save figure to. Default None (do not save figure).
     
    Returns
    -------
        plt.show(), A plot of the original training data points and the true value
    '''
    i1, i2, i3 = idcs_to_plot
    
    #If there are less 2 parameters, plot in 2D
    if len(idcs_to_plot) == 2:
        #Set figure details
        plt.figure(figsize = (6,6))
        plt.xticks(fontsize=other_fontsize)
        plt.yticks(fontsize=other_fontsize)
        plt.tick_params(direction="in",top=True, right=True)
        plt.locator_params(axis='y', nbins=ybins)
        plt.locator_params(axis='x', nbins=xbins)
        plt.minorticks_on() # turn on minor ticks
        plt.tick_params(which="minor",direction="in",top=True, right=True)
        
        #plot training data, testing data, and true values
        plt.scatter(train_data[:,i1],train_data[:,i2], color="green",s=50, label = "Training", marker = "x", zorder = 1)        
        plt.scatter(test_data[:,i1],test_data[:,i2], color="red",s=25, label = "Testing", marker = "x", zorder = 2)
        plt.scatter(val_data[:,i1],val_data[:,i2], color="blue",s=20, label = "Validation", marker = "D", zorder = 3)
        #How to plot theta true given that a combination of x and theta can be chosen?
#         plt.scatter(true_params[i1],true_params[i2], color="blue", label = "True argmin"+r'$(e(\theta))$', s=100, marker=(5,1), zorder = 3)
        #Set plot details
        plt.legend(loc= "upper left", bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
        x_label = r'$\mathbf{'+ param_names[i1] +'}$'
        y_label = r'$\mathbf{'+ param_names[i2] +'}$'

        plt.xlabel(x_label, fontsize=other_fontsize, fontweight='bold')
        plt.ylabel(y_label, fontsize=other_fontsize, fontweight='bold')
        
        #Set axis limits based on the maximum and minimum of the parameter search space      
        x_lim_l = np.amin(np.concatenate((train_p[:,i1], test_p[:,i1]), axis = None))
        y_lim_l = np.amin(np.concatenate((train_p[:,i2], test_p[:,i2]), axis = None))
        x_lim_u = np.amax(np.concatenate((train_p[:,i1], test_p[:,i1]), axis = None))
        y_lim_u = np.amax(np.concatenate((train_p[:,i2], test_p[:,i2]), axis = None))
        plt.xlim((x_lim_l,x_lim_u))
        plt.ylim((y_lim_l,y_lim_u))
        plt.title(title, fontsize=title_fontsize, fontweight='bold')

     
    #Otherwise print in 3D
    elif len(idcs_to_plot) == 3:
        #How should I go about plotting the true value?

        # Create the figure
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.zaxis.set_tick_params(labelsize=other_fontsize)
        ax.yaxis.set_tick_params(labelsize=other_fontsize)
        ax.xaxis.set_tick_params(labelsize=other_fontsize)
        ax.tick_params(direction="in",top=True, right=True) 
        ax.locator_params(axis='y', nbins=ybins)
        ax.locator_params(axis='x', nbins=xbins)
        ax.locator_params(axis='z', nbins=zbins)
        ax.minorticks_on() # turn on minor ticks
        ax.tick_params(which="minor",direction="in",top=True, right=True)

    
        # Plot the values
        ax.scatter(train_data[:,i1], train_data[:,i2], train_data[:,i3], color = "green", s=50, label = "Training", marker='o', zorder = 1)
        ax.scatter(test_data[:,i1],test_data[:,i2], test_data[:,i3], color="red", s=25, label = "Testing", marker = "x", zorder = 2)
        ax.scatter(val_data[:,i1],val_data[:,i2], val_data[:,i3], color="blue", s=20, label = "Validation", marker = "D", zorder = 3)
#         ax.scatter(p_true_3D_full[:,0], p_true_3D_full[:,1], p_true_3D_full[:,2], color="blue", label = "True argmin" + r'$(e(\theta))$', 
#                     s=100, marker = (5,1), zorder = 3)
        
        plt.legend(loc= "upper left", bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
        x_label = r'$\mathbf{'+ param_names[i1] +'}$'
        y_label = r'$\mathbf{'+ param_names[i2] +'}$'
        z_label = r'$\mathbf{'+ param_names[i3] +'}$'
#         plt.xlabel(r'$\mathbf{\theta_1}$', fontsize=16, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=other_fontsize, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=other_fontsize, fontweight='bold')
        ax.set_zlabel(z_label, fontsize=other_fontsize, fontweight='bold')
        
        #How to set bounds given that you could plot any combination of theta and x dimensions on any axis?
#         x_lim_l = np.amin(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
#         y_lim_l = np.amin(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
#         x_lim_u = np.amax(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
#         y_lim_u = np.amax(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
#         plt.xlim((x_lim_l,x_lim_u))
#         plt.ylim((y_lim_l,y_lim_u))
        ax.grid(False)
        fig.suptitle(title, weight='bold', fontsize=title_fontsize)
    else:
        print("Must be a 2D or 3D graph")
        
    if save_path is not None:
        save_fig(save_path, ext='png', close=True, verbose=False)  
    else:
        plt.show()
        plt.close()

    return 

def plot_heat_maps(heat_map_data, theta_true, theta_obj_min, theta_ei_max, train_theta, param_names, levels, idcs_to_plot, xbins, ybins, zbins, title, title_fontsize = 24, other_fontsize = 20, save_path = None):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: list of 3 NxN arrays containing all points that will be plotted for GP_mean, GP standard deviation, and y_sim
        theta_true: ndarray, A 2x1 containing the true input parameters
        theta_obj_min: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        theta_ei_max: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        train_theta: Training data for the iteration under consideration
        levels: int, Number of levels to skip when drawing contour lines. Default is 20
        idcs_to_plot: ndarray, The list of values that will be plotted (0-2). 0=sse, 1 = sse var, 2 = ei
        xbins: int, Number of bins for x
        ybins: int, Number of bins for y
        zbins: int, Number of bins for z
        title: str, Title of graph
        title_fontsize: int, fontisize for title. Default 24
        other_fontsize: int, fontisize for other values. Default 20
        save_path: None or str, Path to save figure to. Default None (do not save figure).    
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Backtrack out number of parameters from given information
    if levels is None:
        tot_lev = None
    elif len(levels) == 1:
        tot_lev = levels*len(z) 
    else:
        tot_lev = levels
        
    #Define figures and x and y data
    unique_theta = heat_map_data.get_unique_theta()
    theta_pts = int(np.sqrt(len(unique_theta)))
    test_mesh = unique_theta.reshape(theta_pts,theta_pts, -1).T
    xx , yy = test_mesh #NxN, NxN
    z = [heat_map_data.sse_mean, heat_map_data.sse_var, heat_map_data.ei]
    titles = ["sse", "sse var", "ei"]
    #Assert sattements
#     assert len(z) == len(titles), "Equal number of data matricies and titles must be given!"
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    
    #Make figures and define number of subplots  
    subplots_needed = len(idcs_to_plot)
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    #Print the title
    fig.suptitle(title, weight='bold', fontsize=18)
    
    #Set plot details
    #Loop over number of subplots
    for i in idcs_to_plot:
        z[i] = z[i].reshape(theta_pts,theta_pts).T
        #Assert statements
        assert z[i].shape==xx.shape, "Array z must be NxN"
        assert isinstance(z[i], np.ndarray)==True, "Heat map values must be numpy arrays"
        assert isinstance(titles[i], str)==True, "Title must be a string" 
        
        #Create a colormap and colorbar for each subplot
        cs_fig = ax[i].contourf(xx, yy,z[i], levels = 900, cmap = "autumn")
        if np.amax(abs(z[i])) < 1e-1 or np.amax(abs(z[i])) > 1000:
            cbar = plt.colorbar(cs_fig, ax = ax[i], format='%.2e')
        else:
            cbar = plt.colorbar(cs_fig, ax = ax[i], format = '%2.2f')
        
        #Create a line contour for each colormap
        if levels is not None:
            cs2_fig = ax[i].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
            ax[i].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=10, inline=1)
    
        #plot true, best, and training values
        ax[i].scatter(theta_true[0],theta_true[1], color="blue", label = "True", s=100, marker = (5,1))
        ax[i].scatter(train_theta[:,0],train_theta[:,1], color="green",s=25, label = "Training", marker = "x")
        ax[i].scatter(theta_obj_min[0],theta_obj_min[1], color="white", s=90, label = "Min Obj", marker = ".", edgecolor= "k", linewidth=0.3)
        ax[i].scatter(theta_ei_max[0],theta_ei_max[1], color="black", s=25, label = "Max EI", marker = ".")
        
        xlabel = r'$\mathbf{'+ param_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ param_names[1]+ '}$'
        
        plot_details(ax[i], xx, yy, xlabel, ylabel, titles[i], xbins, ybins, other_fontsize)
        
        #Get legend information
        if i == len(z)-1:
            handles, labels = ax[i].get_legend_handles_labels()     
          
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", bbox_to_anchor=(1.0, 0.95), borderaxespad=0)

    
    return plt.show()
