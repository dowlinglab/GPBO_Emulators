from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from .GPBO_Classes_New import Data
from.analyze_data import analyze_SF_data_for_plot, analyze_sse_min_sse_ei, analyze_thetas, get_best_data, get_median_data, get_mean_data, analyze_heat_maps, get_mean_med_best_over_sf


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
        

def create_subplots(num_subplots, sharex = "row"):
    """
    Creates Subplots based on the amount of data
    
    Parameters
    ----------
    num_subplots: int, total number of needed subplots
    
    Returns
    -------
    fig: matplotlib.figure, The figure you are plotting
    ax: matplotlib.axes.Axes, 1D array of axes
    len(ax): The number of axes generated total
    """
    assert num_subplots >= 1, "Number of subplots must be at least 1"
    #Make figures and define number of subplots  
    #If you are making more than one figure
    if num_subplots > 1:
        #Make enough rows and columns and get close to equal number of each
        row_num = int(np.floor(np.sqrt(num_subplots)))
        col_num = int(np.ceil(num_subplots/row_num))
        assert row_num * col_num >= num_subplots, "row * col numbers must be at least equal to number of graphs"
        #Creat subplots
        fig, axes = plt.subplots(nrows = row_num, ncols = col_num, figsize = (col_num*6,row_num*6), squeeze = False, sharex = sharex)
        ax = axes.reshape(row_num * col_num)
    else:
        #One subplot if num_subplots = 1
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6,6), squeeze = False, sharex = "row")
        ax = axes.reshape(1)
        
    return fig, ax, len(ax)
    
def subplot_details(ax, plot_x, plot_y, xlabel, ylabel, title, xbins, ybins, fontsize):
    """
    Function for setting plot settings
    
    Parameters
    ----------
    plot_x: ndarray, The x data for plotting
    plot_y: ndarray, The y data for plotting
    xlabel: str or None, the label for the x axis
    ylabel: str or None, the label for the y axis
    title: str or None, The subplot title
    xbins: int, Number of x bins
    ybins: int, Number of y bins
    fontsize: int, fontsize of letters in the subplot
    """
    #Group inputs by type
    none_str_vars = [title, xlabel, ylabel]
    int_vars = [xbins, ybins, fontsize]
    arr_vars = [plot_x, plot_y]
    
    #Assert Statements
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title, xlabel, and ylabel must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, and fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive"
    assert all(isinstance(var, (np.ndarray,pd.core.series.Series)) or var is None for var in arr_vars), "plot_x, plot_y must be np.ndarray or pd.core.series.Series or None"
    
    #Set title, label, and axes
    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontweight='bold')   
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize,fontweight='bold')
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold')
        
    #Turn on tick parameters and bin number
    ax.xaxis.set_tick_params(labelsize=fontsize, direction = "in")
    ax.yaxis.set_tick_params(labelsize=fontsize, direction = "in")
    ax.locator_params(axis='y', nbins=ybins)
    ax.locator_params(axis='x', nbins=xbins)
    ax.minorticks_on() # turn on minor ticks
    ax.tick_params(which="minor",direction="in",top=True, right=True)
    
    #Set bounds and aspect ratio
    
    if plot_x is not None and not np.isclose(np.min(plot_x), np.max(plot_x), rtol = 1e-6):        
        ax.set_xlim(left = np.min(plot_x), right = np.max(plot_x))
 
    ax.set_box_aspect(1)
    
    if plot_y is not None and np.min(plot_y) == 0:
        ax.set_ylim(bottom = np.min(plot_y)-0.05, top = np.max(plot_y)+0.05)
#     elif np.isclose(np.min(plot_y), np.max(plot_y), rtol =1e-6) == False:
#         ax.set_ylim(bottom = np.min(plot_y)-abs(np.min(plot_y)*0.05), top = np.max(plot_y)+abs(np.min(plot_y)*0.05))
#         print(np.min(plot_y)-abs(np.min(plot_y)*0.05), np.max(plot_y)+abs(np.min(plot_y)*0.05))
#     else:
#         ax.set_ylim(bottom = np.min(plot_y)/2, top = np.max(plot_y)*2)    
    return ax
         
def set_plot_titles(fig, title, x_label, y_label, title_fontsize = 24, other_fontsize = 20):
    """
    Helper function to set plot titles and labels for figures with subplots
    """
    
    if title_fontsize is not None:
        fig.suptitle(title, weight='bold', fontsize=title_fontsize)
    if x_label is not None:
        fig.supxlabel(x_label, fontsize=other_fontsize,fontweight='bold')
    if y_label is not None:
        fig.supylabel(y_label, fontsize=other_fontsize,fontweight='bold')
        
    return
    
    
def plot_2D_Data_w_BO_Iter(data, data_names, data_true, xbins, ybins, title, x_label, y_label, log_data = False, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots 2D values of the same value on multiple subplots
    
    Parameters
    -----------
    data: ndarray (n_runs x n_iters x n_params), Array of data from bo workflow runs
    data_names: list of str, List of data names
    data_true: list/ndarray of float/int or None, The true values of each parameter
    xbins: int, Number of bins for x
    ybins: int, Number of bins for y
    title: str or None, Title of graph
    x_label: str or None, title of x-axis
    y_label: str or None, title of y-axis
    title_fontsize: int, fontisize for title. Default 24
    other_fontsize: int, fontisize for other values. Default 20
    save_path: str or None, Path to save figure to. Default None (do not save figure).
    """
    #Assert Statements
    none_str_vars = [x_label, y_label, title, save_path]
    int_vars = [xbins, ybins, title_fontsize, other_fontsize]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title, xlabel, save_path, and ylabel must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive"
    assert isinstance(data, np.ndarray) or data is None, "data must be np.ndarray"
    assert len(data.shape) == 3, "data must be a 3D tensor"
    assert isinstance(data_true, (list, np.ndarray)) or data_true is None, "data_true must be list, ndarray, or None"
    if data_true is not None:
         assert all(isinstance(item, (float,int)) for item in data_true), "data_true elements must be float/int"
    assert isinstance(data_names, list), "data_names must be list"
    assert all(isinstance(item, str) for item in data_names), "data_names elements must be string"
   
    
    #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
    subplots_needed = data.shape[-1]
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)

    #Loop over different hyperparameters (number of subplots)
    for i in range(num_subplots):
        #If you still have data to plot
        if i < data.shape[-1]:
            #The index of the data is i, and one data type is in the last row of the data
            one_data_type = data[:,:,i]
            #Loop over all runs
            for j in range(one_data_type.shape[0]):
                #Create label based on run #
                label = "Run: "+str(j+1) 
                #Remove elements that are numerically 0
                data_df_run = pd.DataFrame(data = one_data_type[j])
                data_df_j = data_df_run.loc[(abs(data_df_run) > 1e-14).any(axis=1),0]
                data_df_i = data_df_run.loc[:,0] #Used to be df_i
                #Ensure we have at least 2 elements to plot
                if len(data_df_j) < 2:
                    data_df_j = data_df_i[0:int(len(data_df_j)+2)] #+2 for stopping criteria + 1 to include last point
                #Define x axis
                bo_len = len(data_df_j)
                bo_space = np.linspace(1,bo_len,bo_len)
                #Set appropriate notation
                if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                    ax[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
                #Plot data
                if log_data == True:
                    data_df_j = np.log(data_df_j)
                ax[i].step(bo_space, data_df_j, label = label)
                #Plot true value if applicable
                if data_true is not None and j == one_data_type.shape[0] - 1:
                    ax[i].axhline(y=data_true[i], color = "red", linestyle='-', label = "True Value")
                #Set plot details 
                title = r'$'+ data_names[i]+ '$'
                subplot_details(ax[i], bo_space, data_df_j, None, None, title, xbins, ybins, other_fontsize)

        #Set axes off if it's an extra
        else:
            ax[i].set_axis_off()
            
        #Fetch handles and labels on last iteration
        if i == num_subplots-1:
            handles, labels = ax[data.shape[-1]-1].get_legend_handles_labels()
            
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  

    return

def plot_compare_method_ei_sse(file_path_list, bo_method_list, run_num_list, string_for_df, data_names, xbins, ybins, title, x_label, y_label, log_data, title_fontsize, other_fontsize, save_path):
    """
    Plots 5 value plots for EI, SSE, Min SSE, and EI values vs BO iter for all 5 methods
    """
     #Assert Statements
    none_str_vars = [x_label, y_label, title, save_path]
    int_vars = [xbins, ybins, title_fontsize, other_fontsize]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title, xlabel, save_path, and ylabel must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive"
    assert isinstance(data_names, list) or data_names is None, "data_names must be list or None"
    if isinstance(data_names, list):
        assert all(isinstance(item, str) for item in data_names), "data_names elements must be string"
    
    #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
    subplots_needed = len(file_path_list)
    fig, ax, num_subplots = create_subplots(subplots_needed, sharex = False)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)
    
    #Loop over different hyperparameters (number of subplots)
    for i in range(num_subplots):
        #If you still have data to plot
        if i < subplots_needed:
            #Get data
            data, data_true = analyze_sse_min_sse_ei(file_path_list[i], run_num_list[i], string_for_df)

            #The index of the data is i, and one data type is in the last row of the data
            one_data_type = data
            #Loop over all runs
            for j in range(one_data_type.shape[0]):
                #Create label based on run #
                label = "Run: "+str(j+1) 
                #Remove elements that are numerically 0
                data_df_run = pd.DataFrame(data = one_data_type[j])
                data_df_j = data_df_run.loc[(abs(data_df_run) > 1e-14).any(axis=1),0]
                data_df_i = data_df_run.loc[:,0] #Used to be data_df_i
                #Ensure we have at least 2 elements to plot
                if len(data_df_j) < 2:
                    data_df_j = data_df_i[0:int(len(data_df_j)+2)] #+2 for stopping criteria + 1 to include last point
                #Define x axis
                bo_len = len(data_df_j)
                bo_space = np.linspace(1,bo_len,bo_len)
                #Set appropriate notation
                if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                    ax[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
                #Plot data
                if log_data == True:
                    data_df_j = np.log(data_df_j)
                ax[i].step(bo_space, data_df_j, label = label)
                #Plot true value if applicable
                if data_true is not None and j == one_data_type.shape[0] - 1:
                    ax[i].axhline(y=data_true[i], color = "red", linestyle='-', label = "True Value")
                #Set plot details 
                title = bo_method_list[i]
                subplot_details(ax[i], bo_space, data_df_j, None, None, title, xbins, ybins, other_fontsize)
        

        #Set axes off if it's an extra
        else:
            ax[i].set_axis_off()
            
        #Fetch handles and labels on last iteration
        if i == 0:
            handles, labels = ax[0].get_legend_handles_labels()
    
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  
        
    return
                    
def plot_compare_mean_med_best(df, cs_name, theta_true, param_name_str, xbins, ybins, title, log_data, title_fontsize, other_fontsize, save_path):
    """
    Compares Number of Evaluations, L2 Norm of Theta, and SSE of data using a mean, median, or mode statistic
    """
    #Set Axes Titles
    if log_data is True:
        ax_title_list = [r'$\log(e(\theta))$', r'$\vert\vert \theta - \theta_{true} \vert \vert_{2}$', 'Evaluations (Best)', 'Evaluations (Termination)']
    else:
        ax_title_list = [r'$e(\theta)$', r'$\vert\vert \theta - \theta_{true} \vert \vert_{2}$', 'Evaluations (Best)', 'Evaluations (Termination)']
    ax_df_key_list = ["Min Obj Cum.", "L2 Norm Theta", "BO Iter", "Max Evals"] 
    
    #Get BO method names
    names = df['BO Method'].unique().tolist()
    sep_fact_list = df['Sep Fact'].unique()
    x_data = np.array(list(map(float, sep_fact_list))) #Make sure SF values are float
    titles = ["Mean", "Median", "Median of Best", "Best"]
    choices = ["mean", "median", "median_best", "best"]

    #Get df containing the best value at each sf for each method
    df_mean, df_med, df_med_best, df_best = get_mean_med_best_over_sf(df, cs_name, theta_true, param_name_str)

    #Loop over mean, best, and median
    for choice,title in zip(choices,titles):        
        #Number of subplots is number of values to plot. This will always be 4 for this graph
        subplots_needed = 4 #log(SSE), L2 norm (Theta), iters for best sse, and termination
        fig, ax, num_subplots = create_subplots(subplots_needed)
        
        #Print the title and labels as appropriate
        set_plot_titles(fig, title, "Separation Factor", None, title_fontsize, other_fontsize)
        
        if choice == "mean":
            df_plot = df_mean
        elif choice == "median":
            df_plot = df_med
        elif choice == "median_best":
            df_plot = df_med_best
        elif choice == "best":
            df_plot = df_best
        else:
            raise Warning("choice must be 'mean', 'median', 'median_best', or 'best'")
                    
        #Loop over different subplots (number of subplots)
        for i in range(num_subplots): 
            y_data_list = []
            #Loop over names
            for name in names:
                df_meth_sf = df_plot[df_plot['BO Method'] == name]                 
                #If plotting log sse, make sure the data is log
                if i == 0:
                    #Make unlogged data from 1A, 2A and 2C in log form
                    if log_data is True and "B" not in name:
                        ax[i].plot(x_data, np.log(df_meth_sf[ax_df_key_list[i]]), label = str(name))
                        y_data = np.log(df_meth_sf[ax_df_key_list[i]])               
                    #Make log data from 2B and 1B in normal form
                    elif log_data is False and "B" in name:
                        ax[i].plot(x_data, np.exp(df_meth_sf[ax_df_key_list[i]].astype(np.float64)), label = str(name))
                        y_data = np.exp(df_meth_sf[ax_df_key_list[i]].astype(np.float64))
                    #Otherwise plot it as is
                    else:
                        ax[i].plot(x_data, df_meth_sf[ax_df_key_list[i]], label = str(name))
                        y_data = df_meth_sf[ax_df_key_list[i]]
                    
                #Add +1 to BO iter for number of iterations when plotting
                elif i == 2:
                    ax[i].plot(x_data, df_meth_sf[ax_df_key_list[i]] + 1, label = str(name))
                    y_data = df_meth_sf[ax_df_key_list[i]] + 1
                  
                #Otherwise plot the number of iterations until termination as is
                else:
                    ax[i].plot(x_data, df_meth_sf[ax_df_key_list[i]], label = str(name))
                    y_data = df_meth_sf[ax_df_key_list[i]]
                
                y_data_list.append(y_data)
                
            #Set plot details 
            title = ax_title_list[i]
            subplot_details(ax[i], x_data, np.array(y_data_list), None, None, title, xbins, ybins, other_fontsize)

            #Fetch handles and labels on 1st iteration
            if i == 0:
                handles, labels = ax[0].get_legend_handles_labels()

        #Plots legend and title
        plt.tight_layout()
        fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)

        #save or show figure
        if save_path is None:
            plt.show()
            plt.close()
        else:
            save_fig(save_path + choice, ext='png', close=True, verbose=False)  

    return

def plot_all_SF_data(df, xbins, ybins, title, x_label, y_label, log_data = False, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots the separation factor analysis
    
    Parameters
    -----------
    df: pd.DataFrame. DataFrame of all SF study data
    xbins: int, Number of bins for x
    ybins: int, Number of bins for y
    title: str or None, Title of graph
    x_label: str or None, title of x-axis
    y_label: str or None, title of y-axis
    title_fontsize: int, fontisize for title. Default 24
    other_fontsize: int, fontisize for other values. Default 20
    save_path: str or None, Path to save figure to. Default None (do not save figure).
    """
    #Assert Statements
    none_str_vars = [x_label, y_label, title, save_path]
    int_vars = [xbins, ybins, title_fontsize, other_fontsize]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title, xlabel, save_path, and ylabel must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive"
    assert isinstance(log_data, bool), "log_data must be bool!"
    
    sep_facts = df["Sep Fact"].unique()
    sep_facts_float = np.array(list(map(float, sep_facts)))
    bo_methods = df["BO Method"].unique()
   
    #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
    subplots_needed = len(bo_methods)
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)

    #Loop over different methods (number of subplots)
    for i in range(num_subplots):
        #If you still have data to plot
        if i < subplots_needed:
            #Get the values 
            y_data, data_names = analyze_SF_data_for_plot(df, bo_methods[i], sep_facts)  
            ax[i].plot(sep_facts_float, y_data[:,0], color="green", label = data_names[0], zorder = 1)        
            ax[i].plot(sep_facts_float, y_data[:,1], color="red", label = data_names[1], zorder = 2)
            
            #Set title and subplot details
            title = df["BO Method"].unique()[i]
            subplot_details(ax[i], sep_facts_float, y_data, None, None, title, xbins, ybins, other_fontsize)

        #Set axes off if it's an extra
        else:
            ax[i].set_axis_off()
            
        #Fetch handles and labels on last iteration
        if i == 0:
            handles, labels = ax[0].get_legend_handles_labels()
     
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  

    return

def plot_SF_data(x_data, y_data, data_names, data_true, xbins, ybins, title, x_label, y_label, log_data = False, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots 2D values of the same value on multiple subplots
    
    Parameters
    -----------
    x_data: ndarray, Array of X data
    y_data: ndarray, Array of y data
    data_names: list of str, List of data names
    data_true: list/ndarray of float/int or None, The true values of each parameter
    xbins: int, Number of bins for x
    ybins: int, Number of bins for y
    title: str or None, Title of graph
    x_label: str or None, title of x-axis
    y_label: str or None, title of y-axis
    title_fontsize: int, fontisize for title. Default 24
    other_fontsize: int, fontisize for other values. Default 20
    save_path: str or None, Path to save figure to. Default None (do not save figure).
    """
    #Assert Statements
    none_str_vars = [x_label, y_label, title, save_path]
    int_vars = [xbins, ybins, title_fontsize, other_fontsize]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title, xlabel, save_path, and ylabel must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive"
    assert isinstance(x_data, (list, np.ndarray)), "x_data must be list of float or ndarray"
    assert isinstance(y_data, np.ndarray) or data is None, "y_data must be np.ndarray"
    assert len(y_data.shape) == 2, "y_data must be a 2D array"
    assert isinstance(data_true, (list, np.ndarray)) or data_true is None, "data_true must be list, ndarray, or None"
    if data_true is not None:
         assert all(isinstance(item, (float,int)) for item in data_true), "data_true elements must be float/int"
    assert isinstance(data_names, list), "data_names must be list"
    assert all(isinstance(item, str) for item in data_names), "data_names elements must be string"
   
    #Set figure details. Set bins turn on ticks
    fig = plt.figure(figsize = (6,6))
    plt.xticks(fontsize=other_fontsize)
    plt.yticks(fontsize=other_fontsize)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=ybins)
    plt.locator_params(axis='x', nbins=xbins)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)

    #plot training data, testing data, and true values
    if log_data == True:
        y_data = np.log(y_data)
        
    plt.plot(x_data, y_data[:,0], color="green", label = data_names[0], zorder = 1)        
    plt.plot(x_data, y_data[:,1], color="red", label = data_names[1], zorder = 2)
    
    #Set plot labels
    plt.legend(loc= "upper left", bbox_to_anchor=(1.05, 0.95), borderaxespad=0)
    plt.xlabel(x_label, fontsize=other_fontsize, fontweight='bold')
    plt.ylabel(y_label, fontsize=other_fontsize, fontweight='bold')

    #Set axis limits based on the maximum and minimum of the parameter search space      
    x_lim_l = np.amin(x_data)
    y_lim_l = np.amin(y_data)
    x_lim_u = np.amax(x_data)
    y_lim_u = np.amax(y_data)
    plt.xlim((x_lim_l,x_lim_u))
    plt.ylim((y_lim_l,y_lim_u))
    #Set plot title
    plt.title(title, fontsize=title_fontsize, fontweight='bold')
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  

    return fig

def plot_x_vs_y_given_theta(data, exp_data, train_data, test_data, xbins, ybins, title, x_label, y_label, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots x data vs y data for any given parameter set theta
    
    Parameters
    ----------
    data: Instance of Data,
    
    """
    subplots_needed = data.get_dim_x_vals()
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)
    
    
    #Loop over different hyperparameters (number of subplots)
    for i in range(num_subplots):
        #If you still have data to plot
        if i < data.get_dim_x_vals():
            #The index of the data is i, and one data type is in the last row of the data
            X_space = data.x_vals[:,i]
            ax[i].plot(X_space, data.gp_mean, lw=2, label="GP_mean", color = "blue")
            ax[i].scatter(X_space, data.y_vals, label = "Y_sim", color = "black")
            if train_data is not None:
                ax[i].scatter(train_data.x_vals[:,i], train_data.y_vals, color = "green",  s=150, marker = "x", label = "Training")
            if test_data is not None:
                ax[i].scatter(test_data.x_vals[:,i], test_data.y_vals, color = "red", s=100, marker = "x", label = "Testing")
            if exp_data is not None:
                ax[i].scatter(exp_data.x_vals[:,i], exp_data.y_vals, color = "black", marker = "o", label = "Experiment")
            ax[i].fill_between(
                X_space,
                data.gp_mean - 1.96 * np.sqrt(data.gp_var),
                data.gp_mean + 1.96 * np.sqrt(data.gp_var),
                alpha=0.3 )
            subplot_details(ax[i], X_space, None, None, None, "X_exp dim " + str(i+1) , xbins, ybins, other_fontsize)
        #Set axes off if it's an extra
        else:
            ax[i].set_axis_off()
            
        #Fetch handles and labels on last iteration
        if i == num_subplots-1:
            handles, labels = ax[i].get_legend_handles_labels()
            
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  
    
    return

def plot_theta_vs_y_given_x(data, theta_idx, data_names, exp_data, train_data, test_data, xbins, ybins, title, x_label, y_label, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots theta data vs y data for any given parameter set theta
    
    Parameters
    ----------
    data: Instance of Data,
    
    """
    subplots_needed = len(data)
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)
    
    #Loop over different hyperparameters (number of subplots)
    for i in range(num_subplots):
        #If you still have data to plot
        if i < subplots_needed:
            data = data[i]
            #The index of the data is i, and one data type is in the last row of the data
            theta_space = data.theta_vals[:,theta_idx]
            ax[i].plot(theta_space, data.gp_mean, lw=2, label="GP_mean", color = "blue")
            ax[i].plot(theta_space, data.y_vals, label = "Y_sim", color = "black", linestyle = "--")
            if train_data is not None:
                ax[i].scatter(train_data.theta_vals[:,theta_idx], train_data.y_vals, color = "green", s=150, marker = "x", label = "Train")
            if test_data is not None:
                ax[i].scatter(test_data.theta_vals[:,theta_idx], test_data.y_vals, color = "red", s=100, marker = "x", label = "Test")
            if exp_data is not None:
                ax[i].scatter(exp_data.theta_vals[:,theta_idx], exp_data.y_vals, color = "black", marker = "o", label = "Exp")

            ax[i].fill_between(
               theta_space,
               data.gp_mean - 1.96 * np.sqrt(data.gp_var),
               data.gp_mean + 1.96 * np.sqrt(data.gp_var),
               alpha=0.3 )
            
            subplot_details(ax[i], theta_space, None, None, None, data_names[i] , xbins, ybins, other_fontsize)
        #Set axes off if it's an extra
        else:
            ax[i].set_axis_off()
            
        #Fetch handles and labels on last iteration
        if i == num_subplots-1:
            handles, labels = ax[i].get_legend_handles_labels()
            
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  
    
    return

#These parameters may need to change
def plot_train_test_val_data(train_data, test_data, val_data, param_names, idcs_to_plot, x_exp, xbins, ybins, zbins, title, title_fontsize = 24, other_fontsize = 20, save_path = None):
    '''
    UPDATE ME LATER
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
    assert len(idcs_to_plot) in [2,3], "idcs_to_plot must be a length 2 or 3"
    
    #If there are less 2 parameters, plot in 2D
    if len(idcs_to_plot) == 2:
        i1, i2 = idcs_to_plot
        #Set figure details. Set bins turn on ticks
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
        if val_data is not None:
            plt.scatter(val_data[:,i1],val_data[:,i2], color="blue",s=20, label = "Validation", marker = "D", zorder = 3)
        #How to plot theta true given that a combination of x and theta can be chosen?
#         plt.scatter(true_params[i1],true_params[i2], color="blue", label = "True argmin"+r'$(e(\theta))$', s=100, marker=(5,1), zorder = 3)
        #Set plot labels
        plt.legend(loc= "upper left", bbox_to_anchor=(1.05, 0.95), borderaxespad=0)
        x_label = r'$\mathbf{'+ param_names[i1] +'}$'
        y_label = r'$\mathbf{'+ param_names[i2] +'}$'

        plt.xlabel(x_label, fontsize=other_fontsize, fontweight='bold')
        plt.ylabel(y_label, fontsize=other_fontsize, fontweight='bold')
        
        #Set axis limits based on the maximum and minimum of the parameter search space      
        x_lim_l = np.amin(np.concatenate((train_data[:,i1], test_data[:,i1]), axis = None))
        y_lim_l = np.amin(np.concatenate((train_data[:,i2], test_data[:,i2]), axis = None))
        x_lim_u = np.amax(np.concatenate((train_data[:,i1], test_data[:,i1]), axis = None))
        y_lim_u = np.amax(np.concatenate((train_data[:,i2], test_data[:,i2]), axis = None))
        plt.xlim((x_lim_l,x_lim_u))
        plt.ylim((y_lim_l,y_lim_u))
        #Set plot title
        plt.title(title, fontsize=title_fontsize, fontweight='bold')
 
    #Otherwise print in 3D
    elif len(idcs_to_plot) == 3:
        i1, i2, i3 = idcs_to_plot
        #How should I go about plotting the true value?

        # Create the figure
        fig = plt.figure(figsize = (6,6))
        #Add 3D axes, set ticks, and bins
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
        ax.scatter(train_data[:,i1], train_data[:,i2], train_data[:,i3], color = "green", s=100, label = "Training", marker='o',zorder = 1)
        ax.scatter(test_data[:,i1],test_data[:,i2], test_data[:,i3], color="red", s=50, label = "Testing", marker = "x", zorder = 2)
        if val_data is not None:
            ax.scatter(val_data[:,i1],val_data[:,i2], val_data[:,i3], color="blue", s=40, label = "Validation", marker = "D", zorder = 3)
#         ax.scatter(p_true_3D_full[:,0], p_true_3D_full[:,1], p_true_3D_full[:,2], color="blue", label = "True argmin" + r'$(e(\theta))$', 
#                     s=100, marker = (5,1), zorder = 3)
        #Set Labels
        x_label = r'$\mathbf{'+ param_names[i1] +'}$'
        y_label = r'$\mathbf{'+ param_names[i2] +'}$'
        z_label = r'$\mathbf{'+ param_names[i3] +'}$'
#         plt.xlabel(r'$\mathbf{\theta_1}$', fontsize=16, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=other_fontsize, fontweight='bold', labelpad=int(other_fontsize))
        ax.set_ylabel(y_label, fontsize=other_fontsize, fontweight='bold', labelpad=int(other_fontsize))
        ax.set_zlabel(z_label, fontsize=other_fontsize, fontweight='bold', labelpad=int(other_fontsize))
        
        #How to set bounds given that you could plot any combination of theta and x dimensions on any axis?
#         x_lim_l = np.amin(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
#         y_lim_l = np.amin(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
#         x_lim_u = np.amax(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
#         y_lim_u = np.amax(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
#         plt.xlim((x_lim_l,x_lim_u))
#         plt.ylim((y_lim_l,y_lim_u))
        #Remove grid, set title and get legend
        ax.grid(False)
        plt.title(title, weight='bold', fontsize=title_fontsize)
        plt.tight_layout()
        plt.legend(loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 1.05), borderaxespad=0)
        
    else:
        print("Must be a 2D or 3D graph")
     
    #Save or show figure
    if save_path is not None:
        save_fig(save_path, ext='png', close=True, verbose=False)  
    else:
        plt.show()
        plt.close()

    return 

def parity_plot(y_data, y_sse_data, sse_data, method, log_plot, xbins, ybins, x_label, y_label, title, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Creates parity plots of data
    """      
        
    #Make figures and define number of subplots  
    if method.emulator == True:
        assert all(isinstance(var, Data) for var in [y_data, y_sse_data]), "y_data and y_sse_data must be type Data and not None!"
        subplots_needed = 2
        y_sim = y_data.y_vals
        gp_mean = y_data.gp_mean
        gp_stdev = np.sqrt(y_data.gp_var)
        sse_sim = y_sse_data.y_vals
        sse_mean = y_data.sse    
        sse_var = y_data.sse_var
    else:
        subplots_needed = 1
        assert isinstance(sse_data, Data), "sse_data must be type Data and not None!"
        sse_sim = sse_data.y_vals
        sse_mean = sse_data.sse  
        sse_var = sse_data.sse_var
        
        
    #If not getting log values    
    if log_plot == False:
        titles = ["sse", "y_sim"]
        #Change sse sim, mean, and stdev to not log for 1B and 2B
        if method.obj.value == 2:
            #SSE variance is var*(e^((log(sse)))^2
            sse_sim = np.exp(sse_sim)
            sse_mean = np.exp(sse_mean)
            sse_var = sse_var*sse_mean**2
            
    #If getting log values
    else:
        titles = ["log(sse)", "y_sim"]
        #Get log data from 1A, 2A, and 2C
        if method.obj.value == 1:            
            #SSE Variance is var/sse**2
            sse_var = sse_var/sse_mean**2
            sse_mean = np.log(sse_mean)
            sse_sim = np.log(sse_sim)
                
    sse_stdev = np.sqrt(sse_var)
        
    fig, ax, num_subplots = create_subplots(subplots_needed, sharex = False)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)
    
    #Set plot details
    #Loop over number of subplots
    for i in range(subplots_needed):
        if i < subplots_needed:
            #When we only want sse_data
            if i == 0:
                gp_upper = sse_mean + sse_stdev*1.96
                gp_lower = sse_mean - sse_stdev*1.96
                y_err = np.array([gp_lower, gp_upper])
                ax[i].errorbar(sse_sim, sse_mean, fmt="o", yerr=y_err, label = "GP", ms=10, zorder=1, mec = "green", mew = 1)
                ax[i].plot(sse_sim, sse_sim, label = "Sim" , zorder=2, color = "black")
                #Set plot details
                subplot_details(ax[i], sse_sim, sse_sim, None, None, titles[i], xbins, ybins, other_fontsize)
            else:
                #The index of the data is i, and one data type is in the last row of the data
                ax[i].errorbar(y_sim, gp_mean, yerr=1.96*gp_stdev, fmt = "o", label ="GP", ms=5, mec = "green", mew = 1, zorder= 1 )
                ax[i].plot(y_sim, y_sim, label = "Sim" , zorder=2, color = "black")
                #Set plot details
                subplot_details(ax[i], y_sim, y_sim, None, None, titles[i], xbins, ybins, other_fontsize)
           
            #Get legend information
            if i == subplots_needed-1:
                handles, labels = ax[i].get_legend_handles_labels()  
        else:
           #Set axes off if it's an extra
            ax[i].set_axis_off() 
              
    #Plots legend and title
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    plt.tight_layout()

    #Save or show figure
    if save_path is not None:
        save_fig(save_path, ext='png', close=True, verbose=False)  
    else:
        plt.show()
        plt.close()
    
    return

def plot_heat_maps(test_mesh, theta_true, theta_obj_min, theta_ei_max, train_theta, param_names, levels, idcs_to_plot, z, z_titles, xbins, ybins, zbins, title, title_fontsize = 24, other_fontsize = 20, cmap = "autumn", save_path = None):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        test_mesh: list of ndarray of length 2, Containing all values of the parameters for the heat map x and y. Gen with np.meshgrid()
        theta_true: ndarray or None, Containing the true input parameters in all dimensions
        theta_obj_min: ndarray or None, Containing the optimal input parameters predicted by the GP
        theta_ei_max: ndarray or None, Containing the input parameters predicted by the GP to have the best EI
        train_theta: ndarray or None, Training data for the iteration under consideration
        param_names: list of str, Parameter names. Length of 2
        levels: int, list of int or None, Number of levels to skip when drawing contour lines
        idcs_to_plot: list of int, Indecies of parameters to plot
        z: list of np.ndarrays, The list of values that will be plotted. Ex. SSE, SSE_Var, EI
        z_titles: list of str, The list of the names of the values in z
        xbins: int, Number of bins for x
        ybins: int, Number of bins for y
        zbins: int, Number of bins for z
        title: str or None, Title of graph
        title_fontsize: int, fontisize for title. Default 24
        other_fontsize: int, fontisize for other values. Default 20
        save_path: str or None, Path to save figure to. Default None (do not save figure).    
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    
    #Assert Statements
    none_str_vars = [title, save_path]
    none_ndarray_list = [theta_true, theta_obj_min, theta_ei_max, train_theta]
    int_vars = [xbins, ybins, zbins, title_fontsize, other_fontsize]
    list_vars = [test_mesh, z, z_titles, param_names]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title and save_path  must be string or None"
    assert all(isinstance(var, np.ndarray) or var is None for var in none_ndarray_list), "theta_true, theta_obj_min, theta_ei_max, train_theta must be array or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive" 
    assert isinstance(levels, (list, int)) or levels is None, "levels must be list of int, int, or None"
    if isinstance(levels, (list)) == True:
        assert all(isinstance(var, int) for var in levels), "If a list, levels must be list of int"
    assert all(isinstance(item, int) for item in idcs_to_plot), "idcs_to_plot elements must be int"
    assert all(isinstance(item, np.ndarray) for item in z), "z elements must be np.ndarray"
    assert all(isinstance(item, np.ndarray) for item in test_mesh), "test_mesh elements must be np.ndarray"
    assert all(isinstance(item, str) for item in z_titles), "z_title elements must be str"
    assert all(isinstance(item, str) for item in param_names), "param_names elements must be str"
    
    #Define plot levels
    if levels is None:
        tot_lev = None
    elif len(levels) == 1:
        tot_lev = levels*len(z) 
    else:
        tot_lev = levels
        
    #Assert sattements
    #Get x and y data from test_mesh
    xx , yy = test_mesh #NxN, NxN
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    
    #Make figures and define number of subplots  
    subplots_needed = len(z)
    fig, ax, num_subplots = create_subplots(subplots_needed)
    
    #Print the title
    if title is not None:
        fig.suptitle(title, weight='bold', fontsize=title_fontsize)

    #Set plot details
    #Loop over number of subplots
    for i in range(num_subplots):
        if i < len(z):
            #Assert statements
            assert z[i].shape==xx.shape, "Array z must be NxN"

            #Set number format based on magnitude
            if np.amax(abs(z[i])) < 1e-1 or np.amax(abs(z[i])) > 1000:
                fmt = '%.2e'           
            else:
                fmt = '%2.2f'

            #Create a colormap and colorbar for each subplot
            try:
                cs_fig = ax[i].contourf(xx, yy, z[i], levels = zbins, cmap = plt.cm.get_cmap(cmap))
            except:
                cs_fig = ax[i].contourf(xx, yy, z[i], levels = [np.max(z[i])-1e-9, np.max(z[i])], cmap = plt.cm.get_cmap(cmap))
            
            divider1 = make_axes_locatable(ax[i])
            cax1 = divider1.append_axes("right", size="8%", pad=0.2)
            cbar = plt.colorbar(cs_fig, ax = ax[i], cax = cax1, format=fmt)
#             cbar = ax.colorbar(cs_fig, ax = ax[i], format=fmt)
            cbar.ax.tick_params(labelsize=other_fontsize)

            #Create a line contour for each colormap
            if levels is not None:   
                cs2_fig = ax[i].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
                ax[i].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1, fmt = fmt)

            #plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax[i].scatter(theta_true[idcs_to_plot[0]],theta_true[idcs_to_plot[1]], color="blue", label = "True", s=200, marker = (5,1))
            if train_theta is not None:
                ax[i].scatter(train_theta[:,idcs_to_plot[0]],train_theta[:,idcs_to_plot[1]],color="green",s=100,label="Train",marker= "x")
            if theta_obj_min is not None:
                ax[i].scatter(theta_obj_min[idcs_to_plot[0]],theta_obj_min[idcs_to_plot[1]], color="white", s=175, label = "Min Obj", marker = ".", edgecolor= "k", linewidth=0.3)
            if theta_ei_max is not None:
                ax[i].scatter(theta_ei_max[idcs_to_plot[0]],theta_ei_max[idcs_to_plot[1]],color="black",s=150,label ="Max EI",marker = ".")

            #Define x and y labels
            if "theta" in param_names[0]:
                xlabel = r'$\mathbf{'+ "\\" + param_names[0]+ '}$'
                ylabel = r'$\mathbf{'+ "\\" + param_names[1]+ '}$'
            else:
                xlabel = r'$\mathbf{'+ param_names[0]+ '}$'
                ylabel = r'$\mathbf{'+ param_names[1]+ '}$'

            #Set plot details
            if title is not None:
                subplot_details(ax[i], xx, yy, xlabel, ylabel, z_titles[i], xbins, ybins, other_fontsize)
            else:
                subplot_details(ax[i], xx, yy, xlabel, ylabel, z_titles[i], xbins, ybins, other_fontsize)

            #Get legend information
            if i == len(z)-1:
                handles, labels = ax[i].get_legend_handles_labels()  
        else:
           #Set axes off if it's an extra
            ax[i].set_axis_off() 
              
    #Plots legend and title
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    fig.tight_layout()

    #Save or show figure
    if save_path is not None:          
        save_path = save_path + "Heat_Maps/" + param_names[0] + "-" + param_names[1] + "/" +'-'.join(z_titles)
        save_fig(save_path, ext='png', close=True, verbose=False)  
    else:
        plt.show()
        plt.close()
    
    return plt.show()

def compare_method_heat_maps(file_path_list, bo_methods_list, run_num_list, bo_iter_list, pair, z_choice, log_data, levels, xbins, ybins, zbins, title, title_fontsize = 24, other_fontsize = 20, cmap = "autumn", save_path = None):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        file_path_list: list of str, The file paths of data we want to make plots for
        bo_methods_list: str, The list of bo method names under consideration
        run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
        bo_iter_list: list of int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
        pair: int or tuple of str, The pair of data parameters
        z_choice: str, "sse_sim", "sse_mean", "sse_var", or "ei". The values that will be plotted
        levels: int, list of int or None, Number of levels to skip when drawing contour lines
        xbins: int, Number of bins for x
        ybins: int, Number of bins for y
        zbins: int, Number of bins for z
        title: str or None, Title of graph
        title_fontsize: int, fontisize for title. Default 24
        other_fontsize: int, fontisize for other values. Default 20
        save_path: str or None, Path to save figure to. Default None (do not save figure).    
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    
    #Assert Statements
    none_str_vars = [title, save_path]
    int_vars = [xbins, ybins, zbins, title_fontsize, other_fontsize]
    list_vars = [file_path_list, bo_methods_list, run_num_list, bo_iter_list]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title and save_path  must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive" 
    assert isinstance(levels, (list, int)) or levels is None, "levels must be list of int, int, or None"
    if isinstance(levels, (list)) == True:
        assert all(isinstance(var, int) for var in levels), "If a list, levels must be list of int"
    assert all(isinstance(item, int) for item in run_num_list), "run_num_list elements must be int"
    assert all(isinstance(item, int) for item in bo_iter_list), "bo_iter_list elements must be int"
    assert all(isinstance(item, str) for item in file_path_list), "file_path_list elements must be str"
    assert all(isinstance(item, str) for item in bo_methods_list), "bo_methods_list elements must be str"
    assert isinstance(z_choice, str), "z_choice must be string"
    assert z_choice in ['sse_sim', 'sse_mean', 'sse_var','ei'], "z_choice must be 'sse_sim', 'sse_mean', 'sse_var', or 'ei'"
    
    
    #Define plot levels
    if levels is None:
        tot_lev = None
    elif len(levels) == 1:
        tot_lev = levels*len(z) 
    else:
        tot_lev = levels
    
    #Make figures and define number of subplots based on number of files (different methods)  
    subplots_needed = len(file_path_list)
    fig, ax, num_subplots = create_subplots(subplots_needed)

    #Set plot details
    #Loop over number of subplots
    for i in range(num_subplots):
        if i < len(file_path_list):
            #Get data
            analysis_list = analyze_heat_maps(file_path_list[i], run_num_list[i], bo_iter_list[i], pair, log_data)
            sim_sse_var_ei, test_mesh, theta_true, theta_opt, theta_next, train_theta, plot_axis_names, idcs_to_plot = analysis_list
            sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
            
            #Assert sattements
            #Get x and y data from test_mesh
            xx , yy = test_mesh #NxN, NxN
            assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
            
            #Find z based on 
            if "sim" in z_choice:
                z = sse_sim
                title2 = r"$e(\theta)_{sim}$"
            elif "mean" in z_choice:
                z = sse_mean
                title2 = r"$e(\theta)_{gp}$"
            elif "var" in z_choice:
                z = sse_var
                title2 = r"$e(\theta)_{gp_{var}}$"
            elif "ei" in z_choice:
                z = ei
                title2 = r"$EI(\theta)$"
            else:
                raise Warning("choice must contain 'sim', 'mean', 'var', or 'ei'")
                
            #Assert statements
            assert z.shape==xx.shape, "Array z must be NxN"

            #Set number format based on magnitude
            if np.amax(abs(z)) < 1e-1 or np.amax(abs(z)) > 1000:
                fmt = '%.2e'           
            else:
                fmt = '%2.2f'

            #Create a colormap and colorbar for each subplot
            cs_fig = ax[i].contourf(xx, yy, z, levels = zbins, cmap = plt.cm.get_cmap(cmap))
            
            #Create a line contour for each colormap
            if levels is not None:   
                cs2_fig = ax[i].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
                ax[i].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1, fmt = fmt)
                
            #plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax[i].scatter(theta_true[idcs_to_plot[0]],theta_true[idcs_to_plot[1]], color="blue", label = "True", s=200, marker = (5,1))
            if train_theta is not None:
                ax[i].scatter(train_theta[:,idcs_to_plot[0]],train_theta[:,idcs_to_plot[1]],color="green",s=100,label="Train",marker= "x")
            if theta_opt is not None:
                ax[i].scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], color="white", s=175, label = "Min Obj", marker = ".", edgecolor= "k", linewidth=0.3)
            if theta_next is not None:
                ax[i].scatter(theta_next[idcs_to_plot[0]],theta_next[idcs_to_plot[1]],color="black",s=150,label ="Max EI",marker = ".")

            #Set plot details
            subplot_details(ax[i], xx, yy, None, None, bo_methods_list[i], xbins, ybins, other_fontsize)
            
            #Get legend information
            if i == len(file_path_list)-1:
                handles, labels = ax[i].get_legend_handles_labels() 

        else:
           #Set axes off if it's an extra
            ax[i].set_axis_off()
            
        #Make colorbar on last plot which is invisible
        if i == num_subplots - 1:
            if num_subplots == subplots_needed:
                if subplots_needed == 1:
                    cbar = plt.colorbar(cs_fig, ax = ax[i], cax = cax1, format=fmt)
                else:
                    side = "right"
                    pad = 0.2
            else:
                side = "left"
                pad = 0.01
            
                divider1 = make_axes_locatable(ax[i])
                cax1 = divider1.append_axes(side, size="5%", pad=pad)
                cbar = plt.colorbar(cs_fig, ax = ax[i], cax = cax1, format=fmt)
            cbar.ax.tick_params(labelsize=other_fontsize)
            if log_data == True:
                title2 = "log(" + title2 + ")"
#             cbar.set_label(title2, rotation=270, fontsize = other_fontsize)
            cbar.ax.set_ylabel(title2, fontsize=other_fontsize, fontweight='bold')
     
                      
    #Print the title
    if title is not None:
        title = title + " " + str(plot_axis_names)
        
    #Print the title and labels as appropriate
    #Define x and y labels
    #Define x and y labels
    if "theta" in plot_axis_names[0]:
        xlabel = r'$\mathbf{'+ "\\" + plot_axis_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ "\\" + plot_axis_names[1]+ '}$'
    else:
        xlabel = r'$\mathbf{'+ plot_axis_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ plot_axis_names[1]+ '}$'
                
#     xlabel = r'$\mathbf{'+ plot_axis_names[0]+ '}$'
#     ylabel = r'$\mathbf{'+ plot_axis_names[1]+ '}$'
    set_plot_titles(fig, title, xlabel, ylabel, title_fontsize, other_fontsize)
    
    #Plots legend and title
    if not subplots_needed == 5:
        fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.95), borderaxespad=0)
    else:
        fig.legend(handles, labels, loc= "lower right", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.1), borderaxespad=0)
    plt.tight_layout()

    #Save or show figure
    if save_path is not None:
        save_fig(save_path, ext='png', close=True, verbose=False)  
    else:
        plt.show()
        plt.close()
    
    return plt.show()

