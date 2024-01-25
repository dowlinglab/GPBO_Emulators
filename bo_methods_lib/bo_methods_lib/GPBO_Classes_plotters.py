from matplotlib import pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import json
from matplotlib import colormaps
from collections.abc import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from .GPBO_Classes_New import Data, Method_name_enum
from.analyze_data import analyze_sse_min_sse_ei, analyze_thetas, get_best_data, get_median_data, get_mean_data, analyze_heat_maps

import warnings
np.warnings = warnings


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
        

def create_subplots(num_subplots, sharex = "row", sharey = 'none'):
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
    assert isinstance(num_subplots, int), "Num subplots must be int"
    #Make figures and define number of subplots  
    #If you are making more than one figure, sharex is always true
    if num_subplots == 1:
        sharex = True

    #Make enough rows and columns and get close to equal number of each
    row_num = int(np.floor(np.sqrt(num_subplots)))
    col_num = int(np.ceil(num_subplots/row_num))
    assert row_num * col_num >= num_subplots, "row * col numbers must be at least equal to number of graphs"
    total_ax_num = row_num * col_num

    #Creat subplots
    gridspec_kw = {'wspace': 0.4, 'hspace': 0.2}
    fig, axes = plt.subplots(row_num, col_num, figsize = (col_num*6,row_num*6), squeeze = False, sharex = sharex, sharey = sharey)

    #Turn off unused axes
    for i, axs in enumerate(axes.flatten()):
        if i >= num_subplots:
            axs.axis('off')

    #Make plot mapping to map an axes to an iterable value
    plot_mapping = {}
    for i in range(row_num):
        for j in range(col_num):
            plot_number = i * col_num + j
            plot_mapping[plot_number] = (i, j)

    return fig, axes, total_ax_num, plot_mapping
    
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
        pad = 6 + 4*title.count("_")
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad = pad)   
    if xlabel is not None:
        pad = 6 + 4*xlabel.count("_")
        ax.set_xlabel(xlabel,fontsize=fontsize,fontweight='bold', labelpad = pad)
    if ylabel is not None:
        pad = 6 + 2*ylabel.count("_")
        ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold', labelpad = pad)
        
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

def make_plot_dict(log_data, title, xlabel, ylabel, line_levels, save_path=None, xbins=5, ybins=5, zbins=900, title_size=24, other_size=24, cmap = "autumn"):
    """
    Function to make dictionary for plotting specifics
    
    Parameters:
    -----------
        log_data: bool, plots data on natural log scale if True
        title: str or None, Title of plot
        xlabel: str or None, the x label of the plot
        ylabel: str or None, the y label of the plot
        line_levels: int, list of int or None, Number of zbins to skip when drawing contour lines
        save_path: str or None, Path to save figure to. Default None (do not save figure).
        xbins: int, Number of bins for x. Default 5
        ybins: int, Number of bins for y. Default 5
        zbins: int or None, Number of bins for z. Default 900
        title_size: int, fontisize for title. Default 24
        other_size: int, fontisize for other values. Default 20
        cmap: str, colormap for matplotlib to use for heat map generation
        
    Returns:
    --------
        plot_dict: dict, a dictionary of the plot details 
        
    Notes:
    -----
        plot_dict has keys "title", "log_data", "title_size", "other_size", "xbins", "ybins", "zbins", "save_path", "cmap", "line_levels", "xlabel", and "ylabel"
        
    """
    assert isinstance(cmap, str) and cmap in list(colormaps), "cmap must be a string in matplotlib.colormaps"
    assert isinstance(log_data, bool), "log_data must be bool"
    assert isinstance(save_path, list) or save_path is None, "save_path must be list of str or None"
    none_str_vars = [title, xlabel, ylabel]
    if save_path is not None:
        none_str_vars += [path for path in save_path]
    int_vars = [xbins, ybins, title_size, other_size]
    assert isinstance(zbins, int) or zbins is None, "zbins must be int > 3 or None"
    if isinstance(zbins, int):
        assert zbins > 3, "zbins must be int > 3 or None"
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title and save_path must be string or None"
    assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, title_fontsize, and other_fontsize must be int"
    assert all(var > 0 or var is None for var in int_vars), "xbins, ybins, title_size, and other_size must be positive int" 
    assert isinstance(line_levels, (list, int)) or line_levels is None, "line_levels must be list of int, int, or None"
    if isinstance(line_levels, (list)) == True:
        assert all(isinstance(var, int) for var in line_levels), "If a list, line_levels must be list of int"
        
    plot_dict = {"log_data":log_data, "title_size": title_size, "other_size":other_size, "xbins":xbins, "ybins":ybins, "zbins":zbins, 
                 "save_path":save_path, "cmap":cmap, "line_levels":line_levels, "title": title, "xlabel":xlabel, 
                 "ylabel":ylabel}
    
    return plot_dict
    
def plot_2D_Data_w_BO_Iter(data, data_names, data_true, plot_dict):
    """
    Plots 2D values of the same data type (ei, sse, min sse) on multiple subplots
    
    Parameters
    -----------
        data: ndarray (n_runs x n_iters x n_params), Array of data from bo workflow runs
        data_names: list of str, List of data names
        data_true: list/ndarray of float/int or None, The true values of each parameter
        plot_dict: dict, a dictionary of the plot details 
    """
    keys_to_check = ["title","log_data","title_size","other_size","xbins","ybins","save_path","xlabel","ylabel"]
    assert all(key in plot_dict for key in keys_to_check), "plot_dict must have all keys. Generate plot_dict with make_plot_dict()"
    keys_not_none = ["xlabel"]
    assert all(plot_dict[key] is not None for key in keys_not_none), "xlabel must not be None!"
    
    #Break down plot dict and check for correct things
    title = plot_dict["title"]
    log_data = plot_dict["log_data"]
    title_fontsize = plot_dict["title_size"]
    other_fontsize = plot_dict["other_size"]
    xbins = plot_dict["xbins"]
    ybins = plot_dict["ybins"]
    save_path = plot_dict["save_path"]
    x_label = plot_dict["xlabel"]
    y_label = plot_dict["ylabel"]
    
    #Assert Statements
    assert isinstance(data, np.ndarray) or data is None, "data must be np.ndarray"
    assert len(data.shape) == 3, "data must be a 3D tensor"
    assert isinstance(data_true, (list, np.ndarray)) or data_true is None, "data_true must be list, ndarray, or None"
    if data_true is not None:
         assert all(isinstance(item, (float,int)) for item in data_true), "data_true elements must be float/int"
    assert isinstance(data_names, (list, np.ndarray)), "data_names must be list or np.ndarray"
    assert all(isinstance(item, str) for item in data_names), "data_names elements must be string"
    
    #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
    subplots_needed = data.shape[-1]
    fig, axes, num_subplots, plot_mapping = create_subplots(subplots_needed, sharex = True)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)

    #Loop over different hyperparameters (number of subplots)
    for i, ax in enumerate(axes.flatten()):
        #Only plot data if axis is visible
        if i < subplots_needed:
            #The index of the data is i, and one data type is in the last row of the data
            #Grab the mapping from plot_mapping
            ax_row, ax_col = plot_mapping[i]
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
                    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
                #Plot data
                if log_data == True:
                    data_df_j = np.log(data_df_j)
                ax.step(bo_space, data_df_j, label = label)
                #Plot true value if applicable
                if data_true is not None and j == one_data_type.shape[0] - 1:
                    ax.axhline(y=data_true[i], color = "red", linestyle='--', label = "True Value")
                
                #Set plot details 
                title = r'$'+ data_names[i]+ '$'
                subplot_details(ax, bo_space, data_df_j, None, None, title, xbins, ybins, other_fontsize)
                
            if not log_data and data_true is None:
                ax.set_yscale("log")

        #Add legends and handles from last subplot that is visible
        if i == subplots_needed -1:
            handles, labels = axes[0, -1].get_legend_handles_labels()
            
    #Plots legend and title
    plt.tight_layout()
    fig.legend(handles, labels, loc= "upper left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.75), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  

    return

def plot_objs_all_methods(file_path_list, run_num_list, z_choices, plot_dict):
    """
    Plots EI, SSE, Min SSE, and EI values vs BO iter for all 6 methods
    
    Parameters
    -----------
        file_path_list: list of str, The file paths of data we want to make plots for
        run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
        z_choices:  list of str, list of strings "sse_sim", "sse_mean", "sse_var", and/or "ei". The values that will be plotted
        plot_dict: dict, a dictionary of the plot details 
        
    """
    keys_to_check = ["title","log_data","title_size","other_size","xbins","ybins","zbins","save_path","cmap","line_levels"]
    assert all(key in plot_dict for key in keys_to_check), "plot_dict must have all keys. Generate plot_dict with make_plot_dict()"
    keys_not_none = ["xlabel"]
    assert all(plot_dict[key] is not None for key in keys_not_none), "x_label must not be None!"
    
    #Break down plot dict and check for correct things
    title = plot_dict["title"]
    log_data = plot_dict["log_data"]
    title_fontsize = plot_dict["title_size"]
    other_fontsize = plot_dict["other_size"]
    xbins = plot_dict["xbins"]
    ybins = plot_dict["ybins"]
    save_path = plot_dict["save_path"]
    x_label = plot_dict["xlabel"]
    
    #Assert Statements
    list_vars = [file_path_list, run_num_list, z_choices]
    assert all(isinstance(var, Iterable) for var in list_vars), "file_path_list, run_num_list, and z_choices must be iterable"
    assert all(isinstance(item, int) for item in run_num_list), "run_num_list elements must be int"
    assert all(isinstance(item, str) for item in file_path_list), "file_path_list elements must be str"
    assert all(isinstance(item, str) for item in z_choices), "z_choices elements must be str"
    for i in range(len(z_choices)):
        assert z_choices[i] in ['min_sse','sse','ei'],"z_choices items must be 'min_sse', 'sse', or 'ei'"
    
    colors = ["red", "blue", "green", "purple", "darkorange", "deeppink"]
    method_names = ["Conventional", "Log Conventional", "Independence", "Log Independence", "Sparse Grid", "Monte Carlo"]
    #Number of subplots is 1 for each ei, sse, sse_sim etc.
    subplots_needed = len(z_choices)
    
    fig, axes, num_subplots, plot_mapping = create_subplots(subplots_needed, sharex = True)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, None, None, title_fontsize, other_fontsize)
    bo_len_max = 1
    #Loop over different methdods (number of subplots)
    for i in range(len(file_path_list)):     
        #Get data
        data, data_names, data_true, GPBO_method_val = analyze_sse_min_sse_ei(file_path_list[i], z_choices)
        #Create label based on method #
        label = method_names[GPBO_method_val-1] 
        
        #Loop over number of data types
        #Loop over different hyperparameters (number of subplots)
        for k, ax in enumerate(axes.flatten()):
            #Only plot data if axis is visible
            if k < subplots_needed:
#             for k in range(data.shape[-1]):
                #The index of the data type is k, and one data type is in the last row of the data
                one_data_type = data[:,:,k]
                 #Set run counter as 1 to start
                run_num_count = 1
                term_loop = False

                #loop as long as there are runs in the file
                while not term_loop:
                    j = run_num_count-1 #Iterable

                    #Remove elements that are numerically 0            
                    data_df_j = get_data_to_bo_iter_term(one_data_type[j])
                    #Define x axis
                    bo_len = len(data_df_j)
                    bo_space = np.linspace(1,bo_len,bo_len)

                    if bo_len > bo_len_max:
                        bo_len_max = bo_len

                    #Set appropriate notation
                    if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))

                    #Plot data
                    if log_data == True:
                        data_df_j = np.log(data_df_j)

                    #For the best result, print a solid line                    
                    if run_num_list[i] == j + 1:
                        ax.plot(bo_space, data_df_j, alpha = 1, color = colors[GPBO_method_val-1], label = label, drawstyle='steps')
                    else:
                        ax.step(bo_space, data_df_j, alpha = 0.2, color = colors[GPBO_method_val-1], linestyle='--', drawstyle='steps')

                    #Add 1 to run number and terminate if the total amount of runs is equal to the total amount
                    if run_num_count == one_data_type.shape[0]:
                        term_loop = True
                    run_num_count += 1

                #Set plot details
                bo_space_org = np.linspace(1,bo_len_max,100)
                subplot_details(ax, bo_space_org, None, x_label, rf"${data_names[k]}$", None, xbins, ybins, other_fontsize)

    #Get legend and handles
    handles, labels = axes[0,0].get_legend_handles_labels()
    if log_data == False:
        for ax in axes.flatten():
            ax.set_yscale("log")
    
    #Plots legend and title
    plt.tight_layout()
    
    fig.legend(handles, labels, loc= "center left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.60), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        for save_path_dir in save_path:
            save_path_to = save_path_dir + "Line_Plots/" + '_'.join(map(str, data_names)).replace(" ", "_").lower() + "_all_runs"
#             print(save_path_to)
            save_fig(save_path_to, ext='png', close=False, verbose=False)  
        plt.close() #Only close figure after for loop
        
    return

def get_data_to_bo_iter_term(data_all_iters):
    """
    Gets data that is not zero for plotting from data array
    """
    #Remove elements that are numerically 0
    data_df_run = pd.DataFrame(data = data_all_iters)
    data_df_j = data_df_run.loc[(abs(data_df_run) > 1e-14).any(axis=1),0]
    data_df_i = data_df_run.loc[:,0] #Used to be data_df_i
    #Ensure we have at least 2 elements to plot
    if len(data_df_j) < 2:
        data_df_j = data_df_i[0:int(len(data_df_j)+2)] #+2 for stopping criteria + 1 to include last point
        
    return data_df_j
                    
def plot_one_obj_all_methods(file_path_list, run_num_list, z_choices, plot_dict):
    """
    Plots SSE, Min SSE, and EI values vs BO iter for a single BO Method
    
    Parameters
    -----------
        file_path_list: list of str, The file paths of data we want to make plots for
        run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
        z_choices:  str, one of "sse_sim", "sse_mean", "sse_var", or "ei". The value that will be plotted
        plot_dict: dict, a dictionary of the plot details 
        
    """
    keys_to_check = ["title","log_data","title_size","other_size","xbins","ybins","zbins","save_path","cmap","line_levels"]
    assert all(key in plot_dict for key in keys_to_check), "plot_dict must have all keys. Generate plot_dict with make_plot_dict()"
    keys_not_none = ["xlabel", "ylabel"]
    assert all(plot_dict[key] is not None for key in keys_not_none), "x_label and y_label must not be None!"
    
    #Break down plot dict and check for correct things
    title = plot_dict["title"]
    log_data = plot_dict["log_data"]
    title_fontsize = plot_dict["title_size"]
    other_fontsize = plot_dict["other_size"]
    xbins = plot_dict["xbins"]
    ybins = plot_dict["ybins"]
    save_path = plot_dict["save_path"]
    x_label = plot_dict["xlabel"]
    y_label = plot_dict["ylabel"]
    
    #Assert Statements
    list_vars = [file_path_list, run_num_list]
    assert all(isinstance(var, Iterable) for var in list_vars), "file_path_list, run_num_list, and z_choices must be iterable"
    assert all(isinstance(item, int) for item in run_num_list), "run_num_list elements must be int"
    assert all(isinstance(item, str) for item in file_path_list), "file_path_list elements must be str"
    assert isinstance(z_choices, str), "z_choices must be str"
    assert z_choices in ['min_sse','sse','ei'],"z_choices must be one of 'min_sse', 'sse', or 'ei'"

    colors = ["red", "blue", "green", "purple", "darkorange", "deeppink"]
    method_names = ["Conventional", "Log Conventional", "Independence", "Log Independence", "Sparse Grid", "Monte Carlo"]
    
    #Number of subplots is number of runs in the run list (which will be the last spot of the shape parameter)
    subplots_needed = len(run_num_list)
    
    fig, ax, num_subplots, plot_mapping = create_subplots(subplots_needed, sharex = False)
    
    #Print the title and labels as appropriate
    set_plot_titles(fig, title, x_label, y_label, title_fontsize, other_fontsize)

    meth_bo_max_evals = np.zeros(len(run_num_list))
    
    #Loop over different files
    for i in range(len(file_path_list)):
        #Get data
        run_num_count = 1
        term_loop = False

        data, data_names, data_true, GPBO_method_val = analyze_sse_min_sse_ei(file_path_list[i], z_choices)
        #The index of the data is i, and one data type is in the last row of the data
        one_data_type = data

        #Set subplot index to the corresponding method value number
        ax_idx = int(GPBO_method_val - 1)
        ax_row, ax_col = plot_mapping[ax_idx]

        #Loop over all runs
        while not term_loop:
            j =  run_num_count - 1
            #Create label based on run #
            label = "Run: "+ str(j+1) 
            data_df_j = get_data_to_bo_iter_term(one_data_type[j])
            #Define x axis
            bo_len = len(data_df_j)
            bo_space = np.linspace(1,bo_len,bo_len)

            #Set appropriate notation
            if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                ax[ax_row, ax_col].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))

            #Plot data
            if log_data == True:
                data_df_j = np.log(data_df_j)

            #For the best result, print a solid line                    
            if run_num_list[ax_idx] == j + 1:
                ax[ax_row, ax_col].plot(bo_space, data_df_j, alpha = 1, color = colors[ax_idx], label = label, drawstyle='steps')
            else:
                ax[ax_row, ax_col].plot(bo_space, data_df_j, alpha = 0.2, color = colors[ax_idx], linestyle='--', drawstyle='steps')

            #Plot true value if applicable
            if data_true is not None and j == one_data_type.shape[0] - 1:
                ax[ax_row, ax_col].axhline(y=data_true[i], color = "black", linestyle='-', label = "True Value")

            #Set plot details 
            title = method_names[ax_idx]
            if bo_len > meth_bo_max_evals[ax_idx]:
                meth_bo_max_evals[ax_idx] = bo_len
                subplot_details(ax[ax_row, ax_col], bo_space, data_df_j, None, None, title, xbins, ybins, other_fontsize)

            #Add 1 to run number and terminate if the total amount of runs is equal to the total amount
            if run_num_count == one_data_type.shape[0]:
                term_loop = True
            run_num_count += 1

    handles, labels = ax[0,0].get_legend_handles_labels()
    for k, axs in enumerate(ax.flatten()):
        if k+1 < subplots_needed:
            h, l = axs.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if log_data == False:
            axs.set_yscale("log")

    #Plots legend and title
    plt.tight_layout()
    # fig.legend(handles, labels, loc= "center left", fontsize = other_fontsize, bbox_to_anchor=(1.0, 0.60), borderaxespad=0)
    
    #save or show figure
    if save_path is None:
        plt.show()
        plt.close()
    else:
        save_fig(save_path, ext='png', close=True, verbose=False)  
        
    return                  

def plot_x_vs_y_given_theta(data, exp_data, train_data, test_data, xbins, ybins, title, x_label, y_label, title_fontsize = 24, other_fontsize = 20, save_path = None):
    """
    Plots x data vs y data for any given parameter set theta
    
    Parameters
    ----------
    data: Instance of Data,
    
    """
    subplots_needed = data.get_dim_x_vals()
    fig, axes, ax, num_subplots = create_subplots(subplots_needed)
    
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
    fig, axes, ax, num_subplots = create_subplots(subplots_needed)
    
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

def plot_hms_gp_compare(file_path, run_num, bo_iter, pair, z_choices, plot_dict):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        file_path_list: list of str, The file paths of data we want to make plots for
        run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
        bo_iter_list: list of int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
        pair: int, The pair of data parameters. pair 0 is the 1st pair
        z_choice: str, "sse_sim", "sse_mean", "sse_var", or "ei". The values that will be plotted
        plot_dict: dict, dictionary of plotting options. Generate with make_plot_dict()
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    keys_to_check = ["title","log_data","title_size","other_size","xbins","ybins","zbins","save_path","cmap","line_levels"]
    assert all(key in plot_dict for key in keys_to_check), "plot_dict must have all keys. Generate plot_dict with make_plot_dict()"
    keys_not_none = ["line_levels", "zbins", "cmap"]
    assert all(plot_dict[key] is not None for key in keys_not_none), "line_levels, zbins, and cmap must not be None!"
    
    #Break down plot dict and check for correct things
    title = plot_dict["title"]
    log_data = plot_dict["log_data"]
    title_fontsize = plot_dict["title_size"]
    other_fontsize = plot_dict["other_size"]
    xbins = plot_dict["xbins"]
    ybins = plot_dict["ybins"]
    zbins = plot_dict["zbins"]
    save_path = plot_dict["save_path"]
    cmap = plot_dict["cmap"]
    levels = plot_dict["line_levels"]
    
    #Assert Statements
    assert isinstance(z_choices, Iterable), "z_choices must be Iterable"
    for z_choice in z_choices:
        assert z_choice in ['sse_sim', 'sse_mean', 'sse_var','ei'], "z_choices elements must be 'sse_sim', 'sse_mean', 'sse_var', or 'ei'"

    method_names = ["Conventional", "Log Conventional", "Independence", "Log Independence", "Sparse Grid", "Monte Carlo"]
    
    #Define plot levels
    if levels is None:
        tot_lev = None
    elif len(levels) == 1:
        tot_lev = levels*len(z_choices) 
    else:
        tot_lev = levels
    
    #Get all data for subplots needed
    get_ei = True if "ei" in z_choices else False
    #Get data
    analysis_list = analyze_heat_maps(file_path, run_num, bo_iter, pair, log_data, get_ei)
    sim_sse_var_ei, test_mesh, param_info_dict = analysis_list
    sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
    theta_true = param_info_dict["true"]
    theta_opt = param_info_dict["min_sse"]
    theta_next = param_info_dict["max_ei"]
    train_theta = param_info_dict["train"]
    plot_axis_names = param_info_dict["names"]
    idcs_to_plot = param_info_dict["idcs"]

    #Assert sattements
    #Get x and y data from test_mesh
    xx , yy = test_mesh #NxN, NxN
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"

    all_z_data = []
    all_z_titles = []
    #Find z based on z_choice
    for z_choice in z_choices:
        if "sse_sim" == z_choice:
            all_z_data.append(sse_sim)
            all_z_titles.append(r"$\mathbf{e(\theta)_{sim}}$")
        elif "sse_mean" == z_choice:
            all_z_data.append(sse_mean)
            all_z_titles.append(r"$\mathbf{e(\theta)_{gp}}$")
        elif "sse_var" == z_choice:
            all_z_data.append(sse_var)
            all_z_titles.append(r"$\mathbf{\sigma^2_{e(\theta)_{gp}}}$")
        elif "ei" == z_choice:
            all_z_data.append(ei)
            all_z_titles.append(r"$\mathbf{EI(\theta)}$")
        else:
            raise Warning("choice must contain 'sim', 'mean', 'var', or 'ei'")
            
    #Make figures and define number of subplots based on number of files (different methods)  
    subplots_needed = len(z_choices)

    fig, axes, num_subplots, plot_mapping = create_subplots(subplots_needed, sharex = True, sharey = True)

    #Get method value from json file
    with open(os.path.dirname(file_path) + "/signac_statepoint.json", 'r') as json_file:
        # Load the JSON data
        GPBO_method_val = json.load(json_file)["meth_name_val"]
        
    #Loop over number of subplots
    for i, ax in enumerate(axes.flatten()):
        if i < subplots_needed:
            #Get data for z_choice
            z = all_z_data[i]
            need_unscale = False

            #Unlog scale the data if vmin is 0 and log_data = True
            if np.min(z) == -np.inf or np.isnan(np.min(z)):
                need_unscale = True 
                if log_data:
                    warnings.warn("Cannot plot log scaled data! Reverting to original")
                    z = np.exp(all_z_data[i])

            #Create normalization
            vmin = np.nanmin(z)
            vmax = np.nanmax(z)
            #Check if data scales 3 orders of magnitude
            mag_diff = math.log10(abs(vmax)) - math.log10(abs(vmin)) > 3.0 if vmin > 0 else False
            

            if need_unscale == False and log_data:
                title2 = "log(" + all_z_titles[i] + ")"
            else:
                title2 = all_z_titles[i]

            #Choose an appropriate colormap and scaling based on vmin, vmax, and log_data
            #If not using log data, vmin > 0, and the data scales 3 orders+ of magnitude use log10 to view plots
            if not log_data and vmin > 0 and mag_diff:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
                cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), zbins)
    #             new_ticks = np.logspace(np.log10(vmin), np.log10(vmax), 7)
                new_ticks = matplotlib.ticker.LogLocator(numticks=7) #Set up to 12 ticks
                def custom_format(x, pos):
                    return f'{eval("10**" + str(int(np.log10(x))))}' if x != 0 else '0'

            #Otherwise do not scale
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False) 
                cbar_ticks = np.linspace(vmin, vmax, zbins)
                new_ticks = matplotlib.ticker.MaxNLocator(nbins=7) #Set up to 12 ticks  
                def custom_format(x, pos):
                    return '{:2.2e}'.format(x) if x != 0 else '0'

            #Create a colormap and colorbar normalization for each subplot   
            cs_fig = ax.contourf(xx, yy, z, levels = cbar_ticks, cmap = plt.cm.get_cmap(cmap), norm = norm)

            #Create a line contour for each colormap
            if levels is not None:  
                cs2_fig = ax.contour(cs_fig,levels=cs_fig.levels[::tot_lev[i]],colors='k',alpha=0.7,linestyles='dashed',linewidths=3,norm=norm)
                # ax[i].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1) #Uncomment for numbers

            #plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax.scatter(theta_true[idcs_to_plot[0]],theta_true[idcs_to_plot[1]], color="blue", label = "True",s=200,marker=(5,1),zorder = 2)
            if train_theta is not None:
                ax.scatter(train_theta[:,idcs_to_plot[0]],train_theta[:,idcs_to_plot[1]],color="green",s=100,label="Train",marker="x",zorder=1)
            if theta_next is not None:
                ax.scatter(theta_next[idcs_to_plot[0]],theta_next[idcs_to_plot[1]],color="black",s=175,label ="Max EI",marker= "^", zorder = 3)
            if theta_opt is not None:
                ax.scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], color="white", s=150, label = "Min Obj", marker = ".", edgecolor= "k", linewidth=0.3, zorder = 4)

            #Set plot details
            subplot_details(ax, xx, yy, None, None, all_z_titles[i], xbins, ybins, other_fontsize)

            # Use a custom formatter for the colorbar
            if not log_data and vmin >0:
                fmt = matplotlib.ticker.FuncFormatter(custom_format)
            else:
                fmt = matplotlib.ticker.FuncFormatter(custom_format) 

            divider1 = make_axes_locatable(ax)
            cax1 = divider1.append_axes("right", size="5%", pad="6%")
            cbar = fig.colorbar(cs_fig, ax = ax, cax = cax1, ticks = new_ticks, use_gridspec=True) #format = fmt
            cbar.ax.yaxis.set_major_formatter(fmt)
            cbar.ax.tick_params(labelsize=int(other_fontsize/2))
#             cbar.ax.set_ylabel(title2, fontsize=int(other_fontsize/2), fontweight='bold')

    #Get legend information and make colorbar on last plot
    handles, labels = axes[0,0].get_legend_handles_labels() 
                      
    #Print the title
    if title is not None:
        title = title + " " + str(plot_axis_names)
        
    #Print the title and labels as appropriate
    #Define x and y labels
    if "theta" in plot_axis_names[0]:
        xlabel = r'$\mathbf{'+ "\\" + plot_axis_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ "\\" + plot_axis_names[1]+ '}$'
    else:
        xlabel = r'$\mathbf{'+ plot_axis_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ plot_axis_names[1]+ '}$'

    for axs in axes[-1]:
        axs.set_xlabel(xlabel, fontsize = other_fontsize)

    for axs in axes[:, 0]:
        axs.set_ylabel(ylabel, fontsize = other_fontsize)

    set_plot_titles(fig, title, None, None, title_fontsize, other_fontsize)
    
    #Plots legend and title
    fig.legend(handles, labels, loc= "upper right", fontsize = other_fontsize, bbox_to_anchor=(-0.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    
    #Save or show figure
    if save_path is not None:
        for save_path_dir in save_path:
            save_path_to = save_path_dir + "Heat_Maps/" + z_choice + "_all_methods/" + plot_axis_names[0] + "-" + plot_axis_names[1]
#             print(save_path_to)
            save_fig(save_path_to, ext='png', close=False, verbose=False)  
        plt.close() #Only close figure after for loop
    else:
        plt.show()
        plt.close()
    
    return plt.show()

def plot_hms_all_methods(file_path_list, run_num_list, bo_iter_list, pair, z_choice, plot_dict):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        file_path_list: list of str, The file paths of data we want to make plots for
        run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
        bo_iter_list: list of int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
        pair: int, The pair of data parameters. pair 0 is the 1st pair
        z_choice: str, "sse_sim", "sse_mean", "sse_var", or "ei". The values that will be plotted
        plot_dict: dict, dictionary of plotting options. Generate with make_plot_dict()
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    keys_to_check = ["title","log_data","title_size","other_size","xbins","ybins","zbins","save_path","cmap","line_levels"]
    assert all(key in plot_dict for key in keys_to_check), "plot_dict must have all keys. Generate plot_dict with make_plot_dict()"
    keys_not_none = ["line_levels", "zbins", "cmap"]
    assert all(plot_dict[key] is not None for key in keys_not_none), "line_levels, zbins, and cmap must not be None!"
    
    #Break down plot dict and check for correct things
    title = plot_dict["title"]
    log_data = plot_dict["log_data"]
    title_fontsize = plot_dict["title_size"]
    other_fontsize = plot_dict["other_size"]
    xbins = plot_dict["xbins"]
    ybins = plot_dict["ybins"]
    zbins = plot_dict["zbins"]
    save_path = plot_dict["save_path"]
    cmap = plot_dict["cmap"]
    levels = plot_dict["line_levels"]
    
    #Assert Statements
    list_vars = [file_path_list, run_num_list, bo_iter_list]
    assert all(isinstance(var, Iterable) for var in list_vars), "file_path_list, run_num_list, and bo_iter_list must be iterable"
    assert all(isinstance(item, int) for item in run_num_list), "run_num_list elements must be int"
    assert all(isinstance(item, int) for item in bo_iter_list), "bo_iter_list elements must be int"
    assert all(isinstance(item, str) for item in file_path_list), "file_path_list elements must be str"
    assert isinstance(z_choice, str), "z_choice must be string"
    assert z_choice in ['sse_sim', 'sse_mean', 'sse_var','ei'], "z_choice must be 'sse_sim', 'sse_mean', 'sse_var', or 'ei'"

    method_names = ["Conventional", "Log Conventional", "Independence", "Log Independence", "Sparse Grid", "Monte Carlo"]
    
    #Define plot levels
    if levels is None:
        tot_lev = None
    elif len(levels) == 1:
        tot_lev = levels*len(z) 
    else:
        tot_lev = levels
    
    #Make figures and define number of subplots based on number of files (different methods)  
    subplots_needed = len(file_path_list)
    fig, ax, num_subplots, plot_mapping = create_subplots(subplots_needed, sharex = True, sharey = True)
    
    all_z_data = []
    all_theta_opt = []
    all_theta_next = []
    all_train_theta = []
    
    #Get all data for subplots needed
    #Loop over number of subplots needed
    for i in range(subplots_needed):
        if "ei" in z_choice:
            get_ei = True
        else:
            get_ei = False
        #Get data
        analysis_list = analyze_heat_maps(file_path_list[i], run_num_list[i], bo_iter_list[i], pair, log_data, get_ei)
        sim_sse_var_ei, test_mesh, param_info_dict = analysis_list
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        
        theta_true = param_info_dict["true"]
        theta_opt = param_info_dict["min_sse"]
        theta_next = param_info_dict["max_ei"]
        train_theta = param_info_dict["train"]
        plot_axis_names = param_info_dict["names"]
        idcs_to_plot = param_info_dict["idcs"]

        #Assert sattements
        #Get x and y data from test_mesh
        xx , yy = test_mesh #NxN, NxN
        assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
        
        #Find z based on z_choice
        if "sim" in z_choice:
            z = sse_sim
            title2 = r"$\mathbf{e(\theta)_{sim}}$"
        elif "mean" in z_choice:
            z = sse_mean
            title2 = r"$\mathbf{e(\theta)_{gp}}$"
        elif "var" in z_choice:
            z = sse_var
            title2 = r"$\mathbf{e(\theta)_{gp_{var}}}$"
        elif "ei" in z_choice:
            z = ei
            title2 = r"$\mathbf{EI(\theta)}$"
        else:
            raise Warning("choice must contain 'sim', 'mean', 'var', or 'ei'")
            
        #Assert statements
        assert z.shape==xx.shape, "Array z must be NxN"
        all_z_data.append(z)
        all_theta_opt.append(theta_opt)
        all_theta_next.append(theta_next)
        all_train_theta.append(train_theta)
                 
    #Initialize need_unscale to False
    need_unscale = False
    
    #Unlog scale the data if vmin is 0 and log_data = True
    if np.amin(all_z_data) == -np.inf or np.isnan(np.amin(all_z_data)):
        need_unscale = True 
        if log_data:
            warnings.warn("Cannot plot log scaled data! Reverting to original")
            z = np.exp(all_z_data[i])

    # Find the maximum and minimum values in your data to normalize the color scale
    vmin = min(np.min(arr) for arr in all_z_data)
    vmax = max(np.max(arr) for arr in all_z_data)
    #Check if data scales 3 orders of magnitude
    mag_diff = math.log10(abs(vmax)) - math.log10(abs(vmin)) > 3.0

    # Create a common color normalization for all subplots
    #Do not use log10 scale if natural log scaling data or the difference in min and max values < 1e-3 
    if log_data == True or need_unscale or mag_diff:
        norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False) 
        cbar_ticks = np.linspace(vmin, vmax, zbins)
        new_ticks = matplotlib.ticker.MaxNLocator(nbins=12) #Set up to 12 ticks
    else:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip = False)
        cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), zbins)
        new_ticks= matplotlib.ticker.LogLocator(numticks=12)

    #Set plot details
    #Loop over number of subplots
    for i in range(subplots_needed):
        #Get method value from json file
        with open(os.path.dirname(file_path_list[i]) + "/signac_statepoint.json", 'r') as json_file:
            # Load the JSON data
            GPBO_method_val = json.load(json_file)["meth_name_val"]
        ax_idx = int(GPBO_method_val - 1)  
        ax_row, ax_col = plot_mapping[ax_idx]
        
        z = all_z_data[i]
        theta_opt = all_theta_opt[i]
        theta_next = all_theta_next[i]
        train_theta = all_train_theta[i]


        #Set number format based on magnitude
        if np.amax(abs(z)) < 1e-1 or np.amax(abs(z)) > 1000:
            fmt = '%.2e'           
        else:
            fmt = '%2.2f'

        #Create a colormap and colorbar for each subplot
        if log_data == True:
            cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, cmap = plt.cm.get_cmap(cmap), norm = norm)
        else:
            cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, cmap = plt.cm.get_cmap(cmap), norm = norm)
#             cs_fig = ax[i].contourf(xx, yy, z, levels = zbins, cmap = plt.cm.get_cmap(cmap), norm = norm)

        #Create a line contour for each colormap
        if levels is not None:  
            cs2_fig = ax[ax_row, ax_col].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', alpha=0.7, linestyles='dashed', linewidths=3, norm = norm)
            # ax[ax_idx].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1, fmt = fmt)

        #plot min obj, max ei, true and training param values as appropriate
        if theta_true is not None:
            ax[ax_row, ax_col].scatter(theta_true[idcs_to_plot[0]], theta_true[idcs_to_plot[1]], color="blue", label = "True", 
                                       s=200, marker = (5,1), zorder = 2)
        if train_theta is not None:
            ax[ax_row, ax_col].scatter(train_theta[:,idcs_to_plot[0]], train_theta[:,idcs_to_plot[1]], color="green", s=100, 
                                       label="Train", marker= "x", zorder = 1)
        if theta_next is not None:
            ax[ax_row, ax_col].scatter(theta_next[idcs_to_plot[0]], theta_next[idcs_to_plot[1]], color="black", s=175,
                                       label ="Max EI",marker = "^", zorder = 3)
        if theta_opt is not None:
            ax[ax_row, ax_col].scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], color="white", s=150, label = "Min Obj", 
                                       marker = ".", edgecolor= "k", linewidth=0.3, zorder = 4)

        #Set plot details
        subplot_details(ax[ax_row, ax_col], xx, yy, None, None, method_names[ax_idx], xbins, ybins, other_fontsize)

    #Get legend information and make colorbar on last plot
    handles, labels = ax[-1, -1].get_legend_handles_labels() 

    cb_ax = fig.add_axes([1.03,0,0.04,1])
    if log_data is True and not need_unscale:
        title2 = "log(" + title2 + ")"
        
    cbar = fig.colorbar(cs_fig, orientation='vertical', ax=ax, cax=cb_ax, ticks = new_ticks)
    cbar.ax.tick_params(labelsize=other_fontsize)
    cbar.ax.set_ylabel(title2, fontsize=other_fontsize, fontweight='bold')
                      
    #Print the title
    if title is not None:
        title = title + " " + str(plot_axis_names)
        
    #Print the title and labels as appropriate
    #Define x and y labels
    if "theta" in plot_axis_names[0]:
        xlabel = r'$\mathbf{'+ "\\" + plot_axis_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ "\\" + plot_axis_names[1]+ '}$'
    else:
        xlabel = r'$\mathbf{'+ plot_axis_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ plot_axis_names[1]+ '}$'
        
    for axs in ax[-1]:
        axs.set_xlabel(xlabel, fontsize = other_fontsize)

    for axs in ax[:, 0]:
        axs.set_ylabel(ylabel, fontsize = other_fontsize)

    set_plot_titles(fig, title, None, None, title_fontsize, other_fontsize)
    
    #Plots legend and title
    fig.legend(handles, labels, loc= "upper right", fontsize = other_fontsize, bbox_to_anchor=(-0.02, 1), borderaxespad=0)

    plt.tight_layout()

    #Save or show figure
    if save_path is not None:
        for save_path_dir in save_path:
            save_path_to = save_path_dir + "Heat_Maps/" + z_choice + "_all_methods/" + plot_axis_names[0] + "-" + plot_axis_names[1]
#             print(save_path_to)
            save_fig(save_path_to, ext='png', close=False, verbose=False)  
        plt.close() #Only close figure after for loop
    else:
        plt.show()
        plt.close()
    
    return plt.show()

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
    val_label = "EI Max"
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
            plt.scatter(val_data[:,i1],val_data[:,i2], color="blue",s=20, label = val_label, marker = "D", zorder = 3)
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
            ax.scatter(val_data[:,i1],val_data[:,i2], val_data[:,i3], color="blue", s=40, label = val_label, marker = "D", zorder = 3)
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
        
    fig, axes, ax, num_subplots = create_subplots(subplots_needed, sharex = False, sharey = False)
    
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

#Fix me
def plot_nlr_heat_maps(test_mesh, all_z_data, theta_true, theta_opt, z_titles, param_names, idcs_to_plot, plot_dict):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        test_mesh: list of ndarray of length 2, Containing all values of the parameters for the heat map x and y. Gen with np.meshgrid()
        theta_true: ndarray or None, Containing the true input parameters in all dimensions
        theta_obj_min: ndarray or None, Containing the optimal input parameters predicted by the GP
        param_names: list of str, Parameter names. Length of 2
        levels: int, list of int or None, Number of levels to skip when drawing contour lines
        idcs_to_plot: list of int, Indecies of parameters to plot
        all_z_data: list of np.ndarrays, The list of values that will be plotted. Ex. SSE, SSE_Var, EI
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
    keys_to_check = ["title","log_data","title_size","other_size","xbins","ybins","zbins","save_path","cmap","line_levels"]
    assert all(key in plot_dict for key in keys_to_check), "plot_dict must have all keys. Generate plot_dict with make_plot_dict()"
    keys_not_none = ["zbins", "cmap"]
    assert all(plot_dict[key] is not None for key in keys_not_none), "zbins, and cmap must not be None!"
    
    #Break down plot dict and check for correct things
    title = plot_dict["title"]
    log_data = plot_dict["log_data"]
    title_fontsize = plot_dict["title_size"]
    other_fontsize = plot_dict["other_size"]
    xbins = plot_dict["xbins"]
    ybins = plot_dict["ybins"]
    zbins = plot_dict["zbins"]
    save_path = plot_dict["save_path"]
    cmap = plot_dict["cmap"]
    levels = plot_dict["line_levels"]
    
    #Assert Statements
    none_str_vars = [save_path]
    none_ndarray_list = [theta_true, theta_opt]
    int_vars = [xbins, ybins, zbins, title_fontsize, other_fontsize]
    list_vars = [test_mesh, all_z_data, z_titles, param_names, idcs_to_plot]
    assert all(isinstance(var, str) or var is None for var in none_str_vars), "title and save_path  must be string or None"
    assert all(isinstance(var, np.ndarray) or var is None for var in none_ndarray_list), "theta_true and theta_obj_min must be array or None"
    assert all(isinstance(item, int) for item in idcs_to_plot), "idcs_to_plot elements must be int"
    assert all(isinstance(item, np.ndarray) for item in all_z_data), "all_z_data elements must be np.ndarray"
    assert all(isinstance(item, np.ndarray) for item in test_mesh), "test_mesh elements must be np.ndarray"
    assert all(isinstance(item, str) for item in z_titles), "z_titles elements must be str"
    assert all(isinstance(item, str) for item in param_names), "param_names elements must be str"
    
    #Define plot levels
    if levels is None:
        tot_lev = None
    elif len(levels) == 1:
        tot_lev = levels*len(all_z_data) 
    else:
        tot_lev = levels

    assert tot_lev is None or len(tot_lev) == len(all_z_data), "levels must be length 1, None, or len(all_z_data)"
        
    #Assert sattements
    #Get x and y data from test_mesh
    xx , yy = test_mesh #NxN, NxN
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    
    #Make figures and define number of subplots  
    subplots_needed = len(all_z_data)
    fig, ax, num_subplots, plot_mapping = create_subplots(subplots_needed, sharex = True, sharey = True)
    
    # Find the maximum and minimum values in your data to normalize the color scale
    vmin = min(np.min(arr) for arr in all_z_data)
    vmax = max(np.max(arr) for arr in all_z_data)
    mag_diff = int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0 if vmin > 0 else False

    # Create a common color normalization for all subplots
    if log_data == True or not mag_diff or vmin <0:
        print(vmin, vmax)
        norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False) 
        cbar_ticks = np.linspace(vmin, vmax, zbins)
    else:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip = False)
        cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), zbins)

    #Set plot details
    #Loop over number of subplots
    for i in range(subplots_needed):
        #Get method value from json file 
        ax_row, ax_col = plot_mapping[i]
        
        z = all_z_data[i]

        #Create a colormap and colorbar for each subplot
        if log_data == True:
            cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, cmap = plt.cm.get_cmap(cmap), norm = norm)
        else:
            cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, cmap = plt.cm.get_cmap(cmap), norm = norm)

        #Create a line contour for each colormap
        if levels is not None:  
            cs2_fig = ax[ax_row, ax_col].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', alpha=0.7, linestyles='dashed', linewidths=3, norm = norm)
            # ax[ax_row, ax_col].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1)

        #plot min obj, max ei, true and training param values as appropriate
        if theta_true is not None:
            ax[ax_row, ax_col].scatter(theta_true[idcs_to_plot[0]], theta_true[idcs_to_plot[1]], color="blue", label = "True", 
                                       s=200, marker = (5,1), zorder = 2)
        if theta_opt is not None:
            ax[ax_row, ax_col].scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], color="white", s=150, label = "Min Obj", 
                                       marker = ".", edgecolor= "k", linewidth=0.3, zorder = 4)

        #Set plot details
        subplot_details(ax[ax_row, ax_col], xx, yy, None, None, z_titles[i], xbins, ybins, other_fontsize)

    #Get legend information and make colorbar on last plot
    handles, labels = ax[-1, -1].get_legend_handles_labels() 

    cb_ax = fig.add_axes([1.03,0,0.04,1])
    if log_data is True or not mag_diff or vmin < 0:
        new_ticks = matplotlib.ticker.MaxNLocator(nbins=7) #Set up to 7 ticks
    else:
        new_ticks = matplotlib.ticker.LogLocator(numticks=7)

    title2 = z_titles[i] 
        
    if "theta" in param_names[0]:
        xlabel = r'$\mathbf{'+ "\\" + param_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ "\\" + param_names[1]+ '}$'
    else:
        xlabel = r'$\mathbf{'+ param_names[0]+ '}$'
        ylabel = r'$\mathbf{'+ param_names[1]+ '}$'
        
    for axs in ax[-1]:
        axs.set_xlabel(xlabel, fontsize = other_fontsize)

    for axs in ax[:, 0]:
        axs.set_ylabel(ylabel, fontsize = other_fontsize)
        
    cbar = fig.colorbar(cs_fig, orientation='vertical', ax=ax, cax=cb_ax, ticks = new_ticks)
    cbar.ax.tick_params(labelsize=other_fontsize)
    cbar.ax.set_ylabel("Function Value", fontsize=other_fontsize, fontweight='bold')
                      
    #Print the title
    if title is not None:
        title = title + " " + str(param_names)
        
    #Print the title and labels as appropriate
    #Define x and y labels
    set_plot_titles(fig, title, None, None, title_fontsize, other_fontsize)
    
    #Plots legend
    if labels:
        fig.legend(handles, labels, loc= "upper right", fontsize = other_fontsize, bbox_to_anchor=(-0.02, 1), borderaxespad=0)

    plt.tight_layout()

    #Save or show figure
    if save_path is not None:   
        if z_save_names:
            path_end =  '-'.join(z_save_names)  
        else:
            path_end = '-'.join(z_titles)
        save_path = save_path + "Heat_Maps/" + path_end + "/" + param_names[0] + "-" + param_names[1]
        save_fig(save_path, ext='png', close=True, verbose=False)  
    else:
        plt.show()
        plt.close()
    
    return plt.show()