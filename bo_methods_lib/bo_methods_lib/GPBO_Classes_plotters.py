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
from.analyze_data import *

import warnings
np.warnings = warnings

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
    assert isinstance(save_path, str) or save_path is None, "save_path must be str or None"
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

class Plotters:
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------

    """
    # Class variables and attributes
    
    def __init__(self, analyzer, save_figs = False):
        """
        Parameters
        ----------
        plt_options: dict, Generate with make_plot_dict()"
        criteria_dict: dict, Signac statepoints to consider for the job. Should include minimum of cs_name_val and param_name_str
        """
        #Asserts
        assert isinstance(save_figs, bool), "save_figs must be boolean"

        # Constructor method
        self.analyzer = analyzer
        self.save_figs = save_figs
        self.cmap = "autumn"
        self.xbins = 5
        self.ybins = 5
        self.zbins = 900
        self.title_fntsz = 24
        self.other_fntsz = 24
        self.colors = ["red", "blue", "green", "purple", "darkorange", "deeppink"]
        self.method_names = ["Conventional", "Log Conventional", "Independence", "Log Independence", 
                             "Sparse Grid", "Monte Carlo"]

    def plot_one_obj_all_methods(self, z_choice, log_data = False, title = None):
        """
        Plots SSE, Min SSE, or EI values vs BO iter for all BO Methods at the best runs
        
        Parameters
        -----------
            file_path_list: list of str, The file paths of data we want to make plots for
            run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
            z_choices:  str, one of "sse_sim", "sse_mean", "sse_var", or "ei". The value that will be plotted
            plot_dict: dict, a dictionary of the plot details 
            
        """
        #Assert Statements
        assert isinstance(z_choice, str), "z_choices must be str"
        assert z_choice in ['min_sse','sse','ei'], "z_choices must be one of 'min_sse', 'sse', or 'ei'"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        assert isinstance(log_data, bool), "log_data must be boolean:"
        
        #Set x and y labels and save path for figure
        x_label = "BO Iterations"
        y_label = self.__set_ylab_from_z(z_choice)
        save_path = self.analyzer.make_dir_name_from_criteria(self.analyzer.criteria_dict)
        
        #Get all jobs
        job_pointer = self.analyzer.get_jobs_from_criteria()
        #Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        #Back out best runs from job_list_best
        emph_runs = df_best["Run Number"].values
        #Initialize list of maximum bo iterations for each method
        meth_bo_max_evals = np.zeros(len(self.method_names))
        #Number of subplots is the length of the best jobs list
        subplots_needed = len(self.method_names)
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = False)
        
        #Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, x_label, y_label)

        #Loop over different jobs
        for i in range(len(job_pointer)):
            #Assert job exists, if it does, great,
            #Get data
            data, data_names, data_true, sp_data = self.analyzer.analyze_obj_vals(job_pointer[i], z_choice)
            GPBO_method_val = sp_data["meth_name_val"]
            shrt_name = Method_name_enum(GPBO_method_val).name

            #Get Number of runs in the job
            runs_in_job = sp_data["bo_runs_in_job"]

            #Set subplot index to the corresponding method value number
            ax_idx = int(GPBO_method_val - 1)
            #Find the run corresponding to the best row for that job
            emph_run = df_best.loc[df_best['BO Method'] == shrt_name, 'Run Number'].iloc[0]
            ax_row, ax_col = plot_mapping[ax_idx]

            #Loop over all runs
            for j in range(runs_in_job):
                #Find job Number
                run_number = sp_data["bo_run_num"] + j
                #Create label based on run #
                label = "Run: "+ str() 
                #Get data until termination
                data_df_j = self.__get_data_to_bo_iter_term(data[j])
                #Define x axis
                bo_len = len(data_df_j)
                bo_space = np.linspace(1,bo_len,bo_len)
                #Set appropriate notation
                if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                    ax[ax_row, ax_col].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
                
                #Plot data
                if log_data == True:
                    data_df_j = np.log(data_df_j)

                #For result where run num list is the number of runs, print a solid line 
                # print(ax_idx, emph_runs[ax_idx], run_number)
                if emph_run == run_number:
                    ax[ax_row, ax_col].plot(bo_space, data_df_j, alpha = 1, color = self.colors[ax_idx], 
                                            label = label, drawstyle='steps')
                else:
                    ax[ax_row, ax_col].plot(bo_space, data_df_j, alpha = 0.2, color = self.colors[ax_idx], 
                                            linestyle='--', drawstyle='steps')
                ls_xset  = False
                #Plot true value if applicable
                if data_true is not None and j == data.shape[0] - 1:
                    data_ls = data_true[z_choice]
                    if isinstance(data_ls, pd.DataFrame):
                        x = data_ls["Iter"].to_numpy()
                        ls_xset = True
                        if z_choice == "min_sse":
                            y = data_ls["Min Obj Cum."].to_numpy()
                        else:
                            y = data_ls["Min Obj Act"].to_numpy()
                    else:
                        x = [1, data.shape[1]]
                        y = [list(data_true.values())[0], list(data_true.values())[0]]
                    ax[ax_row, ax_col].plot(x, y, color = "darkslategrey", linestyle='dashdot', 
                                               label = "Least Squares")

                #Set plot details 
                title = self.method_names[ax_idx]
                if bo_len > meth_bo_max_evals[ax_idx]:
                    meth_bo_max_evals[ax_idx] = bo_len
                    x_space = bo_space
                elif ls_xset and max(x) > meth_bo_max_evals[ax_idx]:
                    meth_bo_max_evals[ax_idx] = max(x)
                    x_space = np.linspace(1, max(x), max(x))
                self.__set_subplot_details(ax[ax_row, ax_col], x_space, data_df_j, None, None, title)

        #Set handles and labels and scale axis if necessary
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
        
        #save or show figure
        if self.save_figs:
            save_path_dir = os.path.join(save_path, "line_plots", "all_meth_1_obj")
            save_path_to = os.path.join(save_path_dir, z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()
            
        return  
    
    def plot_hypers(self, job, title = None):
        data, data_names, data_true, sp_data = self.analyzer.analyze_hypers(job)
        y_label = "Value"
        title = "Hyperparameter Values"
        fig = self.__plot_2D_general(data, data_names, data_true, y_label, title, False)
        #save or show figure
        if self.save_figs:
            save_path_to = os.path.join(job.fn(""), "line_plots", "hyperparams")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

    def plot_thetas(self, job, z_choice, title = None):
        data, data_names, data_true, sp_data = self.analyzer.analyze_thetas(job, z_choice)
        y_label = None
        title = "Min Obj Parameter Values"
        fig = self.__plot_2D_general(data, data_names, data_true, y_label, title, False)
        #save or show figure
        if self.save_figs:
            save_path_to = os.path.join(job.fn(""), "line_plots", "params_" + z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

    def __plot_2D_general(self, data, data_names, data_true, y_label, title, log_data):
        """
        Plots 2D values of the same data type (ei, sse, min sse) on multiple subplots
        
        Parameters
        -----------
            data: ndarray (n_runs x n_iters x n_params), Array of data from bo workflow runs
            data_names: list of str, List of data names
            data_true: list/ndarray of float/int or None, The true values of each parameter
            plot_dict: dict, a dictionary of the plot details 
        """

        #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
        subplots_needed = data.shape[-1]
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = True)
        #Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, None, None)
        x_label = "BO Iterations"

        #Loop over different hyperparameters (number of subplots)
        for i, ax in enumerate(axes.flatten()):
            #Only plot data if axis is visible
            if i < subplots_needed:
                #The index of the data is i, and one data type is in the last row of the data
                one_data_type = data[:,:,i]

                #Loop over all runs
                for j in range(one_data_type.shape[0]):
                    #Create label based on run #
                    label = "Run: "+str(j+1) 
                    data_df_j = self.__get_data_to_bo_iter_term(one_data_type[j])

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
                        ax.axhline(y=list(data_true.values())[i], color = "red", linestyle='--', label = "True Value")
                    
                    #Set plot details 
                    title = r'$'+ data_names[i]+ '$'
                    self.__set_subplot_details(ax, bo_space, data_df_j, None, None, title)
                    
                if not log_data and data_true is None:
                    ax.set_yscale("log")

            #Add legends and handles from last subplot that is visible
            if i == subplots_needed -1:
                handles, labels = axes[0, -1].get_legend_handles_labels()  
        
        for axs in axes[-1]:
            axs.set_xlabel(x_label, fontsize = self.other_fntsz)

        for axs in axes[:, 0]:
            axs.set_ylabel(y_label, fontsize = self.other_fntsz)

        #Plots legend and title
        plt.tight_layout()
        fig.legend(handles, labels, loc= "center left", fontsize = self.other_fntsz, bbox_to_anchor=(1.0, 0.60), 
                   borderaxespad=0)
        
        return fig
    
    def plot_objs_all_methods(self, z_choices, log_data = False, title = None):
        """
        Plots EI, SSE, Min SSE, and EI values vs BO iter for all 6 methods
        
        Parameters
        -----------
            file_path_list: list of str, The file paths of data we want to make plots for
            run_num_list: list of int, The run you want to analyze. Note, run_num 1 corresponds to index 0
            z_choices:  list of str, list of strings "sse_sim", "sse_mean", "sse_var", and/or "ei". The values that will be plotted
            plot_dict: dict, a dictionary of the plot details 
            
        """
        
        #Break down plot dict and check for correct things
        save_path = self.analyzer.make_dir_name_from_criteria(self.analyzer.criteria_dict)
        x_label = "BO Iterations"
        
        #Assert Statements
        assert isinstance(z_choices, (list, str)), "z_choices must be list or string"
        if isinstance(z_choices, str):
            z_choices = list(z_choices)
        assert all(isinstance(item, str) for item in z_choices), "z_choices elements must be str"
        for i in range(len(z_choices)):
            assert z_choices[i] in ['min_sse','sse','ei'],"z_choices items must be 'min_sse', 'sse', or 'ei'"
        
        #Create figure and axes. Number of subplots is 1 for each ei, sse, sse_sim etc.
        subplots_needed = len(z_choices)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = True)
        
        #Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, None, None)
        bo_len_max = 1

        #Get all jobs
        job_pointer = self.analyzer.get_jobs_from_criteria()
        #Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        #Back out best runs from job_list_best
        emph_runs = df_best["Run Number"].values

        #Loop over different methdods (number of subplots)
        for i in range(len(job_pointer)):     
            #Get data
            data, data_names, data_true, sp_data = self.analyzer.analyze_obj_vals(job_pointer[i], z_choices)
            GPBO_method_val = sp_data["meth_name_val"]
            #Create label based on method #
            label = self.method_names[GPBO_method_val-1] 
            
            #Loop over number of data types
            for k, ax in enumerate(axes.flatten()):
                #Only plot data if axis is visible
                if k < subplots_needed:
    #             for k in range(data.shape[-1]):
                    #The index of the data type is k, and one data type is in the last row of the data
                    one_data_type = data[:,:,k]

                    #Get Number of runs in the job
                    runs_in_job = sp_data["bo_runs_in_job"]

                    #loop as long as there are runs in the file
                    for j in range(runs_in_job):
                        #Set run number of run in job
                        run_number = sp_data["bo_run_num"] + j
                        #Remove elements that are numerically 0            
                        data_df_j = self.__get_data_to_bo_iter_term(one_data_type[j])
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
                        if emph_runs[GPBO_method_val -1] == run_number:
                            ax.plot(bo_space, data_df_j, alpha = 1, color = self.colors[GPBO_method_val-1], 
                                    label = label, drawstyle='steps')
                        else:
                            ax.step(bo_space, data_df_j, alpha = 0.2, color = self.colors[GPBO_method_val-1], 
                                    linestyle='--', drawstyle='steps')

                    #Set plot details
                    bo_space_org = np.linspace(1,bo_len_max,100)
                    self.__set_subplot_details(ax, bo_space_org, None, x_label, rf"${data_names[k]}$", None)

        #Get legend and handles
        handles, labels = axes[0,0].get_legend_handles_labels()
        if log_data == False:
            for ax in axes.flatten():
                ax.set_yscale("log")
        
        #Plots legend and title
        plt.tight_layout()
        fig.legend(handles, labels, loc= "center left", fontsize = self.other_fntsz, bbox_to_anchor=(1.0, 0.60), 
                   borderaxespad=0)
        
        #save or show figure
        if self.save_figs:
            z_choices_sort = sorted(z_choices, key=lambda x: ("sse", "min_sse", "ei").index(x))
            save_path_dir = os.path.join(save_path, "line_plots")
            save_path_to = os.path.join(save_path_dir, "all_meth_" + '_'.join(map(str, z_choices_sort)))
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()
            
        return
    
    def __get_z_plot_names_hms(self, z_choices, sim_sse_var_ei):
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        if isinstance(z_choices, str):
            z_choices = [z_choices]
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
        if len(all_z_data) == 1:
            return all_z_data[0], all_z_titles[0]
        else:
            return all_z_data, all_z_titles
        
    def plot_hms_all_methods(self, pair, z_choice, levels, log_data = False, title = None):
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
        #Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        #Back out best runs from job_list_best
        emph_runs = df_best["Run Number"].values
        emph_iters = df_best["BO Iter"].values

        #Make figures and define number of subplots based on number of different methods
        subplots_needed = len(job_list_best)
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = True, sharey = True)
        
        #Define plot levels
        if levels is None:
            tot_lev = None
        elif len(levels) == 1:
            tot_lev = levels*len(z) 
        else:
            tot_lev = levels

        all_z_data = []
        all_sp_data = []
        all_theta_opt = []
        all_theta_next = []
        all_train_theta = []
        
        #Get all data for subplots needed
        #Loop over number of subplots needed
        for i in range(len(job_list_best)):
            if "ei" in z_choice:
                get_ei = True
            else:
                get_ei = False
            #Get data
            analysis_list = self.analyzer.analyze_heat_maps(job_list_best[i], emph_runs[i], emph_iters[i], 
                                                            pair, get_ei = get_ei)
            sim_sse_var_ei, test_mesh, param_info_dict, sp_data = analysis_list
            #Set correct values based on propagation of errors for gp
            sim_sse_var_ei = self.__scale_z_data(sim_sse_var_ei, sp_data, log_data)
            
            theta_true = param_info_dict["true"]
            theta_opt = param_info_dict["min_sse"]
            theta_next = param_info_dict["max_ei"]
            train_theta = param_info_dict["train"]
            plot_axis_names = param_info_dict["names"]
            idcs_to_plot = param_info_dict["idcs"]
            z, title2 = self.__get_z_plot_names_hms(z_choice, sim_sse_var_ei)
            
            #Get x and y data from test_mesh
            xx , yy = test_mesh #NxN, NxN
            #Assert sattements
            assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
            assert z.shape==xx.shape, "Array z must be NxN"

            all_z_data.append(z)
            all_sp_data.append(sp_data)
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
        #Check if data scales 2+ orders of magnitude
        mag_diff = int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0 if vmin > 0 else False

        # Create a common color normalization for all subplots
        #Do not use log10 scale if natural log scaling data or the difference in min and max values < 1e-3 
        if log_data == True or need_unscale or not mag_diff :
            norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False) 
            cbar_ticks = np.linspace(vmin, vmax, self.zbins)
            new_ticks = matplotlib.ticker.MaxNLocator(nbins=7) #Set up to 12 ticks
        else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip = False)
            cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), self.zbins)
            new_ticks= matplotlib.ticker.LogLocator(numticks=7)

        #Set plot details
        #Loop over number of subplots
        for i in range(len(job_list_best)):
            #Get method value from json file
            GPBO_method_val = all_sp_data[i]["meth_name_val"]
            ax_idx = int(GPBO_method_val - 1)  
            ax_row, ax_col = plot_mapping[ax_idx]
            
            z = all_z_data[i]
            theta_opt = all_theta_opt[i]
            theta_next = all_theta_next[i]
            train_theta = all_train_theta[i]

            #Set number format based on magnitude
            fmt = '%.2e' if np.amax(abs(z)) < 1e-1 or np.amax(abs(z)) > 1000 else '%2.2f'

            if np.all(z == z[0]):
                z =  np.random.normal(scale=1e-14, size=z.shape)

            #Create a colormap and colorbar for each subplot
            if log_data == True:
                cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, 
                                                     cmap = plt.cm.get_cmap(self.cmap), norm = norm)
            else:
                cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, 
                                                     cmap = plt.cm.get_cmap(self.cmap), norm = norm)
    #             cs_fig = ax[i].contourf(xx, yy, z, levels = zbins, cmap = plt.cm.get_cmap(cmap), norm = norm)

            #Create a line contour for each colormap
            if levels is not None:  
                cs2_fig = ax[ax_row, ax_col].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], 
                                                     colors='k', alpha=0.7, linestyles='dashed', linewidths=3, 
                                                     norm = norm)
                # ax[ax_idx].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1, fmt = fmt)

            #plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax[ax_row, ax_col].scatter(theta_true[idcs_to_plot[0]], theta_true[idcs_to_plot[1]], color="blue", 
                                           label = "True", s=200, marker = (5,1), zorder = 2)
            if train_theta is not None:
                ax[ax_row, ax_col].scatter(train_theta[:,idcs_to_plot[0]], train_theta[:,idcs_to_plot[1]], 
                                           color="green", s=100, label="Train", marker= "x", zorder = 1)
            if theta_next is not None:
                ax[ax_row, ax_col].scatter(theta_next[idcs_to_plot[0]], theta_next[idcs_to_plot[1]], color="black", 
                                           s=175, label ="Max EI",marker = "^", zorder = 3)
            if theta_opt is not None:
                ax[ax_row, ax_col].scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], color="white", 
                                           s=150, label = "Min Obj", marker = ".", edgecolor= "k", linewidth=0.3, 
                                           zorder = 4)

            #Set plot details
            self.__set_subplot_details(ax[ax_row, ax_col], xx, yy, None, None, self.method_names[ax_idx])

        #Get legend information and make colorbar on last plot
        handles, labels = ax[-1, -1].get_legend_handles_labels() 

        cb_ax = fig.add_axes([1.03,0,0.04,1])
        if log_data is True and not need_unscale:
            title2 = "log(" + title2 + ")"
            
        cbar = fig.colorbar(cs_fig, orientation='vertical', ax=ax, cax=cb_ax, ticks = new_ticks)
        cbar.ax.tick_params(labelsize=self.other_fntsz)
        cbar.ax.set_ylabel(title2, fontsize=self.other_fntsz, fontweight='bold')
                        
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
            axs.set_xlabel(xlabel, fontsize = self.other_fntsz)

        for axs in ax[:, 0]:
            axs.set_ylabel(ylabel, fontsize = self.other_fntsz)

        self.__set_plot_titles(fig, title, None, None)
        
        #Plots legend and title
        fig.legend(handles, labels, loc= "upper right", fontsize = self.other_fntsz, 
                   bbox_to_anchor=(-0.02, 1), borderaxespad=0)

        plt.tight_layout()  

        #save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(self.analyzer.criteria_dict)
            save_path_dir = os.path.join(save_path, "heat_maps", plot_axis_names[0] + "-" + plot_axis_names[1])
            save_path_to = os.path.join(save_path_dir, z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()
        
        return 
    
    def plot_hms_gp_compare(self, job, run_num, bo_iter, pair, z_choices, levels, log_data = False, title = None):
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
        #Assert Statements
        assert isinstance(z_choices, (Iterable, str)), "z_choices must be Iterable or str"
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        for z_choice in z_choices:
            assert z_choice in ['sse_sim', 'sse_mean', 'sse_var','ei'], "z_choices elements must be 'sse_sim', 'sse_mean', 'sse_var', or 'ei'"

        #Define plot levels
        if levels is None:
            tot_lev = None
        elif len(levels) == 1:
            tot_lev = levels*len(z_choices) 
        else:
            tot_lev = levels
        
        #Get all data for subplots needed
        get_ei = True if "ei" in z_choices else False
        analysis_list = self.analyzer.analyze_heat_maps(job, run_num, bo_iter, pair, get_ei = get_ei)
        sim_sse_var_ei, test_mesh, param_info_dict, sp_data = analysis_list
        #Get method value from json file
        GPBO_method_val = sp_data["meth_name_val"]
        #Set correct values based on propagation of errors for gp
        sim_sse_var_ei = self.__scale_z_data(sim_sse_var_ei, sp_data, log_data)
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

        #Make figures and define number of subplots based on number of files (different methods)  
        subplots_needed = len(z_choices)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = True, sharey = True)

        #Find z based on z_choice
        all_z_data, all_z_titles = self.__get_z_plot_names_hms(z_choices, sim_sse_var_ei)
            
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
                #If all z data are the same, add a small amount of noise to each to allow for plotting
                if vmin == vmax:
                    vmin -= 1e-14
                    vmax += 1e-14

                #Check if data scales 3 orders of magnitude
                mag_diff = int(math.log10(abs(vmax)) - math.log10(abs(vmin))) > 2.0 if vmin > 0 else False
                
                if need_unscale == False and log_data:
                    title2 = "log(" + all_z_titles[i] + ")"
                else:
                    title2 = all_z_titles[i]

                #Choose an appropriate colormap and scaling based on vmin, vmax, and log_data
                #If not using log data, vmin > 0, and the data scales 3 orders+ of magnitude use log10 to view plots
                if log_data or vmin < 0 or not mag_diff:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False) 
                    cbar_ticks = np.linspace(vmin, vmax, self.zbins)
                    new_ticks = matplotlib.ticker.MaxNLocator(nbins=7) #Set up to 12 ticks  
                    def custom_format(x, pos):
                        return '{:2.2e}'.format(x) if x != 0 else '0'
                else:
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), self.zbins)
        #             new_ticks = np.logspace(np.log10(vmin), np.log10(vmax), 7)
                    new_ticks = matplotlib.ticker.LogLocator(numticks=7) #Set up to 12 ticks
                    def custom_format(x, pos):
                        return f'{eval("10**" + str(int(np.log10(x))))}' if x != 0 else '0'

                #Create a colormap and colorbar normalization for each subplot
                cs_fig = ax.contourf(xx, yy, z, levels = cbar_ticks, cmap = plt.cm.get_cmap(self.cmap), norm = norm)

                #Create a line contour for each colormap
                if levels is not None:  
                    cs2_fig = ax.contour(cs_fig,levels=cs_fig.levels[::tot_lev[i]],colors='k',alpha=0.7,
                                         linestyles='dashed',linewidths=3,norm=norm)
                    # ax[i].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1) #Uncomment for numbers

                #plot min obj, max ei, true and training param values as appropriate
                if theta_true is not None:
                    ax.scatter(theta_true[idcs_to_plot[0]],theta_true[idcs_to_plot[1]], color="blue", 
                               label = "True",s=200,marker=(5,1),zorder = 2)
                if train_theta is not None:
                    ax.scatter(train_theta[:,idcs_to_plot[0]],train_theta[:,idcs_to_plot[1]],color="green",
                               s=100,label="Train",marker="x",zorder=1)
                if theta_next is not None:
                    ax.scatter(theta_next[idcs_to_plot[0]],theta_next[idcs_to_plot[1]],color="black",s=175,
                               label ="Max EI",marker= "^", zorder = 3)
                if theta_opt is not None:
                    ax.scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], color="white", s=150, 
                               label = "Min Obj", marker = ".", edgecolor= "k", linewidth=0.3, zorder = 4)

                #Set plot details
                self.__set_subplot_details(ax, xx, yy, None, None, all_z_titles[i])

                # Use a custom formatter for the colorbar
                if not log_data and vmin >0:
                    fmt = matplotlib.ticker.FuncFormatter(custom_format)
                else:
                    fmt = matplotlib.ticker.FuncFormatter(custom_format) 

                divider1 = make_axes_locatable(ax)
                cax1 = divider1.append_axes("right", size="5%", pad="6%")
                cbar = fig.colorbar(cs_fig, ax = ax, cax = cax1, ticks = new_ticks, use_gridspec=True) #format = fmt
                cbar.ax.yaxis.set_major_formatter(fmt)
                cbar.ax.tick_params(labelsize=int(self.other_fntsz/2))
    #             cbar.ax.set_ylabel(title2, fontsize=int(other_fontsize/2), fontweight='bold')

        #Get legend information and make colorbar on last plot
        handles, labels = axes[0,0].get_legend_handles_labels() 
                        
        #Print the title
        if title is None:
            title = self.method_names[GPBO_method_val-1]
            
        #Print the title and labels as appropriate
        #Define x and y labels
        if "theta" in plot_axis_names[0]:
            xlabel = r'$\mathbf{'+ "\\" + plot_axis_names[0]+ '}$'
            ylabel = r'$\mathbf{'+ "\\" + plot_axis_names[1]+ '}$'
        else:
            xlabel = r'$\mathbf{'+ plot_axis_names[0]+ '}$'
            ylabel = r'$\mathbf{'+ plot_axis_names[1]+ '}$'

        for axs in axes[-1]:
            axs.set_xlabel(xlabel, fontsize = self.other_fntsz)

        for axs in axes[:, 0]:
            axs.set_ylabel(ylabel, fontsize = self.other_fntsz)

        self.__set_plot_titles(fig, title, None, None)
        
        #Plots legend and title
        fig.legend(handles, labels, loc= "upper right", fontsize = self.other_fntsz, bbox_to_anchor=(-0.02, 1), 
                   borderaxespad=0)
        
        plt.tight_layout()
        
        #save or show figure
        if self.save_figs:
            z_choices_sort = sorted(z_choices, key=lambda x: ('sse_sim', 'sse_mean', 'sse_var','ei').index(x))
            z_choices_str = '_'.join(map(str, z_choices_sort))
            title_str = title.replace(" ", "_").lower()
            save_path = self.analyzer.make_dir_name_from_criteria(self.analyzer.criteria_dict)
            save_path_dir = os.path.join(save_path, "heat_maps", title_str, plot_axis_names[0] + "-" + 
                                        plot_axis_names[1], z_choices_str)
            save_path_to = os.path.join(save_path_dir, "run_"+ str(run_num) + "_" + "iter_" + str(bo_iter))
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()
        
        return plt.show()
    
    def __scale_z_data(self, sim_sse_var_ei, sp_data, log_data):
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        #Get log or unlogged data values        
        if log_data == False:
            #Change sse sim, mean, and stdev to not log for 1B and 2B
            if sp_data["meth_name_val"] in [2,4]:
                #SSE variance is var*(e^((log(sse)))^2
                sse_mean = np.exp(sse_mean)
                sse_var = (sse_var*sse_mean**2)      
                sse_sim = np.exp(sse_sim)

        #If getting log values
        else:
            #Get log data from 1A, 2A, 2C, and 2D
            if not sp_data["meth_name_val"] in [2,4]:            
                #SSE Variance is var/sse**2
                sse_var = sse_var/sse_mean**2
                sse_mean = np.log(sse_mean)
                sse_sim = np.log(sse_sim)

        sim_sse_var_ei = [sse_sim, sse_mean, sse_var, ei]
        return sim_sse_var_ei

    def __set_ylab_from_z(self, z_choice):
        if "sse" == z_choice:
            y_label = r"$\mathbf{e(\theta)}$"
        if "min_sse" == z_choice:
            y_label = r"$\mathbf{Min\,e(\theta)}$"   
        if "ei" == z_choice:
            y_label = r"$\mathbf{Max\,EI(\theta)}$"
        return y_label
    
    def __get_data_to_bo_iter_term(self, data_all_iters):
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

    def __save_fig(self, save_path, ext='png', close=True):                
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
        directory = os.path.split(save_path)[0]
        filename = "%s.%s" % (os.path.split(save_path)[1], ext)
        if directory == '':
            directory = '.'

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # The final path to save to
        savepath = os.path.join(directory, filename)

        # Actually save the figure
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        
        # Close it
        if close:
            plt.close()

    def __create_subplots(self, num_subplots, sharex = "row", sharey = 'none'):
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
    
    def __set_subplot_details(self, ax, plot_x, plot_y, xlabel, ylabel, title):
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
        int_vars = [self.xbins, self.ybins, self.other_fntsz]
        arr_vars = [plot_x, plot_y]
        
        #Assert Statements
        assert all(isinstance(var, str) or var is None for var in none_str_vars), "title, xlabel, and ylabel must be string or None"
        assert all(isinstance(var, int) for var in int_vars), "xbins, ybins, and fontsize must be int"
        assert all(var > 0 or var is None for var in int_vars), "integer variables must be positive"
        assert all(isinstance(var, (np.ndarray,pd.core.series.Series)) or var is None for var in arr_vars), "plot_x, plot_y must be np.ndarray or pd.core.series.Series or None"
        
        #Set title, label, and axes
        if title is not None:
            pad = 6 + 4*title.count("_")
            ax.set_title(title, fontsize=self.other_fntsz, fontweight='bold', pad = pad)   
        if xlabel is not None:
            pad = 6 + 4*xlabel.count("_")
            ax.set_xlabel(xlabel,fontsize=self.other_fntsz,fontweight='bold', labelpad = pad)
        if ylabel is not None:
            pad = 6 + 2*ylabel.count("_")
            ax.set_ylabel(ylabel,fontsize=self.other_fntsz,fontweight='bold', labelpad = pad)
            
        #Turn on tick parameters and bin number
        ax.xaxis.set_tick_params(labelsize=self.other_fntsz, direction = "in")
        ax.yaxis.set_tick_params(labelsize=self.other_fntsz, direction = "in")
        ax.locator_params(axis='y', nbins=self.ybins)
        ax.locator_params(axis='x', nbins=self.xbins)
        ax.minorticks_on() # turn on minor ticks
        ax.tick_params(which="minor",direction="in",top=True, right=True)
        
        #Set a and y bounds and aspect ratio
        if plot_x is not None and not np.isclose(np.min(plot_x), np.max(plot_x), rtol = 1e-6):        
            ax.set_xlim(left = np.min(plot_x), right = np.max(plot_x))
    
        if plot_y is not None and np.min(plot_y) == 0:
            ax.set_ylim(bottom = np.min(plot_y)-0.05, top = np.max(plot_y)+0.05) 

        ax.set_box_aspect(1)

        return ax

    def __set_plot_titles(self, fig, title, x_label, y_label):
        """
        Helper function to set plot titles and labels for figures with subplots
        """
        
        if self.title_fntsz is not None:
            fig.suptitle(title, weight='bold', fontsize=self.title_fntsz)
        if x_label is not None:
            fig.supxlabel(x_label, fontsize=self.other_fntsz,fontweight='bold')
        if y_label is not None:
            fig.supylabel(y_label, fontsize=self.other_fntsz,fontweight='bold')
            
        return 
     
    def plot_nlr_heat_maps(self, test_mesh, all_z_data, z_titles, levels, param_info_dict, log_data, title = None):
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
        
        #Assert Statements
        list_vars = [test_mesh, all_z_data, z_titles]
        assert all(isinstance(item, np.ndarray) for item in all_z_data), "all_z_data elements must be np.ndarray"
        assert all(isinstance(item, np.ndarray) for item in test_mesh), "test_mesh elements must be np.ndarray"
        
        #Define plot levels
        if levels is None:
            tot_lev = None
        elif len(levels) == 1:
            tot_lev = levels*len(all_z_data) 
        else:
            tot_lev = levels

        assert tot_lev is None or len(tot_lev) == len(all_z_data), "levels must be length 1, None, or len(all_z_data)"
            
        #Get info from param dict
        theta_true = param_info_dict["true"]
        theta_opt = param_info_dict["min_sse"]
        param_names = param_info_dict["names"]
        idcs_to_plot = param_info_dict["idcs"]

        #Assert sattements
        #Get x and y data from test_mesh
        xx , yy = test_mesh #NxN, NxN
        assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
        
        #Make figures and define number of subplots  
        subplots_needed = len(all_z_data)
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = True, sharey = True)
        
        # Find the maximum and minimum values in your data to normalize the color scale
        vmin = min(np.min(arr) for arr in all_z_data)
        vmax = max(np.max(arr) for arr in all_z_data)
        mag_diff = int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0 if vmin > 0 else False

        # Create a common color normalization for all subplots
        if log_data == True or not mag_diff or vmin <0:
            # print(vmin, vmax)
            norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False) 
            cbar_ticks = np.linspace(vmin, vmax, self.zbins)
        else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip = False)
            cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), self.zbins)

        #Set plot details
        #Loop over number of subplots
        for i in range(subplots_needed):
            #Get method value from json file 
            ax_row, ax_col = plot_mapping[i]
            
            z = all_z_data[i]

            #Create a colormap and colorbar for each subplot
            if log_data == True:
                cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, 
                                                     cmap = plt.cm.get_cmap(self.cmap), norm = norm)
            else:
                cs_fig = ax[ax_row, ax_col].contourf(xx, yy, z, levels = cbar_ticks, 
                                                     cmap = plt.cm.get_cmap(self.cmap), norm = norm)

            #Create a line contour for each colormap
            if levels is not None:  
                cs2_fig = ax[ax_row, ax_col].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', 
                                                     alpha=0.7, linestyles='dashed', linewidths=3, norm = norm)
                # ax[ax_row, ax_col].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1)

            #plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax[ax_row, ax_col].scatter(theta_true[idcs_to_plot[0]], theta_true[idcs_to_plot[1]], 
                                           color="blue", label = "True", s=200, marker = (5,1), zorder = 2)
            if theta_opt is not None:
                ax[ax_row, ax_col].scatter(theta_opt[idcs_to_plot[0]],theta_opt[idcs_to_plot[1]], 
                                           color="white", s=150, label = "Min Obj", marker = ".", 
                                           edgecolor= "k", linewidth=0.3, zorder = 4)

            #Set plot details
            self.__set_subplot_details(ax[ax_row, ax_col], xx, yy, None, None, z_titles[i])

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
            axs.set_xlabel(xlabel, fontsize = self.other_fntsz)

        for axs in ax[:, 0]:
            axs.set_ylabel(ylabel, fontsize = self.other_fntsz)
            
        cbar = fig.colorbar(cs_fig, orientation='vertical', ax=ax, cax=cb_ax, ticks = new_ticks)
        cbar.ax.tick_params(labelsize=self.other_fntsz)
        cbar.ax.set_ylabel("Function Value", fontsize=self.other_fntsz, fontweight='bold')
                        
        #Print the title
        if title is not None:
            title = title + " " + str(param_names)
            
        #Print the title and labels as appropriate
        #Define x and y labels
        self.__set_plot_titles(fig, title, None, None)
        
        #Plots legend
        if labels:
            fig.legend(handles, labels, loc= "upper right", fontsize = self.other_fntsz, bbox_to_anchor=(-0.02, 1), 
                       borderaxespad=0)

        plt.tight_layout()

        nlr_plot = "func_ls_compare" if theta_true is None else "sse_contour"

        #Save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(self.analyzer.criteria_dict)
            save_path_dir = os.path.join(save_path, "heat_maps", param_names[0] + "-" + param_names[1])
            save_path_to = os.path.join(save_path_dir, "least_squares")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()
        
        return plt.show()
    
    def make_parity_plots(self):
        """
        Makes Parity plots of validation and true data for selected methods in best
        """
        #Get Best Data Runs and iters
        df_best, job_list_best = self.analyzer.get_best_data()
        runs = df_best["Run Number"].to_list()
        iters = df_best["BO Iter"].to_list()
        GPBO_methods = df_best["BO Method"].to_list()
        ax_idxs = [int(Method_name_enum[str].value-1) for str in GPBO_methods]
        #Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
        subplots_needed = len(runs)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(subplots_needed, sharex = False, sharey=False)
        #Print the title and labels as appropriate
        self.__set_plot_titles(fig, None, "True Values", "Predicted Values")

        #Loop over different hyperparameters (number of subplots)
        for i, ax in enumerate(axes.flatten()):
            #Only plot data if axis is visible
            if i < subplots_needed:
                #Get the test data associated with the best job
                test_data = self.analyzer.analyze_parity_plot_data(job_list_best[i], runs[i], iters[i])
                
                #Get data from test_data
                sim_data = test_data.y_vals 
                gp_mean = test_data.gp_mean
                gp_stdev = np.sqrt(abs(test_data.gp_var))

                #Plot x and y data
                ax.plot(sim_data, sim_data, color = "k")
                ax.scatter(sim_data, gp_mean, color = "blue", label = "GP Mean")
                ax.errorbar(sim_data, gp_mean, yerr = 1.96*gp_stdev, alpha=0.3, fmt = 'o', color = "blue")
                
                #Set plot details
                self.__set_subplot_details(ax, sim_data, gp_mean, None, None, self.method_names[ax_idxs[i]])


            #Add legends and handles from last subplot that is visible
            if i == subplots_needed -1:
                handles, labels = axes[0, -1].get_legend_handles_labels()  
                
        #Plots legend
        if labels:
            fig.legend(handles, labels, loc= "upper right", fontsize = self.other_fntsz, bbox_to_anchor=(-0.02, 1), 
                       borderaxespad=0)

        plt.tight_layout()
            
        #Save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(self.analyzer.criteria_dict)
            save_path_dir = os.path.join(save_path)
            save_path_to = os.path.join(save_path_dir, "parity_plots")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()
            
        return 

        

