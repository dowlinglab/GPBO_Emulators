##https://towardsdatascience.com/model-validation-in-python-95e2f041f78c
##Load modules
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as PathEffects
from pylab import *
import torch
import os
import gpytorch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import qmc
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from .bo_functions_generic import define_GP_model, round_time, train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs, train_GP_scikit
from .GP_Vs_True_Param_Sens import eval_GP, eval_GP_emulator, path_name_gp_val
from .CS2_bo_plotters import save_csv, save_fig, plot_xy
    
# from .CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from .CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def Param_Sens_Multi_Theta(all_data, x_space_points, eval_theta_idxs, Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, value_num, skip_param_types = 0, kernel_func = "RBF", set_lengthscale = None, outputscl = False, train_iter = 300, initialize = 1, noise_std = 0.1, verbose = False, DateTime = None, save_csvs = True, save_figure= False, eval_Train = False, CutBounds = False, package = "gpytorch"):  
    """
    Compare GP models to True/Simulation Values
  
    Parameters:
    -----------
        all_data: list, contains all data for GP for different amounts of training data
        x_space_points: list or ndarray, The experimental data point indecies for X over which to evaluate the GP
        Xexp: ndarray, The experimental data for y (the true value)
        Yexp: ndarray, The experimental data for y (the true value)
        true_model_coefficients: ndarray, The array containing the true values of the problem (may be the same as true_p)
        true_p: ndarray, The array containing the true values of theta parameters to regress- flattened array
        Case_Study: int, float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        bounds_p: ndarray, defines the bounds of the parameter space values (unused now, may use to define values later)
        value_num: int, defines the number of values of each parameter to test for the graph
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        kernel_func" str, defines which kernel function to use
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        train_iter: int, number of training iterations to run for GP. Default is 300
        initialize: int, number of times to restart GP model training
        noise_std: float, int: The standard deviation of the noise. Default 0.1
        verbose: bool, Determines whether EI component terms are saved also determines activeness of print statement, Default = False
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        save_csvs: bool, Determines whether csv data will be saved for plots. Default True
        save_figure: bool, Determines whether figures will be saved. Default False
        eval_Train: bool, Whether the GP model will be evaluated at the first training point (T) or the true values (F). Default False
        CutBounds: bool, Used for naming. Set True if bounds are cut from original values. Default False
        package: str, Determines whether gpytorch or scikit learn will be used to build the GP model. Default "gpytorch"
        
    
    Returns:
    --------
        None, prints/saves data and figues
        
    """ 
    assert isinstance(outputscl, bool)==True, "Outputscl must be a boolean!"
    assert isinstance(kernel_func, str) == True, "kernel_func must be a string!"
    assert isinstance(eval_Train, bool)==True, "eval_Train must be a boolean!"
    
    param_dict = {0 : 'a_1', 1 : 'a_2', 2 : 'a_3', 3 : 'a_4',
              4 : 'b_1', 5 : 'b_2', 6 : 'b_3', 7 : 'b_4'}
    n, m = Xexp.shape
    q = true_p.shape[0]
    total_dim = int(m + q + 1) #add 1 because of 1 dimensional y data 
    t = [len(all_data[i])/n for i in range(len(all_data))] #Number of thetas used for training
    print("Training Thetas Used: ", t)
    X_space= np.array([Xexp[m] for m in x_space_points])
    val_num_map = np.linspace(0,value_num-1, value_num)
    
    #Create lists to save GP data and evlauation theta data in
    eval_theta_data = np.zeros((len(eval_theta_idxs), len(all_data), len(x_space_points), total_dim))
    param_sens_data = np.zeros((len(eval_theta_idxs), len(all_data), 3, len(x_space_points), q, value_num)) 
    lenscl_per_idx = np.zeros((len(eval_theta_idxs), len(all_data), q+m))
    noise_per_idx = np.zeros((len(eval_theta_idxs), len(all_data)))
    ops_per_idx = np.zeros((len(eval_theta_idxs), len(all_data)))
    
    #Loop over all theta evaluation points
    for eval_theta_idx in range(len(eval_theta_idxs)):
        eval_p_base = all_data[0][int(eval_theta_idxs[eval_theta_idx]*n),1:q+1]
        print("Theta Train for Sensitivity:", np.round(eval_p_base,6) )
        #Loop over training data sets
        for train_idx in range(len(all_data)):
            data = all_data[train_idx]  
            #Perform Parameter sensitivity analysis
            param_sens_data_res = Param_Sens_Analysis(data, x_space_points, eval_theta_idxs[eval_theta_idx], Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, value_num, skip_param_types, kernel_func, set_lengthscale, outputscl, train_iter, initialize, noise_std, verbose, DateTime, save_csvs, save_figure, eval_Train, CutBounds, package)
            train_xspace_set, all_data_to_plot, lenscl_final, lenscl_noise_final, outputscale_final, values_list = param_sens_data_res
            eval_theta_data[eval_theta_idx, train_idx, :, :] = train_xspace_set
            param_sens_data[eval_theta_idx, train_idx, :, :, :, :] = all_data_to_plot
            lenscl_per_idx[eval_theta_idx, train_idx] = lenscl_final
            noise_per_idx[eval_theta_idx, train_idx] = lenscl_noise_final
            ops_per_idx[eval_theta_idx, train_idx] = outputscale_final
    
    mul_plot_param_many(param_sens_data, set_lengthscale, train_iter, t, Case_Study, CutBounds, X_space, x_space_points, param_dict, values_list, val_num_map, lenscl_per_idx, eval_theta_data, noise_per_idx, ops_per_idx, kernel_func, DateTime, None, save_csvs, save_figure, package)
    return

def Param_Sens_Analysis(data, x_space_points, eval_theta_num, Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, value_num, skip_param_types = 0, kernel_func = "RBF", set_lengthscale = None, outputscl = False, train_iter = 300, initialize = 1, noise_std = 0.1, verbose = False, DateTime = None, save_csvs = True, save_figure= False, eval_Train = False, CutBounds = False, package = "gpytorch"):  
    """
    Compare GP models to True/Simulation Values
  
    Parameters:
    -----------
        data: ndarray, contains all data for a single number of training points for the GP
        x_space_points: list or ndarray, The experimental data point indecies for X over which to evaluate the GP
        Xexp: ndarray, The experimental data for y (the true value)
        Yexp: ndarray, The experimental data for y (the true value)
        true_model_coefficients: ndarray, The array containing the true values of the problem (may be the same as true_p)
        true_p: ndarray, The array containing the true values of theta parameters to regress- flattened array
        Case_Study: int, float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        bounds_p: ndarray, defines the bounds of the parameter space values (unused now, may use to define values later)
        value_num: int, defines the number of values of each parameter to test for the graph
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        kernel_func" str, defines which kernel function to use
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        train_iter: int, number of training iterations to run for GP. Default is 300
        initialize: int, number of times to restart GP model training
        noise_std: float, int: The standard deviation of the noise. Default 0.1
        verbose: bool, Determines whether EI component terms are saved also determines activeness of print statement, Default = False
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        save_csvs: bool, Determines whether csv data will be saved for plots. Default True
        save_figure: bool, Determines whether figures will be saved. Default False
        eval_Train: bool, Whether the GP model will be evaluated at the first training point (T) or the true values (F). Default False
        CutBounds: bool, Used for naming. Set True if bounds are cut from original values. Default False
        package: str, Determines whether gpytorch or scikit learn will be used to build the GP model. Default "gpytorch"
        
    
    Returns:
    --------
        None, prints/saves data and figues
        
    """    
    #Define constants for dimensions of x (m), number of exp data points (n), number of parameters to be regressed (q), and data length (t)
    rand_seed = False
    param_dict = {0 : 'a_1', 1 : 'a_2', 2 : 'a_3', 3 : 'a_4',
              4 : 'b_1', 5 : 'b_2', 6 : 'b_3', 7 : 'b_4'}
    n, m = Xexp.shape
    q = true_p.shape[0]
    t = len(data)
    
    #Set training data
    data_train = data
    #separate into y data and training parameter data
    if m > 1:
        train_p = torch.tensor(data_train[:,1:-m+1]).float() #8 or 10 (emulator) parameters 
    else:
        train_p = torch.tensor(data_train[:,1:-m]).float() #8 or 10 (emulator) parameters 

    train_y = torch.tensor(data_train[:,-1]).float()
    
    #Define X training data
    X_train = train_p[:,-m:]
    
    #Define model and likelihood
    gp_model_params = define_GP_model(package, noise_std, train_p, train_y, kernel_func, set_lengthscale, outputscl, initialize, train_iter, True)
    model, likelihood, lenscl_final, lenscl_noise_final, outputscale_final = gp_model_params
        
    #Print noise and lengthscale hps
    lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
    lenscl_noise_print = '%.3e' % lenscl_noise_final
    outputscale_final_print = '%.3e' % outputscale_final

    #Evaluate at true value or close to a training point doing a sensitivity analysis
    if eval_Train == False:
        eval_p_base = torch.tensor(true_p)   
    else:
        eval_p_base = train_p[eval_theta_num*n,0:q]
#     print("Base Theta Train for Movies:", np.round(eval_p_base.numpy(),6) )
    
    #Define X_space for param sensitivity testing
    train_xspace_set = np.array([data_train[m + eval_theta_num*n,1:] for m in x_space_points])
    X_space= np.array([Xexp[m] for m in x_space_points])
    
    #Create list to save evaluated arrays in and arrays to store GP mean/stdev and true predictions in
    all_data_to_plot = np.zeros((3, len(x_space_points), q, value_num))
    GP_mean_all, GP_stdev_all, y_sim_all, values_array = eval_over_xspace(eval_p_base, bounds_p, value_num, X_space, train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types, noise_std, Case_Study)
                
    #Save data and send necessary items to outer loop
    all_data_to_plot[0,:, :, :],all_data_to_plot[1,:, :, :], all_data_to_plot[2,:, :, :]  = y_sim_all, GP_mean_all, GP_stdev_all
        
    return train_xspace_set, all_data_to_plot, lenscl_final, lenscl_noise_final, outputscale_final, values_array

def eval_over_xspace(eval_p_base, bounds_p, value_num, X_space, train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types = 0, noise_std =  0.01, Case_Study = 2.2):
    """
    Evaluates Muller potential sensitivity w.r.t theta values
    
     Parameters:
    -----------
        eval_p_base: ndarray, contains the parameter set for which to evaluate the GP
        bounds_p: ndarray, defines the bounds of the parameter space values (unused now, may use to define values later)
        value_num: int, defines the number of values of each parameter to test for the graph
        X_space: ndarray, The experimental data point array for X over which to evaluate the GP
        train_y: ndarray, The output variable training data
        true_model_coefficients: ndarray, The array containing the true values of the problem (may be the same as true_p)
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be Gaussian
        verbose: bool, Determines activeness of print statement, Default = False
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        Case_Study: int, float, the number of the case study to be evaluated. Default is 2.2, other option is 1         
    
    Returns:
    --------
        GP_mean_all: ndarray, Array of GP mean values
        GP_stdev_all: ndarray, Array of GP stdev values
        y_sim_all: ndarray, Array of Ysim values
        values_array, ndarray, Array of all values evalauted
    
    """
    q = bounds_p.shape[1]
    ##Evaluate parameter sets at each Xspace value
    #Loop over all parameters
    GP_mean_all = np.zeros((len(X_space), q, value_num) )
    GP_stdev_all = np.zeros((len(X_space), q, value_num) )
    y_sim_all = np.zeros((len(X_space), q, value_num) )
    
    #Create list to save evaluated parameter sets and values in
    eval_p_df = []
    values_list = [] 
    values_array = np.zeros((len(eval_p_base), value_num))
    
    for i in range(len(eval_p_base)):   
        #Clone the base value
        eval_p = eval_p_base.clone()
        #Define upper and lower theta bounds
        lower_theta = bounds_p[0,i]
        upper_theta = bounds_p[1,i]
        #Define Values to test
        values = np.linspace(lower_theta, upper_theta, value_num) #Note: Default to 41 
        values_array[i,:] = values
        values_list.append(values)
        #Save each bound value as a number from 0 to len(percentiles)
        val_num_map = np.linspace(0,len(values)-1, len(values))
        #Define parameter sets to test   
        for j in range(len(values)):   
            # Evaluate at the original point for each parameter and swap a parameter value for a value within the bounds
            new_eval_p = values[j]
            #Change the value to the exact point except for 1 variable that is rounded to 2 sig figs after modification by a percent
#             eval_p[i] = torch.tensor(float('%.2g' % float(new_eval_p)))
            #Or just do the exact value
            eval_p[i] = torch.tensor(float(new_eval_p))
            #Append evaluated value to this list only on 1st iteration of k
            eval_p_df.append(list(eval_p.numpy()))
            #Loop over Xspace Values: #Note. X_space defined as Xexp points we want to test
            for k in range(len(X_space)):
                #Evaluate the values
                eval_components_Xexp = eval_GP(eval_p, X_space[k], train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types = skip_param_types, noise_std = noise_std, CS = Case_Study)
                #Get GP predictions and true values
                GP_mean_Xexp, GP_stdev_Xexp, y_sim_Xexp = eval_components_Xexp
                #Append GP mean, GP_stdev, and true values to arrays outside of loop
                GP_mean_all[k,i,j], GP_stdev_all[k,i,j], y_sim_all[k,i,j] = GP_mean_Xexp, GP_stdev_Xexp, y_sim_Xexp 
    
    return GP_mean_all, GP_stdev_all, y_sim_all, values_array
        

def mul_plot_param_many(param_sens_data, set_lengthscale, train_iter, t, Case_Study, CutBounds, X_space, x_space_points, param_dict, values_list, val_num_map, lenscls, eval_theta_data, noises, opscls, kernel = "RBF", DateTime = None, X_train = None, save_csvs = False, save_figure = False, package = "", plot_one_param = None):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: list of 3 NxN arrays containing all points that will be plotted for GP_mean, GP standard deviation, and y_sim
        minima: ndarray, Array containing the minima of the true parameter set values
        saddle: ndarray, Array containing the saddle points of the true parameter set values
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        train_iter: int, number of training iterations to run for GP. Default is 300
        t: int, the total number of training points used to train the GP
        Case_Study: int, float, the number of the case study to be evaluated. Default is 1, other option is 2.2
        CutBounds: bool, Used for naming. Set True if bounds are cut from original values. Default False
        lenscl_final: "" or ndarray, The final lengthscale used by the GP
        train_xspace_set: ndarray, The base parameter set, state point, and y values, at which the GP was evaluated
        lenscl_noise_final: "" or ndarray, The noise of the final lengthscale used by the GP
        kernel: str, defines which kernel function was used. Default RBF
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        X_train: None or ndarray, The X values used in the training data. Default None
        save_csvs: bool, Determines whether csv data will be saved for plots. Default True
        save_figure: bool, Determines whether figures will be saved. Default False
        Mul_title: str, part of the path to save the CSV data for each piece of this figure. Corresponds to what is being plotted/saved
        param: str, part of the path to save the CSV data for each piece of this figure. Which parameter is being plotted/saved
        percentile: str, part of the path to save the CSV data for each piece of this figure. Which percentile is being plotted/saved
        package: str, Determines whether gpytorch or scikit learn will be used to build the GP model. Default "gpytorch"
        plot_one_param: None or int, if you only want to plot one parameter, define the integer associated w/ that parameter
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''      
    #Define constants
    lenscl_final = lenscls[0,0] #Since they are all the same
    lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
    lenscl_noise_final = noises[0,0]
    lenscl_noise_print = '%.3e' % lenscl_noise_final
    half = int(len(lenscl_print)/2)
    t_save = '_'.join(map(str, t))
    
    all_ones = np.all(opscls == 1)
    outputscl = not all_ones
        
    #Make a new plot for each X Coordinate and parameter value tested
    #loop over Xexp values
    for k in range(param_sens_data.shape[3]):        
        #None if looping over all parameters, use eval_theta_data.shape[1], only loop over 1 specific value otherwise
        if plot_one_param != None:
            range_val = 1   
        else:
            range_val = param_sens_data.shape[4]

        #Loop over parameter values
        for i in range(range_val):
            #Define figure based on number of unique training thetas that were evaluated
            eval_thetas = param_sens_data.shape[0]
            row_num = int(np.floor(np.sqrt(eval_thetas)))
            col_num = int(eval_thetas/row_num)
            assert row_num * col_num == eval_thetas, "row and col numbers must add to number of graphs" 
            fig, axes = plt.subplots(nrows = row_num, ncols = col_num, figsize = (7*col_num,6*row_num), squeeze=False)
            ax = axes.reshape(eval_thetas)
                
            #Define plot title
#             title_str = "Xexp Point " + str(x_space_points[k]+1) + '\n' + r'$\ell = $' + str(lenscl_print[:half]) + '\n' + str(lenscl_print[half:]) + "\n" + r'$\sigma_{\ell} = $' + lenscl_noise_print
            title_str = "Xexp Point " + str(x_space_points[k]+1) + '\n' + r'$\sigma_{\ell} = $' + lenscl_noise_print
            fig.suptitle(title_str)
            
            #Set i as the integer value corresponding to the parameter you want to plot if applicable
            if plot_one_param != None:
                i = plot_one_param   

            #Loop over each axis
            for ax_idx in range(eval_thetas):    
            #Add plot values
                #Loop over all training data values
                colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
                for train_idx in range(param_sens_data.shape[1]):
                    y_sim_data, GP_mean_data, GP_stdev_data = param_sens_data[ax_idx, train_idx]
                    train_xspace_set = eval_theta_data[ax_idx, train_idx]
                    outputscl_org = opscls[ax_idx, train_idx]
                    opscl_print = '%.3e' % outputscl_org
                    t_prnt = str(t[train_idx])  
                    ax[ax_idx].plot(values_list[i], GP_mean_data[k,i], label =str(t[train_idx]) + " TP, " + "ops = " + str(opscl_print), color = colors[train_idx])
                    ax[ax_idx].fill_between(values_list[i],
                                    GP_mean_data[k,i] - 1.96 * GP_stdev_data[k,i],
                                    GP_mean_data[k,i] + 1.96 * GP_stdev_data[k,i],
                                    alpha=0.3, color = colors[train_idx])
                ax[ax_idx].scatter(train_xspace_set[k,i], train_xspace_set[k,-1], label = "Training", marker ="*", color= colors[-1], s=80)
                ax[ax_idx].plot(values_list[i], y_sim_data[k,i], linestyle = "--", label = "Y Sim", color = colors[-1])

                #Set plot details        
            #     plt.legend(loc = "best")
                ax[ax_idx].set_xlabel(r'$' + param_dict[i] +'$', fontsize=16, fontweight='bold')
                ax[ax_idx].set_ylabel("Muller Potential", fontsize=16, fontweight='bold')
                ax_title = "Param Values: " + str(train_xspace_set[k,0:len(param_dict)].round(decimals=3))
                ax[ax_idx].set_title(ax_title, weight='bold',fontsize=10)

                lenscl_val_txt = r'$\ell = $' + str(lenscl_print[:half]) + '\n' + str(lenscl_print[half:]) + "\n" + r'$\sigma_{\ell} = $' + lenscl_noise_print #Since all lengthscales the same
#                 ax[ax_idx].text(0.5, -0.01, parm_val_txt, ha='center', wrap=True, transform = fig.transFigure)
#                 ax[ax_idx].tight_layout()
                
                #Get legend information
                if ax_idx == len(ax)-1:
                    handles, labels = ax[ax_idx].get_legend_handles_labels()
                    
           
            fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)
            fig.tight_layout()
    #             plt.grid(True)

            #Save CSVs and Figures
            if save_csvs == True:
                #Create a list to save paths in
                mul_val_title = ["/Sim_val", "/GP_mean", "/GP_stdev"]
                mul_vals = [param_sens_data[:, :, 0, k, i, :], param_sens_data[:, :, 1, k, i, :], param_sens_data[:, :, 2, k, i, :]]
                mul_val_paths = []               
                #For each values being plotted
                for j in range(len(mul_vals)):
                    #Create a path
                    mul_val_path = path_name_gp_val(set_lengthscale, train_iter, t_save, Case_Study, DateTime, is_figure = False, csv_end = "", CutBounds = CutBounds, X_val = k, val_title = mul_val_title[j], param = param_dict[i], kernel = kernel, package = package, outputscl = outputscl)
                    #Save path to list
                    mul_val_paths.append(mul_val_path) 

                csv_item_list_vals = mul_vals
                make_csv_list_vals = mul_val_paths
                for j in range(len(make_csv_list_vals)):
#                     print(make_csv_list_vals[j])
                    save_csv(csv_item_list_vals[j], make_csv_list_vals[j], ext = "npy")

            #Save figure or show and close figure
            if save_figure == True:
                path = path_name_gp_val(set_lengthscale, train_iter, t_save, Case_Study, DateTime, is_figure = True, CutBounds = CutBounds, X_val = k, val_title = "/Param_Sens_Multi_Theta", param = param_dict[i], kernel = kernel, package = package, outputscl = outputscl)
#                 print(path)
                save_fig(path, ext='png', close=True, verbose=False) 
            else:
                plt.show()
            plt.close()

    #Create paths for x training points and values
    if save_csvs == True:
        csv_ends = ["x_set_param_data", "Param_Values", "X_space"]
        eval_theta_data_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[0], CutBounds = CutBounds, X_val = "", val_title = "", param = "", kernel = "", package = "", outputscl = "")
        param_val_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[1], CutBounds = CutBounds, X_val = "", val_title = "", param = "", kernel = "", package = "", outputscl = "")
        x_spc_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[2], CutBounds = CutBounds, X_val = "", val_title = "", param = "", kernel = "", package = "", outputscl = "")
        # Create a list of extra items to save and corresponding path names
        csv_item_list = [eval_theta_data, np.array(values_list), X_space]
        make_csv_list = [eval_theta_data_path, param_val_path, x_spc_path]
        #Save values as CSVs
        for i in range(len(make_csv_list)):
#             print(make_csv_list[i])
            save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")

    return plt.show()

def path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime = None, is_figure = True, csv_end = None, CutBounds = False, X_val = "", val_title = "", param = "", kernel = "", package = "", outputscl = ""):
    """
    names a path based on given parameters
    
    Parameters
    ----------
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        train_iter: int, number of training iterations to run for GP. Default is 300
        t: int, int, Number of initial training points to use
        Case_Study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        is_figure: bool, used for saving CSVs as part of this function and for calling the data from a CSV to make a plot
        csv_end: str, the name of the csv file
        CutBounds: bool, Used for naming. Set True if bounds are cut from original values. Default False
        X_val: str, part of the path to save the CSV data for each piece of this figure. Corresponds to what X_value is being plotted/saved
        param: str, part of the path to save the CSV data for each piece of this figure. Which parameter is being plotted/saved
        percentile: str, part of the path to save the CSV data for each piece of this figure. Which percentile is being plotted/saved
        kernel: str, defines which kernel function was used. Default RBF
        package: str, Determines whether gpytorch or scikit learn will be used to build the GP model. Default "gpytorch"
        
    Returns:
    --------
        path: str, The path to which the file is saved
    
    """
    if X_val !="":
        X_value = "/" + "X_val_num_" + str(X_val)
    else:
        X_value = ""
        
    if package != "":
        pckg = "/"+ package
    else:
        pckg = ""
        
    if set_lengthscale == "":
        len_scl = ""
    elif set_lengthscale is not None:
        len_scl = "/len_scl_" + "%0.4f" % set_lengthscale
    else:
        len_scl = "/len_scl_varies"
        
    if train_iter == "":
        trn_iter = ""
    else:
        trn_iter = "/train_iter_" + str(train_iter)
        
    if t == "":
        org_TP_str = ""
    else:
        org_TP_str = "/TP_"+ str(t)
        
    CS = "/CS_" + str(Case_Study) 
    kernel_type = "/" + str(kernel)
    
    if outputscl == True:
        kernel_type = kernel_type + "_w_ops"
    
    if param != "":
        param = "-" + str(param)
            
    plot = val_title        
      
    if DateTime is not None:
#         path_org = "../"+DateTime #Will send to the Datetime folder outside of CS1
        path_org = DateTime #Will send to the Datetime folder outside of CS1
    else:
        path_org = "Test_Figs"
        
#         path_org = "Test_Figs"+"/Sep_Analysis2"+"/Figures"
    if is_figure == True:
        path_org = path_org + "/Figures"
    else:
        path_org = path_org + "/CSV_Data"
        
    path_end = CS + trn_iter + org_TP_str + pckg + kernel_type + len_scl + plot + X_value + param  

    if CutBounds == True:
        cut_bounds = "_CB"
    else:
        cut_bounds = ""
        
    path = path_org + "/GP_Vs_Sim_Comp" + cut_bounds + path_end 
        
    if csv_end is not None:
        path = path + csv_end
   
    return path