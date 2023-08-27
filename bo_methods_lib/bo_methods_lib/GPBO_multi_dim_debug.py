import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
import torch
import csv
import gpytorch
import scipy.optimize as optimize
import itertools
from itertools import combinations_with_replacement
from itertools import combinations
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import time
import Tasmanian

#Notes: Change line below when changing test problems: 
# If line 21 is active, the 8D problem is used, if line 22 is active, the 2D problem is used
from .CS2_create_data import calc_muller, create_sse_data, create_y_data, calc_y_exp, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point
# from .CS1_create_data import create_sse_data, create_y_data, calc_y_exp, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point

from .bo_functions_generic import LHS_Design, set_ep, test_train_split, find_train_doc_path, ExactGPModel, train_GP_model, calc_GP_outputs, explore_parameter, ei_approx_ln_term, calc_ei_emulator, get_sparse_grids, eval_GP_sparse_grid, calc_ei_basic, train_test_plot_preparation, clean_1D_arrays, norm_unnorm, train_GP_scikit, define_GP_model

from .normalize import normalize_x, normalize_p_data, normalize_p_set, normalize_p_true, normalize_constants, normalize_general, normalize_p_bounds

from .CS2_bo_plotters import value_plotter, plot_xy, plot_Theta, plot_Theta_min, plot_obj, plot_obj_abs_min, plot_3GP_performance, plot_sep_fact_min, save_fig, save_csv, path_name, plot_EI_abs_max, save_misc_data
# from CS2_bo_plotters import plot_org_train

from .CS2_bo_functions_multi_dim import eval_GP, eval_GP_basic_set, eval_GP_emulator_set, optimize_theta_set, find_opt_and_best_arg, argmax_multiple, find_opt_best_scipy, eval_GP_scipy, eval_GP_mean_std

from .Parm_Sens_Multi_Theta import path_name_gp_val

def bo_iter_w_runs(BO_iters,all_data_doc,t,theta_set,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, runs, sparse_grid, emulator, package, kernel, set_lengthscale, outputscl, initialize, true_model_coefficients, param_dict, bounds_p, bounds_x, verbose = False, save_fig=False, save_CSV = True, shuffle_seed = None, DateTime=None, sep_fact = 1, LHS = False, skip_param_types = 0, eval_all_pairs = False, normalize = True, case_study = 1):
    """
    Performs BO iterations with runs. A run contains of choosing different initial training data.
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iterations
        all_data_doc: csv name as a string, contains all training data for GP
        t: int, Number of total points to use
        theta_set: ndarray (len_set x dim_param), array of Theta values
        Theta_True: ndarray, The array containing the true values of theta parameters to regress- flattened array
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        Xexp: ndarray, The list of Xs that will be used to generate Y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        runs: int, The number of times to choose new training points
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
        emulator: bool, Determines if GP will model the function or the function error
        package: str ("gpytorch" or  "scikit_learn") determines which package to use for GP hyperaparameter optimization
        kernel: str ("Mat_52", Mat_32" or "RBF") Determines which GP Kerenel to use
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        outputscl: bool, Determines whether outputscale is trained
        initialize: int, number of times to restart GP training
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        bounds_p: ndarray, The bounds for searching for Theta_True.
        bounds_x: ndarray, The bounds of Xexp
        verbose: bool, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: bool, Determines whether figures will be saved. Default False
        save_CSV: bool, Determines whether CSVs will be saved. Default True
        shuffle_seed, int, number of seed for shuffling training data. Default is None. 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        LHS: bool, Whether theta_set was generated from an LHS set or a meshgrid. Default False
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        eval_all_pairs: bool, determines whether all pairs of theta are evaluated. Default False
        normalize: bool, determines whether input values are normalized. Default True
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        
    Returns:
    --------
        bo_opt: int, The BO iteration at which the lowest SSE occurs
        run_opt: int, The run at which the lowest SSE occurs
        Theta_Opt_all: ndarray, the theta values/parameter set that maps to the lowest SSE
        SSE_abs_min: float, the absolute minimum SSE found
        Theta_Best_all: ndarray, the theta values/parameter set that maps to the highest EI
    
    """
    #Assert statements
    assert all(isinstance(i, int) for i in [BO_iters, t,runs,train_iter, skip_param_types, runs, initialize]), "integer variables must be integers"
#     print(type(emulator), type(verbose), type(eval_all_pairs), type(outputscl), type(sparse_grid), type(normalize))
    assert all(isinstance(i, (bool, np.bool_)) for i in [emulator, verbose, eval_all_pairs, outputscl, sparse_grid, normalize]), "Boolean variables must be booleans"
    assert BO_iters > 0, "Number of BO Iterations must be greater than 0!"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert package == "scikit_learn" or package == "gpytorch", "Package must be scikit_learn or gpytorch"
    assert isinstance(kernel, str) == True, "kernel_func must be a string!"
    
    #Find values of dimensions of Xexp (m), number of experimental data (n), dimensionality of theta (q), and number of data (t)
    m = Xexp.shape[1]
    n = Xexp.shape[0]
#     m = Xexp[0].size #Dimensions of X
    q = len(Theta_True) #Number of parameters to regress
#     p = theta_mesh.shape[1] #Number of training points to evaluate in each dimension of q
    ep0 = explore_bias
    
    dim = m+q #dimensions in a CSV
    #Read data from a csv
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=",")) 
    
    #Initialize Theta and SSE matricies
    Theta_Opt_matrix = np.zeros((runs,BO_iters,q))
    Theta_Opt_abs_matrix = np.zeros((runs,BO_iters,q))
    Theta_Best_matrix = np.zeros((runs,BO_iters,q))
    SSE_matrix = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    EI_matrix = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    EI_matrix_abs_max = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    SSE_matrix_abs_min = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    Total_BO_iters_matrix = np.zeros(runs)
    time_per_iter_matrix = np.zeros((runs,BO_iters))
    
    GP_mean_matrix = np.zeros((runs,BO_iters,n)) #Saves ln(SSE) values
    GP_var_matrix = np.zeros((runs,BO_iters,n)) #Saves ln(SSE) values
    
    #Loop over # runs
    for i in range(runs):
#         print("Run Number: ",i+1)
        if verbose == True or save_fig == False:
            print("Run Number: ",i+1)
        #Note: sep_fact can be used to use less training data points
        train_data, test_data = test_train_split(all_data, runs = int(i), sep_fact = sep_fact, shuffle_seed=shuffle_seed)
        train_p, train_y = train_data[:,1:-1], train_data[:,-1]
        test_p, test_y = test_data[:,1:-1], test_data[:,-1]

        assert len(train_p) == len(train_y), "Training data must be the same length"
        if emulator == True:
            assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
        else:
            assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
        
        #Plot all training data
        #This works, put it back when we need it (12/13/22)
#         train_test_plot_preparation(q, m, theta_set, train_p, test_p, Theta_True, Xexp, emulator, sparse_grid, obj, ep0, set_lengthscale, i, save_fig, BO_iters, runs, DateTime, verbose, param_dict, sep_fact, normalize)

        #Normalize data as appropriate
        if normalize == True:
            norm_vals, norm_scalers = normalize_general(bounds_p, train_p, test_p, bounds_x, Xexp, theta_set, Theta_True, true_model_coefficients, emulator, skip_param_types, case_study)
            bounds_p_scl, train_p_scl, test_p_scl, bounds_x_scl, Xexp_scl, theta_set_scl, Theta_True_scl, true_model_coefficients_scl = norm_vals
            scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
#             print(norm_vals, norm_scalers)
        else:
            norm_scalers = None
            bounds_p_scl, train_p_scl, test_p_scl, bounds_x_scl, Xexp_scl, theta_set_scl, Theta_True_scl, true_model_coefficients_scl =  bounds_p, train_p, test_p, bounds_x, Xexp, theta_set, Theta_True, true_model_coefficients
                             
        #Run BO Iteration
        BO_results = bo_iter(BO_iters,train_p_scl,train_y,theta_set_scl,Theta_True_scl,train_iter,explore_bias, Xexp_scl, Yexp, noise_std, obj, i, sparse_grid, emulator, package, kernel, set_lengthscale, outputscl, initialize, true_model_coefficients_scl, param_dict, bounds_p_scl, verbose, save_fig, save_CSV, runs, DateTime, test_p_scl, sep_fact = sep_fact, LHS = LHS, skip_param_types = skip_param_types, eval_all_pairs = eval_all_pairs, normalize = normalize, norm_scalers = norm_scalers, case_study = case_study)
        
        #Add all SSE/theta results at each BO iteration for that run
        Theta_Best_matrix[i,:,:] = BO_results[0]
        Theta_Opt_matrix[i,:,:] = BO_results[1]
        SSE_matrix[i,:] = BO_results[2]
        SSE_matrix_abs_min[i] = BO_results[3]
        Total_BO_iters_matrix[i] = BO_results[4]
        Theta_Opt_abs_matrix[i,:,:] = BO_results[5]
        EI_matrix_abs_max[i] = BO_results[6]
        GP_mean_matrix[i,:,:] = BO_results[7]
        GP_var_matrix[i,:,:] = BO_results[8]
        time_per_iter_matrix[i,:] = BO_results[9]
#         print(time_per_iter_matrix)
        
    #Save GP mean and Var and time/iter here
    if save_CSV == True:
        fxn_name_list = ["GP_mean_vals", "GP_var_vals", "time_per_iter"]
        fxn_data_list = [GP_mean_matrix, GP_var_matrix, time_per_iter_matrix]
        for i in range(len(fxn_name_list)):
            save_misc_data(fxn_data_list[i], fxn_name_list[i], t, obj, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, tot_iter=BO_iters, tot_runs=runs, DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
    
    #Calculate median time
    median_time_per_iter = np.median(time_per_iter_matrix[np.nonzero(time_per_iter_matrix)])
    print("Median BO Iteration Time (s): ", median_time_per_iter)
        
    #Plot all SSE/theta results for each BO iteration for all runs
    if runs >= 1 and BO_iters > 1:
        plot_Theta(Theta_Opt_matrix, Theta_True, t, obj,ep0, emulator, sparse_grid,  set_lengthscale, save_fig, param_dict, BO_iters,
                   runs, DateTime, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)
        plot_Theta_min(Theta_Opt_abs_matrix, Theta_True, t, obj,ep0, emulator, sparse_grid, set_lengthscale, save_fig, param_dict,
                       BO_iters, runs, DateTime, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize) 
        
        plot_obj(SSE_matrix, t, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)
        plot_obj_abs_min(SSE_matrix_abs_min, emulator, ep0, sparse_grid, set_lengthscale, t, obj, save_fig, BO_iters, runs, DateTime, 
                         sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)
        plot_EI_abs_max(EI_matrix_abs_max, emulator, ep0, sparse_grid, set_lengthscale, t, obj, save_fig, BO_iters, runs, DateTime, 
                        sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)
          
    
    #Find point corresponding to absolute minimum SSE and max(-ei) at that point
    #Find lowest nonzero sse point
    argmin = np.array(np.where(np.isclose(SSE_matrix, np.amin(SSE_matrix[np.nonzero(SSE_matrix)]),rtol=abs(np.amin(SSE_matrix)*1e-6))==True))
    
    if len(argmin) > 1:
        rand_ind = np.random.randint(argmin.shape[1]) #Chooses a random point with the minimum value
        argmin = argmin[:,rand_ind]    
    SSE_abs_min = np.amin(SSE_matrix[np.nonzero(SSE_matrix)])
#     SSE_abs_min = np.amin(SSE_matrix)
    run_opt = int(argmin[0]+1)
    bo_opt = int(argmin[1]+1)
    
    #Find theta value corresponding to argmin(SSE) and corresponding argmax(ei) at which run and theta value they occur
    Theta_Best_all = np.array(Theta_Best_matrix[tuple(argmin)+(Ellipsis,)])
    Theta_Opt_all = np.array(Theta_Opt_matrix[tuple(argmin)+(Ellipsis,)])
    
    return bo_opt, run_opt, Theta_Opt_all, SSE_abs_min, Theta_Best_all

def bo_iter(BO_iters,train_p,train_y,theta_set,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, run, sparse_grid, emulator, package, kernel, set_lengthscale, outputscl, initialize, true_model_coefficients, param_dict, bounds_p, verbose = False,save_fig=False, save_CSV = True, tot_runs = 1, DateTime=None, test_p = None, sep_fact = 1, LHS = False, skip_param_types = 0, eval_all_pairs = False, normalize = False, norm_scalers = None, case_study = 1):
    """
    Performs BO iterations
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iteratiosn
        train_p: tensor or ndarray, The training parameter space data
        train_y: tensor or ndarray, The training y data
        theta_set: ndarray (len_set x dim_param), array of Theta values
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2 (must be 1D)
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        run: int, The iteration of the number of times new training points have been picked
        sparse_grid: bool: Determines whether a sparse grid or approximation is used for the GP emulator
        emulator: bool, Determines if GP will model the function or the function error
        package: str ("gpytorch" or  "scikit_learn") determines which package to use for GP hyperaparameter optimization
        kernel: str ("Mat_52", Mat_32" or "RBF") Determines which GP Kerenel to use
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        outputscl: bool, Determines whether utfutscale is trained
        initialize: int, number of times to restart GP training
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        bounds_p: ndarray, The bounds for searching for Theta_True.
        verbose: bool, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: bool, Determines whether figures will be saved. Default False
        save_CSV: bool, Determines whether CSVs will be saved. Default True
        tot_runs: The total number of runs to perform. Default 1
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        test_p: None, tensor, or ndarray, The testing parameter space data. Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        LHS: bool, Whether theta_set was generated from an LHS set or a meshgrid. Default False
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        eval_all_pairs: bool, determines whether all pairs of theta are evaluated. Default False
        normalize: bool, determines whether data is normalized. Default False
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
        
        
    Returns:
    --------
        All_Theta_Best: ndarray, Array of all Best Theta values (as determined by max(ei)) for each iteration 
        All_Theta_Opt: ndarray, Array of all Optimal Theta values (as determined by min(sse)) for each iteration
        All_SSE: ndarray, Array of all minimum SSE values (as determined by min(sse)) for each iteration
        All_SSE_abs_min: ndarray, Array of the absolute minimum SSE values (as determined by min(sse)) at each iteration 
        Total_BO_iters: int, The number of BO iteration actually completed    
        All_Theta_abs_Opt: ndarray, Array of all minimum Optimal Theta values (as determined by min(sse)) for each iteration
        All_Max_EI: ndarray, Array of the absolute maximum EI values (as determined by min(-ei)) at each iteration   
        gp_mean_all_mat: ndarray, Array of all GP mean values for emulator approximation, array of zeros for standard
        gp_var_all_mat: ndarray, Array of all GP variance values for emulator approximation, array of zeros for standard 
        time_per_iter: ndarray, Array of timer per iteration
        
    """
    
    #Assert Statments
    assert len(train_p) == len(train_y), "Training data must be the same length"

    #Find values of dimensions of Xexp (m), number of experimental data (n), dimensionality of theta (q), and number of data (t)
    n,m = Xexp.shape #Dimensions of X
    q = len(Theta_True) #Number of parameters to regress
    t = int(len(train_p)) + int(len(test_p)) #Original length of all data

    #Find number of data points and set a parameter for the original exploration bias
    data_points = int(np.sqrt(len(theta_set)))
#     print(data_points)
    ep0 = explore_bias
    
    #Set arrays to track values for every BO iteration
    All_Theta_Best = np.zeros((BO_iters,q)) 
    All_Theta_abs_Opt = np.zeros((BO_iters,q))
    All_Theta_Opt = np.zeros((BO_iters,q)) 
    All_SSE = np.zeros(BO_iters) #Will save ln(SSE) values
    All_SSE_abs_min = np.zeros(BO_iters) #Will save ln(SSE) values  
    All_Max_EI = np.zeros(BO_iters) #Used in stopping criteria
    Total_BO_iters = BO_iters
    gp_mean_all_mat = np.zeros((BO_iters, n))
    gp_var_all_mat = np.zeros((BO_iters, n))
    time_per_iter = np.zeros(BO_iters)
    
    
    #Ensures GP will take correct # of inputs
    if emulator == True:
        GP_inputs = q+m
        assert train_p.shape[1] == q+m, "train_p must have the same number of dimensions as the value of q+m"
    else:
        GP_inputs = q
        assert train_p.shape[1] == q, "train_p must have the same number of dimensions as the value of q"
    
    #Initialize mean of variance and best error number
    mean_of_var = 0
    best_error_num = 0
    ep_init = explore_bias

    theta_set = torch.tensor(theta_set).float()
    
    #Loop over # of BO iterations
    for i in range(BO_iters):
        #Start timer for BO loop
        timestart = time.time()
        #Converts numpy arrays to tensors after 1st pass 
        train_p = torch.from_numpy(train_p).float()
        train_y = torch.from_numpy(train_y).float()

        if torch.is_tensor(test_p) != True:
            test_p = torch.tensor(test_p)
            
        #Redefine likelihood and model based on new training data
        #Define model and likelihood, print lengthscale, noise, and outputscale
        gp_model_params = define_GP_model(package, noise_std, train_p, train_y, kernel, set_lengthscale, outputscl, initialize, train_iter, True)
        model, likelihood, lenscl_final, lenscl_noise_final, outputscale_final = gp_model_params
        print("GP training complete!")
        #Set Exploration parameter
#         explore_bias = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Constant") #Defaulting to exp method
        explore_bias = ep_init #Sets ep to the multiplicative scaler between 0.1 and 1
        
        #Evaluate GP to find sse and ei for optimization step at each theta in the LHS
        eval_components = eval_GP(theta_set, train_y, explore_bias, Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj, skip_param_types, norm_scalers)

        #If the emulator approach is used, gp_mean_all and gp_var_all are saved, and there are no internal EI parameters to report
        if emulator == True:
            ei,sse,var,stdev,best_error,gp_mean_all, gp_var_all = eval_components
        #If emulator is false, gp_mean_all and gp_var_all are redundant, because they are the same as sse and var. 
        else:
            ei,sse,var,stdev,best_error = eval_components

        #solve for opt and best based on the highest ei and lowest sse from the GP evaluations of theta_set
        theta_b, theta_o = optimize_theta_set(Xexp, Yexp, theta_set, true_model_coefficients, train_y, train_p, sse, ei, model, likelihood, explore_bias, emulator, sparse_grid, verbose, obj, bounds_p, skip_param_types, norm_scalers)
#         print(theta_b, theta_o)

        print("Theta Best and Theta Opt Found with Scipy")
        
        ##Evaluate all pairs of theta to make heat maps if necessary
        if eval_all_pairs == True:
            ##Evaluate with theta_o and theta_b both equal to true p
            eval_all_theta_pairs(q, theta_set, data_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, save_CSV, param_dict, i, run, BO_iters, tot_runs, DateTime, t,  true_model_coefficients, bounds_p, sep_fact, skip_param_types, normalize, norm_scalers)
            print("Original Heat Maps Evaluated")
            
#             eval_all_theta_pairs(q, theta_set, data_points, Theta_True, Xexp, Yexp, Theta_True, Theta_True, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, save_CSV, param_dict, i, run, BO_iters, tot_runs, DateTime, t,  true_model_coefficients, bounds_p, sep_fact, skip_param_types, normalize, norm_scalers)
#             print("True Value Heat Maps Evaluated")          
        
        #Evaluate GP for best EI theta set
        torch_theta_b = torch.tensor(np.array([theta_b]))
        eval_components_best = eval_GP(torch_theta_b, train_y, explore_bias, Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj, skip_param_types, norm_scalers)
        
        print("Best Theta Set Evaluated")

        #Determine which parameters will be plotted given the method type and whether verbose is T/F. Save parameters to plot to a list   
        #If the emulator approach is used, gp_mean_all and gp_var_all are saved, and there are no internal EI parameters to report
        if emulator == True:
            ei_best,sse_best,var_best,stdev_best,best_error_best,gp_mean_all_best, gp_var_all_best = eval_components_best
            gp_mean_all_mat[i] = gp_mean_all_best
            gp_var_all_mat[i] = gp_var_all_best
        #If emulator is false, eval_GP does not return the individual parameters important in the calculation of EI and gp_mean_all and gp_var_all are redundant, because they are the same as sse and var. 
        else:
            ei_best,sse_best,var_best,stdev_best,best_error_best = eval_components_best
            gp_mean_all_mat[i] = sse_best
            gp_var_all_mat[i] = var_best
     
        #Save Figures
        #Update timer
        timeend = time.time()
        time_per_iter[i] +=  (timeend - timestart)
              
        #Write code to save these values
#         print(ei,sse,var,stdev,best_error)
        
        #Update timer
        timestart = time.time()

        #Add max_EI to list
        All_Max_EI[i] = ei_best
        
#         mean_of_var = np.average(var)
# #         print("MOV",mean_of_var)
#         best_error_num = best_error
           
        #Save unnormalized theta_best and theta_opt values for iteration. Print values if verbose is true
        if normalize == False:
            theta_b_unscl, theta_o_unscl = theta_b, theta_o
        else:
            scaler_theta = norm_scalers[1]
            theta_b_unscl = normalize_p_true(theta_b, scaler_theta, norm= False)
            theta_o_unscl = normalize_p_true(theta_o, scaler_theta, norm= False)
        
        All_Theta_Best[i], All_Theta_Opt[i] = theta_b_unscl, theta_o_unscl
        
        #Calculate values of y given the GP optimal theta values
        if case_study == 1:
            y_GP_Opt = gen_y_Theta_GP(Xexp, theta_o, true_model_coefficients, skip_param_types, norm_scalers, emulator)
        else:
            y_GP_Opt = gen_y_Theta_GP(Xexp, theta_o, true_model_coefficients, Theta_True, case_study, skip_param_types, norm_scalers, emulator)
        
        #Calculate GP SSE and save value
        ln_error_mag = np.log(np.sum((y_GP_Opt-Yexp)**2)) 
        All_SSE[i] = ln_error_mag
        
        if verbose == True:
            print("Magnitude of ln(SSE) given Theta_Opt = ",theta_o_unscl, "is", "{:.4e}".format(ln_error_mag))
            print("Scipy Theta Best = ",theta_b_unscl)
            print("Scipy Theta Opt = ",theta_o_unscl)

        #Save best value of SSE for plotting          
        if i == 0:
        #At the first iteration, the minimum SSE is the only SSE and the min theta is the first theta
            All_SSE_abs_min[i] = ln_error_mag
            All_Theta_abs_Opt[i] = theta_o_unscl
            improvement = False

        #Otherwise, the minimum SSE and theta_o that correspond are determined by the ln_error_mag
        else:
        #If ln_error_mag is smaller than before it is the new best value, otherwise it remains the same
            if All_SSE_abs_min[i-1] >= ln_error_mag:
                All_SSE_abs_min[i] = ln_error_mag
                All_Theta_abs_Opt[i] = theta_o_unscl
                improvement = True
            else: 
                All_SSE_abs_min[i] = All_SSE_abs_min[i-1]
                All_Theta_abs_Opt[i] = All_Theta_abs_Opt[i-1]
                improvement = False
        
        #Prints certain values at each iteration if verbose is True
        if verbose == True:
            print("BO Iteration = ", i+1)
            print("Exploration Bias Factor = ",explore_bias)
            print("EI_max =", ei_best, "\n")
                    
        ##Append best values to training data 
        #Convert training data to numpy arrays to allow concatenation to work
        train_p = train_p.numpy() #(q x t)
        train_y = train_y.numpy() #(1 x t)
        
        #Save unnormalized Training data for this iteration in CSV          
        if normalize == True:
            train_p_unscl = normalize_p_data(train_p, m, emulator, norm, scaler_theta)[0]
            test_p_unscl = normalize_p_data(train_p, m, emulator, norm, scaler_theta)[0]
        else:
            train_p_unscl = train_p
            test_p_unscl = test_p
            
        #Update timer
        timeend = time.time()
        time_per_iter[i] +=  (timeend - timestart)
        
        if save_CSV == True:
            df_list = [train_p_unscl, test_p_unscl]
            df_list_ends = ["Train_p", "Test_p"]
            fxn = "value_plotter"
            title_save_TT = "Train_Test_Data"

            for j in range(len(df_list)):
                array_df = pd.DataFrame(df_list[j])
                path_csv = path_name(emulator, explore_bias, sparse_grid, fxn, set_lengthscale, t, obj, None, i, title_save_TT, run, tot_iter=Total_BO_iters, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[j], normalize = normalize)
    #             print(path_csv)
                save_csv(array_df, path_csv, ext = "npy") #Note: Iter 3 means the TPs used in calculations for iter 3 to determine iter 4
        
        #Update timer
        timestart = time.time()
        
        #End BO loops if max(EI) < 1e-10 for more than 1 iteration
        if  i > 0:
            #Change to 1e-7
            if abs(All_Max_EI[i-1]) <= 1e-10 and abs(All_Max_EI[i]) <= 1e-10:
                Total_BO_iters = i+1
                break

        #Append theta_b and y/sse value associated with it to training_data
        train_p, train_y = make_next_point(train_p, train_y, theta_b, Xexp, Yexp, emulator, true_model_coefficients, obj, q, skip_param_types, noise_std, norm_scalers)
        
        #Update Timer
        timeend = time.time()
        time_per_iter[i] +=  (timeend - timestart)
        print("Iteration Complete!")
              
    return All_Theta_Best, All_Theta_Opt, All_SSE, All_SSE_abs_min, Total_BO_iters, All_Theta_abs_Opt, All_Max_EI, gp_mean_all_mat, gp_var_all_mat, time_per_iter 


def eval_all_theta_pairs(dimensions, theta_set, n_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, save_CSV, param_dict, bo_iter, run, BO_iters, tot_runs, DateTime, t, true_model_coefficients, bounds_p, sep_fact = 1, skip_param_types = 0, normalize = False, norm_scalers = None):
    """
    Evaluates all combinations of theta pairs to make heat maps
    
    Parameters:
    -----------
        dimensions: int, Number of parameters to regress
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        n_points: int, The number of points in each vector of parameter space
        Theta_True: ndarray, A 2x1 containing the true input parameters
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_o: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        theta_b: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        train_p: tensor or ndarray, The training parameter space data
        train_y: ndarray, The output training data
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be Gaussian
        verbose: bool, Determines whether z and ei terms are printed
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep0: float, float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        explore_bias: float, the current numerical bias towards exploration
        emulator: bool, Determines if GP will model the function or the function error
        sparse_grid: bool, bool: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_fig: bool, Determines whether figures will be saved
        save_CSV: bool, Determines whether CSVs will be saved
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run: int or None, The iteration of the number of times new training points have been picked
        BO_iters: int, total number of BO iterations
        tot_runs: int, total number of runs
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, Number of total data points to use
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default 1
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        normalize: bool, determines whether data is normalized. Default False
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
    Returns:
    --------
        None - Saves graphs and CSVs     
        
    """
    #Create a linspace for the number of dimensions
    dim_list = np.linspace(0,dimensions-1,dimensions)
    
    #Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
    mesh_combos = np.array(list(combinations(dim_list, 2)), dtype = int)
    
    #Loop over all possible theta combinations of 2
    for i in range(len(mesh_combos)):
        #Set the indecies of theta_set to evaluate and plot as each row of mesh_combos
        indecies = mesh_combos[i]
        #Finds the name of the parameters that correspond to each index. There will only ever be 2 here since the purpose of the function called here is to plot in 2D
        param_names_list = [param_dict[indecies[0]], param_dict[indecies[1]]]
        #Evaluate and plot each set of values over a grid
        eval_and_plot_GP_over_grid(theta_set, indecies, n_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, save_CSV, param_names_list, bo_iter, run, BO_iters, tot_runs, DateTime, t, true_model_coefficients, sep_fact, skip_param_types, normalize, norm_scalers)
    return

def eval_and_plot_GP_over_grid(theta_set_org, indecies, n_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, save_CSV, param_names_list, bo_iter, run, BO_iters, tot_runs, DateTime, t, true_model_coefficients, sep_fact, skip_param_types = 0, normalize = False, norm_scalers = None):
    '''
    Makes heat maps given a combination of theta pairs
    
    Parameters:
    -----------
        theta_set_org: ndarray, The original set of theta values to look at from a mashgrid or LHS
        indecies: ndarray, The 2 indecies referring to column/parameter types that will be plotted
        n_points: int, The number of points in each vector of parameter space
        Theta_True: ndarray, A 2x1 containing the true input parameters
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_o: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        theta_b: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        train_p: tensor or ndarray, The training parameter space data
        train_y: ndarray, The output training data
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be Gaussian
        verbose: bool, Determines whether z and ei terms are printed
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep0: float, float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        explore_bias: float, the current numerical bias towards exploration
        emulator: bool, Determines if GP will model the function or the function error
        sparse_grid: bool, bool: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_fig: bool, Determines whether figures will be saved
        save_CSV: bool, Determines whether CSVs will be saved
        param_names_list: list, list of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run: int or None, The iteration of the number of times new training points have been picked
        BO_iters: int, total number of BO iterations
        tot_runs: int, total number of runs
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, Number of total data points to use
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP.
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        normalize: bool, determines whether data is normalized. Default False
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
        
    Returns:
    --------
        None - Saves graphs and CSVs     
        
    '''
    #Add true_p=best_p for titles 
    if all(np.array_equal(Theta_True, arr) for arr in (theta_o, theta_b)) == True:
        true_map = "_best_is_true_p"
    else:
        true_map = ""
    #Clean shape of Xexp from a 1D arrays to shape (len(Xexp), 1)
    Xexp = clean_1D_arrays(Xexp)
    #Clean shape of theta_true from a 1D arrays to shape (1, len(theta_true))
    Theta_True_clean = clean_1D_arrays(Theta_True, param_clean = True)
    
    #Define dimensions and length of Xexp and theta_true
    len_x, dim_x = Xexp.shape
    len_data, dim_data = Theta_True_clean.shape
    
    #Create a set that has the 2 columns changing, but eveything else is based on theta_opt value
    theta_set = theta_set_org.clone()
    
    print(theta_set)
    
    #Loop over all dimenisons (columns) in theta_set_org
    for i in range(theta_set_org.shape[1]):
        #If the column index is the same as either of the indecies we want to change, overwrite theta_set with those values
        if i == indecies[0]:
            theta_set[:,i] = theta_set_org[:,i]
        elif i == indecies[1]:
            theta_set[:,i] = theta_set_org[:,i]
        #If the column is not changing, use the theta_b value
        else:
            theta_set[:,i] = theta_b[i]

    #Evaluate GP at new points
    eval_components = eval_GP(theta_set, train_y, explore_bias, Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj, skip_param_types = skip_param_types, norm_scalers = norm_scalers)
    
    #Determine which parameters will be plotted given the method type. Save parameters to plot to a list    
    if emulator == True:
        #If the emulator approach is used, gp_mean_all and gp_var_all are saved, and there are no internal EI parameters to report
        ei,sse,var,stdev,best_error,gp_mean_all, gp_var_all = eval_components
        list_of_plot_variables = [ei,sse,var,stdev,best_error,gp_mean_all,gp_var_all]
    else:
        #If emulator is false, eval_GP does not return the individual parameters important in the calculation of EI and gp_mean_all and gp_var_all are redundant, because they are the same as sse and var. 
        ei,sse,var,stdev,best_error = eval_components
        list_of_plot_variables = [ei,sse,var,stdev,best_error]
    
    #Loop over all plotting variables
    for i in range(len(list_of_plot_variables)):
        #Reshape plotting variables that are not floats to the correct shapes for plotting. 
        # Note: Only best error will fall outside of this if statement
        if len(list_of_plot_variables[i].shape) == 2: #Only triggers for gp_mean_all and gp_var_all
            list_of_plot_variables[i] = list_of_plot_variables[i].reshape((n_points, n_points, -1)).T
        elif isinstance(list_of_plot_variables[i], (np.float64, np.float32)) == False:
            list_of_plot_variables[i] = list_of_plot_variables[i].reshape((n_points, -1)).T
        else:
            #Any list variable that is only 1 value will not be plotted and doesn't need to be reshaped
            list_of_plot_variables[i] = list_of_plot_variables[i]
            
    #Set titles for plots
    if emulator == False:
        titles = ['E(I(\\theta))','log(e(\\theta))','\sigma^2','\sigma','Best_Error']  
        titles_save = ["EI","ln(SSE)","Var","StDev","Best_Error"] 
    else:
        titles = ['E(I(\\theta))','log(e(\\theta))','\sigma^2', '\sigma', 'Best_Error', 'GP Mean', 'GP Variance']  
        titles_save = ["EI","ln(SSE)","Var","StDev","Best_Error", "GP_Mean", "GP_Var"] 
    
    #If true and theta_o are the same, add marker so that saving names are different
    for i in range(len(titles)):
        titles_save[i] == titles_save[i] + true_map
        
    #Create train_p_unscl
    train_p_unscl = train_p.clone()
    
    #Unscale data before plotting if necessary
    if normalize == True: 
        #Norm = False will unscale data
        norm = False
        #Unpack scalers
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
        #Unscale data to original values
        theta_set_org = normalize_p_set(theta_set, scaler_theta, norm)
        theta_b = normalize_p_true(theta_b, scaler_theta, norm)
        theta_o = normalize_p_true(theta_o, scaler_theta, norm)
        Theta_True = normalize_p_true(Theta_True, scaler_theta, norm)
        #Ensure that only parameter value columns in train_p will be plotted. Since everything after this is redefining or plotting, train_p is simply redefined if the emulator approach is being used
        if emulator == True:
            train_p_unscl[:,0:-dim_x] = normalize_p_data(train_p[:,0:-dim_x], dim_x, emulator, norm, scaler_theta)
            train_p_unscl[:,-dim_x:] = normalize_x(Xexp, train_p[:,-dim_x:], norm, scaler_x)[0]
        else:
            train_p_unscl  = normalize_p_data(train_p, dim_x, emulator, norm, scaler_theta)
            
    #Generate meshgrid and theta_set from unnormalized meshgrid 
    theta_set_org = np.array(theta_set_org)
    Theta1_lin = np.linspace(np.min(theta_set_org[:,indecies[0]]),np.max(theta_set_org[:,indecies[0]]), n_points)
    Theta2_lin = np.linspace(np.min(theta_set_org[:,indecies[1]]),np.max(theta_set_org[:,indecies[1]]), n_points)
    theta_mesh = np.array(np.meshgrid(Theta1_lin, Theta2_lin)) 
    
    #Build training data for new model
    xx,yy = theta_mesh
    
    #Redefine where GP_SSE_min, EI_max, and true values are based on which parameters are being plotted
    theta_o = np.array([theta_o[indecies[0]], theta_o[indecies[1]]])
    theta_b = np.array([theta_b[indecies[0]], theta_b[indecies[1]]])
    #Note, clean_1D_theta arrays assures shape of (1, len(train_p)) for each column of train_p that will be plotted
    #Note: We concatenate the 2 columns for train_p that will be plotted after normalizing the values
    train_p_plot = np.concatenate(( clean_1D_arrays(train_p_unscl[:,indecies[0]]) , clean_1D_arrays(train_p_unscl[:,indecies[1]]) ), axis = 1)
    Theta_True = np.array([Theta_True[indecies[0]], Theta_True[indecies[1]]])
    
    #Plot and save figures for EI
    value_plotter(theta_mesh, list_of_plot_variables[0], Theta_True, theta_o, theta_b, train_p_plot, titles[0],titles_save[0], obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, param_names_list, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)

    #Ensure that a plot of SSE (and never ln(SSE)) is drawn
    if obj == "LN_obj" and emulator == False:
        plot_sse = list_of_plot_variables[1]
    else:
        plot_sse = np.log(list_of_plot_variables[1])

    #Plot and save figures for SSE
    value_plotter(theta_mesh, plot_sse, Theta_True, theta_o, theta_b, train_p_plot, titles[1], titles_save[1], obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, param_names_list, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)

    #Plot and save other figures
    #Loop over remaining variables to be saved and plotted
    for j in range(len(list_of_plot_variables)-2):
        #Define a component to be plotted and find the title to save it by
        component = list_of_plot_variables[j+2]       
        title = titles[j+2]
        title_save = titles_save[j+2]
        #Plot 2D components and print the best error value: GP_mean_all and gp_var_all are not printed or saved for all values
        if isinstance(component, (np.float32, np.float64)) == False:
            if title not in ['GP Mean','GP Variance']: 
#                 print(type(component), component.shape)
                value_plotter(theta_mesh, component, Theta_True, theta_o, theta_b, train_p_plot, title, title_save, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, param_names_list, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)
            else:
                for k in range(len(Xexp)):
                    title_plt_save = title_save+"_Xexp_"+str(k+1)
                    title_plt = title + ": Xexp = " + str(Xexp[k])
                    value_plotter(theta_mesh, component[k], Theta_True, theta_o, theta_b, train_p_plot, title_plt, title_plt_save, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, param_names_list, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact, save_CSV = save_CSV, normalize = normalize)
        elif isinstance(component, (np.float32, np.float64)) == True and title == "Best_Error" :
            Best_Error_Found = np.round(component,4)
            if verbose == True:
                print("Best Error is:", Best_Error_Found)
    return

def eval_GP_mean_std_ysim(theta_set, Xexp, Yexp, true_model_coefficients, model, likelihood, X_space = None, skip_param_types=0, CS=1):
    """ 
    Calculates the expected improvement of the emulator approach
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by (gpytorch ot scikitlearn method)
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood or None
        X_space: None or ndarray, The points for X over which to evaluate the GP (p^2 x dim(x) or n x dim(x))
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
    
    Returns
    -------
        GP_mean: ndaarray, Array of GP mean predictions at X_space and theta_set
        GP_stdev: ndarray, Array of GP variances related to GP means at X_space and theta_set
        y_sim: ndarray, simulated values at X_space and theta_set
    """
    #Set theta_set to only be parameter values
    theta_set_params = np.array( theta_set )
    
    #Ensure correct shapes of data
    if len(X_space.shape) < 2:
        X_space = clean_1D_arrays(X_space, param_clean = True)
    if len(theta_set_params.shape) < 2:
        theta_set_params = clean_1D_arrays(theta_set_params, param_clean = True)
   
    #Define dimensionality of X
    n,m = X_space.shape
      
    #Define the length of theta_set and the number of parameters that will be regressed (q)
    len_set, q = theta_set_params.shape
    
    ##Calculate Values
    #Define a parameter set, point
    theta_set_val = theta_set_params[0]
    point = list(theta_set_val)
    if X_space is not None:
        X_val = list(X_space.flatten()) #astype(np.float)
        #Append Xexp_k to theta_set to evaluate at theta, xexp_k
        x_point_data = list(X_space.flatten()) #astype(np.float)
        #Create point to be evaluated
        point = point + x_point_data
        eval_point = torch.from_numpy(np.array([point])).float()
        #Calculate y_sim
        if CS == 1:
            #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
            y_sim = create_y_data(eval_point)
        else:
            y_sim = create_y_data(eval_point, true_model_coefficients, X_space, skip_param_types)
    else:
        X_val = None
        eval_point = torch.from_numpy(np.array([point])).float()
        if CS == 1:
            #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
            y_sim = create_sse_data(q, eval_point, Xexp, Yexp, obj = "obj")
        else:
            y_sim = create_sse_data(eval_point, Xexp, Yexp, true_model_coefficients, obj = "obj", skip_param_types = skip_param_types)
    
    GP_mean, GP_var = eval_GP_mean_std(theta_set_val, model, likelihood, X_val)

    #Define GP standard deviation   
    GP_stdev = np.sqrt(GP_var)  
    
    return GP_mean, GP_stdev, y_sim 