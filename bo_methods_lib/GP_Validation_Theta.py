##https://towardsdatascience.com/model-validation-in-python-95e2f041f78c
##Load modules
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import torch
import os
import gpytorch
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
from sklearn.model_selection import LeaveOneGroupOut

from .bo_functions_generic import train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs
from .CS2_bo_plotters import save_csv, save_fig
from .normalize import normalize_x, normalize_p_data, normalize_p_bounds, normalize_p_set, normalize_p_true, normalize_constants, normalize_general
    
# from .CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from .CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def LOO_In_Theta_Analysis(all_data, Xexp, Yexp, true_model_coefficients, true_p, emulator, obj, Case_Study, skip_param_types = 0, set_lengthscale = None, train_iter = 300, noise_std = 0.1, verbose = False, DateTime = None, save_figure= True, plot_axis = None, normalize = False, bounds_p = None, bounds_x = None):  
    """
    Run GP Validation using a leave one out scheme
    
    Parameters:
    -----------
        all_data: ndarray, contains all data for GP
        Xexp: ndarray, The list of Xs that will be used to generate Y
        Yexp: ndarray, The experimental data for y (the true value)
        true_model_coefficients: ndarray, The array containing the true values of the problem (may be the same as true_p)
        true_p: ndarray, The array containing the true values of theta parameters to regress- flattened array
        emulator: bool, Determines if GP will model the function or the function error
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        Case_Study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        train_iter: int, number of training iterations to run for GP. Default is 300
        noise_std: float, int: The standard deviation of the noise. Default 0.1
        verbose: bool, Determines whether EI component terms are saved also determines activeness of print statement, Default = False
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        save_figure: bool, Determines whether figures will be saved. Default True
        plot_axis: None or list: Determines which axis to plot parity plot on (0 = Xexp axis (100 graphs), 1 = theta_j axis (5 graphs))
    
    Returns:
    --------
        None, prints/saves graphs and sse numbers 
        
    """   
    
    #Define constants for dimensions of x (m), number of exp data points (n), number of parameters to be regressed (q), and data length (t)
    n,m = Xexp.shape
    q = true_p.shape[0]
    t = len(all_data)
    
    #Create Groups for Data
    groups = np.zeros(len(all_data))
    for i in range(len(all_data)):
        if emulator == True:
            #First row is part of group 1
            if i == 0:
                groups[i] = 1
            #Next group happens after n Xexp values
            elif i%n == 0 and i > 0:
                groups[i] = groups[i-1] + 1
            #If not a new group, same as last group number
            else:
                groups[i] = groups[i-1]
        else: #If emulator is false, each line is its own group
            #First row is part of group 1
            if i == 0:
                groups[i] = 1
            #Next group happens after n Xexp values
            else:
                groups[i] = groups[i-1] + 1
            
    
    #Create empy lists to store index, GP model val, y_sim vals, sse's from emulator vals, SSE from emulator val, and sse from GP vals
    index_list = []
    y_model_tj_xk_list = []
    y_model_stdev_tj_xk_list = []
    sse_GP_tj_xk_list = []
    sse_GP_tj_xj_list = []
    sse_GP_stdev_tj_xk_list = []
    sse_GP_stdev_tj_xj_list = []
    y_sim_tj_xk_list = []
    y_sim_tj_xj_list = []
    sse_y_sim_tj_xk_list = []
    sse_y_sim_tj_xj_list = []
    
    #Normalize these values only once
    if normalize == True:
        norm = True
        bounds_x, scaler_x = normalize_x(bounds_x, None, norm)
        bounds_p, scaler_theta = normalize_p_bounds(bounds_p, norm)
        Xexp = normalize_x(Xexp, None, norm, scaler_x)[0]
        true_p = normalize_p_true(true_p, scaler_theta, norm)
        true_model_coefficients, scaler_C_before, scaler_C_after  = normalize_constants(true_model_coefficients, true_p, scaler_theta, skip_param_types, Case_Study, norm)
        norm_scalers = [scaler_x, scaler_theta, scaler_C_before, scaler_C_after]
    else:
        norm_scalers = None
    all_data = all_data.astype('float32')
    
    #Split all data into theta (p) and y data
    if m > 1:
        data_p = all_data[:,1:-m+1] #8 or 10 (emulator) parameters   
    else:
        data_p = all_data[:,1:-m] #8 or 10 (emulator) parameters 
    data_y = all_data[:,-1]
    
    #Define LOO splits
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups = groups)
    
    #Loop over all test indecies & #Shuffle and split into training and testing data where 5 points are testing data (each theta)
    #Intialize index list for plotting
    index_list = []
#     for train_index, test_index in loo.split(all_data):
    for train_index, test_index in logo.split(data_p, data_y, groups):
        #Separate training and testing data
        index_list.append(test_index)
        train_p, test_p = torch.tensor(np.array([data_p[i] for i in train_index])), torch.tensor(np.array([data_p[i] for i in test_index]))
        train_y, test_y = torch.tensor(np.array([data_y[i] for i in train_index])), torch.tensor(np.array([data_y[i] for i in test_index]))
        
#         print(train_p[0:10], test_p[0:10])
#         print(type(train_p), train_p.dtype)
#         print(type(train_y), train_y.dtype)
        
#         print(len(train_p))
#         print(len(train_y))
        
        #Normalize Data if norm = True
        if normalize == True:
            if Case_Study == 1 or emulator == True:
    #           #Emulator always set to true for normalization for CS1 because training data always includes x values
                train_p[:,-m:] = normalize_x(Xexp, train_p[:,-m:], norm, scaler_x)[0].float()
                train_p[:,0:-m] = normalize_p_data(train_p, m, True, norm, scaler_theta).float()
                test_p[:,-m:] = normalize_x(Xexp, test_p[:,-m:], norm, scaler_x)[0].float()
                test_p[:,0:-m] = normalize_p_data(test_p, m, True, norm, scaler_theta).float()
            else:
                train_p = normalize_p_data(train_p, m, emulator, norm, scaler_theta).float()
                test_p = normalize_p_data(test_p, m, emulator, norm, scaler_theta).float()
               
        #Theta_set will be be only the test value
        #Make theta_set a numpy array. Note: Theta_set is always 1xtrain_p.shape[1]
        theta_set = test_p
        
        #Set model and likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        
        #Train GP
#         print(train_p[0:5], theta_set, Xexp, true_model_coefficients)
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose)      
        
        eval_components = LOO_eval_GP(theta_set, Xexp, train_y, true_model_coefficients, model, likelihood, verbose, emulator, set_lengthscale, true_p, norm_scalers, Case_Study, train_p = train_p, obj = obj, skip_param_types = skip_param_types, noise_std = noise_std)
        
        #eval_components is the GP_SSE, and GP_SSE_StDev
        sse_GP_tj_xj,sse_GP_stdev_tj_xj = eval_components
                
        if emulator == False:
            sse_GP_tj_xj,sse_GP_stdev_tj_xj = eval_components
            #Append data to lists as appropriate
            sse_GP_tj_xj_list.append(sse_GP_tj_xj)
            sse_GP_stdev_tj_xj_list.append(sse_GP_stdev_tj_xj)
        
        else:            
            #Calculate the values using theta_j and Xexp_k
            y_model_tj_xk, y_model_stdev_tj_xk, y_sim_tj_xk, sse_GP_tj_xk,  sse_y_sim_tj_xk, sse_GP_stdev_tj_xk = LOO_eval_GP_emulator_tj_xk(theta_set, Xexp, Yexp,true_model_coefficients, model, likelihood, true_p, norm_scalers, verbose, skip_param_types, Case_Study)
            y_model_tj_xk_list.append(y_model_tj_xk)
            y_model_stdev_tj_xk_list.append(y_model_stdev_tj_xk)
            sse_GP_tj_xk_list.append(sse_GP_tj_xk)
            y_sim_tj_xk_list.append(y_sim_tj_xk)
            sse_y_sim_tj_xk_list.append(sse_y_sim_tj_xk)
            sse_GP_stdev_tj_xk_list.append(sse_GP_stdev_tj_xk)
                
            #Append data to lists as appropriate
            sse_GP_tj_xj_list.append(sse_GP_tj_xj)
            sse_GP_stdev_tj_xj_list.append(sse_GP_stdev_tj_xj)
    
    #Turn lists into arrays
    index_list = np.array(index_list)
    y_model_tj_xk_list = np.array(y_model_tj_xk_list)
    y_model_stdev_tj_xk_list = np.array(y_model_stdev_tj_xk_list)
    sse_GP_tj_xk_list = np.array(sse_GP_tj_xk_list)
    sse_GP_tj_xj_list = np.array(sse_GP_tj_xj_list)
    sse_GP_stdev_tj_xj_list = np.array(sse_GP_stdev_tj_xj_list)
    sse_GP_stdev_tj_xk_list = np.array(sse_GP_stdev_tj_xk_list)
    y_sim_tj_xk_list = np.array(y_sim_tj_xk_list)
    y_sim_tj_xj_list = data_y
    sse_y_sim_tj_xj_list =np.zeros(int(len(y_sim_tj_xj_list)/n))
    sse_y_sim_tj_xk_list = np.array(sse_y_sim_tj_xk_list)
    
    #Plot model vs sim sse
#     print(len(index_list), len(sse_GP_tj_xj_list), len(sse_y_sim_tj_xj_list), len(sse_GP_stdev_tj_xj_list))      
        
    if emulator == False:
        #Depending on obj, ensure sse is sse and not log(sse)
        if obj == "LN_obj":
            sse_y_sim_tj_xj_list = np.exp(all_data[:,-1])
        else:
            sse_y_sim_tj_xj_list = all_data[:,-1]
        sse_y_sim_tj_xj_list = np.array(sse_y_sim_tj_xj_list)
        
        #Plot model vs sim sse
        LOO_Plots_2_Input(index_list, sse_GP_tj_xj_list, sse_y_sim_tj_xj_list, sse_GP_stdev_tj_xj_list, Case_Study, DateTime, obj, set_lengthscale, save_figure, normalize = normalize)
        
        LOO_parity_plot_emul(sse_GP_tj_xj_list, sse_y_sim_tj_xj_list, sse_GP_stdev_tj_xj_list, Case_Study, DateTime, t, emulator, obj, set_lengthscale, save_figure, plot_axis = None, plot_num = None, normalize = normalize)
        
    else:        
        #Plot log(SSE) from GP(theta_j,Xexp) and y_sim(theta_j,Xexp)
        LOO_Plots_2_Input(index_list, sse_GP_tj_xk_list, sse_y_sim_tj_xk_list, sse_GP_stdev_tj_xk_list, Case_Study, DateTime, obj, set_lengthscale, save_figure, emulator, normalize = normalize)
        
        #Plot Parity plot for log(SSE)
        LOO_parity_plot_emul(sse_GP_tj_xk_list, sse_y_sim_tj_xk_list, sse_GP_stdev_tj_xk_list, Case_Study, DateTime, t, emulator, obj, set_lengthscale, save_figure, plot_axis = None, plot_num = None, normalize = normalize)
        
        #Loop over each axis
        for axis in plot_axis:
            #For each row or column of GP(theta_j, Xexp_k), make a parity plot between GP(theta_j, Xexp_k) and y_sim(theta_j, Xexp_k)
#             print((y_model_tj_xk_list.shape, y_model_tj_xk_list.shape[axis]))
            for i in range(y_model_tj_xk_list.shape[axis]):   #Plot axis either 0 or 1
                #If plot_axis == 0, plot on axis Xexp and have j graphs
                if axis == 0: 
                    test_data_to_plot = all_data[int(i*n)] 
                    title_arg = test_data_to_plot[1:-m-1]
                                      
                    LOO_parity_plot_emul(y_model_tj_xk_list[i,:], y_sim_tj_xk_list[i,:], y_model_stdev_tj_xk_list[i,:], Case_Study, DateTime, t, emulator, obj, set_lengthscale, save_figure, axis, plot_num = i, title_arg = title_arg, normalize = normalize)
                #If plot_axis == 1, plot on axis theta_j and have n graphs
                else:  
                    title_arg = Xexp[i]
                    #Un-normalize if necessary
                    if normalize == True:
                        norm = False
                        title_arg = normalize_x(clean_1D_arrays(title_arg, param_clean = True), None, norm, scaler_x)[0]
                    LOO_parity_plot_emul(y_model_tj_xk_list[:,i], y_sim_tj_xk_list[:,i], y_model_stdev_tj_xk_list[:,i], Case_Study, DateTime, t, emulator, obj, set_lengthscale, save_figure, axis, plot_num = i, title_arg = title_arg, normalize = normalize)
        
        #Print and save total sse value to CSV
        fxn = "LOO_Plots_3_Input"
        #Note flatten y_sim_tj_xj_list to prevent an error in the calculation and ensure y_sim_tj_xj_list.shape = y_model_tj_xj_list.shape
        SSE_Total =  sum( sse_GP_tj_xj_list) 
        sse_tot_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/sse_tot", normalize = normalize)
        print("SSE Total = ",'{:.4e}'.format(SSE_Total) )
    return

def LOO_eval_GP(theta_set, Xexp, train_y, true_model_coefficients, model, likelihood, verbose, emulator, set_lengthscale, true_p, norm_scalers = None, CS = 1, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1):
    """
    Evaluates GP
    
    Parameters:
    -----------
        theta_set: ndarray (len_set x dim_param), array of Theta values 
        train_y: tensor or ndarray, The training y data
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
        emulator: True/False: Determiens whether GP is an emulator of the function
        sparse_grd: True/False: Determines whether an assumption or sparse grid is used
        set_lengthscale: float/None: Determines whether Hyperparameter values will be set
        true_p
        norm_scalers
        CS
        train_p: tensor or ndarray, The training parameter space data
        obj: ob or LN_obj: Determines which objective function is used for the 2 input GP. Default = "obj"
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        noise_std: float, int: The standard deviation of the noise. Default 0.1
        
    
    Returns:
    --------
        eval_components: ndarray, The componenets evaluate by the GP. ei, sse, var, stdev, f_best, (z_term, ei_term_1, ei_term_2, CDF, PDF)
    """
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Ensure train_y is a tensor
    if isinstance(train_y, np.ndarray)==True:
        train_y = torch.tensor(train_y) #1xn
    
    #Set hyperparameters
    if set_lengthscale is not None:
        if verbose == True:
            print("Lengthscale Set To: " + set_lengthscale)
        outputscale = torch.tensor([1])
        lengthscale = torch.tensor([set_lengthscale])
        noise = torch.tensor([0.1])

        model.likelihood.noise = noise
        model.covar_module.base_kernel.lengthscale =lengthscale
        model.covar_module.outputscale = outputscale
    
    model.eval()
    #Puts likelihood in evaluation mode
    likelihood.eval()
    
    #Evaluate GP based on error emulator or property emulator
    if emulator == False:
        eval_components = LOO_eval_GP_basic_set(theta_set, train_y, model, likelihood, obj, verbose)
    else:
#         eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose)
        eval_components = LOO_eval_GP_emulator_set(theta_set, Xexp, true_model_coefficients, model, likelihood, true_p, norm_scalers, CS, verbose, train_p, obj, skip_param_types = skip_param_types,  noise_std = noise_std)
    
    return eval_components

def LOO_eval_GP_basic_set(theta_set, train_sse, model, likelihood, obj = "obj", verbose = False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        train_sse: ndarray (1 x t), Training data for sse
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated. Default "obj"
        verbose: True/False: Determines whether z and ei terms are printed
    
    Returns
    -------
        sse: ndarray, the sse/ln(sse) of the GP model
        stdev: ndarray, the standard deviation of the GP model
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"

    #Define constants for number of parameters to be regressed (q), and theta_set length (len_set)
    if len(theta_set.shape) > 1:
        len_set, q = theta_set.shape[0], theta_set.shape[1]
    else:
        len_set, q = 1, theta_set.shape[0]
    #Initalize matricies to save GP outputs and calculations using GP outputs
    sse = np.zeros(len_set)
    var = np.zeros(len_set)
    stdev = np.zeros(len_set)
        
    #Choose and evaluate point
    point = list(theta_set[0])
    eval_point = torch.tensor(np.array([point]))
#     print(eval_point[0:1].dtype)

    #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
    GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
    #Save GP outputs
    model_sse = GP_Outputs[3].numpy()[0] #1xn
    model_variance= GP_Outputs[1].detach().numpy()[0] #1xn

    #Ensures sse is saved instead of ln(sse)
    if obj == "obj":
        sse = model_sse
    else:
        sse = np.exp(model_sse)
    var = model_variance
    stdev = np.sqrt(model_variance)  

    return sse, stdev 

def LOO_eval_GP_emulator_set(theta_set, Xexp, true_model_coefficients, model, likelihood, p_true, norm_scalers, CS = 1, verbose = False, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1):
    """ 
    Calculates the expected improvement of the emulator approach
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        Xexp: ndarray, "experimental" x values
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: bool, Determines whether output is verbose. Default False
        train_p: tensor or ndarray, The training parameter space data
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated. Default "obj"
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        noise_std: float, int: The standard deviation of the noise. Default 0.1
    
    Returns
    -------
        GP_mean_all: ndarray, Array of GP mean predictions
        GP_stdev_all: ndarray, Array of GP standard deviations
        SSE: ndarray, The SSE of the model 
        SSE_var_GP: ndarray, The varaince of the SSE pf the GP model
        SSE_stdev_GP: ndarray, The satndard deviation of the SSE of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    
    #Define length (len_set) and dimsenionality (q) of theta_set and dimensionality of Xexp (m)
    if len(theta_set.shape) > 1:
        len_set, q = theta_set.shape[0], theta_set.shape[1]
    else:
        len_set, q = 1, theta_set.shape[0]

    m = Xexp.shape[1]
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    
    #Initialize values
    
    SSE_var_GP = 0
    SSE_stdev_GP = 0
    SSE = 0
         
    ##Calculate Values
    #Loop over testing data:
    for i in range(len(theta_set)):
        #Caclulate GP vals for each value given theta_j and x_j
        point = list(theta_set[i])
        eval_point = torch.tensor(np.array([point])).float()
    #     print(eval_point)
        #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])

        #Save GP Values
        model_mean = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
#         print(model_mean)
        #Calculate corresponding experimental data from theta_set value
        if str(norm_scalers) != "None":
            scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
            Xexp_unscl = normalize_x(Xexp, np.array(theta_set[:,-m:]), False, scaler_x)[0]
            true_model_coefficients_unscl = normalize_constants(true_model_coefficients, p_true, scaler_theta, skip_param_types, CS, False, scaler_C_before, scaler_C_after)[0]
        else:
            Xexp_unscl = np.array(theta_set[:,-m:])
            true_model_coefficients_unscl = true_model_coefficients

        calc_exp_point = clean_1D_arrays(Xexp_unscl)
    #     print(calc_exp_point)
    #     print(true_model_coefficients_unscl)

        ##Compute SSE and SSE variance for that point
        #Copute Yexp
        Yexp = calc_y_exp(true_model_coefficients_unscl, calc_exp_point, noise_std) 

        SSE += (np.array(model_mean) - Yexp[i])**2

        error_point = (model_mean - Yexp[i]) #This SSE_variance CAN be negative
        SSE_var_GP += 2*error_point*model_variance #Error Propogation approach

        #Ensure positive standard deviations are saved for plotting purposes
#         print(SSE_var_GP, SSE_var_GP.shape)
        if SSE_var_GP > 0:
            SSE_stdev_GP += np.sqrt(SSE_var_GP)
        else:
            SSE_stdev_GP += np.sqrt(np.abs(SSE_var_GP))

        #Save values for each value in theta_set (in this case only 1 value)
#         GP_mean_all[i] = model_mean
#         GP_var_all[i] = model_variance
        
#     GP_stdev_all = np.sqrt(GP_var_all)
#     print(GP_mean_all, GP_var_all, GP_stdev_all)

    return SSE, SSE_stdev_GP
    
def LOO_eval_GP_emulator_tj_xk(theta_set, Xexp, Yexp,true_model_coefficients, model, likelihood, p_true, norm_scalers = None, verbose=False, skip_param_types=0, CS= 1):
    """ 
    Calculates the expected improvement of the emulator approach
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        Xexp: ndarray, "experimental" x values
        Yexp: ndarray, experimental y values
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        p_true: ndarray, true values of coefficients
        verbose: bool, Determines whether output is verbose. Default False
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
    
    Returns
    -------
        GP_mean: ndaarray, Array of GP mean predictions at Xexp and theta_set
        GP_stdev: ndarray, Array of GP variances related to GP means at Xexp and theta_set
        y_sim: ndarray, simulated values at Xexp and theta_set
        SSE_model: ndarray, The SSE of the model at Xexp and theta_set
        SSE_sim: ndarray, The SSE of the simulated data at Xexp and theta_set
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"

    #Define dimensionality of X
    m = Xexp.shape[1]
    n = Xexp.shape[0]
    
    #Set theta_set to only be parameter values instead of theta_j, x_j
    theta_set_params = theta_set[:, 0:-m]
    
    #Define the length of theta_set and the number of parameters that will be regressed (q)
    if len(theta_set_params.shape) > 1:
        len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    else:
        len_set, q = 1, theta_set_params.shape[0]
    
    #Initialize values for saving data
    GP_mean = np.zeros((len(theta_set_params)))
    GP_var = np.zeros((len(theta_set_params)))
    y_sim = np.zeros((len(theta_set_params)))
    
    #Loop over experimental data 
    for k in range(len(theta_set_params)):
        ##Calculate Values
        #Caclulate GP vals for each value given theta_j and x_j
        point = list(theta_set[k])
        eval_point = torch.tensor(np.array([point])).float()
        #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])

        model_mean = GP_Outputs[3].numpy()[0] #1xn
        GP_mean[k] = model_mean
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
        GP_var[k] = model_variance
        
        #Unnormalize data
        if str(norm_scalers) != "None":
#             eval_point_unscl = eval_point.copy()
            eval_point_unscl = np.array(eval_point.clone())
            scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
            Xexp_unscl = normalize_x(Xexp, None, norm = False, scaler = scaler_x)[0]
            eval_point_unscl[:,0:-m] = normalize_p_data(eval_point_unscl[:,0:-m], m, True, norm = False, scaler = scaler_theta)
            eval_point_unscl[:,-m:] = normalize_x(Xexp, eval_point_unscl[:,-m:], norm = False, scaler = scaler_x)[0]  
            true_model_coefficients_unscl = normalize_constants(true_model_coefficients, p_true, scaler_theta, skip_param_types, CS, False, scaler_C_before, scaler_C_after)[0]
        else:
            eval_point_unscl = np.array(eval_point)
            true_model_coefficients_unscl = true_model_coefficients
            Xexp_unscl = Xexp
        
        #Calculate y_sim & sse_sim
        if CS == 1:
            #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
            y_sim[k] = create_y_data(eval_point_unscl)
        else:
            y_sim[k] = create_y_data(eval_point_unscl, true_model_coefficients_unscl, Xexp_unscl, skip_param_types)

    #Compute GP SSE and SSE_sim for that point
    SSE_model = np.sum((np.array(GP_mean) - Yexp)**2)
    error_point = (np.array(GP_mean) - Yexp) #This SSE_variance CAN be negative
    SSE_var_GP = sum(2*error_point*model_variance) #Error Propogation approach
    
    SSE_sim = np.sum((y_sim - Yexp)**2)
    
    #Ensure positive standard deviations are saved for plotting purposes
    if SSE_var_GP > 0:
        SSE_model_stdev = np.sqrt(SSE_var_GP)
    else:
        SSE_model_stdev = np.sqrt(np.abs(SSE_var_GP))
        
    GP_stdev = np.sqrt(GP_var)  
    
    return GP_mean, GP_stdev, y_sim, SSE_model, SSE_sim, SSE_model_stdev

def LOO_Plots_2_Input(iter_space, GP_mean, sse_sim, GP_stdev, Case_Study, DateTime, obj, set_lengthscale = None, save_figure= True, emulator = False, normalize = False, save_csvs = True):
    """ 
    Creates plots of sse_sim and sse_model vs test space point
    Parameters
    ----------
        iter_space: ndarray, a linspace of the number of testing points evaluated
        GP_mean: ndarray, Array of GP mean predictions
        sse_sim: ndarray, Array of sse_sim values
        GP_stdev: ndarray, Array of GP standard deviations
        Case_Study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, int, Number of initial training points to use
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: bool, Determines whether figures will be saved. Default True
        emulator: bool, whether or not emulator SSEs are being plotted (used for savinf CSV data and figures)
    
    Returns
    -------
        None
    """
    #If emulator, change indecies of GP_mean to match actual indecies given that theta values are repeated n times
    if emulator == True:
        n = len(iter_space)/len(GP_mean)
        iter_space = np.linspace(0,len(GP_mean), len(GP_mean))
        iter_space = iter_space*n
        t = int(len(iter_space)*n)
    else:
        t = len(iter_space)
    #Flatten GP mean to ensure smooth plotting
    GP_mean = GP_mean.flatten()
    
    #Define function, length of GP mean predictions (p), and number of tests (t)   
    p = GP_mean.shape[0]
    fxn = "LOO_Plots_2_Input"
    

    # Compare the GP mean to the true model (simulated model)
    plt.figure(figsize = (6.4,4))
#     plt.scatter(iter_space,Y_space, label = "$y_{exp}$")
#     label = "$log(SSE_{model})$"

    #Only plot error bars if a standard deviation is given
#     print(GP_mean.shape, sse_sim.shape)
    GP_stdev = GP_stdev.flatten()
    GP_upper = np.log(GP_mean + GP_stdev)
    GP_lower = np.log(GP_mean - GP_stdev)
    y_err = np.array([GP_lower, GP_upper])
#         yerr=1.96*GP_stdev
#     print(len(iter_space), len(np.log(GP_mean)))
    plt.errorbar(iter_space,np.log(GP_mean), fmt="o", yerr=y_err, label = r'$log(e(\theta))_{model}$', ms=10, zorder=1, mec = "green", mew = 1 )
    plt.scatter(iter_space,np.log(sse_sim), label = r'$log(e(\theta))_{sim}$' , s=50, color = "orange", zorder=2, marker = "*")
    
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left", fontsize=16)
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    plt.xlabel("Index", fontsize=16, fontweight='bold')
    plt.ylabel("Natural Log Error", fontsize=16, fontweight='bold')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")

    #Save CSVs
    if save_csvs == True:
        iter_space_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/iter_space", normalize = normalize)
        GP_mean_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/log_sse_model", normalize = normalize)
        sse_sim_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/log_sse_sim", normalize = normalize)
        csv_item_list = [iter_space, np.log(GP_mean), np.log(sse_sim)]
        make_csv_list = [iter_space_path, GP_mean_path, sse_sim_path]

        for i in range(len(make_csv_list)):
            save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")
    
    #Save figure or show and close figure
    if save_figure == True:
        path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = True, normalize = normalize)
        save_fig(path, ext='png', close=True, verbose=False) 
    else:
        plt.show()
        plt.close()
        
    return

def LOO_Plots_3_Input(iter_space, GP_mean, y_sim, GP_stdev, Case_Study, DateTime, set_lengthscale = None, save_figure = True, normalize = False, save_csvs = True):
    """ 
    Creates plots of y_sim and y_model vs test space point
    Parameters
    ----------
        iter_space: ndarray, a linspace of the number of testing points evaluated
        GP_mean: ndarray, Array of GP mean predictions
        y_sim: ndarray, Array of y_sim values
        GP_stdev: ndarray, Array of GP standard deviations
        Case_Study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: bool, Determines whether figures will be saved. Default True
    
    Returns
    -------
        None
    """
    #Flatten values to ensure no plotting errors from shape (len(vals),1)
    GP_mean = GP_mean.flatten()
    GP_stdev = GP_stdev.flatten()
    
    #Define function (fxn), length of GP mean predictions (p), and number of tests (t), and obj ("obj") 
    fxn = "LOO_Plots_3_Input"
    emulator = True
    p = GP_mean.shape[0]
    t = len(iter_space)
    obj = "obj"

    # Compare the GP Mean to the true model (simulated model ysim)
    plt.figure(figsize = (6.4,4))
    plt.errorbar(iter_space,GP_mean, fmt = "o", yerr = 1.96*GP_stdev, label = "$y_{model}$", ms=10, zorder =1, mec = "green", mew = 1 )
    plt.scatter(iter_space,y_sim, label = "$y_{sim}$" , s=50, color = "orange", zorder=2, marker = "*")
    
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left", fontsize=16)
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    plt.xlabel("Index", fontsize=16, fontweight='bold')
    plt.ylabel("y value", fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")

    #Save CSVs
    if save_csvs == True:
        iter_space_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/iter_space", normalize = normalize)
        GP_mean_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/y_model", normalize = normalize)
        y_sim_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/y_sim", normalize = normalize)
        csv_item_list = [iter_space, GP_mean, y_sim]
        make_csv_list = [iter_space_path, GP_mean_path, y_sim_path]
    
    for i in range(len(make_csv_list)):
        save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")
    
    #Save figure or show and close figure
    if save_figure == True:
        path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = True, normalize= normalize)
        save_fig(path, ext='png', close=True, verbose=False) 
    else:
        plt.show()
        plt.close()
        
    return

def LOO_parity_plot_emul(GP_mean, y_sim, GP_stdev, Case_Study, DateTime, t, emulator, obj, set_lengthscale = None, save_figure = True, plot_axis = 0, plot_num = 0, title_arg = "None", save_csvs = True, normalize = False):
    """ 
    Creates parity plots of y_sim and y_model along axis for theta_j or Xexp
    Parameters
    ----------
        GP_mean: ndarray, Array of GP mean predictions
        y_sim: ndarray, Array of y_sim values
        GP_stdev: ndarray, Array of GP standard deviations
        Case_Study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, int, Number of initial training points to use
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: bool, Determines whether figures will be saved. Default True
        plot_axis: None or list: Determines which axis to plot parity plot on (0 = Xexp axis (100 graphs), 1 = theta_j axis (5 graphs))
        plot_num: None or int, The number of the parity plot w.r.t Xexp or thet_j indecies
    
    Returns
    -------
        None
    """
    #Flatten values to ensure no plotting errors from shape (len(vals),1)
    y_sim = y_sim.flatten()
    GP_mean = GP_mean.flatten()
    GP_stdev = GP_stdev.flatten()
    
    #Define function (fxn), length of GP mean predictions (p), and number of tests (t), and obj ("obj")
    fxn = "LOO_parity_plot_emul"
    p = GP_mean.shape[0]
    
    #Create figure
    plt.figure(figsize = (6.4,4))
    # Compare the GP Mean to the true model (simulated model ysim)
    #Plot y_sim vs y_GP for axis plots or plot log(sse_sim) vs log(sse_GP)
    if plot_axis != None:
        y_lab = "$y_{sim}$"
        plt.errorbar(y_sim,GP_mean, yerr=1.96*GP_stdev, fmt = "o", label = "$y_{model}$", ms=5, mec = "green", mew = 1, zorder = 1 )
        plt.plot(y_sim, y_sim, label = y_lab , zorder=2, color = "black")
    else:
        y_lab = r'$log(e(\theta))_{sim}$'
        GP_upper = np.log(GP_mean + GP_stdev)
        GP_lower = np.log(GP_mean - GP_stdev)
        y_err = np.array([GP_lower, GP_upper])
    #         yerr=1.96*GP_stdev
        plt.errorbar(np.log(y_sim), np.log(GP_mean), fmt="o", yerr=y_err, label = r'$log(e(\theta))_{model}$', ms=10, zorder=1, mec = "green", mew = 1)
        plt.plot(np.log(y_sim), np.log(y_sim), label = y_lab , zorder=2, color = "black")

    #Set plot details        
#     plt.legend(loc = "best")
    if plot_axis == 1:
        plt.title("Xexp = " + str(np.round(title_arg,2)), fontsize=16, fontweight='bold')
    elif plot_axis == 0:
        plt.title(r'$\theta_{j}$' + "=" + str(np.round(title_arg,2)), fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left", fontsize=16)
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    if plot_axis != None:
        plt.xlabel(r'$\mathbf{y_{sim}}$', fontsize=16, fontweight='bold')
        plt.ylabel(r'$\mathbf{y_{model}}$', fontsize=16, fontweight='bold')
    else:
        plt.xlabel("Simulated Natural Log Error", fontsize=16, fontweight='bold')
        plt.ylabel("Model Natural Log Error", fontsize=16, fontweight='bold')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)

    #Save CSVs
    if save_csvs == True:
        if plot_axis == None:
            csv_ends = ["/y_model", "/y_sim", "/y_stdev"]
        else:
            csv_ends = ["/sse_model", "/sse_sim", "/sse_gp_stdev"]
        GP_mean_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, plot_axis = plot_axis, plot_num = plot_num, csv_end = csv_ends[0], normalize = normalize)
        y_sim_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, plot_axis = plot_axis, plot_num = plot_num, csv_end = csv_ends[1], normalize = normalize )
        y_stdev_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, plot_axis = plot_axis, plot_num = plot_num, csv_end = csv_ends[2], normalize = normalize )
        csv_item_list = [GP_mean, y_sim, GP_stdev]
        make_csv_list = [GP_mean_path, y_sim_path, y_stdev_path]

        for i in range(len(make_csv_list)):
            save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")

    #Save figure or show and close figure
    if save_figure == True:
        path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = True, plot_axis =plot_axis, plot_num= plot_num, normalize = normalize )
        save_fig(path, ext='png', close=True, verbose=False) 
    else:
        plt.show()
        plt.close()
        
    return

def path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime = None, is_figure = True, csv_end = None, plot_axis = None, plot_num = None, normalize = False):
    """
    names a path
    
    Parameters
    ----------
        emulator: True/False, Determines if GP will model the function or the function error
        fxn: str, The name of the function whose file path name will be created
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        Case_Study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        is_figure: bool, used for saving CSVs as part of this function and for calling the data from a CSV to make a plot
        csv_end: str, the name of the csv file
        plot_axis: None or list: Determines which axis to plot parity plot on (0 = Xexp axis (100 graphs), 1 = theta_j axis (5 graphs))
        plot_num: None or int, The number of the parity plot w.r.t Xexp or thet_j indecies
        
    Returns:
    --------
        path: str, The path to which the file is saved
    
    """

    obj_str = "/"+str(obj)
    len_scl = "/len_scl_varies"
    org_TP_str = "/TP_"+ str(t)
    CS = "/CS_" + str(Case_Study) 
    
    if plot_axis == 1 or plot_axis == 0:
        parity_end = "/axis_val_" +str(plot_axis) + "/plot_num_" + str(plot_num).zfill(len(str(t)))
    else:
        parity_end = "/sse_parity_plot"

        
    if emulator == False:
        Emulator = "/GP_Error_Emulator"
        method = ""
    else:
        Emulator = "/GP_Emulator"
            
    fxn_dict = {"LOO_Plots_2_Input":"/SSE_gp_val" , "LOO_Plots_3_Input":"/y_gp_val", "LOO_parity_plot_emul":"/parity_plots"+parity_end}
    plot = fxn_dict[fxn]        
      
    if DateTime is not None:
#         path_org = "../"+DateTime #Will send to the Datetime folder outside of CS1
        path_org = DateTime #Will send to the Datetime folder outside of CS1
    else:
        path_org = "Test_Figs_GP_Val"
        
#         path_org = "Test_Figs"+"/Sep_Analysis2"+"/Figures"
    if normalize == True:
        path_org = path_org + "/Norm_Data"
    
    if is_figure == True:
        path_org = path_org + "/Figures"
    else:
        path_org = path_org + "/CSV_Data"
        
    path_end = CS + Emulator + org_TP_str + obj_str + len_scl + plot     

    path = path_org + "/GP_Validation_Figs" + path_end 
        
    if csv_end is not None:
        path = path + csv_end
   
    return path