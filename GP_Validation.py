##https://towardsdatascience.com/model-validation-in-python-95e2f041f78c
##Load modules
import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc
from sklearn.model_selection import LeaveOneOut

from bo_functions_generic import train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs, eval_GP

import matplotlib as mpl

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def LOO_Analysis(all_data, ep, Xexp, Yexp, true_model_coefficients, emulator, sparse_grid, obj, skip_param_types = 0, set_lengthscale = None, train_iter = 300,verbose = False):
    ep_init = ep
    loo = LeaveOneOut()
    loo.get_n_splits(all_data)
    #Loop over all test indecies & #Shuffle and split into training and testing data where 1 point is testing data
    for train_index, test_index in loo.split(all_data):
        data_train = all_data[train_index]
        data_test = all_data[test_index]
        
        #separate into y data and parameter data
        train_p = data_train[:,0:-1]
        test_p = data_test[:,0:-1]
        
        train_y = data_train[:,-1]
        test_y = data_test[:,-1]
         
        #Set likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose)
        #Set exploration parameter (in thic case, 1)
        explore_bias = ep_init #Sets ep to the multiplicative scaler between 0.1 and 1
        #Evaluate GP
        # QUESTION: What do I actually want to evaluate? The test theta at all values of X1 and X2?
        #Create new functions to do the LOO GP analysis
        #Theta_set will be be only the correct value
        eval_components = LOO_eval_GP(theta_set, train_y, explore_bias, Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p = train_p, obj = obj, skip_param_types = skip_param_types)
        if emulator == False:
            GP_mean,GP_var,GP_stdev = eval_components
        else:
            GP_mean,GP_var,GP_stdev, sse, sse_GP_var, sse_GP_stdev  = eval_components
        
#         eval_components = eval_GP(test_p, train_p, explore_bias,Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj, skip_param_types = skip_param_types)
        
#         if verbose == True and emulator == False:
#             ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
#         elif emulator == True:
#             ei,sse,var,stdev,best_error,gp_mean_all, gp_var_all = eval_components
#         else:
#             ei,sse,var,stdev,best_error = eval_components
            
        #Plot GP_mean test vs train for X1 and X2 vs Muller Potential
        #Fix these plotters to be what I want
        if emulator == True:
            LOO_Plots_3_Input(model, likelihood, Xexp, noise_std, emulator, set_lengthscale, t, obj, sep_fact, verbose = verbose, runs = runs, DateTime = DateTime, test_p = test_p, LOO = LOO, LSO = LSO, save_figure = save_fig)
        else:
            LOO_Plots_2_Input(model, likelihood, Xexp, noise_std, emulator, set_lengthscale, t, obj, sep_fact, verbose = verbose, runs = runs, DateTime = DateTime, test_p = test_p, LOO = LOO, LSO = LSO, save_figure = save_fig)
            
        #Calculate SSE
        #Make residual plots

def LOO_eval_GP(theta_set, train_y, explore_bias, Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p = None, obj = "obj", skip_param_types = 0):
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
        train_p: tensor or ndarray, The training parameter space data
        obj: ob or LN_obj: Determines which objective function is used for the 2 input GP
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
    
    Returns:
    --------
        eval_components: ndarray, The componenets evaluate by the GP. ei, sse, var, stdev, f_best, (z_term, ei_term_1, ei_term_2, CDF, PDF)
    """
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    ##Set Hyperparameters to 1
#     print(skip_param_types)
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
        eval_components = LOO_eval_GP_basic_set(theta_set, train_y, model, likelihood, explore_bias, verbose)
    else:
#         eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose)
        eval_components = LOO_eval_GP_emulator_set(Xexp, Yexp, theta_set, true_model_coefficients, model, likelihood, sparse_grid, explore_bias, verbose, train_p, obj, skip_param_types = skip_param_types)
    
    return eval_components

def LOO_eval_GP_basic_set(theta_set, train_sse, model, likelihood, explore_bias=0.0, verbose = False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        train_sse: ndarray (1 x t), Training data for sse
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: True/False: Determines whether z and ei terms are printed
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
        sse: ndarray, the sse/ln(sse) of the GP model
        var: ndarray, the variance of the GP model
        stdev: ndarray, the standard deviation of the GP model
        f_best: ndarray, the best value so far
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Calculate and save best error
    #Negative sign because -max(-train_sse) = min(train_sse)
    best_error = -max(-train_sse).numpy() 
#     best_error = max(-train_sse).numpy()

#     print(theta_set.shape)
    if len(theta_set.shape) > 1:
        len_set, q = theta_set.shape[0], theta_set.shape[1]
    else:
        len_set, q = 1, theta_set.shape[0]
    
    #These will be redone
    #Initalize matricies to save GP outputs and calculations using GP outputs
    sse = np.zeros(len_set)
    var = np.zeros(len_set)
    stdev = np.zeros(len_set)
        
        
    #Create all iteration permulations - Takes a very long time for 8 dimensions
    #Theta = np.linspace(-2,2,10) (insert this instead of a a theta mesh (everything will be scaled from 0-1 in Muller problem)
    #df = pd.DataFrame(list(itertools.product(Theta, repeat=8)))
    #df2 = df.drop_duplicates()
    #theta_list = df2.to_numpy()
    
    for i in range(len_set):
        #Choose and evaluate point
        point = theta_set[i]
#         point = [theta_set[i]]
        eval_point = np.array([point])
#         print(eval_point)
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_sse = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
#             if verbose == True:
#                 print("Point",eval_point)
#                 print("Model Mean",model_sse)
#                 print("Model Var", model_variance)
        #Save GP outputs
        sse[i] = model_sse
        var[i] = model_variance
        stdev[i] = np.sqrt(model_variance)  

        

    return sse, var, stdev #Prints just the value
    
def LOO_eval_GP_emulator_set(Xexp, Yexp, theta_set, true_model_coefficients, model, likelihood, sparse_grid, explore_bias = 0.0, verbose = False, train_p = None, obj = "obj", skip_param_types = 0):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        Xexp: ndarray, "experimental" x values
        Yexp: ndarray, "experimental" y values
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        sparse_grid: True/False: Determines whether an assumption or sparse grid method is used
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: bool, Determines whether output is verbose
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        (NOT USED NOW) optimize: bool, Determines whether scipy will be used to find the best point for 
    
    Returns
    -------
        EI: ndarray, the expected improvement of the GP model
        SSE: ndarray, The SSE of the model 
        SSE_var_GP: ndarray, The varaince of the SSE pf the GP model
        SSE_stdev_GP: ndarray, The satndard deviation of the SSE of the GP model
        best_error: ndarray, The best_error of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(Xexp)==len(Yexp), "Experimental data must have same length"
    n = len(Xexp)
    len_set , q = theta_set.shape
    
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    
    #Initialize values
    SSE_var_GP = np.zeros(len_set)
    SSE_stdev_GP = np.zeros(len_set)
    SSE = np.zeros(len_set)
    GP_mean_all = np.zeros((len_set,n))
    GP_var_all = np.zeros((len_set,n))
    
    ##Calculate Best Error
    # Loop over theta 1
    for i in range(len_set):
        best_error = eval_GP_emulator_BE(Xexp, Yexp, train_p, true_model_coefficients, obj = "obj", skip_param_types = skip_param_types)
        GP_mean = np.zeros(n)
        GP_var = np.zeros(n)
        
        ##Calculate Values
        for k in range(n):
            #Caclulate EI for each value n given the best error
            point = list(theta_set[i])
#             point.append(float(Xexp[k]))
            x_point_data = list(Xexp[k]) #astype(np.float)
            point = point + x_point_data
#             print(point)
#             point.append(x_point_data) 
            point = np.array(point)  
            eval_point = np.array([point])
#             eval_point = np.array([point])[0]
#             print(eval_point)
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_mean = GP_Outputs[3].numpy()[0] #1xn
            model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
            
            GP_mean[k] = model_mean
            GP_var[k] = model_variance               

            #Compute SSE and SSE variance for that point
            SSE[i] += (model_mean - Yexp[k])**2

            error_point = (model_mean - Yexp[k]) #This SSE_variance CAN be negative
            SSE_var_GP[i] += 2*error_point*model_variance #Error Propogation approach

            if SSE_var_GP[i] > 0:
                SSE_stdev_GP[i] = np.sqrt(SSE_var_GP[i])
            else:
                SSE_stdev_GP[i] = np.sqrt(np.abs(SSE_var_GP[i]))


        GP_mean_all[i] = GP_mean
        GP_var_all[i] = GP_var
        GP_stdev = np.sqrt(GP_var)
    
    return GP_mean_all, GP_var_all, GP_stdev, SSE, SSE_var_GP, SSE_stdev_GP