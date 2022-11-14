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
from sklearn.model_selection import LeaveOneOut

from bo_functions_generic import train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs
# from CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def LOO_Analysis(all_data, ep, Xexp, Yexp, true_model_coefficients, true_p, emulator, obj, skip_param_types = 0, set_lengthscale = None, train_iter = 300, noise_std = 0.1, verbose = False):
    ep_init = ep
    m = Xexp.shape[1]
#     print(m)
    loo = LeaveOneOut()
    loo.get_n_splits(all_data)
    
    index_list = []
    model_list = []
    sse_model_list = []
    #Loop over all test indecies & #Shuffle and split into training and testing data where 1 point is testing data
    for train_index, test_index in loo.split(all_data):
        index_list.append(test_index)
        data_train = all_data[train_index]
#         print(data_train[0:5])
        data_test = all_data[test_index]
#         print(data_test)
        #separate into y data and parameter data
        if m > 1:
            train_p = torch.tensor(data_train[:,1:-m+1]) #8 or 10 (emulator) parameters 
            test_p = torch.tensor(data_test[:,1:-m+1])
        else:
            train_p = torch.tensor(data_train[:,1:-m]) #8 or 10 (emulator) parameters 
            test_p = torch.tensor(data_test[:,1:-m])
#         print(test_p, test_p.shape)
#         print(train_p[:,-m:])
            
        train_y = torch.tensor(data_train[:,-1])
        test_y = torch.tensor(data_test[:,-1])
         
        #Set likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose)
        #Set exploration parameter (in thic case, 1)
        explore_bias = ep_init #Sets ep to the multiplicative scaler between 0.1 and 1
        #Evaluate GP on test set
        #Theta_set will be be only the test value
#         print(test_p)
        test_p_reshape = test_p.numpy()
#         print(test_p_reshape, test_p_reshape.shape)
        eval_components = LOO_eval_GP(test_p_reshape, Xexp, train_y, explore_bias, true_model_coefficients, model, likelihood, verbose, emulator, set_lengthscale, train_p = train_p, obj = obj, skip_param_types = skip_param_types, noise_std = noise_std)
#         print("Evaluated")
        if emulator == False:
            GP_mean,GP_var,GP_stdev = eval_components

        else:
            GP_mean,GP_var,GP_stdev, sse, sse_GP_var, sse_GP_stdev  = eval_components
#             print(sse.shape, sse)
            sse_model_list.append(sse)
            
        if test_index%50 == 0:
            print("Loop")
        model_list.append(GP_mean)
        #Plot GP_mean test vs train for X1 and X2 vs Muller Potential
        #Fix these plotters to be what I want
    index_list = np.array(index_list)
    model_list = np.array(model_list)
    sse_model_list = np.array(sse_model_list)
    if emulator == False:
        LOO_Plots_2_Input(index_list, model_list, GP_stdev, true_p)  
    else:
        if m > 1:
            Y_space = calc_y_exp(true_model_coefficients, all_data[:,-m:], noise_std = noise_std)
        else:
            calc_point = clean_1D_arrays(all_data[:,-m])
            Y_space = calc_y_exp(true_model_coefficients, calc_point, noise_std = noise_std)
            Y_space = Y_space.reshape((index_list.shape))
        ##NEED TO ADD WHAT THE SSE SHOULD BE GIVEN THE TEST POINT WHEN PLOTTING ERROR!!!
        LOO_Plots_3_Input(index_list, Y_space, model_list, GP_stdev, true_p)
        LOO_Plots_2_Input(index_list, sse_model_list, GP_stdev, true_p)
        #Calculate SSE (maybe)
        #Make residual plots (maybe)
    return

def LOO_eval_GP(theta_set, Xexp, train_y, explore_bias, true_model_coefficients, model, likelihood, verbose, emulator, set_lengthscale, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1):
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
        eval_components = LOO_eval_GP_basic_set(theta_set, train_y, model, likelihood, obj, explore_bias, verbose)
    else:
#         eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose)
        eval_components = LOO_eval_GP_emulator_set(theta_set, Xexp, true_model_coefficients, model, likelihood, explore_bias, verbose, train_p, obj, skip_param_types = skip_param_types,  noise_std = noise_std)
    
    return eval_components

def LOO_eval_GP_basic_set(theta_set, train_sse, model, likelihood, obj = "obj", explore_bias=0.0, verbose = False):
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
        sse: ndarray, the sse/ln(sse) of the GP model
        var: ndarray, the variance of the GP model
        stdev: ndarray, the standard deviation of the GP model
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"

#     print(theta_set.shape)
    if len(theta_set.shape) > 1:
        len_set, q = theta_set.shape[0], theta_set.shape[1]
    else:
        len_set, q = 1, theta_set.shape[0]
#     print(theta_set.shape)
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
#         print(eval_point, train_sse)
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_sse = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
#             if verbose == True:
#                 print("Point",eval_point)
#                 print("Model Mean",model_sse)
#                 print("Model Var", model_variance)
        #Save GP outputs
        
        if obj == "obj":
            sse[i] = model_sse
        else:
            sse[i] = np.exp(model_sse)
        var[i] = model_variance
        stdev[i] = np.sqrt(model_variance)  

        
#     print(sse)
    return sse, var, stdev #Prints just the value
    
def LOO_eval_GP_emulator_set(theta_set, Xexp, true_model_coefficients, model, likelihood, explore_bias = 0.0, verbose = False, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1):
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
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: bool, Determines whether output is verbose
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        (NOT USED NOW) optimize: bool, Determines whether scipy will be used to find the best point for 
    
    Returns
    -------
        SSE: ndarray, The SSE of the model 
        SSE_var_GP: ndarray, The varaince of the SSE pf the GP model
        SSE_stdev_GP: ndarray, The satndard deviation of the SSE of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    
    if len(theta_set.shape) > 1:
        len_set, q = theta_set.shape[0], theta_set.shape[1]
    else:
        len_set, q = 1, theta_set.shape[0]
#     print(theta_set.shape)
    m = Xexp.shape[1]
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    
    #Initialize values
    SSE_var_GP = 0
    SSE_stdev_GP = 0
    SSE = 0
    GP_mean_all = np.zeros((len_set))
    GP_var_all = np.zeros((len_set))
    
    ##Calculate Best Error
    # Loop over theta 1
    for i in range(len_set):        
    ##Calculate Values
        #Caclulate EI for each value n given the best error
        point = list(theta_set[i])
        eval_point = np.array([point])
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        #3-Input GP not well trained
        model_mean = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
        calc_exp_point = clean_1D_arrays(theta_set[:,-m:])
        
        Yexp = calc_y_exp(true_model_coefficients, calc_exp_point, noise_std)
#         print(Yexp)   

        #Compute SSE and SSE variance for that point
        SSE += (model_mean - Yexp)**2

        error_point = (model_mean - Yexp) #This SSE_variance CAN be negative
        SSE_var_GP += 2*error_point*model_variance #Error Propogation approach

        if SSE_var_GP > 0:
            SSE_stdev_GP += np.sqrt(SSE_var_GP)
        else:
            SSE_stdev_GP += np.sqrt(np.abs(SSE_var_GP))


        GP_mean_all[i] = model_mean
        GP_var_all[i] = model_variance
        GP_stdev = np.sqrt(GP_var_all)
#     print(GP_mean_all)
    return GP_mean_all, GP_var_all, GP_stdev, SSE, SSE_var_GP, SSE_stdev_GP

def LOO_Plots_2_Input(iter_space, GP_mean, GP_stdev, Theta):
    p = GP_mean.shape[0]
#     print(X1.shape, X2.shape, GP_mean.shape)
    # Compare the experiments to the true model
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
#     plt.scatter(iter_space,Y_space, label = "$y_{exp}$")
    label = "$log(SSE_{sim})$"
    plt.scatter(iter_space,np.log(GP_mean), label = label )
    
#     ax.fill_between(iter_space,
#     GP_mean - 1.96 * GP_stdev,
#     GP_mean + 1.96 * GP_stdev,
#     alpha=0.3 )
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    plt.xlabel("Index", fontsize=16, fontweight='bold')
    plt.ylabel(label, fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")
#     plt.grid(True)
    return plt.show()

def LOO_Plots_3_Input(iter_space, Y_space, GP_mean, GP_stdev, Theta):
    p = GP_mean.shape[0]
#     print(X1.shape, X2.shape, GP_mean.shape)
    # Compare the experiments to the true model
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
    plt.scatter(iter_space,Y_space, label = "$y_{exp}$")
    label = "Muller Potential"
    plt.scatter(iter_space,np.log(GP_mean), label = label )
    
#     ax.fill_between(iter_space,
#     GP_mean - 1.96 * GP_stdev,
#     GP_mean + 1.96 * GP_stdev,
#     alpha=0.3 )
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    plt.xlabel("Index", fontsize=16, fontweight='bold')
    plt.ylabel(label, fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")
#     plt.grid(True)
    return plt.show()