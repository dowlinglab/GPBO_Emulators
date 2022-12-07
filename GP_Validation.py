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
from CS2_bo_plotters import save_csv, save_fig
    
from CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
# from CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def LOO_Analysis(all_data, Xexp, Yexp, true_model_coefficients, true_p, emulator, obj, Case_Study, skip_param_types = 0, set_lengthscale = None, train_iter = 300, noise_std = 0.1, verbose = False, DateTime = None, save_figure= True):  
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
        noise_std: float, int: The standard deviation of the noise. Default 0,1
        verbose: bool, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        save_figure: bool, Determines whether figures will be saved. Default True
    
    Returns:
    --------
        None, prints/saves graphs and sse numbers 
        
    """
    
    #Define constants for dimensions of x (m), number of parameters to be regressed (q), and data length (t)
    m = Xexp.shape[1]
    q = true_p.shape[0]
    t = len(all_data)
#     print(m)

    #Define LOO splits
    loo = LeaveOneOut()
    loo.get_n_splits(all_data)
    
    #Create empy lists to store index, GP model val, y_sim vals, sse's from emulator vals, SSE from emulator val, and sse from GP vals
    index_list = []
    model_list = []
    y_sim_list = []
    y_sim_sse_list = []
    GP_SSE_model_list = []
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
            
        train_y = torch.tensor(data_train[:,-1])
        test_y = torch.tensor(data_test[:,-1])
         
        #Set likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose)
        #Theta_set will be be only the test value
        #Make theta_set a numpy array
        test_p_reshape = test_p.numpy()
#         print(test_p_reshape, test_p_reshape.shape)
        #Evaluate GP on test set
        eval_components = LOO_eval_GP(test_p_reshape, Xexp, train_y, true_model_coefficients, model, likelihood, verbose, emulator, set_lengthscale, train_p = train_p, obj = obj, skip_param_types = skip_param_types, noise_std = noise_std)
#         print("Evaluated")
        #If emulator is false, eval_components is the GP mean, StDev, and variance
        if emulator == False:
            GP_mean,GP_var,GP_stdev = eval_components
        #If emulator is true, eval_components is the GP mean, StDev, and variance, and the corresponding sse value, variance, and stdev
        else:
            GP_mean,GP_var,GP_stdev, sse, sse_GP_var, sse_GP_stdev  = eval_components #sse here is w/ theta_j x_j
            #Calculate the SSE value from Y_sim and GP_SSE using theta_j and Xexp 
            GP_SSE, Y_sim_SSE = LOO_eval_GP_emulator_sse(test_p_reshape, Xexp, Yexp,true_model_coefficients, model, likelihood, verbose, skip_param_types, Case_Study)
            
            #Append data to lists as appropriate
            GP_SSE_model_list.append(GP_SSE)
            y_sim_sse_list.append(Y_sim_SSE)
            
            sse_model_list.append(sse)
            y_sim_list.append(data_test[:,-1])
            
#         if test_index%50 ==0:
#             print("Loop")
        model_list.append(GP_mean)
    
    #Turn lists into arrays
    index_list = np.array(index_list)
    model_list = np.array(model_list)
    sse_model_list = np.array(sse_model_list)
    
    if emulator == False:
        #Depending on obj, ensure sse is sse and not log(sse)
        if obj == "LN_obj":
            sse_sim = np.exp(all_data[:,-1])
        else:
            sse_sim = all_data[:,-1]
        #Plot model vs sim sse
        LOO_Plots_2_Input(index_list, model_list, sse_sim, GP_stdev, true_p, Case_Study, DateTime, obj, set_lengthscale, save_figure)
        
    else:
        #turn lists into arrays
        y_sim_list = np.array(y_sim_list)
        y_sim_sse_list = np.array(y_sim_sse_list)
        GP_SSE_model_list = np.array(GP_SSE_model_list)
        
        #Plot GP vs y_sim
        LOO_Plots_3_Input(index_list, model_list, all_data[:,-1], GP_stdev, true_p, Case_Study, DateTime, set_lengthscale, save_figure)
        #Plot log(SSE) from GP(theta_j,Xexp) and y_sim(theta_j,Xexp)
        LOO_Plots_2_Input(index_list, GP_SSE_model_list, y_sim_sse_list, None, true_p, Case_Study, DateTime, obj, set_lengthscale, save_figure, emulator)
        
        #Print and save total sse value to CSV
        fxn = "LOO_Plots_3_Input"
        SSE_Total =  sum( (y_sim_list - model_list)**2 ) 
        sse_tot_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/sse_tot")
#         print(sse_tot_path)
        print("SSE Total = ",'{:.4e}'.format(SSE_Total) )
    return

def LOO_eval_GP(theta_set, Xexp, train_y, true_model_coefficients, model, likelihood, verbose, emulator, set_lengthscale, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1):
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
        noise_std: float, int: The standard deviation of the noise. Default 0,1
        
    
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
        eval_components = LOO_eval_GP_basic_set(theta_set, train_y, model, likelihood, obj, verbose)
    else:
#         eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose)
        eval_components = LOO_eval_GP_emulator_set(theta_set, Xexp, true_model_coefficients, model, likelihood, verbose, train_p, obj, skip_param_types = skip_param_types,  noise_std = noise_std)
    
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
        
    
    for i in range(len_set):
        #Choose and evaluate point
        point = theta_set[i]
#         point = [theta_set[i]]
        eval_point = np.array([point])
#         print(eval_point, train_sse)
        #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        #Save GP outputs
        model_sse = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
#             if verbose == True:
#                 print("Point",eval_point)
#                 print("Model Mean",model_sse)
#                 print("Model Var", model_variance)
        
        #Ensures sse is saved instead of ln(sse)
        if obj == "obj":
            sse[i] = model_sse
        else:
            sse[i] = np.exp(model_sse)
        var[i] = model_variance
        stdev[i] = np.sqrt(model_variance)  

    return sse, var, stdev #Prints just the value

def LOO_eval_GP_emulator_set(theta_set, Xexp, true_model_coefficients, model, likelihood, verbose = False, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1):
    """ 
    Calculates the expected improvement of the emulator approach
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
    
    #Define length and dimsenionality of theta_set
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
    
    # Loop over theta 1
    for i in range(len_set):        
    ##Calculate Values
        #Caclulate GP vals for each value given theta_j and x_j
        point = list(theta_set[i])
        eval_point = np.array([point])
        #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        #Save GP Valuez
        model_mean = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
        
        #Calculate corresponding experimental data from theta_set value
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

        #Save values for each value in theta_set (in this case only 1 value)
        GP_mean_all[i] = model_mean
        GP_var_all[i] = model_variance
        GP_stdev = np.sqrt(GP_var_all)
#     print(GP_mean_all)
    return GP_mean_all, GP_var_all, GP_stdev, SSE, SSE_var_GP, SSE_stdev_GP
    
def LOO_eval_GP_emulator_sse(theta_set, Xexp, Yexp,true_model_coefficients, model, likelihood, verbose=False, skip_param_types=0, CS = 1):
    """ 
    Calculates the expected improvement of the emulator approach
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
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
    
    Returns
    -------
        SSE: ndarray, The SSE of the model 
        SSE_var_GP: ndarray, The varaince of the SSE pf the GP model
        SSE_stdev_GP: ndarray, The satndard deviation of the SSE of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"

    #Define dimensionality of X
    m = Xexp.shape[1]
    
    #Set theta_set to only be parameter values instead of theta_j, x_j
    theta_set_params = theta_set[:, 0:-m]
    
    #Define the length of theta_set and the number of parameters that will be regressed (q)
    if len(theta_set_params.shape) > 1:
        len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    else:
        len_set, q = 1, theta_set_params.shape[0]
    
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    
    #Initialize values
    SSE_model = np.zeros((len_set))
    SSE_sim = np.zeros((len_set))
    ##Calculate Best Error
    # Loop over theta 1
    for i in range(len_set): 
    #Initialize values for saving data
        GP_mean = np.zeros((Xexp.shape[0]))
        y_sim = np.zeros((Xexp.shape[0]))
        SSE = 0
        #Loop over experimental data 
        for k in range(Xexp.shape[0]):
            ##Calculate Values
            #Caclulate sse for each value theta_j, xexp_k
            point = list(theta_set_params[i])
            #Append Xexk_k to theta_set to evaluate at theta_j, xexp_k
            x_point_data = list(Xexp[k]) #astype(np.float)
            #Create point to be evaluated
            point = point + x_point_data
#             print(point, type(point))
            eval_point = np.array([point])
            #Evaluate GP model
            #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_mean = GP_Outputs[3].numpy()[0] #1xn
            GP_mean[i] = model_mean
            #Calculate y_sim & sse_sim
            if CS == 1:
                #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
                y_sim[k] = create_y_data(eval_point)
            else:
                y_sim[k] = create_y_data(eval_point, true_model_coefficients, Xexp, skip_param_types)
        
        #Compute GP SSE and SSE_sim for that point
        SSE_model[i] = np.sum((GP_mean - Yexp)**2)
        SSE_sim[i] = np.sum((y_sim - Yexp)**2)
        
    return SSE_model, SSE_sim

def LOO_Plots_2_Input(iter_space, GP_mean, sse_sim, GP_stdev, Theta, Case_Study, DateTime, obj, set_lengthscale = None, save_figure= True, emulator = False):
#     print(sse_sim.shape, GP_mean.shape, iter_space.shape)
    
    p = GP_mean.shape[0]
    fxn = "LOO_Plots_2_Input"
    t = len(iter_space)
#     print(X1.shape, X2.shape, GP_mean.shape)
    # Compare the experiments to the true model
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
#     plt.scatter(iter_space,Y_space, label = "$y_{exp}$")
#     label = "$log(SSE_{model})$"
    plt.scatter(iter_space,np.log(GP_mean), label = r'$\mathbf{log(e(\theta))_{model}}$', s=100 )
    plt.scatter(iter_space,np.log(sse_sim), label = r'$\mathbf{log(e(\theta))_{sim}}$' , s=50)
    
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
    plt.ylabel(r'$\mathbf{log(e(\theta))}$', fontsize=16, fontweight='bold')
    
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

    #Save CSVs
    iter_space_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/iter_space")
    GP_mean_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/log_sse_model")
    sse_sim_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/log_sse_sim")
    csv_item_list = [iter_space, np.log(GP_mean), np.log(sse_sim)]
    make_csv_list = [iter_space_path, GP_mean_path, sse_sim_path]
    
    for i in range(len(make_csv_list)):
        save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")
#         print("2", make_csv_list[i])
    
    #Save Figures (if applicable)
    if save_figure == True:
        path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = True)
        save_fig(path, ext='png', close=True, verbose=False) 
#         print(path)
    else:
        plt.show()
        plt.close()
        
    return

def LOO_Plots_3_Input(iter_space, GP_mean, y_sim, GP_stdev, Theta, Case_Study, DateTime, set_lengthscale = None, save_figure = True):
    fxn = "LOO_Plots_3_Input"
    emulator = True
    p = GP_mean.shape[0]
    t = len(iter_space)
    obj = "obj"
#     print(X1.shape, X2.shape, GP_mean.shape)
    # Compare the experiments to the true model
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
    plt.scatter(iter_space,GP_mean, label = "$y_{model}$", s=100 )
    plt.scatter(iter_space,y_sim, label = "$y_{sim}$" , s=50)
    
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
    plt.ylabel("y value", fontsize=16, fontweight='bold')
    
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
    #Save CSVs
    iter_space_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/iter_space")
    GP_mean_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/y_model")
    y_sim_path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = False, csv_end = "/y_sim")
    csv_item_list = [iter_space, GP_mean, y_sim]
    make_csv_list = [iter_space_path, GP_mean_path, y_sim_path]
    
    for i in range(len(make_csv_list)):
        save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")
#         print("3", make_csv_list[i])
    
    #Save Figures (if applicable)
    if save_figure == True:
        path = path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime, is_figure = True)
#         print(path)
        save_fig(path, ext='png', close=True, verbose=False) 
    else:
        plt.show()
        plt.close()
        
    return

def path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime = None, is_figure = True, csv_end = None):
    """
    names a path
    
    Parameters
    ----------
        emulator: True/False, Determines if GP will model the function or the function error
        ep: float, float,int,tensor,ndarray (1 value) The original exploration bias parameter
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        fxn: str, The name of the function whose file path name will be created
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        mesh_combo: str, the name of the combination of parameters - Used to make a folder name
        bo_iter: int, integer, number of the specific BO iterations
        title_save: str or None,  A string containing the title of the file of the plot
        run, int or None, The iteration of the number of times new training points have been picked
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        is_figure: bool, used for saving CSVs as part of this function and for calling the data from a CSV to make a plot
        csv_end: str, the name of the csv file
    Returns:
        path: str, The path to which the file is saved
    
    """

    obj_str = "/"+str(obj)
    len_scl = "/len_scl_varies"
    org_TP_str = "/TP_"+ str(t)
    CS = "/CS_" + str(Case_Study)

    if emulator == False:
        Emulator = "/GP_Error_Emulator"
        method = ""
    else:
        Emulator = "/GP_Emulator"
            
    fxn_dict = {"LOO_Plots_2_Input":"/SSE_gp_val" , "LOO_Plots_3_Input":"/y_gp_val"}
    plot = fxn_dict[fxn]        
      
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
        
    path_end = CS + Emulator + org_TP_str + obj_str + len_scl + plot     

    path = path_org + "/GP_Validation_Figs" + path_end 
        
    if csv_end is not None:
        path = path + csv_end
#     print(path)   
    return path