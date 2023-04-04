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
from sklearn.gaussian_process.kernels import RBF, Matern

from .bo_functions_generic import round_time, train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs, train_GP_scikit
from .CS2_bo_plotters import save_csv, save_fig, plot_xy
    
# from .CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from .CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def Compare_GP_True_Movie(all_data, X_space, Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, percentiles, skip_param_types = 0, kernel_func = "RBF", set_lengthscale = None, train_iter = 300, initialize = 1, noise_std = 0.1, verbose = False, DateTime = None, save_csvs = True, save_figure= False, eval_Train = False, CutBounds = False, package = "gpytorch"):  
    """
    Run GP Validation using a leave one out scheme
    
    Parameters:
    -----------
        all_data: ndarray, contains all data for GP
        X_space: ndarray, The p^2 x dim(x) meshgrid points for X over which to evaluate the GP
        Xexp: ndarray, The experimental data for y (the true value)
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
    param_dict = {0 : 'a_1', 1 : 'a_2', 2 : 'a_3', 3 : 'a_4',
              4 : 'b_1', 5 : 'b_2', 6 : 'b_3', 7 : 'b_4'}
    n, m = Xexp.shape
    p_sq = len(X_space)
    p = int(np.sqrt(p_sq))
    q = true_p.shape[0]
    t = len(all_data)
    
    #Create empy lists to store index, GP model val, y_sim vals, sse's from emulator vals, SSE from emulator val, and sse from GP vals
    
    #Set training data
    data_train = all_data
    #separate into y data and parameter data
    if m > 1:
        train_p = torch.tensor(data_train[:,1:-m+1]).float() #8 or 10 (emulator) parameters 
    else:
        train_p = torch.tensor(data_train[:,1:-m]).float() #8 or 10 (emulator) parameters 

    train_y = torch.tensor(data_train[:,-1]).float()
    X_train = train_p[:,-m:]
    #Define model and likelihood
    if package == "gpytorch":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood, kernel = kernel_func) 
        # Train GP
    #     print(train_p.dtype, train_y.dtype)
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose, set_lenscl = set_lengthscale, initialize = initialize, rand_seed = False)
        best_hps = train_GP
        lenscl_final = best_hps[0]
        lenscl_noise_final = best_hps[1]
        outputscale_final = best_hps[2]
        print("Lengthscale", np.round(model.covar_module.base_kernel.lengthscale.detach().numpy(),4))
        print("Noise for lengthscale", np.round(model.likelihood.noise.item(),4))
        print("Outputscale", np.round(model.covar_module.outputscale.item(),4))
#     print('lengthscale: %.3f   noise: %.3f'% (model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()) )
#     print(type(model.covar_module.base_kernel.lengthscale.item()), type(model.likelihood.noise.item()))

    elif package == "scikit_learn":
        lenscl_final, gaussian_process = train_GP_scikit(train_p, train_y, noise_std, kernel_func, verbose, set_lengthscale, initialize, rand_seed = False)
        print("Lengthscale", np.round(np.array(lenscl_final),4))
        lenscl_noise_final = 0

    #Create list to save evaluated arrays in
    eval_p_df = []
    #Save each percentile as a number from 0 to len(percentiles)
    pct_num_map = np.linspace(0,len(percentiles)-1, len(percentiles))
#     
    #Evaluate GP with true parameters OR a close training point over meshgrid X1 X2
    if eval_Train == False:
        eval_p_base = true_p
    else:
        eval_p_base = train_p[0,0:q] #For Either Cut Bound Problem
#         eval_p = torch.tensor([-1.29, -1.63, -8.65,  0.92,  0.63,  1.11, 10.53,  1.91]) #For TP = 40
#         eval_p = torch.tensor([1.05, -1.61, -7.16, -1.39, -0.47,  1.68, 14.31,  0.04]) #For TP = 100
#         print(len(eval_p))
    print("Base Theta Train for Movies:", eval_p_base)
    X_space_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/X_space_unmeshed", CutBounds = CutBounds, kernel = kernel_func, package = package)
#     print("X space path", X_space_path)
    save_csv(X_space, X_space_path, ext = "npy")
    
    #Evaluate at true value or close to a training point doing a sensitivity analysis
    #Loop over number of thetas
    for i in range(len(eval_p_base)):   
        # Evaluate at the lower bound for each theta and add or subtract 100% of org value from theta to see what happens
        #Note: Could modify this to be dependent on problem bounds
#         print(percentiles)
        eval_p = eval_p_base.clone()
        for j in range(len(percentiles)):
#             lower_theta = bounds_p[0,i]
#             upper_theta = bounds_p[1,i]
            base_theta = eval_p_base[i] 
            new_eval_p = base_theta + base_theta*percentiles[j]
            #Change the value to the exact point except for 1 variable that is rounded to 2 sig figs after modification by a percent
            eval_p[i] = torch.tensor(float('%.2g' % float(new_eval_p)))
#             print("Eval_p: \n", eval_p)
            eval_p_df.append(list(eval_p.numpy()))
#             print("eval_p_df: \n", eval_p)
            if package == "gpytorch":
                #Evaluate values at each X_space point
                eval_components = eval_GP_x_space(eval_p, X_space, train_y, true_model_coefficients, model, likelihood, verbose, train_p = train_p, skip_param_types = skip_param_types, noise_std = noise_std, CS = Case_Study, Xspace_is_Xexp = False) 
                #Evaluate the values at the training point
                eval_components_Xexp = eval_GP_x_space(eval_p, Xexp, train_y, true_model_coefficients, model, likelihood, verbose, train_p = train_p, skip_param_types = skip_param_types, noise_std = noise_std, CS = Case_Study, Xspace_is_Xexp = True)
                         
            elif package == "scikit_learn":
                #Evaluate values at each X_space point
                eval_components = eval_GP_scipy(eval_p, X_space, true_model_coefficients, gaussian_process, skip_param_types = skip_param_types, 
                                                CS = Case_Study, Xspace_is_Xexp = False)
                #Evaluate the values at the training point
                eval_components_Xexp = eval_GP_scipy(eval_p, X_space, true_model_coefficients, gaussian_process, skip_param_types = skip_param_types, 
                                                     CS = Case_Study, Xspace_is_Xexp = True)
                
            GP_mean, GP_stdev, y_sim = eval_components
            GP_mean_Xexp, GP_stdev_Xexp, y_sim_Xexp = eval_components_Xexp
            
            #Make pandas df of values evaluated at training points and set indecies to start at 1 and save it as npy
            APE_Exp_Preds = 100*abs((y_sim_Xexp - GP_mean_Xexp)/y_sim_Xexp)
            Exp_Preds = [Xexp[:,x] for x in range(m)] + [y_sim_Xexp, GP_mean_Xexp, GP_stdev_Xexp, APE_Exp_Preds]
#             Exp_Preds = np.array( [Xexp[:,x] for x in range(m)] + [y_sim_Xexp, GP_mean_Xexp, GP_stdev_Xexp, APE_Exp_Preds] )
#             print(Exp_Preds.shape)
            Exp_Preds_df = pd.DataFrame(data = [Exp_Preds], columns= ['Xexp '+str(x+1) for x in range(m)] +["Y sim", "GP Mean", "GP Stdev", "APE"])
            Exp_Preds_df.index += 1
            Exp_Preds_df_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "", CutBounds = CutBounds, Mul_title = "/Exp_Preds", param = param_dict[i], percentile = pct_num_map[j], kernel = kernel_func,
                                                package = package)
            save_csv(Exp_Preds_df, Exp_Preds_df_path, ext = "csv")
        
            #Plot true shape
            if Case_Study == 2.2:
                minima = np.array([[-0.558,1.442],
                              [-0.050,0.467],
                              [0.623,0.028]])

                saddle = np.array([[-0.82,0.62],
                              [0.22,0.30]])

                if eval_Train == False:
                    title1 = "True Values"
#                     print("True Theta" + eval_p.tolist())
                else:
#                     print("Test TP Rounded:", np.round(eval_p.tolist(),2))
        #             print(eval_p)
                    theta_eval = np.round(eval_p.tolist(),4)
                    title1 = "Sim Val " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$' 
#                     print(title1)
                X_mesh = X_space.reshape(p,p,-1).T
                title2 = "GP Mean " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$'
                title3 = "GP StDev " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$'
    
                z = [y_sim.T, GP_mean.T, GP_stdev.T]     
                Mul_title = ["/Sim_val", "/GP_mean", "/GP_stdev"]
                title = [title1, title2, title3]
                
                #Make heat maps
                Muller_plotter(X_mesh, z, minima, saddle, title, set_lengthscale, train_iter, t, Case_Study, CutBounds, lenscl_final,
lenscl_noise_final, kernel_func, DateTime, Xexp, save_csvs, save_figure, Mul_title = Mul_title, param = param_dict[i], percentile = pct_num_map[j], package = package)

            elif Case_Study == 1:
                plot_xy(X_space, Xexp, Yexp, None ,GP_mean, y_sim,title = "XY Comparison")

#         print("eval_p_df loop:", eval_p_df)
    eval_p_df = pd.DataFrame(eval_p_df, columns = list(param_dict.values()))
    eval_p_df_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/eval_p_df", CutBounds = CutBounds, kernel = kernel_func, package = package)
    save_csv(eval_p_df, eval_p_df_path, ext = "npy")
#     print("eval_p_df_path", eval_p_df_path)
    return

def eval_GP_x_space(theta_set, X_space, train_y, true_model_coefficients, model, likelihood, verbose, train_p = None, skip_param_types = 0, noise_std = 0.1, CS = 1, Xspace_is_Xexp = False):
    """
    Evaluates GP
    
    Parameters:
    -----------
        theta_set: ndarray (len_set x dim_param), array of Theta values 
        train_y: tensor or ndarray, The training y data
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        X_space: ndarray, The p^2 x dim(x) meshgrid points for X over which to evaluate the GP
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
        emulator: True/False: Determiens whether GP is an emulator of the function
        sparse_grd: True/False: Determines whether an assumption or sparse grid is used
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
      
    model.eval()
    #Puts likelihood in evaluation mode
    likelihood.eval()
    
    #Evaluate GP based on property emulator
    eval_components = eval_GP_emulator_x_space(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types, CS, Xspace_is_Xexp)
    
    return eval_components

def eval_GP_scipy(theta_set, X_space, true_model_coefficients, model, skip_param_types=0, CS=1, Xspace_is_Xexp = False):
    """ 
    Evaluates the GP over some set of X_values and some set of parameter values
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of parameter values to test
        X_space: ndarray, The p^2 x dim(x) meshgrid points for X over which to evaluate the GP
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: gaussian_process, the model from scikitlearn
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
    
    Returns
    -------
        GP_mean: ndaarray, Array of GP mean predictions at X_space and theta_set
        GP_stdev: ndarray, Array of GP variances related to GP means at X_space and theta_set
        y_sim: ndarray, simulated values at X_space and theta_set
    """
    #Define dimensionality of X
    m = X_space.shape[1]
    p_sq = X_space.shape[0]
    p = int(np.sqrt(p_sq))
    
    #Set theta_set to only be parameter values
    theta_set_params = theta_set
    
    #Define the length of theta_set (len_set) and the number of parameters that will be regressed (q)
    if len(theta_set_params.shape) > 1:
        len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    else:
        theta_set_params = clean_1D_arrays(theta_set_params, param_clean = True)
        len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    
    #Initialize values for saving data
    GP_mean = np.zeros((p_sq))
    GP_var = np.zeros((p_sq))
    y_sim = np.zeros((p_sq))
    
    #Loop over number of X values
    for k in range(p_sq):
        ##Calculate Values
        #Define a parameter set, point
        point = list(theta_set_params[0])
        #Append Xexp_k to theta_set to evaluate at theta, xexp_k
        x_point_data = list(X_space[k]) #astype(np.float)
        #Create point to be evaluated
        point = point + x_point_data
        eval_point = torch.from_numpy(np.array([point])).float()
        #Evaluate GP given parameter set theta and state point value
        model_mean, model_variance = model.predict(eval_point, return_std=True)
        #Add values to lists for GP mean and standard deviation
        GP_mean[k] = model_mean
        GP_var[k] = model_variance
        
        #Calculate y_sim and save the value for each individual point
        if CS == 1:
            #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
            y_sim[k] = create_y_data(eval_point)
        else:
            y_sim[k] = create_y_data(eval_point, true_model_coefficients, X_space, skip_param_types)

    #Define GP standard deviation    
    GP_stdev = np.sqrt(GP_var)  
    
    if m > 1 and Xspace_is_Xexp == False:
        #Turn GP_mean, GP_stdev, and y_sim back into meshgrid form
        GP_stdev = np.array(GP_stdev).reshape((p, p))
        GP_mean = np.array(GP_mean).reshape((p, p))
        y_sim = np.array(y_sim).reshape((p, p))
    
    return GP_mean, GP_stdev, y_sim

def eval_GP_emulator_x_space(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types=0, CS=1, Xspace_is_Xexp = False):
    """ 
    Calculates the expected improvement of the emulator approach
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        X_space: ndarray, The p^2 x dim(x) meshgrid points for X over which to evaluate the GP
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: bool, Determines whether output is verbose. Default False
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
    
    Returns
    -------
        GP_mean: ndaarray, Array of GP mean predictions at X_space and theta_set
        GP_stdev: ndarray, Array of GP variances related to GP means at X_space and theta_set
        y_sim: ndarray, simulated values at X_space and theta_set
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"

    #Define dimensionality of X
    m = X_space.shape[1]
    p_sq = X_space.shape[0]
    p = int(np.sqrt(p_sq))
    #Set theta_set to only be parameter values
    theta_set_params = theta_set
    
    #Define the length of theta_set and the number of parameters that will be regressed (q)
    if len(theta_set_params.shape) > 1:
        len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    else:
        theta_set_params = clean_1D_arrays(theta_set_params, param_clean = True)
        len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    
    #Initialize values for saving data
    GP_mean = np.zeros((p_sq))
    GP_var = np.zeros((p_sq))
    y_sim = np.zeros((p_sq))
    
    #Loop over experimental data 
    for k in range(p_sq):
        ##Calculate Values
        #Caclulate sse for each value theta_j, xexp_k
        point = list(theta_set_params[0])
        #Append Xexk_k to theta_set to evaluate at theta_j, xexp_k
        x_point_data = list(X_space[k]) #astype(np.float)
        #Create point to be evaluated
        point = point + x_point_data
        eval_point = torch.from_numpy(np.array([point])).float()
#         if k <10:
#             print(eval_point[0:1])
        #Evaluate GP model
        #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_mean = GP_Outputs[3].numpy()[0] #1xn
        GP_mean[k] = model_mean
#         if k <10:
#             print(model_mean)
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
        GP_var[k] = model_variance
        #Calculate y_sim
        if CS == 1:
            #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
            y_sim[k] = create_y_data(eval_point)
        else:
            y_sim[k] = create_y_data(eval_point, true_model_coefficients, X_space, skip_param_types)

        
    GP_stdev = np.sqrt(GP_var)  
    
#     if Xspace_is_Xexp == True:
#         print(GP_mean.shape, y_sim.shape, GP_stdev.shape)
        
    if m > 1 and Xspace_is_Xexp == False:
        #Turn GP_mean, GP_stdev, and y_sim back into meshgrid form
        GP_stdev = np.array(GP_stdev).reshape((p, p))
        GP_mean = np.array(GP_mean).reshape((p, p))
        y_sim = np.array(y_sim).reshape((p, p))
    
    return GP_mean, GP_stdev, y_sim 

def Muller_plotter(test_mesh, z, minima, saddle, title, set_lengthscale, train_iter, t, Case_Study, CutBounds, lenscl_final = "", lenscl_noise_final = "", kernel = "RBF", DateTime = None, X_train = None, save_csvs = False, save_figure = False, Mul_title = "", param = "", percentile = "", package = "", tot_lev = [40,40,75]):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: list of 3 NxN arrays containing all points that will be plotted for GP_mean, GP standard deviation, and y_sim
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        p_GP_Best: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        title: list of str, A string containing the title of the plots ex: ["Y_sim, GP Mean", "GP Stdev"]
        title_save: str, A string containing the title of the file of the plot
        obj: str, The name of the objective function. Used for saving figures
        ep: int or float, the exploration parameter
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        Bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run, int or None, The iteration of the number of times new training points have been picked
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Define figures and x and y data
    xx , yy = test_mesh #NxN, NxN
    
    #Assert sattements
    assert len(z) == len(title), "Equal number of data matricies and titles must be given!"
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    
    #Make figures and define number of subplots  
    fig, axes = plt.subplots(nrows = 1, ncols = len(z), figsize = (18,6))
    ax = axes
    
#     title_str = r'$\ell =' +  str(format(lenscl_final, '.3f')) + '$ & '+ r'$\sigma_{\ell} = ' + str(format(lenscl_noise_final, '.3f') + '$')
    title_str = r'$\ell = $' + str(np.round(lenscl_final,3)) + ' & ' + r'$\sigma_{\ell} = $' + str(np.round(lenscl_noise_final,5))
    if type(lenscl_final) != str:
        fig.suptitle(title_str, weight='bold', fontsize=18)
    
    #Set plot details
    #Loop over number of subplots
    for i in range(len(z)):
        #Assert statements
        assert z[i].shape==xx.shape, "Array z must be NxN"
        assert isinstance(z[i], np.ndarray)==True or torch.is_tensor(z[i])==True, "Heat map values must be numpy arrays or torch tensors."
        assert isinstance(title[i], str)==True, "Title must be a string" 
        
        #Create a colormap and colorbar for each subplot
        cs_fig = ax[i].contourf(xx, yy,z[i], levels = 900, cmap = "jet")
        if np.amax(z[i]) < 1e-1 or np.amax(z[i]) > 1000:
            cbar = plt.colorbar(cs_fig, ax = ax[i], format='%.2e')
        else:
            cbar = plt.colorbar(cs_fig, ax = ax[i], format = '%2.2f')
        
        #Create a line contour for each colormap
        cs2_fig = ax[i].contour(cs_fig, levels=cs_fig.levels[::tot_lev[i]], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
        ax[i].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=10, inline=1)
    
        #plot saddle pts and local minima, and training data X values if it's given
        if str(X_train) != "None":
            ax[i].scatter(X_train[:,0], X_train[:,1], color = "goldenrod", label = "Training", marker = "o")
            for index in range(len(X_train)):
                txt = ax[i].text(X_train[index,0], X_train[index,1], str(index+1), size=10, color ="white")
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])

        ax[i].scatter(minima[:,0], minima[:,1], color="black", label = "Minima", s=25, marker = (5,1))
        ax[i].scatter(saddle[:,0], saddle[:,1], color="white", label = "Saddle", s=25, marker = "X", edgecolor='k')   
        
        #Get legend information
        if i == len(z)-1:
            handles, labels = ax[i].get_legend_handles_labels()
        
        #Plots axes such that they are scaled the same way (eg. circles look like circles)
        ax[i].axis('scaled')  
        ax[i].set_xlabel('$x_1$',weight='bold',fontsize=16)
        ax[i].set_ylabel('$x_2$',weight='bold',fontsize=16)
        
        #Plot title and set axis scale
        ax[i].set_title(title[i], weight='bold',fontsize=16)
        ax[i].set_xlim(left = np.amin(xx), right = np.amax(xx))
        ax[i].set_ylim(bottom = np.amin(yy), top = np.amax(yy))      
          
    #Plots legend and title
    plt.tight_layout()
#     print(type(lenscl_noise_final), type(lenscl_final))
    fig.legend(handles, labels, loc="upper left")  #bbox_to_anchor=(-0.1, 1)

    #Save CSVs and Figures
    if save_csvs == True:
        z_csv_ends = Mul_title #optional
        csv_ends = ["/Minima", "/Saddle", "/X_train"]
        z_paths = []
        for i in range(len(z)):
            z_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "", CutBounds = CutBounds, Mul_title = Mul_title[i], param = param, percentile = percentile, kernel = kernel, package = package)
#             print(z_path)
            z_paths.append(z_path)  
        min_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[0], CutBounds = CutBounds, Mul_title = "", param = "", percentile = "",  package = "")
#         print("min_path", min_path)
        sad_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[1], CutBounds = CutBounds, Mul_title = "", param = "", percentile = "",  package = "")
#         print("sad_path", sad_path)
        x_trn_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[2], CutBounds = CutBounds, Mul_title = "", param = "", percentile = "",  package = "")
#         print("x_trn_path", x_trn_path)
        csv_item_list = z + [minima, saddle, X_train]
        make_csv_list = z_paths + [min_path, sad_path, x_trn_path]
        for i in range(len(make_csv_list)):
            save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")

    #Save figure or show and close figure
    if save_figure == True:
        path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = True, CutBounds = CutBounds, Mul_title = "/Mul_Comp_Figs", param = param, percentile = percentile, kernel = kernel, package = package)
#         print("fig_path", path)
        save_fig(path, ext='png', close=True, verbose=False) 
    else:
        plt.show()
        plt.close()
    
    plt.close()
    return plt.show()

def path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime = None, is_figure = True, csv_end = None, CutBounds = False, Mul_title = "", param = "", percentile = "", kernel = "", package = ""):
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
    
    if param != "":
        param = "/" + str(param)
    
    if percentile != "":
        percent = "_pct_" + str(int(percentile)).zfill(len(str(50))) #Note, 2 places is the default.
    else:
        percent = ""
            
    plot = Mul_title        
      
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
        
    path_end = CS + trn_iter + org_TP_str + pckg + kernel_type + len_scl + plot + param + percent  

    if CutBounds == True:
        cut_bounds = "_CB"
    else:
        cut_bounds = ""
        
    path = path_org + "/GP_Vs_Sim_Comp" + cut_bounds + path_end 
        
    if csv_end is not None:
        path = path + csv_end
   
    return path