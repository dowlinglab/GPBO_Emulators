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

from .bo_functions_generic import round_time, train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs, train_GP_scikit
from bo_methods_lib.GP_Vs_True_Sensitivity import eval_GP_x_space, eval_GP_emulator_x_space, Muller_plotter, path_name_gp_val
from .CS2_bo_plotters import save_csv, save_fig, plot_xy
    
# from .CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from .CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def scikit_gpytorch_mul_maps(all_data, X_space, Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, percentiles, skip_param_types = 0, kernel_func = "RBF", set_lengthscale = None, outputscl = False, train_iter = 300, initialize = 1, noise_std = 0.1, verbose = False, DateTime = None, save_csvs = True, save_figure= False, eval_Train = False, CutBounds = False, package = "gpytorch"):  
    """
    Compare GP models to True/Simulation Values
    
    Parameters:
    -----------
        all_data: ndarray, contains all data for GP
        X_space: ndarray, The p^2 x dim(x) meshgrid points for X over which to evaluate the GP
        Xexp: ndarray, The experimental data for y (the true value)
        Yexp: ndarray, The experimental data for y (the true value)
        true_model_coefficients: ndarray, The array containing the true values of the problem (may be the same as true_p)
        true_p: ndarray, The array containing the true values of theta parameters to regress- flattened array
        Case_Study: int, float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        bounds_p: ndarray, defines the bounds of the parameter space values (unused now, may use to define percentiles later)
        percentiles: ndarray, defines the percentiles at which to alter each parameter at each iteration for the movie
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
    p_sq = len(X_space)
    p = int(np.sqrt(p_sq))
    q = true_p.shape[0]
    t = len(all_data)
    
    #Set training data
    data_train = all_data
    #separate into y data and training parameter data
    if m > 1:
        train_p = torch.tensor(data_train[:,1:-m+1]).float() #8 or 10 (emulator) parameters 
    else:
        train_p = torch.tensor(data_train[:,1:-m]).float() #8 or 10 (emulator) parameters 

    train_y = torch.tensor(data_train[:,-1]).float()
    
    #Define X training data
    X_train = train_p[:,-m:]
    
    #Define model and likelihood
    if package == "gpytorch":
#         likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = torch.tensor(noise_level**2))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #noise = torch.tensor(np.ones(train_p.shape[0])*noise_level**2))
        model = ExactGPModel(train_p, train_y, likelihood, kernel = kernel_func, outputscl = outputscl) 
        hyperparameters  = train_GP_model(model, likelihood, train_p, train_y, noise_std, kernel_func, verbose, set_lengthscale, outputscl,
                                          initialize, train_iter, rand_seed)
        
        lenscl_final, lenscl_noise_final, outputscale_final = hyperparameters
        outputscale_final_print = '%.3e' % outputscale_final

        print("Outputscale", outputscale_final_print)
#     print('lengthscale: %.3f   noise: %.3f'% (model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()) )
#     print(type(model.covar_module.base_kernel.lengthscale.item()), type(model.likelihood.noise.item()))

    elif package == "scikit_learn":
        likelihood = None
        model_params = train_GP_scikit(train_p, train_y, noise_std, kernel_func, verbose, set_lengthscale, initialize, rand_seed= rand_seed)
        lenscl_final, lenscl_noise_final, model = model_params
        
    #Print noise and lengthscale hps
    lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
    lenscl_noise_print = '%.3e' % lenscl_noise_final
    
    print("Lengthscale", lenscl_print)
    print("Noise for lengthscale", lenscl_noise_print)
    
    #Create list to save evaluated arrays in
    eval_p_df = []
    
    #Save each percentile as a number from 0 to len(percentiles)
    pct_num_map = np.linspace(0,len(percentiles)-1, len(percentiles))
  
    #Evaluate GP with true parameters OR a close training point over meshgrid X1 X2
    eval_theta_num = 0 
    if eval_Train == False:
        eval_p_base = torch.tensor(true_p)
    else:
        eval_p_base = train_p[eval_theta_num,0:q]
    print("Base Theta Train for Movies:", np.round(eval_p_base.numpy(),6) )
    
    #Save the values of the meshgrid for this run to a CSV
    X_space_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/X_space_unmeshed", CutBounds = CutBounds, kernel = kernel_func, package = package, outputscl = outputscl)
    if save_csvs == True:
        save_csv(X_space, X_space_path, ext = "npy")
    
    #Evaluate at true value or close to a training point doing a sensitivity analysis
    #Loop over number of parameters
    for i in range(len(eval_p_base)):   
        #Clone the base value
        eval_p = eval_p_base.clone()
        #Loop over each percentile value    
        for j in range(len(percentiles)):
#             lower_theta = bounds_p[0,i]
#             upper_theta = bounds_p[1,i]
            base_theta = eval_p_base[i] 
            # Evaluate at the original point for each parameter and add or subtract j% of org value from theta
            #Note: Could modify this to be dependent on problem bounds
            new_eval_p = base_theta + base_theta*percentiles[j]
            #Change the value to the exact point except for 1 variable that is rounded to 2 sig figs after modification by a percent
            
            if percentiles[j] != 0:
                eval_p[i] = torch.tensor(float('%.2g' % float(new_eval_p)))
            #Or just set it to the original
            else:
                eval_p[i] = torch.tensor(float(new_eval_p))
                              
            #Append evaluated value to this list
            eval_p_df.append(list(eval_p.numpy()))
            #Evaluate values at each X_space point
            eval_components = eval_GP_x_space(eval_p, X_space, train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types = skip_param_types, noise_std = noise_std, CS = Case_Study, Xspace_is_Xexp = False) 
            #Evaluate the values at each X training point (Xexp)
            eval_components_Xexp = eval_GP_x_space(eval_p, Xexp, train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types = skip_param_types, noise_std = noise_std, CS = Case_Study, Xspace_is_Xexp = True)
            
            #Get GP predictions and true values
            GP_mean, GP_stdev, y_sim = eval_components
            GP_mean_Xexp, GP_stdev_Xexp, y_sim_Xexp = eval_components_Xexp
            
            #Make pandas df of values evaluated at training points and set indecies to start at 1 and save it as npy
            #Calculate APE for training points
            APE_Exp_Preds = 100*abs((y_sim_Xexp - GP_mean_Xexp)/y_sim_Xexp)
            #Put Xexp, GP prediction, true values, and APE in a list 
            Exp_Preds = [Xexp[:,x] for x in range(m)] + [y_sim_Xexp, GP_mean_Xexp, GP_stdev_Xexp, APE_Exp_Preds]
            #Convert list to array
            Exp_Preds = np.array( Exp_Preds )
            #Turn array into pandas df
            Exp_Preds_df = pd.DataFrame(data = Exp_Preds.T, columns= ['Xexp '+str(x+1) for x in range(m)] +["Y sim", "GP Mean", "GP Stdev", "APE"])
            #Fix index to have correctly labeled Xexp point
            Exp_Preds_df.index += 1
            #Save to csv
            Exp_Preds_df_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "", CutBounds = CutBounds, Mul_title = "/Exp_Preds", param = param_dict[i], percentile = pct_num_map[j], kernel = kernel_func, package = package, outputscl = outputscl)
            if save_csvs == True:
                save_csv(Exp_Preds_df, Exp_Preds_df_path, ext = "csv")
        
            #Plot Values
            if Case_Study == 2.2:
                #Define minimal and saddle points of true function
                minima = np.array([[-0.558,1.442],
                              [-0.050,0.467],
                              [0.623,0.028]])

                saddle = np.array([[-0.82,0.62],
                              [0.22,0.30]])
                
                theta_eval = np.round(eval_p.tolist(),4)
                X_mesh = X_space.reshape(p,p,-1).T
                
                #Create titles for subplots
                if eval_Train == False:
                    title1 = "True Values " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$'
                else:
                    title1 = "Sim Val " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$' 

                title2 = "GP Mean " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$'
                title3 = "GP StDev " + r'$' + param_dict[i] + "=" + str(theta_eval[i]) + '$'
                
                #Define values to plot and how to save them as csvs
                z = [y_sim.T, GP_mean.T, GP_stdev.T]     
                Mul_title = ["/Sim_val", "/GP_mean", "/GP_stdev"]
                title = [title1, title2, title3]
                
                #Make heat maps
                Muller_plotter(X_mesh, z, minima, saddle, title, set_lengthscale, train_iter, t, Case_Study, CutBounds, lenscl_final,
lenscl_noise_final, kernel_func, DateTime, Xexp, save_csvs, save_figure, Mul_title = Mul_title, param = param_dict[i], percentile = pct_num_map[j], package = package, outputscl = outputscl)

            elif Case_Study == 1:
                plot_xy(X_space, Xexp, Yexp, None ,GP_mean, y_sim,title = "XY Comparison")
            
        #Break loop over params if maps will all be the same for each parameter (i,e. percentile is 1 number and that number is 0)
        if len( percentiles ) <= 1 and percentiles[-1] == 0:
                break  
                
    #Save all evaluated parameter values to a csv
    eval_p_df = pd.DataFrame(eval_p_df, columns = list(param_dict.values()))
    eval_p_df_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/eval_p_df", CutBounds = CutBounds, kernel = kernel_func, package = package, outputscl = outputscl)
    if save_csvs == True:
        save_csv(eval_p_df, eval_p_df_path, ext = "npy")
    return