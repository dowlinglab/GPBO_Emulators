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
from .CS2_bo_plotters import save_csv, save_fig, plot_xy
    
# from .CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from .CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def Compare_GP_True_Param_Sens(all_data, x_space_points, eval_theta_num, Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, value_num, skip_param_types = 0, kernel_func = "RBF", set_lengthscale = None, outputscl = False, train_iter = 300, initialize = 1, noise_std = 0.1, verbose = False, DateTime = None, save_csvs = True, save_figure= False, eval_Train = False, CutBounds = False, package = "gpytorch"):  
    """
    Compare GP models to True/Simulation Values
  
    Parameters:
    -----------
        all_data: ndarray, contains all data for GP
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
        #If the noise is larger than 0.01, set it appropriately, otherwise use a regular GaussianLikelihood noise and set it manually 
        noise = torch.tensor(noise_std**2)
        if noise_std >= 0.01:
            noise = torch.ones(train_p.shape[0])*noise_std**2
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise, learn_additional_noise=False)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
            likelihood.noise = noise  # Some small value. Try 1e-4
            likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise
            
        model = ExactGPModel(train_p, train_y, likelihood, kernel = kernel_func, outputscl = outputscl) 
        hyperparameters  = train_GP_model(model, likelihood, train_p, train_y, noise_std, kernel_func, verbose, set_lengthscale, outputscl,
                                          initialize, train_iter, rand_seed)
        
        lenscl_final, lenscl_noise_final, outputscale_final = hyperparameters
        
#     print('lengthscale: %.3f   noise: %.3f'% (model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()) )
#     print(type(model.covar_module.base_kernel.lengthscale.item()), type(model.likelihood.noise.item()))

    elif package == "scikit_learn":
        likelihood = None
        model_params = train_GP_scikit(train_p, train_y, noise_std, kernel_func, verbose, set_lengthscale, outputscl, initialize, 
                                       rand_seed= rand_seed)
        lenscl_final, lenscl_noise_final, outputscale_final, model = model_params
        
    #Print noise and lengthscale hps
    lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
    lenscl_noise_print = '%.3e' % lenscl_noise_final
    outputscale_final_print = '%.3e' % outputscale_final
    
    print("Lengthscale", lenscl_print)
    print("Noise for lengthscale", lenscl_noise_print)
    print("Outputscale", outputscale_final_print)

    #Evaluate at true value or close to a training point doing a sensitivity analysis
    if eval_Train == False:
        eval_p_base = torch.tensor(true_p)   
    else:
        eval_p_base = train_p[eval_theta_num,0:q]
    print("Base Theta Train for Movies:", np.round(eval_p_base.numpy(),6) )
    
    #Define X_space for param sensitivity testing
    train_xspace_set = np.array([data_train[m + eval_theta_num*n,1:] for m in x_space_points])
    X_space= np.array([Xexp[m] for m in x_space_points])
    
    #Create list to save evaluated arrays in and arrays to store GP mean/stdev and true predictions in
    GP_mean_all = np.zeros((len(X_space), q, value_num) )
    GP_stdev_all = np.zeros((len(X_space), q, value_num) )
    y_sim_all = np.zeros((len(X_space), q, value_num) )
    
    #Create list to save evaluated parameter sets and values in
    eval_p_df = []
    values_list = [] 
    
    ##Evaluate parameter sets at each Xspace value
    #Loop over all parameters
    for i in range(len(eval_p_base)):   
        #Clone the base value
        eval_p = eval_p_base.clone()
        #Define upper and lower theta bounds
        lower_theta = bounds_p[0,i]
        upper_theta = bounds_p[1,i]
        #Define Values to test
        values = np.linspace(lower_theta, upper_theta, value_num) #Note: Default to 41 
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
                
    #Plot GP vs y_sim predictions for each Xexp Value and save data
    all_data_to_plot = [y_sim_all, GP_mean_all, GP_stdev_all] 
#     print(y_sim_all.shape)
    mul_plot_param(all_data_to_plot, set_lengthscale, train_iter, t, Case_Study, CutBounds, X_space, x_space_points, param_dict, values_list, val_num_map, lenscl_final, train_xspace_set, lenscl_noise_final, outputscale_final, kernel_func, DateTime, Xexp, save_csvs, save_figure, package)
    
    #Save all evaluated parameter values to a csv
    eval_p_df = pd.DataFrame(eval_p_df, columns = list(param_dict.values()))
    eval_p_df_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/eval_p_df", CutBounds = CutBounds, kernel = kernel_func, package = package)
    train_xspace_set_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/train_xspace_data", CutBounds = CutBounds, kernel = kernel_func, package = package)
    if save_csvs == True:
        save_csv(eval_p_df, eval_p_df_path, ext = "npy")
    
    return

def eval_GP(theta_set, X_space, train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types = 0, noise_std = 0.1, CS=1):
    """
    Preps GP model for evaluation
    
    Parameters:
    -----------
        theta_set: ndarray (len_set x dim_param), array of Theta values 
        X_space: ndarray, The p^2 x dim(x) meshgrid points for X over which to evaluate the GP
        train_y: tensor or ndarray, The training y data
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        noise_std: float, int: The standard deviation of the noise. Default 0.1
        CS: int, float, the number of the case study to be evaluated. Default is 1, other option is 2.2
        Xspace_is_Xexp: whether X_space is is a set of meshgrid values or Xexp values
        
    
    Returns:
    --------
        eval_components: ndarray, The componenets evaluate by the GP and true simulation values (GP_mean, GP_stdev, y_sim)
    """
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Ensure train_y is a tensor
    if isinstance(train_y, np.ndarray)==True:
        train_y = torch.tensor(train_y) #1xn
    
    #If there is a likelihood, we are using gpytorch. Otherwise, we're using scikit learn
    if likelihood != None:
        #Assert correct likelihood and model  types
        assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
        assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
        #Put model and likelihood in evaluation mode
        model.eval()
        likelihood.eval()
    
    #Evaluate GP based on property emulator
    eval_components = eval_GP_emulator(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types, CS)
    
    return eval_components

def eval_GP_emulator(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types=0, CS=1):
    """ 
    Calculates the expected improvement of the emulator approach
    Parameters
    ----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        X_space: ndarray, The points for X over which to evaluate the GP (p^2 x dim(x) or n x dim(x))
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        model: bound method, The model that the GP is bound by (gpytorch ot scikitlearn method)
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood or None
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. Default 0
        CS: float, the number of the case study to be evaluated. Default is 1, other option is 2.2 
        Xspace_is_Xexp: bool, whether X_space is is a set of meshgrid values or Xexp values
    
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
    m = X_space.shape[1]
    n = X_space.shape[0]
      
    #Define the length of theta_set and the number of parameters that will be regressed (q)
    len_set, q = theta_set_params.shape[0], theta_set_params.shape[1]
    
    #Initialize values for saving data
    GP_mean = 0
    GP_var = 0
    y_sim = 0
    
    ##Calculate Values
    #Define a parameter set, point
    point = list(theta_set_params[0])
    #Append Xexp_k to theta_set to evaluate at theta, xexp_k
    x_point_data = list(X_space.flatten()) #astype(np.float)
    #Create point to be evaluated
    point = point + x_point_data
    eval_point = torch.from_numpy(np.array([point])).float()
    #Evaluate GP model
    #If there is a likelihood, we are using gpytorch. Otherwise, we're using scikit learn
    if likelihood != None: 
        #Evaluate GP given parameter set theta and state point value
        #Note: eval_point[0:1] prevents a shape error from arising when calc_GP_outputs is called
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_mean = GP_Outputs[3].numpy()[0] #1xn
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
    else:
        #Evaluate GP given parameter set theta and state point value
        model_mean, model_variance = model.predict(eval_point, return_std=True)

    #Save values of GP mean and variance
    GP_mean = model_mean
    GP_var = model_variance

    #Calculate y_sim
    if CS == 1:
        #Case study 1, the 2D problem takes different arguments for its function create_y_data than 2.2
        y_sim = create_y_data(eval_point)
    else:
        y_sim = create_y_data(eval_point, true_model_coefficients, X_space, skip_param_types)
    
    #Define GP standard deviation   
    GP_stdev = np.sqrt(GP_var)  
    
    return GP_mean, GP_stdev, y_sim 

def mul_plot_param(data, set_lengthscale, train_iter, t, Case_Study, CutBounds, X_space, x_space_points, param_dict, values_list, val_num_map, lenscl_final, train_xspace_set, lenscl_noise_final, outputscale_final, kernel = "RBF", DateTime = None, X_train = None, save_csvs = False, save_figure = False, package = "", plot_one_param = None):
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
    
    #Get data
    y_sim_data, GP_mean_data, GP_stdev_data = data
    
    #Make a new plot for each X Coordinate and parameter value tested
    #loop over Xexp values
    for k in range(y_sim_data.shape[0]):
        lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
        half = int(len(lenscl_print)/2)
        #Define a title for the whole plot based on lengthscale and lengthscale noise values
        lenscl_noise_print = '%.3e' % lenscl_noise_final
        ops_print = '%.3e' % outputscale_final
        title_str = "Xexp Point " + str(x_space_points[k]+1) + '\n' + r'$\ell = $' + str(lenscl_print[:half]) + '\n' + str(lenscl_print[half:]) + "\n" + r'$\sigma_{\ell} = $' + lenscl_noise_print + ' & ' + r'$\tau = $' + ops_print
            
        #None if looping over all parameters, use y_sim_data.shape[1], only loop over 1 specific value otherwise
        if plot_one_param != None:
            range_val = 1   
        else:
            range_val = y_sim_data.shape[1]
            
        #Loop over parameter values
        for i in range(range_val):
            #Set i as the integer value corresponding to the parameter you want to plot if applicable
            if plot_one_param != None:
                i = plot_one_param   
            #Create figure
            fig = plt.figure(figsize = (6.4,4))
            #Add plot values
            plt.plot(values_list[i], GP_mean_data[k,i], label = "GP mean")
            plt.scatter(train_xspace_set[k,i], train_xspace_set[k,-1], label = "Training", color = "red", marker = "*")
            plt.plot(values_list[i], y_sim_data[k,i], label = "Y Sim")
            plt.fill_between(values_list[i],
                            GP_mean_data[k,i] - 1.96 * GP_stdev_data[k,i],
                            GP_mean_data[k,i] + 1.96 * GP_stdev_data[k,i],
                            alpha=0.3)

            #Set plot details        
        #     plt.legend(loc = "best")
            plt.title(title_str)
            plt.xlabel(r'$' + param_dict[i] +'$', fontsize=16, fontweight='bold')
            plt.ylabel("Muller Potential", fontsize=16, fontweight='bold')
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(direction="in",top=True, right=True)
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=5)
            plt.minorticks_on() # turn on minor ticks
            plt.tick_params(which="minor",direction="in",top=True, right=True)
            parm_val_txt = ("Param Values: " + str(train_xspace_set[k,0:len(param_dict)].round(decimals=3)))
            plt.text(0.5, -0.01, parm_val_txt, ha='center', wrap=True, transform = fig.transFigure)
            plt.tight_layout()
#             plt.grid(True)

            #Save CSVs and Figures
            if save_csvs == True:
                #Create a list to save paths in
                mul_val_title = ["/Sim_val", "/GP_mean", "/GP_stdev"]
                mul_vals = [y_sim_data[k,i], GP_mean_data[k,i], GP_stdev_data[k,i]]
                mul_val_paths = []               
                #For each values being plotted
                for j in range(len(mul_vals)):
                    #Create a path
                    mul_val_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "", CutBounds = CutBounds, X_val = k, val_title = mul_val_title[j], param = param_dict[i], kernel = kernel, package = package)
                    #Save path to list
                    mul_val_paths.append(mul_val_path) 
                
                csv_item_list_vals = mul_vals
                make_csv_list_vals = mul_val_paths
                for j in range(len(make_csv_list_vals)):
                    save_csv(csv_item_list_vals[j], make_csv_list_vals[j], ext = "npy")
                         
            #Save figure or show and close figure
            if save_figure == True:
                path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = True, CutBounds = CutBounds, X_val = k, val_title = "/Mul_Comp_Figs_Param", param = param_dict[i], kernel = kernel, package = package)
                save_fig(path, ext='png', close=True, verbose=False) 
            else:
                plt.show()
            plt.close()
                    
    #Create paths for x training points and values
    if save_csvs == True:
        csv_ends = ["/X_train", "/Param_Values", "X_space"]
        x_trn_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[0], CutBounds = CutBounds, X_val = "", val_title = "", param = "", kernel = "", package = "")                   
        param_val_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[1], CutBounds = CutBounds, X_val = "", val_title = "", param = "", kernel = "", package = "")
        x_spc_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[2], CutBounds = CutBounds, X_val = "", val_title = "", param = "", kernel = "", package = "")
        # Create a list of extra items to save and corresponding path names
        csv_item_list = [X_train, np.array(values_list), X_space]
        make_csv_list = [x_trn_path, param_val_path, x_spc_path]
        #Save values as CSVs
        for i in range(len(make_csv_list)):
            save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")

    return plt.show()

def path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime = None, is_figure = True, csv_end = None, CutBounds = False, X_val = "", val_title = "", param = "", kernel = "", package = ""):
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