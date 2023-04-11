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
def Compare_GP_True_Movie(all_data, X_space, Xexp, Yexp, true_model_coefficients, true_p, Case_Study, bounds_p, percentiles, skip_param_types = 0, kernel_func = "RBF", set_lengthscale = None, outputscl = False, train_iter = 300, initialize = 1, noise_std = 0.1, verbose = False, DateTime = None, save_csvs = True, save_figure= False, eval_Train = False, CutBounds = False, package = "gpytorch"):  
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
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
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
        model_params  = train_GP_scikit(train_p, train_y, noise_std, kernel_func, verbose, set_lengthscale, initialize, rand_seed = False)
        lenscl_final, lenscl_noise_final, model = model_params
        
    #Print noise and lengthscale hps
    lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
    lenscl_noise_print = '%.3e' % lenscl_noise_final
    
    print("Noise for lengthscale", lenscl_noise_print)
    print("Lengthscale", lenscl_print)

    #Create list to save evaluated arrays in
    eval_p_df = []
    
    #Save each percentile as a number from 0 to len(percentiles)
    pct_num_map = np.linspace(0,len(percentiles)-1, len(percentiles))
  
    #Evaluate GP with true parameters OR a close training point over meshgrid X1 X2
    if eval_Train == False:
        eval_p_base = torch.tensor(true_p)
    else:
        eval_p_base = train_p[0,0:q]
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
            eval_p[i] = torch.tensor(float('%.2g' % float(new_eval_p)))
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
                
    #Save all evaluated parameter values to a csv
    eval_p_df = pd.DataFrame(eval_p_df, columns = list(param_dict.values()))
    eval_p_df_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "/eval_p_df", CutBounds = CutBounds, kernel = kernel_func, package = package, outputscl = outputscl)
    if save_csvs == True:
        save_csv(eval_p_df, eval_p_df_path, ext = "npy")
    return

def eval_GP_x_space(theta_set, X_space, train_y, true_model_coefficients, model, likelihood, verbose, skip_param_types = 0, noise_std = 0.1, CS = 1, Xspace_is_Xexp = False):
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
    eval_components = eval_GP_emulator_x_space(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types, CS, Xspace_is_Xexp)
    
    return eval_components

def eval_GP_emulator_x_space(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types=0, CS=1, Xspace_is_Xexp = False):
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
        #Define a parameter set, point
        point = list(theta_set_params[0])
        #Append Xexp_k to theta_set to evaluate at theta, xexp_k
        x_point_data = list(X_space[k]) #astype(np.float)
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
        GP_mean[k] = model_mean
        GP_var[k] = model_variance
        
        #Calculate y_sim
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

def Muller_plotter(test_mesh, z, minima, saddle, title, set_lengthscale, train_iter, t, Case_Study, CutBounds, lenscl_final = "", lenscl_noise_final = "", kernel = "RBF", DateTime = None, X_train = None, save_csvs = False, save_figure = False, Mul_title = "", param = "", percentile = "", package = "", outputscl = "", tot_lev = [40,40,75]):
    '''
    Plots comparison of y_sim, GP_mean, and GP_stdev
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: list of 3 NxN arrays containing all points that will be plotted for GP_mean, GP standard deviation, and y_sim
        minima: ndarray, Array containing the minima of the true parameter set values
        saddle: ndarray, Array containing the saddle points of the true parameter set values
        title: list of str, A string containing the title of the plots ex: ["Y_sim, GP Mean", "GP Stdev"]
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        train_iter: int, number of training iterations to run for GP. Default is 300
        t: int, the total number of training points used to train the GP
        Case_Study: int, float, the number of the case study to be evaluated. Default is 1, other option is 2.2
        CutBounds: bool, Used for naming. Set True if bounds are cut from original values. Default False
        lenscl_final: "" or ndarray, The final lengthscale used by the GP
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
        tot_lev: ndarray or list, the values at which to set levels for the heat maps
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
     
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

    #Define a title for the whole plot based on lengthscale and lengthscale noise values
    lenscl_print = ['%.3e' % lenscl_final[i] for i in range(len(lenscl_final))]
    half = int(len(lenscl_print)/2)
    if lenscl_noise_final != "": 
        lenscl_noise_print = '%.3e' % lenscl_noise_final
        title_str = r'$\ell = $' + str(lenscl_print[:half]) + '\n' +  str(lenscl_print[half:]) + ' & ' + r'$\sigma_{\ell} = $' + lenscl_noise_print
    else:
        title_str = r'$\ell = $' + str(lenscl_print[:half]) + '\n' +  str(lenscl_print[half:])
    
    #If final lengthscale isn't a string, print the title
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
        
        #Plots axes such that they are scaled the same way (eg. circles look like circles) and name axes
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
    fig.legend(handles, labels, loc="upper left")  #bbox_to_anchor=(-0.01, 0.9), borderaxespad=0

    #Save CSVs and Figures
    if save_csvs == True:
        #Set Csv ends
        z_csv_ends = Mul_title
        csv_ends = ["/Minima", "/Saddle", "/X_train"]
        #Create a list to save paths in
        z_paths = []
        #For each values being plotted
        for i in range(len(z)):
            #Create a path
            z_path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = False, csv_end = "", CutBounds = CutBounds, Mul_title = Mul_title[i], param = param, percentile = percentile, kernel = kernel, package = package, outputscl = outputscl)
            #Save path to list
            z_paths.append(z_path)  
         #Create paths for minima, saddle, and x training points   
        min_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[0], CutBounds = CutBounds, Mul_title = "", param = "", percentile = "",  package = "", outputscl = "")
        sad_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[1], CutBounds = CutBounds, Mul_title = "", param = "", percentile = "",  package = "", outputscl = "")
        x_trn_path = path_name_gp_val("", "", "", Case_Study, DateTime, is_figure = False, csv_end = csv_ends[2], CutBounds = CutBounds, Mul_title = "", param = "", percentile = "",  package = "", outputscl = "")
        # Create a list of items to save and corresponding path names
        csv_item_list = z + [minima, saddle, X_train]
        make_csv_list = z_paths + [min_path, sad_path, x_trn_path]
        #Save values as CSVs
        for i in range(len(make_csv_list)):
            save_csv(csv_item_list[i], make_csv_list[i], ext = "npy")

    #Save figure or show and close figure
    if save_figure == True:
        path = path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime, is_figure = True, CutBounds = CutBounds, Mul_title = "/Mul_Comp_Figs", param = param, percentile = percentile, kernel = kernel, package = package, outputscl = outputscl)
        save_fig(path, ext='png', close=True, verbose=False) 
    else:
        plt.show()
        plt.close()
    
    plt.close()
    return plt.show()

def path_name_gp_val(set_lengthscale, train_iter, t, Case_Study, DateTime = None, is_figure = True, csv_end = None, CutBounds = False, Mul_title = "", param = "", percentile = "", kernel = "", package = "", outputscl = ""):
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
        Mul_title: str, part of the path to save the CSV data for each piece of this figure. Corresponds to what is being plotted/saved
        param: str, part of the path to save the CSV data for each piece of this figure. Which parameter is being plotted/saved
        percentile: str, part of the path to save the CSV data for each piece of this figure. Which percentile is being plotted/saved
        kernel: str, defines which kernel function was used. Default RBF
        package: str, Determines whether gpytorch or scikit learn will be used to build the GP model. Default "gpytorch"
        
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
    
    if outputscl == True:
        kernel_type = kernel_type + "_w_ops"
    
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
#         path_org = "Test_Figs2"
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

def Muller_plotter_simple(test_mesh, z, minima, saddle, title, X_train = None):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        minima: ndarray, Array containing the minima of the true parameter set values
        saddle: ndarray, Array containing the saddle points of the true parameter set values
        title: str, A string containing the title of the plot
        X_train: None or ndarray, The X values used in the training data. Default None
        
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    xx , yy = test_mesh #NxN, NxN
    #Assert sattements
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert isinstance(title, str)==True, "Title must be a string" 
    
    #Set plot details
    #Set contour details
    cs = plt.contourf(xx, yy,z, levels = 1000, cmap = "jet")
    if np.amax(z) < 1e-1 or np.amax(z) > 1000:
        cbar = plt.colorbar(cs, format='%.2e')
    else:
        cbar = plt.colorbar(cs, format = '%2.2f')
    #Set line contour details    
    cs2 = plt.contour(cs, levels=cs.levels[::40], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
    
    #Plot training data if given
    if str(X_train) != "None":
        plt.scatter(X_train[:,0], X_train[:,1], color = "goldenrod", label = "Training", marker = "o")
    
    #Plot all minima and only label first instance
    for i in range(len(minima)):
        if i == 0:
            plt.scatter(minima[i,0], minima[i,1], color="black", label = "Minima", s=25, marker = (5,1))
        else:
            plt.scatter(minima[i,0], minima[i,1], color="black", s=25, marker = (5,1))
    
    #Plot all saddle and only label first instance
    for j in range(len(saddle)):
        if j == 0:
            plt.scatter(saddle[j,0], saddle[j,1], color="white", label = "Saddle", s=25, marker = "X", edgecolor='k')
        else:
            plt.scatter(saddle[j,0], saddle[j,1], color="white", s=25, marker = "X", edgecolor='k')
       
    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')    
    
    #Plots grid and legend
#     plt.grid()
    legend(loc='upper right', bbox_to_anchor=(-0.1, 1))

    #Creates axis labels and title
    plt.xlabel('$x_1$',weight='bold')
    plt.ylabel('$x_2$',weight='bold')
    plt.xlim((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy),np.amax(yy)))
    plt.title("Muller Potential "+title, weight='bold',fontsize=16)
           
    return plt.show() 