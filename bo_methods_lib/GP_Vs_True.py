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

from .bo_functions_generic import train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs
from .CS2_bo_plotters import save_csv, save_fig, plot_xy
    
# from .CS1_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data
from .CS2_create_data import gen_y_Theta_GP, calc_y_exp, create_y_data

###Load data
###Get constants
##Note: X and Y should be 400 points long generated from meshgrid values and calc_y_exp :)
def Compare_GP_True(all_data, X_space, Xexp, Yexp, true_model_coefficients, true_p, obj, Case_Study, skip_param_types = 0, set_lengthscale = None, train_iter = 300, noise_std = 0.1, verbose = False, DateTime = None, save_figure= True):  
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
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_p, train_y, likelihood)
    
    #Train GP
#     print(train_p.dtype, train_y.dtype)
    train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose)

    #Evaluate GP with true parameters over meshgrid X1 X2
    eval_components = eval_GP_x_space(true_p, X_space, train_y, true_model_coefficients, model, likelihood, verbose, set_lengthscale, train_p = train_p, obj = obj, skip_param_types = skip_param_types, noise_std = noise_std, CS = Case_Study)
    GP_mean, GP_stdev, y_sim = eval_components
    
    #Plot true shape
    if Case_Study == 2.2:
        minima = np.array([[-0.558,1.442],
                      [-0.050,0.467],
                      [0.623,0.028]])

        saddle = np.array([[-0.82,0.62],
                      [0.22,0.30]])

        title1 = "- True Value"
        X_mesh = X_space.reshape(p,p,-1).T
        Muller_plotter(X_mesh, y_sim.T, minima, saddle, title1, X_train = X_train)

        #Plot GP shape
        title2 = "- GP Mean"
        Muller_plotter(X_mesh, GP_mean.T, minima, saddle, title2, X_train = X_train)
    elif Case_Study == 1:
        plot_xy(X_space, Xexp, Yexp, None ,GP_mean, y_sim,title = "XY Comparison")
    
    return

def eval_GP_x_space(theta_set, X_space, train_y, true_model_coefficients, model, likelihood, verbose, set_lengthscale, train_p = None, obj = "obj", skip_param_types = 0, noise_std = 0.1, CS = 1):
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
        set_lengthscale: float/None: Determines whether Hyperparameter values will be set
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
    
    #Evaluate GP based on property emulator
    eval_components = eval_GP_emulator_x_space(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types, CS)
    
    return eval_components

    
def eval_GP_emulator_x_space(theta_set, X_space, true_model_coefficients, model, likelihood, skip_param_types=0, CS=1):
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
    
    if m > 1:
        #Turn GP_mean, GP_stdev, and y_sim back into meshgrid form
        GP_stdev = np.array(GP_stdev).reshape((p, p))
        GP_mean = np.array(GP_mean).reshape((p, p))
        y_sim = np.array(y_sim).reshape((p, p))
    
    return GP_mean, GP_stdev, y_sim

def Muller_plotter(test_mesh, z, minima, saddle, title, X_train = None):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        p_GP_Best: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        title: str, A string containing the title of the plot
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
    xx , yy = test_mesh #NxN, NxN
    #Assert sattements
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert isinstance(title, str)==True, "Title must be a string" 
    
    #Set plot details
#     plt.figure(figsize=(8,4))
    cs = plt.contourf(xx, yy,z, levels = 1000, cmap = "jet")
    if np.amax(z) < 1e-1 or np.amax(z) > 1000:
        cbar = plt.colorbar(cs, format='%.2e')
    else:
        cbar = plt.colorbar(cs, format = '%2.2f')
        
    cs2 = plt.contour(cs, levels=cs.levels[::25], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
    
    #plot saddle pts and local minima, only label 1st instance
    if str(X_train) != "None":
        plt.scatter(X_train[:,0], X_train[:,1], color = "brown", label = "Training", marker = "o")
    
    for i in range(len(minima)):
        if i == 0:
            plt.scatter(minima[i,0], minima[i,1], color="black", label = "Minima", s=25, marker = (5,1))
        else:
            plt.scatter(minima[i,0], minima[i,1], color="black", s=25, marker = (5,1))

    for j in range(len(saddle)):
        if j == 0:
            plt.scatter(saddle[j,0], saddle[j,1], color="white", label = "Saddle", s=25, marker = "x")
        else:
            plt.scatter(saddle[j,0], saddle[j,1], color="white", s=25, marker = "x")
       
    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')    
    
    #Plots grid and legend
#     plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('$x_1$',weight='bold')
    plt.ylabel('$x_2$',weight='bold')
    plt.xlim((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy),np.amax(yy)))
    plt.title("Muller Potential "+title, weight='bold',fontsize=16)
           
    return plt.show() 

def path_name_gp_val(emulator, fxn, set_lengthscale, t, obj, Case_Study, DateTime = None, is_figure = True, csv_end = None, plot_axis = None, plot_num = None):
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
    if is_figure == True:
        path_org = path_org + "/Figures"
    else:
        path_org = path_org + "/CSV_Data"
        
    path_end = CS + Emulator + org_TP_str + obj_str + len_scl + plot     

    path = path_org + "/GP_Validation_Figs" + path_end 
        
    if csv_end is not None:
        path = path + csv_end
   
    return path