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
import pandas as pd
import os
import Tasmanian

from bo_functions_generic import LHS_Design, calc_y_exp, calc_muller, create_sse_data, create_y_data, set_ep, gen_y_Theta_GP, test_train_split, find_train_doc_path, ExactGPModel, train_GP_model, calc_GP_outputs, explore_parameter, ei_approx_ln_term, calc_ei_emulator, eval_GP_emulator_BE, get_sparse_grids, eval_GP_sparse_grid, calc_ei_basic, train_test_plot_preparation

from CS2_bo_plotters import value_plotter
# from CS2_bo_plotters import plot_org_train
from CS2_bo_plotters import plot_xy
from CS2_bo_plotters import plot_Theta
from CS2_bo_plotters import plot_obj
from CS2_bo_plotters import plot_obj_abs_min
from CS2_bo_plotters import plot_3GP_performance
from CS2_bo_plotters import plot_sep_fact_min
from CS2_bo_plotters import save_fig, save_csv, path_name

def optimize_theta_set(Xexp, Yexp, theta_set, true_model_coefficients, train_y, train_p, sse, ei, model, likelihood, explore_bias, emulator, sparse_grid, verbose, obj):
    """
    Finds the lowest sse and highest EI parameter sets using scipy
    
    Parameters:
    -----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_set: ndarray (n x p), sets of Theta values
        train_y: ndarray, The output training data
        train_p: tensor or ndarray, The training parameter space data
        sse: ndarray, The SSE of the model 
        ei: ndarray, the expected improvement of the GP model
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        explore_bias: float, the numerical bias towards exploration, zero is the default
        emulator: True/False: Determines whether the GP is a property emulator of error emulator
        sparse_grid: True/False: Determines whether an assumption or sparse grid method is used
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
   Returns:
   --------
       theta_b: ndarray, The point where the objective function is minimized in theta space
       theta_o: ndarray, The point where the ei is maximized in theta space   
    """
    #Could modify to chec every point
    theta0_b, theta0_o = find_opt_and_best_arg(theta_set, sse, ei, train_p)
    theta_b, theta_o = find_opt_best_scipy(Xexp, Yexp, theta_set, true_model_coefficients, train_y,train_p, theta0_b,theta0_o,sse,ei,model,likelihood,explore_bias,emulator,sparse_grid,verbose,obj)
    return theta_b, theta_o

def eval_all_theta_pairs(dimensions, theta_set, n_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = 1):
    """
    Evaluates all combinations of theta pairs to make heat maps
    
    Parameters:
    -----------
        dimensions: int, Number of parameters to regress
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        theta_opt: The optimal set of theta_parameters
        n_points:, int, number of meshgrid points/ parameter or number of LHS samples
        Theta_True: ndarray, A 2x1 containing the true input parameters
        theta_o: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        theta_b: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        train_p: tensor or ndarray, The training parameter space data
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep0: float, float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_fig: True/False, Determines whether figures will be saved
        bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run: int or None, The iteration of the number of times new training points have been picked
        BO_iters: int, total number of BO iterations
        tot_runs: int, total number of runs
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, Number of total data points to use
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP.
    Returns:
    --------
        None - Saves graphs and CSVs     
        
    """
    dim_list = np.linspace(0,dimensions-1,dimensions)
    mesh_combos = np.array(list(combinations(dim_list, 2)), dtype = int)
    for i in range(len(mesh_combos)):
        indecies = mesh_combos[i]
        eval_GP_over_grid(theta_set, indecies, n_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)
    return
    
def eval_GP_over_grid(theta_set_org, indecies, n_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact):
    """
    Evaluates all combinations of theta pairs to make heat maps. Makes heat maps and saves them
    
    Parameters:
    -----------
        dimensions: int, Number of parameters to regress
        indecies: indecies of theta_set that will be changed to create meshgrids and heat maps
        theta_set_org: ndarray (num_LHS_points x dimensions), list of theta combinations
        n_points:, int, number of meshgrid points/ parameter or number of LHS samples
        Theta_True: ndarray, A 2x1 containing the true input parameters
        theta_o: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        theta_b: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        train_p: tensor or ndarray, The training parameter space data
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep0: float, float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_fig: True/False, Determines whether figures will be saved
        bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run: int or None, The iteration of the number of times new training points have been picked
        BO_iters: int, total number of BO iterations
        tot_runs: int, total number of runs
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, Number of total data points to use
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
    Returns:
    --------
        None - Saves graphs and CSVs     
        
    """
    
    #Generate meshgrid and theta_set from meshgrid
    Theta1_lin = np.linspace(np.min(theta_set_org[:,indecies[0]]),np.max(theta_set_org[:,indecies[0]]), n_points)
    Theta2_lin = np.linspace(np.min(theta_set_org[:,indecies[1]]),np.max(theta_set_org[:,indecies[1]]), n_points)
    theta_mesh = np.array(np.meshgrid(Theta1_lin, Theta2_lin)) 
    xx,yy = theta_mesh
#     print(xx.shape)
    #Not sure if this is right
    theta_set = theta_mesh.T.reshape((-1,2))

    eval_components = eval_GP(theta_set, train_y, explore_bias, Xexp, Yexp, Theta_True, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj)
#     print(eval_components)

    #Determines whether debugging parameters are saved for 2 Input GP       
    if verbose == True and emulator == False:
        ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
        list_of_plot_variables = [ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF]
    else:
        ei,sse,var,stdev,best_error = eval_components
        list_of_plot_variables = [ei,sse,var,stdev,best_error]
    
    for i in range(len(list_of_plot_variables)):
        try:
            list_of_plot_variables[i] = list_of_plot_variables[i].reshape((n_points,-1)).T
        except:
            list_of_plot_variables[i] = list_of_plot_variables[i]

            #Prints figures if more than 1 BO iter is happening
    if emulator == False:
        titles = ['E(I(\\theta))','log(e(\\theta))','\sigma^2','\sigma','Best_Error','z','EI_term_1','EI_term_2','CDF','PDF']  
        titles_save = ["EI","ln(SSE)","Var","StDev","Best_Error","z","ei_term_1","ei_term_2","CDF","PDF"] 
    else:
        titles = ['E(I(\\theta))','log(e(\\theta))','\sigma^2', '\sigma', 'Best_Error']  
        titles_save = ["EI","ln(SSE)","Var","StDev","Best_Error"] 

    #Plot and save figures for all figrues for EI and SSE
    value_plotter(theta_mesh, list_of_plot_variables[0], Theta_True, theta_o, theta_b, train_p, titles[0],titles_save[0], obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)

    #Ensure that a plot of SSE (and never ln(SSE)) is drawn
    if obj == "LN_obj" and emulator == False:
        plot_sse = list_of_plot_variables[1]
    else:
        plot_sse = np.log(list_of_plot_variables[1])

    value_plotter(theta_mesh, plot_sse, Theta_True, theta_o, theta_b, train_p, titles[1], titles_save[1], obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)

    #Save other figures
    for j in range(len(list_of_plot_variables)-2):
        component = list_of_plot_variables[j+2]
        title = titles[j+2]
        title_save = titles_save[j+2]
        try:
            value_plotter(theta_mesh, component, Theta_True, theta_o, theta_b, train_p, title, title_save, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, bo_iter, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)
        except:
            Best_Error_Found = np.round(list_of_plot_variables[j+2],4)
            if verbose == True:
                print("Best Error is:", Best_Error_Found)

    return

def eval_GP_emulator_set(Xexp, Yexp, theta_set, true_model_coefficients, model, likelihood, sparse_grid, explore_bias = 0.0, verbose = False, train_p = None, obj = "obj"):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_set: ndarray (n x p), sets of Theta values
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        sparse_grid: True/False: Determines whether an assumption or sparse grid method is used
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: bool, Determines whether output is verbose
        train_p: tensor or ndarray, The training parameter space data
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
        optimize: bool, Determines whether scipy will be used to find the best point for 
    
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
    EI = np.zeros(len_set) #(p1 x p2) 
    SSE_var_GP = np.zeros(len_set)
    SSE_stdev_GP = np.zeros(len_set)
    SSE = np.zeros(len_set)
    
    ##Calculate Best Error
    # Loop over theta 1
    for i in range(len_set):
        best_error = eval_GP_emulator_BE(Xexp, Yexp, train_p, true_model_coefficients, obj = "obj")
        GP_mean = np.zeros(n)
        GP_var = np.zeros(n)
        
        ##Calculate Values
        for k in range(n):
            #Caclulate EI for each value n given the best error
            point = list(theta_set[i])
            point.append(Xexp[k])
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

            if sparse_grid == False:
                #Compute EI w/ approximation
                EI_temp = calc_ei_emulator(best_error, model_mean, model_variance, Yexp[k], explore_bias, obj)
#                     print(EI_temp)
                EI[i] += EI_temp

        GP_stdev = np.sqrt(GP_var)

        #Get testing values for integration
#             if i == j == 0:
#                 print("Model mean", GP_mean)
#                 print("Model stdev", GP_stdev)
#                 print("EP", explore_bias)
#                 print("best error", best_error)
#                 print("y_target", Yexp)

        if sparse_grid == True:
            #Compute EI using eparse grid (Note theta_mesh not actually needed here)
            EI[i] = eval_GP_sparse_grid(Xexp, Yexp, GP_mean, GP_stdev, best_error, explore_bias, verbose)

    if verbose == True:
        print(EI)
    
    return EI, SSE, SSE_var_GP, SSE_stdev_GP, best_error

def eval_GP_basic_set(theta_set, train_sse, model, likelihood, explore_bias=0.0, verbose = False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        theta_mesh: ndarray (n x p), sets of Theta values
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
    ei = np.zeros(len_set)
    sse = np.zeros(len_set)
    var = np.zeros(len_set)
    stdev = np.zeros(len_set)

    if verbose == True:
        z_term = np.zeros(len_set)
        ei_term_1 = np.zeros(len_set)
        ei_term_2 = np.zeros(len_set)
        CDF = np.zeros(len_set)
        PDF = np.zeros(len_set)
        
        
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

        #Negative sign because -max(-train_sse) = min(train_sse)
        #Print and save certain values based on verboseness
        if verbose == True:
            out1, out2, out3, out4, out5, out6 = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)
#                 out1, out2, out3, out4, out5, out6 = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
            ei[i] = out1
            z_term[i] = out2
            ei_term_1[i] = out3
            ei_term_2[i] = out4
            CDF[i] = out5
            PDF[i] = out6

        else:
            ei[i] = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)

    if verbose == True:
        return ei, sse, var, stdev, best_error, z_term, ei_term_1, ei_term_2, CDF, PDF
    else:
        return ei, sse, var, stdev, best_error #Prints just the value
#         return ei, sse, var, stdev, f_best

def find_opt_and_best_arg(theta_set, sse, ei, train_p): #Not quite sure how to fix setting of points yet
    """
    Finds the Theta value where min(sse) or min(-ei) is true using argmax and argmin
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
        train_p: tensor or ndarray, The training parameter space data
    
    Returns:
    --------
        Theta_Opt_GP: ndarray, The point where the objective function is minimized in theta_mesh
        Theta_Best: ndarray, The point where the ei is maximized in theta_mesh
    """    
    #Point that the GP thinks is best has the lowest SSE
    #Find point in sse matrix where sse is lowest (argmin(SSE))
    len_set, q = theta_set.shape
    argmin = np.array(np.where(np.isclose(sse, np.amin(sse),rtol=abs(np.amin(sse)*1e-6))==True))[0]
    
    #ensures that only one point is used if multiple points yield a minimum
    
    if len(argmin) > 1:
        rand_ind = np.random.randint(np.max(argmin.shape)) #Chooses a random point with the minimum value
        argmin = argmin[rand_ind]
#     print(argmin)
    #Find theta value corresponding to argmin(SSE)
    
    #Initialize Theta_Opt_GP
    Theta_Opt_GP = theta_set[argmin]
    Theta_Opt_GP = Theta_Opt_GP[0:q]
#     Theta_Opt_GP = np.array([Theta_Opt_GP[0:q]])
    
    #calculates best theta value
    #Find point in ei matrix where ei is highest (argmax(EI))
    argmax = np.array(np.where(np.isclose(ei, np.amax(ei),rtol=abs(np.amax(ei)*1e-6))==True))[0]

    #ensures that only one point is used if multiple points yield a maximum
    if len(argmax) > 1:
        argmax = argmax_multiple(argmax, train_p, theta_set)
            
    #Find theta value corresponding to argmax(EI)
    #Initialize Theta_Best
    Theta_Best = theta_set[argmax]
    Theta_Best = Theta_Best[0:q]
    
#     print(Theta_Best)
#     print(Theta_Opt_GP)
    return Theta_Best, Theta_Opt_GP

def argmax_multiple(argmax, train_p, theta_set): #not sure how to fix setting of points here either
    """
    Finds the best ei point argument when more than one point has the maximum ei
    
    Parameters:
    -----------
        argmax: ndarray, The indecies of all parameters that have the maximum ei
        train_p: tensor or ndarray, The training parameter space data
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        
    Returns:
    --------
        argmax_best: ndarray, The indecies of the parameters that have the maximum ei that is furthest from the rest of the training points
    """
    #Initialize max distance and theta arrays
    max_distance_sq = 0
    len_set, q = theta_set.shape[0], theta_set.shape[1]
    
    #Initialize 
    argmax_best = np.zeros(1)
    #Only use this algorithm when >1 points have the max ei
    #Create avg x y pt for training data for only values of parameters to be regressed
#     train_T12_avg = np.average(train_p, axis =0)
    train_T12_avg = np.average(train_p, axis =0)
    train_T12_avg = train_T12_avg[0:q] #Only save the values corresponding to parameters
#     print(q)
#     print(train_T12_avg)
#     print(theta_set[0])

    #Check each point in argmax with all training points and find max distance
    #Loop over all coord points
    for i in range(len(argmax)):
        #Find theta value corresponding to argmax(EI)
        point = argmax[i]
        
        #Initialize Theta_Arr
        Theta_Arr = theta_set[i]

        #Calculate Distance
        distance_sq = np.sum((train_T12_avg - Theta_Arr)**2)

        #Set distance to max distance if it is applicable. At the end of the loop, argmax will be the point with the greatest distance.
        if distance_sq > max_distance_sq:
            max_distance_sq = distance_sq
            argmax_best = np.array([point])
            
    return argmax_best
             
##FOR USE WITH SCIPY##################################################################
def eval_GP_scipy(theta_guess, train_sse, train_p, Xexp,Yexp, theta_set, model, likelihood, emulator, sparse_grid, true_model_coefficients, explore_bias=0.0, ei_sse_choice = "ei", verbose = False, obj = "obj"):
    """ 
    Calculates either -ei or sse (a function to be minimized). To be used in calculating best and optimal parameter sets.
    Parameters
    ----------
        theta_guess: ndarray (1xp), The theta value that will be guessed to optimize 
        train_sse: ndarray (1 x t), Training data for sse
        train_p: tensor or ndarray, The training parameter space data
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_set: ndarray (len_set x q), array of Theta values
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        emulator: True/False: Determines whether the GP is a property emulator of error emulator
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        true_model_coefficients: ndarray, True values of Muller potential constants
        explore_bias: float, Exploration parameter used for calculating 2-Input GP expected improvement
        ei_sse_choice: "neg_ei" or "sse" - Choose which one to optimize
        verbose: True/False - Determines verboseness of output
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
    
    Returns
    -------
        -ei: ndarray, the negative expected improvement of the GP model
        OR
        sse: ndarray, the sse/ln(sse) of the GP model
        
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert ei_sse_choice == "neg_ei" or ei_sse_choice == "sse", "ei_sse_choice must be string 'ei' or 'sse'"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Back out parameters
    len_set, q = theta_set.shape[0], theta_set.shape[1]#Infer from something else
    n = len(Xexp)

    #Evaluate a point with the GP and save values for GP mean and var
    if emulator == False:
#         point = [theta_guess]
        point = theta_guess
        eval_point = np.array([point])
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_sse = GP_Outputs[3].numpy()[0] #1xn 
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn

        #Calculate best error and sse
        #Does the objective function change this? No - As long as they're both logs this will work
        best_error = -max(-train_sse) #Negative sign because -max(-train_sse) = min(train_sse)
        sse = model_sse
            #Calculate ei. If statement depends whether ei is the only thing returned by calc_ei_basic function
        if verbose == True:
            ei = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)[0]
        else:
            ei = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)
    
    else:
        ei = 0
        sse = 0
        best_error = eval_GP_emulator_BE(Xexp, Yexp, train_p, true_model_coefficients, obj = "obj")
        GP_mean = np.zeros(n)
        GP_stdev = np.zeros(n)
        for k in range(n):
            #Caclulate EI for each value n given the best error
#             point = [theta_guess,Xexp[k]]
            point = list(theta_guess)
            point.append(Xexp[k])
            point = np.array(point)
            eval_point = np.array([point])
#             point = theta_guess,Xexp[k]
#             eval_point = np.array([point])
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_mean = GP_Outputs[3].numpy()[0] #1xn
            model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
            
            GP_mean[k] = model_mean
            GP_stdev[k] = np.sqrt(model_variance) 
            sse += (model_mean - Yexp[k])**2

            if sparse_grid == False:
                #Compute EI w/ approximation
                ei += calc_ei_emulator(best_error, model_mean, model_variance, Yexp[k], explore_bias, obj)
           
        if sparse_grid == True:
            #Compute EI using sparse grid #Note theta_mesh not actually needed here
            ei = eval_GP_sparse_grid(Xexp, Yexp, GP_mean, GP_stdev, best_error, explore_bias)
                
            
    #Return either -ei or sse as a minimize objective function
    if ei_sse_choice == "neg_ei":
#         print("EI chosen")
        return -ei #Because we want to maximize EI and scipy.optimize is a minimizer by default
    else:
#         print("sse chosen")
        return sse #We want to minimize sse or ln(sse)

def find_opt_best_scipy(Xexp, Yexp, theta_set, true_model_coefficients, train_y,train_p, theta0_b,theta0_o,sse,ei,model,likelihood,explore_bias,emulator,sparse_grid,verbose,obj):
    """
    Finds the Theta value where min(sse) or min(-ei) is true using scipy.minimize and the L-BFGS-B method
    
    Parameters:
    -----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (len_set x q), array of Theta values 
        train_y: tensor or ndarray, The training y data
        train_p: tensor or ndarray, The training parameter space data
        theta0_b: Initial guess of the Theta value where ei is maximized
        theta0_o: Initial guess of the Theta value where sse is minimized
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        emulator: True/False: Determines whether the GP is a property emulator of error emulator
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        obj: ob or LN_obj: Determines which objective function is used for the 2 input GP
    
    Returns:
    --------
        theta_b: ndarray, The point where the objective function is minimized in theta_mesh
        theta_o: ndarray, The point where the ei is maximized in theta_mesh
    """
    #Assert statements to ensure no bugs
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(theta0_b) == len(theta0_o), "Initial guesses must be the same length."
    
    # bound = np.array([0,1])
    # bounds = (np.repeat(x,8))
    # bnds = bounds.reshape(-1,8).T
    len_set, q = theta_set.shape[0], theta_set.shape[1]
    bnds = np.zeros((q,2)) #Upper and lower bound for each dimension
    
    for i in range(q):
        bnds[i] = np.amin(theta_set[:,i]), np.amax(theta_set[:,i])
    
#     print(bnds)
        
#     bnds = [[np.amin(theta1_mesh), np.amax(theta1_mesh)], [np.amin(theta2_mesh), np.amax(theta2_mesh)]]
    
    #Use L-BFGS Method with scipy.minimize to find theta_opt and theta_best
    ei_sse_choice1 ="neg_ei"
    ei_sse_choice2 = "sse"
    
    #Set arguments and calculate best and optimal solutions
    #remove theta_mesh from these eventually
    argmts_best = ((train_y, train_p, Xexp, Yexp, theta_set, model, likelihood, emulator, sparse_grid, true_model_coefficients, explore_bias, ei_sse_choice1, verbose, obj))
    argmts_opt =  ((train_y, train_p, Xexp, Yexp, theta_set, model, likelihood, emulator, sparse_grid, true_model_coefficients, explore_bias, ei_sse_choice2, verbose, obj))

#     print(argmts_best)
    Best_Solution = optimize.minimize(eval_GP_scipy, theta0_b,bounds=bnds, method = "L-BFGS-B", args=argmts_best)
    Opt_Solution = optimize.minimize(eval_GP_scipy, theta0_o,bounds=bnds,method = "L-BFGS-B",args=argmts_opt)
    
    #save best and optimal values
    theta_b = Best_Solution.x
    theta_o = Opt_Solution.x  
    
    return theta_b, theta_o

# def eval_GP(theta_mesh, train_y, explore_bias, Xexp, Yexp, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale):  
def eval_GP(theta_set, train_y, explore_bias, Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p = None, obj = "obj"):
    """
    Evaluates GP
    
    Parameters:
    -----------
        theta_set: ndarray (num_LHS_points x dimensions), list of theta combinations
        train_y: tensor or ndarray, The training y data
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
        emulator: True/False: Determiens whether GP is an emulator of the function
        sparse_grd: True/False: Determines whether an assumption or sparse grid is used
        set_lengthscale: float/None: Determines whether Hyperparameter values will be set
        train_p: tensor or ndarray, The training parameter space data
        obj: ob or LN_obj: Determines which objective function is used for the 2 input GP
    
    Returns:
    --------
        eval_components: ndarray, The componenets evaluate by the GP. ei, sse, var, stdev, f_best, (z_term, ei_term_1, ei_term_2, CDF, PDF)
    """
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    ##Set Hyperparameters to 1
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
        eval_components = eval_GP_basic_set(theta_set, train_y, model, likelihood, explore_bias, verbose)
    else:
#         eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose)
        eval_components = eval_GP_emulator_set(Xexp, Yexp, theta_set, true_model_coefficients, model, likelihood, sparse_grid, explore_bias, verbose, train_p, obj)
    
    return eval_components

def bo_iter(BO_iters,train_p,train_y,theta_set,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, run, sparse_grid, emulator, set_lengthscale, verbose = False,save_fig=False, tot_runs = 1, DateTime=None, test_p = None, sep_fact = 0.8, LHS = False):
    """
    Performs BO iterations
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iteratiosn
        train_p: tensor or ndarray, The training parameter space data
        train_y: tensor or ndarray, The training y data
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        run: int, The iteration of the number of times new training points have been picked
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        emulator: True/False, Determines if GP will model the function or the function error
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: True/False, Determines whether figures will be saved
        tot_runs: The total number of runs to perform
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        test_p: None, tensor, or ndarray, The testing parameter space data. Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        
        
    Returns:
    --------
        All_Theta_Best: ndarray, Array of all Best Theta values (as determined by max(ei)) for each iteration 
        All_Theta_Opt: ndarray, Array of all Optimal Theta values (as determined by min(sse)) for each iteration
        All_SSE: ndarray, Array of all minimum SSE values (as determined by min(sse)) for each iteration
        All_SSE_abs_min: ndarray, Array of the absolute minimum SSE values (as determined by min(sse)) at each iteration 
        Total_BO_iters: int, The number of BO iteration actually completed    
    """
    #Assert Statments
    assert all(isinstance(i, int) for i in [BO_iters, train_iter]), "BO_iters and train_iter must be integers"
    assert len(train_p) == len(train_y), "Training data must be the same length"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    assert emulator==True or emulator==False, "Verbose must be True/False"
    
    #Find parameters
    m = Xexp[0].size #Dimensions of X
    n = len(Xexp) #Length of experimental data
    q = len(Theta_True) #Number of parameters to regress
#     p = theta_mesh.shape[1] #Number of points to evaluate the GP at in any dimension of q
    t = int(len(train_p)) + int(len(test_p)) #Original length of all data
    if LHS == True:
        data_points = len(theta_set)
    else:
        data_points = int(np.sqrt(len(theta_set)))
#     print(data_points)
    ep0 = explore_bias
    
    #Set arrays to track theta_best, theta_opt, and SSE for every BO iteration
    All_Theta_Best = np.zeros((BO_iters,q)) 
    All_Theta_abs_Opt = np.zeros((BO_iters,q))
    All_Theta_Opt = np.zeros((BO_iters,q)) 
    All_SSE = np.zeros(BO_iters) #Will save ln(SSE) values
    All_SSE_abs_min = np.zeros(BO_iters) #Will save ln(SSE) values  
    All_Max_EI = np.zeros(BO_iters) #Used in stopping criteria
    Total_BO_iters = BO_iters

    #Ensures GP will take correct # of inputs
    if emulator == True:
        GP_inputs = q+m
        assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
    else:
        GP_inputs = q
        assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
    
    mean_of_var = 0
    best_error_num = 0
    ep_init = explore_bias
    
    #Loop over # of BO iterations
    for i in range(BO_iters):
        #Converts numpy arrays to tensors
        if torch.is_tensor(train_p) != True:
            train_p = torch.from_numpy(train_p)
        if torch.is_tensor(train_y) != True:
            train_y = torch.from_numpy(train_y)
        if torch.is_tensor(test_p) != True:
            test_p = torch.from_numpy(test_p)
            
        #Redefine likelihood and model based on new training data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=False)
        
        #Set Exploration parameter
#         explore_bias = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Constant") #Defaulting to exp method
        explore_bias = ep_init #Sets ep to the multiplicative scaler between 0.1 and 1
        
        #Evaluate GP to find sse and ei for optimization step
        eval_components = eval_GP(theta_set, train_y, explore_bias,Xexp, Yexp, Theta_True, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj)

        #Determines whether debugging parameters are saved for 2 Input GP       
        if verbose == True and emulator == False:
            ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
        else:
            ei,sse,var,stdev,best_error = eval_components

        #solve for opt and best based on theta_set
        theta_b, theta_o = optimize_theta_set(Xexp, Yexp, theta_set, Theta_True, train_y, train_p, sse, ei, model, likelihood, explore_bias, emulator, sparse_grid, verbose, obj)
        
        #Evaluate GP for best EI theta set
        eval_components = eval_GP(np.array([theta_b]), train_y, explore_bias,Xexp, Yexp, Theta_True, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj)

        #Determines whether debugging parameters are saved for 2 Input GP       
        if verbose == True and emulator == False:
            ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
        else:
            ei,sse,var,stdev,best_error = eval_components
        
        #Save Figures
#         if save_fig == True:
        eval_all_theta_pairs(q, theta_set, data_points, Theta_True, Xexp, Yexp, theta_o, theta_b, train_p, train_y, model, likelihood, verbose, obj, ep0, explore_bias, emulator, sparse_grid, set_lengthscale, save_fig, i, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)
        
        All_Max_EI[i] = np.max(ei)
        
#         mean_of_var = np.average(var)
# #         print("MOV",mean_of_var)
#         best_error_num = best_error
           
        #Save theta_best and theta_opt values for iteration
        All_Theta_Best[i], All_Theta_Opt[i] = theta_b, theta_o
        
        #Calculate values of y given the GP optimal theta values
        y_GP_Opt = gen_y_Theta_GP(Xexp, theta_o)
        
        #Calculate GP SSE and save value
        ln_error_mag = np.log(np.sum((y_GP_Opt-Yexp)**2)) #Should SSE be calculated like this or should we use the GP approximation?
        
#       sse_opt = eval_GP_scipy(theta_o, train_p, Xexp,Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias, ei_sse_choice = "sse", verbose = False)
#         ln_error_mag = sse_opt

        All_SSE[i] = ln_error_mag
        
        #Save best value of SSE for plotting 
        #Make its own function         
        if i == 0:
            All_SSE_abs_min[i] = ln_error_mag
            All_Theta_abs_Opt[i] = theta_o
            improvement = False
#             All_SSE_abs_min[i] = sse_opt
        else:
            if All_SSE_abs_min[i-1] >= ln_error_mag:
                All_SSE_abs_min[i] = ln_error_mag
                All_Theta_abs_Opt[i] = theta_o
                improvement = True
            else: 
                All_SSE_abs_min[i] = All_SSE_abs_min[i-1]
                All_Theta_abs_Opt[i] = All_Theta_abs_Opt[i-1]
                improvement = False
        
        #Prints certain values at each iteration if verbose is True
        if verbose == True:
            print("BO Iteration = ", i+1)
#             Jas_ep = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Jasrasaria")
#             print("Jasrasaria EP:", Jas_ep)
#             Boy_ep = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Boyle", improvement = improvement)
#             print("Boyle EP:", Boy_ep)
#             Exp_ep = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init)
#             print("Exp EP:", Exp_ep)
#             print("Exploration Bias = ",explore_bias)
            print("Exploration Bias Factor = ",explore_bias)
            print("Scipy Theta Best = ",theta_b)
            print("Scipy Theta Opt = ",theta_o)
            print("EI_max =", np.amax(ei), "\n")
        
        ##Append best values to training data 
        #Convert training data to numpy arrays to allow concatenation to work
        train_p = train_p.numpy() #(q x t)
        train_y = train_y.numpy() #(1 x t)
        
        #Save Training data for this iteration in CSV
        df_list = [train_p, test_p]
        df_list_ends = ["Train_p", "Test_p"]
        fxn = "value_plotter"
        title_save_TT = "Train_Test_Data"

        for j in range(len(df_list)):
            array_df = pd.DataFrame(df_list[j])
            path_csv = path_name(emulator, explore_bias, sparse_grid, fxn, set_lengthscale, t, obj, i, title_save_TT, run, tot_iter=Total_BO_iters, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[j])
#             print(path_csv)
            save_csv(array_df, path_csv, ext = "csv") #Note: Iter 3 means the training points used in calculation of iter 3
        
        if  i > 0:
            #Change to 1e-7
            if abs(All_Max_EI[i-1]) <= 1e-10 and abs(All_Max_EI[i]) <= 1e-10:
                Total_BO_iters = i+1
                break
        
        #Make this a new function
        if emulator == False:   
            #Call the expensive function and evaluate at Theta_Best
            sse_Best = create_sse_data(q,theta_b, Xexp, Yexp, obj) #(1 x 1)
#             sse_Best = create_sse_data(theta_b, Xexp, Yexp, Constants, obj)
            #Add Theta_Best to train_p and y_best to train_y
            train_p = np.concatenate((train_p, [theta_b]), axis=0) #(q x t)
            train_y = np.concatenate((train_y, [sse_Best]),axis=0) #(1 x t)
            
        else:
            #Loop over experimental data
            for k in range(n):
                Best_Point = theta_b
                Best_Point = np.append(Best_Point, Xexp[k])
                #Create y-value/ experimental data ---- #Should use calc_y_exp correct?
                y_Best = calc_y_exp(theta_b, Xexp[k], noise_std, noise_mean=0,random_seed=6)
                #y_Best = calc_y_exp(Constants_True, Xexp[k], noise_std)
                train_p = np.append(train_p, [Best_Point], axis=0) #(q x t)
                train_y = np.append(train_y, [y_Best]) #(1 x t)
        
        if verbose == True:
            print("Magnitude of ln(SSE) given Theta_Opt = ",theta_o, "is", "{:.4e}".format(ln_error_mag))
    
    #Plots a single line of objective/theta values vs BO iteration if there are no runs
    if tot_runs == 1 and verbose == True:
        #Plot X vs Y for Yexp and Y_GP
        X_line = np.linspace(np.min(Xexp),np.max(Xexp),100)
        y_true = calc_y_exp(Theta_True, X_line, noise_std = noise_std, noise_mean=0)
        #y_true = calc_y_exp(Constants_True, X_line, noise_std)
        y_GP_Opt_100 = gen_y_Theta_GP(X_line, theta_o)   
        plot_xy(X_line,Xexp, Yexp, y_GP_Opt,y_GP_Opt_100,y_true)
              
    return All_Theta_Best, All_Theta_Opt, All_SSE, All_SSE_abs_min, Total_BO_iters

def bo_iter_w_runs(BO_iters,all_data_doc,t,theta_set,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, runs, sparse_grid, emulator,set_lengthscale, verbose = True,save_fig=False, shuffle_seed = None, DateTime=None, sep_fact = 1, LHS = False):
    """
    Performs BO iterations with runs. A run contains of choosing different initial training data.
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iterations
        all_data_doc: csv name as a string, contains all training data for GP
        t: int, Number of total points to use
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        runs: int, The number of times to choose new training points
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        emulator: True/False, Determines if GP will model the function or the function error
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: True/False, Determines whether figures will be saved
        shuffle_seed, int, number of seed for shuffling training data. Default is None. 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        
    Returns:
    --------
        bo_opt: int, The BO iteration at which the lowest SSE occurs
        run_opt: int, The run at which the lowest SSE occurs
        Theta_Opt_all: ndarray, the theta values/parameter set that maps to the lowest SSE
        SSE_abs_min: float, the absolute minimum SSE found
        Theta_Best_all: ndarray, the theta values/parameter set that maps to the highest EI
    
    """
    #Assert statements
    assert all(isinstance(i, int) for i in [BO_iters, t,runs,train_iter]), "BO_iters, t, runs, and train_iter must be integers"
    assert BO_iters > 0, "Number of BO Iterations must be greater than 0!"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    assert emulator==True or emulator==False, "Verbose must be True/False"
    assert isinstance(runs, int) == True, "Number of runs must be an integer"
    
    #Find constants
    if len(Xexp.shape) >1:
        m = Xexp.shape[0]
        n = Xexp.shape[1]
    else:
        m = 1
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
    Theta_Best_matrix = np.zeros((runs,BO_iters,q))
    SSE_matrix = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    EI_matrix = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    SSE_matrix_abs_min = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    Total_BO_iters_matrix = np.zeros(runs)
    
    #Loop over # runs
    for i in range(runs):
#         print("Run Number: ",i+1)
        if verbose == True or save_fig == False:
            print("Run Number: ",i+1)
        #Note: sep_fact can be used to use less training data points
        train_data, test_data = test_train_split(all_data, runs = runs, sep_fact = sep_fact, shuffle_seed=shuffle_seed)
        if emulator == True:
            train_p = train_data[:,1:(q+m+1)]
            test_p = test_data[:,1:(q+m+1)]
        else:
            train_p = train_data[:,1:(q+1)]
            test_p = test_data[:,1:(q+1)]
            
#         print(len(test_p))
#         print(test_p.shape)
#         print(train_p.shape)
        train_y = train_data[:,-1]
        assert len(train_p) == len(train_y), "Training data must be the same length"
        
        if emulator == True:
            assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
        else:
            assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
        
        #Plot all training data
        #Test this
        train_test_plot_preparation(q, m, theta_set, train_p, test_p, Theta_True, Xexp, emulator, sparse_grid, obj, ep0, set_lengthscale, i, save_fig, BO_iters, runs, DateTime, verbose, sep_fact)

        #Run BO iteration
        BO_results = bo_iter(BO_iters,train_p,train_y,theta_set,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, i, sparse_grid, emulator, set_lengthscale, verbose, save_fig, runs, DateTime, test_p, sep_fact = sep_fact)
        
        #Add all SSE/theta results at each BO iteration for that run
        Theta_Best_matrix[i,:,:] = BO_results[0]
        Theta_Opt_matrix[i,:,:] = BO_results[1]
        SSE_matrix[i,:] = BO_results[2]
        SSE_matrix_abs_min[i] = BO_results[3]
        Total_BO_iters_matrix[i] = BO_results[4]
        
#         print(Theta_Best_matrix)
    #Plot all SSE/theta results for each BO iteration for all runs
    if runs >= 1:
        plot_obj(SSE_matrix, t, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact)
        plot_Theta(Theta_Opt_matrix, Theta_True, t, BO_iters, obj,ep0, emulator, sparse_grid,  set_lengthscale, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact)
        plot_obj_abs_min(SSE_matrix_abs_min, emulator, ep0, sparse_grid, set_lengthscale, t, obj, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact)
    
    #Find point corresponding to absolute minimum SSE and max(-ei) at that point
#     print(SSE_matrix)
    argmin = np.array(np.where(np.isclose(SSE_matrix, np.amin(SSE_matrix),rtol=abs(np.amin(SSE_matrix)*1e-6))==True)) #Use rtol so that graphs match data in matricies
#     print("Argmin 1", argmin)
    #Not sure how to generalize this last part
    
    if len(argmin) > 1:
        rand_ind = np.random.randint(argmin.shape[1]) #Chooses a random point with the minimum value
        argmin = argmin[:,rand_ind]
#     if len(argmin) != q: #How to generalize next line?
#         argmin = np.array([[argmin[0]],[argmin[1]]])
#     print("Argmin 2", argmin)
#     argmin = argmin.reshape(1,-1)[0]
#     print("Argmin 3", argmin)
#     print(Theta_Opt_matrix)
    #Find theta value corresponding to argmin(SSE) and corresponding argmax(ei) at which run and theta value they occur
#     Theta_Best_all = np.array(Theta_Best_matrix[argmin])
#     Theta_Opt_all = np.array(Theta_Opt_matrix[argmin])
    Theta_Best_all = np.array(Theta_Best_matrix[tuple(argmin)+(Ellipsis,)])
    Theta_Opt_all = np.array(Theta_Opt_matrix[tuple(argmin)+(Ellipsis,)])
#     print(Theta_Opt_all)
    SSE_abs_min = np.amin(SSE_matrix)
    run_opt = int(argmin[0]+1)
    bo_opt = int(argmin[1]+1)
    
    return bo_opt, run_opt, Theta_Opt_all, SSE_abs_min, Theta_Best_all, SSE_matrix