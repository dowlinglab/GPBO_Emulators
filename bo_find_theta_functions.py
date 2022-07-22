import numpy as np
import math
from scipy.stats import norm
import torch
import csv
import gpytorch
import scipy.optimize as optimize
import pandas as pd
import os
import Tasmanian

# from bo_initialize_functions import LHS_Design
# from bo_initialize_functions import calc_y_exp
# from bo_initialize_functions import create_sse_data
# from bo_initialize_functions import gen_y_Theta_GP
# from bo_initialize_functions import create_y_data
# from bo_initialize_functions import test_train_split
# from bo_initialize_functions import find_train_doc_path
# from bo_initialize_functions import explore_parameter

# from bo_GP_functions import ExactGPModel
# from bo_GP_functions import train_GP_model 
# from bo_GP_functions import calc_GP_outputs 

# from bo_basic_ei_functions import calc_ei_basic
# from bo_basic_ei_functions import eval_GP_basic_tot

# from bo_emul_ei_functions import eval_GP_emulator_BE
# from bo_emul_ei_functions import calc_ei_emulator
# from bo_emul_ei_functions import get_sparse_grids
# from bo_emul_ei_functions import eval_GP_sparse_grid
# from bo_emul_ei_functions import eval_GP_emulator_tot

# from bo_plotters import value_plotter
# from bo_plotters import plot_org_train
# from bo_plotters import plot_xy
# from bo_plotters import plot_Theta
# from bo_plotters import plot_obj
# from bo_plotters import plot_obj_abs_min
# from bo_plotters import plot_3GP_performance

def find_opt_and_best_arg(theta_mesh, sse, ei):
    """
    Finds the Theta value where min(sse) or min(-ei) is true using argmax and argmin
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
    
    Returns:
    --------
        Theta_Opt_GP: ndarray, The point where the objective function is minimized in theta_mesh
        Theta_Best: ndarray, The point where the ei is maximized in theta_mesh
    """
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    
    #Point that the GP thinks is best has the lowest SSE
    #Find point in sse matrix where sse is lowest (argmin(SSE))
    argmin = np.array(np.where(np.isclose(sse, np.amin(sse),atol=np.amin(sse)*1e-6)==True))
    
    #ensures that only one point is used if multiple points yield a minimum
    if len(argmin[0]) != 1:
        argmin = np.array([[argmin[0,1]],[argmin[1,1]]])
    
    #Find theta value corresponding to argmin(SSE)
    Theta_1_Opt = float(theta1_mesh[argmin[0],argmin[1]])
    Theta_2_Opt = float(theta2_mesh[argmin[0],argmin[1]])
    Theta_Opt_GP = np.array((Theta_1_Opt,Theta_2_Opt))
    
    #calculates best theta value
    #Find point in ei matrix where ei is highest (argmax(EI))
    argmax = np.array(np.where(np.isclose(ei, np.amax(ei),atol=np.amax(ei)*1e-6)==True))

    #ensures that only one point is used if multiple points yield a maximum
    if len(argmax[0]) != 1:
        argmax = np.array([[argmax[0,1]],[argmax[1,1]]])
        
    #Find theta value corresponding to argmax(EI)
    Theta_1_Best = float(theta1_mesh[argmax[0],argmax[1]])
    Theta_2_Best = float(theta2_mesh[argmax[0],argmax[1]])
    Theta_Best = np.array((Theta_1_Best,Theta_2_Best))  
    
    return Theta_Best, Theta_Opt_GP

##FOR USE WITH SCIPY##################################################################
def eval_GP_scipy(theta_guess, train_sse, Xexp,Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias=0.0, ei_sse_choice = "ei", verbose = False):
    """ 
    Calculates either -ei or sse (a function to be minimized). To be used in calculating best and optimal parameter sets.
    Parameters
    ----------
        theta_guess: ndarray (1xp), The theta value that will be guessed to optimize 
        train_sse: ndarray (1 x t), Training data for sse
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        emulator: True/False: Determines whether the GP is a property emulator of error emulator
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        explore_bias: float, Exploration parameter used for calculating 2-Input GP expected improvement
        ei_sse_choice: "neg_ei" or "sse" - Choose which one to optimize
        verbose: True/False - Determines verboseness of output
    
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
    
    #Separates meshgrid
    q = len(theta_guess)
    p = theta_mesh.shape[1]
    n = len(Xexp)
    
    theta1_guess = theta_guess[0]
    theta2_guess = theta_guess[1]

    #Evaluate a point with the GP and save values for GP mean and var
    if emulator == False:
        point = [theta1_guess,theta2_guess]
        eval_point = np.array([point])
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_sse = GP_Outputs[3].numpy()[0] #1xn 
        model_variance= GP_Outputs[1].numpy()[0] #1xn

        #Calculate best error and sse
        #Does the objective function change this? No - As long as they're both logs this will work
        best_error = max(-train_sse) #Negative sign because -max(-train_sse) = min(train_sse)
        sse = model_sse
            #Calculate ei. If statement depends whether ei is the only thing returned by calc_ei_basic function
        if verbose == True:
            ei = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[0]
        else:
            ei = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
    
    else:
        ei = 0
        sse = 0
        best_error = eval_GP_emulator_BE(Xexp,Yexp, theta_mesh)
        GP_mean = np.zeros(n)
        GP_stdev = np.zeros(n)
        for k in range(n):
            #Caclulate EI for each value n given the best error
            point = [theta1_guess,theta2_guess,Xexp[k]]
            eval_point = np.array([point])
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_mean = GP_Outputs[3].numpy()[0] #1xn
            model_variance= GP_Outputs[1].numpy()[0] #1xn
            
            GP_mean[k] = model_mean
            GP_stdev[k] = np.sqrt(model_variance) 
            sse += (model_mean - Yexp[k])**2

            if sparse_grid == False:
                #Compute EI w/ approximation
                ei += calc_ei_emulator(best_error, model_mean, model_variance, Yexp[k], explore_bias)
           
        if sparse_grid == True:
            #Compute EI using sparse grid
            ei = eval_GP_sparse_grid(Xexp, Yexp, theta_mesh, GP_mean, GP_stdev, best_error)
                
            
    #Return either -ei or sse as a minimize objective function
    if ei_sse_choice == "neg_ei":
#         print("EI chosen")
        return -ei #Because we want to maximize EI and scipy.optimize is a minimizer by default
    else:
#         print("sse chosen")
        return sse #We want to minimize sse or ln(sse)

def find_opt_best_scipy(Xexp, Yexp, theta_mesh, train_y,theta0_b,theta0_o,sse,ei,model,likelihood,explore_bias,emulator,sparse_grid,obj):
    """
    Finds the Theta value where min(sse) or min(-ei) is true using scipy.minimize and the L-BFGS-B method
    
    Parameters:
    -----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_y: tensor or ndarray, The training y data
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
    
    #Set theta meshes and bounds
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    bnds = [[np.amin(theta1_mesh), np.amax(theta1_mesh)], [np.amin(theta2_mesh), np.amax(theta2_mesh)]]
    
    #Use L-BFGS Method with scipy.minimize to find theta_opt and theta_best
    ei_sse_choice1 ="neg_ei"
    ei_sse_choice2 = "sse"
    
    #Set arguments and calculate best and optimal solutions
    argmts_best = ((train_y, Xexp, Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias, ei_sse_choice1))
    argmts_opt = ((train_y, Xexp, Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias, ei_sse_choice2))
    Best_Solution = optimize.minimize(eval_GP_scipy, theta0_b,bounds=bnds,method = "L-BFGS-B",args=argmts_best)
    Opt_Solution = optimize.minimize(eval_GP_scipy, theta0_o,bounds=bnds,method = "L-BFGS-B",args=argmts_opt)
    
    #save best and optimal values
    theta_b = Best_Solution.x
    theta_o = Opt_Solution.x  
    
    return theta_b, theta_o
