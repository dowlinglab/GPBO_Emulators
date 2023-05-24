# import numpy as np
# import math
# from scipy.stats import norm
# from scipy import integrate
# import torch
# import csv
# import gpytorch
# import scipy.optimize as optimize
# import itertools
# from itertools import combinations_with_replacement
# from itertools import combinations
# from itertools import permutations
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# import os
# import time
# import Tasmanian

# #Notes: Change line below when changing test problems: 
# # If line 21 is active, the 8D problem is used, if line 22 is active, the 2D problem is used
# # from .CS2_create_data import calc_muller, create_sse_data, create_y_data, calc_y_exp, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point
# from .CS1_create_data import create_sse_data, create_y_data, calc_y_exp, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point

# from .bo_functions_generic import LHS_Design, set_ep, test_train_split, find_train_doc_path, ExactGPModel, train_GP_model, calc_GP_outputs, explore_parameter, ei_approx_ln_term, calc_ei_emulator, get_sparse_grids, eval_GP_sparse_grid, calc_ei_basic, train_test_plot_preparation, clean_1D_arrays, norm_unnorm, train_GP_scikit, define_GP_model

# from .normalize import normalize_x, normalize_p_data, normalize_p_set, normalize_p_true, normalize_constants, normalize_general, normalize_p_bounds

# from .CS2_bo_plotters import value_plotter, plot_xy, plot_Theta, plot_Theta_min, plot_obj, plot_obj_abs_min, plot_3GP_performance, plot_sep_fact_min, save_fig, save_csv, path_name, plot_EI_abs_max, save_misc_data
# from CS2_bo_plotters import plot_org_train
class CaseStudyParameters:
    """
    The base class for any GPBO Case Study and Evaluation
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, cs_name, true_params, true_model_coefficients, param_dict, skip_param_types, ep0, sep_fact, normalize, num_data, LHS_gen_theta, eval_all_pairs, package, noise_std, kernel, set_lenscl, outputscl, retrain_GP, GP_train_iter, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime):
        """
        Parameters
        ----------
        cs_name, str, The name associated with the case study being evaluated
        true_params: ndarray, The array containing the true parameter values for the problem (must be 1D)
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        ep0: float, The original  exploration bias. Default 1
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, Determines whether feature data will be normalized for problem analysis
        num_data: int, number of available data for training/testing
        LHS_gen_theta: bool, Whether theta_set will be generated from an LHS (True) set or a meshgrid (False). Default False
        eval_all_pairs: bool, determines whether all pairs of theta are evaluated to create heat maps. Default False
        package: str ("gpytorch" or  "scikit_learn") determines which package to use for GP hyperaparameter optimization
        noise_std: float, int: The standard deviation of the noise
        kernel: str ("Mat_52", Mat_32" or "RBF") Determines which GP Kerenel to use
        set_lenscl: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        outputscl: bool, Determines whether utfutscale is trained
        retrain_GP: int, number of times to restart GP training
        GP_train_iter: int, number of training iterations to run. Default is 300
        bo_iter_tot: int, total number of BO iterations per restart
        bo_run_tot: int, total number of BO algorithm restarts
        save_fig: bool, Determines whether figures will be saved. Default False
        save_data: bool, Determines whether data will be saved. Default True
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        
        """
        # Constructor method
        self.cs_name = cs_name
        self.true_params = true_params
        self.true_model_coefficients = true_model_coefficients
        self.param_dict = param_dict
        self.skip_param_types = skip_param_types
        self.ep0 = ep0
        self.sep_fact = sep_fact
        self.normalize = normalize
        self.num_data = num_data
        self.LHS_gen_theta = LHS_gen_theta
        self.eval_all_pairs = eval_all_pairs
        self.bo_iter_tot = bo_iter_tot
        self.bo_run_tot = bo_run_tot
        self.save_fig = save_fig
        self.save_data = save_data
        self.DateTime = DateTime
        self.package = package
        self.noise_std = noise_std
        self.kernel = kernel
        self.set_lenscl = set_lenscl
        self.outputscl = outputscl
        self.retrain_GP = retrain_GP
        self.GP_train_iter = GP_train_iter
        
        
class GPBO_Methods:
    """
    The base class for any GPBO Method
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, method_name, emulator, obj, sparse_grid, GP_training_dims):
        """
        Parameters
        ----------
        method_name, str, The name associated with the method being tested
        emulator: bool, Determines if GP will model the function or the function error
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
        GP_training_dims: int, Number of features for GP
        """
        # Constructor method
        self.method_name = method_name
        self.emulator = emulator
        self.obj = obj
        self.sparse_grid = sparse_grid
        self.GP_training_dims = GP_training_dims
        
class Data:
    """
    The base class for any Data used in this workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, theta_vals, theta_true, x_vals, y_vals, gp_mean, gp_var, sse, ei, hyperparams):
        """
        Parameters
        ----------
        theta_vals: ndarray, The arrays of theta_values
        theta_true: ndarray, The array of true theta_values
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental y data
        gp_mean: ndarray, GP mean prediction values associated with theta_vals and x_vals
        gp_var: ndarray, GP variance prediction values associated with theta_vals and x_vals
        sse: ndarray, sum of squared error values associated with theta_vals and x_vals
        ei: ndarray, expected improvement values associated with theta_vals and x_vals
        hyperparams: ndarray, Array of lengthscale + outputscale values
        """
        # Constructor method
        self.theta_vals = theta_vals
        self.theta_true = theta_true
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.sse = sse
        self.ei = ei
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.hyperparams = hyperparams
        
    def get_dims(self):
        """
        Gets dimenisons of data
        
        Returns
        ---------
        len: int, length of data
        dim: int, number of dimensions of data
        """
        # Method definition
        # Code logic goes here
        pass
    
    def normalize(self, bounds):
        """
        Normalizes data between 0 and 1

        Parameters
        ----------
        bounds: ndarray, The unscaled bounds of the data
        
        Returns:
        ---------
        data_norm: ndarray, the data normalized between 1 and 0 based on the bounds
        scaler: MinMaxScaler(), to scaler used to normalize data ##Can we set this as a parameter once it's calculated?
        """
        # Method definition
        # Code logic goes here
        pass
    
    def unnormalize(self, bounds):
        """
        Normalizes data back to original values 
        
        Parameters
        ----------
        bounds: ndarray, The unscaled bounds of the data
        
        Returns
        ---------
        data: ndarray, the original daat renormalized based on the original bounds
        """
        # Method definition
        # Code logic goes here
        pass

#Stopped Here. Work on this 5/24/23
class GP_Emulator(CaseStudyParameters):
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    type_1
    type_2
    """
    # Class variables and attributes
    
    def __init__(self):
        """
        Parameters
        ----------
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        """
        # Constructor method
        super().__init__(normalize, package, noise_std, kernel, set_lenscl, outputscl, retrain_GP, GP_train_iter)
    
    def type_1(self, param_set): #Where should I account for finding the SSE of these values?
        """
        Evaluates GP model for a standard GPBO
        
        Parameters
        ----------
        param_set: ndarray (n_train x n_dim), Array of GP evaluation data
        
        Returns
        --------
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_var: tensor, The GP model's variance evaluated over param_set 
        
        """
    def type_2(self, param_set, x_vals):
        """
        Evaluates GP model for an emulator GPBO
        
        Parameters
        ----------
        param_set: ndarray (n_train x n_dim), Array of GP evaluation data
        x_vals: ndarray, experimental state points (x data)
        
        Returns
        --------
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_var: tensor, The GP model's variance evaluated over param_set 
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        
class Acquisition_Function(CaseStudyParameters):  
    """
    The base class for acquisition functions
    Parameters
    
    Methods
    --------------
    __init__
    calc_best_error
    type_1_ei
    type_2_ei
    """
    def __init__(self, ep, gp_mean, gp_var):
        """
        Parameters
        ----------
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        ep: float, the exploration parameter of the function  
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_var: tensor, The GP model's variance evaluated over param_set
        """
        
        # Constructor method
        super().__init__(cs_name, theta_true, true_model_coefficients, theta_dict, skip_param_types, normalize)
        self.ep = ep
        self.gp_mean = gp_mean
        self.gp_var = gp_var
    
    def calc_best_error(self, train_data, x_vals, y_vals, Method):
        """
        Calculates the best error of the model
        
        Parameters
        ----------
        train_data: ndarray, The training data
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental output data (y data)
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        best_error: float, the best error of the method
        
        """
    def type_1_ei(self, param_set, best_error, Method):
        """
        Calculates expected improvement of type 1 (standard) GPBO
        
        Parameters:
        -----------
        param_set: ndarray (1 x n_dim), Array of GP evaluation data
        best_error: float, the best error of the problem so far
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        ei: float, The expected improvement of the parameter set
        """
        
    def type_2_ei(self, param_set, best_error, Method):
        """
        Calculates expected improvement of type 2 (emulator) GPBO
        
        Parameters:
        -----------
        param_set: ndarray (1 x n_dim), Array of GP evaluation data
        best_error: float, the best error of the problem so far
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        ei: float, The expected improvement of the parameter set
        """
        
    
class GPBO_Driver(CaseStudyParameters):
    """
    The base class for running the GPBO Workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self):
        """
        Parameters
        ----------
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        """
        # Constructor method
        super().__init__(cs_name, theta_true, true_model_coefficients, theta_dict, skip_param_types, ep0, sep_fact, normalize, num_data, LHS_gen_theta, eval_all_pairs,  package, noise_std, kernel, set_lenscl, outputscl, retrain_GP, GP_train_iter, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime)
        
    def create_exp_data(self):
        """
        Creates experimental data based on x, theta_true, and the case study
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns:
        --------
        Yexp: ndarray. Value of y given state points x and theta_true
        """
        
    def create_sim_data(self, Method):
        """
        Creates simulation data based on x, theta_vals, the GPBO method, and the case study
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns:
        --------
        Ysim: ndarray. Value of y given state points x and theta_vals
        """
        
    def train_GP(self, train_data, verbose = False):
        """
        Trains the GP model
        
        Parameters
        ----------
        train_data: ndarray, The training data
        verbose: bool, Whether to print hyperparameters. Default False.
        
        Returns
        -------
        model: bound method, The model that the GP is bound by
        likelihood: None or bound method, The likelihood of the GP model. In this case, must be Gaussian
        lenscl_final: ndarray, array of optimized lengthscale parameters
        outputscale_final: ndarray (1x1), array of optimized outputscale parameter
        """
        
    def optimize_acquisition_func(self, Method, param_set):
        """
        Optimizes the acquisition function
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        param_set: ndarray (n_train x n_dim), Array of GP evaluation data
        
        Returns
        -------
        Theta_Best: ndarray, Array of Best Theta values (as determined by max(ei)) for each iteration 
        Theta_Opt: ndarray, Array of Optimal Theta values (as determined by min(sse)) for each iteration
        """
        
    def eval_GP_over_grid(self, Method, bounds_p, x_vals, best_error)
        """
        Evaluates GP for data values over a grid (GP Mean, GP var, EI, SSE)
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        bounds_p: ndarray, The bounds for searching for Theta_True.
        x_vals: ndarray, experimental state points (x data)
        best_error: float, the best error given the Method and case study so far
        
        Returns:
        --------
        theta_mesh: ndarray, the meshgrid over which the GP was evaluated
        ei: ndarray, the expected improvement over theta_mesh
        gp_mean: tensor, The GP model's mean evaluated over theta_mesh 
        gp_var: tensor, The GP model's variance evaluated over theta_mesh 
        sse: ndarray, the sum of squared errors over theta_mesh
        sse_var: ndarray, the variance over the sum of squared errors over theta_mesh
        best_error: ndarray, the best_error over theta_mesh
        
        """
        
    def augment_train_data(self, param_set, add_param_set):
        """
        Augments training data given a new point
        
        Parameters
        ----------
        param_set: ndarray (n_train x n_dim), Array of training data
        add_param_set: ndarray (1 x n_dim), New training data to augment
        
        Returns:
        --------
        param_set: ndarray. The parameter set with the augmented values
        """
        
    def run_bo_iter(self, train_data, test_data, param_set, x_vals, y_vals, GP_model, Method):
        """
        Runs a single GPBO iteration
        
        Parameters
        ----------
        train_data: ndarray, The training data
        test_data: ndarray, The testing data
        param_set: ndarray (n_train x n_dim), Array of GP evaluation data
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental output data (y data)
        GP_model: Class, Fully defined model for GP
        Method: class, fully defined methods class which determines which method will be used
        
        Returns:
        --------
        Saves relavent data and figures for one BO iterations
        
        Theta_Best: ndarray, Array of Best Theta values (as determined by max(ei)) for each iteration 
        Theta_Opt: ndarray, Array of Optimal Theta values (as determined by min(sse)) for each iteration
        Min_SSE: float, minimum SSE values (as determined by min(sse)) for each iteration
        Max_EI: float, absolute maximum EI values (as determined by max(ei)) at each iteration   
        gp_mean_vals: ndarray, Array of GP mean values for GP approximation
        gp_var_vals: ndarray,  Array of GP variance values for GP approximation
        time_per_iter: float, time of iteration
        final_hyperparams: ndarray, array of hyperparameters used for GP predictions
        """
    def run_bo_to_term(self, train_data, test_data, param_set, x_vals, y_vals, GP_model, Method, EI_tol = 1e-7):
        """
        Runs multiple GPBO iterations
        
        Parameters
        ----------
        train_data: ndarray, The training data
        test_data: ndarray, The testing data
        param_set: ndarray (n_train x n_dim), Array of GP evaluation data
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental output data (y data)
        GP_model: Class, Fully defined model for GP
        Method: class, fully defined methods class which determines which method will be used
        EI_tol: float, tolerance for EI_max maximum value for GPBO early termination. Default 1e-7.
        
        Returns:
        --------
        Saves relavent data and figures for one BO iterations
        
        BO_Theta_Best: ndarray, Array of all Best Theta values (as determined by max(ei)) for each iteration 
        BO_Theta_Opt: ndarray, Array of all Optimal Theta values (as determined by min(sse)) for each iteration
        BO_Min_SSE: ndarray, all minimum SSE values (as determined by min(sse)) for each iteration
        BO_SSE_abs_min: ndarray, absolute minimum SSE values (as determined by min(sse)) showing only the lowest over all iterations 
        BO_Max_EI: ndarray, absolute maximum EI values (as determined by max(ei)) at each iteration   
        BO_gp_mean_vals: ndarray, Array of GP mean values for GP approximation for all BO iterations
        BO_gp_var_vals: ndarray,  Array of GP variance values for GP approximation for all BO iterations
        BO_median_time_per_iter: float, median time of BO iterations
        BO_final_hyperparams: ndarray, array of hyperparameters used for GP predictions for all BO iterations
        """
        
    def bo_restart(self, train_data, test_data, param_set, x_vals, y_vals, GP_model, Method, EI_tol = 1e-7):
        """
        Runs multiple GPBO iterations
        
        Parameters
        ----------
        train_data: ndarray, The training data
        test_data: ndarray, The testing data
        param_set: ndarray (n_train x n_dim), Array of GP evaluation data
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental output data (y data)
        GP_model: Class, Fully defined model for GP
        Method: class, fully defined methods class which determines which method will be used
        EI_tol: float, tolerance for EI_max maximum value for GPBO early termination. Default 1e-7.
        
        Returns:
        --------
        Saves relavent data and figures for one BO iterations
        
        Run_Theta_Best: ndarray, Array of all Best Theta values (as determined by max(ei)) for each iteration for each GPBO restart
        Run_Theta_Opt: ndarray, Array of all Optimal Theta values (as determined by min(sse)) for each iteration for each GPBO restart
        Run_Min_SSE: ndarray, all minimum SSE values (as determined by min(sse)) for each iteration for each GPBO restart
        Run_SSE_abs_min: ndarray, absolute minimum SSE values showing only the lowest over all iterations for each GPBO restart
        Run_Max_EI: ndarray, absolute maximum EI values (as determined by max(ei)) at each iteration for each GPBO restart
        Run_gp_mean_vals: ndarray, Array of GP mean values for GP approximation for all BO iterations for each GPBO restart
        Run_gp_var_vals: ndarray,  Array of GP variance values for GP approximation for all BO iterations for each GPBO restart
        Run_median_time_per_iter: ndarray, median time of iteration for each GPBO restart
        Run_final_hyperparams: ndarray, array of hyperparameters used for GP predictions for all BO iterations for each GPBO restart
        """