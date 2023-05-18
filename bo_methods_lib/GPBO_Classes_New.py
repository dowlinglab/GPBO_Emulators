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
    
    def __init__(self, cs_name, true_params, true_model_coefficients, param_dict, skip_param_types, ep0, sep_fact, normalize, num_data, bo_iter_tot, bo_run_tot):
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
        bo_iter_tot: int, total number of BO iterations per restart
        bo_run_tot: int, total number of BO algorithm restarts
        
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
        self.bo_iter_tot = bo_iter_tot
        self.bo_run_tot = bo_run_tot
        
        
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
    
    def __init__(self, theta_vals, theta_true, x_vals, y_vals, gp_mean, gp_var, sse, ei):
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
    
def LHS_Design(num_points, dimensions, seed = None, bounds = None):
    """
    Design LHS Samples
    
    Parameters
    ----------
        num_points: int, number of points in LHS, should be greater than # of dimensions
        dimensions: int, number of parameters
        seed (optional): int, seed of random generation
        bounds (optional): ndarray, array containing upper and lower bounds of elements in LHS sample. Defaults of 0 and 1
    
    Returns
    -------
        LHS: ndarray, Array of LHS sampling points
    """
    sampler = qmc.LatinHypercube(d=dimensions, seed = seed)
    LHS = sampler.random(n=num_points)
    
    if bounds is not None:
        LHS = qmc.scale(LHS, bounds[0], bounds[1])

    return LHS

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
        super().__init__(cs_name, theta_true, true_model_coefficients, theta_dict, skip_param_types, ep0, sep_fact, normalize, num_data, bo_iter_tot, bo_run_tot)
        
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
    def run_bo_iter(self, Method):
    def bo_to_term(self, Method):