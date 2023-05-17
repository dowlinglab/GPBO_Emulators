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
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import time
import Tasmanian

#Notes: Change line below when changing test problems: 
# If line 21 is active, the 8D problem is used, if line 22 is active, the 2D problem is used
# from .CS2_create_data import calc_muller, create_sse_data, create_y_data, calc_y_exp, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point
from .CS1_create_data import create_sse_data, create_y_data, calc_y_exp, gen_y_Theta_GP, eval_GP_emulator_BE, make_next_point

from .bo_functions_generic import LHS_Design, set_ep, test_train_split, find_train_doc_path, ExactGPModel, train_GP_model, calc_GP_outputs, explore_parameter, ei_approx_ln_term, calc_ei_emulator, get_sparse_grids, eval_GP_sparse_grid, calc_ei_basic, train_test_plot_preparation, clean_1D_arrays, norm_unnorm, train_GP_scikit, define_GP_model

from .normalize import normalize_x, normalize_p_data, normalize_p_set, normalize_p_true, normalize_constants, normalize_general, normalize_p_bounds

from .CS2_bo_plotters import value_plotter, plot_xy, plot_Theta, plot_Theta_min, plot_obj, plot_obj_abs_min, plot_3GP_performance, plot_sep_fact_min, save_fig, save_csv, path_name, plot_EI_abs_max, save_misc_data
# from CS2_bo_plotters import plot_org_train

class Case_Study:
    """
    The base class for any GPBO Method
    Parameters
    
    Methods
    --------------
    __init__
    define_cs()
    
    """
    # Class variables and attributes
    
    def __init__(self, CS_name):
        """
        Parameters
        ----------
        method_name, str, The name associated with the case study being evaluated
        """
        # Constructor method
        self.method_name = method_name
    def define_method(self):
        """
        Returns:
        ---------
        true_params: ndarray, The array containing the true parameter values for the problem (must be 1D)
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        """
        #Logic for determining obj, sparse_grid, and emulator


class GPBO_Methods:
    """
    The base class for any GPBO Method
    Parameters
    
    Methods
    --------------
    __init__
    define_method()
    
    """
    # Class variables and attributes
    
    def __init__(self, method_name):
        """
        Parameters
        ----------
        method_name, str, The name associated with the method being tested
        """
        # Constructor method
        self.method_name = method_name
    def define_method(self):
        """
        Parameters:
        -----------
        Returns:
        ---------
        emulator: bool, Determines if GP will model the function or the function error
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
        GP_training_dims: int, Number of features for GP
        """
        #Logic for determining obj, sparse_grid, and emulator
    
class Parameters_And_State_Points:
    """
    The base class for any parameter set or Experimental X data among any method

    Methods
    -------
    __init__
    get_dims()
    normalize()
    unnormalize()

    """
    # Class variables and attributes
    def __init__(self, bounds_p, bounds_x):
        """
        Parameters
        -----------
        bounds_p: ndarray, The bounds for searching for Theta_True.
        bounds_x: ndarray, The bounds of Xexp
        """
        # Constructor method
        self.bounds_p = bounds_p
        self.bounds_x = bounds_x
   
    def get_dims(self, point):
        """
        Gets dimenisons of a parameter set, point, or state point
        
        Parameters
        ----------
        point: ndarray, the parameter set or state point we want to evaluate
        
        Returns
        ---------
        len: int, length of parameter set/point/theta_values 
        dim: int, number of dimensions of parameter set/point/theta_values 
        """
        # Method definition
        # Code logic goes here
        pass
    
    def normalize(self, point):
        """
        Normalizes a point, parameter set, or state point between 0 and 1
        
        Parameters
        ----------
        point: ndarray, the parameter set or state point we want to scale
        Returns:
        ---------
        point_norm: ndarray, the parameter set or state point normalized between 1 and 0 based on the bounds
        scaler: MinMaxScaler(), to scaler used to normalize data ##Can we set this as a parameter once it's calculated?
        """
        # Method definition
        # Code logic goes here
        pass
    
    def unnormalize(self, point_norm, scaler):
        """
        Normalizes a point, parameter set, or state point back to original values 
        
        Parameters
        ----------
        point_norm: ndarray, the parameter set or state point we want to rescale
        scaler: MinMaxScaler(), used to un-normalize data
        ---------
        point: ndarray, the original parameter set or state point renormalized based on the original bounds
        """
        # Method definition
        # Code logic goes here
        pass
    
class Parameter_Sets(Parameters_And_State_Points):
    """
    The base class for any Parameter Set that will be used to train or evaluate a GP
    
    Methods
    --------
    __init__
    generate_set()
    train_test_split()
    eval_set()
    optimize_set()
    create_y_data()
    create_sse_data()
    
    """
    # Class variables and attributes
    def __init__(self, Case_Study, Method, bounds_p, bounds_x):
        """
        Parameters
        -----------
        Case_Study: Case_Study, The Case_Study that is being evalauted
        Method: Method, The method that is being tested
        bounds_p: ndarray, The bounds for searching for Theta_True
        bounds_x: ndarray, The bounds of Xexp
        """
        # Constructor method
        self.Case_Study = Case_Study
        self.Method = Method
        super().__init__(bounds_p, bounds_x)
        
    def generate_set(self, LHS, n_points, n_dims, seed = None):
        """
        Generates a theta_set for initial evaluation
        
        Parameters
        ----------
        LHS: bool, Determines whether a meshgrid or LHS sample is generated, default True
        n_points:, int, number of meshgrid points/ parameter or the square root of the number of LHS samples
        n_dims: int, Number of parameters to regress
        seed: None or int, Optional Seed for LHS generation
        ---------
        Returns:
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        """
        # Method definition
        # Code logic goes here
        pass
    
    
    def train_test_split(self, point, sep_fact, seed = None):
        """
        Splits available data into training and tetsing data
        
        Parameters
        ----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets which are available as training data
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        seed: None or int, Optional Seed for test/train split
        
        Returns:
        ---------
        train_data: ndarray, The GP training data
        test_data: ndarray, The GP testing data
        """
        # Method definition
        # Code logic goes here
        pass
    
    def eval_set_GP(self, param_set, model, likelihood):
        """
        Evaluates GP mean and standard deviation prediction for GP model
        
        Parameters
        ----------
        param_set: ndarray (1 x n_dims) matrix of parameter sets to evaluate
        model: bound method, The model that the GP is bound by
        likelihood: bound method or None, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        
        Returns
        ---------
        model_mean: float, The GP mean
        model_variance: float, the GP variance
        """
        # Method definition
        # Code logic goes here
        pass
    
    def optimize_set(self, param_set, Yexp, train_data, model, likelihood, Method, Case_Study, ep, derived_val):
        """
        Optimizes theta set to find best SSE or EI
        
        Parameters
        ----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        Yexp: ndarray, Experimental output data
        train_data: ndarray, The GP training data
        model: bound method, The model that the GP is bound by
        likelihood: bound method or None, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        Method: class, The method with which to evaluate the GP
        Case_Study: class, The case study which the method is applied to
        ep: float, The exploration bias used for calculation of EI
        derived_val: ndarray, either "SSE" or "EI". The array values we want to optimize over theta
        ---------
        point: ndarray, the original parameter set or state point renormalized based on the original bounds
        """
        # Method definition
        # Code logic goes here
        pass
    def create_y_data(self, param_set, Case_Study, Method, noise_std, noise_mean = 0, seed = None):
        """
        Creates y data for a given case study and method
        
        Parameters
        ----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        Method: class, The method with which to evaluate the GP
        Case_Study: class, The case study which the method is applied to
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise. Default 0
        seed: None or int, Optional Seed for noise generation. Default None
        ---------
        y_data: ndarray, the y data associated with a given parameter set
        """
        # Method definition
        # Code logic goes here
        pass
    
    def create_sse_data(self, param_set, Yexp, Case_Study, Method, noise_std, noise_mean = 0, seed = None):
        """
        Creates sse data for a given case study and method
        
        Parameters
        ----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        Yexp: ndarray, Experimental output data
        Method: class, The method with which to evaluate the GP
        Case_Study: class, The case study which the method is applied to
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise. Default 0
        seed: None or int, Optional Seed for noise generation. Default None
        ---------
        sse_data: ndarray, the sse data associated with a given parameter set
        """
        # Method definition
        # Code logic goes here
        pass
    
class Hyperparameters:
    """
    The base class for any best error
    
    Methods
    --------------
    __init__
    calc_be()

    """
    # Class variables and attributes
    
    def __init__(self, hp_name, constant):
        """
        Parameters
        ----------
        hp_name: str, The name associated with the hyperparameter
        constant: bool, Whether the hyperparameter is a constant (T) or will be optimized (F)
        
        """
        # Constructor method
        self.hp_name = hp_name
        self.constant = constant
        self.value = value
        
    def set_hps(self, value = 1):
        """
        Evaluates value of hyperparameters
        
        Parameters:
        -----------
        value: int, If constant, value of the hyperparameter, If not constant, starting value of the hyperparameter
        
        Returns:
        ---------
        hyperparam: float or ndarray, A model hyperparameter
        """
        #Logic
    
    def print_hps(self, hyperparam):
        """
        prints evaluated value of hyperparameters
        
        Parameters:
        -----------
        hyperparam: float or ndarray, A model hyperparameter
        
        Returns:
        ---------
        hyperparam_pp: str, The hyperparameter as a string rounded appropriately
        """
        #Logic 
        
class Exploration_Bias(ep_bias = 1):
    """
    The base class for any Exploration Parameter
    
    Methods
    --------------
    __init__
    calc_ep()
    
    Variables
    --------------
    ep0: float,  The original  exploration bias. Default 1
    """
    # Class variables and attributes
    ep0 = ep_bias
    
    def __init__(self, ep_calc_method, ep):
        """
        Parameters
        ----------
        method_name, str, The name associated with the method being tested
        """
        # Constructor method
        self.ep_calc_method = ep_calc_method
        self.ep = ep
        
    def calc_ep(self, Bo_iter, ep, mean_of_var, best_error, ep_inc = 1.5, ep_f = 0.01, improvement = False):
        """
        Evaluates value of exploration parameter at each BO iteration
        
        Parameters:
        -----------
        Bo_iter: int, The value of the current BO iteration
        mean_of_var: float, The value of the average of all posterior variances
        best_error: float, The best error of the GP
        e_inc: float, the increment for the Boyle's method for calculating exploration parameter: Default is 1.5
        ep_f: float, The final exploration parameter value: Default is 0.01
        ep_method: float, determines if Boyle, Jasrasaria, or exponential method will be used: Defaults to constant ep
        improvement: Bool, Determines whether last objective was an improvement
        
        Returns:
        ---------
        ep: float, The current value of exploration bias
        """
        #Logic
        
class Best_Error:
    """
    The base class for any best error
    
    Methods
    --------------
    __init__
    calc_be()

    """
    # Class variables and attributes
    
    def __init__(self):
        """
        Parameters
        ----------
        method_name, str, The name associated with the method being tested
        """
        # Constructor method
        
    def calc_be(self, Method, Case_Study, train_data, Xexp, Yexp):
        """
        Evaluates value of exploration parameter at each BO iteration
        
        Parameters:
        -----------
        Method: class, The method with which to evaluate the GP
        Case_Study: class, The case study which the method is applied to
        train_data: ndarray, The GP training data
        Xexp: ndarray, Experimental input data
        Yexp: ndarray, Experimental output data
        
        Returns:
        ---------
        best_error: float, The best error of the GP model
        """
        #Logic for determining be 
        
class GP_Derived_Values:
    """
    The base class for parameters derived from GP predictions GP stdev and GP mean
    
    Methods
    --------------
    __init__
    eval_GP_param()

    """
    # Class variables and attributes
    
    def __init__(self, derived_val_name):
        """
        Parameters
        ----------
        derived_val_name: str, Either "SSE" or "EI". The value we want to calculate
        """
        # Constructor method
        self.derived_val_name = derived_val_name
        
    def calc_EI(self, Method, train_data, Xexp, Yexp, model_mean, model_variance, ep):
        """
        Evaluates value of expected improvement
        
        Parameters:
        -----------
        Method: class, The method with which to evaluate the GP
        train_data: ndarray, The GP training data
        Xexp: ndarray, Experimental input data
        Yexp: ndarray, Experimental output data
        model_mean: float, The GP mean
        model_variance: float, the GP variance
        ep: float, The current value of exploration bias
        
        Returns:
        ---------
        ei: float, The expected improvement of the given values
        """
        
    def find_argmax_param_set(self, param_set, derived_val):
        """
        Evaluates value of expected improvement
        
        Parameters:
        -----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        derived_val: ndarray, Either SSE or EI. The value we want to find the maximum of
        
        Returns:
        ---------
        argmax_param_set: ndarray, The parameter sets associated with the argmax of the derived property value
        """    
        
    def calc_sse(self, Method, train_data, Xexp, Yexp, model_mean):
        """
        Evaluates value of exploration parameter at each BO iteration

        Parameters:
        -----------
        Method: class, The method with which to evaluate the GP
        train_data: ndarray, The GP training data
        Xexp: ndarray, Experimental input data
        Yexp: ndarray, Experimental output data
        model_mean: float, The GP mean

        Returns:
        ---------
        sse: float, The sse of the given values
        """
        
    def find_argmin_param_set(self, param_set, derived_val):
        """
        Evaluates value of expected improvement
        
        Parameters:
        -----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        derived_val: ndarray, Either SSE or EI. The value we want to find the maximum of
        
        Returns:
        ---------
        argmin_param_set: ndarray, The parameter sets associated with the argmin of the derived property value
        """ 
        
# QUESTIONS:
#     1) What is the best way to save csvs of the necessary data? How can I write that to a class such that each function doesn't need 100 parameters?
#     2) Is there a better way to define the Best_Error/GP_Derived_Values classes together? As it is, what would I initialize for Best_Error?
#     3) What is the best way to deal with the shape differences between standard and emulator BO?
#     4) When do I use a function vs a class?
#     5) Where would I add a method for GP validation? Parameter_Set?
#     6) Which class should creating data be part of?