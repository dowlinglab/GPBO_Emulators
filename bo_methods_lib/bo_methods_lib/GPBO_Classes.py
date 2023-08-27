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

class Case_Study:
    """
    The base class for any GPBO Case Study and Evaluation
    Parameters
    
    Methods
    --------------
    __init__
    define_cs()
    
    """
    # Class variables and attributes
    
    def __init__(self, cs_name, ep0, sep_fact, normalize, num_data, bo_iter_tot, bo_run_tot):
        """
        Parameters
        ----------
        cs_name, str, The name associated with the case study being evaluated
        ep0: float, The original  exploration bias. Default 1
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, Determines whether feature data will be normalized for problem analysis
        num_data: int, number of available data for training/testing
        bo_iter_tot: int, total number of BO iterations per restart
        bo_run_tot: int, total number of BO algorithm restarts
        """
        # Constructor method
        self.cs_name = cs_name
        self.ep0 = ep0
        self.sep_fact = sep_fact
        self.normalize = normalize
        self.num_data = num_data
        self.bo_iter_tot = bo_iter_tot
        self.bo_run_tot = bo_run_tot
        
    def define_cs(self):
        """
        Returns:
        ---------
        true_params: ndarray, The array containing the true parameter values for the problem (must be 1D)
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        """

        
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
    
    def __init__(self, Case_Study, Method, bounds_p, bounds_x): #Am I able to initialize a class with another class? I need the params from it.
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
    
    def train_test_split(self, point, seed = None):
        """
        Splits available data into training and tetsing data
        
        Parameters
        ----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets which are available as training data
        seed: None or int, Optional Seed for test/train split
        #Note - Separation factor inherited from Case_Study
        
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
    
    def optimize_set(self, param_set, Yexp, train_data, model, likelihood, Case_Study, Method, ep, derived_val):
        """
        Optimizes theta set to find best SSE or EI
        
        Parameters
        ----------
        param_set: ndarray (n_points x n_dims) matrix of parameter sets to evaluate
        Yexp: ndarray, Experimental output data
        train_data: ndarray, The GP training data
        model: bound method, The model that the GP is bound by
        likelihood: bound method or None, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        Case_Study: class, The case study which the method is applied to
        Method: class, The method with which to evaluate the GP
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
        Case_Study: class, The case study which the method is applied to
        Method: class, The method with which to evaluate the GP
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise. Default 0
        seed: None or int, Optional Seed for noise generation. Default None
        
        Returns
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
        Case_Study: class, The case study which the method is applied to
        Method: class, The method with which to evaluate the GP
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
    set_hps()
    print_hps
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
    ep0: float, The original exploration bias. Default 1
    """
    # Class variables and attributes
    ep0 = ep_bias
    
    def __init__(self, ep_calc_method, ep):
        """
        Parameters
        ----------
        ep_calc_method: str, The name associated with the exploration parameter method being tested
        ep: float, the current exploration parameter value
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
        e_inc: float, The increment for the Boyle's method for calculating exploration parameter: Default is 1.5
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
        .....
        """
        # Constructor method
        
    def calc_be(self, Case_Study, Method, train_data, Xexp, Yexp):
        """
        Evaluates value of exploration parameter at each BO iteration
        
        Parameters:
        -----------
        Case_Study: class, The case study which the method is applied to
        Method: class, The method with which to evaluate the GP
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
    calc_ei()
    find_argmax_param_set()
    calc_sse()
    find_argmin_param_set()

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
        
    def calc_ei(self, Method, train_data, Xexp, Yexp, model_mean, model_variance, ep):
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
        
def Plotter_Class:
    """
    The base class for making plots
    
    Methods
    --------------
    __init__
    plot_heat_map()
    plot_line_graph()
    plot_scatter_graph()
    plot_3d_scatter_graph()
    
    """
    
    def __init__(self):
        """
        Parameters
        ----------
        
        """
        # Constructor method
        
    def plot_heat_map(self, param_vals, param_names, z, z_title, scatter_points, scatter_labels, title = "", levels = 100):
        """
        Creates a heat map given a mesh grid (x-y) and corresponding z plot values
        
        Parameters
        ----------
        param_vals: ndarray, a meshgrid of parameter values to plot on the x and y axes
        param_names: list, names of parameter values to plot on the x and y axes
        z: list of ndarray, an array of the value(s) corresponding to the heat maps
        z_title: list of str, list of titles for what the value of z represents 
        scatter_points: list of ndarray, list of values of scatter points to add to the heat maps
        scatter_labels: list of str, list of names of scatter points to add to the heat maps
        title: str, Title of the graph. Default empty string
        levels: int, the value of levels to include during heat map plotting (black lines). Default = 100
        
        Returns:
        --------
        Heat maps of given data
        """

    def plot_line_graph(self, param_vals, param_names, z, z_title, title = ""):
        """
        Creates a line plot given x-axis values and corresponding y plot values
        
        Parameters
        ----------
        param_vals: ndarray, a meshgrid of parameter values to plot on the x axes
        param_names: list, names of parameter values to plot on the x axes
        z: list of ndarray, an array of the value(s) corresponding to the y-axis
        z_title: list of str, list of titles for what the value of z represents 
        title: str, Title of the graph. Default empty string
        
        Returns:
        --------
        Line plots of given data
        """   
        
    def plot_scatter_graph(self, param_vals, param_names, scatter_points, scatter_labels, title = ""):
        """
        Creates a scatter plot given x-axis values and corresponding y plot values
        
        Parameters
        ----------
        param_vals: ndarray, a meshgrid of parameter values to plot on the x axes
        param_names: list, names of parameter values to plot on the x and y axes
        scatter_points: list of ndarray, list of values of scatter points to add to the heat maps
        scatter_labels: list of str, list of names of scatter points to add to the heat maps
        title: str, Title of the graph. Default empty string
        
        Returns:
        --------
        Scatter plots of given data
        """ 
      
    def plot_3d_scatter_graph(self, param_vals, param_names, scatter_points, scatter_labels, title = ""):
         """
        Creates a scatter plot given x-axis and y-axis values and corresponding z plot values
        
        Parameters
        ----------
        param_vals: ndarray, a meshgrid of parameter values to plot on the x axes
        param_names: list, names of parameter values to plot on the x and y axes
        scatter_points: list of ndarray, list of values of scatter points to add to the heat maps
        scatter_labels: list of str, list of names of scatter points to add to the heat maps
        title: str, Title of the graph. Default empty string
        
        Returns:
        --------
        3D Scatter plots of given data
        """ 
    
    
class Set_File_Path:
    """
    The base class for saving CSVs

    Methods
    --------------
    __init__
    save_data_file()
    save_plot_figure()
    """

    def __init__(self):
        """
        Parameters
        ----------

        """
        # Constructor method
    
    def save_data_file(self, DateTime, Case_Study, Method, GP_Model, bo_iter, bo_restart):
        """
        Saves data files to a .npy file
        
        Parameters
        ----------
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        Case_Study: class, The case study being evaluated
        Method: class, The method being tested
        GP_model: class, The GP model used in the analysis
        bo_iter: int, The BO iter number for the problem
        bo_restart: int, The restart iter number for the problem
        
        """        
        
    def save_plot_figure(self, DateTime, Case_Study, Method, GP_Model, bo_iter, bo_restart):
        """
        Saves figures to a .png file
        
        Parameters
        ----------
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        Case_Study: class, The case study being evaluated
        Method: class, The method being tested
        GP_model: class, The GP model used in the analysis
        bo_iter: int, The BO iter number for the problem
        bo_restart: int, The restart iter number for the problem
        """
    
    # Class variables and attributes
        
# QUESTIONS:
#     1) What is the best way to save csvs of the necessary data? How can I write that to a class such that each function doesn't need 100 parameters?
#     2) Is there a better way to define the Best_Error/GP_Derived_Values classes together? As it is, what would I initialize for Best_Error?
#     3) What is the best way to deal with the shape differences between standard and emulator BO?
#     4) When do I use a function vs a class?
#     5) Where would I add a method for GP validation? Parameter_Set?
#     6) Which class should creating data be part of?
#     7) How should I make a class for plotters given that at any point I may want multiple different objects that would appear on one graph but not another? What would I even initialize for a plotting class?

