import numpy as np
# import math
# from scipy.stats import norm
# from scipy import integrate
# import torch
# import csv
# import gpytorch
# import scipy.optimize as optimize
# import pandas as pd
# import os
# import time
# import Tasmanian
from scipy.stats import qmc
import pandas as pd

from .GPBO_Class_fxns import vector_to_1D_array, calc_muller, calc_cs1_polynomial, lhs_design, calc_y_exp, calc_y_sim, calc_sse
import itertools
from itertools import combinations_with_replacement, combinations, permutations

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
    
    def __init__(self, cs_name, true_params, true_model_coefficients, param_dict, indecies_to_consider, ep0, sep_fact, normalize, num_x_data, num_theta_data, num_data, LHS_gen_theta, x_data_vals, eval_all_pairs, package, noise_mean, noise_std, kernel, set_lenscl, outputscl, retrain_GP, GP_train_iter, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime, seed):
        """
        Parameters
        ----------
        cs_name, str, The name associated with the case study being evaluated
        true_params: ndarray, The array containing the true parameter values for the problem (must be 1D)
        true_model_coefficients: ndarray, The array containing the true values of problem constants
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        ep0: float, The original  exploration bias. Default 1
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, Determines whether feature data will be normalized for problem analysis
        num_x_data: int, number of available x data for training/testing
        num_theta_data: int, number of available theta data for training/testing
        num_data: int, number of available data for training/testing 
        lhs_gen_theta: bool, Whether theta_set will be generated from an LHS (True) set or a meshgrid (False). Default False
        x_data_vals: ndarray or none: Values of X data. If none, these values must be generated
        eval_all_pairs: bool, determines whether all pairs of theta are evaluated to create heat maps. Default False
        package: str ("gpytorch" or  "scikit_learn") determines which package to use for GP hyperaparameter optimization
        noise_mean:float, int: The mean of the noise
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
        seed: int or None, Determines seed for randomizations. None if seed is random
        
        """
        # Constructor method
        self.cs_name = cs_name
        self.true_params = true_params
        self.true_model_coefficients = true_model_coefficients
        self.param_dict = param_dict
        self.indecies_to_consider = indecies_to_consider
        self.ep0 = ep0
        self.sep_fact = sep_fact
        self.normalize = normalize
        self.num_x_data = num_x_data
        self.num_theta_data = num_theta_data
        self.num_data = num_data
        self.lhs_gen_theta = LHS_gen_theta
        self.x_data_vals = x_data_vals
        self.eval_all_pairs = eval_all_pairs
        self.bo_iter_tot = bo_iter_tot
        self.bo_run_tot = bo_run_tot
        self.save_fig = save_fig
        self.save_data = save_data
        self.DateTime = DateTime
        self.package = package
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.kernel = kernel
        self.set_lenscl = set_lenscl
        self.outputscl = outputscl
        self.retrain_GP = retrain_GP
        self.GP_train_iter = GP_train_iter
        self.seed = seed
        
        
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
    def set_GP_training_dims(self, x_data, theta_vals):
        """
        Sets the number of GP training inputs based on method properties and data dimensions
        
        Parameters:
        -----------
        x_data_vals: ndarray: Values of x data
        theta_vals: ndarray, The arrays of theta_values
        
        Returns:
        --------
        GP_training_dims: int, number of GP inputs
        """
        if self.emulator == True:
            GP_training_dims = (theta_vals.shape[1]) + (x_data.shape[1])
        else:
            GP_training_dims = (theta_vals.shape[1])
        
        return GP_training_dims
        
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
    #What is the syntax to allow data to be any self parameter?
    def normalize(self, data, bounds):
        """
        Normalizes data between 0 and 1

        Parameters
        ----------
        bounds: ndarray, The unscaled bounds of the data
        
        Returns:
        ---------
        scaled_data: ndarray, the data normalized between 1 and 0 based on the bounds
        """
        # Method definition
        # Code logic goes here
        #Define a scaler for normalization
        bounds = vector_to_1D_array(bounds)
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        scaled_data = (data - lower_bound) / (upper_bound - lower_bound)
        
        return scaled_data
    
    def unnormalize(self, scaled_data, bounds):
        """
        Normalizes data back to original values 
        
        Parameters
        ----------
        scaler: MinMaxScaler(), The scaler used to normalize the data
        bounds: ndarray, The unscaled bounds of the data
        
        Returns
        ---------
        data: ndarray, the original data renormalized based on the original bounds
        """
        #Transform/unnormalize data (#How to generalize data to work for any variable type given?)
        bounds = vector_to_1D_array(bounds)
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        data = scaled_data*(upper_bound - lower_bound) + lower_bound
        
        return data

#https://www.geeksforgeeks.org/inheritance-and-composition-in-python/
#AD: Use composition instead of inheritance here, pass an instance of CaseStudyParameters to the init function
class Type_1_GP_Emulator(CaseStudyParameters):
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
    
    def eval_gp(self, param_set): #Where should I account for finding the SSE of these values?
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
        
class Type_2_GP_Emulator(CaseStudyParameters):
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
    
    def eval_gp(self, param_set, x_vals):
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
##Again, composition instead of inheritance      
class Expected_Improvement(CaseStudyParameters):  
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
    #AD Comment: What part of the acquisition function code can be generalized and what is specific to type1 and type2? 
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
    def type_1(self, param_set, best_error, Method):
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
        
    def type_2(self, param_set, best_error, Method):
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
        
##Again, composition instead of inheritance       
class GPBO_Driver:
    """
    The base class for running the GPBO Workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, CaseStudyParameters):
        """
        Parameters
        ----------
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        """
        # Constructor method
        self.CaseStudyParameters = CaseStudyParameters
        
    #Not sure how to generalize this between different params effectively
    def create_param_data(self, n_points, bounds, gen_meth):
        """
        Generates data based off of bounds, and an LHS generation number
        
        Parameters
        ----------
        bounds: ndarray, array of x bounds
        gen_meth: str, ("LHS", "Meshgrid", or None). Determines whether x data will be generated with an LHS or meshgrid (for 2D X data)
        
        Returns:
        --------
        x_data: ndarray, a list of x data
        
        Notes: Meshgrid generated data will output n_points in each dimension, LHS generates n_points of data
        """
        assert gen_meth == "LHS" or gen_meth == "Meshgrid", "gen_meth must be LHS or Meshgrid"
        
        seed = self.CaseStudyParameters.seed
        dimensions = bounds.shape[1]
        
        if gen_meth == "Meshgrid":
            #Generate mesh_grid data for theta_set in 2D
            #Define linspace for theta
            params = np.linspace(0,1, n_points)
            #Generate the equivalent of all meshgrid points
            df = pd.DataFrame(list(itertools.product(params, repeat=dimensions)))
            df2 = df.drop_duplicates()
            scaled_data = df2.to_numpy()
            #Normalize to bounds 
            if bounds is not None:
                lower_bound = bounds[0]
                upper_bound = bounds[1]
                data = scaled_data*(upper_bound - lower_bound) + lower_bound  
            
        elif gen_meth == "LHS":
            #Generate LHS sample
            data = lhs_design(n_points, dimensions, seed, bounds = bounds)
        
        else:
            pass
#             assert self.CaseStudyParameters.x_data_vals is not None, "X must be provided if not generated"
#             assert self.CaseStudyParameters.x_data_vals.shape[1] == exp_d, "Provided X values must have the same dimension as bounds!"
#             sim_data = self.CaseStudyParameters.x_data_vals
        
        #How do I make the data an instance of the data class?
        return data
        
    def create_y_exp_data(self):
        """
        Creates experimental y data based on x, theta_true, and the case study
        
        Returns:
        --------
        Yexp: ndarray. Value of y given state points x and theta_true
        """
        true_p = self.CaseStudyParameters.true_params
        noise_std = self.CaseStudyParameters.noise_std
        noise_mean = self.CaseStudyParameters.noise_mean
        true_model_coefficients = self.CaseStudyParameters.true_model_coefficients
        seed = self.CaseStudyParameters.seed
        #Is there an easier way to do this? Should I rename the functions such that none of the names are the same?
        #Note - Can only use this function after generating x data
        x_data = self.CaseStudyParameters.x_data_vals
        if self.CaseStudyParameters.cs_name == "CS1":
            y_exp = calc_y_exp(calc_cs1_polynomial, true_model_coefficients, x_data, noise_std, noise_mean, seed)   
        elif self.CaseStudyParameters.cs_name == "CS2":
            y_exp = calc_y_exp(calc_muller, true_model_coefficients, x_data, noise_std, noise_mean, seed)
        else:
            print("cs_name must be CS1 or CS2!")
        return y_exp
        
    def create_sim_data(self, method, sim_data, exp_data):
        """
        Creates simulation data based on x, theta_vals, the GPBO method, and the case study
        
        Parameters
        ----------
        method: class, fully defined methods class which determines which method will be used
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        
        Returns:
        --------
        Ysim: ndarray. Value of y given state points x and theta_vals
        """
        cs_name = self.CaseStudyParameters.cs_name
        gp_method = method.method_name
        obj = method.obj
        indecies_to_consider = self.CaseStudyParameters.indecies_to_consider
        noise_mean = self.CaseStudyParameters.noise_mean
        noise_std = self.CaseStudyParameters.noise_std
        true_model_coefficients = self.CaseStudyParameters.true_model_coefficients
        seed =  self.CaseStudyParameters.seed
        
        if cs_name == "CS1":
            if gp_method in ["1A", "1B"]:
                #Calculate sse for sim data
                y_sim = calc_sse(calc_cs1_polynomial, sim_data, exp_data, true_model_coefficients, indecies_to_consider, obj = "obj")
            else:
                #Calculate y_sim for sim data
#                 y_sim = cs1_calc_y_sim(sim_data, exp_data)
                y_sim = calc_y_sim(calc_cs1_polynomial, sim_data, exp_data, true_model_coefficients, indecies_to_consider)
                
        
        elif cs_name == "CS2":
            if gp_method in ["1A", "1B"]:
                #Calculate sse for sim data. Need new functions
                y_sim = calc_sse(calc_muller, sim_data, exp_data, true_model_coefficients, indecies_to_consider, obj = "obj")
            else:  
                #Calculate y_sim for sim data. Need new functions
#                 y_sim = cs2_calc_y_sim(sim_data, true_model_coefficients, indecies, noise_mean, noise_std, seed)
                y_sim = calc_y_sim(calc_muller, sim_data, exp_data, true_model_coefficients, indecies_to_consider)
        
        else:
            raise ValueError("self.CaseStudyParameters.cs_name must be CS1 or CS2!")
            
        return y_sim
            
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
        
    def eval_GP_over_grid(self, Method, bounds_p, x_vals, best_error):
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
        
    def augment_train_data(self, sim_data, best_theta):
        """
        Augments training data given a new point

        Parameters
        ----------
        sim_data: Class, Class containing at least the theta_vals and y data for simulations
        best_theta: Class, Class containing at least the best_theta value as defined by max(EI) and the corresponding y_sim value

        Returns:
        --------
        sim_data: ndarray. The training parameter set with the augmented theta values
        """
        sim_data.theta_vals = np.vstack((sim_data.theta_vals, best_theta.theta_vals)) #(q x t)
        sim_data.y_vals = np.concatenate((sim_data.y_vals, best_theta.y_vals)) #(q x t)

        return sim_data

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