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
from enum import Enum

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


#Question: Which parameters in Simulator vs CaseStudyParameters?
class CaseStudyParameters:
    """
    The base class for any GPBO Case Study and Evaluation
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, cs_name, ep0, sep_fact, normalize, eval_all_pairs, package, noise_mean, noise_std, kernel, set_lenscl, outputscl, retrain_GP, GP_train_iter, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime, seed):
        """
        Parameters
        ----------
        cs_name: Class, The name/enumerator associated with the case study being evaluated   
        ep0: float, The original  exploration bias. Default 1
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, Determines whether feature data will be normalized for problem analysis
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
        self.ep0 = ep0
        self.sep_fact = sep_fact
        self.normalize = normalize
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

#I'm having trouble defining how to update num_x_data, num_theta_data and dim_x depending on the situation. Especially when adding new data or when using the meshgrid options
class Simulator:
    """
    The base class for differet simulators. Defines a simulation
    """
    def __init__(self, dim_x, indecies_to_consider, theta_ref, theta_names, calc_y_fxn):
        """
        Parameters
        ----------
        dim_x: int, The number of dimensions of x
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        theta_ref: ndarray, The array containing the true values of problem constants
        theta_names: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        calc_y_fxn: function, The function to calculate ysim data with
        """
        # Constructor method
        self.dim_x = dim_x
        self.dim_theta = len(indecies_to_consider) #Length of theta is equivalent to the number of indecies to consider
        self.indecies_to_consider = indecies_to_consider
        self.theta_ref = theta_ref
        self.theta_names = theta_names
        self.theta_true, self.theta_true_names = self.set_true_params()
        self.calc_y_fxn = calc_y_fxn
    
    def set_num_theta_data(self, gen_meth):
        """
        Set the number of theta_data given gen_meth
        """
        if gen_meth.value == 1:
            self.num_theta_data = self.num_theta_data
        elif gen_meth.value == 2:
            self.num_theta_data = self.num_theta_data**2
        else:
            raise ValueError("gen_meth.value must be 1 or 2!")            
        
    def set_true_params(self):
        """
        Sets true parameter value array and the corresponding names based on parameter dictionary and indecies to consider
        
        Parameters
        ----------
        CaseStudyParameters: class, class containing at least the true_model_coefficients and param_dict
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        
        Returns
        -------
        true_params: ndarray, The true parameter of the model
        """
        theta_ref = self.theta_ref
        theta_names = self.theta_names
        indecies_to_consider = self.indecies_to_consider
        
        assert all(0 <= idx <= len(theta_ref)-1 for idx in indecies_to_consider)==True, "indecies to consider must be in range of theta_ref"
        
        true_params = theta_ref[indecies_to_consider]
        true_param_names = [theta_names[idx] for idx in indecies_to_consider]
        
        return true_params, true_param_names
    
    def create_sim_data(self, method, CaseStudyParameters, sim_data, exp_data):
        """
        Creates simulation data based on x, theta_vals, the GPBO method, and the case study
        
        Parameters
        ----------
        method: class, fully defined methods class which determines which method will be used
        CaseStudyParameters: class, class containing at least the true_model_coefficients
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        
        Returns:
        --------
        Ysim: ndarray. Value of y given state points x and theta_vals
        """
        gp_method = method.method_name.value

        if gp_method in [1,2]:
            #Calculate sse for sim data
            y_sim = calc_sse(CaseStudyParameters, self, method, sim_data, exp_data)
        else:
            #Calculate y_sim for sim data
            y_sim = calc_y_sim(CaseStudyParameters, self, sim_data, exp_data)
            
        return y_sim 
      
class Method_name_enum(Enum):
    """
    The base class for any GPBO Method names
    
    """
    A1 = 1
    B1 = 2
    A2 = 3
    B2 = 4
    C2 = 5
    #Note use Method_name_enum.enum.name to call "A1"
    
class Gen_meth_enum(Enum):
    """
    The base class for any GPBO Method names
    
    """
    LHS = 1
    MESHGRID = 2
    
class Obj_enum(Enum):
    """
    The base class for any GPBO Method names
    
    """
    OBJ = 1
    LN_OBJ = 2
    
class CS_name_enum(Enum):
    """
    The base class for any GPBO case study names
    
    """
    CS1 = 1
    CS2 = 2

class GPBO_Methods:
    """
    The base class for any GPBO Method
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, method_name):
        """
        Parameters
        ----------
        method_name, Class, The name associated with the method being tested. Enum type
        emulator: bool, Determines if GP will model the function or the function error
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        sparse_grid: bool, Determines whether a sparse grid or approximation is used for the GP emulator
        """
        # Constructor method
        self.method_name = method_name
        self.emulator = self.get_emulator()
        self.obj = self.get_obj()
        self.sparse_grid = self.get_sparse_grid()
        
    def get_emulator(self):
        """
        Function to get emulator status based on method name
        
        Returns:
        --------
        emulator, bool: Status of whether the GP emulates the function directly
        """
        
        if "2" in self.method_name.name:
            emulator = True
        else:
            emulator = False
        
        return emulator
    
    def get_obj(self):
        """
        Function to get objective function status based on method name
        
        Returns:
        --------
        obj_enum, class instance: Determines whether log scaling is used
        """
        if "B" in self.method_name.name:
            obj = Obj_enum(2)
        else:
            obj = Obj_enum(1)
        return obj
    
    def get_sparse_grid(self):
        """
        Function to get sparse grid status based on method name
        
        Returns:
        --------
        sparse_grid: bool, Determines whether a sparse grid is used to evaluate the EI integral
        """
        if "C" in self.method_name.name:
            sparse_grid = True
        else:
            sparse_grid = False
        
        return sparse_grid
        
class Data:
    """
    The base class for any Data used in this workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, theta_vals, x_vals, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u, y_vals, gp_mean, gp_var, sse, ei):
        """
        Parameters
        ----------
        theta_vals: ndarray, The arrays of theta_values
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental y data
        bounds_theta_l: list, lower bounds of theta
        bounds_x_l: list, lower bounds of x
        bounds_theta_u: list, upper bounds of theta
        bounds_x_u: list, upper bounds of x
        gp_mean: ndarray, GP mean prediction values associated with theta_vals and x_vals
        gp_var: ndarray, GP variance prediction values associated with theta_vals and x_vals
        sse: ndarray, sum of squared error values associated with theta_vals and x_vals
        ei: ndarray, expected improvement values associated with theta_vals and x_vals
        """
        # Constructor method
        self.theta_vals = theta_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.bounds_theta = np.array([bounds_theta_l, bounds_theta_u])
        self.bounds_x = np.array([bounds_x_l, bounds_x_u])        
        self.sse = sse
        self.ei = ei
        self.gp_mean = gp_mean
        self.gp_var = gp_var
#         self.hyperparams = hyperparams
        
    
    def get_num_theta(self):
        """
        Defines the total number of theta data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_theta_data: int, the number of data the GP will have access to
        """
        num_theta_data = len(self.theta_vals)
        
        return num_theta_data
    
    def get_dim_theta(self):
        """
        Defines the total dimensions of theta data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_theta_data: int, the number of data the GP will have access to
        """
        if len(self.theta_vals) == 1:
            theta_vals = self.theta_vals.reshape(1,-1)
        else:
            theta_vals = self.theta_vals
            
        dim_theta_data = theta_vals.shape[1]
        
        return dim_theta_data
    
    def get_num_x_vals(self):
        """
        Defines the total number of x data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_x_data: int, the number of data the GP will have access to
        """
        num_x_data = len(self.x_vals)
        
        return num_x_data
    
    def get_dim_x_vals(self):
        """
        Defines the total dimensions of x data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_x_data: int, the number of data the GP will have access to
        """
        dim_x_data = vector_to_1D_array(self.x_vals).shape[1]
        
        return dim_x_data
        
    def get_num_gp_data(self, Method):
        """
        Defines the total number of data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_data: int, the number of data the GP will have access to
        """
        if Method.emulator == True:
            num_gp_data = int(self.get_num_x_vals()*self.get_num_theta())
        else:
            num_gp_data = int(self.get_num_theta())
        
        return num_gp_data
    
    def get_dim_gp_data(self, Method):
        """
        Defines the total dimension of data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_data: int, the number of data the GP will have access to
        """
        if Method.emulator == True:
            dim_gp_data = int(self.get_dim_x_vals() + self.get_dim_theta())
        else:
            dim_gp_data = int(self.get_dim_theta())
        
        return dim_gp_data
    
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
    eval_gp
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
    
    def calc_best_error(self, method, CaseStudyParameters, sim_data, exp_data):
        """
        Calculates the best error of the model
        
        Parameters
        ----------
        method: class, fully defined methods class which determines which method will be used
        CaseStudyParameters: class, class containing at least the true_model_coefficients
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        
        Returns
        -------
        best_error: float, the best error of the method
        
        """
        
    def eval_gp(self, param_set):
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
        
class Type_2_GP_Emulator():
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    eval_gp
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
class Expected_Improvement():  
    """
    The base class for acquisition functions
    Parameters
    
    Methods
    --------------
    __init__
    type_1_ei
    type_2_ei
    """
    #AD Comment: What part of the acquisition function code can be generalized and what is specific to type1 and type2? 
    def __init__(self, ep, gp_mean, gp_var, CaseStudyParameters):
        """
        Parameters
        ----------
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        ep: float, the exploration parameter of the function  
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_var: tensor, The GP model's variance evaluated over param_set
        """
        
        # Constructor method
        self.CaseStudyParameters = CaseStudyParameters
        self.ep = ep
        self.gp_mean = gp_mean
        self.gp_var = gp_var
    
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
             
class GPBO_Driver:
    """
    The base class for running the GPBO Workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, CaseStudyParameters, Simulator, exp_data, sim_data):
        """
        Parameters
        ----------
        Simulator: Class, class containing values associated with simulation parameters not covered in the GPBO_Method class
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        exp_data: Instance of Data class, Class containing at least bounds for experimental data
        sim_data: Instance of Data class
        
        """
        # Constructor method
        self.CaseStudyParameters = CaseStudyParameters
        self.Simulator = Simulator
        self.exp_data = exp_data
        self.sim_data = sim_data
        
    def gen_exp_data(self, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u, num_x_data, gen_meth_x):
        """
        Generates experimental data in an instance of the Data class
        
        Parameters
        ----------
        bounds_theta_l: list, lower bounds of theta
        bounds_x_l: list, lower bounds of x
        bounds_theta_u: list, upper bounds of theta
        bounds_x_u: list, upper bounds of x
        num_x_data: int, number of experiments
        gen_meth_x: bool: Whether to generate X data with LHS or grid method
        
        Returns:
        --------
        exp_data: instance of a class filled in with experimental x and y data along with parameter bounds
        """
        theta_true = self.Simulator.theta_true
        exp_data = Data(theta_true, None, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u, None, None, None, None, None)
        exp_data.x_vals = vector_to_1D_array(self.create_param_data(num_x_data, exp_data.bounds_x, gen_meth_x))
        exp_data.y_vals = self.create_y_exp_data(exp_data)
        
        self.exp_data = exp_data
        
        return exp_data
        
        
    def gen_sim_data(self, num_theta_data, gen_meth_theta, num_x_data, gen_meth_x, share_x_from_exp = True):
        """
        Generates experimental data in an instance of the Data class
        
        Parameters
        ----------
        num_theta_data: int, number of theta values
        gen_meth_theta: bool: Whether to generate theta data with LHS or grid method
        num_x_data: int, number of experiments
        gen_meth_x: bool: Whether to generate X data with LHS or grid method
        share_x_from_exp:bool, whether to take same x as xexp data. Default True
        
        Returns:
        --------
        sim_data: instance of a class filled in with experimental x and y data along with parameter bounds
        """
        bounds_theta = self.exp_data.bounds_theta
        bounds_x = self.exp_data.bounds_x
        
        if not share_x_from_exp:
            x_data = vector_to_1D_array(self.create_param_data(num_x_data, bounds_x, gen_meth_x))
        else:
            x_data = self.exp_data.x_vals
     
        sim_data = Data(None, x_data, bounds_theta[0], bounds_x[0], bounds_theta[1], bounds_x[1], None, None, None, None, None)
        sim_data.theta_vals = vector_to_1D_array(self.create_param_data(num_theta_data, bounds_theta, gen_meth_theta))
        sim_data.y_vals = self.create_y_exp_data(self.exp_data)
        
        self.sim_data = sim_data
        
        return sim_data
        
        
    def create_param_data(self, n_points, bounds, gen_meth):
        """
        Generates data based off of bounds, and an LHS generation number
        
        Parameters
        ----------
        n_points: int, number of data to generate
        bounds: array, array of parameter bounds
        gen_meth: class (Gen_meth_enum), ("LHS", "Meshgrid"). Determines whether x data will be generated with an LHS or meshgrid
        
        Returns:
        --------
        x_data: ndarray, a list of x data
        
        Notes: Meshgrid generated data will output n_points in each dimension, LHS generates n_points of data
        """        
        seed = self.CaseStudyParameters.seed
        dimensions = bounds.shape[1] #Want to do it this way to make it general for either x or theta parameters
        
        if gen_meth.value == 2:
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
            
        elif gen_meth.value == 1:
            #Generate LHS sample
            data = lhs_design(n_points, dimensions, seed, bounds = bounds)
        
        else:
            raise ValueError("gen_meth.value must be 1 or 2!")
#             assert self.CaseStudyParameters.x_data_vals is not None, "X must be provided if not generated"
#             assert self.CaseStudyParameters.x_data_vals.shape[1] == exp_d, "Provided X values must have the same dimension as bounds!"
#             sim_data = self.CaseStudyParameters.x_data_vals
        
        #How do I make the data an instance of the data class?
        return data
        
    def create_y_exp_data(self, exp_data):
        """
        Creates experimental y data based on x, theta_true, and the case study
        
        Parameters:
        -----------
        exp_data: instance of a class. Contains at least the experimental x data
        
        Returns:
        --------
        Yexp: ndarray. Value of y given state points x and theta_true
        """        
        CaseStudyParameters = self.CaseStudyParameters
        Simulator = self.Simulator
        
        #Note - Can only use this function after generating x data
        y_exp = calc_y_exp(CaseStudyParameters, Simulator, exp_data)   
        
        return y_exp
            
    def train_test_split(self, sim_data):
        """
        Splits data into training and testing data

        Parameters
        ----------
            sim_data: class, The simulated parameter space and y data
        Returns:
            train_data: ndarray, The training data
            test_data: ndarray, The testing data

        """
        sep_fact = self.CaseStudyParameters.sep_fact
        seed = self.CaseStudyParameters.seed
        
        #Assert statements check that the types defined in the doctring are satisfied and sep_fact is between 0 and 1 
        assert isinstance(sep_fact, (float, int))==True or torch.is_tensor(sep_fact)==True, "Separation factor must be a float or int"
        assert 0 <= sep_fact <= 1, "Separation factor must be between 0 and 1"
        
        theta_data = sim_data.theta_vals
        x_data = sim_data.y_vals
        y_data = sim_data.y_vals

        #Shuffles Random Data
        if shuffle_seed is not None:
            #Set seed to number specified by shuffle seed
            np.random.seed(shuffle_seed)

        #How do I shuffle and split data given that my "data points" are not actually stored in arrays?
        
        return train_data, test_data
    
    def train_GP(self, train_data, verbose = False):
        """
        Trains the GP model
        
        Parameters
        ----------
        train_data: instance of Data class, The training data
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