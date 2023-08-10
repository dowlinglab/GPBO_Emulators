import numpy as np
import random
import math
from scipy.stats import norm
from scipy import integrate
# import torch
# import csv
# import gpytorch
import scipy.optimize as optimize
# import pandas as pd
# import os
# import time
import Tasmanian
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.stats import qmc
import pandas as pd
from enum import Enum

from .GPBO_Class_fxns import vector_to_1D_array, calc_muller, calc_cs1_polynomial
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

class Method_name_enum(Enum):
    """
    The base class for any GPBO Method names
    
    """
    #Ensure that only values 1 to 5 are chosen
    if Enum in range(1, 6) == False:
        raise ValueError("There are only five options for Enum: 1 to 5")
        
    A1 = 1
    B1 = 2
    A2 = 3
    B2 = 4
    C2 = 5
    #Note use Method_name_enum.enum.name to call "A1"

class Kernel_enum(Enum):
    """
    Base class for kernel choices
    """
    #Check that values are only 1 to 3
    if Enum in range(1, 4) == False:
        raise ValueError("There are only three options for Enum: 1 to 3")
        
    MAT_52 = 1
    MAT_32 = 2
    RBF = 3
    
class Gen_meth_enum(Enum):
    """
    The base class for any GPBO Method names
    
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 to 2")
        
    LHS = 1
    MESHGRID = 2
    
class Obj_enum(Enum):
    """
    The base class for any GPBO Method names
    
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 to 2")
        
    OBJ = 1
    LN_OBJ = 2
    
class CS_name_enum(Enum):
    """
    The base class for any GPBO case study names
    
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 to 2")
        
    CS1 = 1
    CS2 = 2
    
class Ep_enum(Enum):
    """
    The base class for any Method for calculating the decay of the exploration parameter
    
    """
    #Ensure that only values 1 to 5 are chosen
    if Enum in range(1, 4) == False:
        raise ValueError("There are only four options for Enum: 1 to 4")
        
    CONSTANT = 1
    DECAY = 2
    BOYLE = 3
    JASRASARIA = 4

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
        #Objective function uses emulator GP if class 2
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
        
        #Objective function is ln_obj if it includes the letter B
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
        #GP EMulator uses sparse grid if it contains C
        if "C" in self.method_name.name:
            sparse_grid = True
        else:
            sparse_grid = False
        
        return sparse_grid

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
    
    def __init__(self, cs_name, ep0, sep_fact, normalize, eval_all_pairs, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime, seed, ei_tol = 1e-6):
        """
        Parameters
        ----------
        cs_name: Class, The name/enumerator associated with the case study being evaluated   
        ep0: float, The original  exploration bias. Default 1
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, Determines whether feature data will be normalized for problem analysis
        eval_all_pairs: bool, determines whether all pairs of theta are evaluated to create heat maps. Default False
        noise_mean:float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise
        bo_iter_tot: int, total number of BO iterations per restart
        bo_run_tot: int, total number of BO algorithm restarts
        save_fig: bool, Determines whether figures will be saved. Default False
        save_data: bool, Determines whether data will be saved. Default True
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        seed: int or None, Determines seed for randomizations. None if seed is random
        ei_tol: float, ei at which to be terminate algorithm
        
        """
        #Assert statements
        #Check for strings
        assert isinstance(cs_name, (Enum, str)) == True, "cs_name must be a string or Enum" #Will figure this one out later
        #Check for float/int
        assert all(isinstance(var, (float,int)) for var in [sep_fact,ep0]) == True, "sep_fact and ep0 must be float or int"
        #Check for sep fact number between 0 and 1
        assert 0 <= sep_fact <= 1, "Separation factor must be between 0 and 1"
        #Chrck for bool
        assert all(isinstance(var, (bool)) for var in [normalize, eval_all_pairs, save_fig, save_data]) == True, "normalize, eval_all_pairs, save_fig, and save_data must be bool"
        #Check for int
        assert all(isinstance(var, (int)) for var in [bo_iter_tot, bo_run_tot, seed]) == True, "bo_iter_tot, bo_run_tot, and seed must be int"
        #Check for > 0
        assert all(var > 0 for var in [bo_iter_tot, bo_run_tot, seed]) == True, "bo_iter_tot, bo_run_tot, and seed must be > 0"
        #Check for str or None
        assert isinstance(DateTime, (str)) == True or DateTime == None, "DateTime must be str or None"
        
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
        self.seed = seed
        #Set seed
        if  self.seed != None:
            assert isinstance(self.seed, int) == True, "Seed number must be an integer or None"
            random.seed(self.seed)
        self.ei_tol = ei_tol

#I'm having trouble defining how to update num_x_data, num_theta_data and dim_x depending on the situation. Especially when adding new data or when using the meshgrid options
class Simulator:
    """
    The base class for differet simulators. Defines a simulation
    """
    def __init__(self, indecies_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u, noise_mean, noise_std, case_study_params, calc_y_fxn):
        """
        Parameters
        ----------
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        theta_ref: ndarray, The array containing the true values of problem constants
        theta_names: list, list of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        bounds_theta_l: list, lower bounds of theta
        bounds_x_l: list, lower bounds of x
        bounds_theta_u: list, upper bounds of theta
        bounds_x_u: list, upper bounds of x
        noise_mean:float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise
        case_study_params: instance of the CaseStudyParameters class
        calc_y_fxn: function, The function to calculate ysim data with
        """
        #Check for float/int
        assert all(isinstance(var,(float,int)) for var in [noise_std, noise_mean]) == True, "noise_mean and noise_std must be int or float"
        #Check for list or ndarray
        list_vars = [indecies_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u]
        assert all(isinstance(var,(list,np.ndarray)) for var in list_vars) == True, "indecies_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must be list or np.ndarray"
        #Check for list lengths > 0
        assert all(len(var) > 0 for var in list_vars) == True, "indecies_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must have length > 0"
        #Check that bound_x and bounds_theta have same lengths
        assert len(bounds_theta_l) == len(bounds_theta_u) and len(bounds_x_l) == len(bounds_x_u), "bounds lists for x and theta must be same length"
        #Check indecies to consider in theta_ref
        assert all(0 <= idx <= len(theta_ref)-1 for idx in indecies_to_consider)==True, "indecies to consider must be in range of theta_ref"
        #How to write assert statements for case_study_params and calc_y_fxn
        
        # Constructor method
        self.dim_x = len(bounds_x_l)
        self.dim_theta = len(indecies_to_consider) #Length of theta is equivalent to the number of indecies to consider
        self.indecies_to_consider = indecies_to_consider
        self.theta_ref = theta_ref
        self.theta_names = theta_names
        self.theta_true, self.theta_true_names = self.__set_true_params()
        #How to acount for this in the doctring?
        self.bounds_theta = np.array([bounds_theta_l, bounds_theta_u])
        self.bounds_theta_reg = self.bounds_theta[:,self.indecies_to_consider] #This is the theta_bounds for parameters we will regress
        self.bounds_x = np.array([bounds_x_l, bounds_x_u])
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.case_study_params = case_study_params
        self.calc_y_fxn = calc_y_fxn
        
    def __set_true_params(self):
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
        #Define theta_true and theta_true_names from theta_ref, theta_names, and indecies to consider
        true_params = self.theta_ref[self.indecies_to_consider]
        true_param_names = [self.theta_names[idx] for idx in self.indecies_to_consider]
        
        return true_params, true_param_names
    
    def __grid_sampling(self, num_points, bounds):
        """
        Generates Grid sampled data
        
        Parameters
        ----------
        num_points: int, number of points in LHS, should be greater than # of dimensions
        bounds: ndarray, array containing upper and lower bounds of elements in LHS sample. Defaults of 0 and 1
        
        Returns:
        ----------
        grid_data: ndarray, (num_points)**bounds.shape[1] grid sample of data
        
        """
        #Generate mesh_grid data for theta_set in 2D
        #Define linspace for theta
        params = np.linspace(0,1, num_points)
        #Define dimensions of parameter
        dimensions = bounds.shape[1]
        #Generate the equivalent of all meshgrid points
        df = pd.DataFrame(list(itertools.product(params, repeat=dimensions)))
        df2 = df.drop_duplicates()
        scaled_data = df2.to_numpy()
        #Normalize to bounds 
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        grid_data = scaled_data*(upper_bound - lower_bound) + lower_bound 
        return grid_data
    
    def __lhs_sampling(self, num_points, bounds, seed):
        """
        Design LHS Samples

        Parameters
        ----------
            num_points: int, number of points in LHS, should be greater than # of dimensions
            bounds: ndarray, array containing upper and lower bounds of elements in LHS sample. Defaults of 0 and 1
            seed: int, seed of random generation

        Returns
        -------
            LHS: ndarray, array of LHS sampling points with length (num_points) 
        """
        #Define number of dimensions
        dimensions = bounds.shape[1]
        #Define sampler
        sampler = qmc.LatinHypercube(d=dimensions, seed = seed)
        lhs_data = sampler.random(n=num_points)

        #Generate LHS data given bounds
        lhs_data = qmc.scale(lhs_data, bounds[0], bounds[1]) #Using this because I like that bounds can be different shapes

        return lhs_data
    
    def __create_param_data(self, num_points, bounds, gen_meth, seed):
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
        #Set dimensions 
        dimensions = bounds.shape[1] #Want to do it this way to make it general for either x or theta parameters
        
        #Decide on a method to use based on gen_meth_value. LHS or Grid
        if gen_meth.value == 2:
            data = self.__grid_sampling(num_points, bounds) 
            
        elif gen_meth.value == 1:
            #Generate LHS sample
            data =  self.__lhs_sampling(num_points, bounds, seed)
        
        else:
            raise ValueError("gen_meth.value must be 1 or 2!")
        
        return data
    
    def gen_y_data(self, data, noise_mean, noise_std):
        """
        Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
        Parameters
        ----------
            CaseStudyParameters: class, class containing at least the theta_true, x_data, noise_mean, noise_std, and seed
            Simulator: Class, class containing at least calc_y_fxn
            sim_data: Class, Class containing at least the theta_vals for simulation

        Returns
        -------
            y_data: ndarray, The simulated y training data
        """        
        #Set seed
        if self.case_study_params.seed is not None:
            np.random.seed(self.case_study_params.seed)
        #Define an array to store y values in
        y_data = []
        #Get number of points
        len_points = data.get_num_theta()
        #Loop over all theta values
        for i in range(len_points):
            #Create model coefficient from true space substituting in the values of param_space at the correct indecies
            model_coefficients = self.theta_ref.copy()
            #Replace coefficients a specified indecies with their theta_val counterparts
            model_coefficients[self.indecies_to_consider] = data.theta_vals[i]
            #Create model coefficients
            y_data.append(self.calc_y_fxn(model_coefficients, data.x_vals[i])) 

        #Convert list to array and flatten array
        y_data = np.array(y_data).flatten()

        #Creates noise values with a certain stdev and mean from a normal distribution
        noise = np.random.normal(size=len(y_data), loc = noise_mean, scale = noise_std)
        
        #Add noise to data
        y_data = y_data + noise

        return y_data
       
    def gen_exp_data(self, num_x_data, gen_meth_x):
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
        #check that num_data > 0 
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError('num_x_data must be a positive integer')
            
        #Create x vals based on bounds and num_x_data
        x_vals = vector_to_1D_array(self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, self.case_study_params.seed))
        #Reshape theta_true to correct dimensions and stack it once for each xexp value
        theta_true = self.theta_true.reshape(1,-1)
        theta_true_repeated = np.vstack([theta_true]*len(x_vals))
        #Create exp_data class and add valies
        exp_data = Data(theta_true_repeated, x_vals, None, None, None, None, None, None, self.bounds_theta_reg, self.bounds_x)
        #Generate y data for exp_data calss instance
        exp_data.y_vals = self.gen_y_data(exp_data, self.noise_mean, self.noise_std)
        
        return exp_data
    
    def gen_sim_data(self, num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, gen_val_data = False):
        """
        Generates experimental data in an instance of the Data class
        
        Parameters
        ----------
        num_theta_data: int, number of theta values
        gen_meth_theta: bool: Whether to generate theta data with LHS or grid method
        num_x_data: int, number of experiments
        gen_meth_x: bool: Whether to generate X data with LHS or grid method
        gen_val_data: bool, Whether validation data (no y vals) or simulation data (has y vals) will be generated. 
        
        
        Returns:
        --------
        sim_data: instance of a class filled in with experimental x and y data along with parameter bounds
        """
        if isinstance(gen_val_data, bool) == False:
            raise ValueError('gen_val_data must be bool')
            
        #Chck that num_data > 0
        if num_theta_data <= 0 or isinstance(num_theta_data, int) == False:
            raise ValueError('num_theta_data must be a positive integer')
            
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError('num_x_data must be a positive integer')
        
        #Set bounds on theta which we are regressing given bounds_theta and indecies to consider
        #X data we always want the same between simulation and validation data
        x_data = vector_to_1D_array(self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, self.case_study_params.seed))
            
        #Infer how many times to repeat theta and x values given whether they were generated by LHS or a meshgrid
        #X and theta repeated at least once per time the other is generated
        repeat_x = num_theta_data
        repeat_theta = len(x_data)
        
        #If using a meshgrid this number is exponentiated by the number of dimensions of itself
        if gen_meth_theta.value == 2:
            repeat_x = num_theta_data**(self.dim_theta)
        if gen_meth_x.value == 2:
            repeat_theta = num_x_data**(self.dim_x)
            
        #Warn user if >5000 pts generated
        if repeat_x*repeat_theta > 5000:
            raise Warning("More than 5000 points will be generated!")
     
        #Generate all rows of simulation data
        sim_data = Data(None, None, None, None, None, None, None, None, self.bounds_theta_reg, self.bounds_x)
       
        #For validation theta, change the seed by 1 to ensure validation and sim data are never the same
        if gen_val_data == False:
            seed = self.case_study_params.seed
        else:
            seed = int(self.case_study_params.seed + 1)
            
        #Generate simulation data theta_vals and create instance of data class   
        sim_theta_vals = vector_to_1D_array(self.__create_param_data(num_theta_data, self.bounds_theta_reg, gen_meth_theta, seed))
        
        #Add repeated theta_vals and x_data to sim_data
        sim_data.theta_vals = np.repeat(sim_theta_vals, repeat_theta , axis =0)
        sim_data.x_vals = np.vstack([x_data]*repeat_x)
        
        #Add y_vals for sim_data only
        if gen_val_data == False:
            sim_data.y_vals = self.gen_y_data(sim_data, 0, 0)
        
        return sim_data
   
    def sim_data_to_sse_sim_data(self, method, sim_data, exp_data, gen_val_data = False):
        """
        Creates simulation data based on x, theta_vals, the GPBO method, and the case study

        Parameters
        ----------
        method: class, fully defined methods class which determines which method will be used
        sim_data: Class, Class containing at least the theta_vals, x_vals, and y_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        gen_val_data: bool, Whether validation data (no y vals) or simulation data (has y vals) will be generated.

        Returns:
        --------
        sse: ndarray. Value of sse given state points x and theta_vals
        """
        if isinstance(gen_val_data, bool) == False:
            raise ValueError('gen_val_data must be bool')
            
        #Find length of theta and x in data arrays
        len_theta = sim_data.get_num_theta()
        len_x = exp_data.get_num_x_vals()
      
        #Q: For this dataset does it make more sense to have all theta and x values or just the unique thetas and x values?
        #A: Just the unique ones. No need to store extra data if we won't use it and it will be saved somewhere else regardless
        #Assign unique theta indecies and create an array of them
        unique_indexes = np.unique(sim_data.theta_vals, axis = 0, return_index=True)[1]
        unique_theta_vals = np.array([sim_data.theta_vals[index] for index in sorted(unique_indexes)])
        #Add the unique theta_vals and exp_data x values to the new data class instance
        sim_sse_data = Data(unique_theta_vals, exp_data.x_vals, None, None, None, None, None, None, self.bounds_theta, self.bounds_x)
        
        if gen_val_data == False:
            #Make sse array equal length to the number of total unique thetas
            sum_error_sq = []
            #Define all y_sims
            y_sim = sim_data.y_vals
            #Iterates over evey combination of theta to find the SSE for each combination
            #Note to do this Xexp and X **must** use the same values
            for i in range(0, len_theta, len_x):
                sum_error_sq.append(sum((y_sim[i:i+len_x] - exp_data.y_vals)**2))#Scaler

            sum_error_sq = np.array(sum_error_sq)

            #objective function only explicitly log if using 1B
            if method.method_name.name == "B1":
                sum_error_sq = np.log(sum_error_sq) #Scaler

            #Add y_values to data class instance
            sim_sse_data.y_vals = sum_error_sq
        
        return sim_sse_data         
        
        
class Data:
    """
    The base class for any Data used in this workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, sse_var, ei, bounds_theta, bounds_x):
        """
        Parameters
        ----------
        theta_vals: ndarray, The arrays of theta_values
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental y data
        gp_mean: ndarray, GP mean prediction values associated with theta_vals and x_vals
        gp_var: ndarray, GP variance prediction values associated with theta_vals and x_vals
        sse: ndarray, sum of squared error values associated with theta_vals and x_vals
        ei: ndarray, expected improvement values associated with theta_vals and x_vals
        """
        list_vars = [theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, ei]
        assert all(isinstance(var, np.ndarray) or var is None for var in list_vars), "theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, and ei must be list, np.ndarray, or None"
        # Constructor method
        self.theta_vals = theta_vals
        self.x_vals = x_vals
        self.y_vals = y_vals        
        self.sse = sse
        self.sse_var = sse_var
        self.ei = ei
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.bounds_theta = bounds_theta
        self.bounds_x = bounds_x
        
    
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
        assert self.theta_vals is not None, "theta_vals must be defined"
        num_theta_data = len(self.theta_vals)
        
        return num_theta_data
    
    def __get_unique(self, all_vals):
        """
        Gets unique instances of a certain type of data
        
        Parameters:
        -----------
        all_vals: ndarray, array of parameters with duplicates
        
        Returns:
        unique_vals: ndarray, array of parameters without duplicates
        """
        
        unique_indexes = np.unique(all_vals, axis = 0, return_index=True)[1]
        unique_vals = np.array([all_vals[index] for index in sorted(unique_indexes)])
        
        return unique_vals
    
    def get_unique_theta(self):
        """
        Defines the unique theta data in an array of theta_vals
        
        Returns:
        unique_theta_vals: ndarray, array of unique theta vals 
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        unique_theta_vals = self.__get_unique(self.theta_vals)
        return unique_theta_vals
    
    def get_unique_x(self):
        """
        Defines the unique x data in an array of x_vals
        
        Returns:
        unique_x_vals: ndarray, array of unique x vals 
        """
        assert self.x_vals is not None, "x_vals must be defined"
        unique_x_vals = self.__get_unique(self.x_vals)
        return unique_x_vals
        
    
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
        assert self.theta_vals is not None, "theta_vals must be defined"
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
        assert self.x_vals is not None, "x_vals must be defined"
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
        assert self.x_vals is not None, "x_vals must be defined"
        dim_x_data = vector_to_1D_array(self.x_vals).shape[1]
        
        return dim_x_data
    
    def norm_feature_data(self):
        """
        Normalizes all feature data. Only call this method on unscaled data
        
        Returns
        -------
        scaled_data: Instance of Data class containing all data in original class with scaled feature values
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        assert self.x_vals is not None, "x_vals must be defined"
        assert self.bounds_theta is not None, "bounds_theta must be defined"
        assert self.bounds_x is not None, "bounds_x must be defined"
        
        scaled_theta_vals = self.__normalize(self.theta_vals, self.bounds_theta)
        scaled_x_vals = self.__normalize(self.x_vals, self.bounds_x)
        
        scaled_data = Data(scaled_theta_vals, scaled_x_vals, self.y_vals, self.gp_mean, self.gp_var, self.sse, self.sse_var, self.ei, self.bounds_theta, self.bounds_x) 
        
        return scaled_data
    
    def unnorm_feature_data(self):
        """
        Normalizes all feature data and stores it in a new instance of the data class. Only call this method on scaled data

        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        assert self.x_vals is not None, "x_vals must be defined"
        assert self.bounds_theta is not None, "bounds_theta must be defined"
        assert self.bounds_x is not None, "bounds_x must be defined"
        
        reg_theta_vals = self.__unnormalize(self.theta_vals, self.bounds_theta)
        reg_x_vals = self.__unnormalize(self.x_vals, self.bounds_x)
        
        unscaled_data = Data(reg_theta_vals, reg_x_vals, self.y_vals, self.gp_mean, self.gp_var, self.sse, self.sse_var, self.ei, self.bounds_theta, self.bounds_x) 
        
        return unscaled_data
    
    def __normalize(self, data, bounds):
        """
        Normalizes data between 0 and 1

        Parameters
        ----------
        data: ndarray: The data you want to scale
        bounds: ndarray, The bounds of the type of data you want to normalize
        
        Returns:
        ---------
        scaled_data: ndarray, the data normalized between 1 and 0 based on the bounds
        """
        #Define lower/upper bounds
        bounds = vector_to_1D_array(bounds)
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        #Scale data
        scaled_data = (data - lower_bound) / (upper_bound - lower_bound)
        
        return scaled_data
    
    def __unnormalize(self, scaled_data, bounds):
        """
        Normalizes data back to original values 
        
        Parameters
        ----------
        scaler: MinMaxScaler(), The scaler used to normalize the data
        bounds: ndarray, The bounds of the type of data you want to normalize
        
        Returns
        ---------
        data: ndarray, the original data renormalized based on the original bounds
        """
        #Define upper/lower bounds
        bounds = vector_to_1D_array(bounds)
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        #scale data
        data = scaled_data*(upper_bound - lower_bound) + lower_bound
        
        return data
    
    def train_test_idx_split(self, cs_params):
        """
        Splits data indecies into training and testing indecies

        Parameters
        ----------
            cs_params: class, class containing at least the separation factor and seed
        Returns:
        --------
            train_idx: ndarray, The training theta data identifiers
            test_idx: ndarray, The testing theta data identifiers

        """
        assert self.theta_vals is not None, "data must have theta_vals"
        #Define sep_fact and shuffle_seed
        sep_fact = cs_params.sep_fact
        shuffle_seed = cs_params.seed

        #Find number of unique thetas and calculate length of training data
        len_theta = len(self.get_unique_theta())
        len_train_idc = int(len_theta*sep_fact)
        
        #Create an index for each theta
        all_idx = np.arange(0,len_theta)

        #Shuffles Random Data. Will calling this once in case study parameters mean I don't need this? (No. It doesn't)
        if shuffle_seed is not None:
            #Set seed to number specified by shuffle seed
            random.seed(shuffle_seed)
            
        #Shuffle all_idx data in such a way that theta values will be randomized
        #Set train test indecies
        random.shuffle(all_idx)
        train_idx = all_idx[:len_train_idc]
        test_idx = all_idx[len_train_idc:]
        
        return train_idx, test_idx

class GP_Emulator:
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    eval_gp
    """
    # Class variables and attributes
    
    def __init__(self, gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, __feature_train_data, __feature_test_data, __feature_val_data, __feature_cand_data):
        """
        Parameters
        ----------
        gp_sim_data: instance of Data class, GP data containg y_vals
        gp_val_data: instance of Data class, GP data for evaluation
        kernel: enum class instance, Determines which GP Kerenel to use
        lenscl: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        outputscl: float or None, Determines value of outputscale
        retrain_GP: int, number of times to restart GP training
        seed: int or None, random seed
        
        """
        #Assert statements
        #Check for int/float
        assert all(isinstance(var, (float,int)) or var is None for var in [lenscl, outputscl]) == True, "lenscl and outputscl must be float, int, or None"
        assert sum(value is None or value > 0 for value in [lenscl, outputscl]) >=2, "lenscl and outputscl must positive or None"
        #CHeck for int
        assert isinstance(retrain_GP, int) == True, "retrain_GP must be int"
        #Check for > 0
        assert all(var >= 0 for var in [retrain_GP]) == True, "retrain_GP must be greater than or equal to 0"
        #Check for Enum
        assert isinstance(kernel, Enum) == True, "kernel must be type Enum"
        #Check for instance of Data class or None
        assert isinstance(gp_sim_data, (Data)) == True or gp_sim_data == None, "gp_sim_data must be an instance of the Data class or None"
        assert isinstance(gp_val_data, (Data)) == True or gp_sim_data == None, "gp_sim_data must be an instance of the Data class or None"
        
        # Constructor method
        self.gp_sim_data = gp_sim_data
        self.gp_val_data = gp_val_data
        self.cand_data = cand_data
        self.kernel = kernel
        self.lenscl = lenscl
        self.noise_std = noise_std
        self.outputscl = outputscl
        self.retrain_GP = retrain_GP
        self.seed = seed
        self.__feature_train_data = None #Added using child class
        self.__feature_test_data = None #Added using child class
        self.__feature_val_data = None #Added using child class
        self.__feature_cand_data = None #Added using child class
        self.cand_data = cand_data
        
    def get_num_gp_data(self):
        """
        Defines the total number of data the GP will have access to to train on
        
        Parameters
        ----------
        Method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        num_data: int, the number of data the GP will have access to
        """
        
        #Number of available gp data determined by number of sim data
        num_gp_data = int(self.gp_sim_data.get_num_theta())
        
        return num_gp_data
    
    def set_gp_model(self):
        """
        Generates the GP model for the process in sklearn
        
        Parameters
        -----------
            noise_std: float, int, the noise associated with the true data
            
        Returns
        --------
            gp_model: instance of GaussianProcessRegressor in Sklearn containing kernel, optimizer, etc.
        """
        
        #Don't optimize anything if lengthscale and outputscale are being set
        if self.lenscl != None and self.outputscl != None:
            optimizer = None
        else:
            optimizer = "fmin_l_bfgs_b"
            
        kernel = self.__set_kernel()
        kernel = self.__set_lenscl(kernel)
        kernel = self.__set_outputscl(kernel)

        #Define model
        gp_model = GaussianProcessRegressor(kernel=kernel, alpha=self.noise_std**2, n_restarts_optimizer=self.retrain_GP, 
                                            random_state = self.seed, optimizer = optimizer)
        
        return gp_model
    
    def __set_kernel(self):
        """
        Sets kernel of the model
        
        Parameters
        ----------
        simulator: instance of Simulator class containinf at least the noise stdev
        
        """ 
        #Set noise kernel
        noise_kern = WhiteKernel(noise_level=self.noise_std**2, noise_level_bounds= "fixed") #bounds = "fixed"
    
        if self.kernel.value == 3: #RBF
            kernel = ConstantKernel(constant_value=1, constant_value_bounds = (1e-2,10))*RBF(length_scale_bounds=(1e-5, 1e5)) + noise_kern 
        elif self.kernel.value == 2: #Matern 3/2
            kernel = ConstantKernel(constant_value=1, constant_value_bounds = (1e-2,10))*Matern(length_scale_bounds=(1e-05, 1e5), nu=1.5) + noise_kern
        else: #Matern 5/2
            kernel = ConstantKernel(constant_value=1, constant_value_bounds = (1e-2,10))*Matern(length_scale_bounds=(1e-05, 1e5), nu=2.5) + noise_kern 
            
        return kernel
    
    def __set_lenscl(self, kernel):
        """
        Set the lengthscale of the model. Need to have training data before 
        
        Parameters
        ----------
        kernel: The kernel of the model defined by __set_kernel
        method: Instance of Method class.
        
        Returns
        -------
        kernel: The kernel of the model defined by __set_kernel with the lengthscale bounds set
        """
        #If setting lengthscale, ensure lengthscale values are fixed and that there is 1 lengthscale/dim, otherwise initialize them at 1
        if self.lenscl != None:
            assert self.lenscl > 0, "lenscl must be positive"
            lengthscale_val = np.ones(self.get_dim_gp_data())*self.lenscl
            kernel.k1.k2.length_scale_bounds = "fixed"
        else:
            lengthscale_val = np.ones(self.get_dim_gp_data())

        #Set model lengthscale
        kernel.k1.k2.length_scale = lengthscale_val
        
        return kernel
    
    def __set_outputscl(self, kernel):
        """
        Set the outputscale of the model
        
        Parameters
        ----------
        kernel: The kernel of the model defined by __set_kernel with the lengthscale bounds set
        
        Returns
        -------
        kernel: The kernel of the model defined by __set_kernel with the outputscale bounds set
        """
        #Set outputscl kernel to be optimized if necessary or set it to the default of 1 to be optimized
        if self.outputscl != None:
            assert self.outputscl> 0, "outputscl must be positive"
            kernel.k1.k1.constant_value = self.outputscl
            kernel.k1.k1.constant_value_bounds = "fixed"
        else:
            kernel.k1.k1.constant_value == 1.0
            
        return kernel
        
        
    def train_gp(self, gp_model):
        """
        Trains the GP given training data
        
        Parameters
        ----------
            gp_model: The untrained, fully defined gp model
            
        Returns
        -------
            trained_hyperparams: list, a list of the hyperparameters. Order: lenscl, noise, outputscl
            git_gp_model: GaussianProcessRegressor instance. Fit GP model
        """  
        assert self.feature_train_data is not None, "Must have training data. Run set_train_test_data() to generate"
        
        #Train GP
        fit_gp_model = gp_model.fit(self.feature_train_data, self.train_data.y_vals)

        #Pull out kernel parameters after GP training
        opt_kern_params = fit_gp_model.kernel_
        outputscl_final = opt_kern_params.k1.k1.constant_value
        lenscl_final = opt_kern_params.k1.k2.length_scale
        noise_final = opt_kern_params.k2.noise_level
        
        #Put hyperparameters in a list
        trained_hyperparams = [lenscl_final, noise_final, outputscl_final] 
        
        #Assign self parameters
        self.trained_hyperparams = trained_hyperparams
        self.fit_gp_model = fit_gp_model
        
    def __eval_gp_mean_var(self, data):
        """
        Calculates the GP mean and variance given each point and adds it to the instance of the data class
        
        Parameters:
        -----------
            data: instance of the Data class, data to evaluate GP for containing at least theta_vals and x_vals
        
        Returns:
        -------
            data: instance of the Data class, data containing at least theta_vals, x_vals, gp_mean, and gp_var
        
        """       
        #Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        gp_mean = np.zeros(len(data))
        gp_var = np.zeros(len(data))
        
        #Loop over all eval points
        for i in range(len(data)):
            eval_point = np.array([data[i]])
            #Evaluate GP given parameter set theta and state point value
            model_mean, model_std = self.fit_gp_model.predict(eval_point[0:1], return_std=True)
            model_variance = model_std**2
            #Add values to list
            gp_mean[i] = model_mean
            gp_var[i] = model_variance
                  
        return gp_mean, gp_var
    
    def eval_gp_mean_var_test(self):
        """
        Evaluate the GP mean and variance for the test set
        
        Returns:
        -------
        test_gp_mean: ndarray, array of gp_mean for the test set
        test_gp_var: ndarray, array of gp variance for the test set
        """
        
        assert self.feature_test_data is not None, "Must have testing data. Run set_train_test_data() to generate"
        #Evaluate test data for GP
        test_gp_mean, test_gp_var = self.__eval_gp_mean_var(self.feature_test_data)
        
        #Set data parameters
        self.test_data.gp_mean = test_gp_mean
        self.test_data.gp_var = test_gp_var

        return test_gp_mean, test_gp_var
    
    def eval_gp_mean_var_val(self):
        """
        Evaluate the GP mean and variance for the validation set
        
        Returns:
        -------
        val_gp_mean: ndarray, array of gp_mean for the test set
        val_gp_var: ndarray, array of gp variance for the test set
        """
        
        assert self.feature_val_data is not None, "Must have validation data. Run set_train_test_data() to generate"
        #Evaluate test data for GP
        val_gp_mean, val_gp_var = self.__eval_gp_mean_var(self.feature_val_data)
        
        #Set data parameters
        self.gp_val_data.gp_mean = val_gp_mean
        self.gp_val_data.gp_var = val_gp_var
        
        return val_gp_mean, val_gp_var
    
    def eval_gp_mean_var_cand(self):
        """
        Evaluate the GP mean and variance for the validation set
        
        Returns:
        -------
        cand_gp_mean: ndarray, array of gp_mean for the test set
        cand_gp_var: ndarray, array of gp variance for the test set
        """
        
        assert self.feature_cand_data is not None, "Must have validation data. Run set_train_test_data() to generate"
        #Evaluate test data for GP
        cand_gp_mean, cand_gp_var = self.__eval_gp_mean_var(self.feature_cand_data)
        
        #Set data parameters
        self.cand_data.gp_mean = cand_gp_mean
        self.cand_data.gp_var = cand_gp_var
#         print(self.feature_cand_data, cand_gp_mean, cand_gp_var)
        return cand_gp_mean, cand_gp_var
    
            
#https://www.geeksforgeeks.org/inheritance-and-composition-in-python/
#AD: Use composition instead of inheritance here, pass an instance of CaseStudyParameters to the init function
class Type_1_GP_Emulator(GP_Emulator):
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    eval_gp
    """
    # Class variables and attributes
    
    def __init__(self, gp_sim_data, gp_val_data, cand_data, train_data, test_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, feature_train_data, feature_test_data, feature_val_data, feature_cand_data):
        """
        Parameters
        ----------
        """
        # Constructor method
        super().__init__(gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, feature_train_data, feature_test_data, feature_val_data, feature_cand_data)
        self.train_data = train_data
        self.test_data = test_data 
        
    def get_dim_gp_data(self):
        """
        Defines the total dimension of data the GP will have access to to train on
        
        Returns
        -------
        num_data: int, the number of data the GP will have access to
        """
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have simulation theta to get a number of theta"
        
        #Just use number of theta dimensions for Type 1
        dim_gp_data = int(self.gp_sim_data.get_dim_theta())
        
        return dim_gp_data
    
    def set_train_test_data(self, cs_params):
        """
        finds the simulation data to use as training data
        Parameters
        ----------
            cs_params: Instance of CaseStudyParameters class. Contains at least sep_fact and seed
        Returns
        -------
            train_data: Instance of data class. Contains all theta, x, and y data for training data
            test_data: Instance of data class. Contains all theta, x, and y data for testing data
        """
        assert np.all(self.gp_sim_data.x_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.y_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.bounds_x is not None), "Must have simulation x bounds to create train/test data"
        assert np.all(self.gp_sim_data.bounds_theta is not None), "Must have simulation theta bounds to create train/test data"
        
        #Get train test idx
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split(cs_params)
        
        #Get train data
        theta_train = self.gp_sim_data.theta_vals[train_idx]
        x_train = self.gp_sim_data.x_vals #x_vals for Type 1 is the same as exp_data. No need to index x
        y_train = self.gp_sim_data.y_vals[train_idx]
        train_data = Data(theta_train, x_train, y_train, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x)
        self.train_data = train_data
        
        #Get test data
        theta_test = self.gp_sim_data.theta_vals[test_idx]
        x_test = self.gp_sim_data.x_vals #x_vals for Type 1 is the same as exp_data. No need to index x
        y_test = self.gp_sim_data.y_vals[test_idx]
        test_data = Data(theta_test, x_test, y_test, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x)
        self.test_data = test_data
        
        #Set training and validation data features in GP_Emulator base class
        feature_train_data = self.__featurize_data(train_data)
        feature_test_data = self.__featurize_data(test_data)
        feature_val_data = self.__featurize_data(self.gp_val_data)
        
        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data
        self.feature_val_data = feature_val_data
        
        return train_data, test_data
    
    def __featurize_data(self, data):
        """
        Calculates the GP mean and variance given each point and adds it to the instance of the data class
        
        Parameters:
        -----------
            data: instance of the Data class, data to evaluate GP for containing at least theta_vals and x_vals
        
        Returns:
        -------
            data: instance of the Data class, data containing at least theta_vals, x_vals, gp_mean, and gp_var
        
        """
        assert np.all(data.theta_vals is not None), "Must have validation data theta_vals and x_vals to evaluate the GP"
        
        #Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = data.theta_vals
        
        return feature_eval_data
       
    def eval_gp_sse_var_test(self):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for testing data
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        assert np.all(self.test_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(self.test_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        
        #For type 1, sse is the gp_mean
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data)
                    
        return test_sse_mean, test_sse_var
    
    def eval_gp_sse_var_val(self):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for validation data
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        assert np.all(self.gp_val_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.gp_val_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        
        #For type 1, sse is the gp_mean
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data)
                    
        return val_sse_mean, val_sse_var  
    
    def eval_gp_sse_var_cand(self):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for validation data
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        assert np.all(self.cand_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.cand_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        
        #For type 1, sse is the gp_mean
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data)
                    
        return cand_sse_mean, cand_sse_var
    
    def __eval_gp_sse_var(self, data):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        #For type 1, sse is the gp_mean
        sse_mean = data.gp_mean
        sse_var = data.gp_var
        
        #Set attributes
        data.sse = data.gp_mean
        data.sse_var = data.gp_var
                    
        return sse_mean, sse_var
    
    def calc_best_error(self):
        """
        Calculates the best error of the model
        
        Returns
        -------
        best_error: float, the best error of the method
        
        """   
        assert np.all(self.train_data.theta_vals is not None), "Must have simulation theta and y data to calculate best error"
        assert np.all(self.train_data.y_vals is not None), "Must have simulation theta and y data to calculate best error"
        
        #Best error is the minimum sse value of the training data for Type 1
#         lowest_sse_idx = np.argmin(self.train_data.y_vals)
#         best_error = self.train_data.theta_vals[lowest_sse_idx]
        best_error = np.min(self.train_data.y_vals)
        
        return best_error
    
    
    def __eval_gp_ei(self, sim_data, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        """
        #Call instance of expected improvement class
        ei_class = Expected_Improvement(ep_bias, sim_data.gp_mean, sim_data.gp_var, exp_data, best_error)
        #Call correct method of ei calculation
        ei = ei_class.type_1()
        #Add ei data to validation data class
        sim_data.ei = ei
        
        return ei
    
    def eval_ei_test(self, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        """
        ei = self.__eval_gp_ei(self.test_data, exp_data, ep_bias, best_error)
        return ei
    
    def eval_ei_val(self, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in gp_val_data
        """
        ei = self.__eval_gp_ei(self.gp_val_data, exp_data, ep_bias, best_error)
        
        return ei
    
    def eval_ei_cand(self, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in candidate data
        """
        ei = self.__eval_gp_ei(self.cand_data, exp_data, ep_bias, best_error)
        
        return ei
    
class Type_2_GP_Emulator(GP_Emulator):
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    eval_gp
    """
    # Class variables and attributes
    def __init__(self, gp_sim_data, gp_val_data, cand_data, train_data, test_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, feature_train_data, feature_test_data, feature_val_data, feature_cand_data):
        """
        Parameters
        ----------
        CaseStudyParameters: Class, class containing the values associated with CaseStudyParameters
        """
        # Constructor method
        super().__init__(gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, feature_train_data, feature_test_data, feature_val_data, feature_cand_data)
        self.train_data = train_data
        self.test_data = test_data
                  
    def get_dim_gp_data(self):
        """
        Defines the total dimension of data the GP will have access to to train on
        
        Returns
        -------
        num_data: int, the number of data the GP will have access to
        """
        assert np.all(self.gp_sim_data.x_vals is not None), "Must have sim data theta_vals and x_vals"
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have sim data theta_vals and x_vals"
        
        #Number of theta dimensions + number of x dimensions
        dim_gp_data = int(self.gp_sim_data.get_dim_x_vals() + self.gp_sim_data.get_dim_theta())
        
        return dim_gp_data
    
    def set_train_test_data(self, cs_params):
        """
        finds the simulation data to use as training data
        Parameters
        ----------
            cs_params: Instance of CaseStudyParameters class. Contains at least sep_fact and seed
        Returns
        -------
            train_data: Instance of data class. Contains all theta, x, and y data for training data
            test_data: Instance of data class. Contains all theta, x, and y data for testing data
        """
        assert np.all(self.gp_sim_data.x_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.y_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.bounds_x is not None), "Must have simulation x bounds to create train/test data"
        assert np.all(self.gp_sim_data.bounds_theta is not None), "Must have simulation theta bounds to create train/test data"
        
        #Find train indecies
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split(cs_params)
        
        #Find unique theta_values
        unique_theta_vals = self.gp_sim_data.get_unique_theta()
        
        # Check which rows in True_array match the rows in Theta_unique based on theta_idx
        train_mask = np.isin(self.gp_sim_data.theta_vals, unique_theta_vals[train_idx])
        test_mask = np.isin(self.gp_sim_data.theta_vals, unique_theta_vals[train_idx], invert = True)

        # Get the indices of the matching rows
        train_rows_idx = np.all(train_mask, axis=1)
        test_rows_idx = np.all(test_mask, axis=1)

        # Use the indices to select the specific rows from True_array
        theta_train = self.gp_sim_data.theta_vals[train_rows_idx]
        x_train = self.gp_sim_data.x_vals[train_rows_idx]
        y_train = self.gp_sim_data.y_vals[train_rows_idx]
        train_data = Data(theta_train, x_train, y_train, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x)
        self.train_data = train_data
        
        #Get test data
        theta_test = self.gp_sim_data.theta_vals[test_rows_idx]
        x_test = self.gp_sim_data.x_vals[test_rows_idx]
        y_test = self.gp_sim_data.y_vals[test_rows_idx]
        test_data = Data(theta_test, x_test, y_test, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x)
        self.test_data = test_data
        
        #Set training and validation data features in GP_Emulator base class
        feature_train_data = self.__featurize_data(train_data)
        feature_test_data = self.__featurize_data(test_data)
        feature_val_data = self.__featurize_data(self.gp_val_data)
        
        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data
        self.feature_val_data = feature_val_data

        return train_data, test_data
    
    def __featurize_data(self, data):
        """
        Calculates the GP mean and variance given each point and adds it to the instance of the data class
        
        Parameters:
        -----------
            data: instance of the Data class, data to evaluate GP for containing at least theta_vals and x_vals
        
        Returns:
        -------
            data: instance of the Data class, data containing at least theta_vals, x_vals, gp_mean, and gp_var
        
        """
        assert np.all(data.x_vals is not None), "Must have validation data theta_vals and x_vals to evaluate the GP"
        assert np.all(data.theta_vals is not None), "Must have validation data theta_vals and x_vals to evaluate the GP"
        
        #Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = np.concatenate((data.theta_vals, data.x_vals), axis =1)
        
        return feature_eval_data
    
    def eval_gp_sse_var_test(self, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for test data
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        assert np.all(self.test_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.test_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.test_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(self.test_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data, exp_data)
        
        return test_sse_mean, test_sse_var
    
    def eval_gp_sse_var_val(self, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for validation data
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        assert np.all(self.gp_val_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.gp_val_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.gp_val_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.gp_val_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data, exp_data)
        
        return val_sse_mean, val_sse_var
    
    def eval_gp_sse_var_cand(self, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for candidate feature data
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        cand_sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        cand_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        assert np.all(self.cand_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.cand_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.cand_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.cand_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data, exp_data)
        
        return cand_sse_mean, cand_sse_var
    
    
    def __eval_gp_sse_var(self, data, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        feature_eval_data = self.__featurize_data(data)
        
        #Find length of theta and number of unique x in data arrays
        len_theta = data.get_num_theta()
        len_x = len(data.get_unique_x())
      
        #Assign unique theta indecies and create an array of them
        unique_theta_vals = data.get_unique_theta()
     
        #Make sse array equal length to the number of total unique thetas
        sse_mean = np.zeros(len(unique_theta_vals))
        sse_var = np.zeros(len(unique_theta_vals))
        
        #Iterates over evey combination of theta to find the sse for each combination
        #Note to do this Xexp and X **must** use the same values
        sse_idx = 0 #Used to set data in array
        for i in range(0, len_theta, len_x):
            sse_mean[sse_idx] = sum((data.gp_mean[i:i+len_x] - exp_data.y_vals)**2) #Scaler 
            error_point = (data.gp_mean[i:i+len_x] - exp_data.y_vals) #This SSE_variance CAN be negative
            sse_var[sse_idx] = 2*error_point@data.gp_var[i:i+len_x] #Error Propogation approach
            sse_idx += 1
        
        #Set class parameters
        data.sse = sse_mean
        data.sse_var = sse_var
        
        return sse_mean, sse_var
    
    def calc_best_error(self, exp_data):
        """
        Calculates the best error of the model
        
        Returns
        -------
        best_error: float, the best error of the method
        
        """  
        assert np.all(self.train_data.x_vals is not None), "Must have simulation x, theta, and y data to calculate best error"
        assert np.all(self.train_data.theta_vals is not None), "Must have simulation x, theta, and y data to calculate best error"
        assert np.all(self.train_data.y_vals is not None), "Must have simulation x, theta, and y data to calculate best error"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        #Find length of theta and x in data arrays
        len_theta = self.train_data.get_num_theta()
        len_x = len(self.train_data.get_unique_x())
      
        #Assign unique theta indecies and create an array of them
        unique_theta_vals = self.train_data.get_unique_theta()
     
        #Make sse array equal length to the number of total unique thetas
        sse_train_vals = np.zeros(len(unique_theta_vals))
        true_idx_list = [] #Used for error checking
        
        sse_idx = 0 #Used to set data in array
        #Evaluate SSE by looping over the x values for each combination of theta and calculating SSE
        for i in range(0, len_theta, len_x):
            sse_train_vals[sse_idx] = sum((self.train_data.y_vals[i:i+len_x] - exp_data.y_vals)**2) #Scaler
            true_idx_list.append(i) #Used for error checking
            sse_idx += 1
                
        #Best error is the minimum of these values
        best_error = np.amin(sse_train_vals)
#         print(self.train_data.theta_vals[true_idx_list[np.argmin(sse_train_vals)]]) #For Error Checking, Returns theta associated with best value
        
        return best_error
    
    def __eval_gp_ei(self, sim_data, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        method: instance of Method class, method for GP Emulation
        
        Returns
        -------
        ei: The expected improvement of all the data in sim_data
        """
        assert method.method_name.value >= 3, "Must be using method 2A, 2B, or 2C"
        #Call instance of expected improvement class
        ei_class = Expected_Improvement(ep_bias, sim_data.gp_mean, sim_data.gp_var, exp_data, best_error)
        #Call correct method of ei calculation
        ei = ei_class.type_2(method)
        #Add ei data to validation data class
        self.gp_val_data.ei = ei
        
        return ei
    
    def eval_ei_test(self, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        method: instance of Method class, method for GP Emulation
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        """
        ei = self.__eval_gp_ei(self.test_data, exp_data, ep_bias, best_error, method)
        return ei
    
    def eval_ei_val(self, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        method: instance of Method class, method for GP Emulation
        
        Returns
        -------
        ei: The expected improvement of all the data in gp_val_data
        """
        ei = self.__eval_gp_ei(self.gp_val_data, exp_data, ep_bias, best_error, method)
        
        return ei
        
    def eval_ei_cand(self, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        method: instance of Method class, method for GP Emulation
        
        Returns
        -------
        ei: The expected improvement of all the data in candidate feature
        """
        ei = self.__eval_gp_ei(self.cand_data, exp_data, ep_bias, best_error, method)
        
        return ei
    
##Again, composition instead of inheritance      
class Expected_Improvement():  
    """
    The base class for acquisition functions
    Parameters
    
    Methods
    --------------
    __init__
    set_ep
    type_1_ei
    type_2_ei
    """
    #AD Comment: What part of the acquisition function code can be generalized and what is specific to type1 and type2? 
    def __init__(self, ep_bias, gp_mean, gp_var, exp_data, best_error):
        """
        Parameters
        ----------
        ep_bias: instance of Exploration_Bias, class with information of exploration bias parameter
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_var: tensor, The GP model's variance evaluated over param_set
        best_error: float, the smallest value of the error in the training data
        """
        assert len(gp_mean) == len(gp_var), "gp_mean and gp_var must be arrays of the same length"
        assert all(isinstance(arr, np.ndarray) for arr in (gp_mean, gp_var, exp_data.y_vals)), "gp_mean, gp_var, and exp_data.y_vals must be ndarrays"
        assert isinstance(ep_bias, Exploration_Bias), "ep_bias must be instance of Exploration_Bias"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(best_error, (float, int)), "best_error must be float or int. Calculate with GP_Emulator.calc_best_error()"
        
        # Constructor method
        self.ep_bias = ep_bias
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.exp_data = exp_data
        self.best_error = best_error
    
    def type_1(self):
        """
        Calculates expected improvement of type 1 (standard) GPBO given gp_mean, gp_var, and best_error data
        
        Returns
        -------
        ei: ndarray, The expected improvement of the parameter set
        """
        ei = np.zeros(len(self.gp_mean))

        for i in range(len(self.gp_mean)):
            pred_stdev = np.sqrt(self.gp_var[i]) #1xn_test
            #Checks that all standard deviations are positive
            if pred_stdev > 0:
                #Calculates z-score based on Ke's formula
                z = (self.best_error*self.ep_bias.ep_curr - self.gp_mean[i])/pred_stdev #scaler
                #Calculates ei based on Ke's formula
                #Explotation term
                ei_term_1 = (self.best_error*self.ep_bias.ep_curr - self.gp_mean[i])*norm.cdf(z) #scaler
                #Exploration Term
                ei_term_2 = pred_stdev*norm.pdf(z) #scaler
                ei[i] = ei_term_1 +ei_term_2 #scaler

#                 print("z",z)
#                 print("Exploitation Term",ei_term_1)
#                 print("CDF", norm.cdf(z))
#                 print("Exploration Term",ei_term_2)
#                 print("PDF", norm.pdf(z))
#                 print("EI",ei,"\n")
            else:
                #Sets ei to zero if standard deviation is zero
                ei[i] = 0
            
        return ei
        
    def type_2(self, method):
        """
        Calculates expected improvement of type 2 (emulator) GPBO
        
        Parameters:
        -----------
        method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        ei: float, The expected improvement of the parameter set
        """
        #Num thetas = #gp mean pts/number of x_vals for Type 2
        num_thetas = int(len(self.gp_mean)/self.exp_data.get_num_x_vals()) 
        #Define n as the number of x values
        n = self.exp_data.get_num_x_vals()
        #Initialize array of eis for eacch theta
        ei = np.zeros(num_thetas)
        #Loop over number of thetas in theta_val_set
        for i in range(num_thetas): #1 ei per theta and also 1 sse per theta   
            #For method 2C
            if method.method_name.value == 5: #2C
                #for ei, ensure that a gp mean and gp_var corresponding to a certain theta are sent to self.__calc_ei_sparse()
                ei[i] = self.__calc_ei_sparse(self.gp_mean[i*n:i*n+n], self.gp_var[i*n:i*n+n], self.exp_data.y_vals)
            
            elif method.method_name.value in (3,4): #2A and 2B
                #Initialize ei for specific theta
                ei_theta = 0
                #Loop over number of exp data points
                for j in range(self.exp_data.get_num_x_vals()):
                    #Calculate ei for a given theta (sum of ei for each x over all thetas)
                    if method.method_name.value == 3: #2A
                        ei_temp = self.__calc_ei_emulator(self.gp_mean[i*n+j], self.gp_var[i*n+j], self.exp_data.y_vals[j])

                    else: #2B
                        ei_temp = self.__calc_ei_log_emulator(self.gp_mean[i*n+j], self.gp_var[i*n+j], self.exp_data.y_vals[j])
                    
                    ei_theta += ei_temp

                    #Save ei to array
                    ei[i] = ei_theta            
                
            else:
                raise ValueError("method.method_name.value must be 3 (2A), 4 (2B), or 5 (2C)")
        
        return ei  
        
    def __calc_ei_emulator(self, gp_mean, gp_var, y_target): #Will need obj toggle soon
        """ 
        Calculates the expected improvement of the emulator approach without log scaling (2A)
        Parameters
        ----------
            gp_mean: ndarray, model mean at same state point x and experimental data value y
            gp_variance: ndarray, model variance at same state point x and experimental data value y
            y_target: ndarray, the expected value of the function from data or other source

        Returns
        -------
            ei: ndarray, the expected improvement for one term of the GP model
        """
        #Defines standard devaition
        pred_stdev = np.sqrt(gp_var) #1xn
        
        if pred_stdev > 0:
            #If variance is close to zero this is important
            with np.errstate(divide = 'warn'):
                #Creates upper and lower bounds and described by Alex Dowling's Derivation
                bound_a = ((y_target - gp_mean) +np.sqrt(self.best_error*self.ep_bias.ep_curr))/pred_stdev #1xn
                bound_b = ((y_target - gp_mean) -np.sqrt(self.best_error**self.ep_bias.ep_curr))/pred_stdev #1xn
                bound_lower = np.min([bound_a,bound_b])
                bound_upper = np.max([bound_a,bound_b])        

                #Creates EI terms in terms of Alex Dowling's Derivation
                ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower) #1xn
                ei_term1_comp2 = (self.best_error**self.ep_bias.ep_curr) - (y_target - gp_mean)**2 #1xn

                ei_term2_comp1 = 2*(y_target - gp_mean)*pred_stdev #1xn
                ei_eta_upper = -np.exp(-bound_upper**2/2)/np.sqrt(2*np.pi)
                ei_eta_lower = -np.exp(-bound_lower**2/2)/np.sqrt(2*np.pi)
                ei_term2_comp2 = (ei_eta_upper-ei_eta_lower)

                ei_term3_comp1 = bound_upper*ei_eta_upper #1xn
                ei_term3_comp2 = bound_lower*ei_eta_lower #1xn

                ei_term3_comp3 = (1/2)*math.erf(bound_upper/np.sqrt(2)) #1xn
                ei_term3_comp4 = (1/2)*math.erf(bound_lower/np.sqrt(2)) #1xn  

                ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp3 #1xn
                ei_term3_psi_lower = ei_term3_comp2 + ei_term3_comp4 #1xn
                
                ei_term1 = ei_term1_comp1*ei_term1_comp2 #1xn
                ei_term2 = ei_term2_comp1*ei_term2_comp2 #1xn
                ei_term3 = -gp_var*(ei_term3_psi_upper-ei_term3_psi_lower) #1xn
                print(ei_term1, ei_term2, ei_term3 )
                ei = ei_term1 + ei_term2 + ei_term3 #1xn
        else:
            ei = 0

        return ei

    def __calc_ei_log_emulator(self, gp_mean, gp_var, y_target):
        """ 
        Calculates the expected improvement of the emulator approach with log scaling (2B)
        Parameters
        ----------
            gp_mean: ndarray, model mean at same state point x and experimental data value y
            gp_variance: ndarray, model variance at same state point x and experimental data value y
            y_target: ndarray, the expected value of the function from data or other source

        Returns
        -------
            ei: ndarray, the expected improvement for one term of the GP model
        """
        #Defines standard devaition
        pred_stdev = np.sqrt(gp_var) #1xn
            
        if pred_stdev > 0:
            with np.errstate(divide = 'warn'):
                #Creates upper and lower bounds and described by Alex Dowling's Derivation
                bound_a = ((y_target - gp_mean) +np.sqrt(np.exp(self.best_error**self.ep_bias.ep_curr)))/pred_stdev #1xn
                bound_b = ((y_target - gp_mean) -np.sqrt(np.exp(self.best_error**self.ep_bias.ep_curr)))/pred_stdev #1xn
                bound_lower = np.min([bound_a,bound_b])
                bound_upper = np.max([bound_a,bound_b])
                args = (self.best_error, gp_mean, pred_stdev, y_target, self.ep_bias.ep_curr)
                ei_term_1 = (self.best_error*self.ep_bias.ep_curr)*( norm.cdf(bound_upper)-norm.cdf(bound_lower) )
                ei_term_2_out = integrate.quad(self.__ei_approx_ln_term, bound_lower, bound_upper, args = args, full_output = 1)
                ei_term_2 = (-2)*ei_term_2_out[0] 
                term_2_abs_err = ei_term_2_out[1]
                ei = ei_term_1 + ei_term_2
        else:
            ei = 0
  
        return ei

    def __ei_approx_ln_term(self, epsilon, best_error, gp_mean, gp_stdev, y_target, ep): 
        """ 
        Calculates the integrand of expected improvement of the emulator approach using the log version
        Parameters
        ----------
            epsilon: The random variable. This is the variable that is integrated w.r.t
            best_error: float, the best predicted error encountered
            gp_mean: ndarray, model mean
            gp_stdev: ndarray, model stdev
            y_target: ndarray, the expected value of the function from data or other source
            ep: float, the numerical bias towards exploration, zero is the default

        Returns
        -------
            ei_term_2_integral: ndarray, the expected improvement for term 2 of the GP model for method 2B
        """
        #Define inside term
        #In the case that this is zero, what should happen?
        inside_term = max(1e-12, abs((y_target - gp_mean - gp_stdev*epsilon)) )
        
        #Check that inside term is > numerical 0
        if inside_term > 0:
            ei_term_2_integral = math.log( inside_term )*norm.pdf(epsilon) 
            
        else:
            #If it is 0, then ei tern 2 int is negative infinity
            ei_term_2_integral = -np.inf
        
        return ei_term_2_integral

    def __calc_ei_sparse(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a sparse grid approach (2C)
        Parameters
        ----------
            gp_mean: ndarray, model mean at same state point x and experimental data value y
            gp_var: ndarray, model variance at same state point x and experimental data value y
            y_target: ndarray, the expected value of the function from data or other source
            
        Returns
        -------
            ei: ndarray, the expected improvement for one term of the GP model
        """
        #Obtain Sparse Grid points and weights
        points_p, weights_p = self.__get_sparse_grids(len(y_target),output=0,depth=3, rule='gauss-hermite', verbose = False)
        
        #Initialize EI
        ei_temp = 0
        #Loop over sparse grid weights and nodes
        for i in range(len(points_p)):
            #Initialize SSE
            sse_temp = 0
            #Loop over experimental data points
            
            for j in range(len(y_target)):
                sse_temp += (y_target[j] - gp_mean[j] - gp_var[j]*points_p[i,j])**2
    #             SSE_Temp += (Yexp[j] - GP_mean[j] - ep - GP_stdev[j]*points_p[i,j])**2 #If there is an ep, need to add
            #Apply max operator (equivalent to max[SSE_Temp - (best_error*ep),0])
            min_list = [sse_temp - (self.best_error*self.ep_bias.ep_curr),0] 
            ei_temp += weights_p[i]*(-np.min(min_list)) 
            
        return ei_temp

    def __get_sparse_grids(self, dim, output=0,depth=3, rule="gauss-hermite", verbose = False, alpha = 0):
        '''
        This function shows the sparse grids generated with different rules
        Parameters:
        -----------
            dim: int, sparse grids dimension. Default is zero
            output: int, output level for function that would be interpolated
            depth: int, depth level. Controls density of abscissa points
            rule: str, quadrature rule. Default is 'gauss-legendre'
            verbose: bool, determines Whether or not plot of sparse grid is shown. False by default
            alpha: int, specifies $\alpha$ parameter for the integration weight $\rho(x)$, ignored when rule doesn't have this parameter

        Returns:
        --------
            points_p: ndarray, The sparse grid points
            weights_p: ndarray, The Gauss-Legendre Quadrature Rule Weights    

        Other:
        ------
            A figure shows 2D sparse grids (if verbose = True)
        '''
        #Get grid points and weights
        grid_p = Tasmanian.SparseGrid()
        grid_p.makeGlobalGrid(dim,output,depth,"level",rule)
        points_p = grid_p.getPoints()
        weights_p = grid_p.getQuadratureWeights()
        if verbose == True:
            #If verbose is true print the sparse grid
            for i in range(len(points_p)):
                plt.scatter(points_p[i,0], points_p[i,1])
                plt.title('Sparse Grid of '+ rule.title(), fontsize = 20)
                plt.xlabel(r"$ϵ$ Dimension 1", fontsize = 20)
                plt.ylabel(r"$ϵ$ Dimension 2", fontsize = 20)
            plt.show()
        return points_p, weights_p

class Exploration_Bias():
    """
    Base class for methods of calculating explroation bias at each bo iter
    """
    def __init__(self, ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var):
        """
        Parameters
        ----------
        ep0: float, the original exploration parameter of the function
        ep_curr: float, the current exploration parameter value
        ep_enum: Enum, whether Boyle, Jasrasaria, Constant, or Decay ep method will be used
        bo_iter: int, The value of the current BO iteration
        bo_iter: int, The maximum values of BO iterations
        e_inc: float, the increment for the Boyle's method for calculating exploration parameter: Default is 1.5
        ep_f: float, The final exploration parameter value: Default is 0.01
        improvement: Bool, Determines whether last objective was an improvement. Default False
        best_error: float, The lowest error value in the training data
        mean_of_var: float, The value of the average of all posterior variances
        """
        assert all((isinstance(param, (float, int)) or param is None) for param in [ep0, ep_curr, ep_inc, ep_f, best_error, mean_of_var]), "ep0, ep_curr, ep_inc, ep_f, best_error, and mean_of_var must be int, float, or None"
        assert isinstance(ep_enum, Enum) == True, "ep_enum must be an Enum instance of Class Ep_enum"
        assert isinstance(improvement, bool) == True or improvement is None, "improvement must be bool or None"
        assert all((isinstance(param, (int)) or param is None) for param in [bo_iter, bo_iter_max]), "bo_iter and bo_iter_max must be int or None"
        # Constructor method
        self.ep0 = ep0
        self.ep_curr = ep_curr
        self.ep_enum = ep_enum
        self.bo_iter = bo_iter
        self.bo_iter_max = bo_iter_max
        self.ep_inc = ep_inc
        self.ep_f = ep_f
        self.improvement = improvement
        self.best_error = best_error
        self.mean_of_var = mean_of_var
        
    def set_ep(self):
        """
        Updates value of exploration parameter based on one of the four alpha heuristics
        
        Returns:
        --------
        ep: The current exploration parameter
        
        """        
        if self.ep_enum.value == 1: #Constant if using constant method
            assert self.ep0 is not None
            ep = self.__set_ep_constant()
            
        elif self.ep_enum.value == 2: #Decay
            assert self.ep0 is not None
            assert self.ep_f is not None 
            assert self.ep0*self.ep_f > 0 #Assert same sign
            assert self.ep0 !=0 #Assert that the starting value is not zero (can't decay from 0)
            assert self.bo_iter is not None
            assert self.bo_iter_max is not None
            assert self.bo_iter_max-1 >= self.bo_iter >= 0
            
            ep = self.__set_ep_decay()
            
        elif self.ep_enum.value == 3: #Boyle
            assert self.ep0 is not None
            assert self.ep_inc is not None
            assert self.improvement is not None
            ep = self.__set_ep_boyle()
            
        else: #Jasrasaria
            assert self.best_error is not None
            assert self.mean_of_var is not None
            ep = self.__set_ep_jasrasaria()
        
        #Set current ep to new ep
        self.ep_curr = ep
            
    def __set_ep_constant(self):
        """
        Creates a value for the exploration parameter based off of a constant value

        Returns
        --------
            ep: The exploration parameter for the iteration
        """
        ep = self.ep0
        
        return ep
    
    def __set_ep_decay(self):
        """
        Creates a value for the exploration parameter based off of a decay heuristic
        
        Returns
        --------
            ep: The exploration parameter for the iteration
        """
        decay_steps = int(self.bo_iter_max/2)
        if self.bo_iter < decay_steps:
#             ep = self.ep0*((self.ep_f/self.ep0)**(self.bo_iter/decay_steps))
            ep = self.ep0 + (self.ep_f - self.ep0)*(self.bo_iter/self.bo_iter_max)
        else: 
            ep = self.ep_f
            
        return ep
    
    def __set_ep_boyle(self):
        """
        Creates a value for the exploration parameter
            
        Returns
        --------
            ep: The exploration parameter for the iteration
        """
        if self.ep_curr is None:
            self.ep_curr = self.ep0
            
        if self.improvement == True:
            ep = self.ep_curr*self.ep_inc
        else:
            ep = self.ep_curr/self.ep_inc

        return ep
    
    def __set_ep_jasrasaria(self):
        """
        Creates a value for the exploration parameter based off of Jasrasaria's heuristic

        Returns
        --------
            ep: The exploration parameter for the iteration
        """
        
        ep = self.mean_of_var/self.best_error
        
        return ep

    
class GPBO_Driver:
    """
    The base class for running the GPBO Workflow
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, gp_emulator, ep_bias):
        """
        Parameters
        ----------
        cs_params: Instance of CaseStudyParameters class, class containing the values associated with CaseStudyParameters
        method: Instance of GPBO_Methods class, Class containing GPBO method information
        simulator: Instance of Simulator class, class containing values of simulation parameters
        exp_data: Instance of Data class, Class containing at least theta, x, and y experimental data
        sim_data: Instance of Data class, class containing at least theta, x, and y simulation data
        sim_sse_data: Instance of Data class, class containing at least theta, x, and sse simulation data
        val_data: Instance of Data class, class containing at least theta and x simulation data
        val_sse_data: Instance of Data class, class containing at least theta and x sse simulation data
        
        """
        # Constructor method
        self.cs_params = cs_params
        self.method = method
        self.simulator = simulator
        self.exp_data = exp_data
        self.sim_data = sim_data
        self.sim_sse_data = sim_sse_data
        self.val_data = val_data
        self.val_sse_data = val_sse_data
        self.gp_emulator = gp_emulator
        self.ep_bias = ep_bias
               
    
    def gen_emulator(self, kernel, lenscl, outputscl, retrain_GP):
        """
        Sets GP Emulator class (equipped with training data) and validation data based on the method class instance
        
        Returns:
        --------
            gp_emulator: Instance of the GP_Emulator class. Class for the GP emulator
        """
        #Determine Emulator Status, set gp_data data, and ininitalize correct GP_Emulator child class
        if self.method.emulator == False:
            all_gp_data = self.sim_sse_data
            all_val_data = self.val_sse_data
            gp_emulator = Type_1_GP_Emulator(all_gp_data, all_val_data, None, None, None, kernel, lenscl, self.simulator.noise_std, outputscl, retrain_GP, self.cs_params.seed, None, None, None, None)
        else:
            all_gp_data = self.sim_data
            all_val_data = self.val_data
            gp_emulator = Type_2_GP_Emulator(all_gp_data, all_val_data, None, None, None, kernel, lenscl, self.simulator.noise_std, outputscl, retrain_GP, self.cs_params.seed, None, None, None, None)
            
        return gp_emulator
    
    
    def opt_with_scipy(self, neg_ei, reoptimize):
        """
        Optimizes a function with scipy.optimize
        
        Parameters
        ----------
        neg_ei: bool, whether to calculate neg_ei (True) or sse (False)
        reoptimize: int, how many times to reinitialize optimization with other starting points
        
        Returns:
        --------
        best_val: float, The optimized value of the function
        best_theta: ndarray, The theta set corresponding to val_best
        """
        
        assert isinstance(reoptimize, int) and reoptimize > 0, "reoptimize must be int > 0"

        #Find unique theta vals
        unique_thetas = self.gp_emulator.gp_val_data.get_unique_theta()
        assert reoptimize <= len(unique_thetas), "Can not reoptimize more times than there are starting points"
        
        #Set seed
        if self.cs_params.seed is not None:
            np.random.seed(self.cs_params.seed)
            
        #Initialize val_best and best_theta
        best_vals = np.full(reoptimize, np.inf)
        best_thetas = np.zeros((reoptimize, self.gp_emulator.gp_val_data.get_dim_theta()))
        
        #Calc best error
        if self.method.emulator == False:
            best_error = self.gp_emulator.calc_best_error()
        else:
            best_error = self.gp_emulator.calc_best_error(self.exp_data)
            
        #Find bounds and arguments for function
        bnds = self.simulator.bounds_theta_reg.T #Transpose bounds to work with scipy.optimize
        
        #Choose values of theta from validation set at random
        theta_val_idc = list(range(len(unique_thetas)))
    
        ## Loop over each validation point/ a certain number of validation point thetas
        for i in range(reoptimize):
            #Choose a random index of theta to start with
            unique_theta_index = random.sample(theta_val_idc, 1)
            theta_guess = unique_thetas[unique_theta_index]
            
            #Call scipy method to optimize EI given theta
            best_result = optimize.minimize(self.__scipy_fxn, theta_guess, bounds=bnds, method = "L-BFGS-B", args=(neg_ei, best_error))
            #Add ei and best_thetas to lists as appropriate
            best_vals[i] = best_result.fun
            best_thetas[i] = best_result.x
        
        #Choose a single value with the lowest -ei or sse
        min_value = min(best_vals) #Find lowest value
        min_indices = np.where( np.isclose(best_vals, min_value, rtol=1e-7) )[0] #Find all indecies where there may be a tie
        rand_min_idx = np.random.choice(min_indices) #Choose one at random as the next step
        
        best_val = best_vals[rand_min_idx]
        best_theta = best_thetas[rand_min_idx]
        
        if neg_ei == True:
            best_val = best_val*-1
                    
        return best_val, best_theta
        
    def __scipy_fxn(self, theta, neg_ei, best_error):
        """
        Calculates either -ei or sse objective at a candidate theta value
        
        Parameters
        -----------
        theta: ndarray, the array of theta values to optimize
        neg_ei: bool, whether to calculate neg_ei (True) or sse (False)
        best_error: float, the best error of the method so far
        
        Returns:
        --------
        obj: float, Either neg_ei or sse for candidate theta
        
        """
        #Note, theta must be in array form ([ [1,2] ])
        #copy theta into candidate point in GP Emulator (to be added)
        candidate = Data(None, self.exp_data.x_vals, None, None, None, None, None, None, self.simulator.bounds_theta_reg, self.simulator.bounds_x)
        
        #Create feature data for candidate point
        if self.method.emulator == False:
            candidate_theta_vals = theta.reshape(1,-1)
        else:
            candidate_theta_vals = np.repeat(theta.reshape(1,-1), self.exp_data.get_num_x_vals() , axis =0)
        
        candidate.theta_vals = candidate_theta_vals  
        self.gp_emulator.cand_data = candidate
        
        #Set candidate point feature data
        if self.method.emulator == False:
            self.gp_emulator.feature_cand_data = self.gp_emulator._Type_1_GP_Emulator__featurize_data(self.gp_emulator.cand_data)
        else:
            self.gp_emulator.feature_cand_data = self.gp_emulator._Type_2_GP_Emulator__featurize_data(self.gp_emulator.cand_data)
        
        #Evaluate GP mean/ stdev at theta
        cand_mean, cand_var = self.gp_emulator.eval_gp_mean_var_cand()
        
        #Evaluate SSE & SSE stdev at theta
        if self.method.emulator == False:
            cand_sse_mean, cand_sse_var = self.gp_emulator.eval_gp_sse_var_cand()
        else:
            cand_sse_mean, cand_sse_var = self.gp_emulator.eval_gp_sse_var_cand(self.exp_data)
        
        #Calculate objective fxn
        if neg_ei == False:
            obj = cand_sse_mean
        else:
            if self.method.emulator == False:
                obj = -1*self.gp_emulator.eval_ei_cand(self.exp_data, self.ep_bias, best_error)
            else:
                obj = -1*self.gp_emulator.eval_ei_cand(self.exp_data, self.ep_bias, best_error, self.method)

        return obj

    def eval_GP_over_grid(self):
        """
        Evaluates GP for data values over a grid (GP Mean, GP var, EI, SSE)
        
        Parameters
        ----------
        
        Returns:
        --------
        Grid_Data: list of Data instances, contains all data from evaluating the model over a grid
        
        """
        #Make list of Data classes
        #Loop over number of theta combinations
            #Make instance of Data class and add following to it
            #Make meshgrid of theta combination in 2 parameters keeping the rest of the values at their true value
            #Evaluate model mean, stdev, sse, and sse_var
            #Evaluate ei
                
    def augment_train_data(self):
        """
        Augments training data given a new point

        Parameters
        ----------

        Returns:
        --------
        sim_data: ndarray. The training parameter set with the augmented theta values
        """
        #Augment theta_best to training data

    def run_bo_iter(self, gp_model):
        """
        Runs a single GPBO iteration
        
        Parameters
        ----------
        gp_emulator: Instance of GP_Emulator, class for GP
        
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
        #Train GP model
        gp_emulator.train_gp(gp_model)
               
        #Set trained gp_emulator as a self parameter
        self.gp_emulator = gp_emulator
        
        #Calcuate best error
        if self.method.emulator == False:
            best_error = self.gp_emulator.calc_best_error() #For Type 1
        else:
            best_error = self.gp_emulator.calc_best_error(exp_data) #For Type 2
            
        #Call optimize acquistion fxn
        max_ei, max_ei_theta = self.opt_with_scipy(True, reoptimize)
        
        #Call optimize objective function
        min_sse, min_sse_theta = self.opt_with_scipy(False, reoptimize)
        
        #Call eval_all_data (optional) #Instead just make the data you would need to make heat maps?
        
        #Set Data in new Bo_results class
        
        #Call augment_train_data
        
        
    def run_bo_to_term(self, terminate):
        """
        Runs multiple GPBO iterations
        
        Params:
        -------
        max_ei: the maximum ei value for the iteration
        
        Returns:
        --------
        terminate: bool, Whether to terminate BO iterations
        """
        while terminate == False:
            ei_prev = np.inf
            for i in range(self.cs_params.bo_iter_tot):
                #Set exploration bias
                self.ep_bias.set_ep()
                
                bo_results, terminate = self.run_bo_iter(gp_model)
                #Call stopping criteria
                if all(max_ei < self.cs_params.ei_tol and ei_prev < self.cs_params.ei_tol) == True:
                    terminate = True
                else:
                    ei_prev = max_ei
                    terminate = False
        
    def run_bo_restarts(self):
        """
        Runs multiple GPBO restarts
        
        Returns:
        --------
        ???
        """
        for i in range(cs_params.bo_run_tot):
            self.bo_restart()
        
    def run_bo_workflow(self, kernel, lenscl, outputscl, retrain_GP):
        """
        Runs multiple GPBO iterations
        
        Parameters
        ----------
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
        
        #Initialize gp_emualtor class
        gp_emulator = self.gen_emulator(kernel, lenscl, outputscl, retrain_GP)
        
        #Choose training data
        train_data, test_data = gp_emulator.set_train_test_data(cs_params)
        
        #Initilize gp model
        gp_model = gp_emulator.set_gp_model()
        
        ##Call bo_iter
        self.run_bo_to_term(False)
        
        return run_theta_best, run_theta_opt, run_min_sse, run_min_abs_sse, run_max_ei, run_gp_mean, run_gp_var, run_hps