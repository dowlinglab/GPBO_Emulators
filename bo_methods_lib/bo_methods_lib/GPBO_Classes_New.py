import numpy as np
import random
import warnings
import math
from scipy.stats import norm
from scipy import integrate
import scipy.optimize as optimize
import os
import time
import Tasmanian
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import pandas as pd
from enum import Enum
import pickle
import gzip
import itertools
from itertools import combinations
import copy
import scipy
import matplotlib.pyplot as plt

class Method_name_enum(Enum):
    """
    The base class for any GPBO Method names
    
    Notes: 
    -------
    1 = A1
    2 = B1
    3 = A2
    4 = B2
    5 = C2
    
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
    
    Notes: 
    -------
    1 = Matern 52
    2 = Matern 32
    3 = RBF
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
    
    Notes: 
    -------
    1 = LHS
    2 = Meshgrid
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 (LHS) to 2 (Meshgrid)")
        
    LHS = 1
    MESHGRID = 2
    
class Obj_enum(Enum):
    """
    The base class for any objective function
    
    Notes: 
    -------
    1 = SSE
    2 = ln(SSE)
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 (obj) to 2 (ln obj)")
        
    OBJ = 1
    LN_OBJ = 2
    
class CS_name_enum(Enum):
    """
    The base class for any GPBO case study names
    
    Notes: 
    -------
    1 = CS1 2 Param Polynomial
    2 = CS2 4 Param Muller Potential
    3 = CS2 8 Param Muller Potential
    4 = CS2 12 Param Muller Potential
    5 = CS2 16 Param Muller Potential
    6 = CS2 20 Param Muller Potential
    7 = CS2 24 Param Muller Potential
    8 = CS3 5 Param Polynomial
    9 = CS4 4 Param Isotherm
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 10) == False:
        raise ValueError("There are 9 options for Enum: 1 to 9")
        
    CS1 = 1
    CS2_4 = 2
    CS2_8 = 3
    CS2_12 = 4
    CS2_16 = 5
    CS2_20 = 6
    CS2_24 = 7
    CS3 = 8
    CS4 = 9
    
class Ep_enum(Enum):
    """
    The base class for any Method for calculating the decay of the exploration parameter
    
    Notes: 
    -------
    1 = Constant
    2 = Decay
    3 = Boyle
    4 = Jasrasaria
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
    get_emulator()
    get_obj()
    get_sparse_grid()
    """
    # Class variables and attributes
    
    def __init__(self, method_name):
        """
        Parameters
        ----------
        method_name, Method_name_enum Class instance, The name associated with the method being tested. Enum type
        """
        assert isinstance(method_name, Method_name_enum), "method_name must be an instance of Method_name_enum"
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
        obj_enum: class instance, Determines whether log scaling is used
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

class CaseStudyParameters:
    """
    The base class for any GPBO Case Study and Evaluation
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, seed, obj_tol, ei_tol):
        """
        Parameters
        ----------
        cs_name: string, The name associated with the case study being evaluated   
        ep0: float, The original  exploration bias. Default 1
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, Determines whether feature data will be normalized for problem analysis
        kernel: enum class instance, Determines which GP Kerenel to use
        lenscl: float, np.ndarray, or None, Value of the lengthscale hyperparameter - None if hyperparameters will be trained
        outputscl: float or None, Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_GP: int, number of times to restart GP training. Note, 0 = 1 optimization
        reoptimize_obj: int, number of times to reoptimize ei/sse with different starting values. Note, 0 = 1 optimization
        gen_heat_map_data: bool, determines whether validation data are generated to create heat maps
        noise_mean:float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise
        bo_iter_tot: int, maximum number of BO iterations per restart
        bo_run_tot: int, total number of BO algorithm restarts
        save_fig: bool, Determines whether figures will be saved. Default False
        save_data: bool, Determines whether data will be saved. Default True
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        seed: int or None, Determines seed for randomizations. None if seed is random
        ei_tol: float, ei at which to terminate algorithm
        obj_tol: float, obj at which to terminate algorithm after int(bo_iter_tot*0.3) iters
        
        """
        #Assert statements
        #Check for strings
        assert isinstance(cs_name, (str, Enum)) == True, "cs_name must be a string or Enum" #Will figure this one out later
        #Check for enum 
        assert isinstance(kernel, (Enum)) == True, "kernel must be type Enum" #Will figure this one out later
        #Check for float/int
        assert all(isinstance(var, (float,int)) for var in [sep_fact,ep0]) == True, "sep_fact and ep0 must be float or int"
        #Check for bool
        assert all(isinstance(var, (bool)) for var in [normalize, gen_heat_map_data, save_data]) == True, "normalize, gen_heat_map_data, save_fig, and save_data must be bool"
        #Check for int
        assert all(isinstance(var, (int)) for var in [bo_iter_tot, bo_run_tot, seed, retrain_GP, reoptimize_obj]) == True, "bo_iter_tot, bo_run_tot, seed, retrain_GP, and reoptimize_obj must be int"
        assert isinstance(outputscl, (float, int)) or outputscl is None, "outputscl must be float, int, or None"
        #Outputscl must be >0 if not None
        if outputscl is not None:
            assert outputscl > 0, "outputscl must be > 0 initially if it is not None"
            
        #Check lenscl, float, int, array, or None
        if isinstance(lenscl, list):
            lenscl = np.array(lenscl)
            
        assert isinstance(lenscl, (float, int, np.ndarray)) or lenscl is None, "lenscl must be float, int, np.ndarray, or None"
        if lenscl is not None:
            if isinstance(lenscl, (float, int)):
                assert lenscl > 0, "lenscl must be > 0 initially if lenscl is not None"
            else:
                assert all(isinstance(var, (np.int64, np.float64, float, int)) for var in lenscl), "All lenscl elements must float or int"
                assert all(item > 0 for item in lenscl), "lenscl elements must be > 0 initially if lenscl is not None"
        #Check for sep fact number between 0 and 1        
        assert 0 < sep_fact <= 1, "Separation factor must be between 0 and 1. Not including zero"
        #Check for > 0
        assert all(var > 0 for var in [bo_iter_tot, bo_run_tot, seed]) == True, "bo_iter_tot, bo_run_tot, and seed must be > 0"        
        #Check for >=0
        assert all(var >= 0 for var in [retrain_GP, reoptimize_obj]) == True, "retrain_GP and reoptimize_obj must be >= 0"
        #Check for str or None
        assert isinstance(DateTime, (str)) == True or DateTime == None, "DateTime must be str or None"
        assert isinstance(ei_tol, (float,int)) and ei_tol >= 0, "ei_tol must be a positive float or integer"
        assert isinstance(obj_tol, (float,int)) and obj_tol >= 0, "obj_tol must be a positive float or integer"
        
        # Constructor method
        #Ensure name is a string
        if isinstance(cs_name, Enum) == True:
            self.cs_name = cs_name.name
        else:
            self.cs_name = cs_name
        self.ep0 = ep0
        self.sep_fact = sep_fact
        self.normalize = normalize
        self.kernel = kernel
        self.lenscl = lenscl
        self.outputscl = outputscl
        self.retrain_GP = retrain_GP
        self.reoptimize_obj = reoptimize_obj
        self.gen_heat_map_data = gen_heat_map_data
        self.bo_iter_tot = bo_iter_tot
        self.bo_run_tot = bo_run_tot
        self.save_data = save_data
        self.DateTime = DateTime
        self.seed = seed
        #Set seed
        if  self.seed != None:
            assert isinstance(self.seed, int) == True, "Seed number must be an integer or None"
            random.seed(self.seed)
        self.ei_tol = ei_tol
        self.obj_tol = obj_tol

class Simulator:
    """
    The base class for differet simulators. Defines a simulation
    
    Methods
    --------------
    __init__
    __set_true_params()
    __grid_sampling(n_points, bounds
    __lhs_sampling(n_points, bounds, seed)
    __create_param_data(n_points, bounds, gen_meth, seed)
    gen_y_data(data, noise_mean, noise_std)
    gen_exp_data(num_x_data, gen_meth_x)
    gen_sim_data(num_theta_data, gen_meth_theta, num_x_data, gen_meth_x, gen_val_data)
    sim_data_to_sse_sim_data(method, sim_data, exp_data, gen_val_data)
    """
    def __init__(self, indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u, noise_mean, noise_std, normalize, seed, calc_y_fxn):
        """
        Parameters
        ----------
        indeces_to_consider: list of int, The indeces corresponding to which parameters are being guessed
        theta_ref: ndarray, The array containing the true values of problem constants
        theta_names: list, list of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        bounds_theta_l: list, lower bounds of theta
        bounds_x_l: list, lower bounds of x
        bounds_theta_u: list, upper bounds of theta
        bounds_x_u: list, upper bounds of x
        noise_mean:float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise
        normalize: bool, Determines whether feature data will be normalized for problem analysis
        seed: int or None, Determines seed for randomizations. None if seed is random
        calc_y_fxn: function, The function to calculate ysim data with
        """
        #Check for float/int
        assert all(isinstance(var,(float,int)) for var in [noise_std, noise_mean]) == True, "noise_mean and noise_std must be int or float"
        assert isinstance(seed, int) or seed is None, "Seed must be int or None"
        #Check for list or ndarray
        list_vars = [indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u]
        assert all(isinstance(var,(list,np.ndarray)) for var in list_vars) == True, "indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must be list or np.ndarray"
        #Check for list lengths > 0
        assert all(len(var) > 0 for var in list_vars) == True, "indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must have length > 0"
        #Check that bound_x and bounds_theta have same lengths
        assert len(bounds_theta_l) == len(bounds_theta_u) and len(bounds_x_l) == len(bounds_x_u), "bounds lists for x and theta must be same length"
        #Check indeces to consider in theta_ref
        assert all(0 <= idx <= len(theta_ref)-1 for idx in indeces_to_consider)==True, "indeces to consider must be in range of theta_ref"
        #How to write assert statements for cs_params and calc_y_fxn
        #Assert normalize is bool
        assert isinstance(normalize, bool), "normalize"
        
        # Constructor method
        self.dim_x = len(bounds_x_l)
        self.dim_theta = len(indeces_to_consider) #Length of theta is equivalent to the number of indeces to consider
        self.indeces_to_consider = indeces_to_consider
        self.theta_ref = theta_ref
        self.theta_names = theta_names
        self.theta_true, self.theta_true_names = self.__set_true_params() #Would this be better as a dictionary?
        self.bounds_theta = np.array([bounds_theta_l, bounds_theta_u])
        self.bounds_theta_reg = self.bounds_theta[:,self.indeces_to_consider] #This is the theta_bounds for parameters we will regress
        self.bounds_x = np.array([bounds_x_l, bounds_x_u])
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.calc_y_fxn = calc_y_fxn
        self.normalize = normalize
        self.seed = seed
        #Find theta_ref normalized if applicable
        if self.normalize == True:
            self.theta_true_norm = (self.theta_true - self.bounds_theta_reg[0]) / (self.bounds_theta_reg[1] - self.bounds_theta_reg[0])
    
    def __set_true_params(self):
        """
        Sets true parameter value array and the corresponding names based on parameter dictionary and indeces to consider
        
        Returns
        -------
        true_params: ndarray, The true parameter of the model
        true_param_names: list of string, The names of the true parameter of the model
        """
        #Define theta_true and theta_true_names from theta_ref, theta_names, and indeces to consider
        true_params = self.theta_ref[self.indeces_to_consider]
        true_param_names = [self.theta_names[idx] for idx in self.indeces_to_consider]
        
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
        lhs_data: ndarray, array of LHS sampling points with length (num_points) 
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
        gen_meth: class (Gen_meth_enum), ("LHS", "Meshgrid"). Determines whether data will be generated with an LHS or meshgrid
        seed: int, seed of random generation
        
        Returns:
        --------
        data: ndarray, an array of data
        
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
    
    def __vector_to_1D_array(self, array):
        """
        Turns arrays that are shape (n,) into (n, 1) arrays

        Parameters
        ----------
        array: ndarray, n dimensions

        Returns
        -------
        array: ndarray,  if n > 1, return original array. Otherwise, return 2D array with shape (-1,n)
        """
        #If array is not 2D, give it shape (len(array), 1)
        if not len(array.shape) > 1:
            array = array.reshape(-1,1)
        return array
    
    
    #Need to account for normalization here
    def gen_y_data(self, data, noise_mean, noise_std):
        """
        Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
        Parameters
        ----------
        data, Instance of Data: Data to generate y data for
        noise_mean:float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise

        Returns
        -------
        y_data: ndarray, The simulated y training data
        """        
        #Set seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        #Unnormalize feature data for calculation if necessary
        if self.normalize == True:
            data = data.unnorm_feature_data()
                
        #Define an array to store y values in
        y_data = []
        #Get number of points
        len_points = data.get_num_theta()
        #Loop over all theta values
        for i in range(len_points):
            #Create model coefficient from true space substituting in the values of param_space at the correct indeces
            model_coefficients = self.theta_ref.copy()
            #Replace coefficients a specified indeces with their theta_val counterparts
            model_coefficients[self.indeces_to_consider] = data.theta_vals[i]               
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
        x_vals = self.__vector_to_1D_array(self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, self.seed))
        #Reshape theta_true to correct dimensions and stack it once for each xexp value
        theta_true = self.theta_true.reshape(1,-1)
        theta_true_repeated = np.vstack([theta_true]*len(x_vals))
        #Create exp_data class and add values
        exp_data = Data(theta_true_repeated, x_vals, None, None, None, None, None, None, self.bounds_theta_reg, self.bounds_x, None, self.seed)
        #Normalize feature data if noramlize is true
        if self.normalize == True:
            exp_data = exp_data.norm_feature_data()
        #Generate y data for exp_data calss instance
        exp_data.y_vals = self.gen_y_data(exp_data, self.noise_mean, self.noise_std)
        
        return exp_data
    
    def gen_sim_data(self, num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, gen_val_data = False):
        """
        Generates experimental data in an instance of the Data class
        
        Parameters
        ----------
        num_theta_data: int, number of theta values
        num_x_data: int, number of experiments
        gen_meth_theta: bool: Whether to generate theta data with LHS or grid method
        gen_meth_x: bool: Whether to generate X data with LHS or grid method
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        gen_val_data: bool, Whether validation data (no y vals) or simulation data (has y vals) will be generated. 
        
        Returns:
        --------
        sim_data: instance of a class filled in with experimental x and y data along with parameter bounds
        """
        assert isinstance(sep_fact, (float,int)), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        
        if isinstance(gen_val_data, bool) == False:
            raise ValueError('gen_val_data must be bool')
            
        #Chck that num_data > 0
        if num_theta_data <= 0 or isinstance(num_theta_data, int) == False:
            raise ValueError('num_theta_data must be a positive integer')
            
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError('num_x_data must be a positive integer')
        
        #Set bounds on theta which we are regressing given bounds_theta and indeces to consider
        #X data we always want the same between simulation and validation data
        x_data = self.__vector_to_1D_array(self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, self.seed))
            
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
            warnings.warn("More than 5000 points will be generated!")
     
        #Generate all rows of simulation data
        sim_data = Data(None, None, None, None, None, None, None, None, self.bounds_theta_reg, self.bounds_x, sep_fact, self.seed)
       
        #For validation theta, change the seed by 1 to ensure validation and sim data are never the same
        if gen_val_data == False:
            seed = self.seed
        else:
            seed = int(self.seed + 1)
            
        #Generate simulation data theta_vals and create instance of data class   
        sim_theta_vals = self.__vector_to_1D_array(self.__create_param_data(num_theta_data, self.bounds_theta_reg, gen_meth_theta, seed))
        
        #Add repeated theta_vals and x_data to sim_data
        sim_data.theta_vals = np.repeat(sim_theta_vals, repeat_theta , axis = 0)
        sim_data.x_vals = np.vstack([x_data]*repeat_x)
        
        #Normalize feature data if noramlize is true
        if self.normalize == True:
            sim_data = sim_data.norm_feature_data()
        
        #Add y_vals for sim_data only
        if gen_val_data == False:
            sim_data.y_vals = self.gen_y_data(sim_data, 0, 0)
        
        return sim_data
   
    def sim_data_to_sse_sim_data(self, method, sim_data, exp_data, sep_fact, gen_val_data = False):
        """
        Creates simulation data based on x, theta_vals, the GPBO method, and the case study

        Parameters
        ----------
        method: class, fully defined methods class which determines which method will be used
        sim_data: Class, Class containing at least the theta_vals, x_vals, and y_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        gen_val_data: bool, Whether validation data (no y vals) or simulation data (has y vals) will be generated.

        Returns:
        --------
        sim_sse_data: ndarray, sse data generated from y_vals
        """
        
        assert isinstance(sep_fact, (float,int)), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
            
        if isinstance(gen_val_data, bool) == False:
            raise ValueError('gen_val_data must be bool')
            
        #Find length of theta and x in data arrays
        len_theta = sim_data.get_num_theta()
        len_x = exp_data.get_num_x_vals()
      
        #Q: For this dataset does it make more sense to have all theta and x values or just the unique thetas and x values?
        #A: Just the unique ones. No need to store extra data if we won't use it and it will be saved somewhere else regardless
        #Assign unique theta indeces and create an array of them
        unique_indexes = np.unique(sim_data.theta_vals, axis = 0, return_index=True)[1]
        unique_theta_vals = np.array([sim_data.theta_vals[index] for index in sorted(unique_indexes)])
        #Add the unique theta_vals and exp_data x values to the new data class instance
        sim_sse_data = Data(unique_theta_vals, exp_data.x_vals, None, None, None, None, None, None, self.bounds_theta, self.bounds_x, sep_fact, self.seed)
        
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

            #objective function only log if using 1B or 2B
            if method.obj.value == 2:
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
    __get_unique(all_vals)
    get_unique_theta()
    get_unique_x()
    get_num_theta()
    get_dim_theta()
    get_num_x_vals()
    get_dim_x_vals()
    __normalize(data, bounds)
    __unnormalize(data, bounds)
    norm_feature_data() 
    unnorm_feature_data()
    train_test_idx_split() 
    """
    # Class variables and attributes
    
    def __init__(self, theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, sse_var, ei, bounds_theta, bounds_x, sep_fact, seed):
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
        bounds_theta: ndarray, bounds of theta
        bounds_x: ndarray, bounds of x
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        seed: int or None, Determines seed for randomizations. None if seed is random
        """
        list_vars = [theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, ei]
        assert all(isinstance(var, np.ndarray) or var is None for var in list_vars), "theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, and ei must be np.ndarray, or None"
        assert isinstance(seed, int) or seed is None, "Seed must be int or None"
        assert isinstance(sep_fact, (float,int)) or sep_fact is None, "Separation factor must be float or int > 0 or None (exp_data)"
        if sep_fact is not None:
            assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        # Constructor method
        self.theta_vals = theta_vals
        self.x_vals = x_vals
        self.y_vals = y_vals  
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.sse = sse
        self.sse_var = sse_var
        self.ei = ei
        self.bounds_theta = bounds_theta
        self.bounds_x = bounds_x
        self.sep_fact = sep_fact
        self.seed = seed
    
    def __get_unique(self, all_vals):
        """
        Gets unique instances of a certain type of data
        
        Parameters:
        -----------
        all_vals: ndarray, array of parameters with duplicates
        
        Returns:
        --------
        unique_vals: ndarray, array of parameters without duplicates
        """
        #Get unique indecies and use them to get the values
        unique_indexes = np.unique(all_vals, axis = 0, return_index=True)[1]
        unique_vals = np.array([all_vals[index] for index in sorted(unique_indexes)])
        
        return unique_vals
    
    def get_unique_theta(self):
        """
        Defines the unique theta data in an array of theta_vals
        
        Returns:
        --------
        unique_theta_vals: ndarray, array of unique theta vals 
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        #Get unique indecies and use them to get the values
        unique_theta_vals = self.__get_unique(self.theta_vals)
        return unique_theta_vals
    
    def get_unique_x(self):
        """
        Defines the unique x data in an array of x_vals
        
        Returns:
        --------
        unique_x_vals: ndarray, array of unique x vals 
        """
        assert self.x_vals is not None, "x_vals must be defined"
        #Get unique indecies and use them to get the values
        unique_x_vals = self.__get_unique(self.x_vals)
        return unique_x_vals
    
    def get_num_theta(self):
        """
        Defines the total number of theta data the GP will have access to to train on
        
        Returns
        -------
        num_theta_data: int, the number of theta data the GP will have access to
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        num_theta_data = len(self.theta_vals)
        
        return num_theta_data
    
    def get_dim_theta(self):
        """
        Defines the total dimensions of theta data the GP will have access to to train on
        
        Returns
        -------
        dim_theta_data: int, the dim of theta data the GP will have access to
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
        
        Returns
        -------
        num_x_data: int, the number of x data the GP will have access to
        """
        assert self.x_vals is not None, "x_vals must be defined"
        #Length is the number of data
        num_x_data = len(self.x_vals)
        
        return num_x_data
    
    def get_dim_x_vals(self):
        """
        Defines the total dimensions of x data the GP will have access to to train on
        
        Returns
        -------
        dim_x_data: int, the dimensiosn of x data the GP will have access to
        """
        assert self.x_vals is not None, "x_vals must be defined"
        #Get dim of x data
        dim_x_data = self.__vector_to_1D_array(self.x_vals).shape[1]
        
        return dim_x_data
    
    def __vector_to_1D_array(self, array):
        """
        Turns arrays that are shape (n,) into (n, 1) arrays

        Parameters
        ----------
        array: ndarray, n dimensions

        Returns
        -------
        array: ndarray,  if n > 1, return original array. Otherwise, return 2D array with shape (-1,n)
        """
        #If array is not 2D, give it shape (len(array), 1)
        if not len(array.shape) > 1:
            array = array.reshape(-1,1)
        return array
    
    def __normalize(self, data, bounds):
        """
        Normalizes data between 0 and 1

        Parameters
        ----------
        data: ndarray: The data you want to scale
        bounds: ndarray, The bounds of the type of data you want to normalize
        
        Returns:
        ---------
        scaled_data: ndarray, the data normalized between 0 and 1 based on the bounds
        """
        #Define lower/upper bounds
        bounds = self.__vector_to_1D_array(bounds)
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
        scaled_data: Instance of Data, data with features scaled between 0 and 1
        bounds: ndarray, The bounds of the type of data you want to normalize
        
        Returns
        ---------
        data: ndarray, the original data renormalized based on the original bounds
        """
        #Define upper/lower bounds
        bounds = self.__vector_to_1D_array(bounds)
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        #scale data
        data = scaled_data*(upper_bound - lower_bound) + lower_bound
        
        return data
    
    def norm_feature_data(self):
        """
        Normalizes all feature data. Only call this method on unscaled data
        
        Returns
        -------
        scaled_data: Instance of Data class, data with features scaled between 0 and 1
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        assert self.x_vals is not None, "x_vals must be defined"
        assert self.bounds_theta is not None, "bounds_theta must be defined"
        assert self.bounds_x is not None, "bounds_x must be defined"
        
        #Scale theta and x values
        scaled_theta_vals = self.__normalize(self.theta_vals, self.bounds_theta)
        scaled_x_vals = self.__normalize(self.x_vals, self.bounds_x)
        
        #Create an instance of the data class with scaled values
        scaled_data = Data(scaled_theta_vals, scaled_x_vals, self.y_vals, self.gp_mean, self.gp_var, self.sse, self.sse_var, self.ei, self.bounds_theta, self.bounds_x, self.sep_fact, self.seed) 
        
        return scaled_data
    
    def unnorm_feature_data(self):
        """
        Unnormalizes all feature data and stores it in a new instance of the data class. Only call this method on scaled data
        
        Returns
        -------
        unscaled_data: Instance of Data class, data with features scaled between the original bounds
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        assert self.x_vals is not None, "x_vals must be defined"
        assert self.bounds_theta is not None, "bounds_theta must be defined"
        assert self.bounds_x is not None, "bounds_x must be defined"
        
        #Unscale theta and x values
        reg_theta_vals = self.__unnormalize(self.theta_vals, self.bounds_theta)
        reg_x_vals = self.__unnormalize(self.x_vals, self.bounds_x)
        
        #Create instance of data class for new unscaled values
        unscaled_data = Data(reg_theta_vals, reg_x_vals, self.y_vals, self.gp_mean, self.gp_var, self.sse, self.sse_var, self.ei, self.bounds_theta, self.bounds_x, self.sep_fact, self.seed) 
        
        return unscaled_data
    
    def train_test_idx_split(self):
        """
        Splits data indeces into training and testing indeces
        
        Returns:
        --------
        train_idx: ndarray, The training theta data identifiers
        test_idx: ndarray, The testing theta data identifiers
        
        Notes
        -----
        The training and testing data is split such that the number train_data is always rounded up. Ensures there is always training data

        """
        assert self.sep_fact is not None, "Data must have a separation factor that is not None!"
        assert self.theta_vals is not None, "data must have theta_vals"
        #Find number of unique thetas and calculate length of training data
        len_theta = len(self.get_unique_theta())
        len_train_idc = int(np.ceil(len_theta*self.sep_fact)) #Ensure there will always be at least one training point by using np.ceil
        
        #Create an index for each theta
        all_idx = np.arange(0,len_theta)

        #Shuffles Random Data. Will calling this once in case study parameters mean I don't need this? (No. It doesn't)
        if self.seed is not None:
            #Set seed to number specified by shuffle seed
            random.seed(self.seed)
            
        #Shuffle all_idx data in such a way that theta values will be randomized
        random.shuffle(all_idx)
        #Set train test indeces
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
    get_num_gp_data()
    __set_kernel()
    __set_lenscl(kernel)
    __set_outputscl(kernel)
    set_model()
    train_gp(gp_model)
    __eval_gp_mean_var(data)
    eval_gp_mean_var_heat_map(heat_map_data, featurized_hm_data)
    eval_gp_mean_var_test()
    eval_gp_mean_var_val()
    eval_gp_mean_var_cand()
    """
    # Class variables and attributes
    
    def __init__(self, gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, __feature_train_data, __feature_test_data, __feature_val_data, __feature_cand_data):
        """
        Parameters
        ----------
        gp_sim_data: instance of Data class, all simulation data for the GP
        gp_val_data: instance of Data class, the validation data for the GP. None if not saving parameter data
        cand_data: instance of Data class, candidate theta value for evaluation with GPBO_Driver.opt-with_scipy()
        kernel: enum class instance, Determines which GP Kerenel to use
        lenscl: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int: The standard deviation of the noise - None if hyperparameters will be updated during training
        outputscl: float or None, Determines value of outputscale
        retrain_GP: int, number of times to restart GP training
        seed: int or None, random seed
        __feature_train_data: ndarray, the feature data for the training data in ndarray form
        __feature_test_data: ndarray, the feature data for the testing data in ndarray form
        __feature_val_data: ndarray, the feature data for the validation data in ndarray form
        __feature_cand_data: ndarray, the feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()
        """
        #Assert statements
        #Check for int/float
        #Outputscl must be >0 if not None
        assert isinstance(outputscl, (float, int)) or outputscl is None, "outputscl must be float, int, or None"
        if outputscl is not None:
            assert outputscl > 0, "outputscl must be > 0 initially if it is not None"
            
        #Check lenscl, float, int, array, or None
        if isinstance(lenscl, list):
            lenscl = np.array(lenscl)
        assert isinstance(lenscl, (float, int, np.ndarray)) or lenscl is None, "lenscl must be float, int, np.ndarray, or None"
        if lenscl is not None:
            if isinstance(lenscl, (float, int)):
                assert lenscl > 0, "lenscl must be > 0 initially if lenscl is not None"
            else:
                assert all(isinstance(var, (np.int64, np.float64, float, int)) for var in lenscl), "All lenscl elements must float or int"
                assert all(item > 0 for item in lenscl), "lenscl elements must be > 0 initially if lenscl is not None"
        
        #Check for int
        assert isinstance(retrain_GP, int) == True, "retrain_GP must be int"
        #Check for > 0
        assert all(var >= 0 for var in [retrain_GP]) == True, "retrain_GP must be greater than or equal to 0"
        #Check for Enum
        assert isinstance(kernel, Enum) == True, "kernel must be type Enum"
        #Check for instance of Data class or None
        assert isinstance(gp_sim_data, (Data)) == True or gp_sim_data == None, "gp_sim_data must be an instance of the Data class or None"
        assert isinstance(gp_val_data, (Data)) == True or gp_val_data == None, "gp_sim_data must be an instance of the Data class or None"
        
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
        self.scaler = StandardScaler()
        self.__feature_train_data = None #Added using child class
        self.__feature_test_data = None #Added using child class
        self.__feature_val_data = None #Added using child class
        self.__feature_cand_data = None #Added using child class
        
    def get_num_gp_data(self):
        """
        Defines the total number of data the GP will have access to to train on
        
        Returns
        -------
        num_data: int, the number of data the GP will have access to
        """
        assert isinstance(self.gp_sim_data, Data), "self.gp_sim_data must be instance of Data class"
        #Number of available gp data determined by number of sim data
        num_gp_data = int(self.gp_sim_data.get_num_theta())
        
        return num_gp_data
    
    def __set_kernel(self):
        """
        Sets kernel of the model
        
        Returns
        ----------
        kernel: The original kernel of the model
        
        """ 
        #Set noise kernel
        noise_kern = WhiteKernel(noise_level=self.noise_std**2, noise_level_bounds= "fixed") #bounds = "fixed"
        #Set Constant Kernel
        cont_kern = ConstantKernel(constant_value = 1, constant_value_bounds = (1e-3,1e4))
        #Set the rest of the kernel
        if self.kernel.value == 3: #RBF
            kernel = cont_kern*RBF(length_scale_bounds=(1e-03, 1e3)) + noise_kern
        elif self.kernel.value == 2: #Matern 3/2
            kernel = cont_kern*Matern(length_scale_bounds=(1e-03, 1e3), nu=1.5) + noise_kern 
        else: #Matern 5/2
            kernel = cont_kern*Matern(length_scale_bounds=(1e-03, 1e3), nu=2.5) + noise_kern 
            
        return kernel
    
    def __set_lenscl(self, kernel):
        """
        Set the lengthscale of the model. Need to have training data before 
        
        Parameters
        ----------
        kernel: The kernel of the model defined by __set_kernel
        
        Returns
        -------
        kernel: The kernel of the model defined by __set_kernel with the lengthscale bounds set
        """
        if isinstance(self.lenscl, np.ndarray):
            assert len(self.lenscl) >= self.get_dim_gp_data(), "Length of self.lenscl must be at least self.get_gim_gp_data()!"
            #Cut the lengthscale to correct length if too long, by cutting the ends
            if len(self.lenscl) > self.get_dim_gp_data():
                self.lenscl =  self.lenscl[:self.get_dim_gp_data()]
        
            #Anisotropic but different
            lengthscale_val = self.lenscl
            kernel.k1.k2.length_scale_bounds = "fixed"
            
        #If setting lengthscale, ensure lengthscale values are fixed and that there is 1 lengthscale/dim,\
        elif isinstance(self.lenscl, (float, int)):            
            #Anisotropic but the same
            lengthscale_val = np.ones(self.get_dim_gp_data())*self.lenscl
            kernel.k1.k2.length_scale_bounds = "fixed"
            
        #Otherwise initialize them at 1 (lenscl is trained) 
        else:
            #Anisotropic but initialized to 1
            lengthscale_val = np.ones(self.get_dim_gp_data())

        #Set initial model lengthscale
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
            kernel.k1.k1.constant_value = 1.0
            
        return kernel
    
    def set_gp_model(self):
        """
        Generates the GP model for the process in sklearn
            
        Returns
        --------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor containing kernel, optimizer, etc.
        """
        
        #Don't optimize anything if lengthscale and outputscale are being fixed
        if isinstance(self.lenscl, np.ndarray) and all(var is not None for var in self.lenscl) and self.outputscl != None:
            optimizer = None
        elif isinstance(self.lenscl, (float, int)) and self.lenscl != None and self.outputscl != None:
            optimizer = None
        else:
            optimizer = "fmin_l_bfgs_b"
        
        #Set kernel
        kernel = self.__set_kernel()
        kernel = self.__set_lenscl(kernel)
        kernel = self.__set_outputscl(kernel)

        #Define model
        gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=self.retrain_GP, 
                                            random_state = self.seed, optimizer = optimizer, normalize_y = True)
        
        return gp_model
        
    def train_gp(self, gp_model):
        """
        Trains the GP given training data. Sets self.trained_hyperparams and self.fit_gp_model
        
        Parameters
        ----------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor, The untrained, fully defined gp model
            
        """  
        assert isinstance(gp_model, GaussianProcessRegressor), "gp_model must be GaussianProcessRegressor"
        assert isinstance(self.feature_train_data, np.ndarray), "self.feature_train_data must be np.ndarray"
        assert self.feature_train_data is not None, "Must have training data. Run set_train_test_data() to generate"
        #Train GP
        #Preprocess Training data
        #Update scaler to be the fitted scaler. This scaler will change as the training data is updated
        self.scaler = self.scaler.fit(self.feature_train_data)
        feature_train_data_scaled = self.scaler.transform(self.feature_train_data)
        fit_gp_model = gp_model.fit(feature_train_data_scaled, self.train_data.y_vals)
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
        gp_mean: ndarray, array of gp_mean for the data set
        gp_var: ndarray, array of gp variance for the data set
        
        """       
        #Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        gp_mean = np.zeros(len(data))
        gp_var = np.zeros(len(data))
        
        #Loop over all eval points
        for i in range(len(data)):
            eval_point = np.array([data[i]])
            #scale eval _point
            eval_point = self.scaler.transform(eval_point)
            #Evaluate GP given parameter set theta and state point value
            model_mean, model_std = self.fit_gp_model.predict(eval_point[0:1], return_std=True)            
            model_variance = model_std**2
            #Add values to list
            gp_mean[i] = model_mean
            gp_var[i] = model_variance
                  
        return gp_mean, gp_var
    
    def eval_gp_mean_var_misc(self, misc_data, featurized_misc_data):
        """
        Evaluate the GP mean and variance for a heat map set
        
        Parameters:
        -----------
        misc_data: instance of the Data class, data to evaluate gp mean and variance for containing at least theta_vals and x_vals
        featurized_misc_data: ndarray, featurized data to evaluate containing at least theta_vals and x_vals
        
        Returns:
        -------
        misc_gp_mean: ndarray, array of gp_mean for the test set
        misc_gp_var: ndarray, array of gp variance for the test set
        """
        
        assert isinstance(misc_data , Data), "misc_data must be type Data"
        assert isinstance(featurized_misc_data, np.ndarray), "featurized_misc_data must be np.ndarray"
        
        #Evaluate heat map data for GP
        misc_gp_mean, misc_gp_var = self.__eval_gp_mean_var(featurized_misc_data)
        
        #Set data parameters
        misc_data.gp_mean = misc_gp_mean
        misc_data.gp_var = misc_gp_var

        return misc_gp_mean, misc_gp_var
    
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
        val_gp_mean: ndarray, array of gp_mean for the validation set
        val_gp_var: ndarray, array of gp variance for the validation set
        """
        
        assert self.feature_val_data is not None, "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(self.feature_val_data, np.ndarray), "self.feature_val_data must by np.ndarray"
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
        cand_gp_mean: ndarray, array of gp_mean for the candidate theta set
        cand_gp_var: ndarray, array of gp variance for the candidate theta set
        """
        
        assert self.feature_cand_data is not None, "Must have validation data. Run set_train_test_data() to generate"
        #Evaluate test data for GP
        cand_gp_mean, cand_gp_var = self.__eval_gp_mean_var(self.feature_cand_data)
        
        #Set data parameters
        self.cand_data.gp_mean = cand_gp_mean
        self.cand_data.gp_var = cand_gp_var
        return cand_gp_mean, cand_gp_var
    
class Type_1_GP_Emulator(GP_Emulator):
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    get_dim_gp_data()
    featurize_data(data)
    set_train_test_data(sep_fact, seed)
    __eval_gp_sse_var(data)
    eval_gp_sse_var_heat_map(heat_map_data)
    eval_gp_sse_var_test/val/cand()
    calc_best_error()
    __eval_gp_ei(sim_data, exp_data, ep_bias, best_error)
    eval_ei_heat_map(heat_map_data, exp_data, ep_bias, best_error)
    eval_ei_test(exp_data, ep_bias, best_error)
    eval_ei_val(exp_data, ep_bias, best_error)
    eval_ei_cand(exp_data, ep_bias, best_error)
    """
    # Class variables and attributes
    
    def __init__(self, gp_sim_data, gp_val_data, cand_data, train_data, test_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, feature_train_data, feature_test_data, feature_val_data, feature_cand_data):
        """
        Parameters
        ----------
        gp_sim_data: instance of Data class, all simulation data for the GP
        gp_val_data: instance of Data class, the validation data for the GP. None if not saving heat map data
        cand_data: instance of Data class, candidate theta value for evaluation with GPBO_Driver.opt-with_scipy()
        train_data: instance of Data class, the training data for the GP
        testing_data: instance of Data class, the testing data for the GP
        kernel: enum class instance, Determines which GP Kerenel to use
        lenscl: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int: The standard deviation of the noise
        outputscl: float or None, Determines value of outputscale
        retrain_GP: int, number of times to restart GP training
        seed: int or None, random seed
        feature_train_data: ndarray, the feature data for the training data in ndarray form
        feature_test_data: ndarray, the feature data for the testing data in ndarray form
        feature_val_data: ndarray, the feature data for the validation data in ndarray form
        feature_cand_data: ndarray, the feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()
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
        dim_gp_data: int, the dimensions of the training data that the GP will have access to
        """
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have simulation theta to get a number of theta"
        
        #Just use number of theta dimensions for Type 1
        dim_gp_data = self.gp_sim_data.get_dim_theta()
        
        return dim_gp_data
    
    def featurize_data(self, data):
        """
        Collects the featues of the GP into ndarray form from an instance of the Data class
        
        Parameters:
        -----------
        data: instance of the Data class, data to evaluate GP for containing at least theta_vals and x_vals
        
        Returns:
        -------
        feature_eval_data: ndarray, The feature data for the GP
        
        """
        assert isinstance(data, Data), "data must be an instance of Data"
        assert np.all(data.theta_vals is not None), "Must have validation data theta_vals and x_vals to evaluate the GP"
        
        #Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = data.theta_vals
        
        return feature_eval_data
    
    def set_train_test_data(self, sep_fact, seed):
        """
        Finds the simulation data to use as training/testing data.
        
        Parameters
        ----------
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        seed: int or None, Determines seed for randomizations. None if seed is random
        
        Returns
        -------
        train_data: Instance of data class. Contains all theta, x, and y data for training data
        test_data: Instance of data class. Contains all theta, x, and y data for testing data
        
        Notes
        -----
        Sets feature_train_data, feature_test_data, and feature_val_data
        """
        assert isinstance(sep_fact, (float,int)), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        assert isinstance(seed, int), "seed must be int!"
        assert isinstance(self.gp_sim_data, Data), "self.gp_sim_data must be instance of Data"
        assert np.all(self.gp_sim_data.x_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.y_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.bounds_x is not None), "Must have simulation x bounds to create train/test data"
        assert np.all(self.gp_sim_data.bounds_theta is not None), "Must have simulation theta bounds to create train/test data"
        
        #Get train test idx
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split()
        
        #Get train data and set it as an instance of the data class
        theta_train = self.gp_sim_data.theta_vals[train_idx]
        x_train = self.gp_sim_data.x_vals #x_vals for Type 1 is the same as exp_data. No need to index x
        y_train = self.gp_sim_data.y_vals[train_idx]
        train_data = Data(theta_train, x_train, y_train, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, seed)
        self.train_data = train_data
        
        #Get test data and set it as an instance of the data class
        theta_test = self.gp_sim_data.theta_vals[test_idx]
        x_test = self.gp_sim_data.x_vals #x_vals for Type 1 is the same as exp_data. No need to index x
        y_test = self.gp_sim_data.y_vals[test_idx]
        test_data = Data(theta_test, x_test, y_test, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, seed)
        self.test_data = test_data
        
        #Set training and validation data features in GP_Emulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)
        
        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data
        
        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data
        
        
        return train_data, test_data
       
    def __eval_gp_sse_var(self, data):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the data
        
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over the data 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the data 
        
        """
        #For type 1, sse is the gp_mean
        sse_mean = data.gp_mean
        sse_var = data.gp_var
        
        #Set attributes
        data.sse = data.gp_mean
        data.sse_var = data.gp_var
                    
        return sse_mean, sse_var
    
    
    def eval_gp_sse_var_misc(self, misc_data):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for any data. Including heat map data
        
        Parameters
        -----------
        misc_data: Instance of Data, the data to evaluate the sse mean and variance for
        
        Returns
        --------
        misc_sse_mean: tensor, The sse derived from gp_mean evaluated over the data 
        misc_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the data 
        
        """
        assert isinstance(misc_data , Data), "misc_data must be type Data"
        assert np.all(misc_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(misc_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        
        #For type 1, sse is the gp_mean
        misc_sse_mean, misc_sse_var = self.__eval_gp_sse_var(misc_data)
                    
        return misc_sse_mean, misc_sse_var
    
    def eval_gp_sse_var_test(self):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the testing data
        
        Returns
        --------
        test_sse_mean: tensor, The sse derived from gp_mean evaluated over the test data 
        test_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the test data 
        
        """
        assert isinstance(self.test_data , Data), "self.test_data must be type Data"
        assert np.all(self.test_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(self.test_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        
        #For type 1, sse is the gp_mean
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data)
                    
        return test_sse_mean, test_sse_var
    
    def eval_gp_sse_var_val(self):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the validation data
        
        Returns
        --------
        val_sse_mean: tensor, The sse derived from gp_mean evaluated over the validation data 
        val_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the validation data 
        
        """
        assert isinstance(self.gp_val_data , Data), "self.gp_val_data must be type Data"
        assert np.all(self.gp_val_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.gp_val_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        
        #For type 1, sse is the gp_mean
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data)
                    
        return val_sse_mean, val_sse_var  
    
    def eval_gp_sse_var_cand(self):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the candidate theta data
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over the candidate theta data 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the candidate theta data 
        
        """
        assert isinstance(self.cand_data , Data), "self.cand_data must be type Data"
        assert np.all(self.cand_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.cand_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        
        #For type 1, sse is the gp_mean
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data)
                    
        return cand_sse_mean, cand_sse_var
    
    def calc_best_error(self):
        """
        Calculates the best error of the model
        
        Returns
        -------
        best_error: float, the best error of the method
        
        """   
        assert self.train_data is not None, "Must have self.train_data"
        assert isinstance(self.train_data , Data), "self.train_data must be type Data"
        assert np.all(self.train_data.y_vals is not None), "Must have simulation theta and y data to calculate best error"
        
        #Best error is the minimum sse value of the training data for Type 1
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
        ei, ei_terms_df = ei_class.type_1()
        #Add ei data to validation data class
        sim_data.ei = ei
        
        return ei, ei_terms_df
    
    def eval_ei_misc(self, misc_data, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        misc_data, Instance of Data class, data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        ei, ei_terms_df = self.__eval_gp_ei(misc_data, exp_data, ep_bias, best_error)
        return ei, ei_terms_df
    
    def eval_ei_test(self, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        """
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        ei, ei_terms_df = self.__eval_gp_ei(self.test_data, exp_data, ep_bias, best_error)
        return ei, ei_terms_df
    
    def eval_ei_val(self, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in gp_val_data
        """
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        ei, ei_terms_df = self.__eval_gp_ei(self.gp_val_data, exp_data, ep_bias, best_error)
        
        return ei, ei_terms_df
    
    def eval_ei_cand(self, exp_data, ep_bias, best_error):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in candidate theta data
        """
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        ei, ei_terms_df = self.__eval_gp_ei(self.cand_data, exp_data, ep_bias, best_error)
        
        return ei, ei_terms_df
    
    def add_next_theta_to_train_data(self, theta_best_sse_data):
        """
        Adds the theta with the highest ei to the training data set
        
        Parameters
        ----------
        theta_best_sse_data: Instance of Data, The class containing the data relavent to theta_best for a Type 1 GP
        """
        assert self.train_data is not None, "self.train_data must be Data"
        assert isinstance(self.train_data, Data), "self.train_data must be Data"
        assert isinstance(theta_best_sse_data, Data), "theta_best_sse_data must be Data"
        assert all(isinstance(var, np.ndarray) for var in [self.train_data.theta_vals,self.train_data.y_vals]), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"
        assert all(isinstance(var, np.ndarray) for var in [theta_best_sse_data.theta_vals, theta_best_sse_data.y_vals]), "theta_best_sse_data.theta_vals and self.theta_best_sse_data.y_vals must be np.ndarray"
        #Update training theta, x, and y separately
        self.train_data.theta_vals = np.vstack((self.train_data.theta_vals, theta_best_sse_data.theta_vals))
        self.train_data.y_vals = np.concatenate((self.train_data.y_vals, theta_best_sse_data.y_vals))
        feature_train_data = self.featurize_data(self.train_data)  
        
        #Reset training data feature array
        self.feature_train_data = feature_train_data
    
class Type_2_GP_Emulator(GP_Emulator):
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    get_dim_gp_data()
    featurize_data(data)
    set_train_test_data(sep_fact, seed)
    __eval_gp_sse_var(data, exp_data)
    eval_gp_sse_var_heat_map(heat_map_data, exp_data)
    eval_gp_sse_var_test/val/cand(exp_data)
    calc_best_error(method, exp_data)
    __eval_gp_ei(sim_data, exp_data, ep_bias, best_error, method)
    eval_ei_heat_map(heat_map_data, exp_data, ep_bias, best_error, method)
    eval_ei_test(exp_data, ep_bias, best_error, method)
    eval_ei_val(exp_data, ep_bias, best_error, method)
    eval_ei_cand(exp_data, ep_bias, best_error, method)
    """
    # Class variables and attributes
    def __init__(self, gp_sim_data, gp_val_data, cand_data, train_data, test_data, kernel, lenscl, noise_std, outputscl, retrain_GP, seed, feature_train_data, feature_test_data, feature_val_data, feature_cand_data):
        """
        Parameters
        ----------
        gp_sim_data: instance of Data class, all simulation data for the GP
        gp_val_data: instance of Data class, the validation data for the GP. None if not saving validation data
        cand_data: instance of Data class, candidate theta value for evaluation with GPBO_Driver.opt-with_scipy()
        train_data: instance of Data class, the training data for the GP
        testing_data: instance of Data class, the testing data for the GP
        kernel: enum class instance, Determines which GP Kerenel to use
        lenscl: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int: The standard deviation of the noise
        outputscl: float or None, Determines value of outputscale
        retrain_GP: int, number of times to restart GP training
        seed: int or None, random seed
        feature_train_data: ndarray, the feature data for the training data in ndarray form
        feature_test_data: ndarray, the feature data for the testing data in ndarray form
        feature_val_data: ndarray, the feature data for the validation data in ndarray form
        feature_cand_data: ndarray, the feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()
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
        dim_gp_data: int, the dimensions of the training data the GP will have access to
        """
        assert isinstance(self.gp_sim_data, Data), "self.gp_sim_data must be instance of Data"
        assert np.all(self.gp_sim_data.x_vals is not None), "Must have sim data theta_vals and x_vals"
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have sim data theta_vals and x_vals"
        
        #Number of theta dimensions + number of x dimensions
        dim_gp_data = int(self.gp_sim_data.get_dim_x_vals() + self.gp_sim_data.get_dim_theta())
        
        return dim_gp_data
    
    def featurize_data(self, data):
        """
        Collects the featues of the GP into ndarray form from an instance of the Data class
        
        Parameters:
        -----------
        data: instance of the Data class, data to evaluate GP for containing at least theta_vals and x_vals
        
        Returns:
        --------
        feature_eval_data: ndarray, The feature data for the GP
        
        """
        assert isinstance(data, Data), "data must be instance of Data"
        assert np.all(data.x_vals is not None), "Must have validation data theta_vals and x_vals to evaluate the GP"
        assert np.all(data.theta_vals is not None), "Must have validation data theta_vals and x_vals to evaluate the GP"
        
        #Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = np.concatenate((data.theta_vals, data.x_vals), axis =1)
        
        return feature_eval_data
    
    def set_train_test_data(self, sep_fact, seed):
        """
        Finds the simulation data to use as training/testing data
        
        Parameters
        ----------
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        seed: int or None, Determines seed for randomizations. None if seed is random
        
        Returns
        -------
        train_data: Instance of data class. Contains all theta, x, and y data for training data
        test_data: Instance of data class. Contains all theta, x, and y data for testing data
        
        Notes
        -----
        Sets feature_train_data, feature_test_data, and feature_val_data
        """
        assert isinstance(sep_fact, (float,int)), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        assert isinstance(seed, int), "seed must be int!"
        assert isinstance(self.gp_sim_data, Data), "self.gp_sim_data must be instance of Data"
        assert np.all(self.gp_sim_data.x_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.theta_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.y_vals is not None), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(self.gp_sim_data.bounds_x is not None), "Must have simulation x bounds to create train/test data"
        assert np.all(self.gp_sim_data.bounds_theta is not None), "Must have simulation theta bounds to create train/test data"
        
        #Find train indeces
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split()
        
        #Find unique theta_values
        unique_theta_vals = self.gp_sim_data.get_unique_theta()
        
        # Check which rows in theta_vals match the rows in Theta_unique based on theta_idx
        train_mask = np.isin(self.gp_sim_data.theta_vals, unique_theta_vals[train_idx])
        test_mask = np.isin(self.gp_sim_data.theta_vals, unique_theta_vals[train_idx], invert = True)

        # Get the indices of the matching rows
        train_rows_idx = np.all(train_mask, axis=1)
        test_rows_idx = np.all(test_mask, axis=1)

        # Use the indices to select the specific rows from theta_vals 
        #Set training data and set it as an instance of the data class
        theta_train = self.gp_sim_data.theta_vals[train_rows_idx]
        x_train = self.gp_sim_data.x_vals[train_rows_idx]
        y_train = self.gp_sim_data.y_vals[train_rows_idx]
        train_data = Data(theta_train, x_train, y_train, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, seed)
        self.train_data = train_data
        
        #Get test data and set it as an instance of the data class
        theta_test = self.gp_sim_data.theta_vals[test_rows_idx]
        x_test = self.gp_sim_data.x_vals[test_rows_idx]
        y_test = self.gp_sim_data.y_vals[test_rows_idx]
        test_data = Data(theta_test, x_test, y_test, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, seed)
        self.test_data = test_data
        
        #Set training and validation data features in GP_Emulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)
        
        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data
         
        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data
            
        return train_data, test_data
    
    def __eval_gp_sse_var(self, data, method, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO
        
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        """
        #Featurize data
        feature_eval_data = self.featurize_data(data)
        
        #Find length of theta and number of unique x in data arrays
        len_theta = data.get_num_theta()
        len_x = len(data.get_unique_x())
     
        #Make sse arrays as an empty lists. Will add one value for each training point
        sse_mean = []
        sse_var = []
        
        #Iterates over evey combination of theta to find the sse for each combination
        #Note to do this Xexp and X **must** use the same values
        if len_theta > 0: #Only do this if you actually have data
            for i in range(0, len_theta, len_x):
                sse_mean.append( sum((data.gp_mean[i:i+len_x] - exp_data.y_vals)**2) ) #Vector 
                error_points_sq = (2*(data.gp_mean[i:i+len_x] - exp_data.y_vals))**2 #Vector
                #sse_var = (2*(GP -y ))**2 * var
                sse_var.append( (error_points_sq@data.gp_var[i:i+len_x]) ) #This SSE_variance CAN'T be negative
        
        #Lists to arrays
        sse_mean = np.array(sse_mean)
        sse_var = np.array(sse_var)
        
        
        #For Method 2B, make sse and sse_var data in the log form
        if method.obj.value == 2:
            #Propogation of errors: stdev_ln(val) = stdev/val
            sse_var = sse_var/(sse_mean**2)
            #Set mean to new value
            sse_mean = np.log(sse_mean)
            
        
        #Set class parameters
        data.sse = sse_mean
        data.sse_var = sse_var
        
        return sse_mean, sse_var
    
    def eval_gp_sse_var_misc(self, misc_data, method, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the heat map data
        
        Parameters
        ----------
        misc_data, Instance of Data class, data to evaluate gp sse and sse variance for
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        misc_sse_mean: tensor, The sse derived from gp_mean evaluated over the test data 
        misc_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the test data 
        
        """
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods class"
        assert isinstance(misc_data , Data), "misc_data must be type Data"
        assert np.all(misc_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(misc_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(misc_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(misc_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        misc_sse_mean, misc_sse_var = self.__eval_gp_sse_var(misc_data, method, exp_data)
        
        return misc_sse_mean, misc_sse_var
    
    def eval_gp_sse_var_test(self, method, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the test data
        
        Parameters
        ----------
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        test_sse_mean: tensor, The sse derived from gp_mean evaluated over the test data 
        test_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the test data 
        
        """
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods class"
        assert all(isinstance(var , Data) for var in [self.test_data, exp_data]), "self.test_data and exp_data must be type Data"
        assert np.all(self.test_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.test_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.test_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(self.test_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data, method, exp_data)
        
        return test_sse_mean, test_sse_var
    
    def eval_gp_sse_var_val(self, method, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the validation data
        
        Parameters
        ----------
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        val_sse_mean: tensor, The sse derived from gp_mean evaluated over the validation data 
        val_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the validation data 
        
        """
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods class"
        assert all(isinstance(var , Data) for var in [self.gp_val_data, exp_data]), "self.gp_val_data and exp_data must be type Data"
        assert np.all(self.gp_val_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.gp_val_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.gp_val_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.gp_val_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data, method, exp_data)
        
        return val_sse_mean, val_sse_var
    
    def eval_gp_sse_var_cand(self, method, exp_data):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the candidate theta data
        
        Parameters
        ----------
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        
        Returns
        --------
        cand_sse_mean: tensor, The sse derived from gp_mean evaluated over the candidate theta data 
        cand_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the candidate theta data 
        
        """
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods class"
        assert all(isinstance(var , Data) for var in [self.cand_data, exp_data]), "self.cand_data and exp_data must be type Data"
        assert np.all(self.cand_data.x_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.cand_data.theta_vals is not None), "Must have testing data theta_vals and x_vals to evaluate the GP"
        assert np.all(self.cand_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.cand_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data, method, exp_data)
        
        return cand_sse_mean, cand_sse_var
    
    def calc_best_error(self, method, exp_data):
        """
        Calculates the best error of the model
        
        Parameters
        ----------
        method: Instance of GPBO_Methods, Class containing method information
        exp_data: Instance of Data class, Class containing at least theta, x, and y experimental data
        
        Returns
        -------
        best_error: float, the best error of the method
        
        """ 
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods class"
        assert all(isinstance(var , Data) for var in [self.train_data, exp_data]), "self.tain_data and exp_data must be type Data"
        assert np.all(self.train_data.x_vals is not None), "Must have simulation x, theta, and y data to calculate best error"
        assert np.all(self.train_data.theta_vals is not None), "Must have simulation x, theta, and y data to calculate best error"
        assert np.all(self.train_data.y_vals is not None), "Must have simulation x, theta, and y data to calculate best error"
        assert np.all(exp_data.x_vals is not None), "Must have exp_data x and y to calculate best error"
        assert np.all(exp_data.y_vals is not None), "Must have exp_data x and y to calculate best error"
        
        #Find length of theta and x in data arrays
        len_theta = self.train_data.get_num_theta()
        len_x = len(self.train_data.get_unique_x())
     
        #Make sse array as an empty list. Will add one value for each training point
        sse_train_vals = []
#         true_idx_list = [] #Used for error checking
        
        #Evaluate SSE by looping over the x values for each combination of theta and calculating SSE
        for i in range(0, len_theta, len_x):
            sse_train_vals.append( sum((self.train_data.y_vals[i:i+len_x] - exp_data.y_vals)**2) )#Scaler
#             true_idx_list.append(i) #Used for error checking

        #List to array
        sse_train_vals = np.array(sse_train_vals)   
        
        #Best error is the minimum of these values
        best_error = np.amin(sse_train_vals)
#         print(self.train_data.theta_vals[true_idx_list[np.argmin(sse_train_vals)]]) #For Error Checking, Returns theta associated with best value
        
        #For method 2B, use a log scaled best error
        if method.obj.value == 2:
            best_error = np.log(best_error)
        
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
        sim_data.ei = ei
        
        return ei
    
    def eval_ei_misc(self, misc_data, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        misc_data, Instance of Data class, data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error: float, the best error of the method
        method: instance of Method class, method for GP Emulation
        
        Returns
        -------
        ei: The expected improvement of all the data in sim_data
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
        ei = self.__eval_gp_ei(misc_data, exp_data, ep_bias, best_error, method)
        
        return ei
    
    def eval_ei_test(self, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function for testing data. In this case, ei
        
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
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
                   
        ei = self.__eval_gp_ei(self.test_data, exp_data, ep_bias, best_error, method)
        return ei
    
    def eval_ei_val(self, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function for validation data. In this case, ei
        
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
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
        ei = self.__eval_gp_ei(self.gp_val_data, exp_data, ep_bias, best_error, method)
        
        return ei
        
    def eval_ei_cand(self, exp_data, ep_bias, best_error, method):
        """
        Evaluates gp acquisition function for the candidate theta data. In this case, ei
        
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
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error, (float, int)), "best_error must be float or int"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
        ei = self.__eval_gp_ei(self.cand_data, exp_data, ep_bias, best_error, method)
        
        return ei
    
    def add_next_theta_to_train_data(self, theta_best_data):
        """
        Adds the theta with the highest ei to the training data set
        
        Parameters
        ----------
        theta_best: Instance of Data, The class containing the data relavent to theta_best
        """
        assert self.train_data is not None, "self.train_data must be Data"
        assert isinstance(self.train_data, Data), "self.train_data must be Data"
        assert isinstance(theta_best_data, Data), "theta_best_data must be Data"
        assert all(isinstance(var, np.ndarray) for var in [self.train_data.theta_vals,self.train_data.y_vals]), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"
        assert all(isinstance(var, np.ndarray) for var in [theta_best_data.theta_vals, theta_best_data.y_vals]), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"
        
        #Update training theta, x, and y separately
        self.train_data.theta_vals = np.vstack((self.train_data.theta_vals, theta_best_data.theta_vals))
        self.train_data.x_vals = np.vstack((self.train_data.x_vals, theta_best_data.x_vals))  
        self.train_data.y_vals = np.concatenate((self.train_data.y_vals, theta_best_data.y_vals))
        feature_train_data = self.featurize_data(self.train_data)
        
        #Reset training data feature array
        self.feature_train_data = feature_train_data
                                                                   
##Again, composition instead of inheritance      
class Expected_Improvement():  
    """
    The base class for acquisition functions
    Parameters
    
    Methods
    --------------
    __init__
    type_1(method)
    type_2(method)
    __calc_ei_emulator(gp_mean, gp_var, y_target)
    __calc_ei_log_emulator(gp_mean, gp_var, y_target)
    __ei_approx_ln_term(epsilon, best_error, gp_mean, gp_stdev, y_target, ep)
    __calc_ei_sparse(gp_mean, gp_var, y_target)
    __get_sparse_grids(dim, output=0,depth=3, rule="gauss-hermite", verbose = False, alpha = 0)
    """
    #AD Comment: What part of the acquisition function code can be generalized and what is specific to type1 and type2? 
    def __init__(self, ep_bias, gp_mean, gp_var, exp_data, best_error):
        """
        Parameters
        ----------
        ep_bias: instance of Exploration_Bias, class with information of exploration bias parameter
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_var: tensor, The GP model's variance evaluated over param_set
        exp_data: Instance of Data class, the experimental data to evaluate ei with
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
        columns = ["best_error", "z", "cdf", "pdf", "ei_term_1", "ei_term_2", "ei"]
        ei_term_df = pd.DataFrame(columns=columns)
    
        ei = np.zeros(len(self.gp_mean))

        for i in range(len(self.gp_mean)):
            pred_stdev = np.sqrt(self.gp_var[i]) #1xn_test
            #Checks that all standard deviations are positive
            if pred_stdev > 0:
                #Calculates z-score based on Eq. 6b in Wang and Dowling (2022), COCHE
                z = (self.best_error*self.ep_bias.ep_curr - self.gp_mean[i])/pred_stdev #scaler
                #Calculates ei based on Eq. 6a in Wang and Dowling (2022), COCHE
                #Explotation term
                ei_term_1 = (self.best_error*self.ep_bias.ep_curr - self.gp_mean[i])*norm.cdf(z) #scaler
                #Exploration Term
                ei_term_2 = pred_stdev*norm.pdf(z) #scaler
                ei[i] = ei_term_1 +ei_term_2 #scaler

                # Create a temporary DataFrame for the current row
                row_data = pd.DataFrame([[self.best_error, z, norm.cdf(z), norm.pdf(z), ei_term_1, ei_term_2, ei[0]]], columns=columns)

            else:
                #Sets ei to zero if standard deviation is zero
                ei[i] = 0
                # Create a temporary DataFrame for the current row
                row_data = pd.DataFrame([[self.best_error, None, None, None, None, None, ei]], columns=columns)
                
            # Concatenate the temporary DataFrame with the main DataFrame
            ei_term_df = pd.concat([ei_term_df.astype(row_data.dtypes), row_data], ignore_index=True)
        return ei, ei_term_df
        
    def type_2(self, method):
        """
        Calculates expected improvement of type 2 (emulator) GPBO
        
        Parameters
        ----------
        method: class, fully defined methods class which determines which method will be used
        
        Returns
        -------
        ei: float, The expected improvement of the parameter set
        """        
        columns = ["best_error", "z", "cdf", "pdf", "ei_term_1", "ei_term_2", "ei"]
        ei_term_df = pd.DataFrame(columns=columns)
        
        assert isinstance(method, GPBO_Methods), "method must be type GPBO_Methods"
        #Num thetas = #gp mean pts/number of x_vals for Type 2
        num_thetas = int(len(self.gp_mean)/self.exp_data.get_num_x_vals()) 
        #Define n as the number of x values
        n = self.exp_data.get_num_x_vals()
        #Initialize array of eis for eacch theta
        ei = np.zeros(num_thetas)
        #Loop over number of thetas in theta_val_set
        for i in range(num_thetas): #1 ei per theta and also 1 sse per theta   
            #Get gp mean and var for each set of x values
            #for ei, ensure that a gp mean and gp_var corresponding to a certain theta are sent
            gp_mean_i = self.gp_mean[i*n:(i+1)*n]
            gp_var_i = self.gp_var[i*n:(i+1)*n]
            
            #Calculate ei for a given theta (ei for all x over each theta)
            
            if method.method_name.value == 3: #2A
                #Calculate ei for a given theta (ei for all x over each theta)
                ei[i] = self.__calc_ei_emulator(gp_mean_i, gp_var_i, self.exp_data.y_vals)
                
            elif method.method_name.value == 4: #2B
                ei[i] = self.__calc_ei_log_emulator(gp_mean_i, gp_var_i, self.exp_data.y_vals)
                
            elif method.method_name.value == 5: #2C
                ei[i] = self.__calc_ei_sparse(gp_mean_i, gp_var_i, self.exp_data.y_vals)

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
        
        #Initialize ei as all zeros
        ei = np.zeros(len(gp_var))

        #Create a mask for values where var > 0. Set a value of 1e-14?
        pos_stdev_mask = (gp_var > 0)

        #Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            #Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_var_val = gp_var[valid_indices]
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]

            #If variance is close to zero this is important
            with np.errstate(divide = 'warn'):
                #Creates upper and lower bounds and described by Equation X in Manuscript
                bound_a = ((y_target_val - gp_mean_val) + np.sqrt(self.best_error*self.ep_bias.ep_curr))/pred_stdev_val
                bound_b = ((y_target_val - gp_mean_val) - np.sqrt(self.best_error*self.ep_bias.ep_curr))/pred_stdev_val
                bound_lower = np.minimum(bound_a,bound_b)
                bound_upper = np.maximum(bound_a,bound_b)        

                #Creates EI terms in terms of Equation X in Manuscript
                ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
                ei_term1_comp2 = (self.best_error*self.ep_bias.ep_curr) - (y_target_val - gp_mean_val)**2

                ei_term2_comp1 = 2*(y_target_val - gp_mean_val)*pred_stdev_val
                ei_eta_upper = -np.exp(-bound_upper**2/2)/np.sqrt(2*np.pi)
                ei_eta_lower = -np.exp(-bound_lower**2/2)/np.sqrt(2*np.pi)
                ei_term2_comp2 = (ei_eta_upper-ei_eta_lower)

                ei_term3_comp1 = bound_upper*ei_eta_upper
                ei_term3_comp2 = bound_lower*ei_eta_lower

                ei_term3_comp3 = (1/2)*scipy.special.erf(bound_upper/np.sqrt(2))
                ei_term3_comp4 = (1/2)*scipy.special.erf(bound_lower/np.sqrt(2))  

                ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp3
                ei_term3_psi_lower = ei_term3_comp2 + ei_term3_comp4

                ei_term1 = ei_term1_comp1*ei_term1_comp2
                ei_term2 = ei_term2_comp1*ei_term2_comp2
                ei_term3 = -gp_var_val*(ei_term3_psi_upper-ei_term3_psi_lower)

                #Set EI values of indecies where pred_stdev > 0 
                ei[valid_indices] = ei_term1 + ei_term2 + ei_term3
        
        #The Ei is the sum of the ei at each value of x
        ei_temp = np.sum(ei)
        
        return ei_temp

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
            
        #Initialize ei as all zeros
        ei = np.zeros(len(gp_var))

        #Create a mask for values where pred_stdev > 0 
        pos_stdev_mask = (gp_var > 0)

        #Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            #Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
        
            #Important when stdev is close to 0
            with np.errstate(divide = 'warn'):
                #Creates upper and lower bounds and described by Alex Dowling's Derivation
                bound_a = ((y_target_val - gp_mean_val) +np.sqrt(np.exp(self.best_error*self.ep_bias.ep_curr)))/pred_stdev_val #1xn
                bound_b = ((y_target_val - gp_mean_val) -np.sqrt(np.exp(self.best_error*self.ep_bias.ep_curr)))/pred_stdev_val #1xn
                bound_lower = np.minimum(bound_a,bound_b)
                bound_upper = np.maximum(bound_a,bound_b) 

                #Calculate EI
                args = (self.best_error, gp_mean_val, pred_stdev_val, y_target_val, self.ep_bias.ep_curr)
                ei_term_1 = (self.best_error*self.ep_bias.ep_curr)*( norm.cdf(bound_upper)-norm.cdf(bound_lower) )
                ei_term_2_out = np.array([integrate.quad(self.__ei_approx_ln_term, bl, bu, args=(self.best_error, gm, ps, yt, self.ep_bias.ep_curr)) for bl, bu, gm, ps, yt in zip(bound_lower, bound_upper, gp_mean_val, pred_stdev_val, y_target_val)])

                ei_term_2 = (-2)*ei_term_2_out[:,0] 
                term_2_abs_err = ei_term_2_out[:,1]
                
                #Add ei values to correct indecies.
                ei[valid_indices] = ei_term_1 + ei_term_2
        
        #The Ei is the sum of the ei at each value of x
        ei_temp = np.sum(ei)
  
        return ei_temp

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
        #Define inside term as the maximum of 1e-14 or abs((y_target - gp_mean - gp_stdev*epsilon))
        inside_term = max(1e-14, abs((y_target - gp_mean - gp_stdev*epsilon)) )

        ei_term_2_integral = math.log( inside_term )*norm.pdf(epsilon) 
        
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
        
        #Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0) 
        pos_stdev_mask = (gp_var >= 0)

        #Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            #Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            gp_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
        
            #Defines standard devaition
            gp_stdev = np.sqrt(gp_var) #1xn

            #Obtain Sparse Grid points and weights
            points_p, weights_p = self.__get_sparse_grids(len(y_target_val), output=0, depth=3, rule='gauss-hermite', verbose=False)

            # Calculate gp_var multiplied by points_p
            gp_stdev_points_p = gp_stdev_val * points_p

            # Calculate the SSE for all data points simultaneously
            sse_temp = np.sum((y_target_val[:, np.newaxis] - gp_mean_val[:, np.newaxis] - gp_stdev_points_p.T)**2, axis=0)

            # Apply -min operator (equivalent to max[SSE_Temp - (best_error*ep),0])
            min_list = -np.minimum(sse_temp - (self.best_error*self.ep_bias.ep_curr), 0)

            # Calculate EI_temp using vectorized operations
            ei_temp = np.dot(weights_p, min_list)
            
        else:
            ei_temp = 0
            
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
    
    Methods
    -------
    set_ep(Ep_enum)
    __set_ep_boyle()
    __set_ep_jasrasaria()
    __set_ep_constant()
    __set_ep_decay()
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
        e_inc: float, the increment for the Boyle's method for calculating exploration parameter: Recommendation is 1.5
        ep_f: float, The final exploration parameter value: Recommendation is 0
        improvement: Bool, Determines whether last objective was an improvement. Default False
        best_error: float, The lowest error value in the training data
        mean_of_var: float, The value of the average of all posterior variances
        
        
        Notes
        ------
        For all methods, ep is on domain [0, best_error (initial)] inclusive
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
        self.ep_max = 2
        self.ep_min = 0.5
        
    def __bound_ep(self, ep_val):
        """
        Bounds the value of a given exploration parameter between the minimum and maximum value
        
        Parameters
        ----------
        ep_val: int or float, the value of the exploration parameter
        
        Returns:
        --------
        ep_val: int/float, the value of the exploration parameter within self.ep_min and self.ep_max
        """
        assert isinstance(ep_val, (float, int)), "ep_val must be float or int!"
        if ep_val > self.ep_max:
            warnings.warn("setting ep_val to self.ep_max because it was too large")
            ep_val = self.ep_max
        elif ep_val < self.ep_min:
            warnings.warn("setting ep_val to self.ep_min because it was too small")
            ep_val = self.ep_min
        else:
            assert self.ep_max >= ep_val >= self.ep_min, "Starting exploration bias (ep0) must be greater than or equal to 0.5!"
                
        return ep_val
    
    def set_ep(self):
        """
        Updates value of exploration parameter based on one of the four alpha heuristics
        
        Notes
        --------
        Sets the current exploration parameter self.ep_curr, but does not return anything. Use Exploration_Bias.ep_curr() to return it
        
        """
        #Set ep0 and ep_f to the max if they are too large
        if self.ep0 is not None:
            self.ep0 = self.__bound_ep(self.ep0)
        if self.ep_f is not None:
            self.ep_f = self.__bound_ep(self.ep_f)
                
        if self.ep_enum.value == 1: #Constant if using constant method
            assert self.ep0 is not None 
            ep = self.__set_ep_constant()
            
        elif self.ep_enum.value == 2: #Decay
            assert self.ep0 is not None
            assert self.ep_f is not None 
            assert self.bo_iter_max is not None
            ep = self.__set_ep_decay()
            
        elif self.ep_enum.value == 3: #Boyle
            assert self.ep0 is not None
            assert self.ep_inc is not None
            ep = self.__set_ep_boyle()
            
        else: #Jasrasaria
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
        Creates a value for the exploration parameter based off of a decay heuristic. Note. Full decay by 1/2 max BO iters
        
        Returns
        --------
        ep: The exploration parameter for the iteration
        """
        assert self.bo_iter is not None
        assert self.bo_iter_max-1 >= self.bo_iter >= 0
        
        #Set ep_f to max value if it is too big
        #Initialize number of decay steps
        decay_steps = int(self.bo_iter_max/2)
        #Apply heuristic on 1st iteration and all steps until end of decay steps
        if self.bo_iter < decay_steps or self.bo_iter == 0:
            ep = self.ep0 + (self.ep_f - self.ep0)*(self.bo_iter/self.bo_iter_max)
        else: 
            ep = self.ep_f
            
        return ep
    
    def __set_ep_boyle(self):
        """
        Creates a value for the exploration parameter based on Boyle's Heuristic for GPO bounds
            
        Returns
        --------
        ep: The exploration parameter for the iteration
        
        Notes
        -----
        Based on Heuristic from Boyle, P., Gaussian Processes for regression and Optimisation, Ph.D, Victoria University of Wellington, Wellington, New Zealand, 2007
        For these parameters, ep gets normalized between 0 and 2 given an even mix of 1 is the starting point
        """
        #Set ep_curr as ep0 if it is not set
        if self.ep_curr is None:
            ep = self.ep0
        else:
            #Assert that improvement is not None
            assert self.improvement is not None
            #Apply a version of Boyle's heuristic
            #In original Boyle, you want to gradually expand or shrink your bounds
            #We take this concept for ep to increase exploration when improvement is FALSE and increase it when TRUE
            if self.improvement == True:
                #If we improved last time, Decrease exploration
                ep = self.ep_curr/self.ep_inc
            else:
                #If we did not, Increase Exploration
                ep = self.ep_curr*self.ep_inc
                
        # Ensure that ep stays within the bounds
        ep = self.__bound_ep(ep)

        return ep
    
    def __set_ep_jasrasaria(self):
        """
        Creates a value for the exploration parameter based off of Jasrasaria's heuristic

        Returns
        --------
        ep: The exploration parameter for the iteration
        
        Notes
        -----
        Heuristic from Jasrasaria, D., & Pyzer-Knapp, E. O. (2018). Dynamic Control of Explore/Exploit Trade-Off In Bayesian Optimization. http://arxiv.org/abs/1807.01279
        """
        assert self.best_error is not None
        assert self.mean_of_var is not None
            
        #Apply Jasrasaria's Heuristic
        if self.best_error > 0:
            ep = 1 + (self.mean_of_var/self.best_error**2)
        else:
            ep = self.ep_max
            
        # Ensure that ep stays within the bounds
        ep = self.__bound_ep(ep)
        
        return ep

class BO_Results:
    """
    The base class for storing important BO Results
    
    Methods:
    --------
    __init__
    """
    
    # Class variables and attributes
    def __init__(self, configuration, simulator_class, exp_data_class, list_gp_emulator_class, results_df, 
                 max_ei_details_df, why_term, heat_map_data_dict):
        """
        Parameters
        ----------
        configuration: dictionary, dictionary containing the configuration of the BO algorithm
        simulator_class: Instance of Simulator class, class containing values of simulation parameter data at each BO iteration
        exp_data_class: The experimental data for the workflow
        list_gp_emulator_class: list of GP_Emulator instances, contains all gp_emulator information at each BO iter
        results_df: pandas dataframe, dataframe including the values pertinent to BO for all BO runs
        max_ei_details_df: pandas dataframe, dataframe including ei components of the best EI at each iter
        heat_map_data_dict: dict, heat map data for each set of 2 parameters indexed by parameter names "param_1-param_2"
        """
        # Constructor method
        self.configuration = configuration
        self.simulator_class = simulator_class
        self.exp_data_class = exp_data_class
        self.results_df = results_df
        self.max_ei_details_df = max_ei_details_df
        self.why_term = why_term
        self.list_gp_emulator_class = list_gp_emulator_class
        self.heat_map_data_dict = heat_map_data_dict     
    
class GPBO_Driver:
    """
    The base class for running the GPBO Workflow
    
    Methods
    --------------
    __init__
    __gen_emulator()
    __opt_with_scipy(neg_ei)
    __scipy_fxn(theta,neg_ei, best_error)
    create_heat_map_param_data()
    __augment_train_data(theta_best)
    create_data_instance_from_theta(theta_array)
    __run_bo_iter(gp_model, iteration)
    __run_bo_to_term(gp_model)
    __run_bo_workflow()
    run_bo_restarts()
    save_data(restart_bo_results)
    """
    # Class variables and attributes
    
    def __init__(self, cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, gp_emulator, ep_bias, gen_meth_theta):
        """
        Parameters
        ----------
        cs_params: Instance of CaseStudyParameters class, class containing the values associated with CaseStudyParameters
        method: Instance of GPBO_Methods class, Class containing GPBO method information
        simulator: Instance of Simulator class, class containing values of simulation parameters
        exp_data: Instance of Data class, Class containing at least theta, x, and y experimental data
        sim_data: Instance of Data class, class containing at least theta, x, and y simulation data
        sim_sse_data: Instance of Data class, class containing at least theta, x, and sse simulation data
        val_data: Instance of Data class or None, class containing at least theta and x simulation data or None
        val_sse_data: Instance of Data class or None, class containing at least theta and x sse simulation data or None
        gp_emulator: Instance of GP_Emulator class, class containing gp_emulator data (set after training)
        ep_bias: Instance of Exploration_Bias class, class containing exploration parameter info
        gen_meth_theta: Instance of Gen_meth_enum or None: The method by which simulation data is generated. For heat map making
        """
        assert isinstance(cs_params, CaseStudyParameters), "cs_params must be instance of CaseStudyParameters"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert isinstance(simulator, Simulator), "simulator must be instance of Simulator"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(sim_data, Data), "sim_data must be instance of Data"
        assert isinstance(sim_sse_data, Data), "sim_sse_data must be instance of Data"
        assert isinstance(val_data, Data) or val_data is None, "val_data must be instance of Data or None"
        assert isinstance(val_sse_data, Data) or val_sse_data is None, "val_sse_data must be instance of Data or None"
        assert isinstance(gp_emulator, (Type_1_GP_Emulator, Type_2_GP_Emulator)) or gp_emulator is None, "gp_emulator must be instance of Type_1_GP_Emulator, Type_2_GP_Emulator, or None"
        assert isinstance(ep_bias, Exploration_Bias), "ep_bias must be instance of Exploration_Bias"
        assert isinstance(gen_meth_theta, Gen_meth_enum), "gen_meth_theta must be instance of Gen_meth_enum"

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
        self.gen_meth_theta = gen_meth_theta
        self.bo_iter_term_frac = 0.3 #The fraction of iterations after which to terminate bo if no sse improvement is made
        self.sse_penalty = 1e7 #The penalty the __scipy_opt function gets for choosing nan theta values
               
    
    def __gen_emulator(self):
        """
        Sets GP Emulator class (equipped with training data) and validation data based on the method class instance
        
        Parameters:
        -----------
        kernel: enum class instance, Determines which GP Kerenel to use
        lenscl: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        outputscl: float or None, Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_GP: int, number of times to restart GP training
        
        Returns:
        --------
        gp_emulator: Instance of the GP_Emulator class. Class for the GP emulator
        """
        #Determine Emulator Status, set gp_data data, and ininitalize correct GP_Emulator child class
        if self.method.emulator == False:
            all_gp_data = self.sim_sse_data
            all_val_data = self.val_sse_data
            gp_emulator = Type_1_GP_Emulator(all_gp_data, all_val_data, None, None, None, self.cs_params.kernel, self.cs_params.lenscl, self.simulator.noise_std, self.cs_params.outputscl, self.cs_params.retrain_GP, self.cs_params.seed, None, None, None, None)
        else:
            all_gp_data = self.sim_data
            all_val_data = self.val_data
            gp_emulator = Type_2_GP_Emulator(all_gp_data, all_val_data, None, None, None, self.cs_params.kernel, self.cs_params.lenscl, self.simulator.noise_std, self.cs_params.outputscl, self.cs_params.retrain_GP, self.cs_params.seed, None, None, None, None)
            
        return gp_emulator
    
    
    def __get_best_error(self):
        """
        Helper function to calculate the best error given the method.
        
        Returns
        -------
        best_error: float, the best error of the GPBO workflow
        """
        
        if self.method.emulator == False:
            #Type 1 best error is inferred from training data 
            best_error = self.gp_emulator.calc_best_error()
        else:
            #Type 2 best error must be calculated given the experimental data
            best_error = self.gp_emulator.calc_best_error(self.method, self.exp_data)
        
        return best_error
        
    def __make_starting_opt_pts(self):
        """
        Makes starting point for optimization with scipy
        """
        #If validation data doesn't exist or is shorter than the number of times you want to retrain
        if self.gp_emulator.gp_val_data is None or len(self.gp_emulator.gp_val_data.get_unique_theta()) < self.cs_params.retrain_GP:
            #Create validation points equal to number of retrain_GP
            #Number of x points will always be 1 because we only need the theta values
            num_x = 1
            #Gen method will always be LHS for starting points theta and x
            gen_meth = Gen_meth_enum(1) 
            #Create starting point data
            sp_data = self.simulator.gen_sim_data(self.cs_params.retrain_GP, num_x, gen_meth, gen_meth, self.cs_params.sep_fact, True)
            #Find unique theta values
            starting_pts = sp_data.get_unique_theta()
        #Otherwise, your starting point array is your validation data unique theta values
        else:
            #Find unique theta values
            starting_pts = self.gp_emulator.gp_val_data.get_unique_theta()
            
        return starting_pts
    
    
    def __opt_with_scipy(self, neg_ei):
        """
        Optimizes a function with scipy.optimize
        
        Parameters
        ----------
        neg_ei: bool, whether to use -1*ei (True) or sse (False) as the objective function for minimization
        
        Returns:
        --------
        best_val: float, The optimized value of the function
        best_theta: ndarray, The theta set corresponding to val_best
        """
        
        assert isinstance(neg_ei, bool), "neg_ei must be bool!"
        #Note add +1 because index 0 counts as 1 reoptimization
        if self.cs_params.reoptimize_obj > 50:
            warnings.warn("The objective will be reoptimized more than 50 times!")
        
        #Set seed
        if self.cs_params.seed is not None:
            np.random.seed(self.cs_params.seed)
                           
        #Note. For reoptimizing -ei/sse generate and use a validation point as a starting point for optimization
        unique_val_thetas = self.__make_starting_opt_pts()
#         unique_val_thetas = self.gp_emulator.train_data.get_unique_theta()
            
        #Initialize val_best and best_theta
        best_vals = np.full(self.cs_params.reoptimize_obj+1, np.inf)
        best_thetas = np.zeros((self.cs_params.reoptimize_obj+1, self.gp_emulator.train_data.get_dim_theta()))
        
        #Calc best error
        best_error = self.__get_best_error()
            
        #Find bounds and arguments for function
        #Unnormalize feature data for calculation if necessary
        if self.cs_params.normalize == True:
            #If noramlized, bounds for theta normalization are between 0 and 1
            bnds = np.tile([0, 1], (self.gp_emulator.train_data.get_dim_theta(), 1))
        else:
            #Otherwise bounds are defined as they are
            bnds = self.simulator.bounds_theta_reg.T #Transpose bounds to work with scipy.optimize
        #Need to account for normalization here (make bounds array of [0,1]^dim_theta)
        
        #Choose values of theta from validation set at random
        theta_val_idc = list(range(len(unique_val_thetas)))
    
        ## Loop over each validation point/ a certain number of validation point thetas
        for i in range(self.cs_params.reoptimize_obj+1):
            #Choose a random index of theta to start with
            unique_theta_index = random.sample(theta_val_idc, 1)
            theta_guess = unique_val_thetas[unique_theta_index].flatten()

            # try:
            print(theta_guess, bnds, neg_ei, best_error)
            #Call scipy method to optimize EI given theta
            #Using L-BFGS-B instead of BFGS because it allowd for bounds
            best_result = optimize.minimize(self.__scipy_fxn, theta_guess, bounds=bnds, method = "L-BFGS-B", args=(neg_ei,best_error))
            #Add ei and best_thetas to lists as appropriate
            best_vals[i] = best_result.fun
            best_thetas[i] = best_result.x
            # except ValueError: 
            #     #If the intialized theta causes scipy.optimize to choose nan values, set the value of min sse and its theta to non
            #     best_vals[i] = np.nan
            #     best_thetas[i] = np.full(self.gp_emulator.train_data.get_dim_theta(), np.nan)
        
        #Choose a single value with the lowest -ei or sse
        #In the case that 2 point have the same -ei or sse and this point is the lowest, this lets us pick one at random rather than always just choosing a certain point
        min_value = np.nanmin(best_vals) #Find lowest value
        min_indices = np.where( np.isclose(best_vals, min_value, rtol=1e-7) )[0] #Find all indeces where there may be a tie
        rand_min_idx = np.random.choice(min_indices) #Choose one at random as the next step
        
        best_val = best_vals[rand_min_idx]
        best_theta = best_thetas[rand_min_idx]
        
        #Since we minimize -ei, multiply by -1 to get the maximum value of ei
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
        #Check that any of the values are not NaN
        #If they are nan
        if np.isnan(theta).any():
            #If there are nan values, set neg ei to -1 
            if neg_ei == True:
                obj = -1
            #Set sse to self.sse_penalty
            else:
                obj = self.sse_penalty
                
        #If not, continue the algorithm normally
        else:
            candidate = Data(None, self.exp_data.x_vals, None, None, None, None, None, None, self.simulator.bounds_theta_reg, self.simulator.bounds_x, self.cs_params.sep_fact, self.cs_params.seed)

            #Create feature data for candidate point
            if self.method.emulator == False:
                candidate_theta_vals = theta.reshape(1,-1)
            else:
                candidate_theta_vals = np.repeat(theta.reshape(1,-1), self.exp_data.get_num_x_vals() , axis =0)

            candidate.theta_vals = candidate_theta_vals  
            self.gp_emulator.cand_data = candidate

            #Set candidate point feature data
            self.gp_emulator.feature_cand_data = self.gp_emulator.featurize_data(self.gp_emulator.cand_data)

            #Evaluate GP mean/ stdev at theta
            cand_mean, cand_var = self.gp_emulator.eval_gp_mean_var_cand()

            #Evaluate SSE & SSE stdev at theta
            if self.method.emulator == False:
                #For Type 1 GP, the sse and sse_var are directly inferred from the gp_mean and gp_var
                cand_sse_mean, cand_sse_var = self.gp_emulator.eval_gp_sse_var_cand()
            else:
                #For Type 2 GP, the sse and sse_var are calculated from the gp_mean, gp_var, and experimental data
                cand_sse_mean, cand_sse_var = self.gp_emulator.eval_gp_sse_var_cand(self.method, self.exp_data)

            #Calculate objective fxn
            if neg_ei == False:
                #Objective to minimize is log(sse) if using 1B or 2B, and sse for all other methods
                obj = cand_sse_mean
                
            else:
                if self.method.emulator == False:
                    ei_output = self.gp_emulator.eval_ei_cand(self.exp_data, self.ep_bias, best_error)
                    obj = -1*ei_output[0]
                    # ei_terms = ei_output[1]
                    print(obj)
                else:
                    obj = -1*self.gp_emulator.eval_ei_cand(self.exp_data, self.ep_bias, best_error, self.method)
            
        return obj

    def create_heat_map_param_data(self, n_points_set = None):
        """
        Creates parameter sets that can be used to create heat maps of data at any given iteration
               
        Returns:
        --------
        heat_map_data_dict: dict, heat map data for each set of 2 parameters indexed by parameter name tuple ("param_1,param_2")
        """      
        assert isinstance(self.gp_emulator, (Type_1_GP_Emulator, Type_2_GP_Emulator)), "self.gp_emulator must be instance of Type_1_GP_Emulator or Type_2_GP_Emulator"
        assert isinstance(self.gp_emulator.gp_sim_data, Data), "self.gp_emulator.gp_sim_data must be an instance of Data!"
        assert isinstance(self.gen_meth_theta, Gen_meth_enum), "self.gen_meth_theta must be instance of Gen_meth_enum"
        assert isinstance(self.exp_data.x_vals, (np.ndarray)), "self.exp_data.x_vals must be np.ndarray"
        
        #Create list of heat map theta data
        heat_map_data_dict = {}
        
        #Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(0,self.simulator.dim_theta-1,self.simulator.dim_theta)
        #Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype = int)
        
        #If no number of points is set, use the length of the unique simulation thetas
        if n_points_set == None:
            n_thetas_points = len(self.gp_emulator.gp_sim_data.get_unique_theta()) #Use 20 pts for heat map data
            #Initialze meshgrid-like set of theta values at their true values 
            #If points were generated with an LHS, the number of points per parameter is n_thetas_points for the meshgrid
            if self.gen_meth_theta.value == 1:
                n_points = n_thetas_points
            else:
                #For a meshgrid, the number of theta values/ parameter is n_thetas_points for the meshgrid ^(1/theta_dim)
                n_points = int((n_thetas_points)**(1/self.simulator.dim_theta))

        #Meshgrid set always defined by n_points**2
        theta_set = np.tile(np.array(self.simulator.theta_true), (n_points**2, 1))
        
        #Unnormalize x_vals if necessary
        norm_x_vals = self.exp_data.x_vals
        if self.cs_params.normalize == True:
            lower_bound = self.simulator.bounds_x[0]
            upper_bound = self.simulator.bounds_x[1]
            norm_x_vals = norm_x_vals*(upper_bound - lower_bound) + lower_bound
        
        #Infer how many times to repeat theta and x values given that heat maps are meshgrid form by definition
        #The meshgrid of parameter values created below is symmetric, therefore, x is repeated by n_points**2 for a 2D meshgrid
        repeat_x = n_points**2 #Square because only 2 values at a time change
        x_vals = np.vstack([norm_x_vals]*repeat_x)
        repeat_theta = self.exp_data.get_num_x_vals()

        #Loop over all possible theta combinations of 2
        for i in range(len(mesh_combos)):
            #Create a copy of the true values to change the mehsgrid valus on
            theta_set_copy = np.copy(theta_set)
            #Set the indeces of theta_set for evaluation as each row of mesh_combos
            idcs = mesh_combos[i]
            #define name of parameter set as tuple ("param_1,param_2")
            data_set_name = (self.simulator.theta_true_names[idcs[0]], self.simulator.theta_true_names[idcs[1]])

            #Create a meshgrid of values of the 2 selected values of theta and reshape to the correct shape
            #Assume that theta1 and theta2 have equal number of points on the meshgrid
            theta1 = np.linspace(self.simulator.bounds_theta_reg[0][idcs[0]], self.simulator.bounds_theta_reg[1][idcs[0]], n_points)
            theta2 = np.linspace(self.simulator.bounds_theta_reg[0][idcs[1]], self.simulator.bounds_theta_reg[1][idcs[1]], n_points)
            theta12_mesh = np.array(np.meshgrid(theta1, theta2))
            theta12_vals = np.array(theta12_mesh).T.reshape(-1,2)
            
            #Set initial values for evaluation (true values) to meshgrid values
            theta_set_copy[:,idcs] = theta12_vals
            
            #Put values into instance of data class
            #Create data set based on emulator status
            if self.method.emulator == True:
                #Repeat the theta vals for Type 2 methods to ensure that theta and x values are in the correct form for evaluation with gp_emulator.eval_gp_mean_heat_map()
                theta_vals =  np.repeat(theta_set_copy, repeat_theta , axis =0)
                data_set = Data(theta_vals, x_vals, None,None,None,None,None,None, self.simulator.bounds_theta_reg, self.simulator.bounds_x, self.cs_params.sep_fact, self.cs_params.seed)
            else:
                data_set = Data(theta_set_copy, norm_x_vals, None,None,None,None,None,None, self.simulator.bounds_theta_reg, self.simulator.bounds_x, self.cs_params.sep_fact, self.cs_params.seed)
#             #normalize values between 0 and 1 if necessary
            if self.cs_params.normalize == True:
                data_set = data_set.norm_feature_data()
            #Append data set to dictionary with name
            heat_map_data_dict[data_set_name] = data_set
            
        return heat_map_data_dict
                
    def __augment_train_data(self, theta_best):
        """
        Augments training data given a new point

        Parameters
        ----------
        theta_best: ndarray, The theta value associated with the scipy optimize calculated best theta
        
        Returns:
        --------
        train_data: ndarray. The training parameter set with the augmented theta values
        """
        theta_best_data = self.create_data_instance_from_theta(theta_best)

        #Augment training theta, x, and y/sse data
        self.gp_emulator.add_next_theta_to_train_data(theta_best_data)
                   
    def create_data_instance_from_theta(self, theta_array):
        """
        Creates instance of Data from an nd.array theta set
        
        Parameters
        ----------
        theta_array: np.ndarray, Array of theta values to turn into an instance of Data
        
        Returns
        --------
        theta_data: instance of Data, Data for the theta_array
        """
        assert isinstance(theta_array, np.ndarray), "theta_array must be np.ndarray"
        assert len(theta_array.shape) == 1, "theta_array must be 1D"
        assert isinstance(self.exp_data.x_vals, (np.ndarray)), "self.exp_data.x_vals must be np.ndarray"
        
        #Repeat the theta best array once for each x value
        #Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.gen_y_data
        theta_arr_repeated = np.repeat(theta_array.reshape(1,-1), self.exp_data.get_num_x_vals() , axis =0)
        #Add instance of Data class to theta_best
        theta_arr_data = Data(theta_arr_repeated, self.exp_data.x_vals, None, None, None, None, None, None, self.simulator.bounds_theta_reg, self.simulator.bounds_x, self.cs_params.sep_fact, self.cs_params.seed)
        #Calculate y values and sse for theta_best with noise
        theta_arr_data.y_vals = self.simulator.gen_y_data(theta_arr_data, self.simulator.noise_mean, self.simulator.noise_std)  
        
        #Set the best data to be in sse form if using a type 1 GP
        if self.method.emulator == False:
            theta_arr_data = self.simulator.sim_data_to_sse_sim_data(self.method, theta_arr_data, self.exp_data, self.cs_params.sep_fact, False)
            
        return theta_arr_data
        
    def __run_bo_iter(self, gp_model, iteration):
        """
        Runs a single GPBO iteration
        
        Parameters
        ----------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor, GP emulator for workflow
        iteration: int, The iteration of bo in progress
        
        Returns:
        --------
        iter_df: pd.DataFrame, Dataframe containing the results from the GPBO Workflow for iteration
        gp_emulator_curr: Instance of GP_Emulator, The class used for this iteration of the GPBO workflow
        """
        #Start timer
        time_start = time.time()
        
        #Train GP model (this step updates the model to a trained model)
        self.gp_emulator.train_gp(gp_model)
        
        #Calcuate best error
        best_error = self.__get_best_error()
        
        #Add not log best error to ep_bias
        if iteration == 0 or self.ep_bias.ep_enum.value == 4:
            #Since best error is squared when used in Jasrasaria calculations, the value will always be >=0      
            self.ep_bias.best_error = best_error
                        
        #Calculate mean of var for validation set if using Jasrasaria heuristic
        if self.ep_bias.ep_enum.value == 4:
            #Calculate average gp mean and variance of the validation set
            val_gp_mean, val_gp_var = self.gp_emulator.eval_gp_mean_var_val()
            #For emulator methods, the mean of the variance should come from the sse variance
            if self.method.emulator == True:
                #Redefine gp_mean and gp_var to be the mean and variane of the sse
                val_gp_mean, val_gp_var = self.gp_emulator.eval_gp_sse_var_val(self.method, self.exp_data)
                
            #Check for ln(sse) values
            if self.method.obj.value == 2:
                #For 2B and 1B, propogate errors associated with an unlogged sse value
                val_gp_var = val_gp_var*np.exp(val_gp_mean)**2             

            #Set mean of sse variance
            mean_of_var = np.average(val_gp_var)
            self.ep_bias.mean_of_var = mean_of_var
            
        #Set initial exploration bias and bo_iter
        if self.ep_bias.ep_enum.value == 2:
            self.ep_bias.bo_iter = iteration
        
        #Calculate new ep. Note. It is extemely important to do this AFTER setting the ep_max
        self.ep_bias.set_ep()
        
        print("max ei to be optimized")

        #Call optimize acquistion fxn
        max_ei, max_ei_theta = self.__opt_with_scipy(True)
        
        #Create data class instance for max_ei_theta
        max_ei_theta_data = self.create_data_instance_from_theta(max_ei_theta)
        #Evaluate GP mean/ stdev at max_ei_theta
        feat_max_ei_theta_data = self.gp_emulator.featurize_data(max_ei_theta_data)
        max_ei_theta_data.gp_mean, max_ei_theta_data.gp_var = self.gp_emulator.eval_gp_mean_var_misc(max_ei_theta_data, feat_max_ei_theta_data)
        
        

        #Call optimize objective function
        min_sse, min_sse_theta = self.__opt_with_scipy(False)
        
        print("min sse optimized")
        #Find min sse using the true function value
        #Turn min_sse_theta into a data instance (including generating y_data)
        min_theta_data = self.create_data_instance_from_theta(min_sse_theta)
        
        #If type 2, turn it into sse_data
        #Set the best data to be in sse form if using a type 2 GP and find the min sse
        if self.method.emulator == True:
            min_sse_theta_data = self.simulator.sim_data_to_sse_sim_data(self.method, min_theta_data, self.exp_data, self.cs_params.sep_fact, False)
            min_sse_sim = min_sse_theta_data.y_vals
            #Evaluate SSE & SSE stdev at max ei theta
            max_ei_theta_data.sse, max_ei_theta_data.sse_var = self.gp_emulator.eval_gp_sse_var_misc(max_ei_theta_data, self.method,
                                                                                                    self.exp_data)
            #Evaluate max EI terms at theta
            ei_max = self.gp_emulator.eval_ei_misc(max_ei_theta_data, self.exp_data, self.ep_bias, best_error, self.method)
            #Dummy variable for now
            iter_max_ei_terms = pd.DataFrame()
        #Otherwise the sse data is the original (scaled) data
        else:
            min_sse_sim = min_theta_data.y_vals           
            #Evaluate SSE & SSE stdev at max ei theta
            max_ei_theta_data.sse, max_ei_theta_data.sse_var = self.gp_emulator.eval_gp_sse_var_misc(max_ei_theta_data)
            #Evaluate max EI terms at theta
            ei_max, iter_max_ei_terms = self.gp_emulator.eval_ei_misc(max_ei_theta_data, self.exp_data, self.ep_bias, best_error)
        
        #Turn min_sse_sim value into a float (this makes analyzing data from csvs and dataframes easier)
        min_sse_sim = min_sse_sim[0]
                
        #calculate improvement if using Boyle's method to update the exploration bias
        if self.ep_bias.ep_enum.value == 3:
            #Improvement is true if the min sim sse found is lower than (not log) best error, otherwise it's false
            if min_sse_sim < best_error:
                improvement = True
            else:
                improvement = False
            #Set ep improvement
            self.ep_bias.improvement = improvement
                 
        #Calc time/ iter
        time_end = time.time()
        time_per_iter = time_end-time_start
        
        #Create Results Pandas DataFrame for 1 iter
        #Return SSE and not log(SSE) for 'Min Obj', 'Min Obj Act', 'Theta Min Obj'
        column_names = ['Best Error', 'Exploration Bias', 'Max EI', 'Theta Max EI', 'Min Obj', 'Min Obj Act', 'Theta Min Obj', 'Time/Iter']
        iter_df = pd.DataFrame(columns=column_names)
        bo_iter_results = [best_error, float(self.ep_bias.ep_curr), max_ei, max_ei_theta, min_sse, min_sse_sim, min_sse_theta, time_per_iter]
        # Add the new row to the DataFrame
        iter_df.loc[0] = bo_iter_results
        
        #Create a copy of the GP Emulator Class for this iteration
        gp_emulator_curr = copy.deepcopy(self.gp_emulator)
              
        #Call __augment_train_data to append training data
        self.__augment_train_data(max_ei_theta)
        
        return iter_df, iter_max_ei_terms, gp_emulator_curr
    
    def __run_bo_to_term(self, gp_model):
        """
        Runs multiple GPBO iterations
        
        Params:
        -------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor, GP emulator for workflow
        
        Returns:
        --------
        iter_df: pd.DataFrame, Dataframe containing the results from the GPBO Workflow for all iterations
        list_gp_emulator_class: list of instances of GP_Emulator, The classes used for all iterations of the GPBO workflow
        """
        assert 0 < self.bo_iter_term_frac <= 1, "self.bo_iter_term_frac must be between 0 and 1"
        #Initialize bo params
        column_names = ['Best Error', 'Exploration Bias', 'Max EI', 'Theta Max EI', 'Min Obj', 'Min Obj Act', 'Theta Min Obj', 'Min Obj Cum.', 'Theta Min Obj Cum.', 'Time/Iter']
        results_df = pd.DataFrame(columns=column_names)
        max_ei_details_df = pd.DataFrame()
        list_gp_emulator_class = []
        why_term = "max_budget"
        #Initilize terminate
        terminate = False
        
        #Do Bo iters while stopping criteria is not met
        while terminate == False:
            #Initialize count
            count = 0
            #Loop over number of max bo iters
            for i in range(self.cs_params.bo_iter_tot):
                #Output results of 1 bo iter and the emulator used to get the results
                iter_df, iter_max_ei_terms, gp_emulator_class = self.__run_bo_iter(gp_model, i) #Change me later
                #Add results to dataframe
                results_df = pd.concat([results_df.astype(iter_df.dtypes), iter_df], ignore_index=True)
                max_ei_details_df = pd.concat([max_ei_details_df, iter_max_ei_terms])
                #At the first iteration
                if i == 0:
                    #Then the minimimum is defined by the first value of the objective function you calculate
                    # results_df["Min Obj Cum."].iloc[i] = results_df["Min Obj Act"].iloc[i]
                    results_df.loc[i, "Min Obj Cum."] = results_df.loc[i, "Min Obj Act"]
                    #The Theta values are then inferred
                    results_df.at[i, "Theta Min Obj Cum."] = results_df["Theta Min Obj"].iloc[i]
                    #improvement is defined as infinity on 1st iteration (something is always better than nothing)
                    improvement = np.inf 
                #If it is not the 1st iteration and your current Min Obj value is smaller than your previous Overall Min Obj
                elif results_df["Min Obj Act"].iloc[i] < results_df["Min Obj Cum."].iloc[i-1]:
                    #Then the New Cumulative Minimum objective value is the current minimum objective value
                    results_df.loc[i, "Min Obj Cum."] = results_df.loc[i, "Min Obj Act"]
                    #The Thetas are inferred
                    results_df.at[i, "Theta Min Obj Cum."] = results_df["Theta Min Obj"].iloc[i].copy()
                    #And the improvement is defined as the difference between the last Min Obj Cum. and current Obj Min (unscaled)
                    if self.method.obj.value == 1:
                        improvement = results_df["Min Obj Cum."].iloc[i-1] - results_df["Min Obj Act"].iloc[i]
                    else:
                        improvement = np.exp(results_df["Min Obj Cum."].iloc[i-1]) - np.exp(results_df["Min Obj Act"].iloc[i])
                #Otherwise
                else:
                    #The minimum objective for all the runs is the same as it was before
                    results_df.loc[i, "Min Obj Cum."] = results_df.loc[i-1, "Min Obj Cum."]
                    #And so are the thetas
                    results_df.at[i, "Theta Min Obj Cum."] = results_df['Theta Min Obj Cum.'].iloc[i-1].copy()
                    #And the improvement is defined as 0, since it must be non-negative
                    improvement = 0

                #Add gp emulator data from that iteration to list
                list_gp_emulator_class.append( gp_emulator_class )
                
                #Call stopping criteria after 1st iteration and update improvement counter
                if improvement < self.cs_params.obj_tol:
                    count +=1
                if i > 0:
                    #Terminate if max ei is less than the tolerance twice in a row
                    if results_df["Max EI"].iloc[i] < self.cs_params.ei_tol and results_df["Max EI"].iloc[i-1] < self.cs_params.ei_tol:
                        why_term = "ei"
                        break
                    #Terminate if small sse progress over 1/3 of total iteration budget
                    elif count >= int(self.cs_params.bo_iter_tot*self.bo_iter_term_frac):
                        why_term = "obj"
                        break
                    else:
                        terminate = False
                        
            #Terminate if you hit the max budget of iterations or the loop is broken
            terminate = True
            
        #Reset the index of the pandas df
        results_df = results_df.reset_index()
        
        #Create df for ei and add those results here
        max_ei_details_df.columns=iter_max_ei_terms.columns.tolist()
        max_ei_details_df = max_ei_details_df.reset_index(drop=True)

        return results_df, max_ei_details_df, list_gp_emulator_class, why_term
        
        
    def __run_bo_workflow(self):
        """
        Runs a single GPBO method through all bo iterations and reports the data for that run of the method
        
        Returns:
        --------
        bo_results: Instance of BO_Results class, Includes the results related to a set of BO iters.
        """
        
        #Initialize gp_emualtor class
        gp_emulator = self.__gen_emulator()
        self.gp_emulator = gp_emulator
        
        #Choose training data
        train_data, test_data = self.gp_emulator.set_train_test_data(self.cs_params.sep_fact, self.cs_params.seed)
        
        #Initilize gp model
        gp_model = self.gp_emulator.set_gp_model()
        
        #Reset ep_bias to None for each workflow restart
        self.ep_bias.ep_curr = None
        
        ##Call bo_iter
        results_df, max_ei_details_df, list_gp_emulator_class, why_term = self.__run_bo_to_term(gp_model)
        
        #Set results
        bo_results = BO_Results(None, None, self.exp_data, list_gp_emulator_class, results_df, 
                                max_ei_details_df, why_term, None)
        
        return bo_results

    
    def run_bo_restarts(self):
        """
        Runs multiple GPBO restarts
        
        Returns:
        --------
        restart_bo_results, list of instances of BO_Results, Includes the results related to a set of Bo iters for all restarts
        """
        restart_bo_results = []
        simulator_class = self.simulator
        configuration = {"DateTime String" : self.cs_params.DateTime,
                         "Method Name Enum Value" : self.method.method_name.value,
                         "Case Study Name" : self.cs_params.cs_name,
                         "Number of Parameters": len(self.simulator.theta_true_names),
                         "Exploration Bias Method Value" : self.ep_bias.ep_enum.value,
                         "Separation Factor" : self.cs_params.sep_fact,
                         "Normalize" : self.cs_params.normalize,
                         "Initial Kernel": self.cs_params.kernel,
                         "Initial Lengthscale": self.cs_params.lenscl,
                         "Initial Outputscale": self.cs_params.outputscl,
                         "Retrain GP": self.cs_params.retrain_GP,
                         "Reoptimize Obj": self.cs_params.reoptimize_obj,
                         "Heat Map Points Generated" : self.cs_params.gen_heat_map_data,
                         "Max BO Iters" : self.cs_params.bo_iter_tot,
                         "Number of Workflow Restarts" : self.cs_params.bo_run_tot,
                         "Seed" : self.cs_params.seed,
                         "EI Tolerance" : self.cs_params.ei_tol,
                         "Obj Improvement Tolerance" : self.cs_params.obj_tol,
                         "Theta Generation Enum Value": self.gen_meth_theta.value}
                
        for i in range(self.cs_params.bo_run_tot):
            bo_results = self.__run_bo_workflow()
            #Update the seed in configuration
            configuration["Seed"] = self.cs_params.seed
            #Add this updated copy of configuration with the new seed to the bo_results
            bo_results.configuration = configuration.copy()           
            #Add simulator class
            bo_results.simulator_class = simulator_class
            #On the 1st iteration, create heat map data if we are actually generating the data           
            if i == 0:
                if self.cs_params.gen_heat_map_data == True:
                    #Generate heat map data for each combination of parameter values stored in a dictionary
                    heat_map_data_dict = self.create_heat_map_param_data()
                    # Save these heat map values in the bo_results object 
                    # Only store in first list entry to avoid repeated data which stays the same for each iteration.
                    bo_results.heat_map_data_dict = heat_map_data_dict
            restart_bo_results.append(bo_results)
            #Add 2 to the seed for each restart (1 for the sim/exp data seed and 1 for validation data seed) to get completely new seeds
            self.cs_params.seed += 2
                   
        #Save data automatically if save_data is true
        if self.cs_params.save_data == True:
            self.save_data(restart_bo_results)

        return restart_bo_results
    
    def save_data(self, restart_bo_results):
        """
        Defines where to save data to and saves data accordingly
        
        Parameters
        ----------
        restart_bo_results: list of class instances of BO_results, The results of all restarts of the BO workflow for reproduction
        """
        ##Define a path for the data. (Use the name of the case study and date)
        #Get Date only from DateTime String
        if self.cs_params.DateTime is not None:
            #Note, This one uses / in DateTime and not -
            split_date_parts = self.cs_params.DateTime.split("/")
            Run_Date = "/".join(split_date_parts[:-1])    
        else:
            Run_Date = "No_Date"
        
        path = Run_Date + "/" + "Data_Files/" + self.cs_params.cs_name

        ##Create directory if it doesn't already exist
        # Extract the directory and filename from the given path
        directory = os.path.split(path)[0]
        filename = "%s.%s" % (os.path.split(path)[1], "gz")
        if directory == '':
            directory = '.'

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # The final path to save to is
        savepath = os.path.join(directory, filename)
        
        #Open the file
        fileObj = gzip.open(savepath, 'wb', compresslevel  = 1)
        
        #Turn this class into a pickled object and save to the file
        pickled_results = pickle.dump(restart_bo_results, fileObj)

        # Close the file
        fileObj.close()
        