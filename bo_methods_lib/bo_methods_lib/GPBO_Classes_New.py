import numpy as np
import random
from numpy.random import default_rng
import warnings
np.warnings = warnings
import math
from scipy.stats import norm, multivariate_normal
from scipy import integrate
import scipy.optimize as optimize
import scipy.spatial.distance as distance
import os
import time
import Tasmanian
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
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
import gpflow
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
from sklearn.utils.validation import check_is_fitted


class Method_name_enum(Enum):
    """
    The base class for any GPBO Method names
    
    Notes: 
    -------
    1 = A1 (Conventional GPBO, no obj scaling)
    2 = B1 (Conventional GPBO, ln obj scaling)
    3 = A2 (Emulator GPBO, independende approx. EI)
    4 = B2 (Emulator GPBO, log independence approx. EI)
    5 = C2 (Emulator GPBO, sparse grid integrated EI)
    6 = D2 (Emulator GPBO, monte carlo integrated EI)
    7 = A3 (Emulator GPBO, E[SSE] acquisition function)
    
    """
    #Ensure that only values 1 to 5 are chosen
    if Enum in range(1, 8) == False:
        raise ValueError("There are only seven options for Enum: 1 to 7")
        
    A1 = 1
    B1 = 2
    A2 = 3
    B2 = 4
    C2 = 5
    D2 = 6
    A3 = 7
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
    2 = CS2 4 Param 2 State Point (x0) Muller Potential
    3 = CS3 4 Param 2 State Point (y0) Muller Potential
    4 = CS4 8 Param 2 State Point (x0y0) Muller Potential
    5 = CS5 12 Param 2 State Point (Ax0y0) Muller Potential
    6 = CS6 8 Param 2 State Point (Ax0) Muller Potential
    7 = CS7 8 Param 2 State Point (Ay0) Muller Potential
    8 = CS8 5 Param Polynomial with varying scale parameters
    9 = CS9 4 Param Isotherm
    10 = CS10 5 Param Polynomial with same scale parameters
    11 = CS11 2 Param BOD Example from Bates and Watts
    Examples from: https://www.statforbiology.com/nonlinearregression/usefulequations
    12 = CS12 3 Param Yield-Loss Model
    13 = CS13 4 Param Log Logistic Model
    14 = CS14 4 Param 2 State Point Log Logistic 2D Model
    15 = CS15 5 Param 1 State Point exp() hybrid with many local minima
    16 = CS16 4 Param 2 State Point sin/cos model with many local minima
    17 = CS17 4 Param 1 State Point exp/cos model with many local minima
    """
    #Check that values are only 1 to 2
    if Enum in range(1, 18) == False:
        raise ValueError("There are 16 options for Enum: 1 to 17")
        
    CS1 = 1
    CS2_x0 = 2
    CS3_y0 = 3
    CS4_x0y0 = 4
    CS5_Ax0y0 = 5
    CS6_Ax0 = 6
    CS7_Ay0 = 7
    CS8 = 8
    CS9 = 9
    CS10 = 10
    CS11 = 11
    CS12 = 12
    CS13 = 13
    CS14 = 14
    CS15 = 15
    CS16 = 16
    CS17 = 17
    
    
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
    get_sparse_mc()
    """
    # Class variables and attributes
    
    def __init__(self, method_name):
        """
        Parameters
        ----------
        method_name: Method_name_enum Class instance, The name associated with the method being tested. Enum type
        """
        assert isinstance(method_name, Method_name_enum), "method_name must be an instance of Method_name_enum"
        # Constructor method
        self.method_name = method_name
        self.emulator = self.get_emulator()
        self.obj = self.get_obj()
        self.report_name = self.get_name_long()
        self.sparse_grid, self.mc = self.get_sparse_mc()
        
    def get_name_long(self):
        """
        Gets the shorthand name of the method that appears in the manuscript
        """

        if self.method_name.name == "A1":
            report_name = "Conventional"
        elif self.method_name.name == "B1":
            report_name = "Log Conventional"
        elif self.method_name.name == "A2":
            report_name = "Independence"
        elif self.method_name.name == "B2":
            report_name = "Log Independence"
        elif self.method_name.name == "C2":
            report_name = "Sparse Grid"
        elif self.method_name.name == "D2":
            report_name = "Monte Carlo"
        elif self.method_name.name == "A3":
            report_name = "E[SSE]"
        return report_name
    
    def get_emulator(self):
        """
        Function to get emulator status based on method name
        
        Returns:
        --------
        emulator: bool, Status of whether the GP emulates the function directly
        """
        #Objective function uses emulator GP if class 2
        if not "1" in self.method_name.name:
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
        if self.method_name.name == "1B":
            obj = Obj_enum(2)
        else:
            obj = Obj_enum(1)
        return obj
    
    def get_sparse_mc(self):
        """
        Function to get sparse grid status based on method name
        
        Returns:
        --------
        sparse_grid: bool, Determines whether a sparse grid is used to evaluate the EI integral
        mc: bool, Determines whether an mc is used to evaluate the EI integral
        """
        #Sparse grid and MC default false
        sparse_grid = False
        mc = False
        
        #Check Emulator status
        if self.emulator == True:
            #Method 2C is Sparse Grid
            if "C" in self.method_name.name:
                sparse_grid = True
            #Method 2D is Monte Carlo
            elif "D" in self.method_name.name:
                mc = True
        
        return sparse_grid, mc

class CaseStudyParameters:
    """
    The base class for any GPBO Case Study and Evaluation
    Parameters
    
    Methods
    --------------
    __init__
    
    """
    # Class variables and attributes
    
    def __init__(self, cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, set_seed, obj_tol, acq_tol):
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
        bo_iter_tot: int, maximum number of BO iterations per restart
        bo_run_tot: int, total number of BO algorithm restarts
        save_data: bool, Determines whether ei data for argmax(ei) theta will be saved
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        set_seed: int or None, Determines seed for randomizations. None if seed is random
        obj_tol: float, obj at which to terminate algorithm after int(bo_iter_tot*0.3) iters
        acq_tol: float, acquisition function value at which to terminate algorithm
        """
        #Assert statements
        #Check for strings
        # if not isinstance(cs_name, (str, CS_name_enum)) == True:
        #     warnings.warn("cs_name will be converted to string if it is not an instance of CS_name_enum")
        #Check for enum 
        assert isinstance(kernel, (Enum)) == True, "kernel must be type Enum" #Will figure this one out later
        #Check for float/int
        assert all(isinstance(var, (float,int)) for var in [sep_fact,ep0]) == True, "sep_fact and ep0 must be float or int"
        #Check for bool
        assert all(isinstance(var, (bool)) for var in [normalize, gen_heat_map_data, save_data]) == True, "normalize, gen_heat_map_data, save_fig, and save_data must be bool"
        #Check for int
        assert all(isinstance(var, (int)) for var in [bo_iter_tot, bo_run_tot, set_seed, retrain_GP, reoptimize_obj]) == True, "bo_iter_tot, bo_run_tot, seed, retrain_GP, and reoptimize_obj must be int"
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
        assert all(var > 0 for var in [bo_iter_tot, bo_run_tot, set_seed]) == True, "bo_iter_tot, bo_run_tot, and seed must be > 0"        
        #Check for >=0
        assert all(var >= 0 for var in [retrain_GP, reoptimize_obj]) == True, "retrain_GP and reoptimize_obj must be >= 0"
        #Check for str or None
        assert isinstance(DateTime, (str)) == True or DateTime == None, "DateTime must be str or None"
        assert isinstance(acq_tol, (float,int)) and acq_tol >= 0, "acq_tol must be a positive float or integer"
        assert isinstance(obj_tol, (float,int)) and obj_tol >= 0, "obj_tol must be a positive float or integer"
        
        # Constructor method
        #Ensure name is a string
        if isinstance(cs_name, Enum) == True:
            self.cs_name = cs_name.name
        else:
            self.cs_name = str(cs_name)
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
        self.seed = set_seed
        #Set seed
        if  self.seed != None:
            assert isinstance(self.seed, int) == True, "Seed number must be an integer or None"
            np.random.seed(self.seed)
        self.acq_tol = acq_tol
        self.obj_tol = obj_tol

class Simulator:
    """
    The base class for differet simulators. Defines a simulation
    
    Methods
    --------------
    __init__
    __set_true_params()
    __grid_sampling(num_points, bounds
    __lhs_sampling(num_points, bounds, seed)
    __create_param_data(num_points, bounds, gen_meth, seed)
    __vector_to_1D_array(array)
    gen_y_data(data, noise_mean, noise_std)
    gen_exp_data(num_x_data, gen_meth_x)
    gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, gen_val_data)
    gen_theta_vals(num_theta_data)
    sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, gen_val_data)
    """
    def __init__(self, indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u, noise_mean, noise_std, set_seed, calc_y_fxn, calc_y_fxn_args):
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
        noise_std: float, int: The standard deviation of the noise. If None, 5% of mean of Y-exp will be used
        set_seed: int or None, Determines seed for randomizations. None if seed is random
        calc_y_fxn: function, The function to calculate ysim data with
        calc_y_fxn_args: dict, dictionary of arguments other than parameters and x to pass to calc_y_fxn
        """
        #Check for float/int
        assert isinstance(noise_mean,(float,int)), "noise_mean must be int or float"
        assert isinstance(noise_std,(float,int)) or noise_std is None, "noise_std must be int, float, or None"
        assert isinstance(set_seed, int) or set_seed is None, "Seed must be int or None"
        #Check for list or ndarray
        list_vars = [indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, bounds_x_u]
        assert all(isinstance(var,(list,np.ndarray)) for var in list_vars) == True, "indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must be list or np.ndarray"
        #Check for list lengths > 0
        assert all(len(var) > 0 for var in list_vars) == True, "indeces_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must have length > 0"
        #Check that bound_x and bounds_theta have same lengths
        assert len(bounds_theta_l) == len(bounds_theta_u) and len(bounds_x_l) == len(bounds_x_u), "bounds lists for x and theta must be same length"
        #Check indeces to consider in theta_ref
        assert all(0 <= idx <= len(theta_ref)-1 for idx in indeces_to_consider)==True, "indeces to consider must be in range of theta_ref"
        assert isinstance(calc_y_fxn_args, dict) or calc_y_fxn_args is None, "calc_y_fxn_args must be dict or None"
        
        #How to write assert statements for calc_y_fxn?
        
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
        self.calc_y_fxn_args = calc_y_fxn_args
        self.seed = set_seed
    
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
    
    def __lhs_sampling(self, num_points, bounds, set_seed):
        """
        Design LHS Samples

        Parameters
        ----------
        num_points: int, number of points in LHS, should be greater than # of dimensions
        bounds: ndarray, array containing upper and lower bounds of elements in LHS sample. Defaults of 0 and 1
        set_seed: int, seed of random generation

        Returns
        -------
        lhs_data: ndarray, array of LHS sampling points with length (num_points) 
        """
        #Define number of dimensions
        dimensions = bounds.shape[1]
        #Define sampler
        sampler = qmc.LatinHypercube(d=dimensions, seed = set_seed)
        lhs_data = sampler.random(n=num_points)

        #Generate LHS data given bounds
        lhs_data = qmc.scale(lhs_data, bounds[0], bounds[1]) #Using this because I like that bounds can be different shapes

        return lhs_data
    
    def __create_param_data(self, num_points, bounds, gen_meth, set_seed):
        """
        Generates data based off of bounds, and an LHS generation number
        
        Parameters
        ----------
        num_points: int, number of data to generate
        bounds: array, array of parameter bounds
        gen_meth: class (Gen_meth_enum), ("LHS", "Meshgrid"). Determines whether data will be generated with an LHS or meshgrid
        set_seed: int, seed of random generation
        
        Returns:
        --------
        data: ndarray, an array of data
        
        Notes: Meshgrid generated data will output num_points in each dimension, LHS generates num_points of data
        """        
        #Set dimensions 
        dimensions = bounds.shape[1] #Want to do it this way to make it general for either x or theta parameters
        
        #Decide on a method to use based on gen_meth_value. LHS or Grid
        if gen_meth.value == 2:
            data = self.__grid_sampling(num_points, bounds) 
            
        elif gen_meth.value == 1:
            #Generate LHS sample
            data =  self.__lhs_sampling(num_points, bounds, set_seed)
        
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
        noise_mean: float, int: The mean of the noise
        noise_std: float, int, None: The standard deviation of the noise

        Returns
        -------
        y_data: ndarray, The simulated y training data
        """   
        #Set seed
        if self.seed is not None:
            np.random.seed(self.seed)
                
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
            #Create y data coefficients
            y_data.append(self.calc_y_fxn(model_coefficients, data.x_vals[i], self.calc_y_fxn_args))

        #Convert list to array and flatten array
        y_data = np.array(y_data).flatten()

        #Creates noise values with a certain stdev and mean from a normal distribution
        if noise_std is None:
            #If noise is none, set the noise as 5% of the mean value
            noise_std = np.abs(np.mean(y_data))*0.05
        else:
            noise_std = noise_std

        noise = np.random.normal(size=len(y_data), loc = noise_mean, scale = noise_std)
        
        #Add noise to data
        y_data = y_data + noise
        
        return y_data
       
    def gen_exp_data(self, num_x_data, gen_meth_x, set_seed = None):
        """
        Generates experimental data in an instance of the Data class
        
        Parameters
        ----------
        num_x_data: int, number of experiments
        gen_meth_x: bool: Whether to generate X data with LHS or grid method
        set_seed: int or None, Seed with which t0 generate experimental data. None sets the seed to the class seed

        Returns:
        --------
        exp_data: instance of a class filled in with experimental x and y data along with parameter bounds
        """
        assert isinstance(set_seed, int) or set_seed is None, "set_seed must be int or None"
        #Set generation data seed
        if set_seed is not None:
            new_seed = set_seed
        else:
            new_seed = self.seed
        #check that num_data > 0 
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError('num_x_data must be a positive integer')
            
        #Create x vals based on bounds and num_x_data
        x_vals = self.__vector_to_1D_array(self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, new_seed))
        #Reshape theta_true to correct dimensions and stack it once for each xexp value
        theta_true = self.theta_true.reshape(1,-1)
        theta_true_repeated = np.vstack([theta_true]*len(x_vals))
        #Create exp_data class and add values
        exp_data = Data(theta_true_repeated, x_vals, None, None, None, None, None, None, self.bounds_theta_reg, self.bounds_x, None, new_seed)
        #Generate y data for exp_data calss instance
        exp_data.y_vals = self.gen_y_data(exp_data, self.noise_mean, self.noise_std)
        
        return exp_data
    
    def gen_sim_data(self, num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, set_seed = None, gen_val_data = False):
        """
        Generates experimental data in an instance of the Data class
        
        Parameters
        ----------
        num_theta_data: int, number of theta values
        num_x_data: int, number of experiments
        gen_meth_theta: bool: Whether to generate theta data with LHS or grid method
        gen_meth_x: bool: Whether to generate X data with LHS or grid method
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None, seed to generate initial training data with. If None, seed will be the seed of the class
        gen_val_data: bool, Whether validation data (no y vals) or simulation data (has y vals) will be generated. Default False
        
        Returns:
        --------
        sim_data: instance of a class filled in with experimental x and y data along with parameter bounds
        """
        assert isinstance(sep_fact, (float,int)), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        assert isinstance(set_seed, int) or set_seed is None, "set_seed must be int or None!"

        if set_seed is not None:
            sim_seed = set_seed
        else:
            sim_seed = self.seed
        
        if isinstance(gen_val_data, bool) == False:
            raise ValueError('gen_val_data must be bool')
            
        #Chck that num_data > 0
        if num_theta_data <= 0 or isinstance(num_theta_data, int) == False:
            raise ValueError('num_theta_data must be a positive integer')
            
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError('num_x_data must be a positive integer')
        
        #Set bounds on theta which we are regressing given bounds_theta and indeces to consider
        #X data we always want the same between simulation and validation data
        x_data = self.__vector_to_1D_array(self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, sim_seed))
            
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
        sim_data = Data(None, None, None, None, None, None, None, None, self.bounds_theta_reg, self.bounds_x, sep_fact, sim_seed)
       
        #For validation theta, change the seed by 1 to ensure validation and sim data are never the same
        if gen_val_data == False:
            val_seed = sim_seed
        else:
            val_seed = int(sim_seed + 1)
            
        #Generate simulation data theta_vals and create instance of data class   
        sim_theta_vals = self.__vector_to_1D_array(self.__create_param_data(num_theta_data, self.bounds_theta_reg, gen_meth_theta, val_seed))
        
        #Add repeated theta_vals and x_data to sim_data
        sim_data.theta_vals = np.repeat(sim_theta_vals, repeat_theta , axis = 0)
        sim_data.x_vals = np.vstack([x_data]*repeat_x)
        
        #Add y_vals for sim_data only
        if gen_val_data == False:
            #Training data should be generated with the same mean and variance as the experimental data
            sim_data.y_vals = self.gen_y_data(sim_data, self.noise_mean, self.noise_std)
        
        return sim_data
    
    def gen_theta_vals(self, num_theta_data):
        """
        Generates theta points for an instance of the Data class
        
        Parameters
        ----------
        num_theta_data: int, number of theta values
        
        Returns:
        --------
        theta_vals: ndarray, theta values generated
        """
        assert isinstance(num_theta_data, int) and num_theta_data > 0, "num_theta_data must be int > 0"
        gen_meth_theta = Gen_meth_enum(1)
            
        #Chck that num_data > 0
        if num_theta_data <= 0 or isinstance(num_theta_data, int) == False:
            raise ValueError('num_theta_data must be a positive integer')
            
        #Warn user if >5000 pts generated
        if num_theta_data > 5000:
            warnings.warn("More than 5000 points will be generated!")
            
        #Generate simulation data theta_vals and create instance of data class   
        theta_vals = self.__vector_to_1D_array(self.__create_param_data(num_theta_data, self.bounds_theta_reg, gen_meth_theta, self.seed))
        
        return theta_vals
   
    def sim_data_to_sse_sim_data(self, method, sim_data, exp_data, sep_fact, gen_val_data = False):
        """
        Creates simulation data based on x, theta_vals, the GPBO method, and the case study

        Parameters
        ----------
        method: GPBO_Methods, fully defined methods class which determines which method will be used
        sim_data: Data, Class containing at least the theta_vals, x_vals, and y_vals for simulation
        exp_data: Data, Class containing at least the x_data and y_data for the experimental data
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        gen_val_data: bool, Whether validation data (no y vals) or simulation data (has y vals) will be generated. Default False

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
        sim_sse_data = Data(unique_theta_vals, exp_data.x_vals, None, None, None, sim_data.sse, sim_data.sse_var, 
                            sim_data.acq, self.bounds_theta, self.bounds_x, sep_fact, self.seed)
        
        if gen_val_data == False:
            #Define all y_sims
            y_sim = sim_data.y_vals

            #Reshape y_sim into n_theta rows x n_x columns
            indices = np.arange(0, len_theta, len_x)
            n_blocks = len(indices)
            # Slice y_sim into blocks of size len_x and calculate squared errors for each block
            y_sim_resh = y_sim.reshape(n_blocks, len_x)
            block_errors = (y_sim_resh - exp_data.y_vals[np.newaxis,:])**2
            # Sum squared errors for each block
            sum_error_sq = np.sum(block_errors, axis=1)
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
    __vector_to_1D_array(array)
    get_unique_theta()
    get_unique_x()
    get_num_theta()
    get_dim_theta()
    get_num_x_vals()
    get_dim_x_vals()
    train_test_idx_split() 
    """
    # Class variables and attributes
    
    def __init__(self, theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, sse_var, acq, bounds_theta, bounds_x, sep_fact, set_seed):
        """
        Parameters
        ----------
        theta_vals: ndarray, The arrays of theta_values
        x_vals: ndarray, experimental state points (x data)
        y_vals: ndarray, experimental y data
        gp_mean: ndarray, GP mean prediction values associated with theta_vals and x_vals
        gp_var: ndarray, GP variance prediction values associated with theta_vals and x_vals
        sse: ndarray, GP based sum of squared error values associated with theta_vals and x_vals
        sse_var: ndarray, GP based variance of sum of squared error values associated with theta_vals and x_vals
        acq: ndarray, acquisition function values associated with theta_vals and x_vals
        bounds_theta: ndarray, bounds of theta
        bounds_x: ndarray, bounds of x
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None, Determines seed for randomizations. None if seed is random
        """
        list_vars = [theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, acq]
        assert all(isinstance(var, np.ndarray) or var is None for var in list_vars), "theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, and ei must be np.ndarray, or None"
        assert isinstance(set_seed, int) or set_seed is None, "Seed must be int or None"
        assert isinstance(sep_fact, (float,int)) or sep_fact is None, "Separation factor must be float or int > 0 or None (exp_data)"
        if sep_fact is not None:
            assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        # Constructor method
        self.theta_vals = theta_vals
        self.x_vals = x_vals
        self.y_vals = y_vals  
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.gp_covar = None #This is calculated later
        self.sse = sse
        self.sse_var = sse_var
        self.sse_covar = None #This is calculated later
        self.acq = acq
        self.bounds_theta = bounds_theta
        self.bounds_x = bounds_x
        self.sep_fact = sep_fact
        self.seed = set_seed
    
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
            np.random.seed(self.seed)
            
        #Shuffle all_idx data in such a way that theta values will be randomized
        np.random.shuffle(all_idx)
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
    __set_lenscl_guess(self)
    __set_white_kern(self)
    __set_base_kernel(self, c_guess, set_c_trainable, ls_guess, set_ls_trainable)
    __set_outputscl(self)
    set_gp_model_data(self)
    __set_gp_kernel(self, edu_guess = True)
    __init_hyper_parameters(self, count_fix)
    __fit_GP(self, count_fix)
    train_gp(self)
    __eval_gp_mean_var(data)
    eval_gp_mean_var_misc(misc_data, featurized_misc_data, covar=False)
    eval_gp_mean_var_test(covar=False)
    eval_gp_mean_var_val(covar=False)
    eval_gp_mean_var_cand(covar=False)
    """
    # Class variables and attributes
    
    def __init__(self, gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, set_seed, normalize, __feature_train_data, __feature_test_data, __feature_val_data, __feature_cand_data):
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
        set_seed: int or None, random seed
        normalize: bool, determines whether data is normalized w/ Yeo-Johnson transformation + zero-mean, unit-variance normalization
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
        self.seed = set_seed
        self.normalize = normalize
        #If normalize, create the scalers
        if normalize == True:
            # self.scalerX = PowerTransformer(method = 'yeo-johnson', standardize = True)
            self.scalerX = RobustScaler(unit_variance = True)
            self.scalerY = RobustScaler(unit_variance = True)
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
    
    def bounded_parameter(self, low, high, initial_value):
        sigmoid = tfb.Sigmoid(low=tf.cast(low, dtype=tf.float64), high=tf.cast(high, dtype=tf.float64))
        return gpflow.Parameter(initial_value, transform=sigmoid, dtype=tf.float64)
    
    # def __set_lenscl_guess(self):
    #     """
    #     Gets an upper and lower bound for the lengthscales

    #     Returns:
    #     --------
    #     lenscl_guess: ndarray, array of lengthscale guesses
    #     """

    #     #Set lenscl bounds using the original training data to ensure distance
    #     #Between min and max lengthscales does not collapse as iterations progress
    #     assert isinstance(self.train_data_init, np.ndarray), "self.train_data_init must be an array"
    #     if self.normalize:
    #         org_scalerX = RobustScaler(unit_variance = True)
    #         points = org_scalerX.fit_transform(self.train_data_init)
    #     else:
    #         points = self.train_data_init

    #     # Compute pairwise differences for each column
    #     pairwise_diffs = np.abs(points[:, :, None] - points[:, :, None].transpose(0, 2, 1))
    #     # Compute Euclidean distances
    #     euclidean_distances = np.sqrt(np.sum(pairwise_diffs ** 2, axis=1))
    #     # Set diagonal elements (distance between the same point) to infinity
    #     np.fill_diagonal(euclidean_distances, np.inf)
    #     euclidean_distances = np.ma.masked_invalid(euclidean_distances)
    #     # Find the smallest/largest distance for each column
    #     min_distance = np.min(euclidean_distances, axis=0)
    #     max_distance = np.max(euclidean_distances, axis=0)

    #     lenscl_guess = np.random.uniform(min_distance, max_distance, size=len(max_distance))
    #     return lenscl_guess

    def __set_lenscl_guess(self, lb, ub):
        """
        Gets an upper and lower bound for the lengthscales

        Returns:
        --------
        lenscl_guess: ndarray, array of lengthscale guesses
        """

        #Set lenscl bounds using the original training data to ensure distance
        #Between min and max lengthscales does not collapse as iterations progress
        assert isinstance(self.train_data_init, np.ndarray), "self.train_data_init must be an array"
        if self.normalize:
            org_scalerX = RobustScaler(unit_variance = True)
            points = org_scalerX.fit_transform(self.train_data_init)
        else:
            points = self.train_data_init

        # Compute pairwise differences for each column
        pairwise_diffs = np.abs(points[:, :, None] - points[:, :, None].transpose(0, 2, 1))
        # Compute Euclidean distances
        euclidean_distances = np.sqrt(np.sum(pairwise_diffs ** 2, axis=1))
        # Set diagonal elements (distance between the same point) to infinity
        np.fill_diagonal(euclidean_distances, np.inf)
        euclidean_distances = np.ma.masked_invalid(euclidean_distances)
        # Find the smallest/largest distance for each column and ensure it is within the bounds
        min_distance = np.min(euclidean_distances, axis=0)
        max_distance = np.max(euclidean_distances, axis=0)

        lb_array = np.ones(len(min_distance))*lb
        ub_array = np.ones(len(max_distance))*ub
        lower = np.maximum(min_distance, lb_array)
        upper = np.minimum(max_distance, ub_array)

        lenscl_guess = np.random.uniform(lower, upper, size=len(max_distance))
        return lenscl_guess
        
    
    # def __set_white_kern(self):
    #     """
    #     Sets the white kernel value guess or allows gp to tune the noise parameter

    #     Returns:
    #     --------    
    #     noise_kern: kernel, the noise kernel of the model
    #     """
    #     #Set the noise guess or allow gp to tune the noise parameter
    #     if self.normalize:
    #         self.scalerY.fit(self.train_data.y_vals.reshape(-1,1))
    #         sclr = np.float64(self.scalerY.scale_)
    #     else:
    #         sclr = 1.0

    #     if self.noise_std is not None:
    #         #If we know the noise, use it
    #         noise_guess = float((self.noise_std/sclr)**2)
            
    #     else:
    #         #Otherwise, set the guess as 5% the taining data mean
    #         data_mean = np.abs(np.mean(self.gp_sim_data.y_vals))
    #         noise_guess = np.float64(data_mean*0.05/sclr)**2
        
    #     noise_guess_f = np.maximum(1.01e-6, noise_guess)

    #     noise_kern = gpflow.kernels.White(variance = noise_guess_f)

    #     if self.noise_std is not None:
    #         gpflow.set_trainable(noise_kern.variance, False)

    #     return noise_kern

    def __set_white_kern(self, lb, ub):
        """
        Sets the white kernel value guess or allows gp to tune the noise parameter

        Returns:
        --------    
        noise_kern: kernel, the noise kernel of the model
        """
        #Set the noise guess or allow gp to tune the noise parameter
        if self.normalize:
            self.scalerY.fit(self.train_data.y_vals.reshape(-1,1))
            sclr = np.float64(self.scalerY.scale_)
        else:
            sclr = 1.0

        if self.noise_std is not None:
            #If we know the noise, use it
            noise_guess = float((self.noise_std/sclr)**2)
            
        else:
            #Otherwise, set the guess as 5% the taining data mean
            data_mean = np.abs(np.mean(self.gp_sim_data.y_vals))
            noise_guess = np.float64(data_mean*0.05/sclr)**2
        
        if not lb < noise_guess < ub:
            noise_guess = 1.0

        return noise_guess

    # def __set_base_kernel(self, c_guess, set_c_trainable, ls_guess, set_ls_trainable):
    #     """
    #     Sets the base kernel of the model
        
    #     Returns
    #     ----------
    #     kernel: The base kernel of the model
        
    #     """ 
    #     #Set the type of kernel
    #     if self.kernel.value == 3: #RBF
    #         kernel_base = gpflow.kernels.RBF(variance=c_guess, lengthscales = ls_guess)
    #     elif self.kernel.value == 2: #Matern 3/2
    #         kernel_base = gpflow.kernels.Matern32(variance=c_guess, lengthscales = ls_guess) 
    #     else: #Matern 5/2
    #         kernel_base = gpflow.kernels.Matern52(variance=c_guess, lengthscales = ls_guess) 

    #     #Set scale parameter on base kernel w/ a Half Cauchy Prior w/ mean 1
    #     kernel_base.variance.prior = tfp.distributions.HalfCauchy(np.float64(1.0), np.float64(5.0))

    #     #Set scale values
    #     if not set_c_trainable:
    #         gpflow.set_trainable(kernel_base.variance, False)
    #     if not set_ls_trainable:
    #         gpflow.set_trainable(kernel_base.lengthscales, False)

    #     return kernel_base
    
    # def __set_lenscl(self):
    #     """
    #     Set the original lengthscale of the model and determines whether lengthscale is tunable
        
    #     Returns
    #     -------
    #     lenscl_guess: ndarray, array of lengthscale guesses
    #     set_lenscl_trainable: bool, whether the lengthscale is tunable or not

    #     Notes:
    #     ------
    #     Need to have training data before using this function
    #     """
    #     set_lenscl_trainable = True
    #     if isinstance(self.lenscl, np.ndarray):
    #         assert len(self.lenscl) >= self.get_dim_gp_data(), "Length of self.lenscl must be at least self.get_gim_gp_data()!"
    #         #Cut the lengthscale to correct length if too long, by cutting the ends
    #         if len(self.lenscl) > self.get_dim_gp_data():
    #             self.lenscl =  self.lenscl[:self.get_dim_gp_data()]
        
    #         #Anisotropic but different and set
    #         lenscl_guess = self.lenscl
    #         set_lenscl_trainable = False
            
    #     #If setting lengthscale, ensure lengthscale values are fixed and that there is 1 lengthscale/dim,\
    #     elif isinstance(self.lenscl, (float, int)):            
    #         #Anisotropic but the same
    #         lenscl_guess = self.lenscl*np.ones(self.get_dim_gp_data())
    #         set_lenscl_trainable = False
            
    #     #Otherwise initialize them at 1 (lenscl is trained) 
    #     else:
    #         #Anisotropic but initialized to 1
    #         lenscl_guess = self.__set_lenscl_guess()
        
    #     return lenscl_guess, set_lenscl_trainable
    
    # def __set_outputscl(self):
    #     """
    #     Set the initial outputscale of the model and determines whether it is tunable
        
    #     Returns
    #     -------
    #     tau_guess: float, initial outputscale guess
    #     set_c_trainable: bool, whether the outputscale is tunable or not

    #     Notes:
    #     ------
    #     Need to have training data before using this function
    #     """
    #     set_c_trainable = True
        
    #     #Set outputscl kernel to be optimized based on guess if desired
    #     if self.outputscl == None:
    #         train_y = self.train_data.y_vals.reshape(-1,1)
    #         if self.normalize: 
    #             scl_y = self.scalerY.fit_transform(train_y)
    #         else:
    #             scl_y = train_y

    #         c_guess= sum(scl_y.flatten()**2)/len(scl_y)
    #         tau = c_guess
    #     elif isinstance(self.outputscl, (float, int, np.float64)):
    #         assert self.outputscl > 0, "outputscl must be positive int or float"
    #         tau = self.outputscl
    #         set_c_trainable = False
    #     else:
    #         tau = 1.0
    #         set_c_trainable = False
            
    #     tau_guess_min = 1.01e-6
    #     if tau_guess_min < tau:
    #         tau_guess = tau  
    #     else:
    #         tau_guess = 1.0

    #     return tau_guess, set_c_trainable

    def __set_outputscl(self, lb, ub):
        """
        Set the initial outputscale of the model and determines whether it is tunable
        
        Returns
        -------
        tau_guess: float, initial outputscale guess
        set_c_trainable: bool, whether the outputscale is tunable or not

        Notes:
        ------
        Need to have training data before using this function
        """

        #Set outputscl kernel to be optimized based on guess if desired
        if self.outputscl == None:
            train_y = self.train_data.y_vals.reshape(-1,1)
            if self.normalize: 
                scl_y = self.scalerY.fit_transform(train_y)
            else:
                scl_y = train_y

            c_guess= sum(scl_y.flatten()**2)/len(scl_y)
            tau = c_guess

        elif isinstance(self.outputscl, (float, int, np.float64)):
            assert self.outputscl > 0, "outputscl must be positive int or float"
            tau = self.outputscl
        else:
            tau = 1.0
            
        if not lb < tau < ub:
            tau = 1.0

        return tau

    
    def set_gp_model_data(self):
        """
        Sets the training data for the GP model

        Returns
        -------
        data: tuple or 2 np.ndarrays, the feature and output training data for the GP model
        """
        assert self.feature_train_data is not None, "self.feature_train_data must be defined"
        assert self.train_data.y_vals is not None, "self.train_data.y_vals must be defined"
        #Set new model data
        #Preprocess Training data
        if self.normalize == True:
            #Update scaler to be the fitted scaler. This scaler will change as the training data is updated
            #Scale training data if necessary
            ft_td_scl = self.scalerX.fit_transform(self.feature_train_data)
            y_td_scl = self.scalerY.fit_transform(self.train_data.y_vals.reshape(-1,1))
        else:
            ft_td_scl = self.feature_train_data
            y_td_scl = self.train_data.y_vals.reshape(-1,1)
        data = (ft_td_scl, y_td_scl)
        return data
    
    # def __set_gp_kernel(self, edu_guess = True):
    #     """
    #     Generates the full gp kernel by combining the noise and base kernels

    #     Parameters
    #     ----------
    #     edu_guess: bool, whether to use an educated guess for the hyperparameters or not
            
    #     Returns
    #     --------
    #     kernel: kernel, the full gp kernel
    #     """  
    #     #Get tau val
    #     c_guess, set_c_trainable = self.__set_outputscl()
    #     #Get lescale values
    #     ls_guess, set_ls_trainable = self.__set_lenscl()

    #     if not edu_guess:
    #         if set_c_trainable:
    #             c_guess = np.exp(np.random.uniform(0. , 5.) )
    #             # c_guess = scipy.stats.halfcauchy.rvs(loc = 1.0, scale = 5.0)
    #         if set_ls_trainable:
    #             ls_guess = np.exp(np.random.uniform(0 , 3, size = len(ls_guess)))

    #     #Set base kernel
    #     kernel_base = self.__set_base_kernel(c_guess, set_c_trainable, ls_guess, set_ls_trainable)
    #     #Set Noise kern
    #     noise_kern = self.__set_white_kern()
    #     #Set whole kernel
    #     kernel = kernel_base + noise_kern
        
    #     return kernel
    
    # def set_gp_model(self, kernel=None):
    #     """
    #     Generates the GP model for the process in sklearn
            
    #     Returns
    #     --------
    #     gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor containing kernel, optimizer, etc.
    #     """  
    #     #Get Data
    #     data = self.set_gp_model_data() 

    #     if kernel == None:
    #         kernel_use = self.__set_gp_kernel(edu_guess = True)
    #     else:
    #         kernel_use = kernel

    #     #Get likelihood noise
    #     lik_noise = float(kernel_use.kernels[1].variance.numpy())

    #     gp_model = gpflow.models.GPR(data, kernel=kernel, noise_variance = lik_noise)
    #     # gpflow.utilities.print_summary(gp_model)
    #     return gp_model

    # Hyper-parameters initialization
    def __init_hyper_parameters(self, retrain_count):
        """
        Initializes the kernel and its hyperparameters for the GP model

        Parameters
        ----------
        count_fix: int, the number of times the GP has been retrained

        Returns
        --------
        kernel: kernel, the full gp kernel to optimize
        """
        tf.compat.v1.get_default_graph()
        tf.compat.v1.set_random_seed(retrain_count)
        tf.random.set_seed(retrain_count)
        np.random.seed(retrain_count)
        gpflow.config.set_default_float(np.float64)
        
        lenscl_bnds = [0.00001, 1000.0]
        var_bnds = [0.00001, 100.0]
        white_var_bnds = [0.00001, 10.0]

        #Get X and Y Data
        data = self.set_gp_model_data() 
        x_train, y_train = data

        #On the 1st iteration, use initial guesses initialized to 1
        if retrain_count == 0:
            lengthscale_1 = self.bounded_parameter(lenscl_bnds[0], lenscl_bnds[1], 1.0)
            lenscls = np.ones(x_train.shape[1])*lengthscale_1
            tau = self.bounded_parameter(var_bnds[0], var_bnds[1], 1.0)
            white_var = self.bounded_parameter(white_var_bnds[0], white_var_bnds[1], 1.0)
        #On second iteration, base guesses on initial data values
        elif retrain_count == 1:
            initial_lenscls = np.array(self.__set_lenscl_guess(lenscl_bnds[0], lenscl_bnds[1]) , dtype='float64')
            lenscls = self.bounded_parameter(lenscl_bnds[0], lenscl_bnds[1], initial_lenscls)
            initial_tau = np.array(self.__set_outputscl(var_bnds[0], var_bnds[1]) , dtype='float64')
            tau = self.bounded_parameter(var_bnds[0], var_bnds[1], initial_tau)
            initial_white_var = np.array(self.__set_white_kern(white_var_bnds[0], white_var_bnds[1]), dtype='float64')
            white_var = self.bounded_parameter(white_var_bnds[0], white_var_bnds[1], initial_white_var)
        #On all other iterations, use random guesses
        else:
            initial_lenscls = np.array(np.random.uniform(0.1, 100.0, x_train.shape[1]), dtype='float64')
            lenscls = self.bounded_parameter(lenscl_bnds[0], lenscl_bnds[1], initial_lenscls)
            tau = self.bounded_parameter(var_bnds[0], var_bnds[1], np.array(np.random.lognormal(0.0, 1.0), dtype='float64'))
            white_var = self.bounded_parameter(white_var_bnds[0], white_var_bnds[1], np.array(np.random.uniform(0.05, 10), dtype='float64'))

        # gpflow.utilities.print_summary(kernel)
        
        return lenscls, tau, white_var

    def set_gp_model(self, retrain_count):
        """
        Generates the GP model for the process in sklearn
            
        Returns
        --------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor containing kernel, optimizer, etc.
        """  
        data = self.set_gp_model_data()
        lenscls, tau, white_var = self.__init_hyper_parameters(retrain_count)

        if self.kernel.value == 3:
            gpKernel=gpflow.kernels.SquaredExponential(variance=tau, lengthscales=lenscls)
        elif self.kernel.value == 2:
            gpKernel=gpflow.kernels.Matern32(variance=tau, lengthscales=lenscls)
        else:
            gpKernel=gpflow.kernels.Matern52(variance=tau, lengthscales=lenscls)
        # Add White kernel
        gpKernel=gpKernel+gpflow.kernels.White(variance = white_var)
                
        # Build GP model    
        gp_model=gpflow.models.GPR(data,kernel = gpKernel, noise_variance=10**-5)
        # model_pretrain = deepcopy(model)
        # # print(gpflow.utilities.print_summary(model))
        # condition_number = np.linalg.cond(model.kernel(X_Train))
        # Select whether the likelihood variance is trained
        gpflow.utilities.set_trainable(gp_model.likelihood.variance,False)
        if isinstance(self.lenscl, np.ndarray) or isinstance(self.lenscl, (float, int)):
            gpflow.utilities.set_trainable(gp_model.kernel.kernel[0].lengthscales,False)
        if self.outputscl is not None:
            gpflow.utilities.set_trainable(gp_model.kernel.kernel[0].variance,False)

        return gp_model
    
    # # Hyper-parameters initialization
    # def __init_hyper_parameters(self, count_fix):
    #     """
    #     Initializes the kernel and its hyperparameters for the GP model

    #     Parameters
    #     ----------
    #     count_fix: int, the number of times the GP has been retrained

    #     Returns
    #     --------
    #     kernel: kernel, the full gp kernel to optimize
    #     """
    #     tf.compat.v1.get_default_graph()
    #     tf.compat.v1.set_random_seed(count_fix)
    #     tf.random.set_seed(count_fix)
    #     gpflow.config.set_default_float(np.float64)
    #     #On the 1st iteration, use initial guesses based on training data values
    #     edu_guess = True if count_fix == 0 else False
    #     kernel = self.__set_gp_kernel(edu_guess)

    #     # gpflow.utilities.print_summary(kernel)
        
    #     return kernel

    # def __fit_GP(self, count_fix):
    #     """
    #     Fit a GP and fix Cholesky decomposition failure and optimization failure by random initialization

    #     Parameters
    #     ----------
    #     count_fix: int, the number of times the GP has been retrained

    #     Returns
    #     --------
    #     fit_successed: bool, whether the GP was fit successfully
    #     model: instance of gpflow.models.GPR, the trained GP model
    #     count_fix: int, the number of times the GP has been retrained
    #     """
    #     #Randomize Seed
    #     np.random.seed(count_fix+1)
    #     #Initialize fit_sucess as true
    #     fit_successed = True   
    #     #Get hyperparam guess list
    #     kernel = self.__init_hyper_parameters(count_fix)
    #     try:
    #         #Make model and optimizer and get results
    #         model = self.set_gp_model(kernel)
    #         o = gpflow.optimizers.Scipy()
    #         res = o.minimize(model.training_loss, variables=model.trainable_variables)
    #         # gpflow.utilities.print_summary(model)
    #         #If result isn't successful, remake and retrain model w/ different hyperparameters
    #         if not(res.success):
    #             if count_fix < self.retrain_GP:
    #                 count_fix += 1 
    #                 fit_successed,model,count_fix = self.__fit_GP(count_fix)
    #             else:
    #                 fit_successed = False
    #     #If an error is thrown becauuse of bad hyperparameters, reoptimize them
    #     except tf.errors.InvalidArgumentError as e:
    #         if count_fix < self.retrain_GP:
    #             count_fix += 1
    #             fit_successed,model,count_fix = self.__fit_GP(count_fix)
    #         else:
    #             fit_successed = False

    #     return fit_successed,model, count_fix
        
    # def train_gp(self):
    #     """
    #     Trains the GP given training data. Sets self.trained_hyperparams and self.fit_gp_model
        
    #     Notes:
    #     ------
    #     Sets the following parameters of self
    #     self.trained_hyperparams: list, the trained hyperparameters of the GP model
    #     self.fit_gp_model: instance of gpflow.models.GPR, the trained GP model
    #     self.posterior: instance of gpflow.mean_field.KFGaussian, the posterior of the GP model 
    #     """  
    #     assert isinstance(self.feature_train_data, np.ndarray), "self.feature_train_data must be np.ndarray"
    #     assert self.feature_train_data is not None, "Must have training data. Run set_train_test_data() to generate"

    #     # Train the model multiple times and keep track of the model with the lowest minimum training loss
    #     best_minimum_loss = float('inf')
    #     best_model = None

    #     #Initialize number of counters
    #     count_fix_tot = 0
    #     #While you still have retrains left
    #     while count_fix_tot <= self.retrain_GP:
    #         #Create and fit the model
    #         fit_successed, gp_model, count_fix = self.__fit_GP(count_fix_tot)
    #         #The new counter total is the number of counters used + 1
    #         count_fix_tot += count_fix + 1
    #         #If the fit succeeded 
    #         if fit_successed:
    #             # Compute the training loss of the model
    #             training_loss = gp_model.training_loss().numpy()
    #             # Check if this model has the best minimum training loss
    #             if training_loss < best_minimum_loss:
    #                 best_minimum_loss = training_loss
    #                 best_model = gp_model
    #         #or we have no good models
    #         elif count_fix_tot >= self.retrain_GP:
    #             if best_model is None:
    #                 best_model = gp_model

    #     # gpflow.utilities.print_summary(best_model)
        
    #     #Pull out kernel parameters after GP training
    #     outputscl_final = float(best_model.kernel.kernels[0].variance.numpy())
    #     lenscl_final = best_model.kernel.kernels[0].lengthscales.numpy()
    #     noise_final = float(best_model.kernel.kernels[1].variance.numpy())
        
    #     #Put hyperparameters in a list
    #     trained_hyperparams = [lenscl_final, noise_final, outputscl_final] 
        
    #     #Assign self parameters
    #     self.trained_hyperparams = trained_hyperparams
    #     self.fit_gp_model = best_model
    #     self.posterior = self.fit_gp_model.posterior()

    def train_gp(self):
        """
        Trains the GP given training data. Sets self.trained_hyperparams and self.fit_gp_model
        
        Notes:
        ------
        Sets the following parameters of self
        self.trained_hyperparams: list, the trained hyperparameters of the GP model
        self.fit_gp_model: instance of gpflow.models.GPR, the trained GP model
        self.posterior: instance of gpflow.mean_field.KFGaussian, the posterior of the GP model 
        """  
        assert isinstance(self.feature_train_data, np.ndarray), "self.feature_train_data must be np.ndarray"
        assert self.feature_train_data is not None, "Must have training data. Run set_train_test_data() to generate"

        # Train the model multiple times and keep track of the model with the lowest minimum training loss
        best_minimum_loss = float('inf')
        best_model = None

        #While you still have retrains left
        for i in range(self.retrain_GP):
            #Create and fit the model
            gp_model = self.set_gp_model(i)
            # Build optimizer
            optimizer=gpflow.optimizers.Scipy()
            # Fit GP to training data
            aux=optimizer.minimize(gp_model.training_loss,
                                gp_model.trainable_variables,
                                options={'maxiter':10**9},
                                method="L-BFGS-B")
            training_loss = gp_model.training_loss().numpy()
            if i == 0:
                first_model = gp_model
                first_loss = training_loss
            if aux.success:
                # Check if this model has the best minimum training loss
                if training_loss < best_minimum_loss:
                    best_minimum_loss = training_loss
                    best_model = gp_model

        #If we have no good models, use the first one
        if best_model is None:
            best_model = first_model
            best_minimum_loss = first_loss

        # gpflow.utilities.print_summary(best_model)
        #Pull out kernel parameters after GP training
        outputscl_final = float(best_model.kernel.kernels[0].variance.numpy())
        lenscl_final = best_model.kernel.kernels[0].lengthscales.numpy()
        noise_final = float(best_model.kernel.kernels[1].variance.numpy())
        
        #Put hyperparameters in a list
        trained_hyperparams = [lenscl_final, noise_final, outputscl_final] 
        
        #Assign self parameters
        self.trained_hyperparams = trained_hyperparams
        self.fit_gp_model = best_model
        self.posterior = self.fit_gp_model.posterior()

        # gpflow.utilities.print_summary(best_model)
        
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
        gp_covar: ndarray, array of gp covariance for the data set
        
        """     
        #Get data in vector form into array form
        if len(data.shape) < 2:
            data.reshape(1,-1)
        #scale eval_point if necessary
        if self.normalize == True:
            eval_points = self.scalerX.transform(data)
        else:
            eval_points = data
        
        eval_points_tf = tf.convert_to_tensor(eval_points)  

        with tf.GradientTape(persistent=True) as tape:
            # By default, only Variables are watched. For gradients with respect to tensors,
            # we need to explicitly watch them:
            tape.watch(eval_points_tf)
            #Evaluate GP given parameter set theta and state point value
            gp_mean_scl, gp_covar_scl = self.posterior.predict_f(eval_points_tf, full_cov=True)

        grad_mean_scl = tape.gradient(gp_mean_scl, eval_points_tf).numpy()
        # grad_var_scl = tape.gradient(gp_covar_scl, eval_points_tf).numpy()

        #Remove dimensions of 1
        gp_mean_scl = gp_mean_scl.numpy()
        gp_covar_scl = np.squeeze(gp_covar_scl.numpy(), axis = 0)

        #Unscale gp_mean and gp_covariance
        if self.normalize == True:
            gp_mean = self.scalerY.inverse_transform(gp_mean_scl.reshape(-1,1)).flatten()
            gp_covar = float(self.scalerY.scale_**2) * gp_covar_scl  
            grad_mean = self.scalerY.inverse_transform(grad_mean_scl.reshape(-1,1)).flatten()
        else:
            gp_mean = gp_mean_scl
            gp_covar = gp_covar_scl
            grad_mean = grad_mean_scl
        
        gp_var = np.diag(gp_covar)

        return gp_mean, gp_var, gp_covar
    
    def eval_gp_mean_var_misc(self, misc_data, featurized_misc_data, covar = False):
        """
        Evaluate the GP mean and variance for a heat map set
        
        Parameters:
        -----------
        misc_data: instance of the Data class, data to evaluate gp mean and variance for containing at least theta_vals and x_vals
        featurized_misc_data: ndarray, featurized data to evaluate containing at least theta_vals and x_vals
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns:
        -------
        misc_gp_mean: ndarray, array of gp_mean for the test set
        misc_var_return: ndarray, array of gp (co)variance for the test set
        
        Notes:
        ------
        Also calculates the gp covariance matrix for the misc. data as a class object stored in misc_data.gp_covar
        """
        
        assert isinstance(misc_data , Data), "misc_data must be type Data"
        assert isinstance(featurized_misc_data, np.ndarray), "featurized_misc_data must be np.ndarray"
        assert len(featurized_misc_data) > 0, "Must have data"
        assert isinstance(covar, bool), "covar must be bool!"
        
        #Evaluate heat map data for GP
        misc_gp_mean, misc_gp_var, misc_gp_covar = self.__eval_gp_mean_var(featurized_misc_data)
        
        #Set data parameters
        misc_data.gp_mean = misc_gp_mean
        misc_data.gp_var = misc_gp_var
        misc_data.gp_covar = misc_gp_covar
        
        if covar == False:
            misc_var_return = misc_gp_var
        else:
            misc_var_return = misc_gp_covar

        return misc_gp_mean, misc_var_return
    
    def eval_gp_mean_var_test(self, covar = False):
        """
        Evaluate the GP mean and variance for the test set
        
        Parameters:
        -----------
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns:
        -------
        test_gp_mean: ndarray, array of gp_mean for the test set
        test_var_return: ndarray, array of gp (co)variance for the test set
        
        Notes:
        ------
        Also calculates the gp covariance matrix for the testing data as a class object stored in GP_Emulator.test_data.gp_covar
        """
        
        assert self.feature_test_data is not None, "Must have testing data. Run set_train_test_data() to generate"
        assert len(self.feature_test_data) > 0, "Must have testing data. Run set_train_test_data() to generate"
        assert isinstance(covar, bool), "covar must be bool!"
        #Evaluate test data for GP
        test_gp_mean, test_gp_var, test_gp_covar = self.__eval_gp_mean_var(self.feature_test_data)
        
        #Set data parameters
        self.test_data.gp_mean = test_gp_mean
        self.test_data.gp_var = test_gp_var
        self.test_data.gp_covar = test_gp_covar
        
        if covar == False:
            test_var_return = test_gp_var
        else:
            test_var_return = test_gp_covar

        return test_gp_mean, test_var_return
    
    def eval_gp_mean_var_val(self, covar = False):
        """
        Evaluate the GP mean and variance for the validation set
        
        Parameters:
        -----------
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns:
        -------
        val_gp_mean: ndarray, array of gp_mean for the validation set
        val_var_return: ndarray, array of gp (co)variance for the validation set
        
        Notes:
        ------
        Also calculates the gp covariance matrix for the validation data as a class object stored in GP_Emulator.gp_val_data.gp_covar
        """
        
        assert self.feature_val_data is not None, "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(self.feature_val_data, np.ndarray), "self.feature_val_data must by np.ndarray"
        assert len(self.feature_val_data) > 0, "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(covar, bool), "covar must be bool!"
        
        #Evaluate test data for GP
        val_gp_mean, val_gp_var, val_gp_covar = self.__eval_gp_mean_var(self.feature_val_data)
        
        #Set data parameters
        self.gp_val_data.gp_mean = val_gp_mean
        self.gp_val_data.gp_var = val_gp_var
        self.gp_val_data.gp_covar = val_gp_covar
        
        if covar == False:
            val_var_return = val_gp_var
        else:
            val_var_return = val_gp_covar
        
        return val_gp_mean, val_var_return
    
    def eval_gp_mean_var_cand(self, covar = False):
        """
        Evaluate the GP mean and variance for the candidate set
        
        Parameters:
        -----------
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns:
        -------
        cand_gp_mean: ndarray, array of gp_mean for the candidate theta set
        cand_var_return: ndarray, array of gp (co)variance for the candidate theta set
        
        Notes:
        ------
        Also calculates the gp covariance matrix for the candidate data as a class object stored in GP_Emulator.cand_data.gp_covar
        """
        
        assert self.feature_cand_data is not None, "Must have validation data. Run set_train_test_data() to generate"
        assert len(self.feature_cand_data) > 0, "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(covar, bool), "covar must be bool!"
        #Evaluate test data for GP
        cand_gp_mean, cand_gp_var, cand_gp_covar = self.__eval_gp_mean_var(self.feature_cand_data)
        
        #Set data parameters
        self.cand_data.gp_mean = cand_gp_mean
        self.cand_data.gp_var = cand_gp_var
        self.cand_data.gp_covar = cand_gp_covar
        
        if covar == False:
            cand_var_return = cand_gp_var
        else:
            cand_var_return = cand_gp_covar

        return cand_gp_mean, cand_var_return
    
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
    __eval_gp_sse_var(data, covar)
    eval_gp_sse_var_misc(misc_data, covar)
    eval_gp_sse_var_test(covar)
    eval_gp_sse_var_val(covar)
    eval_gp_sse_var_cand(covar)
    calc_best_error()
    __eval_gp_ei(sim_data, exp_data, ep_bias, best_error_metrics)
    eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics)
    eval_ei_test(exp_data, ep_bias, best_error_metrics)
    eval_ei_val(exp_data, ep_bias, best_error_metrics)
    eval_ei_cand(exp_data, ep_bias, best_error_metrics)
    add_next_theta_to_train_data(theta_best_sse_data)
    """
    # Class variables and attributes
    
    def __init__(self, gp_sim_data, gp_val_data, cand_data, train_data, test_data, kernel, lenscl, noise_std, outputscl, retrain_GP, set_seed, normalize, feature_train_data, feature_test_data, feature_val_data, feature_cand_data):
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
        set_seed: int or None, random seed
        normalize: bool, determines whether data is normalized w/ Yeo-Johnson transformation + zero-mean, unit-variance normalization
        feature_train_data: ndarray, the feature data for the training data in ndarray form
        feature_test_data: ndarray, the feature data for the testing data in ndarray form
        feature_val_data: ndarray, the feature data for the validation data in ndarray form
        feature_cand_data: ndarray, the feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()
        """
        # Constructor method
        # Inherit objects from GP_Emulator Base Class
        super().__init__(gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, set_seed, normalize, feature_train_data, feature_test_data, feature_val_data, feature_cand_data)
        #Add training and testing data as child features
        self.train_data = train_data
        self.test_data = test_data 
        self.train_data_init = None #Will be populated with the 1st instance of train data
        
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
    
    def set_train_test_data(self, sep_fact, set_seed):
        """
        Finds the simulation data to use as training/testing data.
        
        Parameters
        ----------
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None, Determines seed for randomizations. None if seed is random
        
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
        assert isinstance(set_seed, int), "seed must be int!"
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
        train_data = Data(theta_train, x_train, y_train, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, set_seed)
        self.train_data = train_data
        
        #Get test data and set it as an instance of the data class
        theta_test = self.gp_sim_data.theta_vals[test_idx]
        x_test = self.gp_sim_data.x_vals #x_vals for Type 1 is the same as exp_data. No need to index x
        y_test = self.gp_sim_data.y_vals[test_idx]
        test_data = Data(theta_test, x_test, y_test, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, set_seed)
        self.test_data = test_data
        
        #Set training and validation data features in GP_Emulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)
        
        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data
        
        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data

        #Set the initial training data for the GP Emulator upon creation
        if self.train_data_init is None:
            self.train_data_init = feature_train_data
            
        return train_data, test_data
       
    def __eval_gp_sse_var(self, data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the data
        
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns
        --------
        data.sse: np.ndarray, The sse derived from gp_mean evaluated over the data 
        var_return: np.ndarray, The sse (co)variance derived from the GP model's (co)variance evaluated over the data 
        
        Notes:
        ------
        Also stores the gp sse covariance matrix for the data as a class object stored in data.sse_covar
        
        """
        assert isinstance(covar, bool), "covar must be bool!"
        #For type 1, sse is the gp_mean
        data.sse = data.gp_mean
        data.sse_var = data.gp_var
        data.sse_covar = data.gp_covar

        if covar == False:
            var_return = data.sse_var
        else:
            var_return = data.sse_covar

        return data.sse, var_return
    
    def eval_gp_sse_var_misc(self, misc_data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for any data. Including heat map data
        
        Parameters
        -----------
        misc_data: Instance of Data, the data to evaluate the sse mean and variance for
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns
        --------
        misc_sse_mean: tensor, The sse derived from gp_mean evaluated over the data 
        misc_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the data 
        
        """
        assert isinstance(misc_data , Data), "misc_data must be type Data"
        assert np.all(misc_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(misc_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        
        #For type 1, sse is the gp_mean
        misc_sse_mean, misc_sse_var = self.__eval_gp_sse_var(misc_data, covar)
                    
        return misc_sse_mean, misc_sse_var
    
    def eval_gp_sse_var_test(self, covar = False):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the testing data
        
        Parameters
        -----------
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns
        --------
        test_sse_mean: tensor, The sse derived from gp_mean evaluated over the test data 
        test_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the test data 
        
        """
        assert isinstance(self.test_data , Data), "self.test_data must be type Data"
        assert np.all(self.test_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(self.test_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        
        #For type 1, sse is the gp_mean
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data, covar)
                    
        return test_sse_mean, test_sse_var
    
    def eval_gp_sse_var_val(self, covar = False):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the validation data
        
        Parameters
        -----------
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns
        --------
        val_sse_mean: tensor, The sse derived from gp_mean evaluated over the validation data 
        val_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the validation data 
        
        """
        assert isinstance(self.gp_val_data , Data), "self.gp_val_data must be type Data"
        assert np.all(self.gp_val_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.gp_val_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        
        #For type 1, sse is the gp_mean
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data, covar)
                    
        return val_sse_mean, val_sse_var  
    
    def eval_gp_sse_var_cand(self, covar = False):
        """
        Evaluates GP model sse and sse variance and for an standard GPBO for the candidate theta data
        
        Parameters
        -----------
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns
        --------
        cand_sse_mean: tensor, The sse derived from gp_mean evaluated over the candidate theta data 
        cand_sse_var: tensor, The sse variance derived from the GP model's variance evaluated over the candidate theta data 
        
        """
        assert isinstance(self.cand_data , Data), "self.cand_data must be type Data"
        assert np.all(self.cand_data.gp_mean is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(self.cand_data.gp_var is not None), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        
        #For type 1, sse is the gp_mean
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data, covar)
                    
        return cand_sse_mean, cand_sse_var
    
    def calc_best_error(self):
        """
        Calculates the best error of the model
        
        Returns
        -------
        best_error: float, the best error of the method
        be_theta: np.ndarray, the parameter set associated with the best error of the method
        train_idc: int, the index of the best error in the training data
        
        """   
        assert self.train_data is not None, "Must have self.train_data"
        assert isinstance(self.train_data , Data), "self.train_data must be type Data"
        assert np.all(self.train_data.y_vals is not None), "Must have simulation theta and y data to calculate best error"
        
        #Best error is the minimum sse value of the training data for Type 1
        best_error = np.min(self.train_data.y_vals)
        train_idc = np.argmin(self.train_data.y_vals)
        be_theta = self.train_data.theta_vals[train_idc]
        
        return best_error, be_theta, train_idc
    
    
    def __eval_gp_ei(self, sim_data, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        #Call instance of expected improvement class
        ei_class = Expected_Improvement(ep_bias, sim_data.gp_mean, sim_data.gp_covar, exp_data, best_error_metrics, self.seed, None)
        #Call correct method of ei calculation
        ei, ei_terms_df = ei_class.type_1()
        #Add ei data to validation data class
        sim_data.acq = ei
        
        return ei, ei_terms_df
    
    def eval_ei_misc(self, misc_data, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        misc_data, Instance of Data class, data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and  best_error_x (squared error) values of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(misc_data, exp_data, ep_bias, best_error_metrics)
        return ei, ei_terms_df
    
    def eval_ei_test(self, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error theta, and best_error_x (squared error) values of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(self.test_data, exp_data, ep_bias, best_error_metrics)
        return ei, ei_terms_df
    
    def eval_ei_val(self, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in gp_val_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(self.gp_val_data, exp_data, ep_bias, best_error_metrics)
        
        return ei, ei_terms_df
    
    def eval_ei_cand(self, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        
        Returns
        -------
        ei: The expected improvement of all the data in candidate theta data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(self.cand_data, exp_data, ep_bias, best_error_metrics)
        
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
    __eval_gp_sse_var(data, method, exp_data, covar)
    eval_gp_sse_var_misc(misc_data, method, exp_data, covar)
    eval_gp_sse_var_test(method, exp_data, covar)
    eval_gp_sse_var_val(method, exp_data, covar)
    eval_gp_sse_var_cand(method, exp_data, covar)
    calc_best_error(method, exp_data)
    __eval_gp_ei(sim_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
    eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
    eval_ei_test(exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
    eval_ei_val(exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
    eval_ei_cand(exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
    add_next_theta_to_train_data(theta_best_data)
    """
    # Class variables and attributes
    def __init__(self, gp_sim_data, gp_val_data, cand_data, train_data, test_data, kernel, lenscl, noise_std, outputscl, retrain_GP, set_seed, normalize, feature_train_data, feature_test_data, feature_val_data, feature_cand_data):
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
        set_seed: int or None, random seed
        normalize: bool, determines whether data is normalized w/ Yeo-Johnson transformation + zero-mean, unit-variance normalization
        feature_train_data: ndarray, the feature data for the training data in ndarray form
        feature_test_data: ndarray, the feature data for the testing data in ndarray form
        feature_val_data: ndarray, the feature data for the validation data in ndarray form
        feature_cand_data: ndarray, the feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()
        """
        # Constructor method
        #Inherit objects from GP_Emulator Base Class
        super().__init__(gp_sim_data, gp_val_data, cand_data, kernel, lenscl, noise_std, outputscl, retrain_GP, set_seed, normalize, feature_train_data, feature_test_data, feature_val_data, feature_cand_data)
        #Set training and testing data as child class specific objects
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_init = None #This will be populated with the first set of training thetas
                  
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
    
    def set_train_test_data(self, sep_fact, set_seed):
        """
        Finds the simulation data to use as training/testing data
        
        Parameters
        ----------
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None, Determines seed for randomizations. None if seed is random
        
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
        assert isinstance(set_seed, int), "seed must be int!"
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
        train_data = Data(theta_train, x_train, y_train, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, set_seed)
        self.train_data = train_data
        
        #Get test data and set it as an instance of the data class
        theta_test = self.gp_sim_data.theta_vals[test_rows_idx]
        x_test = self.gp_sim_data.x_vals[test_rows_idx]
        y_test = self.gp_sim_data.y_vals[test_rows_idx]
        test_data = Data(theta_test, x_test, y_test, None, None, None, None, None, self.gp_sim_data.bounds_theta, self.gp_sim_data.bounds_x, sep_fact, set_seed)
        self.test_data = test_data
        
        #Set training and validation data features in GP_Emulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)
        
        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data
         
        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data
            
        #Set the initial training data for the GP Emulator upon creation
        if self.train_data_init is None:
            self.train_data_init = feature_train_data
            
        return train_data, test_data
    
    def __eval_gp_sse_var(self, data, method, exp_data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO
        
        Parameters
        ----------
        data, instance of Data class, parameter sets you want to evaluate the sse and sse variance for
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
        Returns
        --------
        sse_mean: tensor, The sse derived from gp_mean evaluated over param_set 
        sse_var: tensor, The sse variance derived from the GP model's variance evaluated over param_set 
        
        Notes:
        ------
        Also stores the gp sse covariance matrix for the data as a class object stored in data.sse_covar
        Covariance only saved if more than 1 unique theta is present. This value corresponds to the covariance of all the different thetas
        
        """
        assert isinstance(covar, bool), "covar must be bool!"
        
        #Find length of theta and number of unique x in data arrays
        len_theta = data.get_num_theta()
        len_x = len(data.get_unique_x())
        #Infer number of thetas
        num_uniq_theta = int(len_theta/len_x)

        #Reshape y_sim into n_theta rows x n_x columns
        indices = np.arange(0, len_theta, len_x)
        n_blocks = len(indices)
        # Slice y_sim into blocks of size len_x and calculate squared errors for each block
        gp_mean_resh = data.gp_mean.reshape(n_blocks, len_x)
        block_errors = gp_mean_resh - exp_data.y_vals[np.newaxis,:]
        residuals = block_errors.reshape(data.gp_covar.shape[0], -1)
        # Sum squared errors for each block
        sse_mean_org = np.sum((block_errors)**2, axis=1)
        sse_mean = sse_mean_org.flatten()
        
        #Calculate the sse variance. This SSE_variance CAN'T be negative
        sse_var_all = 2*np.trace(data.gp_covar**2) + 4*residuals.T@data.gp_covar@residuals

        #Calculate individual variances Var(SSE[t1]), and Var(SSE[t2])
        if num_uniq_theta == 1:
            sse_var = sse_var_all
            sse_covar = sse_var
        else:
            sse_var = np.zeros(n_blocks)
            for i in range(n_blocks):
                #Get section of covariance matrix that corresponds to the covariance of the different thetas
                covar_t_t = data.gp_covar[i*len_x:(i+1)*len_x, i*len_x:(i+1)*len_x]
                #Get row of block error corresponding to this matrix
                res_theta = block_errors[i].reshape(-1,1)
                #Calculate Variance
                sse_var[i] = 2*np.trace(covar_t_t**2) + 4*res_theta.T@covar_t_t@res_theta
            if num_uniq_theta == 2 and covar == True:
                sse_covar = sse_var_all
            else:
                sse_covar = None      

        #For Method 2B, make sse and sse_covar data in the log form
        # if method.obj.value == 2:
        #     #Propogation of errors: stdev_ln(val) = stdev/val           
        #     sse_var = sse_var/float(residuals.T@residuals)
        #     if sse_covar is not None:
        #         sse_covar = sse_covar/float(residuals.T@residuals)
        #     #Set mean to new value
        #     sse_mean = np.log(sse_mean)

        #Set class parameters
        data.sse = sse_mean
        data.sse_var = sse_var
        data.sse_covar = sse_covar

        if covar == False:
            var_return = data.sse_var
        else:
            var_return = data.sse_covar
        
        return sse_mean, var_return
    
    def eval_gp_sse_var_misc(self, misc_data, method, exp_data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the heat map data
        
        Parameters
        ----------
        misc_data: Instance of Data class, data to evaluate gp sse and sse variance for
        method: Instance of GPBO_Methods, containing data for methods
        exp_data: instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
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
        
        misc_sse_mean, misc_sse_var = self.__eval_gp_sse_var(misc_data, method, exp_data, covar)
        
        return misc_sse_mean, misc_sse_var
    
    def eval_gp_sse_var_test(self, method, exp_data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the test data
        
        Parameters
        ----------
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
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
        
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data, method, exp_data, covar)
        
        return test_sse_mean, test_sse_var
    
    def eval_gp_sse_var_val(self, method, exp_data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the validation data
        
        Parameters
        ----------
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
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
        
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data, method, exp_data, covar)
        
        return val_sse_mean, val_sse_var
    
    def eval_gp_sse_var_cand(self, method, exp_data, covar = False):
        """
        Evaluates GP model sse and sse variance and for an emulator GPBO for the candidate theta data
        
        Parameters
        ----------
        method, Instance of GPBO_Methods, containing data for methods
        exp_data, instance of the Data class, The experimental data of the class. Needs at least the x_vals and y_vals
        covar: bool, determines whether covariance (True) or variance (False) of sse is returned with the gp mean. Default False
        
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
        
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data, method, exp_data, covar)
        
        return cand_sse_mean, cand_sse_var
    
    def calc_best_error(self, method, exp_data):
        """
        Calculates the best error of the model (sse) and squared error for each state point x (squared error)
        
        Parameters
        ----------
        method: Instance of GPBO_Methods, Class containing method information
        exp_data: Instance of Data class, Class containing at least theta, x, and y experimental data
        
        Returns
        -------
        best_error: float, the best error (sse) of the method
        be_theta: np.ndarray, The parameter set associated with the best error value
        best_sq_error: np.ndarray, array of squared errors for each value of x
        org_train_idcs: list, the original training indices of be_theta
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

        # #Reshape y_sim into n_theta rows x n_x columns
        indices = np.arange(0, len_theta, len_x)
        n_blocks = len(indices)
        # Slice y_sim into blocks of size len_x and calculate squared errors for each block
        train_y_resh = self.train_data.y_vals.reshape(n_blocks, len_x)
        ind_errors = (train_y_resh - exp_data.y_vals[np.newaxis,:])**2

        # Sum squared errors for each block
        sse_vals = np.sum(ind_errors, axis=1)
        # print("sse_vals", sse_vals.shape)
        # print("trn shape", self.train_data.theta_vals.shape)
        sse_train_vals = sse_vals.flatten()
        # print(np.argmin(sse_train_vals))

        #List to array
        be_theta = self.train_data.theta_vals[int(np.argmin(sse_train_vals)*len_x)]
        org_train_idcs = [int(np.argmin(sse_train_vals)*len_x), int((np.argmin(sse_train_vals)+1)*len_x)]
        
        #Best error is the minimum of these values
        best_error = np.amin(sse_train_vals)
        best_sq_error =  ind_errors[np.argmin(sse_vals)]

        ##For method 2B, use a log scaled best error
        # if method.obj.value == 2:
        #     best_error = np.log(best_error)
        #     best_sq_error[best_sq_error == 0] += 1e-15 #Add a small value to any zero value to avoid problems in ei calculations
        #     best_sq_error = np.log(best_sq_error) 
            
        return best_error, be_theta, best_sq_error, org_train_idcs
    
    def __eval_gp_ei(self, sim_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples = 2000):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: instance of Method class, method for GP Emulation
        sg_mc_samples: Number of to use for the Tasmanian sparse grid or MC approaches
        
        Returns
        -------
        ei: The expected improvement of all the data in sim_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert 6 >= method.method_name.value >=3, "Must be using method 2A, 2B, 2C, or 2D"
        #Set sparse grid depth if applicable
        if method.sparse_grid == True or method.mc == True:
            assert isinstance(sg_mc_samples, int) and sg_mc_samples > 0, "sg_mc_samples must be positive int for sparse grid and Monte Carlo methods"
        #Call instance of expected improvement class
        ei_class = Expected_Improvement(ep_bias, sim_data.gp_mean, sim_data.gp_covar, exp_data, best_error_metrics, self.seed, sg_mc_samples)
        #Call correct method of ei calculation
        ei, ei_terms_df = ei_class.type_2(method)
        #Add ei data to validation data class
        sim_data.acq = ei
        
        return ei, ei_terms_df
    
    def eval_ei_misc(self, misc_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples = 2000):
        """
        Evaluates gp acquisition function. In this case, ei
        
        Parmaeters
        ----------
        misc_data, Instance of Data class, data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: instance of Method class, method for GP Emulation
        sg_mc_samples: Number of to use for the Tasmanian sparse grid or MC approaches
        
        Returns
        -------
        ei: The expected improvement of all the data in sim_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "Error metric must be a tuple of length 3"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert 6 >= method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
        ei, ei_terms_df = self.__eval_gp_ei(misc_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
        
        return ei, ei_terms_df
    
    def eval_ei_test(self, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples = 2000):
        """
        Evaluates gp acquisition function for testing data. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: instance of Method class, method for GP Emulation
        sg_mc_samples: Number of to use for the Tasmanian sparse grid or MC approaches
        
        Returns
        -------
        ei: The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "Error metric must be a tuple of length 3"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert 6 >= method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"          
        ei, ei_terms_df = self.__eval_gp_ei(self.test_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)

        return ei, ei_terms_df
    
    def eval_ei_val(self, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples = 2000):
        """
        Evaluates gp acquisition function for validation data. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: instance of Method class, method for GP Emulation
        sg_mc_samples: Number of to use for the Tasmanian sparse grid or MC approaches
        
        Returns
        -------
        ei: The expected improvement of all the data in gp_val_data
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "best_error_metrics must be tuple of length 3"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert 6 >= method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
        ei, ei_terms_df = self.__eval_gp_ei(self.gp_val_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
        
        return ei, ei_terms_df
        
    def eval_ei_cand(self, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples = 2000):
        """
        Evaluates gp acquisition function for the candidate theta data. In this case, ei
        
        Parmaeters
        ----------
        sim_data, Instance of Data class, sim data to evaluate ei for
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        ep_bias, Instance of Exploration_Bias, The exploration bias class
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: instance of Method class, method for GP Emulation
        sg_mc_samples: Number of to use for the Tasmanian sparse grid or MC approaches
        
        Returns
        -------
        ei: The expected improvement of all the data in candidate feature
        ei_terms_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter sets
        """
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(ep_bias, Exploration_Bias),  "ep_bias must be type Exploration_bias"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics)==3, "best_error_metrics must be tuple of length 3"
        assert isinstance(method, GPBO_Methods), "method must be instance of GPBO_Methods"
        assert 6 >= method.method_name.value > 2, "method must be Type 2. Hint: Must have method.method_name.value > 2"
        ei, ei_terms_df = self.__eval_gp_ei(self.cand_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples)
        
        return ei, ei_terms_df
    
    def add_next_theta_to_train_data(self, theta_best_data):
        """
        Adds the theta with the highest ei to the training data set
        
        Parameters
        ----------
        theta_best_data: Instance of Data, The class containing the data relavent to theta_best
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
    type_1()
    type_2(method)
    __calc_ei_emulator(gp_mean, gp_var, y_target)
    __calc_ei_log_emulator(gp_mean, gp_var, y_target)
    __ei_approx_ln_term(epsilon, gp_mean, gp_stdev, y_target)
    __calc_ei_sparse(gp_mean, gp_var, y_target)
    __get_sparse_grids(dim, output=0,depth=10, rule="gauss-hermite", verbose = False, alpha = 0)
    __calc_ei_mc(gp_mean, gp_var, y_target)
    __bootstrap(self, pilot_sample, ns=100, alpha=0.05, seed = None)
    """
    def __init__(self, ep_bias, gp_mean, gp_covar, exp_data, best_error_metrics, set_seed, sg_mc_samples = 2000):
        """
        Parameters
        ----------       
        ep_bias: instance of Exploration_Bias, class with information of exploration bias parameter
        gp_mean: tensor, The GP model's mean evaluated over param_set 
        gp_covar: tensor, The GP model's covariance evaluated over param_set
        exp_data: Instance of Data class, the experimental data to evaluate ei with
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        set_seed: int or None, Determines seed for randomizations. None if seed is random
        sg_mc_samples: int, The number of points to use for the Tasmanian sparse grid and Monte Carlo
        """
        assert len(gp_mean) == len(gp_covar), "gp_mean and gp_covar must be arrays of the same length"
        assert isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3, "best_error_metrics must be a tuple of length 3"
        assert all(isinstance(arr, np.ndarray) for arr in (gp_mean, gp_covar, exp_data.y_vals)), "gp_mean, gp_var, and exp_data.y_vals must be ndarrays"
        assert isinstance(ep_bias, Exploration_Bias), "ep_bias must be instance of Exploration_Bias"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(best_error_metrics[0], (float, int)), "best_error_metrics[0] must be float or int. Calculate with GP_Emulator.calc_best_error()"
        assert isinstance(best_error_metrics[1], np.ndarray), "best_error_metrics[1] must be np.ndarray"
        assert isinstance(best_error_metrics[2], np.ndarray) or best_error_metrics[2] is None, "best_error_metrics[2] must be np.ndarray (type 2 ei) or None (type 1 ei)"
        assert isinstance(sg_mc_samples, int) or sg_mc_samples is None, "sg_mc_samples must be int (MC and sparse grid) or None (other)"
        
        # Constructor method
        self.ep_bias = ep_bias
        self.gp_mean = gp_mean
        self.exp_data = exp_data
        self.seed = set_seed
        self.gp_covar = gp_covar
        self.gp_var = np.diag(gp_covar)
        self.best_error = best_error_metrics[0]
        self.be_theta = best_error_metrics[1]
        self.best_error_x = best_error_metrics[2]
        self.samples_mc_sg = sg_mc_samples
        
    def __set_sg_def(self, dim):
        depth = 0
        num_points = 0
        # Compute the maximum depth based on the budget
        while num_points <= self.samples_mc_sg:
            depth += 1
            # Generate the global grid with the current depth
            grid_p = Tasmanian.makeGlobalGrid(dim, 1, depth, "qphyperbolic", 'gauss-hermite-odd')
            
            # Get the number of points on the grid
            num_points = grid_p.getNumPoints()

            # Check if the number of points exceeds the budget
            if num_points > self.samples_mc_sg:
                if depth > 1:
                    depth -= 1
                break
        return depth
    
    def __set_rand_vars(self, mean = None, covar=None):
        """
        Sets random variables for MC integration
        
        Returns:
        ---------
        random_vars: np.ndarray, array of multivariate normal random variables
        """
        dim = len(self.exp_data.y_vals)
        mc_samples = self.samples_mc_sg #Set 2000 MC samples
        #Use set seed for integration 
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
        #Always use the same seed if one is not set
            np.random.seed(1)
            
        eigvals, eigvecs = np.linalg.eigh(covar)

        #Get random standard variables
        rng = np.random.default_rng(self.seed)
        random_vars_stand = rng.multivariate_normal(np.zeros(dim), np.eye(dim), mc_samples)
        #If we have a mean and a variance
        if mean is not None or covar is not None:
            #Use the mvn function directly to get the random variables if matrix is Positive Definite
            if np.all(eigvals > 1e-7):
                random_vars = rng.multivariate_normal(mean, np.real(covar), mc_samples, tol=1e-5, method='eigh')
                # print(random_vars[0:3,:])
            #Otherwise, use the LDL decomposition
            else:
                lu, d, perm = scipy.linalg.ldl(np.real(covar), lower=True) # Use the lower part
                sqrt_d = np.sqrt(np.diag(d))[:, np.newaxis]
                random_vars = (mean[:, np.newaxis] + lu[:, perm] @ (sqrt_d * random_vars_stand.T)).T
                # print(random_vars[0:3,:])
                np.save("mean_mc.npy", mean)
                np.save("covar_mc.npy", covar)

        return random_vars
    
    def type_1(self):
        """
        Calculates expected improvement of type 1 (standard) GPBO given gp_mean, gp_var, and best_error data
        
        Returns
        -------
        ei: ndarray, The expected improvement of the parameter set
        ei_term_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter set
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
        ei_term_df: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter set
        """        
        ei_term_df = pd.DataFrame()
        assert isinstance(self.best_error_x, np.ndarray), "best_error_metrics[1] must be np.ndarray for type 2 ei calculations"
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
                ei[i], row_data = self.__calc_ei_emulator(gp_mean_i, gp_var_i, self.exp_data.y_vals)
                
            elif method.method_name.value == 4: #2B
                ei[i], row_data = self.__calc_ei_log_emulator(gp_mean_i, gp_var_i, self.exp_data.y_vals)
                
            elif method.method_name.value == 5: #2C
                ei[i], row_data = self.__calc_ei_sparse(gp_mean_i, gp_var_i, self.exp_data.y_vals)

            elif method.method_name.value == 6: #2D
                ei[i], row_data = self.__calc_ei_mc(gp_mean_i, gp_var_i, self.exp_data.y_vals)

            else:
                raise ValueError("method.method_name.value must be 3 (2A), 4 (2B), 5 (2C), or 6 (2D)")
        
        # Concatenate the temporary DataFrame with the main DataFrame
        ei_term_df = pd.concat([ei_term_df, row_data], ignore_index=True)
        ei_term_df.columns = row_data.columns.tolist()
        
        return ei, ei_term_df 
        
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
        ei_temp: ndarray, the expected improvement for one term of the GP model
        row_data: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
         #Create column names
        columns = ["bound_l", "bound_u", "cdf_l", "cdf_u","eta_l", "eta_u", "psi_l", "psi_u", "ei_term1", "ei_term2",
                   "ei_term3", "ei", "ei_total"]

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
            best_errors_x = self.best_error_x[valid_indices]

            #If variance is close to zero this is important
            with np.errstate(divide = 'warn'):
                #Creates upper and lower bounds and described by Equation X in Manuscript
                bound_a = ((y_target_val - gp_mean_val) + np.sqrt(best_errors_x*self.ep_bias.ep_curr))/pred_stdev_val
                bound_b = ((y_target_val - gp_mean_val) - np.sqrt(best_errors_x*self.ep_bias.ep_curr))/pred_stdev_val
                bound_lower = np.minimum(bound_a,bound_b)
                bound_upper = np.maximum(bound_a,bound_b)        

                #Creates EI terms in terms of Equation X in Manuscript
                ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
                ei_term1_comp2 = (best_errors_x*self.ep_bias.ep_curr) - (y_target_val - gp_mean_val)**2

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
            row_data_lists = pd.DataFrame([[bound_lower, bound_upper, norm.cdf(bound_lower), norm.cdf(bound_upper), 
                                  ei_eta_lower, ei_eta_upper, ei_term3_psi_lower, ei_term3_psi_upper,
                                  ei_term1, ei_term2, ei_term3, ei, ei_temp]], columns=columns)
        else:
            ei_temp = 0
            row_data_lists = pd.DataFrame([["N/A", "N/A", "N/A", "N/A", 
                                  "N/A", "N/A", "N/A", "N/A",
                                  "N/A", "N/A", "N/A", "N/A", ei_temp]], columns=columns)
     
        row_data = row_data_lists.apply(lambda col: col.explode(ignore_index = True), axis=0).reset_index(drop=True)
        
        return ei_temp, row_data

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
        ei_temp: ndarray, the expected improvement for one term of the GP model
        row_data: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        columns = ["best_error", "bound_l", "bound_u", "ei_term1", "ei_term2", "ei", "ei_total"]

        #Initialize ei as all zeros
        ei = np.zeros(len(gp_var))

        #Create a mask for values where pred_stdev > 0 
        pos_stdev_mask = (gp_var > 0)
        best_errors_x_all = np.log(self.best_error_x)

        #Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            #Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            # best_errors_x = self.best_error_x[valid_indices]
            best_errors_x = copy.deepcopy(best_errors_x_all)[valid_indices]
            best_errors_x[best_errors_x == 0] += 1e-15 #Add a small value to any zero value to avoid problems in ei calculations
            #Important when stdev is close to 0
            with np.errstate(divide = 'warn'):
                #Creates upper and lower bounds and described by Alex Dowling's Derivation
                bound_a = ((y_target_val - gp_mean_val) +np.sqrt(np.exp(best_errors_x*self.ep_bias.ep_curr)))/pred_stdev_val #1xn
                bound_b = ((y_target_val - gp_mean_val) -np.sqrt(np.exp(best_errors_x*self.ep_bias.ep_curr)))/pred_stdev_val #1xn
                bound_lower = np.minimum(bound_a,bound_b)
                bound_upper = np.maximum(bound_a,bound_b) 

                #Calculate EI
                args = (gp_mean_val, pred_stdev_val, y_target_val, self.ep_bias.ep_curr)
                ei_term_1 = (best_errors_x*self.ep_bias.ep_curr)*( norm.cdf(bound_upper)-norm.cdf(bound_lower) )
                ei_term_2_out = np.array([integrate.quad(self.__ei_approx_ln_term, bl, bu, args=(gm, ps, yt)) for bl, bu, gm, ps, yt in zip(bound_lower, bound_upper, gp_mean_val, pred_stdev_val, y_target_val)])

                ei_term_2 = (-2)*ei_term_2_out[:,0] 
                term_2_abs_err = ei_term_2_out[:,1]
                
                #Add ei values to correct indecies.
                ei[valid_indices] = ei_term_1 + ei_term_2
        
            #The Ei is the sum of the ei at each value of x
            ei_temp = np.sum(ei)
            row_data_lists = pd.DataFrame([[best_errors_x, bound_lower, bound_upper, ei_term_1, ei_term_2, ei, 
                                  ei_temp]], columns=columns)
        else:
            ei_temp = 0
            # row_data_lists = pd.DataFrame([[self.best_error_x, "N/A", "N/A", "N/A", "N/A", "N/A", ei_temp]], columns=columns)
            row_data_lists = pd.DataFrame([[best_errors_x_all, "N/A", "N/A", "N/A", "N/A", "N/A", ei_temp]], columns=columns)
        
        row_data = row_data_lists.apply(lambda col: col.explode(ignore_index = True), axis=0).reset_index(drop=True)
  
        return ei_temp, row_data

    def __ei_approx_ln_term(self, epsilon, gp_mean, gp_stdev, y_target): 
        """ 
        Calculates the integrand of expected improvement of the emulator approach using the log version
        
        Parameters
        ----------
        epsilon: The random variable. This is the variable that is integrated w.r.t
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
        ei_temp: ndarray, the expected improvement for one term of the GP model
        row_data: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        columns = ["best_error", "sse_temp", "improvement", "ei_total"]

        #Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0) 
        pos_stdev_mask = (gp_var >= 0)

        #Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            ndims = len(y_target)
            #Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            gp_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            gp_mean_min_y = y_target_val - gp_mean_val

            # #Obtain Sparse Grid points and weights
            # points_p, weights_p = self.__get_sparse_grids(len(y_target_val), output=0, depth=self.sg_depth, rule='gauss-hermite', 
            #                                               verbose=False)         
            # # Calculate gp_var multiplied by points_p
            # gp_stdev_points_p = gp_stdev_val * (np.sqrt(2)*points_p)
            # # Calculate the SSE for all data points simultaneously
            # sse_temp = np.sum((gp_mean_min_y[:, np.newaxis].T - gp_stdev_points_p)**2, axis=1)
            #Get maximum depth given number of points p
            sg_depth = self.__set_sg_def(ndims)
            points_p, weights_p = self.__get_sparse_grids(ndims, output=1, depth=sg_depth, rule='gauss-hermite-odd', 
                                                           verbose=False) 
            
            #Diagonalize covariance matrix
            try:
                #As long as the covariance matrix is positive definite use Cholesky decomposition
                L = scipy.linalg.cholesky(np.real(self.gp_covar), lower=True)  
            except:
                #If it is not, use LDL decomposition instead
                lu, d, perm = scipy.linalg.ldl(np.real(self.gp_covar), lower=True) # Use the upper part
                L = lu[:, perm]@np.diag(np.sqrt(d))
                np.save("covar_sg.npy", self.gp_covar)

            transformed_points = L@points_p.T
            gp_random_vars = self.gp_mean[:, np.newaxis] + np.sqrt(2)*(transformed_points)
            sse_temp = np.sum((y_target[:, np.newaxis] - gp_random_vars)**2, axis=0)
            # Apply max operator (equivalent to max[(best_error*ep) - SSE_Temp,0])
            error_diff = self.best_error*self.ep_bias.ep_curr - sse_temp
            # improvement = np.maximum(error_diff, 0)
            #Smooth max improvement function
            improvement = (0.5)*(error_diff + np.sqrt(error_diff**2 + 1e-7))

            # Calculate EI_temp using vectorized operations
            ei_temp = (np.pi**(-ndims/2))*np.dot(weights_p, improvement)
            
        else:
            ei_temp = 0
            sse_temp = "N/A"
            improvement = "N/A"

        row_data_lists = pd.DataFrame([[self.best_error, sse_temp, improvement, ei_temp]], columns=columns)
        row_data = row_data_lists.apply(lambda col: col.explode(ignore_index = True), axis=0).reset_index(drop=True)
            
        return ei_temp, row_data

    def __get_sparse_grids(self, dim, output=1,depth=10, rule="gauss-hermite-odd", verbose = False, alpha = 0):
        '''
        This function shows the sparse grids generated with different rules
        
        Parameters:
        -----------
        dim: int, sparse grids dimension
        output: int, output level for function that would be interpolated. Default is zero
        depth: int, depth level. Controls density of abscissa points. Uses hyperbolic level system. Default 10
        rule: str, quadrature rule. Default is 'gauss-hermite'
        verbose: bool, determines Whether or not plot of sparse grid is shown. Default False
        alpha: int, specifies $\alpha$ parameter for the integration weight $\rho(x)$. Default 0

        Returns:
        --------
        points_p: ndarray, The sparse grid points
        weights_p: ndarray, The Gauss-Legendre Quadrature Rule Weights    

        Other:
        ------
        A figure shows 2D sparse grids (if verbose = True)
        '''
        #Get grid points and weights
        grid_p = Tasmanian.makeGlobalGrid(dim,output,depth,"qphyperbolic",rule)
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
    
    def __calc_ei_mc(self, gp_mean, gp_var, y_target):
        """ 
        Calculates the expected improvement of the emulator approach with log scaling (2B)
        
        Parameters
        ----------
        gp_mean: ndarray, model mean at same state point x and experimental data value y
        gp_variance: ndarray, model variance at same state point x and experimental data value y
        y_target: ndarray, the expected value of the function from data or other source

        Returns
        -------
        ei_mean: ndarray, the expected improvement for one term of the GP model
        row_data: pd.DataFrame, pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        #Set column names
        columns = ["best_error", "sse_temp", "improvement", "ci_lower", "ci_upper", "ei_total"]

        #Calc EI
        #Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0) 
        pos_stdev_mask = (gp_var >= 0)

        #Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # valid_indices = np.where(pos_stdev_mask)[0]
            # gp_stdev_val = np.sqrt(gp_var[valid_indices])
            # gp_mean_val = gp_mean[valid_indices]
            # y_target_val = y_target[valid_indices]
            # mean_min_y = y_target_val - gp_mean_val
        
            # # Calculate gp_var multiplied by points_p
            # gp_stdev_rand_var = gp_stdev_val * self.random_vars

            # # Calculate the SSE for all data points simultaneously
            # sse_temp = np.sum((mean_min_y[:, np.newaxis].T - gp_stdev_rand_var)**2, axis=1)

            # # Apply max operator (equivalent to max[(best_error*ep) - SSE_Temp,0])
            #Set random variables for MC integration
            self.random_vars = self.__set_rand_vars(self.gp_mean, self.gp_covar)
            sse_temp = np.sum((y_target[:, np.newaxis].T - self.random_vars)**2, axis=1)
            error_diff = self.best_error*self.ep_bias.ep_curr - sse_temp 
            ## improvement = np.maximum(error_diff, 0).reshape(-1,1)
            #Smooth max improvement function
            improvement = (0.5)*(error_diff + np.sqrt(error_diff**2 + 1e-7)).reshape(-1,1)

            # Flatten improvement
            ei_temp = improvement.flatten()

        else:
            ei_temp = 0
            sse_temp = "N/A"
            improvement = "N/A"
            
        #Calc monte carlo integrand for each theta and add it to the total
        ei_mean = np.average(ei_temp) #y.sum()/len(y)
        #Note: Domain for random variable is 0-1, so V for MC is 1

        #Perform bootstrapping
        ci_interval = self.__bootstrap(ei_temp, ns=100, alpha=0.05, set_seed = self.seed)
        
        ci_l = ci_interval[0]
        ci_u = ci_interval[1]
        
        row_data_lists = pd.DataFrame([[self.best_error, sse_temp, improvement, ci_l, ci_u, ei_temp]], columns=columns)
        row_data = row_data_lists.apply(lambda col: col.explode(ignore_index = True), axis=0).reset_index(drop=True)
  
        return ei_mean, row_data

    def __bootstrap(self, pilot_sample, ns=100, alpha=0.05, set_seed = None):
        """
        Bootstrapping code for Monte Carlo method. Generously provided by Ryan Smith.
        
        Parameters
        ----------
        pilot_sample: np.ndarray (n_samples x dim param set), the samples to perform bootstrapping on
        ns: int, number of bootstrapping samples. Default 100
        alpha: float, On interval (0,1). The level of significance associated with the bootstrapping. Default 0.05
        set_seed: int or None, seed associated with bootstrapping. Default None
        
        Returns:
        --------
        ci_percentile: np.ndarray, The confidence interval of the MC samples        
        """
        # pilot_sample has one column per rv, one row per observation
        # alpha is the level of significance; 0.05 for 95% confidence interval
        quantiles = np.array([alpha*0.5, 1.0-alpha*0.5])

        #Set seed
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.seed(1)

        #Determine mean of all original samples and its shape
        theta_orig = np.mean(pilot_sample,axis=0)

        #Initialize bootstrap samples as zeros
        theta_bs = np.zeros(tuple([ns]+list(theta_orig.shape)))

        #Create bootstrap samples
        for ibs in range(ns):
            samples = np.random.choice(pilot_sample, size= pilot_sample.shape[0], replace=True)
            theta_bs[ibs,...] = np.mean(samples, axis = 0)

        # percentile CI
        ci_percentile = np.quantile(theta_bs, quantiles, 0)

        # return theta_orig, theta_bs, CI_percentile
        return ci_percentile

class Exploration_Bias():
    """
    Base class for methods of calculating explroation bias at each bo iter
    
    Methods
    -------
    __bound_ep(ep_val)
    set_ep()
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
        bo_iter_max: int, The maximum values of BO iterations
        e_inc: float, the increment for the Boyle's method for calculating exploration parameter: Recommendation is 1.5
        ep_f: float, The final exploration parameter value: Recommendation is 0
        improvement: Bool, Determines whether last objective was an improvement. Default False
        best_error: float, The lowest (sse) error value in the training data
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
        #Set ep max and min based off of mathematical bound reasoning
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
        ep_val: int or float, the value of the exploration parameter within self.ep_min and self.ep_max
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
        For these parameters, ep gets normalized between 0 and 2 given a neutral value of 1 as the starting point
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
        configuration: dict, dictionary containing the configuration of the BO algorithm
        simulator_class: Instance of Simulator class, class containing values of simulation parameter data at each BO iteration
        exp_data_class: Instance of Data class, The experimental data for the workflow
        list_gp_emulator_class: list of GP_Emulator instances, contains all gp_emulator information at each BO iter
        results_df: pd.DataFrame, dataframe including the values pertinent to BO for all BO runs
        max_ei_details_df: pd.DataFrame, dataframe including ei components of the best EI at each iter
        why_term: str, string detailing the reason for algorithm termination
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
    __get_best_error()
    __make_starting_opt_pts(best_error_metrics)
    __gen_start_pts_mc_sparse(best_error_metrics)
    __gen_start_pts_not_mc_sparse
    __opt_with_scipy(opt_obj, beta)
    __scipy_fxn(theta, opt_obj, best_error_metrics, beta)
    create_heat_map_param_data(n_points_set)
    __augment_train_data(theta_best_data)
    create_data_instance_from_theta(theta_array)
    # __get_kappa(beta)
    # __get_regret_term(min_sse_theta_data, max_ei_theta_data)
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
        self.sg_mc_samples = 2000 #This can be changed at will

        self.__min_obj_temp = None
        self.__min_obj_class = None
               
    
    def __gen_emulator(self):
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
            k = np.maximum(self.exp_data.get_num_x_vals()-1,1)
            #If using objective sse use var of a chi^2 distribution (2k)
            if not self.method.obj.value == 2:
                noise_scl_fact = np.sqrt(2*k)
                noise_std = self.simulator.noise_std*noise_scl_fact 
            #If using objective ln(sse) guess the noise std
            else:
                noise_std = None #np.sqrt(float(scipy.special.polygamma(1, (k*self.simulator.noise_std**2)/2)))

            gp_emulator = Type_1_GP_Emulator(all_gp_data, all_val_data, None, None, None, self.cs_params.kernel, self.cs_params.lenscl,
                                             noise_std, self.cs_params.outputscl, self.cs_params.retrain_GP, self.cs_params.seed, 
                                             self.cs_params.normalize, None, None, None, None)
        else:
            all_gp_data = self.sim_data
            all_val_data = self.val_data
            noise_std = self.simulator.noise_std #Yexp_std is exactly the noise_std of the GP Kernel
            gp_emulator = Type_2_GP_Emulator(all_gp_data, all_val_data, None, None, None, self.cs_params.kernel, self.cs_params.lenscl, noise_std, self.cs_params.outputscl, 
                                             self.cs_params.retrain_GP, self.cs_params.seed, self.cs_params.normalize, None, None, None, None)
            
        return gp_emulator
    
    
    def __get_best_error(self):
        """
        Helper function to calculate the best error (sse) and squared error calculations over x (squared error) given the method.
        
        Returns
        -------
        be_data: Instance of Data class, contains best_error as an instance of the data class
        be_metrics: tuple of (float, np.ndarray, np.ndarray). The min_SSE, param at min_SSE, and squared residuals
        """
        
        if self.method.emulator == False:
            #Type 1 best error is inferred from training data 
            best_error, be_theta, train_idx = self.gp_emulator.calc_best_error()
            best_errors_x = None
            be_data = self.create_data_instance_from_theta(be_theta.flatten(), get_y = False)
            be_data.y_vals = np.atleast_1d(self.gp_emulator.train_data.y_vals[train_idx])
        else:
            #Type 2 best error must be calculated given the experimental data
            best_error, be_theta, best_errors_x, train_idx = self.gp_emulator.calc_best_error(self.method, self.exp_data)
            be_data = self.create_data_instance_from_theta(be_theta.flatten(), get_y = False)
            be_data.y_vals = self.gp_emulator.train_data.y_vals[train_idx[0]:train_idx[1]]
        
        be_metrics = best_error, be_theta, best_errors_x

        return be_data, be_metrics
        
    def __make_starting_opt_pts(self, best_error_metrics):
        """
        Makes starting point for optimization with scipy
        
        Parameters:
        -----------
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        
        Returns:
        --------
        starting_pts: np.ndarray, array of parameter set initializations for self.__opt_with_scipy
        """
        #Note: Could make this generate 2 sets of starting points based on whether you want to optimize sse or ei
        #For sparse grid and mc methods
        if self.method.sparse_grid == True or self.method.mc == True:
            starting_pts = self.__gen_start_pts_mc_sparse(best_error_metrics)
        else:
            starting_pts = self.__gen_start_pts_not_mc_sparse()
            
        return starting_pts
    
    def __gen_start_pts_mc_sparse(self, best_error_metrics):
        """
        Makes starting point for optimization with scipy if using sparse grid or monte carlo methods
        
        Parameters:
        -----------
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        
        Returns:
        --------
        starting_pts: np.ndarray, array of parameter set initializations for self.__opt_with_scipy
        """
        #Generate n LHS Theta vals
        num_mc_theta = 500
        theta_vals = self.simulator.gen_theta_vals(num_mc_theta)
        
        #Add repeated theta_vals and experimental x values
        rep_theta_vals = np.repeat(theta_vals, len(self.exp_data.x_vals) , axis = 0)
        rep_x_vals = np.vstack([self.exp_data.x_vals]*num_mc_theta)
        
        #Create instance of Data Class
        sp_data = Data(rep_theta_vals, rep_x_vals, None, None, None, None, None, None, self.simulator.bounds_theta_reg, self.simulator.bounds_x, self.cs_params.sep_fact, self.cs_params.seed)
        
        #Evaluate GP mean and Var (This is the slowest step)
        feat_sp_data = self.gp_emulator.featurize_data(sp_data)
        sp_data.gp_mean, sp_data.gp_var = self.gp_emulator.eval_gp_mean_var_misc(sp_data, feat_sp_data)
        # print(sp_data.gp_covar)
        # print(np.linalg.det(sp_data.gp_covar))

        #Evaluate GP SSE and SSE_Var (This is the 2nd slowest step)
        sp_data_sse_mean, sp_data_sse_var = self.gp_emulator.eval_gp_sse_var_misc(sp_data, self.method, self.exp_data)
        
        #Note - Use Sparse grid EI for approximations
        #Evaluate EI using Sparse Grid or EI (This is relatively quick)
        method_3 = GPBO_Methods(Method_name_enum(3))
        sp_data_ei, iter_max_ei_terms = self.gp_emulator.eval_ei_misc(sp_data, self.exp_data, self.ep_bias, best_error_metrics, 
                                                                  method_3)
        
        ##Sort by min(-ei)
        # Create a list of tuples containing indices and values
        indexed_values = list(enumerate(-1*sp_data_ei)) #argmin(-ei) = argmax(ei)

        # Sort the list of tuples based on values
        sorted_values = sorted(indexed_values, key=lambda x: x[1])

        # Extract the indices from the sorted list
        min_indices = [index for index, _ in sorted_values]
        #Sets the points in order based on the indices
        all_pts = theta_vals[min_indices]
        
        #Choose top retrain_GP points as starting points
        starting_pts = all_pts[:self.cs_params.reoptimize_obj+1]
        
        return starting_pts
    
    def __gen_start_pts_not_mc_sparse(self):
        """
        Makes starting point for optimization with scipy if not using sparse grid or monte carlo methods
        
        Returns:
        --------
        starting_pts: np.ndarray, array of parameter set initializations for self.__opt_with_scipy
        """
        #If validation data doesn't exist or is shorter than the number of times you want to retrain
        if self.gp_emulator.gp_val_data is None or len(self.gp_emulator.gp_val_data.get_unique_theta()) < self.cs_params.reoptimize_obj+1:
            #Create validation points equal to number of retrain_GP
            starting_pts = self.simulator.gen_theta_vals(self.cs_params.reoptimize_obj+1)
        #Otherwise, your starting point array is your validation data unique theta values
        else:
            #Set seed
            if self.cs_params.seed is not None:
                np.random.seed(self.cs_params.seed)
                
            #Find unique theta values and make array of indices
            points = self.gp_emulator.gp_val_data.get_unique_theta()
            idcs = np.arange(len(points))
            #Get random indices to use (sample w/out replacement)
            idcs_to_use = np.random.choice(idcs, self.cs_params.reoptimize_obj+1, False)
            #Get theta values associated with those indices
            starting_pts = points[idcs_to_use]

        return starting_pts
    
    def __opt_with_scipy(self, opt_obj, beta = None):
        """
        Optimizes a function with scipy.optimize
        
        Parameters
        ----------
        opt_obj: str, which objective to calculate. neg_ei, sse, or lcb
        beta: float or None, The value of beta for calculating the lcb. Only necessary when opt_obj == 'lcb'
        
        Returns:
        --------
        best_val: float, The optimized value of the function
        best_theta: ndarray, The theta set corresponding to val_best
        """
        self.__min_obj_class = None

        assert isinstance(opt_obj, str), "opt_obj must be string!"
        assert opt_obj in ["neg_ei", "E_sse", "sse", "lcb"], "opt_obj must be 'neg_ei', 'sse', or 'lcb'"

        #Note add +1 because index 0 counts as 1 reoptimization
        if self.cs_params.reoptimize_obj > 50:
            warnings.warn("The objective will be reoptimized more than 50 times!")
        
        #Calc best error        
        be_data, best_error_metrics = self.__get_best_error()
        
        #Find bounds and arguments for function
        bnds = self.simulator.bounds_theta_reg.T #Transpose bounds to work with scipy.optimize
        #Need to account for normalization here (make bounds array of [0,1]^dim_theta)
    
        ## Loop over each validation point/ a certain number of validation point thetas
        for i in range(self.cs_params.reoptimize_obj+1):
            #Choose a random index of theta to start with
            theta_guess = self.opt_start_pts[i].flatten()
            
            #Initialize L-BFGS-B as default optimization method
            obj_opt_method = "L-BFGS-B"
                
            # try:
            #Call scipy method to optimize EI given theta
            #Using L-BFGS-B instead of BFGS because it allowd for bounds
            best_result = optimize.minimize(self.__scipy_fxn, theta_guess, bounds=bnds, method = obj_opt_method, args=(opt_obj, 
                                                                                                                        best_error_metrics,
                                                                                                                        beta))
            # except ValueError: 
            #     #If the intialized theta causes scipy.optimize to choose nan values, skip it
            #     pass
        
        best_val = self.__min_obj_class.acq
        best_class = self.__min_obj_class

        if opt_obj != "lcb":
            best_class_simple = self.create_data_instance_from_theta(self.__min_obj_class.theta_vals[0])
            best_class.y_vals = best_class_simple.y_vals
        else:
            best_class = self.__min_obj_class
                    
        return best_val, best_class
        
    def __scipy_fxn(self, theta, opt_obj, best_error_metrics, beta):
        """
        Calculates either -ei [0], sse objective[1], or lower confidence bound[2] at a candidate theta value
        
        Parameters
        -----------
        theta: ndarray, the array of theta values to optimize
        opt_obj: str, which objective to calculate. neg_ei, sse, or lcb
        best_error_metrics: tuple, the best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        beta: float or None, The value of beta for calculating the lcb. Only necessary when opt_obj == 'lcb'
        
        Returns:
        --------
        obj: float, Either neg_ei or sse for candidate theta
        
        """             
        #Note, theta must be in array form ([ [1,2] ])
        #copy theta into candidate point in GP Emulator (to be added)
        #Check that any of the values are not NaN
        #If they are nan
        #Set seed
        if self.cs_params.seed is not None:
            np.random.seed(self.cs_params.seed)
            
        if np.isnan(theta).any():
            #If there are nan values, set neg ei to 1 (ei = -1) 
            if opt_obj == "neg_ei":
                obj = 1
            #Set sse and lcb to self.sse_penalty
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
            if opt_obj == "sse":
                #Objective to minimize is log(sse) if using 1B, and sse for all other methods
                obj = cand_sse_mean
            elif opt_obj == "E_sse":
                #Objective to minimize is (E)[sse] for method ESSE
                obj = cand_sse_mean + np.sum(cand_sse_var)
            elif opt_obj == "lcb":
                assert isinstance(beta, (int, float, np.float64)), "beta must be float or int"
                #Objective to minimize is gp_mean - beta_gp_var if using 1B or 2B, and sse for all other methods
                obj = cand_sse_mean + np.sqrt(abs(beta*cand_sse_var))
            else:
                #Otherwise objective is ei
                if self.method.emulator == False:
                    ei_output = self.gp_emulator.eval_ei_cand(self.exp_data, self.ep_bias, best_error_metrics)
                else:
                    ei_output = self.gp_emulator.eval_ei_cand(self.exp_data, self.ep_bias, best_error_metrics, self.method, self.sg_mc_samples)
                obj = -1*ei_output[0]

            set_acq_val = True

            #Save candidate class if there is no current value
            if self.__min_obj_class == None:
                self.__min_obj_class = self.gp_emulator.cand_data
            #The sse/lcb objective is smaller than what we have so far
            elif self.__min_obj_class.acq > obj and opt_obj != "neg_ei":
                self.__min_obj_class = self.gp_emulator.cand_data
            #The ei objective is larger than what we have so far
            elif self.__min_obj_class.acq*-1 > obj and opt_obj == "neg_ei":
                self.__min_obj_class = self.gp_emulator.cand_data
            #For SSE, if the objective is the same, randomly choose between the two (since sse is an objective fxn)
            elif np.isclose(self.__min_obj_class.acq, obj, rtol=1e-7) and opt_obj == "sse":
                random_number = random.randint(0, 1)
                if random_number > 0:
                    self.__min_obj_class = self.gp_emulator.cand_data
                else:
                    set_acq_val = False
            #For EI/E_sse/lcb (acquisition fxns) switch to the value farthest from any training data
            elif np.isclose(self.__min_obj_class.acq, obj, rtol=1e-7):
                #Get the distance between the candidate and the current min_obj_class value and the training data
                dist_old = distance.cdist(self.gp_emulator.train_data.theta_vals, 
                                 self.__min_obj_class.theta_vals[0,:].reshape(1,-1),
                                 metric='euclidean').ravel().max()
                dist_new = distance.cdist(self.gp_emulator.train_data.theta_vals, 
                                 self.gp_emulator.cand_data.theta_vals[0,:].reshape(1,-1),
                                 metric='euclidean').ravel().max()
                #If the distance of the new point is larger or equal to the old point, keep the new point
                if dist_new >= dist_old:
                    self.__min_obj_class = self.gp_emulator.cand_data
                else:
                    set_acq_val = False
            else:
                set_acq_val = False

            if set_acq_val and opt_obj != "neg_ei":
                self.__min_obj_class.acq = obj
        
        return obj

    def create_heat_map_param_data(self, n_points_set = None):
        """
        Creates parameter sets that can be used to create heat maps of data at any given iteration

        Parameters:
        -----------
        n_points_set: int or None, the number of points to use per axis for creating heat maps. Default None. If None, the number of unique simulation points is used
        
        Returns:
        --------
        heat_map_data_dict: dict, heat map data for each set of 2 parameters indexed by parameter name tuple ("param_1,param_2")
        """      
        assert isinstance(self.gp_emulator, (Type_1_GP_Emulator, Type_2_GP_Emulator)), "self.gp_emulator must be instance of Type_1_GP_Emulator or Type_2_GP_Emulator"
        assert isinstance(self.gp_emulator.gp_sim_data, Data), "self.gp_emulator.gp_sim_data must be an instance of Data!"
        assert isinstance(self.gen_meth_theta, Gen_meth_enum), "self.gen_meth_theta must be instance of Gen_meth_enum"
        assert isinstance(self.exp_data.x_vals, (np.ndarray)), "self.exp_data.x_vals must be np.ndarray"
        assert isinstance(n_points_set, int) or n_points_set is None, "n_points_set must be None or int"
        
        #Create list of heat map theta data
        heat_map_data_dict = {}
        
        #Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(0,self.simulator.dim_theta-1,self.simulator.dim_theta)
        #Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype = int)

        #Set x_vals
        norm_x_vals = self.exp_data.x_vals
        num_x = self.exp_data.get_num_x_vals()

        #If no number of points is set, use the length of the unique simulation thetas
        if n_points_set == None:
            #Use number of training theta for number of theta points
            n_thetas_points = len(self.gp_emulator.gp_sim_data.get_unique_theta())
            #Initialze meshgrid-like set of theta values at their true values 
            #If points were generated with an LHS, the number of points per parameter is n_thetas_points for the meshgrid
            if self.gen_meth_theta.value == 1:
                n_points = n_thetas_points
            else:
                #For a meshgrid, the number of theta values/ parameter is n_thetas_points for the meshgrid ^(1/theta_dim)
                n_points = int((n_thetas_points)**(1/self.simulator.dim_theta))
        else:
            n_points = n_points_set

        #Ensure we will never generate more than 5000 pts per heat map
        # if self.method.emulator == True:
        if num_x*n_points**2 >= 5000:
            n_points = int(np.sqrt(5000/(num_x)))

        #Meshgrid set always defined by n_points**2
        #Set thetas for meshgrid. Never use more than 10000 points
        theta_set = np.tile(np.array(self.simulator.theta_true), (n_points**2, 1))
        
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

            #Append data set to dictionary with name
            heat_map_data_dict[data_set_name] = data_set
            
        return heat_map_data_dict
                
    def __augment_train_data(self, theta_best_data):
        """
        Augments training data given a new point

        Parameters
        ----------
        theta_best_data: Instance of data class, The theta value associated with the scipy optimize calculated best theta
        """

        #Augment training theta, x, and y/sse data
        self.gp_emulator.add_next_theta_to_train_data(theta_best_data)
                   
    def create_data_instance_from_theta(self, theta_array, get_y=True):
        """
        Creates instance of Data from an nd.array theta set
        
        Parameters
        ----------
        theta_array: np.ndarray, Array of theta values to turn into an instance of Data
        get_y: bool, Whether to calculate y values for the theta_array
        
        Returns
        --------
        theta_arr_data: instance of Data, Data class instance for the theta_array
        """
        assert isinstance(theta_array, np.ndarray), "theta_array must be np.ndarray"
        assert len(theta_array.shape) == 1, "theta_array must be 1D"
        assert isinstance(self.exp_data.x_vals, (np.ndarray)), "self.exp_data.x_vals must be np.ndarray"
        
        #Repeat the theta best array once for each x value
        #Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.gen_y_data
        theta_arr_repeated = np.repeat(theta_array.reshape(1,-1), self.exp_data.get_num_x_vals() , axis =0)
        #Add instance of Data class to theta_best
        theta_arr_data = Data(theta_arr_repeated, self.exp_data.x_vals, None, None, None, None, None, None, self.simulator.bounds_theta_reg, self.simulator.bounds_x, self.cs_params.sep_fact, self.cs_params.seed)
        if get_y:
            #Calculate y values and sse for theta_best with noise
            theta_arr_data.y_vals = self.simulator.gen_y_data(theta_arr_data, self.simulator.noise_mean, self.simulator.noise_std)  
        
        #Set the best data to be in sse form if using a type 1 GP
        if self.method.emulator == False:
            theta_arr_data = self.simulator.sim_data_to_sse_sim_data(self.method, theta_arr_data, self.exp_data, self.cs_params.sep_fact, not get_y)
            
        return theta_arr_data
    
    def __run_bo_iter(self, iteration):
        """
        Runs a single GPBO iteration
        
        Parameters
        ----------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor, GP emulator for workflow
        iteration: int, The iteration of bo in progress
        
        Returns:
        --------
        iter_df: pd.DataFrame, Dataframe containing the results from the GPBO Workflow for iteration
        iter_max_ei_terms: pd.DataFrame or None, contains ei calculation terms for max ei parameter set if self.cs_params.save_data
        gp_emulator_curr: Instance of GP_Emulator, The class used for this iteration of the GPBO workflow
        r_stop: bool, termination criteria. Whether regret < speed
        """
        #Start timer
        #Initialize iter_max_ei df to None
        iter_max_ei_terms = None
        time_start = time.time()
        
        #Train GP model (this step updates the model to a trained model)
        self.gp_emulator.train_gp()

        #Calcuate best error
        best_err_data, best_error_metrics = self.__get_best_error()
        
        #Add not log best error to ep_bias
        if iteration == 0 or self.ep_bias.ep_enum.value == 4:
            #Since best error is squared when used in Jasrasaria calculations, the value will always be >=0      
            self.ep_bias.best_error = best_error_metrics[0]
                        
        #Calculate mean of var for validation set if using Jasrasaria heuristic
        if self.ep_bias.ep_enum.value == 4:
            #Calculate average gp mean and variance of the validation set
            val_gp_mean, val_gp_var = self.gp_emulator.eval_gp_mean_var_val()
            #For emulator methods, the mean of the variance should come from the sse variance
            if self.method.emulator == True:
                #Redefine gp_mean and gp_var to be the mean and variane of the sse
                val_gp_mean, val_gp_var = self.gp_emulator.eval_gp_sse_var_val(self.method, self.exp_data)
                
            #Check for ln(sse) values
            #For 1B, propogate errors associated with an unlogged sse value
            val_gp_var = val_gp_var*np.exp(val_gp_mean)**2             

            #Set mean of sse variance
            mean_of_var = np.average(val_gp_var)
            self.ep_bias.mean_of_var = mean_of_var
            
        #Set initial exploration bias and bo_iter
        if self.ep_bias.ep_enum.value == 2:
            self.ep_bias.bo_iter = iteration
        
        #Calculate new ep. Note. It is extemely important to do this AFTER setting the ep_max
        self.ep_bias.set_ep()
        
        #Set Optimization starting points for this iteration
        self.opt_start_pts = self.__make_starting_opt_pts(best_error_metrics)

        #Call optimize E[SSE] or log(E[SSE]) objective function
        #Note if we didn't want actual sse values, we would have to set get_y = False in create_data_instance_from_theta in __opt_with_scipy
        min_sse, min_theta_data = self.__opt_with_scipy("sse")

        #Call optimize EI acquistion fxn (If not using E[SSE])
        if self.method.method_name.value != 7:
            opt_acq, acq_theta_data = self.__opt_with_scipy("neg_ei")
            if self.method.emulator == True:
                ei_args = (acq_theta_data, self.exp_data, self.ep_bias, best_error_metrics, 
                           self.method, self.sg_mc_samples)
            else:
                ei_args = (acq_theta_data, self.exp_data, self.ep_bias, best_error_metrics)
        else:
            opt_acq, acq_theta_data = self.__opt_with_scipy("E_sse")

        #If type 2, turn it into sse_data
        #Set the best data to be in sse form if using a type 2 GP and find the min sse
        if self.method.emulator == True:
            #Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = self.simulator.sim_data_to_sse_sim_data(self.method, min_theta_data, self.exp_data, 
                                                                         self.cs_params.sep_fact, False)
            acq_sse_theta_data = self.simulator.sim_data_to_sse_sim_data(self.method, min_theta_data, self.exp_data, 
                                                                         self.cs_params.sep_fact, False)
                 
        #Otherwise the sse data is the original (scaled) data
        else:     
            #Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = min_theta_data
            acq_sse_theta_data = acq_theta_data
            
        #Evaluate max EI terms at theta
        if self.cs_params.save_data and not self.method.method_name.value == 7: 
            ei_max, iter_max_ei_terms = self.gp_emulator.eval_ei_misc(*ei_args)
        
        #Turn min_sse_sim value into a float (this makes analyzing data from csvs and dataframes easier)
        min_sse_gp = float(min_sse)
        min_sse_sim = float(min_sse_theta_data.y_vals)
        opt_acq_sim = float(acq_sse_theta_data.y_vals) 
                
        #calculate improvement if using Boyle's method to update the exploration bias
        #Improvement is true if the min sim sse found is lower than (not log) best error, otherwise it's false
        if min_sse_gp < best_error_metrics[0]:
            improvement = True
        else:
            improvement = False
        if self.ep_bias.ep_enum.value == 3:
            #Set ep improvement
            self.ep_bias.improvement = improvement
                    
        #Create a copy of the GP Emulator Class for this iteration
        gp_emulator_curr = copy.deepcopy(self.gp_emulator)
              
        #Call __augment_train_data to append training data
        self.__augment_train_data(acq_theta_data)

        #Calc time/ iter
        time_end = time.time()
        time_per_iter = time_end-time_start
        
        #Create Results Pandas DataFrame for 1 iter
        #Return SSE and not log(SSE) for 'Min Obj', 'Min Obj Act', 'Theta Min Obj'
        column_names = ['Best Error', 'Exploration Bias', 
                        'Theta Opt Acq', 'Opt Acq', 'Acq Obj Act', 'MSE Acq Act',
                        'Theta Min Obj', 'Min Obj GP', 'Min Obj Act',  'MSE Obj GP', 'MSE Obj Act',
                        'Time/Iter']
        num_exp_x = self.exp_data.get_num_x_vals()
        MSE_acq_obj_act = np.exp(opt_acq_sim)/num_exp_x if self.method.obj.value == 2 else opt_acq_sim/num_exp_x
        MSE_obj_act = np.exp(min_sse_sim)/num_exp_x if self.method.obj.value == 2 else min_sse_sim/num_exp_x
        MSE_obj_gp = np.exp(min_sse_gp)/num_exp_x if self.method.obj.value == 2 else min_sse_gp/num_exp_x
        bo_iter_results = [best_error_metrics[0], float(self.ep_bias.ep_curr), 
                           acq_theta_data.theta_vals[0], float(opt_acq), opt_acq_sim, MSE_acq_obj_act,
                           min_sse_theta_data.theta_vals[0], min_sse_gp, min_sse_sim, MSE_obj_gp, MSE_obj_act,
                           time_per_iter]
        # column_names = ['Best Error', 'Exploration Bias', 'Opt Acq', 'Theta Opt Acq', 'Min Obj', 'Min Obj Act', 'Theta Min Obj', 'Time/Iter']
        iter_df = pd.DataFrame(columns=column_names)
        # bo_iter_results = [best_error_metrics[0], float(self.ep_bias.ep_curr), float(opt_acq), acq_theta_data.theta_vals[0],
        #                     min_sse_gp, min_sse_sim, min_sse_theta_data.theta_vals[0], time_per_iter]
        # Add the new row to the DataFrame
        iter_df.loc[0] = bo_iter_results

        return iter_df, iter_max_ei_terms, gp_emulator_curr #, r_stop
    
    def __run_bo_to_term(self):
        """
        Runs multiple GPBO iterations
        
        Params:
        -------
        gp_model: Instance of sklearn.gaussian_process.GaussianProcessRegressor, GP emulator for workflow
        
        Returns:
        --------
        iter_df: pd.DataFrame, Dataframe containing the results from the GPBO Workflow for all iterations
        max_ei_details_df: pd.DataFrame, contains ei data for max ei parameter sets for each bo iter if self.cs_params.save_data
        list_gp_emulator_class: list of instances of GP_Emulator, The classes used for all iterations of the GPBO workflow
        why_term: str, string containing reasons for bo algorithm termination 
        """
        assert 0 < self.bo_iter_term_frac <= 1, "self.bo_iter_term_frac must be between 0 and 1"
        #Initialize pandas dataframes
        # column_names = ['Best Error', 'Exploration Bias', 'Opt Acq', 'Theta Opt Acq', 'Min Obj', 'Min Obj Act', 'Theta Min Obj', 'Min Obj Cum.', 'Theta Min Obj Cum.', 'Regret', 'Speed', 'Time/Iter']
        # column_names = ['Best Error', 'Exploration Bias', 
        #                 'Theta Opt Acq', 'Opt Acq', 'Acq Obj Act', 
        #                 'Theta Acq Act Cum', 'Acq Obj Act Cum', 
        #                 'Theta Min Obj', 'Min Obj GP', 'Min Obj Act',  
        #                 'Theta Obj GP Cum', 'Min Obj GP Cum', 
        #                 'Theta Obj Act Cum', 'Min Obj Act Cum', 
        #                 'Time/Iter']
        column_names = ['Best Error', 'Exploration Bias', 
                        'Theta Opt Acq', 'Opt Acq', 'Acq Obj Act', 'MSE Acq Act',
                        'Theta Min Obj', 'Min Obj GP', 'Min Obj Act',  'MSE Obj GP', 'MSE Obj Act',
                        'Time/Iter']
        results_df = pd.DataFrame(columns=column_names)
        max_ei_details_df = pd.DataFrame()
        list_gp_emulator_class = []
        
        #Initilize terminate flags   
        acq_flag = False
        obj_flag = False
        max_bud_flag = False
        terminate = False
        
        #Set why_term strings
        # why_terms = ["acq", "obj", "regret", "max_budget"]
        why_terms = ["acq", "obj", "max_budget"]
        
        #Initialize count
        obj_counter = 0
        
        #Do Bo iters while stopping criteria is not met
        while terminate == False: 
            #Loop over number of max bo iters
            for i in range(self.cs_params.bo_iter_tot):
                #Output results of 1 bo iter and the emulator used to get the results
                iter_df, iter_max_ei_terms, gp_emulator_class = self.__run_bo_iter(i) #Change me later
                #Add results to dataframe
                results_df = pd.concat([results_df.astype(iter_df.dtypes), iter_df], ignore_index=True)
                if iter_max_ei_terms is not None:
                    max_ei_details_df = pd.concat([max_ei_details_df, iter_max_ei_terms])
                #At the first iteration
                if i == 0:
                    #improvement is defined as infinity on 1st iteration (something is always better than nothing)
                    improvement = np.inf 
                elif results_df["Min Obj GP"].iloc[i] < float(results_df["Min Obj GP"][:-1].min()):
                    #And the improvement is defined as the difference between the last Min Obj Cum. and current Obj Min (unscaled)
                    if self.method.obj.value == 1:
                        improvement = results_df["Min Obj GP"][:-1].min() - results_df["Min Obj GP"].iloc[i]
                    else:
                        improvement = np.exp(results_df["Min Obj GP"][:-1].min()) - np.exp(results_df["Min Obj GP"].iloc[i])
                #Otherwise
                else:
                    #And the improvement is defined as 0, since it must be non-negative
                    improvement = 0

                #Add gp emulator data from that iteration to list
                list_gp_emulator_class.append( gp_emulator_class )
                
                #Call stopping criteria after 1st iteration and update improvement counter
                #If the improvement is negligible, add to counter
                if improvement < self.cs_params.obj_tol:
                    obj_counter += 1
                #Otherwise reset the counter
                else:
                    obj_counter = 0 
                    
                #set flag if opt acq. func val is less than the tolerance 3 times in a row
                if all(results_df["Opt Acq"].tail(3) < self.cs_params.acq_tol) and i > 2:
                    acq_flag = True
                #set flag if small sse progress over 1/3 of total iteration budget
                if obj_counter >= int(self.cs_params.bo_iter_tot*self.bo_iter_term_frac) and i > 0:
                    obj_flag = True

                flags = [acq_flag, obj_flag]
                 
                #Terminate if you meet 2 stopping criteria, hit the budget, or obj has not improved after 1/2 of iterations
                if flags.count(True) >= 2:
                    terminate = True
                    #Pull indecies of list that are true
                    term_flags = [why_terms[index] for index, value in enumerate(flags) if value]
                    why_term = "-".join(term_flags)
                    break
                elif i == self.cs_params.bo_iter_tot - 1:
                    terminate = True
                    why_term = why_terms[-1]
                    break                    
                elif obj_counter >= int(self.cs_params.bo_iter_tot*0.5) and self.cs_params.bo_iter_tot > 10:
                    terminate = True
                    why_term = why_terms[1]
                    break
                #Continue if no stopping criteria are met   
                else:
                    terminate = False

        #Reset the index of the pandas df
        results_df = results_df.reset_index()

        #Fill Cumulative value columns based on results
        #Initialize cum columns as the same as the original columns
        # results_df.insert(1, "BO Iter", results_df.index + 1)
        results_df.rename(columns={'index': 'BO Iter'}, inplace=True) 
        results_df["BO Iter"] += 1
        results_df["BO Method"] = self.method.report_name
        results_df["Max Evals"] = len(results_df)
        results_df['Theta Acq Act Cum'] = results_df['Theta Opt Acq']
        results_df['Theta Obj GP Cum'] = results_df['Theta Min Obj']
        results_df['Theta Obj Act Cum'] = results_df['Theta Min Obj']
        results_df["Termination"] = why_term
        results_df["Total Run Time"] = float(results_df["Time/Iter"].sum())

        results_df["Min Obj GP Cum"] = np.minimum.accumulate(results_df['Min Obj GP'])
        results_df['Min Obj Act Cum'] = np.minimum.accumulate(results_df['Min Obj Act'])
        results_df["Acq Obj Act Cum"] = np.minimum.accumulate(results_df['Acq Obj Act'])

        for i in range(len(results_df)):
            if i > 0:
                if results_df["Acq Obj Act Cum"].iloc[i] >= results_df["Acq Obj Act Cum"].iloc[i-1]:
                    results_df.at[i, 'Theta Acq Act Cum'] = results_df['Theta Acq Act Cum'].iloc[i-1].copy()
                if results_df["Min Obj Act Cum"].iloc[i] >= results_df["Min Obj Act Cum"].iloc[i-1]:
                    results_df.at[i, 'Theta Obj Act Cum'] = results_df['Theta Obj Act Cum'].iloc[i-1].copy()
                if results_df["Min Obj GP Cum"].iloc[i] >= results_df["Min Obj GP Cum"].iloc[i-1]:
                    results_df.at[i, 'Theta Obj GP Cum'] = results_df['Theta Obj GP Cum'].iloc[i-1].copy()
        
        #Create df for ei and add those results here
        if iter_max_ei_terms is not None:
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
        
        #Reset ep_bias to None for each workflow restart
        self.ep_bias.ep_curr = None
        
        ##Call bo_iter
        results_df, max_ei_details_df, list_gp_emulator_class, why_term = self.__run_bo_to_term()
        
        #Set results
        bo_results_res = BO_Results(None, None, self.exp_data, None, results_df, 
                                max_ei_details_df, why_term, None)
        
        bo_results_GPs = BO_Results(None, None, None, list_gp_emulator_class, None, 
                                max_ei_details_df, None, None)
        
        return bo_results_res, bo_results_GPs

    
    def run_bo_restarts(self):
        """
        Runs multiple GPBO restarts
        
        Returns:
        --------
        restart_bo_results, list of instances of BO_Results, Includes the results related to a set of Bo iters for all restarts
        """
        gpbo_res_simple = []
        gpbo_res_GP = []
        simulator_class = self.simulator
        configuration = {"DateTime String" : self.cs_params.DateTime,
                         "Method Name Enum Value" : self.method.method_name.value,
                         "Case Study Name" : self.cs_params.cs_name,
                         "Number of Parameters": len(self.simulator.theta_true_names),
                         "Number of State Points": self.exp_data.get_num_x_vals(),
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
                         "Acq Tolerance" : self.cs_params.acq_tol,
                         "MC SG Max Points": self.sg_mc_samples,
                         "Obj Improvement Tolerance" : self.cs_params.obj_tol,
                         "Theta Generation Enum Value": self.gen_meth_theta.value}
                
        for i in range(self.cs_params.bo_run_tot):
            bo_results_res, bo_results_GPs = self.__run_bo_workflow()
            #Update the seed in configuration
            configuration["Seed"] = self.cs_params.seed
            #Add this updated copy of configuration with the new seed to the bo_results
            bo_results_res.configuration = configuration.copy()           
            #Add simulator class
            bo_results_res.simulator_class = simulator_class
            #On the 1st iteration, create heat map data if we are actually generating the data           
            if i == 0:
                if self.cs_params.gen_heat_map_data == True:
                    #Generate heat map data for each combination of parameter values stored in a dictionary
                    heat_map_data_dict = self.create_heat_map_param_data()
                    # Save these heat map values in the bo_results object 
                    # Only store in first list entry to avoid repeated data which stays the same for each iteration.
                    bo_results_GPs.heat_map_data_dict = heat_map_data_dict
            gpbo_res_simple.append(bo_results_res)
            gpbo_res_GP.append(bo_results_GPs)
            #Add 2 to the seed for each restart (1 for the sim/exp data seed and 1 for validation data seed) to get completely new seeds
            self.cs_params.seed += 2
                   
        #Save data automatically if DateTime is not None
        if self.cs_params.DateTime is not None:
            self.save_data(gpbo_res_simple)

        return gpbo_res_simple, gpbo_res_GP
    
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
        